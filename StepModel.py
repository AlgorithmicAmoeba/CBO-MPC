import numpy
import scipy.linalg
import utils


class StepModel:
    def __init__(self, G: utils.InternalDelay, dt, N, P, M, integrators=True):
        self.G = G
        self.dt = dt
        self.N = N
        self.P = P
        self.M = M
        self.integrators = integrators

        self.outs, self.ins = self.G.D11.shape

        self.__make_step_coeffs()
        self.__make_step_matrix()
        self.__make_Y0_matrix()

        self.Y0 = numpy.zeros(self.A.shape[0])
        self._dUs = [numpy.zeros(self.ins)] * N
        self._dU_old_tot = numpy.zeros(self.ins)

    def step_Y0(self, dU):  # Untested. Will test with MPC implementation
        self._dUs.append(dU)
        dUs_dyn = numpy.array(self._dUs[-self.N:]).T.flatten()
        self._dU_old_tot += self._dUs[-self.N-1]
        y_old_tot = self.y_steps[:, :, -1] @ self._dU_old_tot
        self.Y0 = self.C @ dUs_dyn + y_old_tot.repeat(self.P)

    def reset(self):
        self.Y0 = numpy.zeros(self.A.shape[0])

    def __make_step_coeffs(self):
        t = (self.N+1) * self.dt
        ts = numpy.linspace(0, t, int(t * 100))
        t_step = numpy.arange(self.dt, t, self.dt)
        ind = numpy.searchsorted(ts, t_step)
        y_steps = numpy.zeros_like(self.G.D11).tolist()

        for in_i in range(self.ins):
            def uf(ti):
                ti += 1  # Done to prevent PEP8 error
                return [0] * in_i + [1] + [0] * (self.ins - in_i - 1)
            ys = self.G.simulate(uf, ts)

            for out_i in range(self.outs):
                yi = ys[:, out_i][ind]
                largest = 1e10
                yi[yi > largest] = largest
                yi[yi < -largest] = -largest
                y_steps[out_i][in_i] = yi

        self.y_steps = numpy.array(y_steps)

    def __make_step_matrix(self):
        if self.P < self.N:
            cols = self.y_steps[:, :, :self.P]
        else:
            a = numpy.moveaxis(numpy.tile(self.y_steps[:, :, -1], (self.P - self.N, 1, 1)), 0, -1)
            cols = numpy.append(self.y_steps, a, axis=2)

        rows = numpy.append(self.y_steps[:, :, :1], numpy.zeros((self.outs, self.ins, self.M - 1)), axis=2)
        As = numpy.zeros_like(self.G.D11).tolist()

        for in_i in range(self.ins):
            for out_i in range(self.outs):
                As[out_i][in_i] = scipy.linalg.toeplitz(cols[out_i][in_i], rows[out_i][in_i])

        self.A = numpy.block(As)

        if self.integrators:
            self.__make_integrators_matrix()
            self.A += self.B

    def __make_integrators_matrix(self):
        dys = self.y_steps[:, :, -1] - self.y_steps[:, :, -2]

        if self.P < self.N:
            cols = numpy.zeros((self.outs, self.ins, self.P))
        else:
            a = numpy.tile(numpy.arange(1, self.P - self.N + 1), (2, 2, 1))
            cols = numpy.append(numpy.zeros((self.outs, self.ins, self.N)), a, axis=2)
        rows = numpy.zeros((self.outs, self.ins, self.M))

        Bs = numpy.zeros_like(self.G.D11).tolist()

        for in_i in range(self.ins):
            for out_i in range(self.outs):
                Bs[out_i][in_i] = scipy.linalg.toeplitz(cols[out_i][in_i], rows[out_i][in_i]) * dys[out_i][in_i]

        self.B = numpy.block(Bs)

    def __make_Y0_matrix(self):
        rows = numpy.insert(self.y_steps[:, :, 1:], -1, self.y_steps[:, :, -1], axis=2)
        rows = numpy.flip(rows, axis=2)
        cols = numpy.repeat(self.y_steps[:, :, -1][:, :, numpy.newaxis], self.P, axis=2)
        Cs = numpy.zeros_like(self.G.D11).tolist()

        for in_i in range(self.ins):
            for out_i in range(self.outs):
                Cs[out_i][in_i] = scipy.linalg.toeplitz(cols[out_i][in_i], rows[out_i][in_i])

        self.C = numpy.block(Cs)
