import numpy
import scipy.linalg
import utils


class StepModel:
    """Builds and simulates step models of Laplace transfer function systems.
    It also contains the ability to build the matrices required for MPC
    simulation as described in Seborg. It makes use of the `utils.InternalDelay`
    class for the TF representation.

    Parameters
        ----------
        G : utils.InternalDelay
        The Laplace TF model of the system

        dt : float
            The sampling time

        N : int
            The number of sampling instances for the step response

        P : int
            The number of sampling instants for the prediction horizon

        M : int
            The number of sampling instants for the control horizon

        integrators : bool, optional
            Should be `True` if there are integrators in the system.
            Can be `True` even if there are no integrators.
            Defaults to `True`

        dvs : int, optional
            The number of inputs that are disturbance variables.
            The code assumes that the last `dvs` inputs are disturbance variables.
            Defaults to 0

    Attributes
    -----------
    G : utils.InternalDelay
        The Laplace TF model of the system

    dt : float
        The sampling time

    N : int
        The number of sampling instances for the step response

    P : int
        The number of sampling instants for the prediction horizon

    M : int
        The number of sampling instants for the control horizon

    integrators : bool
        Should be `True` if there are integrators in the system.
        Can be `True` even if there are no integrators.

    dvs : int
        The number of inputs that are disturbance variables.
        The code assumes that the last `dvs` inputs are disturbance variables.

    outs, ins, mvs : int
        The number of outputs, inputs (MV's and DV's) and disturbances

    Y0 : array_like
        The effect of previous inputs on the futrue outputs
    """
    def __init__(self, G: utils.InternalDelay, dt, N, P, M, integrators=True, dvs=0):
        self.G = G
        self.dt = dt
        self.N = N
        self.P = P
        self.M = M
        self.integrators = integrators
        self.dvs = dvs

        self.outs, self.ins = self.G.D11.shape
        self.mvs = self.ins - self.dvs

        self.__make_step_coeffs()
        self.__make_step_matrix()
        self.__make_Y0_matrix()

        self.Y0 = numpy.zeros(self.A.shape[0])
        self._dUs = [numpy.zeros(self.ins)] * N
        self._dU_old_tot = numpy.zeros(self.ins)

    def step_Y0(self, dU):
        """Changes the value of Y0 based on the current inputs

        Parameters
        ----------
        dU : array_like
            Current change in inputs
        """
        self._dUs.append(dU)
        dUs_dyn = numpy.array(self._dUs[-self.N:]).T.flatten()
        self._dU_old_tot += self._dUs[-self.N-1]
        y_old_tot = self.y_steps[:, :, -1] @ self._dU_old_tot
        self.Y0 = self.C @ dUs_dyn + y_old_tot.repeat(self.P)

    def reset(self):
        """Resets the class for next simulation"""
        self.Y0 = numpy.zeros(self.A.shape[0])

    def __make_step_coeffs(self):
        """Private function that finds the step response coefficients from simulation"""
        t = (self.N+1) * self.dt
        ts = numpy.linspace(0, t, int(t * 100))
        t_step = numpy.arange(self.dt, t, self.dt)
        ind = numpy.searchsorted(ts, t_step)
        y_steps = numpy.zeros_like(self.G.D11).tolist()
        assert isinstance(y_steps, list)

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
        """Private function that builds the required matrix for MPC calculations
        from the step response coefficients"""
        if self.P < self.N:
            cols = self.y_steps[:, :, :self.P]
        else:
            a = numpy.moveaxis(numpy.tile(self.y_steps[:, :, -1], (self.P - self.N, 1, 1)), 0, -1)
            cols = numpy.append(self.y_steps, a, axis=2)

        rows = numpy.append(self.y_steps[:, :, :1], numpy.zeros((self.outs, self.ins, self.M - 1)), axis=2)
        self.A = self.__mimo_toeplitz(rows, cols)

        if self.integrators:
            self.__make_integrators_matrix()
            self.A += self.B

    def __make_integrators_matrix(self):
        """Private function that builds the required matrix for integrator calculations
        from the step response coefficients"""
        dys = self.y_steps[:, :, -1] - self.y_steps[:, :, -2]

        if self.P < self.N:
            cols = numpy.zeros((self.outs, self.ins, self.P))
        else:
            a = numpy.tile(numpy.arange(1, self.P - self.N + 1), (2, 2, 1))
            cols = numpy.append(numpy.zeros((self.outs, self.ins, self.N)), a, axis=2)
        rows = numpy.zeros((self.outs, self.ins, self.M))

        Bs = numpy.zeros_like(self.G.D11).tolist()
        assert isinstance(Bs, list)

        for in_i in range(self.ins):
            for out_i in range(self.outs):
                Bs[out_i][in_i] = scipy.linalg.toeplitz(cols[out_i][in_i], rows[out_i][in_i]) * dys[out_i][in_i]

        self.B = numpy.block(Bs)

    def __make_Y0_matrix(self):
        """Private function that builds the required matrix for Y0 calculations
        from the step response coefficients"""
        rows = numpy.insert(self.y_steps[:, :, 1:], -1, self.y_steps[:, :, -1], axis=2)
        rows = numpy.flip(rows, axis=2)
        cols = numpy.repeat(self.y_steps[:, :, -1][:, :, numpy.newaxis], self.P, axis=2)
        self.C = self.__mimo_toeplitz(rows, cols)

    def __mimo_toeplitz(self, rows, cols):
        """Private utility function that builds MIMO toeplitz matrices"""
        As = numpy.zeros_like(self.G.D11).tolist()
        assert isinstance(As, list)

        for in_i in range(self.ins):
            for out_i in range(self.outs):
                As[out_i][in_i] = scipy.linalg.toeplitz(cols[out_i][in_i], rows[out_i][in_i])

        A = numpy.block(As)
        return A
