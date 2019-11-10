import utils
import numpy


class PlantModel:
    """Simulates a MIMO TF object that acts as the real system for the simulation

    Parameters
    ----------
    G : utils.InternalDelay
        A TF representation of the system

    x0 : array_like, optional
        The initial state of the system.
        Will assume a vector of zeros if no value is given

    Attributes
    ----------
    G : utils.InternalDelay
        A TF representation of the system

    x : array_like
        The current state of the system

    zs : 2d-array_like
        The delays inputs to the `InternalDelay` object
    """
    def __init__(self, G: utils.InternalDelay, x0=None):
        self.G = G
        if x0 is None:
            self.x = numpy.zeros(G.A.shape[0])

        self.zs = []

    def step(self, u, dt):
        """Steps the response of the system to the input.
        Uses a Runge-Kutta delay integration routine.
        Parameters
        ----------
        u : array_like
            The input to the system.
        dt : float
        A scalar indicating the time sincde the previous call

        Returns
        -------
        y : array_like
            The output from the system
        """

        dtss = [int(numpy.round(delay / dt)) for delay in self.G.delays]

        def wf():
            ws = []
            for i, dts in enumerate(dtss):
                if len(self.zs) <= dts:
                    ws.append(0)
                elif dts == 0:
                    ws.append(self.zs[-1][i])
                else:
                    ws.append(self.zs[-dts][i])

            return numpy.array(ws)

        def f(x):
            return self.G.A @ x + self.G.B1 @ u + self.G.B2 @ wf()

        # y
        y = self.G.C1 @ numpy.array(self.x) + self.G.D11 @ u + self.G.D12 @ wf()

        # z
        z = self.G.C2 @ numpy.array(self.x) + self.G.D21 @ u + self.G.D22 @ wf()
        self.zs.append(list(z))

        # x integration
        k1 = f(self.x) * dt
        k2 = f(self.x + 0.5 * k1) * dt
        k3 = f(self.x + 0.5 * k2) * dt
        k4 = f(self.x + k3) * dt
        dx = (k1 + k2 + k2 + k3 + k3 + k4) / 6
        self.x = [xi + dxi for xi, dxi in zip(self.x, dx)]

        return y

    def reset(self):
        self.x = numpy.zeros(self.G.A.shape[0])
        self.zs = []
