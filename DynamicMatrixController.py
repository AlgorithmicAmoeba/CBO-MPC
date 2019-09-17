import StepModel
import numpy


class DynamicMatrixController:
    def __init__(self, SM: StepModel.StepModel, Ysp=None, Q=None, R=None):
        self.SM = SM
        self.dU = numpy.zeros(SM.M * SM.ins)
        self.Y = SM.A @ self.dU

        if Ysp is None:
            Ysp = numpy.full(SM.P * SM.outs, 0)

        self.Ysp = Ysp
        self.E = self.Ysp - self.Y

        SM.reset()

        if Q is None:
            Q = numpy.ones(SM.P * SM.outs)

        self.Q = numpy.diag(Q)

        if R is None:
            R = numpy.ones(SM.M * SM.ins)

        self.R = numpy.diag(R)

        self.bias = None

        self.K = numpy.linalg.inv(SM.A.T @ self.Q @ SM.A + self.R) @ SM.A.T @ self.Q

    def step(self, Y_actual, Ysp=None):

        if Ysp is not None:
            self.Ysp = Ysp

        self.bias = numpy.repeat(Y_actual - self.Y[::self.SM.P], self.SM.P)

        self.Y = self.SM.A @ self.dU + self.SM.Y0 + self.bias
        self.E = self.Ysp - self.Y
        E_f = self.Ysp - self.SM.Y0 - self.bias
        dU_out = (self.K @ E_f)

        self.SM.step_Y0(dU_out[::self.SM.M])

        return dU_out[::self.SM.M]
