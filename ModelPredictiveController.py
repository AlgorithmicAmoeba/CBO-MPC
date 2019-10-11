import StepModel
import numpy
import cvxpy


class ModelPredictiveController:
    def __init__(self, SM: StepModel.StepModel, Ysp=None, dDVs=None, Q=None, R=None, dU_max=None, E_max=None):
        self.SM = SM

        if Ysp is None:
            Ysp = numpy.full(self.SM.outs, 0)

        if dDVs is None:
            dDVs = numpy.full(self.SM.dvs * self.SM.M, 0)

        self.dMVs = cvxpy.Variable(SM.M * SM.mvs)
        if self.SM.dvs:
            self.dU = cvxpy.hstack([self.dMVs, dDVs])
        else:
            self.dU = self.dMVs
        self.Y = SM.A @ self.dU

        self.Ysp = Ysp.repeat(self.SM.P)
        self.E = self.Ysp - self.Y

        SM.reset()

        if Q is None:
            Q = numpy.ones(SM.P * SM.outs)

        self.Q = numpy.diag(Q)

        if R is None:
            R = numpy.ones(SM.M * SM.ins)

        self.R = numpy.diag(R)

        self.cons = []
        self.dU_max = dU_max
        if dU_max is not None:
            self.cons.append(cvxpy.abs(self.dU) <= self.dU_max)

        self.E_max = E_max
        if E_max is not None:
            self.cons.append(cvxpy.abs(self.E) <= self.E_max)

        self.J = cvxpy.quad_form(self.E, self.Q) + cvxpy.quad_form(self.dMVs, self.R)
        self.obj = cvxpy.Minimize(self.J)
        self.prob = cvxpy.Problem(self.obj, self.cons)
        self.prob.solve(solver='OSQP')

        self.bias = numpy.full(SM.P * SM.outs, 0)

        self.A4K = self.SM.A[:, :self.SM.M * self.SM.mvs]
        self.K = numpy.linalg.inv(self.A4K.T @ self.Q @ self.A4K + self.R) @ self.A4K.T @ self.Q

    def step(self, Y_actual, Ysp=None, dDVs=None):

        if Ysp is not None:
            self.Ysp = Ysp.repeat(self.SM.P)

        if dDVs is None:
            dDVs = numpy.full(self.SM.dvs * self.SM.M, 0)
        else:
            dDVs = dDVs.repeat(self.SM.M)

        if self.SM.dvs:
            self.dU = cvxpy.hstack([self.dMVs, dDVs])
        else:
            self.dU = self.dMVs

        self.Y = self.SM.A @ self.dU + self.SM.Y0
        self.bias = numpy.repeat(Y_actual - self.Y.value[::self.SM.P], self.SM.P)

        self.Y = self.SM.A @ self.dU + self.SM.Y0 + self.bias
        self.E = self.Ysp - self.Y
        self.J = cvxpy.quad_form(self.E, self.Q) + cvxpy.quad_form(self.dMVs, self.R)
        self.obj = cvxpy.Minimize(self.J)
        self.cons = []
        if self.dU_max is not None:
            self.cons.append(cvxpy.abs(self.dU) <= self.dU_max)

        if self.E_max is not None:
            self.cons.append(cvxpy.abs(self.E) <= self.E_max)

        kwargs = {'solver': 'OSQP', 'warm_start': True, 'parallel': True, 'qcp': True}
        try:
            self.prob = cvxpy.Problem(self.obj, self.cons)
            self.prob.solve(**kwargs)
            dU_out = self.dMVs.value
        except cvxpy.error.SolverError:
            print(self.prob.status)
            self.prob = cvxpy.Problem(self.obj)
            self.prob.solve(verbose=True, **kwargs)
            dU_out = self.dMVs.value
            # E_f = self.Ysp - self.SM.Y0 - self.bias
            # dU_out = (self.K @ E_f)

        if self.prob.solution is None or self.prob.solution.status != 'optimal':
            # print("Status: " + self.prob.solution.status)
            print("Solving problem without constraints")
            self.prob = cvxpy.Problem(self.obj)
            self.prob.solve(**kwargs)
            if self.prob.solution.status == 'optimal':
                print("It worked!")
            dU_out = self.dMVs.value
            # E_f = self.Ysp - self.SM.Y0 - self.bias
            # dU_out = (self.K @ E_f)

        self.SM.step_Y0(self.dU.value[::self.SM.M])

        return dU_out[::self.SM.M]
