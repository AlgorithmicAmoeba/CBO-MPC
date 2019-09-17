import StepModel
import numpy
import cvxpy


class ModelPredictiveController:
    def __init__(self, SM: StepModel.StepModel, Ysp=None, Q=None, R=None, dU_max=None, E_max=None):
        self.SM = SM
        self.dU = cvxpy.Variable(SM.M * SM.ins)
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

        if dU_max is None:
            dU_max = 1e20

        self.dU_max = dU_max

        if E_max is None:
            E_max = 1e20

        self.E_max = E_max

        self.J = cvxpy.quad_form(self.E, self.Q) + cvxpy.quad_form(self.dU, self.R)
        self.obj = cvxpy.Minimize(self.J)
        self.cons = [cvxpy.abs(self.dU) <= self.dU_max, cvxpy.abs(self.E) <= self.E_max]
        self.prob = cvxpy.Problem(self.obj, self.cons)
        self.prob.solve(solver='MOSEK')

        self.bias = None

        self.K = numpy.linalg.inv(SM.A.T @ self.Q @ SM.A + self.R) @ SM.A.T @ self.Q

    def step(self, Y_actual, Ysp=None):

        if Ysp is not None:
            self.Ysp = Ysp

        self.bias = numpy.repeat(Y_actual - self.Y.value[::self.SM.P], self.SM.P)

        self.Y = self.SM.A @ self.dU + self.SM.Y0 + self.bias
        self.E = self.Ysp - self.Y
        self.J = cvxpy.quad_form(self.E, self.Q) + cvxpy.quad_form(self.dU, self.R)
        self.obj = cvxpy.Minimize(self.J)
        self.cons = [cvxpy.abs(self.dU) <= self.dU_max, cvxpy.abs(self.E) <= self.E_max]

        kwargs = {'solver': 'MOSEK', 'warm_start': True, 'parallel': True}
        try:
            self.prob = cvxpy.Problem(self.obj, self.cons)
            self.prob.solve(**kwargs)
            dU_out = self.dU.value
        except cvxpy.error.SolverError:
            print(self.prob.status)
            self.prob = cvxpy.Problem(self.obj)
            self.prob.solve(verbose=True, **kwargs)
            dU_out = self.dU.value
            # E_f = self.Ysp - self.SM.Y0 - self.bias
            # dU_out = (self.K @ E_f)

        if self.prob.solution is None or self.prob.solution.status != 'optimal':
            # print("Status: " + self.prob.solution.status)
            print("Solving problem without constraints")
            self.prob = cvxpy.Problem(self.obj)
            self.prob.solve(**kwargs)
            if self.prob.solution.status == 'optimal':
                print("It worked!")
            dU_out = self.dU.value
            # E_f = self.Ysp - self.SM.Y0 - self.bias
            # dU_out = (self.K @ E_f)

        self.SM.step_Y0(dU_out[::self.SM.M])

        return dU_out[::self.SM.M]
