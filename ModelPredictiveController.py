import StepModel
import numpy
import cvxpy


class ModelPredictiveController:
    """Runs an MPC controller.

    Parameters
    ----------
    SM : StepModel.StepModel
        The step model of the system that we want to control

    Ysp : array_like
        The default set points for control calculations

    dDVs : array_like
        The initial changes in the DV's.
        This holds information useful for known/measured DV's

    Q : array_like, optional
        A 1d-array_like of the diagonal elements of the Q tuning matrix.
        Q tunes the importance of outputs in the control calculations.
        Defaults to all ones

    R : array_like, optional
        A 1d-array_like of the diagonal elements of the R tuning matrix.
        R tunes the importance of inputs in the control calculations.
        Defaults to all ones

    constraints : callable, optional
        A function that takes in a `ModelPredictiveController` object
        and returns A, a 1d array_like of constraints, that must be satisfied in the form:
        A_i <= 0 for all i
        Defaults to no constraints

    Attributes
    ----------
    SM : StepModel.StepModel
        The step model of the system that we want to control

    dMVs : cvxpy variable array_like
        The changes in the MV's.
        Used in the optimization

    dU :  cvxpy array_like
        Contains the changes in the inputs and DV's

    Y :  cvxpy array_like
        Contains the P predicted values for each output

    Ysp : array_like
        The P values for the future set points for each output

    E : cvxpy array_like
        Contains the error between the outputs and the references: Y - Ysp

    Q : 2d-array_like
        A matrix containing the the Q tuning matrix.
        Q tunes the importance of outputs in the control calculations

    R : 2d-array_like
        A matrix containing the the R tuning matrix.
        R tunes the importance of inputs in the control calculations

    cons : array_like of cvxpy inequalities
        Contains the constraints for the optimization

    obj : cvxpy value
        The objective function to be minimized

    K : 2d-array_like
        The DMC (unconstrained) analytical controller
    """
    def __init__(self, SM: StepModel.StepModel,
                 Ysp=None, dDVs=None, Q=None, R=None,
                 constraints=lambda mpc: [], MVs=None):
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

        self.MVs = MVs
        if MVs is None:
            self.MVs = numpy.zeros(self.SM.mvs)

        self.constraints = constraints
        self.cons = self.constraints(self)

        self.J = cvxpy.quad_form(self.E, self.Q) + cvxpy.quad_form(self.dMVs, self.R)
        self.obj = cvxpy.Minimize(self.J)
        self.prob = cvxpy.Problem(self.obj, self.cons)
        self.prob.solve(solver='OSQP')

        self.bias = numpy.full(SM.P * SM.outs, 0)

        self.A4K = self.SM.A[:, :self.SM.M * self.SM.mvs]
        self.K = numpy.linalg.inv(self.A4K.T @ self.Q @ self.A4K + self.R) @ self.A4K.T @ self.Q
        self.Y_old = self.SM.A @ self.dU + self.SM.Y0

    def step(self, Y_actual, MV_actual=None, Ysp=None, dDVs=None):
        """Calculates next receding horizon control action of the controller.

        Parameters
        ----------
        MV_actual : array_like
            The current value of the outputs.
            Used for constraint purposes

        Y_actual : array_like
            The current value of the outputs.
            Used to take into account model error and unmeasured disturbances

        Ysp : array_like
            The set points for the current control calculation.
        dDVs : array_like
            The changes in the DV's since the last iteration.
            This holds information useful for known/measured DV's

        Returns
        -------
            dU_out : array_like
                The optimal changes in the inputs for this iteration
        """
        self.MVs = MV_actual
        if MV_actual is None:
            self.MVs = numpy.zeros(self.SM.mvs)

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

        self.Y = self.Y_old
        self.bias = numpy.repeat(Y_actual - self.Y.value[::self.SM.P], self.SM.P)

        self.Y = self.SM.A @ self.dU + self.SM.Y0 + self.bias
        self.E = self.Ysp - self.Y
        self.J = cvxpy.quad_form(self.E, self.Q) + cvxpy.quad_form(self.dMVs, self.R)
        self.obj = cvxpy.Minimize(self.J)
        self.cons = self.constraints(self)

        kwargs = {'solver': 'OSQP', 'warm_start': True, 'parallel': True, 'qcp': True}
        try:
            self.prob = cvxpy.Problem(self.obj, self.cons)
            self.prob.solve(**kwargs)
            dU_out = self.dMVs.value
        except cvxpy.error.SolverError:
            # print(self.prob.status)
            self.prob = cvxpy.Problem(self.obj)
            self.prob.solve( **kwargs)
            dU_out = self.dMVs.value
            # E_f = self.Ysp - self.SM.Y0 - self.bias
            # dU_out = (self.K @ E_f)

        if self.prob.solution is None or self.prob.solution.status != 'optimal':
            # print("Status: " + self.prob.solution.status)
            # print("Solving problem without constraints")
            self.prob = cvxpy.Problem(self.obj)
            self.prob.solve(**kwargs)
            if self.prob.solution.status == 'optimal':
                pass
                # print("It worked!")
            dU_out = self.dMVs.value
            # E_f = self.Ysp - self.SM.Y0 - self.bias
            # dU_out = (self.K @ E_f)

        self.SM.step_Y0(self.dU.value[::self.SM.M])
        self.Y_old = self.Y - self.bias

        return dU_out[::self.SM.M]

    def reset(self):
        self.Y_old = self.SM.A @ self.dU + self.SM.Y0
