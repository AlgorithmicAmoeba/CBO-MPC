import StepModel
import PlantModel
import ModelPredictiveController
import utils
import numpy
import matplotlib.pyplot as plt
import tqdm
import pandas


class SimulateMPC:
    """Simulates the execution of an MPC controller on a system.

    Parameters
    ----------
    G : utils.InternalDelay
        The system to be controlled

    N : int
        The number of sampling instances for the step response

    P : int
        The number of sampling instants for the prediction horizon

    M : int
        The number of sampling instants for the control horizon

    dt_model : float
        Sampling time of the model

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

    integrators : bool, optional
        Should be `True` if there are integrators in the system.
        Can be `True` even if there are no integrators.
        Defaults to `True`

    dvs : int, optional
        The number of inputs that are disturbance variables.
        The code assumes that the last `dvs` inputs are disturbance variables.
        Defaults to 0

    known_dvs : int, optional
        The number of disturbance variables that are known/measured.
        The code assumes that the first `known_dvs` inputs are measured disturbance variables.
        Defaults to 0

    Attributes
    -----------
    G : utils.InternalDelay
        The system to be controlled

    SM : StepModel.StepModel
        The step model of the system

    MPC : ModelPredictiveController.ModelPredictiveController
        An MPC instance that will control the system

    PM : PlantModel
        The simulated plant that the controller acts on

    N : int
        The number of sampling instances for the step response

    P : int
        The number of sampling instants for the prediction horizon

    M : int
        The number of sampling instants for the control horizon

    dt_model : float
        Sampling time of the model

    Q : 2d-array_like
        A matrix containing the the Q tuning matrix.
        Q tunes the importance of outputs in the control calculations

    R : 2d-array_like
        A matrix containing the the R tuning matrix.
        R tunes the importance of inputs in the control calculations

    constraints : callable
        A function that takes in a `ModelPredictiveController` object
        and returns A, a 1d array_like of constraints, that must be satisfied in the form:
        A_i <= 0 for all i

    integrators : bool
        Should be `True` if there are integrators in the system.
        Can be `True` even if there are no integrators.

    dvs : int
        The number of inputs that are disturbance variables.
        The code assumes that the last `dvs` inputs are disturbance variables.

    known_dvs : int
        The number of disturbance variables that are known/measured.
        The code assumes that the first `known_dvs` inputs are measured disturbance variables.
    """
    def __init__(self, G: utils.InternalDelay, N, M, P, dt_model,
                 Q=None, R=None, constraints=lambda mpc: [],
                 integrators=True, dvs=0, known_dvs=0):
        self.G = G
        self.N = N
        self.M = M
        self.P = P
        self.dt_model = dt_model
        self.Q = Q
        self.R = R
        self.constraints = constraints
        self.dvs = dvs
        self.known_dvs = known_dvs

        # Step model
        self.SM = StepModel.StepModel(G, dt_model, N, P, M, integrators=integrators, dvs=dvs)
        assert self.SM.ins >= self.dvs >= self.known_dvs

        # MPC
        self.MPC = ModelPredictiveController.ModelPredictiveController(self.SM, Q=Q, R=R, constraints=self.constraints)

        # Plant model
        self.PM = PlantModel.PlantModel(G)

    def simulate(self, Ysp, t_sim,
                 dt_control=None, Udv=lambda t: [],
                 show_tqdm=True, live_plot=False, save_data=''):
        """Runs a simulation of the closed loop system
        Parameters
        ----------
        Ysp : callable
            A function that takes one variable `t`, the current time and
            returns an array_like of set points for the outputs

        t_sim : array_like
            The times at which the simulation runs

        dt_control : float, optional
            The period of the control calculations
            Defaults to dt_model

        Udv : callable, optional
            A function that takes one variable `t`, the current time and
            returns an array_like of values for the distrubance variables

        show_tqdm : bool, optional
            If `True` then tqdm.tqdm will be augmented over the main simulation loop.
            Defaults to `True`

        live_plot : bool, optional
            If `True` then the graph of the simulation will be live as the results are calculated.
            Defaults to `False`

        save_data : string, optional
            If the string is not empty, then the results from the simulation are
            saved to an excel file with this name.
            Defaults to an empty string

        Returns
        -------
        df : pandas.DataFrame
            A DataFrame object with the results of the simulation
        """
        dt_sim = t_sim[1]
        ys = []
        us = []
        ysp = []
        dvs = []

        if dt_control is None:
            dt_control = self.dt_model
        t_next_control = dt_control
        dv_prev_control = numpy.zeros(self.dvs)

        us.append(self.MPC.step(Y_actual=[0] * self.SM.outs, Ysp=Ysp(0)))

        u_pm = list(us[-1]) + list(Udv(0))
        ys.append(self.PM.step(u_pm, dt_sim))
        ysp.append(Ysp(0))
        if self.SM.dvs:
            dvs.append(Udv(0))

        if live_plot:
            plt.ion()

        # Simulate
        t_sim_iter = tqdm.tqdm(t_sim[1:]) if show_tqdm else t_sim[1:]
        for t in t_sim_iter:
            dv = Udv(t)
            u_pm = list(us[-1]) + list(dv)
            ys.append(self.PM.step(u_pm, dt_sim))
            ysp.append(Ysp(t))
            if t > t_next_control:
                dDV = list(dv - dv_prev_control)[:self.known_dvs] + [0]*(self.dvs - self.known_dvs)

                du = self.MPC.step(Y_actual=ys[-1], Ysp=Ysp(t), MV_actual=us[-1], dDVs=numpy.array(dDV))
                us.append(us[-1] + du)

                t_next_control += dt_control
                dv_prev_control = dv
            else:
                us.append(us[-1])

            if self.SM.dvs:
                dvs.append(dv)

            if live_plot:
                t_sim_trunc = t_sim[t_sim <= t]

                plt.subplot(2, 1, 1)
                plt.cla()
                plt.plot(t_sim_trunc, ys, '-')
                plt.plot(t_sim_trunc, ysp, '--')
                plt.xlim(numpy.min(t_sim_trunc), numpy.max(t_sim_trunc))
                plt.ylim([numpy.min(ys) - numpy.std(ys), numpy.max(ys) + numpy.std(ys)])
                plt.legend([rf"${name}_{i+1}$" for i in range(self.SM.ins) for name in ['y', 'r']])

                plt.subplot(2, 1, 2)
                plt.cla()
                plt.plot(t_sim_trunc, us, '-')
                plt.xlim(numpy.min(t_sim_trunc), numpy.max(t_sim_trunc))
                plt.ylim([numpy.min(us) - numpy.std(us), numpy.max(us) + numpy.std(us)])
                plt.legend([rf"$u_{i+1}$" for i in range(self.SM.ins)])

                plt.pause(0.01)

        if live_plot:
            plt.ioff()

        data = numpy.concatenate([numpy.array(d) for d in [t_sim[:, numpy.newaxis], us, dvs, ys, ysp]], axis=1)
        cols = ['ts']
        cols += [f"u_{i+1}" for i in range(self.SM.mvs)]
        cols += [f"dv_{i + 1}" for i in range(self.SM.dvs)]
        cols += [f"y_{i + 1}" for i in range(self.SM.outs)]
        cols += [f"r_{i + 1}" for i in range(self.SM.outs)]

        df = pandas.DataFrame(data, columns=cols)

        if save_data != '':
            df.to_csv(save_data + '.csv', index=False)

        return df

    def change_Q(self, Q):
        """Changes the value of the tuning parameter Q for this class
        and it's internal MPC class

        Parameters
        ----------
        Q : 2d-array_like
            A matrix containing the the Q tuning matrix.
            Q tunes the importance of outputs in the control calculations
        """
        self.Q = Q
        self.MPC.Q = Q

    def change_R(self, R):
        """
        Changes the value of the tuning parameter R for this class
        and it's internal MPC class

        Parameters
        ----------
        R : 2d-array_like
            A matrix containing the the R tuning matrix.
            R tunes the importance of inputs in the control calculations
        """
        self.R = R
        self.MPC.R = R
