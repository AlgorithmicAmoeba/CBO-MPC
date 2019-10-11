import StepModel
import PlantModel
import ModelPredictiveController
import utils
import numpy
import matplotlib.pyplot as plt
import tqdm
import pandas


class SimulateMPC:
    def __init__(self, G: utils.InternalDelay, N, M, P, dt_model,
                 Q=None, R=None, integrators=True,
                 dvs=0, known_dvs=0):
        self.G = G
        self.N = N
        self.M = M
        self.P = P
        self.dt_model = dt_model
        self.Q = Q
        self.R = R
        self.dvs = dvs
        self.known_dvs = known_dvs

        # Step model
        self.SM = StepModel.StepModel(G, dt_model, N, P, M, integrators=integrators, dvs=dvs)
        assert self.SM.ins >= self.dvs >= self.known_dvs

        # MPC
        self.MPC = ModelPredictiveController.ModelPredictiveController(self.SM, Q=Q, R=R)

        # Plant model
        self.PM = PlantModel.PlantModel(G)

    def simulate(self, Ysp, t_sim,
                 dt_control=None, Udv=lambda t: [],
                 show_tqdm=True, live_plot=False, save_data=''):
        dt_sim = t_sim[1]
        ys = []
        us = []
        ysp = []
        dvs = []

        if dt_control is None:
            dt_control = self.dt_model
        t_next_control = dt_control
        dv_prev_control = numpy.zeros(self.dvs)

        us.append(self.MPC.step([0] * self.SM.outs))

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
                dDV = list(dv_prev_control - dv)[self.known_dvs:] + [0]*(self.dvs - self.known_dvs)

                du = self.MPC.step(ys[-1], Ysp(t), dDVs=dDV)
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
