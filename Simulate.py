import StepModel
import PlantModel
import ModelPredictiveController
import utils
import numpy
import matplotlib.pyplot as plt
import tqdm
import pandas


class SimulateMPC:
    def __init__(self, G: utils.InternalDelay, N, M, P, dt_model, Q=None, R=None, integrators=True):
        self.G = G
        self.N = N
        self.M = M
        self.P = P
        self.dt_model = dt_model
        self.Q = Q
        self.R = R

        # Step model
        self.SM = StepModel.StepModel(G, dt_model, N, P, M, integrators=integrators)

        # MPC
        self.MPC = ModelPredictiveController.ModelPredictiveController(self.SM, Q=Q, R=R)

        # Plant model
        self.PM = PlantModel.PlantModel(G)

    def simulate(self, Ysp, t_sim, dt_control=None, show_tqdm=True, live_plot=False, save_data=''):
        dt_sim = t_sim[1]
        ys = []
        us = []
        ysp = []

        if dt_control is None:
            dt_control = self.dt_model
        t_next_control = dt_control

        us.append(self.MPC.step([0]*self.SM.ins))
        ys.append(self.PM.step(us[-1], dt_sim))
        ysp.append(Ysp(0)[::self.P])

        if live_plot:
            plt.ion()

        # Simulate
        t_sim_iter = tqdm.tqdm(t_sim[1:]) if show_tqdm else t_sim[1:]
        for t in t_sim_iter:
            ys.append(self.PM.step(us[-1], dt_sim))
            ysp.append(Ysp(t)[::self.P])
            if t > t_next_control:
                du = self.MPC.step(ys[-1], Ysp(t))
                us.append(us[-1] + du)
                t_next_control += dt_control
            else:
                us.append(us[-1])

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

        data = numpy.concatenate([numpy.array(d) for d in [t_sim[:, numpy.newaxis], us, ys, ysp]], axis=1)
        cols = ['ts'] + [f"{name}_{i+1}" for name in ['u', 'y', 'r'] for i in range(self.SM.ins)]
        df = pandas.DataFrame(data, columns=cols)

        if save_data != '':
            df.to_csv(save_data + '.csv', index=False)

        return df
