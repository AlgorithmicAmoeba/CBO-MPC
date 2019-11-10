import Simulate
import Plotting
import Tuner
import ModelPredictiveController
import utils
import numpy
import datetime
import cvxpy
import matplotlib.pyplot as plt

num = [[[-14.4115], [-0.043], [-0.375], [-0.903], [-0.0068], [7.68], [6.60]],
       [[0.277], [8.23e-4], [-3.6e-3], [-9.2e-3], [3.7e-5], [-0.0092], [0]],
       [[0.379], [-2.085e-3], [0.28], [0.686], [0], [0], [0]],
       [[0.00117], [-0.003], [0.4], [1.08], [-3.56], [0], [0]],
       [[7.5e-3], [1.12e-5], [1.34e-4], [3.3e-4], [9.0e-6], [0], [0]]]
den = [[[0.70, 1], [0.84, 1], [0.57, 1], [0.49, 1], [1.37, 1], [1.03, 1], [1.044, 1]],
       [[0.56, 1], [0.586, 1], [0.26, 1], [0.27, 1], [2.24, 1], [0.675, 1], [1]],
       [[0.445, 0.69, 1], [0.086, 0.25, 1], [0.19, 0.55, 1], [0.15, 0.517, 1], [1], [1], [1]],
       [[0.11, 0.2065, 1], [0.119, 0.55, 1], [0.207, 1], [0.27, 1], [2.39, 1], [1], [1]],
       [[0.55, 1], [0.6, 1], [0.064, 1], [0.176, 1], [0.178, 1], [1], [1]]]
delay = [[2.44, 2.09, 1.74, 1.68, 0.625, 1.53, 1.554],
         [1.27, 1.02, 0.58, 0.53, 0.93, 0.97, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0.68, 0.54, 0, 0, 0],
         [1.13, 1.26, 0.7, 0.54, 0.11, 0, 0]]
G = utils.InternalDelay.from_tf_coefficients(num, den, delay)

# Parameters
T = 8
dt_model = 0.2
N = int(T/dt_model)
M = 20
P = 70


def Ysp_fun(t):
    if t < 20:
        ans = numpy.array([1, 0, 0, 0, 0])
    # elif t < 40:
    #     ans = numpy.array([1, 1, 0, 0, 0])
    # elif t < 60:
    #     ans = numpy.array([1, 1, 1, 0, 0])
    # elif t < 80:
    #     ans = numpy.array([1, 1, 1, 1, 0])
    else:
        ans = numpy.array([1, 0, 0, 0, 0])
    return ans


def Udv(t):
    if t < 100:
        ans = numpy.array([0.1785, 0])
    # elif t < 120:
    #     ans = numpy.array([1, 0])
    else:
        ans = numpy.array([0.1785, 0])
    return ans


def constraints(MPC: ModelPredictiveController.ModelPredictiveController):
    ans = []
    # limit on Y
    # v = numpy.full_like(MPC.E, 2)
    # ans.append(cvxpy.abs(MPC.Y) <= v)

    # limit on U
    # N_mv = int(MPC.dMVs.shape[0]/2)
    # v = numpy.concatenate([numpy.full(N_mv, 30), numpy.full(N_mv, 22)])
    # ans.append(cvxpy.abs(MPC.MVs.repeat(MPC.SM.M) + MPC.dMVs) <= v)

    # limit on dU
    # v = numpy.full_like(MPC.dMVs, 5)
    # ans.append(cvxpy.abs(MPC.dMVs) <= v)

    return ans


Q = numpy.concatenate([numpy.full(P, 101), numpy.full(P, 50.8), numpy.full(P, 12.3),numpy.full(P, 9.85),numpy.full(P, 50.7)])
R = numpy.concatenate([numpy.full(M, 0.125),numpy.full(M, 0.307),numpy.full(M, 8.68e-5),numpy.full(M, 6.20), numpy.full(M, 3.39)])
# Simulation setup
t_end = 20
t_sim = numpy.linspace(0, t_end, t_end*10)

sim = Simulate.SimulateMPC(G, N, M, P, dt_model, Q, R, dvs=2, known_dvs=0, constraints=constraints)

df = sim.simulate(Ysp_fun, t_sim, Udv=Udv, save_data="data/cons", live_plot=False)

tuner = Tuner.Tuner(sim, Ysp_fun, t_sim, Udv=Udv, error_method="IAE", weights=[2, 1, 0.1, 0.1, 1])
tuner.df = df
print(tuner.IAE())

print(df.ts[numpy.where(abs(df.y_1 - 1) <1e-2)[0][0]])

# plt.figure(figsize=(20, 20))
# plt.rc('font', size=20)
# Plotting.plot_all(df, show=False)
# plt.savefig("data/exam_untuned")
# plt.show()


# initial = [100, 50, 10, 10, 50, 0.1, 1, 5, 5, 1]
#
# bounds = [(0.1, 1000)] * len(initial)
#
# a = datetime.datetime.now()
# result = tuner.tune(initial, bounds, simple_tune=True)
# b = datetime.datetime.now()
# print("Total time: ", b - a)
# print(result)