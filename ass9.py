import Simulate
import Plotting
import Tuner
import ModelPredictiveController
import utils
import numpy
import datetime
import cvxpy

# Slightly adapted from  H/W assignment 9: includes DV
num = [[[-0.045], [-0.048], [0.004]], [[-0.23], [0.55], [-0.65]]]
den = [[[8.1, 1], [11, 1], [8.5, 1]], [[8.1, 1], [10, 1], [9.2, 1]]]
delay = [[0.5, 0.5, 1], [1.5, 0.5, 1]]
G = utils.InternalDelay.from_tf_coefficients(num, den, delay)

# Parameters
T = 70
dt_model = 4
N = int(T/dt_model)
M = 8
P = N + M


def Ysp_fun(t):
    if t < 50:
        ans = numpy.array([-2, 0])
    else:
        ans = numpy.array([-2, 3])
    return ans


def Udv(t):
    if t < 100:
        ans = numpy.array([0])
    else:
        ans = numpy.array([5])
    return ans


def constraints(MPC: ModelPredictiveController.ModelPredictiveController):
    ans = []
    # limit on Y
    # v = numpy.full_like(MPC.E, 2)
    # ans.append(cvxpy.abs(MPC.Y) <= v)

    # limit on U
    N_mv = int(MPC.dMVs.shape[0]/2)
    v = numpy.concatenate([numpy.full(N_mv, 30), numpy.full(N_mv, 22)])
    ans.append(cvxpy.abs(MPC.MVs.repeat(MPC.SM.M) + MPC.dMVs) <= v)

    # limit on dU
    v = numpy.full_like(MPC.dMVs, 5)
    ans.append(cvxpy.abs(MPC.dMVs) <= v)

    return ans


Q = numpy.append(numpy.full(P, 101.6), numpy.full(P, 100.6))
R = numpy.append(numpy.full(M, 5e-2), numpy.full(M, 5e-5))
# Q = numpy.append(numpy.full(P, 1), numpy.full(P, 1))
# R = numpy.append(numpy.full(M, 1), numpy.full(M, 1))


# Simulation setup
t_end = 200
t_sim = numpy.linspace(0, t_end, t_end*10)

sim = Simulate.SimulateMPC(G, N, M, P, dt_model, Q, R, dvs=1, known_dvs=0, constraints=constraints)

tune = False
if tune:
    tuner = Tuner.Tuner(sim, Ysp_fun, t_sim, Udv=Udv, error_method="ISE")

    initial = [100, 100, 5, 5]

    bounds = [(1, 1000)] * len(initial)

    a = datetime.datetime.now()
    result = tuner.tune(initial, bounds, simple_tune=True)
    b = datetime.datetime.now()
    print("Total time: ", b - a)
    print(result)
else:
    df = sim.simulate(Ysp_fun, t_sim, Udv=Udv, save_data="data/cons", live_plot=False)

    tuner = Tuner.Tuner(sim, Ysp_fun, t_sim, Udv=Udv, error_method="ISE")
    tuner.df = df
    print(tuner.ISE())

    Plotting.plot_all(df, show=False)
    import matplotlib.pyplot as plt

    plt.subplot(2, 1, 1)
    plt.legend([r"$T_4$", r"$T_{14}$"])

    plt.subplot(2, 1, 2)
    plt.plot(df.ts, df.dv_1, '--')
    plt.legend([r"$F_R$", r"$F_S$", r"$X_F$"])
    plt.ylim(ymin=0)
    plt.savefig("data/cons")
    plt.show()
