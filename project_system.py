import Simulate
import Plotting
import ModelPredictiveController
import Tuner
import utils
import numpy
import datetime
import cvxpy

# Slightly adapted from  H/W assignment 9: includes DV
num = [[[4.05], [1.77], [5.88]], [[5.39], [5.72], [6.90]], [[4.38], [4.42], [7.20]]]
den = [[[50, 1], [60, 1], [50, 1]], [[50, 1], [60, 1], [40, 1]], [[33, 1], [44, 1], [19, 1]]]
delay = [[27, 28, 27], [18, 14, 15], [20, 22, 0]]
G = utils.InternalDelay.from_tf_coefficients(num, den, delay)

# Parameters
T = 300
dt_model = 10
N = int(T/dt_model)
M = 13
P = N+M


def Ysp_fun(t):
    if t < 100:
        ans = numpy.array([1, 0, 0])
    elif t < 200:
        ans = numpy.array([1, 1, 0])
    else:
        ans = numpy.array([1, 1, 1])
    return ans


def constraints(MPC: ModelPredictiveController.ModelPredictiveController):
    ans = []
    # limit on Y
    v = numpy.full_like(MPC.Y, 2)
    ans.append(cvxpy.abs(MPC.Y) <= v)

    # limit on U
    # v = numpy.full_like(MPC.dMVs, 20)
    # ans.append(cvxpy.abs(MPC.MVs.repeat(MPC.SM.M) + MPC.dMVs) <= v)

    return ans


Q = numpy.concatenate([numpy.full(P, 0.513), numpy.full(P, 3.25), numpy.full(P, 2.37)])
R = numpy.concatenate([numpy.full(M, 0.468), numpy.full(M, 0.406), numpy.full(M, 0.036)])

# Simulation setup
t_end = 300
t_sim = numpy.linspace(0, t_end, t_end*10)

sim = Simulate.SimulateMPC(G, N, M, P, dt_model, Q, R)
tune = False

if tune:
    tuner = Tuner.Tuner(sim, Ysp_fun, t_sim, error_method="ISE")

    initial = [5, 4.96, 2.91, 1e-3, 2.4e-2, 0.98]

    bounds = [(1e-4, 100)]*len(initial)

    a = datetime.datetime.now()
    result = tuner.tune(initial, bounds, simple_tune=True)
    b = datetime.datetime.now()
    print("Total time: ", b - a)
    print(result)
else:
    df = sim.simulate(Ysp_fun, t_sim, save_data="test", live_plot=False)
    Plotting.plot_all(df)
