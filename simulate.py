import Simulate
import Plotting
import ModelPredictiveController
import utils
import numpy
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
M = 2
P = 4


def Ysp_fun(t):
    if t < 50:
        ans = numpy.array([1, 0])
    else:
        ans = numpy.array([1, 1])
    return ans


def Udv(t):
    if t < 100:
        ans = numpy.array([0])
    else:
        ans = numpy.array([10])
    return ans


def constraints(MPC: ModelPredictiveController.ModelPredictiveController):
    ans = []
    # limit on Y
    # v = numpy.full_like(MPC.E, 2)
    # ans.append(cvxpy.abs(MPC.Y) <= v)

    # limit on U
    # v = numpy.full_like(MPC.dMVs, 20)
    # ans.append(cvxpy.abs(MPC.MVs.repeat(MPC.SM.M) + MPC.dMVs) <= v)

    return ans


Q = numpy.append(numpy.full(P, 101.6), numpy.full(P, 100.6))
R = numpy.append(numpy.full(M, 5e-2), numpy.full(M, 5e-5))


# Simulation setup
t_end = 200
t_sim = numpy.linspace(0, t_end, t_end*10)

sim = Simulate.SimulateMPC(G, N, M, P, dt_model, Q, R, dvs=1, known_dvs=0, constraints=constraints)

if __name__ == "__main__":
    df = sim.simulate(Ysp_fun, t_sim, Udv=Udv, save_data="test", live_plot=False)
    Plotting.plot_all(df)
