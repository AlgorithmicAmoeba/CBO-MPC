import Simulate
import plotting
import utils
import numpy

# Slightly adapted from  H/W assignment 9
num = [[[-0.045], [-0.048]], [[-0.23], [0.55]]]
den = [[[8.1, 1], [11, 1]], [[8.1, 1], [10, 1]]]  # [[[8.1, 1], [11, 1]], [[8.1, 1], [10, 1]]]
delay = [[0.5, 0.5], [1.5, 0.5]]
G = utils.InternalDelay.from_tf_coefficients(num, den, delay)

# Parameters
T = 70
dt_model = 1
N = int(T/dt_model)
M = 2
P = 4


def Ysp_fun(t):
    if t < 50:
        ans = numpy.array([1, 0])
    else:
        ans = numpy.array([1, 1])
    return ans.repeat(P)

Q = numpy.append(numpy.full(P, 100), numpy.full(P, 100))
R = numpy.append(numpy.full(M, 1), numpy.full(M, 1))

# Simulation setup
t_end = 100
t_sim = numpy.linspace(0, t_end, t_end*10)

sim = Simulate.SimulateMPC(G, N, M, P, dt_model, Q, R)
df = sim.simulate(Ysp_fun, t_sim, save_data="test")

plotting.plot_all(df)
