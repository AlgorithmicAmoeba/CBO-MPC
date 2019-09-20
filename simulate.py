import StepModel
import PlantModel
import ModelPredictiveController
import utils
import numpy
import cvxpy
import matplotlib.pyplot as plt
import tqdm

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

# Step model setup
sm = StepModel.StepModel(G, dt_model, N, P, M, integrators=False)


# MPC setup
def Ysp_fun(t):
    if t < 50:
        ans = numpy.array([1, 0])
    else:
        ans = numpy.array([1, 1])
    return ans.repeat(P)


Q = numpy.append(numpy.full(sm.P, 100), numpy.full(sm.P, 100))
R = numpy.append(numpy.full(sm.M, 1), numpy.full(sm.M, 1))

Ysp = Ysp_fun(0)
mpc = ModelPredictiveController.ModelPredictiveController(sm, Ysp=Ysp, Q=Q, R=R)

# Plant model setup
pm = PlantModel.PlantModel(G)

# Simulation setup
t_end = 100
tsim = numpy.linspace(0, t_end, t_end*10)
dt_sim = tsim[1]
ys = []
us = []
ysp = []

dt_control = dt_model*1
t_next_control = dt_control

us.append(mpc.step([0, 0]))
ys.append(pm.step(us[-1], dt_sim))
ysp.append(Ysp_fun(0)[::P])


live_plot = False
if live_plot:
    plt.ion()

# Simulate
for t in tqdm.tqdm(tsim[1:]):
    ys.append(pm.step(us[-1], dt_sim))
    ysp.append(Ysp_fun(t)[::P])
    if t > t_next_control:
        du = mpc.step(ys[-1], Ysp_fun(t))
        us.append(us[-1] + du)
        t_next_control += dt_control
    else:
        us.append(us[-1])

    if live_plot:
        x_data = tsim[tsim <= t]
        y_data = numpy.array(ys)
        ysp_data = numpy.array(ysp)
        u_data = numpy.array(us)

        plt.subplot(2, 1, 1)
        plt.cla()
        plt.plot(x_data, y_data, '-')
        plt.plot(x_data, ysp_data, '--')
        plt.xlim(numpy.min(x_data), numpy.max(x_data))
        plt.ylim([numpy.min(y_data) - numpy.std(y_data), numpy.max(y_data) + numpy.std(y_data)])
        plt.legend([r"$y_1$", r"$y_2$", r"$r_1$", r"$r_2$"])

        plt.subplot(2, 1, 2)
        plt.cla()
        plt.plot(x_data, u_data, '-')
        plt.xlim(numpy.min(x_data), numpy.max(x_data))
        plt.ylim([numpy.min(u_data) - numpy.std(u_data), numpy.max(u_data) + numpy.std(u_data)])
        plt.legend([r"$u_1$", r"$u_2$"])

        plt.pause(0.01)

plt.ioff()

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(tsim, ys, '-')
plt.plot(tsim, ysp, '--')
plt.xlim(numpy.min(tsim), numpy.max(tsim))
plt.ylim([numpy.min(ys) - numpy.std(ys), numpy.max(ys) + numpy.std(ys)])
plt.legend([r"$y_1$", r"$y_2$", r"$r_1$", r"$r_2$"])

plt.subplot(2, 1, 2)
plt.plot(tsim, us, '-')
plt.legend([r"$u_1$", r"$u_2$"])
plt.xlim(numpy.min(tsim), numpy.max(tsim))
plt.ylim([numpy.min(us) - numpy.std(us), numpy.max(us) + numpy.std(us)])
plt.show()
