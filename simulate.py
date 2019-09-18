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
den = [[[1, -4, 5], [11, 1]], [[8.1, 1], [10, 1]]]  # [[[8.1, 1], [11, 1]], [[8.1, 1], [10, 1]]]
delay = [[0.5, 0.5], [1.5, 0.5]]
G = utils.InternalDelay.from_tf_coefficients(num, den, delay)

# Parameters
N = 50
dt_model = 1
M = 20
P = M+N

# Step model setup
sm = StepModel.StepModel(G, dt_model, N, P, M)

# MPC setup
Ysp = numpy.full(P*sm.outs, 10)
mpc = ModelPredictiveController.ModelPredictiveController(sm, Ysp=Ysp, dU_max=10, E_max=10)

# Plant model setup
pm = PlantModel.PlantModel(G)

# Simulation setup
t_end = 30
tsim = numpy.linspace(0, t_end, t_end*5)
dt_sim = tsim[1]
ys = []
us = []

dt_control = dt_model*5
t_next_control = dt_control

us.append(mpc.step([0, 0]))
ys.append(pm.step(us[-1], dt_sim))

for t in tqdm.tqdm(tsim[1:]):
    if t < t_next_control:
        du = mpc.step(ys[-1])
        us.append(us[-1] + du)
        t_next_control += dt_control
    else:
        us.append(us[-1])
    ys.append(pm.step(us[-1], dt_sim))


plt.plot(tsim, ys)
plt.show()

plt.plot(tsim, us)
plt.show()
