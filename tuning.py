import datetime

import numpy
import simulate
import Tuner

sim = simulate.sim
Ysp = simulate.Ysp_fun
Udv = simulate.Udv
t_sim = simulate.t_sim

tuner = Tuner.Tuner(sim, Ysp, t_sim, Udv=Udv)
# initial = [list(simulate.Q) + list(simulate.R)]
initial = [ 5.47606676e+02,  8.34201423e+02,  1.64844807e+03,  1.73601306e+01,
        5.72445677e+01,  3.99744437e+00,  7.95707623e+01,  3.79307415e+01,
        2.08173337e-01, -2.52593735e-03,  4.63515308e-02,  2.01792961e-01]
bounds = [(1, 1000)]*len(initial)
print(datetime.datetime.now())
ans = tuner.tune(bounds, initial)
print(datetime.datetime.now())
print(ans)
