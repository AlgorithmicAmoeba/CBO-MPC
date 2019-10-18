import datetime

import numpy
import simulate
import Tuner

sim = simulate.sim
Ysp = simulate.Ysp_fun
Udv = simulate.Udv
# t_sim = simulate.t_sim
t_sim = numpy.linspace(0, 100, 1000)

tuner = Tuner.Tuner(sim, Ysp, t_sim, Udv=Udv, error_method="ISE")

initial = [100, 100, 5, 5]

bounds = [(1, 1000)]*len(initial)

a = datetime.datetime.now()
ans = tuner.tune(initial, bounds, simple_tune=True)
b = datetime.datetime.now()
print("Total time: ", b - a)
print(ans)
