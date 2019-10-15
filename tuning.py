import simulate
import Tuner

sim = simulate.sim
Ysp = simulate.Ysp_fun
Udv = simulate.Udv
t_sim = simulate.t_sim

tuner = Tuner.Tuner(sim, Ysp, t_sim, Udv=Udv)
print(tuner.ISE())