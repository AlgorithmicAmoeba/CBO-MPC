import numpy
import scipy.integrate
import Simulate


class Tuner:
    def __init__(self, s: Simulate.SimulateMPC, Ysp, t_sim, error_method="ISE", **kwargs):
        self.s = s
        self.Ysp = Ysp
        self.t_sim = t_sim
        self.method = getattr(self, error_method)
        self.sim_kwargs = kwargs

        self.df = self.s.simulate(self.Ysp, t_sim, **kwargs)

    def get_errors(self):
        ts = self.df.ts
        ys = self.df[self.df.columns.str.contains('y')].values
        rs = self.df[self.df.columns.str.contains('r')].values

        es = ys - rs

        return ts, es

    def ISE(self):
        ts, es = self.get_errors()
        E = es**2
        ans = scipy.integrate.trapz(E, ts)
        return ans


