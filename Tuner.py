import numpy
import scipy.integrate
import scipy.optimize
import Simulate


class Tuner:
    def __init__(self, s: Simulate.SimulateMPC, Ysp, t_sim, error_method="ISE", weights=None, **kwargs):
        self.s = s
        self.Ysp = Ysp
        self.t_sim = t_sim
        self.method = getattr(self, error_method)
        self.weights = weights
        if weights is None:
            self.weights = numpy.ones(self.s.SM.outs)
        self.sim_kwargs = kwargs

        self.df = self.s.simulate(self.Ysp, t_sim, **kwargs)

    def get_errors(self):
        ts = self.df.ts
        ys = self.df.loc[:, self.df.columns.str.contains('y')].values
        rs = self.df.loc[:, self.df.columns.str.contains('r')].values

        es = ys - rs

        return ts, es

    def ISE(self):
        ts, es = self.get_errors()
        E = es**2
        integrals = scipy.integrate.trapz(E.T, ts)
        ans = sum(integrals*self.weights)
        return ans

    def IAE(self):
        ts, es = self.get_errors()
        E = abs(es)
        integrals = scipy.integrate.trapz(E.T, ts)
        ans = sum(integrals*self.weights)
        return ans

    def ITAE(self):
        ts, es = self.get_errors()
        E = abs(es)*ts[:, numpy.newaxis]
        integrals = scipy.integrate.trapz(E.T, ts)
        ans = sum(integrals*self.weights)
        return ans

    def tune(self, initial, simple_tune=False):
        if simple_tune:
            n_Qs, n_Rs = self.s.SM.outs, self.s.SM.mvs
        else:
            n_Qs, n_Rs = self.s.P*self.s.SM.outs, self.s.M*self.s.SM.mvs

        def obj(x):
            x = abs(x)
            x[x == 0] = 0.1
            if simple_tune:
                Q = numpy.diag(numpy.repeat(x[:n_Qs], self.s.SM.P))
                R = numpy.diag(numpy.repeat(x[n_Qs:], self.s.SM.M))
            else:
                Q = numpy.diag(x[:n_Qs])
                R = numpy.diag(x[n_Qs:])
            print(x)
            self.s.change_Q(Q)
            self.s.change_R(R)
            self.df = self.s.simulate(self.Ysp, self.t_sim, **self.sim_kwargs)
            value = self.method()
            print(value)
            return value
        options = {'xatol': 0.1, 'fatol': 0.1}
        return scipy.optimize.minimize(obj, initial, method="Nelder-Mead", options=options)
