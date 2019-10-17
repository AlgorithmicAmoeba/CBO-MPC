import numpy
import scipy.integrate
import scipy.optimize
import Simulate


class Tuner:
    """Tunes parameters for Q and R for an MPC controller by minimizing
    the error of the outputs from a simulation.

    Parameters
    ----------
    simulation : Simulate.SimulateMPC
        Simulation object that is used for the objective function

    Ysp : callable
        A function that takes one variable `t`, the current time and
        returns an array_like of set points for the outputs

    t_sim : array_like
            The times at which the simulation runs

    error_method : {'ISE', 'ITAE', 'IAE'}, optional
        Method used to calculate the error
        Defaults to ISE

    weights : array_like, optional
        Holds values relating to the importance of each output
        Defaults to an equal weighting

    **simulation_options
        Keyword arguments for the simulation

    Attributes
    -----------
    simulation : Simulate.SimulateMPC
        Simulation object that is used for the objective function

    Ysp : callable
        A function that takes one variable `t`, the current time and
        returns an array_like of set points for the outputs

    t_sim : array_like
            The times at which the simulation runs

    error_method : {'ISE', 'ITAE', 'IAE'}
        Method used to calculate the error

    weights : array_like
        Holds values relating to the importance of each output

    **simulation_options
        Keyword arguments for the simulation

    df : pandas.Dataframe
        A DataFrame object with the results of the latest simulation
    """
    def __init__(self, simulation: Simulate.SimulateMPC, Ysp, t_sim,
                 error_method="ISE", weights=None, **simulation_options):
        self.simulation = simulation
        self.Ysp = Ysp
        self.t_sim = t_sim
        self.method = getattr(self, error_method)
        self.weights = weights
        if weights is None:
            self.weights = numpy.ones(self.simulation.SM.outs)
        self.simulation_options = simulation_options

        self.df = self.simulation.simulate(self.Ysp, t_sim, **simulation_options)

    def get_errors(self):
        """Calculate the errors of the most recent simulation

        Returns
        -------
        ts : array_like
            List of times of the simulation

        es : 2d-array_like
            An array containing the error for each output at each instance in time
        """
        ts = self.df.ts
        ys = self.df.loc[:, self.df.columns.str.contains('y')].values
        rs = self.df.loc[:, self.df.columns.str.contains('r')].values

        es = ys - rs

        return ts, es

    def ISE(self):
        """Calculates the Integral of the Square of the Error
        Returns
        -------
        error : float
            The total weighted error
        """
        ts, es = self.get_errors()
        E = es**2
        integrals = scipy.integrate.trapz(E.T, ts)
        error = sum(integrals * self.weights)
        return error

    def IAE(self):
        """Calculates the Integral of the Absolute of the Error
        Returns
        -------
        error : float
            The total weighted error
        """
        ts, es = self.get_errors()
        E = abs(es)
        integrals = scipy.integrate.trapz(E.T, ts)
        ans = sum(integrals*self.weights)
        return ans

    def ITAE(self):
        """Calculates the Integral of the Time Absolute of the Error
        Returns
        -------
        error : float
            The total weighted error
        """
        ts, es = self.get_errors()
        E = abs(es)*ts[:, numpy.newaxis]
        integrals = scipy.integrate.trapz(E.T, ts)
        ans = sum(integrals*self.weights)
        return ans

    def tune(self, initial, bounds, simple_tune=False):
        """Tune the parameters using SLSQP optimization

        Parameters
        ----------
        initial : array_like
            An initial guess of the parameters

        bounds : array_like
            An array containing bounds of the form (min, max)
            A bound of None, means unbounded

        simple_tune : bool
            If `True` then a simple tune is done, where only the individual
            input and output parameters are tuned and not the parameters per stage

        Returns
        -------
        ans : array_like
            The optimal parameters
        """
        if simple_tune:
            n_Qs, n_Rs = self.simulation.SM.outs, self.simulation.SM.mvs
        else:
            n_Qs, n_Rs = self.simulation.P * self.simulation.SM.outs, self.simulation.M * self.simulation.SM.mvs

        def obj(x):
            x = abs(x)
            x[x == 0] = 0.1
            if simple_tune:
                Q = numpy.diag(numpy.repeat(x[:n_Qs], self.simulation.SM.P))
                R = numpy.diag(numpy.repeat(x[n_Qs:], self.simulation.SM.M))
            else:
                Q = numpy.diag(x[:n_Qs])
                R = numpy.diag(x[n_Qs:])
            print(x)
            self.simulation.change_Q(Q)
            self.simulation.change_R(R)
            self.df = self.simulation.simulate(self.Ysp, self.t_sim, **self.simulation_options)
            value = self.method()
            print(value)
            return value

        ans = scipy.optimize.minimize(obj, initial, bounds=bounds, method="SLSQP").x
        return ans
