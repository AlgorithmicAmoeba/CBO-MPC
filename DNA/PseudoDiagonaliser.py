import numpy
import scipy.optimize


class PseudoDiagonaliser:
    """Designs a compensator the maximises the diagonalisation of G

    Parameters
    ----------
    order : int
        The order of the numerator and denominator terms

    ws : array_like
        The frequencies over which to evaluate the compensator

    G : callable
        The system which needs to be diagonalised in the form of a function
        that takes in a complex scalar `s` and returns the TF matrix evaluated
        at `s`

    row_col : bool, optional
        If `True` then the compensator is optimized for row dominance
        else it is optimized for column dominance
        Defaults to `True`

    show : bool, optional
            Prints results from optimization if `True`
            Defaults to `False`

    Attributes
    -----------
    order : int
        The order of the numerator and denominator terms

    ws : array_like
        The frequencies over which to evaluate the compensator

    G : callable
        The system which needs to be diagonalised in the form of a function
        that takes in a complex scalar `s` and returns the TF matrix evaluated
        at `s`

    row_col : bool
        If `True` then the compensator is optimized for row dominance
        else it is optimized for column dominance

    show : bool
            Prints results from optimization if `True`
    """

    def __init__(self, G, order, ws, row_col=True, show=False):
        self.G = G
        self.order = order
        self.ws = ws
        self.row_col = row_col
        self.show = show

        self.N = self.G(0).shape[0]

    def __objective(self, params, save_fun=False):
        """Objective function for optimization

        Parameters
        ----------
        params : array_like
            List of parameters for the optimization

        save_fun : bool, optional
            If `True` then the diagonaliser is is stored as a class instance as well
            Defaults to `False`

        Returns
        -------
        biggest : float
            The largest non-diagonally dominant term

        """
        ax = 2 if self.row_col else 1

        things = numpy.split(params, 2 * self.N ** 2)
        nums = numpy.array([numpy.poly1d([0])] + [numpy.poly1d(t) for t in things[::2]])[1:]
        dens = numpy.array([numpy.poly1d([0])] + [numpy.poly1d(t) for t in things[1::2]])[1:]

        def fun(s):
            n = numpy.array([num(s) for num in nums])
            d = numpy.array([den(s) for den in dens])
            matrix = numpy.reshape(n / d, (self.N, self.N))
            return matrix

        fs = numpy.abs([self.G(1j * w) @ fun(1j * w) for i, w in enumerate(self.ws)])

        # Now need to calculate the row diagonal dominance
        diagonals = numpy.array([numpy.diag(f) for f in fs])
        off_diagonals_sum = numpy.sum(fs, axis=ax) - diagonals
        biggest = numpy.max(off_diagonals_sum / diagonals)
        if self.show:
            print(biggest)

        if save_fun:
            self.fun = fun
        return biggest

    def optimize(self, initial=None):
        """Find the optimal diagonaliser

        Parameters
        ----------
        initial : array_like, optional
        An initial guess for the optimizer
        Defaults to a random value

        Returns
        -------
        result : array_like
            The parameters of the optimized diagonaliser
        """
        self.order += 1

        if initial is None:
            initial = numpy.abs(numpy.random.rand(2 * self.order * self.N ** 2))
        result = scipy.optimize.minimize(self.__objective, initial).x
        self.__objective(result, True)
        return result
