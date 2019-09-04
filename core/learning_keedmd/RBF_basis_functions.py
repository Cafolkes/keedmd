
import scipy

from .basis_functions import BasisFunctions

class RBF(BasisFunctions):
    """
    Implements Radial Basis Functions (RBD) as a basis function
    """

    def __init__(self, nodes, type='thin_plate'):
        """
        Parameters
        ----------
        n : int
            number of basis functions
        Nlift : int
            Number of lifing functions
        """
        self.n = n
        self.Nlift = 
        self.Lambda = None
        self.basis = scipy.interpolate.Rbf(nodes,type)

    def lift(self, q):
        """
        Call this function to get the variables in lifted space

        Parameters
        ----------
        q : numpy array
            State vector

        Returns
        -------
        basis applied to q
        """
        return self.basis(q)

    def construct_basis(self):
        pass