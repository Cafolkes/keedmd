from sklearn.metrics.pairwise import rbf_kernel
from .basis_functions import BasisFunctions

class RBF(BasisFunctions):
    """
    Implements Radial Basis Functions (RBD) as a basis function
    """

    def __init__(self, rbf_centers, n, gamma=1., type='gaussian'):
        """
        Parameters
        ----------
        n : int
            number of basis functions
        Nlift : int
            Number of lifing functions
        """
        self.n = n
        self.n_lift = rbf_centers.shape[0]
        self.rbf_centers = rbf_centers
        self.gamma = gamma
        self.type = type
        self.Lambda = None
        self.basis = None

    def lift(self, q, t):
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
        return self.basis(q, t)

    def construct_basis(self):
        if self.type == 'gaussian':
            self.basis = lambda q, t: rbf_kernel(q, self.rbf_centers, self.gamma)
        else:
            raise Exception('RBF kernels other than Gaussian not implemented')

