from sklearn.metrics.pairwise import rbf_kernel
from .basis_functions import BasisFunctions
from numpy import array, atleast_2d, tile, diag

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
        #return atleast_2d(array([self.basis(q[:, ii].reshape((self.n, 1)), t[ii]) for ii in range(q.shape[1])]).squeeze())
        a = atleast_2d(self.basis(q,t).squeeze())
        return a

    def construct_basis(self):
        if self.type == 'gaussian':
            self.basis = lambda q, t: array([diag(rbf_kernel(q.reshape(q.shape[1],self.n), tile(self.rbf_centers[ii,:],(q.shape[1],1)), self.gamma)).transpose() for ii in range(self.n_lift)])
        else:
            raise Exception('RBF kernels other than Gaussian not implemented')

