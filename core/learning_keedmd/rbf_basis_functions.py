from sklearn.metrics.pairwise import rbf_kernel
from .basis_functions import BasisFunctions
from .utils import rbf
from numpy import array, atleast_2d, tile, diag, reshape

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
        self.Nlift = rbf_centers.shape[1]
        self.rbf_centers = rbf_centers
        self.gamma = gamma
        self.type = type
        self.Lambda = None
        self.basis = None

    def lift(self, q, q_d):
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
        if q.ndim == 1:
            q = reshape(q,(q.shape[0],1))

        return atleast_2d(self.basis(q, q_d).squeeze())

    def construct_basis(self):
        if self.type == 'gaussian':
            #self.basis = lambda q, t: array([diag(rbf_kernel(q.reshape(q.shape[1],self.n), tile(self.rbf_centers[ii,:],(q.shape[1],1)), self.gamma)).transpose() for ii in range(self.Nlift)])
            self.basis = lambda q, q_t: rbf(q, self.rbf_centers, eps=self.gamma)
        else:
            raise Exception('RBF kernels other than Gaussian not implemented')

