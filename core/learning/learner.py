from .utils import differentiate_vec
from sklearn import linear_model
from scipy.linalg import expm
from numpy import array, concatenate, zeros, dot, linalg, eye, ones, std, where, divide, multiply, tile, argwhere, diag, copy, ones_like
from .basis_functions import BasisFunctions

class Learner():
    '''
    Base class for learning methods.
    Overload fit and predict for specific methods.
    '''
    def __init__(self, ):
        pass

    def fit(self, X, X_dot, U):
        """
        Fit a learner

        Inputs:
        - X: state with all trajectories, numpy 3d array [NtrajxN, ns]
        - X_dot: time derivative of the state
        - U: control input, numpy 3d array [NtrajxN, nu]
        - t: time, numpy 2d array [Ntraj, N]
        """
        pass

    def process(self, X, X_d, U, t):
        """process filter data
        
        Arguments:
            X {numpy array []} -- state
            X_d {numpy array []} -- desired state
            U {numpy array } -- command
            t {numpy array [Nt,]} -- time vector
        
        Returns:
            [type] --  state flat, desired state flat, lifted state flat, 
                       lifted state dot flat, control flat, nominal control flat, time flat
        """
        # Remove points where not every input is available:
        if U.shape[1] < X.shape[1]:
            X = X[:,:U.shape[1],:]
            X_d = X_d[:, :U.shape[1], :]
            t = t[:,:U.shape[1]]

        Ntraj = X.shape[0]  # Number of trajectories in dataset
        n_data = X.shape[0] * X.shape[1]
        m = U.shape[2]
        n = X.shape[2]
        order = 'F'

        X_dot = array([differentiate_vec(X[ii,:,:],t[ii,:]) for ii in range(Ntraj)])

        #Vectorize remaining data
        X_flat, X_d_flat, X_dot_flat, U_flat, t_flat = X.transpose().reshape((n,n_data),order=order), \
                                                                           X_d.transpose().reshape((n, n_data),order=order), \
                                                                           X_dot.transpose().reshape((n, n_data),order=order), \
                                                                           U.transpose().reshape((m,n_data),order=order), \
                                                                           t.transpose().reshape((1,n_data),order=order)

        return X_flat, X_d_flat, X_dot_flat, U_flat, t_flat

    def predict(self,X, U):
        pass