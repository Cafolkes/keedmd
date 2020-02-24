from .utils import differentiate_vec
from sklearn import linear_model
from scipy.linalg import expm
from numpy import array, concatenate, zeros, dot, linalg, eye, ones, std, where, divide, multiply, tile, argwhere, diag, copy, ones_like
from .basis_functions import BasisFunctions
from .learner import Learner

class InverseKalmanFilter(Learner):
    '''
    Base class for learning methods.
    Overload fit and predict for specific methods.
    '''
    def __init__(self, B_ensemble):
        self.B_ensemble = B_ensemble

    def fit(self, X, X_dot, U):
        """
        Fit a learner

        Inputs:
        - X: state with all trajectories, numpy 3d array [NtrajxN, ns]
        - X_dot: time derivative of the state
        - U: control input, numpy 3d array [NtrajxN, nu]
        - t: time, numpy 2d array [Ntraj, N]
        """

        # TODO: Implement inverse Kalman Filter
        self.B_ensemble = [self.B_ensemble[0]*1.01, self.B_ensemble[1], self.B_ensemble[2]*0.98]

    def predict(self,X, U):
        pass