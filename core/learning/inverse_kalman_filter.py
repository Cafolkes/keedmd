from .utils import differentiate_vec
from sklearn import linear_model
from scipy.linalg import expm
from numpy import array, concatenate, zeros, dot, linalg, eye, ones, std, where, divide, multiply, tile, argwhere, diag, copy, ones_like
from .basis_functions import BasisFunctions
from .learner import Learner
from .eki import EKI
import numpy as np



class InverseKalmanFilter(Learner):
    '''
    Base class for learning methods.
    Overload fit and predict for specific methods.
    '''
    def __init__(self, A, TrueB, eta_0, B_ensemble):

        self.B_ensemble = B_ensemble
        self.Ns = self.B_ensemble.shape[0]
        self.Nu = self.B_ensemble.shape[1]
        self.Ne = self.B_ensemble.shape[2]
        B_ensemble_flat = np.reshape(B_ensemble, (self.Nu*self.Ns,self.Ne))
        G = lambda theta,y: 0

        self.eki = EKI(B_ensemble_flat, G, eta_0, 
              true_theta=TrueB, maxiter=100, max_error= 1e-6)
        self.Bshape = TrueB.shape

    def fit(self, X, X_dot, U):
        """
        Fit a learner

        Inputs:
        - X: state with all trajectories, numpy 3d array [NtrajxN, ns]
        - X_dot: time derivative of the state
        - U: control input, numpy 3d array [NtrajxN, nu]
        - t: time, numpy 2d array [Ntraj, N]
        """

        
        shrink_debug = True
        if (shrink_debug):
            shrink_rate = 0.6
            B_mean = np.mean(self.B_ensemble,axis=2)
            self.new_ensamble = self.B_ensemble
            for i in range(self.Ne):
                self.new_ensamble[:,:,i] = B_mean + shrink_rate*(self.B_ensemble[:,:,i]-B_mean)
            print(f"new ensamble {self.new_ensamble}")
        else:

            nk = 1
            if (Yraw.shape[1]>nk+1):
                Ym = Yraw[:,nk:]-Yraw[:,:-nk]
                Ym_flat = Ym.flatten()
            else:
                Ym_flat = np.array([])
            self.eki.G = lambda Bflat: self.Gdynamics(Bflat,X,X_dot,nk=nk,dt=dt)

            self.new_ensamble = self.eki.solveIP(self.B_ensemble, Yflat)

        self.B_ensemble = self.new_ensamble.copy()
        return self.new_ensamble

    def Gdynamics(self,Bflat, X, X_dot, nk, dt):

        N = Yraw.shape[0]
        Ny = Yraw.shape[1]
        Ng = Ny - nk
        G = np.zeros((N,Ng))

        # #! Prep matrices for prediction
        # # build A^{nk}
        # self.Ank = np.expm(self.A*dt*nk)
        
        # # build ABM
        # self.ABM = np.zeros((Ns,Ns*nk))
        # self.An = A
        # for i in range(nk):
        #     self.An = self.An @ A
        #     self.ABM = np.hstack((self.ABM,self.An))

        # B = Bflat.reshape(self.Bshape)
        # Goutput = np.zeros()
        # Ny = X_dot.len()
        # for i in range(nk,Ny):
        #     Goutput[:,i] = 

        
        

    def predict(self,X, U):
        pass