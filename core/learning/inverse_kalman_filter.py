from .utils import differentiate_vec
from sklearn import linear_model
from scipy.linalg import expm
from numpy import array, concatenate, zeros, dot, linalg, eye, ones, std, where, divide, multiply, tile, argwhere, diag, copy, ones_like
from .basis_functions import BasisFunctions
from .learner import Learner
from .eki import EKI
import numpy as np
import scipy.signal as signal
from scipy.integrate import solve_ivp




class InverseKalmanFilter(Learner):
    '''
    Transforms a parametrized dynamics problem as a Inverse Kalman Inversion problem
    '''
    def __init__(self, A, TrueB, eta_0, B_ensemble, dt, nk):

        self.B_ensemble = B_ensemble
        self.Ns = self.B_ensemble.shape[0]
        self.Nu = self.B_ensemble.shape[1]
        self.Ne = self.B_ensemble.shape[2]
        B_ensemble_flat = np.reshape(B_ensemble, (self.Nu*self.Ns,self.Ne))
        G = lambda theta,y: 0

        self.eki = EKI(B_ensemble_flat, G, eta_0, 
              true_theta=TrueB, maxiter=100, max_error= 1e-6)
        self.Bshape = TrueB.shape
        self.dt = dt


        # #! Prep matrices for prediction
        # build A^{nk}
        lin_model_d = signal.cont2discrete((A,TrueB,zeros((ns,1))),dt)
        Ad = lin_model_d[0]
        Bd = lin_model_d[1]
        xpm = np.expm(self.A*self.dt*nk)
        
        # # build ABM
        self.ABM = np.zeros((Ns,Nu*nk))
        self.An = Bd
        for i in range(nk):
             self.ABM = np.hstack((self.ABM,self.An))
             self.An = self.An @ A
        self.Ank = self.An @ A


        # Test Prep Matrices
        check_ab = True
        if check_ab:
            x0  = np.random.rand(Ns)
            xd = x0.copy()
            xc = x0.copy()

            # Store data Init
            nsim = 100
            xst = np.zeros((Ns,nk))
            ust = np.zeros((Nu,nk))

            # Simulate in closed loop
            for i in range(nk):
                # Fake pd controller
                ctrl = np.zeros(Nu,) #np.random.rand(nu,)
                xd = Ad @ xd + Bd @ ctrl
                xc = solve_ivp(lambda t,x: A@x+TrueB@ctrl, [0, dt], xc, atol=1e-6, rtol=1e-6).y[:, -1] 
         
                # Store Data
                xst[:,i] = xd
                ust[:,i] = ctrl

            x_multistep = self.ABM@x0 + self.ABM@ust.flatten()
            print(f"multistep {x_multistep}")
            print(f"discrete {xd}")
            print(f"continous {xc}")
        

    def fit(self, X, X_dot, U):
        """
        Fit a learner

        Inputs:
        - X: state with all trajectories, numpy 3d array [NtrajxN, ns]
        - X_dot: time derivative of the state
        - U: control input, numpy 3d array [NtrajxN, nu]
        - t: time, numpy 2d array [Ntraj, N]
        """

        
        shrink_debug = False
        if (shrink_debug):
            shrink_rate = 0.6
            B_mean = np.mean(self.B_ensemble,axis=2)
            self.new_ensamble = self.B_ensemble
            for i in range(self.Ne):
                self.new_ensamble[:,:,i] = B_mean + shrink_rate*(self.B_ensemble[:,:,i]-B_mean)
        else:

            nk = 5
            Ym = X[:,nk:]-X[:,:-nk]
            Ym_flat = Ym.flatten()
            self.eki.G = lambda Bflat: self.Gdynamics(Bflat,X,X_dot,nk=nk,dt=dt)

            self.new_ensamble = self.eki.solveIP(self.B_ensemble, Ym_flat)

        self.B_ensemble = self.new_ensamble.copy()
        return self.new_ensamble

    def Gdynamics(self,Bflat, X, X_dot, nk):

        N = X.shape[0]
        Ny = X.shape[1]
        Ng = Ny - nk
        G = np.zeros((N,Ng))



        # B = Bflat.reshape(self.Bshape)
        # Goutput = np.zeros()
        # Ny = X_dot.len()
        # for i in range(nk,Ny):
        #     Goutput[:,i] = 

        
        

    def predict(self,X, U):
        pass