from .utils import differentiate_vec
from sklearn import linear_model
from scipy.linalg import expm
from numpy import array, concatenate, zeros, dot, linalg, eye, ones, std, where, divide, multiply, tile, argwhere, diag, copy, ones_like
from .basis_functions import BasisFunctions
from .learner import Learner
from .eki import EKI
import numpy as np
import scipy
import scipy.signal as signal
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


def hp(x):
    # use to to plot a numpy array
    import matplotlib.pyplot as plt
    plt.matshow(x)
    plt.colorbar()
    plt.show()

class InverseKalmanFilter(Learner):
    '''
    Transforms a parametrized dynamics problem as a Inverse Kalman Inversion problem
    '''
    def __init__(self, A, TrueB, eta_0, B_ensemble, dt, nk):

        self.A = A
        self.B_ensemble = B_ensemble
        self.Ns = self.B_ensemble.shape[0]
        self.Nu = self.B_ensemble.shape[1]
        self.Ne = self.B_ensemble.shape[2]
        B_ensemble_flat = np.reshape(B_ensemble, (self.Nu*self.Ns,self.Ne))
        G = lambda theta,y: 0

        self.eki = EKI(B_ensemble_flat, G, eta_0, 
              true_theta=TrueB.flatten(), maxiter=5, max_error= 1e-6)
        self.Bshape = TrueB.shape
        self.dt = dt
        self.nk = nk    
        self.get_multistep_matrices(TrueB)
    
    def get_multistep_matrices(self,B):
        # #! Prep matrices for prediction
        # build A^{nk}
        lin_model_d = signal.cont2discrete((self.A,B,np.identity(self.Ns),zeros((self.Ns,1))),self.dt)
        Ad = lin_model_d[0]
        Bd = lin_model_d[1]
        xpm = scipy.linalg.expm(self.A*self.dt*self.nk)
        
        # # build ABM as in x(k)=Ad^k+ABM @ uvector
        self.ABM = Bd 
        self.An  = Ad
        for i in range(self.nk-1):
            self.ABM = np.hstack([self.An @ Bd,self.ABM])
            self.An = self.An @ Ad

        # Test Prep Matrices
        check_ab = True
        if check_ab:
            x0  = np.random.rand(self.Ns)
            xd = x0.copy()
            xc = x0.copy()

            # Store data Init
            xst = np.zeros((self.Ns,self.nk))
            ust = np.zeros((self.Nu,self.nk))

            # Simulate in closed loop
            for i in range(self.nk):
                # Fake pd controller
                ctrl = np.zeros(self.Nu,) 
                ctrl = np.random.rand(self.Nu,)
                xd = Ad @ xd + Bd @ ctrl
                xc = solve_ivp(lambda t,x: self.A @ x + B @ ctrl, [0, self.dt], xc, atol=1e-6, rtol=1e-6).y[:, -1] 
         
                # Store Data
                xst[:,i] = xd
                ust[:,i] = ctrl

            #xc2 = solve_ivp(lambda t,x: self.A @ x + B @ ust[:,np.max([np.int(t/self.dt),self.nk-1])], [0, self.dt*self.nk], x0, atol=1e-6, rtol=1e-6).y[:, -1] 
            print(f"cont 2{xc2}")
            x_multistep = self.An@x0 + self.ABM@ust.flatten()
            print(f"multistep {x_multistep}")
            print(f"discrete {xd}")
            print(f"continous {xc}")
            print(f"ctrl")
        

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

            Ym = X[:,self.nk:]#-X[:,:-self.nk]
            Ym_flat = Ym.flatten()
            self.eki.G = lambda Bflat: self.Gdynamics(Bflat,X,U)
            self.B_ensemble_flat =  self.B_ensemble.reshape(-1, self.B_ensemble.shape[-1]) # [NsNu,Ne]
            print(f"new {self.B_ensemble_flat}")
            self.new_ensemble_flat = self.eki.solveIP(self.B_ensemble_flat, Ym_flat)
            print(f"new {self.B_ensemble_flat}")
            self.new_ensamble = self.new_ensemble_flat.reshape((self.Ns,self.Nu,self.Ne))
    
        self.B_ensemble = self.new_ensamble.copy()

    def Gdynamics(self,Bflat, X, U):
    
        Ny = X.shape[1]
        Ng = Ny - self.nk
        G = np.zeros((self.Ns,Ng))

        B = Bflat.reshape(self.Bshape)
        self.get_multistep_matrices(B)
        for i in range(Ng):
            xc = X[:,i]
            for ii in range(self.nk):
                ctrl = 
                xc = solve_ivp(lambda t,x: self.A @ x + B @ ctrl, [0, self.dt], xc, atol=1e-6, rtol=1e-6).y[:, -1] 
            #ctrl = U[:,i:i+self.nk]
            #f_x_dot = lambda t,x: self.A @ x + B @ ctrl[int(t/dt)]
            Xplus = solve_ivp(f_x_dot, [0, dt*nk], X[:,j], atol=1e-6, rtol=1e-6).y[:, -1] 
            G[:,j] = Xplus
            #G[:,i] = self.An @ X[:,i] + self.ABM @ U[:,i:i+self.nk].flatten()#-X[:,i]
        return G.flatten()
        

    def predict(self,X, U):
        pass