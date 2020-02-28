import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import dill
from itertools import combinations_with_replacement 


class EnKF():
    """EnKF 
    Dimensions:
    - n: size of the state space x
    - m: size of the measurements
    - Ne: number of ensamble members

    """
    def __init__(self,Q,R,f,h,dh,Ne):
        """EnKF 
        
        Arguments:
            Q {numpy array [n,n]} -- discrete process noise matrix
            R {numpy array [n,n]} -- measurement noise matrix
            f {lambda function of x and t, numpy array [n,]} -- RHS in \dot{x}=f(x,t)
            h {lambda function of x and t, numpy array [m,]} -- measurement function in y=h(x)
            dh {lambda function of x, numpy array [m,m]} -- dh/dx at current state estimate
            Ne {float>0} -- number of ensamble members
        """
        self.Q = Q
        self.R = R
        self.f = f
        self.h = h
        self.Ne = Ne
        self.dh = dh

        self.n = Q.shape[0]
        self.m = R.shape[0]

    def dynamic_update(self, xe, dt, atol=1e-10, rtol=1e-10):
        """dynamic_update Propagate the filter through time
        
        Arguments:
            xe {numpy array [n,Ne]} -- state space
            P {numpy array [n,n]} -- covariance
            dt {float>0} -- time step
        
        Keyword Arguments:
            atol {float>0} -- absolute tolerance for ivp solver (default: {1e-10})
            rtol {float>0} -- relative tolerance for ivp solver (default: {1e-10})
        
        Returns:
            numpy array [n,Ne] -- xnew, propagated ensambles
            numpy array [n,n] -- Pnew, experimental state covariance
        """
        #xe = np.random.multivariate_normal(np.zeros(self.n),P,self.Ne).T
        xnew = np.zeros((self.n,self.Ne))
        for i in range(self.Ne):
            xnew[:,i] = solve_ivp(self.f, [0, dt], xe[:,i], atol=atol, rtol=rtol).y[:, -1] + np.random.randn(self.n)*np.diag(self.Q)
            
        
        xmean = np.mean(xnew, axis=1)
        Pnew = np.zeros((self.n,self.n))
        for i in range(self.Ne):
            Pnew += (xnew[:,i]-xmean) @ (xnew[:,i]-xmean).T /self.Ne
        
        return xnew, Pnew


    def measurament_update(self,xe,P,y):
        """measurament_update 
        
        Arguments:
            xe {numpy array [n,Ne]} -- ensamble states
            P {numpy array [n,n]} -- experimental state covariance
            y {numpy array [m,]} -- measurement
        
        Returns:
            numpy array [n,] -- xnew,  new state after measurement
            numpy array [n,n] -- Pnew, new state covariance after measurement
        """

        H = self.dh(xe[:,0]) #TODO: implement 'averaged' H or drop non-linear h support
        S = H @ P @ H.T + self.R
        K = P @ H.T @ np.linalg.inv(S)
        xnew = np.zeros((self.n,self.Ne))
        # y = np.reshape(y,(10,1))
        for i in range(self.Ne):
            xnew[:,i] = xe[:,i] + K @ (y - self.h(xe[:,i]))                                    

        return xnew

    def update(self, x, dt, y):
        """update Update Ensamble Kalman Filter with time and measurement updates
        
        Arguments:
            x {numpy array [n,]} -- previous state vector
            dt {float>0} -- time step
            y {numpy array [m,]} -- measurement
        
        Returns:
            x numpy array [n,] -- [description]
        """
        x_plus, P_plus = self.dynamic_update(x, dt)
        return self.measurament_update(x_plus,P_plus,y)
    
class EKI():
    """Implements the Inverse Kalman Inversion. 

    It uses the model 

            y = G(theta) + eta

    where theta is a parametrization of your dynamics, eta is a gaussian noise, and G(theta)  is the forward function
    to compute the expected measurement. The

    Dimensions: 
        u: number of parameters to estimate
        Ne: number of ensemble members
        Ny: number of measurements

    """
    
    def __init__(self, u_0, G, eta_0, true_theta, maxiter, max_error):
        """
        Initialize the Ensemble Kalman Inversion
        
        Arguments:
            u_0 {numpy array [n,Ne]} -- initial ensemble array
            G {lambda function (u[n,Ne])} -- forward function
            eta_0 {float>0} -- measurement noise covariance
            true_theta {numpy array [n,]} -- real parameter value
            maxiter {int>0} -- maximum number of iterations
            max_error {float>0} -- maximum error to stop iterating
        """
        self.u0 = u_0
        self.G = G
        self.eta_0 = eta_0
        self.maxiter = maxiter
        self.max_error = max_error
        self.true_theta = true_theta

        self.n  = u_0.shape[0]
        self.Ne = u_0.shape[1]

        
        
    def solveIP(self,xe,Y):
        """
        Solve the Inverse Problem
        
        Arguments:
            xe {numpy array [n,Ne]} -- previous ensemble
            Y {numpy array [Ny,]} -- vector of measurements
        
        Returns:
            numpy array [n,Ne]  -- next ensemble
        """
        Ny = Y.shape[0]

        R = np.diag(np.ones(Y.size)*self.eta_0)
        xepast = xe

        for j in range(self.maxiter):
            
            ubar = np.mean(xe,axis=1,keepdims=True)
            Gxe = np.zeros((Ny,self.Ne))
            for i in range(self.Ne):
                Gxe[:,i] = self.G(xe[:,i])
            gbar = np.mean(Gxe,axis=1,keepdims=True)

            u_d = xe - ubar
            Gxe_d = Gxe - gbar
            Cug =    u_d @ Gxe_d.T
            Cgg =  Gxe_d @ Gxe_d.T
            K = Cug @ np.linalg.inv(Cgg + R) 
            Yd = np.reshape(Y, (-1, 1)) + np.random.normal(size=(Y.size,self.Ne))*self.eta_0
            xe = xe + K @ (Yd - Gxe)                                 

            error_v = np.mean(xe,axis=1)-self.true_theta
            dtheta_change = xe-xepast
            #error_dtheta = np.sqrt(dtheta_change.dot(dtheta_change))
            xepast = xe
            error = np.sqrt(error_v.dot(error_v))
            print(f'Iteration EKI {j+1}/{self.maxiter}. Error {error:.2f}')
            #if (abs(error) < self.max_error):
            #    break
        return xe