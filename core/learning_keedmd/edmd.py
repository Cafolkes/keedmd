from .utils import differentiate_vec
from sklearn import linear_model
from scipy.linalg import expm
from numpy import array, concatenate, zeros, dot, linalg, eye, ones, std, where, divide, multiply, tile, argwhere, diag
from .basis_functions import BasisFunctions

class Edmd():
    '''
    Base class for edmd-type methods. Implements baseline edmd with the possible addition of l1 and/or l2 regularization.
    Overload fit for more specific methods.
    '''
    def __init__(self, basis=BasisFunctions(0,0), system_dim=0, l1=0., l1_ratio=0.5, acceleration_bounds=None, override_C=True):
        self.A = None
        self.B = None
        self.C = None
        self.basis = basis  # Basis class to use as lifting functions
        self.l1 = l1  # Strength of l1 regularization
        self.l1_ratio = l1_ratio  # Strength of l2 regularization
        self.n = system_dim
        self.n_lift = None
        self.m = None
        self.override_C = override_C
        self.acceleration_bounds = acceleration_bounds #(nx1)
        self.Z_std = ones((basis.Nlift+basis.n+1,1))

    def fit(self, X, X_d, Z, Z_dot, U, U_nom):
        """
        Fit a EDMD object with the given basis function

        Sizes:
        - Ntraj: number of trajectories
        - N: number of timesteps
        - ns: number or original states
        - nu: number of control inputs

        Inputs:
        - X: state with all trajectories, numpy 3d array [NtrajxN, ns]
        - X_d: desired state with all trajectories, numpy 3d array [NtrajxN, ns]
        - Z: lifted state with all trajectories, numpy[NtrajxN, ns]
        - Z: derivative of lifted state with all trajectories, numpy[NtrajxN, ns]
        - U: control input, numpy 3d array [NtrajxN, nu]
        - U_nom: nominal control input, numpy 3d array [NtrajxN, nu]
        - t: time, numpy 2d array [Ntraj, N]
        """

        if self.l1 == 0.:
            # Construct EDMD matrices as described in M. Korda, I. Mezic, "Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control":
            W = concatenate((Z_dot, X), axis=0)
            V = concatenate((Z, U), axis=0)
            VVt = dot(V,V.transpose())
            WVt = dot(W,V.transpose())
            M = dot(WVt, linalg.pinv(VVt))
            self.A = M[:self.n_lift,:self.n_lift]
            self.B = M[:self.n_lift,self.n_lift:]
            self.C = M[self.n_lift:,:self.n_lift]

            if self.override_C:
                self.C = zeros(self.C.shape)
                self.C[:self.n,:self.n] = eye(self.n)
                self.C = multiply(self.C, self.Z_std.transpose())

        else:
            # Construct EDMD matrices using Elastic Net L1 and L2 regularization
            input = concatenate((Z.transpose(),U.transpose()),axis=1)
            output = Z_dot.transpose()

            CV = False
            if CV:
                reg_model = linear_model.MultiTaskElasticNetCV(alphas=None, copy_X=True, cv=5, eps=0.001, fit_intercept=True,
                                        l1_ratio=self.l1_ratio, max_iter=1e6, n_alphas=100, n_jobs=None,
                                        normalize=False, positive=False, precompute='auto', random_state=0,
                                        selection='cyclic', tol=0.0001, verbose=0)
            else:
                reg_model = linear_model.ElasticNet(alpha=self.l1, l1_ratio=self.l1_ratio, fit_intercept=False, normalize=False, max_iter=1e5)
            reg_model.fit(input,output)
            print(reg_model)

            self.A = reg_model.coef_[:self.n_lift,:self.n_lift]
            self.B = reg_model.coef_[:self.n_lift,self.n_lift:]
            if self.override_C:
                self.C = zeros((self.n,self.n_lift))
                self.C[:self.n,:self.n] = eye(self.n)
                self.C = multiply(self.C, self.Z_std.transpose())
            else:
                raise Exception('Warning: Learning of C not implemented for regularized regression.')

    def process(self, X, X_d, U, U_nom, t):
        # Remove points where not every input is available:
        if U.shape[1] < X.shape[1]:
            X = X[:,:U.shape[1],:]
            X_d = X_d[:, :U.shape[1], :]
            t = t[:,:U.shape[1]]

        # Filter data to exclude non-continuous dynamics:
        if self.acceleration_bounds is not None:
            X_filtered, X_d_filtered, U_filtered, U_nom_filtered, t_filtered = self.filter_input(X, X_d, U, U_nom, t)
        else:
            X_filtered, X_d_filtered, U_filtered, U_nom_filtered, t_filtered = X, X_d, U, U_nom, t

        Ntraj = X_filtered.shape[0]  # Number of trajectories in dataset
        Z = array([self.lift(X_filtered[ii,:,:].transpose(), X_d_filtered[ii,:,:].transpose()) for ii in range(Ntraj)])  # Lift x

        # Vectorize Z- data
        n_data = Z.shape[0] * Z.shape[1]
        self.n_lift = Z.shape[2]
        self.m = U_filtered.shape[2]
        order = 'F'
        Z_vec = Z.transpose().reshape((self.n_lift, n_data), order=order)

        # Normalize data
        self.Z_std = std(Z_vec, axis=1)
        self.Z_std[argwhere(self.Z_std == 0.)] = 1.
        #self.Z_std[:self.n] = 1.  # Do not rescale states. Note: Assumes state is added to beginning of observables
        self.Z_std = self.Z_std.reshape((self.Z_std.shape[0], 1))
        Z_norm = array([divide(Z[ii,:,:], self.Z_std.transpose()) for ii in range(Z.shape[0])])

        Z_dot = array([differentiate_vec(Z_norm[ii,:,:],t_filtered[ii,:]) for ii in range(Ntraj)])  #Numerical differentiate lifted state


        #Vectorize remaining data
        X_filtered, X_d_filtered, Z, Z_dot, U_filtered, U_nom_filtered, t_filtered = X_filtered.transpose().reshape((self.n,n_data),order=order), \
                                                                       X_d_filtered.transpose().reshape((self.n, n_data),order=order), \
                                                                        Z_norm.transpose().reshape((self.n_lift, n_data),order=order), \
                                                                        Z_dot.transpose().reshape((self.n_lift,n_data),order=order), \
                                                                        U_filtered.transpose().reshape((self.m,n_data),order=order), \
                                                                        U_nom_filtered.transpose().reshape((self.m, n_data),order=order), \
                                                                        t_filtered.transpose().reshape((1,n_data),order=order)

        return X_filtered, X_d_filtered, Z, Z_dot, U_filtered, U_nom_filtered, t_filtered

    def filter_input(self, X, X_d, U, U_nom, t):
        '''
        Calculates the numerical derivative of the data and removes points that lie outside of the accepted acceleration
        interval (used to filter out data points corresponding to impacts with the environment)
        :param X:
        :param U:
        :return:
        '''
        X_filtered = X
        X_d_filtered = X_d
        U_filtered = U
        U_nom_filtered = U_nom
        t_filtered = t
        return X_filtered, X_d_filtered, U_filtered, U_nom_filtered, t_filtered

    def lift(self, X, X_d):
        Z = self.basis.lift(X, X_d)
        if not X.shape[1] == Z.shape[1]:
            Z = Z.transpose()
        one_vec = ones((1,Z.shape[1]))
        output_norm = divide(concatenate((X,one_vec, Z),axis=0),self.Z_std)
        return output_norm.transpose()

    def predict(self,X, U):
        return dot(self.C, dot(self.A,X) + dot(self.B, U))

    def tune_fit(self, X, X_d, Z, Z_dot, U, U_nom):
        l1_ratio = array([.1, .5, .7, .9, .95, .99, 1])  # Values to test

        # Construct EDMD matrices using Elastic Net L1 and L2 regularization
        input = concatenate((Z.transpose(), U.transpose()), axis=1)
        output = Z_dot.transpose()

        reg_model_cv = linear_model.MultiTaskElasticNetCV(l1_ratio=l1_ratio, fit_intercept=False, normalize=False, cv=5, n_jobs=-1, selection='random')
        reg_model_cv.fit(input, output)

        self.A = reg_model_cv.coef_[:self.n_lift, :self.n_lift]
        self.B = reg_model_cv.coef_[:self.n_lift, self.n_lift:]
        if self.override_C:
            self.C = zeros((self.n, self.n_lift))
            self.C[:self.n, :self.n] = eye(self.n)
            self.C = multiply(self.C,self.Z_std.transpose())

        else:
            raise Exception('Warning: Learning of C not implemented for regularized regression.')

        self.l1 = reg_model_cv.alpha_
        self.l1_ratio = reg_model_cv.l1_ratio_



    def discretize(self,dt):
        '''
        Discretizes continous dynamics
        '''
        self.dt = dt
        self.Bd = self.B*dt
        self.Ad = expm(self.A*self.dt)
        
