from core.learning import differentiate_vec
from sklearn import linear_model
from scipy.linalg import expm
from numpy import array, concatenate, zeros, dot, linalg, eye, ones, std, where, divide, multiply, tile

class Edmd():
    '''
    Base class for edmd-type methods. Implements baseline edmd with the possible addition of l1 and/or l2 regularization.
    Overload fit for more specific methods.
    '''
    def __init__(self, basis, system_dim, l1=0., l2=0., acceleration_bounds=None, override_C=True):
        self.A = None
        self.B = None
        self.C = None
        self.basis = basis  # Basis class to use as lifting functions
        self.l1 = l1  # Strength of l1 regularization
        self.l2 = l2  # Strength of l2 regularization
        self.n = system_dim
        self.n_lift = None
        self.m = None
        self.override_C = override_C
        self.acceleration_bounds = acceleration_bounds #(nx1)

    def fit(self, X, U, U_nom, t):
        X, Z, Z_dot, U, U_nom, t = self.process(X, U, U_nom, t)
        if self.l1 == 0. and self.l2 == 0.:
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

        else:
            input = concatenate((Z.transpose(),U.transpose()),axis=1)
            output = Z_dot.transpose()
            l1_ratio = self.l1/(self.l1+self.l2)
            alpha = self.l1 + self.l2
            reg_model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, normalize=False, max_iter=1e5)
            reg_model.fit(input,output)

            self.A = reg_model.coef_[:self.n_lift,:self.n_lift]
            self.B = reg_model.coef_[:self.n_lift,self.n_lift:]
            if self.override_C:
                self.C = zeros((self.n,self.n_lift))
                self.C[:self.n,:self.n] = eye(self.n)
            else:
                raise Exception('Warning: Learning of C not implemented for regularized regression.')

    def process(self, X, U, U_nom, t):
        # Remove points where not every input is available:
        if U.shape[1] < X.shape[1]:
            X = X[:,:U.shape[1],:]
            t = t[:,:U.shape[1]]

        # Filter data to exclude non-continuous dynamics:
        if self.acceleration_bounds is not None:
            X_filtered, U_filtered, U_nom_filtered, t_filtered = self.filter_input(X, U, U_nom, t)
        else:
            X_filtered, U_filtered, U_nom_filtered, t_filtered = X, U, U_nom, t

        Ntraj = X_filtered.shape[0]  # Number of trajectories in dataset
        Z = array([self.lift(X_filtered[ii,:,:].transpose(), t_filtered[ii,:]) for ii in range(Ntraj)])  # Lift x
        Z_dot = array([differentiate_vec(Z[ii,:,:],t_filtered[ii,:]) for ii in range(Ntraj)])  #Numerical differentiate lifted state

        # Vectorize data
        n_data = Z.shape[0]*Z.shape[1]
        self.n_lift = Z.shape[2]
        self.m = U_filtered.shape[2]
        order = 'F'

        X_filtered, Z, Z_dot, U_filtered, U_nom_filtered, t_filtered = X_filtered.transpose().reshape((self.n,n_data),order=order), \
                                                                        Z.transpose().reshape((self.n_lift,n_data),order=order), \
                                                                        Z_dot.transpose().reshape((self.n_lift,n_data),order=order), \
                                                                        U_filtered.transpose().reshape((self.m,n_data),order=order), \
                                                                        U_nom_filtered.transpose().reshape((self.m, n_data),order=order), \
                                                                        t_filtered.transpose().reshape((1,n_data),order=order)

        return X_filtered, Z, Z_dot, U_filtered, U_nom_filtered, t_filtered

    def filter_input(self, X, U, U_nom, t):
        '''
        Calculates the numerical derivative of the data and removes points that lie outside of the accepted acceleration
        interval (used to filter out data points corresponding to impacts with the environment)
        :param X:
        :param U:
        :return:
        '''
        X_filtered = X
        U_filtered = U
        U_nom_filtered = U_nom
        t_filtered = t
        return X_filtered, U_filtered, U_nom_filtered, t_filtered

    def lift(self, X, t):
        Z = self.basis.lift(X, t)
        if not X.shape[1] == Z.shape[1]:
            Z = Z.transpose()
        one_vec = ones((1,Z.shape[1]))
        return concatenate((X,one_vec, Z),axis=0).transpose()

    def predict(self,X, U):
        return dot(self.C, dot(self.A,X) + dot(self.B, U))

    def discretize(self,dt):
        '''
        Discretizes continous dynamics
        '''
        self.dt = dt
        self.Bd = self.B*dt
        self.Ad = expm(self.A*self.dt)
        
