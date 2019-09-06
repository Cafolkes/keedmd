from core.learning_keedmd.edmd import Edmd
from sklearn import linear_model
from numpy import array, concatenate, zeros, dot, linalg, eye, diag, std, divide, tile, where


class Keedmd(Edmd):
    def __init__(self, basis, system_dim, l1=0., l2=0., acceleration_bounds=None, override_C=True, K_p = None, K_d = None):
        super().__init__(basis, system_dim, l1=l1, l2=l2, acceleration_bounds=acceleration_bounds, override_C=override_C)
        self.K_p = K_d
        self.K_d = K_d
        if self.basis.Lambda is None:
            raise Exception('Basis provided is not an Koopman eigenfunction basis')
        elif self.K_p is None or self.K_p is None:
            raise Exception('Nominal controller gains not defined.')

    def fit(self, X, U, U_nom, t):
        X, Z, Z_dot, U, U_nom, t = self.process(X, U, U_nom, t)
        self.n_lift = Z.shape[0]

        if self.l1 == 0. and self.l2 == 0.:
            # Solve least squares problem to find A and B for velocity terms:
            input_vel = concatenate((Z,U),axis=0).transpose()
            output_vel = Z_dot[int(self.n/2):self.n,:].transpose()
            sol_vel = dot(linalg.pinv(input_vel),output_vel).transpose()
            A_vel = sol_vel[:,:self.n_lift]
            B_vel = sol_vel[:,self.n_lift:]

            # Construct A matrix
            self.A = zeros((self.n_lift, self.n_lift))
            self.A[:int(self.n / 2), int(self.n / 2):self.n] = eye(int(self.n / 2))  # Known kinematics
            self.A[int(self.n/2):self.n,:] = A_vel
            self.A[self.n:,self.n:] = diag(self.basis.Lambda)

            # Solve least squares problem to find B for position terms:
            input_pos = U.transpose()
            output_pos = (Z_dot[:int(self.n/2),:]-dot(self.A[:int(self.n/2),:],Z)).transpose()
            B_pos = dot(linalg.pinv(input_pos),output_pos).transpose()

            # Solve least squares problem to find B for eigenfunction terms:
            input_eig = (U - U_nom).transpose()
            output_eig = (Z_dot[self.n:, :] - dot(self.A[self.n:, :], Z)).transpose()
            B_eig = dot(linalg.pinv(input_eig), output_eig).transpose()

            # Construct B matrix:
            self.B = concatenate((B_pos, B_vel, B_eig), axis=0)

            if self.override_C:
                self.C = zeros((self.n,self.n_lift))
                self.C[:self.n,:self.n] = eye(self.n)
            else:
                raise Exception('Warning: Learning of C not implemented for structured regression.')

        else:
            l1_ratio = self.l1 / (self.l1 + self.l2)
            alpha = self.l1 + self.l2
            reg_model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False,
                                                         normalize=False, max_iter=1e5)

            # Solve least squares problem to find A and B for velocity terms:
            input_vel = concatenate((Z, U), axis=0).transpose()
            output_vel = Z_dot[int(self.n / 2):self.n, :].transpose()


            reg_model.fit(input_vel, output_vel)

            sol_vel = reg_model.coef_
            A_vel = sol_vel[:, :self.n_lift]
            B_vel = sol_vel[:, self.n_lift:]

            # Construct A matrix
            self.A = zeros((self.n_lift, self.n_lift))
            self.A[:int(self.n / 2), int(self.n / 2):self.n] = eye(int(self.n / 2))  # Known kinematics
            self.A[int(self.n / 2):self.n, :] = A_vel
            self.A[self.n:, self.n:] = diag(self.basis.Lambda)

            # Solve least squares problem to find B for position terms:
            input_pos = U.transpose()
            output_pos = (Z_dot[:int(self.n / 2), :] - dot(self.A[:int(self.n / 2), :], Z)).transpose()
            reg_model.fit(input_pos, output_pos)
            B_pos = reg_model.coef_


            # Solve least squares problem to find B for eigenfunction terms:
            input_eig = (U - U_nom).transpose()
            output_eig = (Z_dot[self.n:, :] - dot(self.A[self.n:, :], Z)).transpose()
            reg_model.fit(input_eig, output_eig)
            B_eig = reg_model.coef_

            # Construct B matrix:
            self.B = concatenate((B_pos, B_vel, B_eig), axis=0)

            if self.override_C:
                self.C = zeros((self.n, self.n_lift))
                self.C[:self.n, :self.n] = eye(self.n)
            else:
                raise Exception('Warning: Learning of C not implemented for structured regression.')

        self.A[self.n:,:self.n] -= dot(self.B[self.n:,:],concatenate((self.K_p, self.K_d), axis=1))

    def lift(self, X, t):
        Z = self.basis.lift(X, t)
        return concatenate((X.transpose(), Z),axis=1)
