from numpy import zeros
from numpy.linalg import eigvals

from .controller import Controller
from core.dynamics import AffineQuadCLF

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as sparse
import osqp



        self._osqp_Ad = sparse.eye(nx)+Ac*self.dt
        self._osqp_Bd = Bc*self.dt

        [nx, nu] = self._osqp_Bd.shape
        # Constraints
        
        umin = np.ones(nu)*0.1-self.u_hover
        umax = np.ones(nu)*0.9-self.u_hover
        xmin = np.array([-3,-3,0,-np.inf,-np.inf,-np.inf])
        xmax = np.array([ 5.0,5.0,10.0,3.,3.,4.])

        # Sizes
        ns = 6 # p_x, p_y, p_z, v_x, v_y, v_z
        nu = 3 # f_x, f_y, f_z

        # Objective function
        Q = sparse.diags([10., 10., 10., 1., 1., 1.])
        QN = Q
        R = 3.5*sparse.eye(nu)

        # Initial and reference states
        x0 = np.array([0.0,0.0,1.0,0.0,0.0,0.0])
        xr = np.array([0.,4.,8.0,0.,0.,0.0])

        # Prediction horizon
        N = int(self.rate*7.0)
        self._osqp_N = N

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                            sparse.kron(sparse.eye(N), R)]).tocsc()
        # - linear objective
        q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
               np.zeros(N*nu)])
        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), self._osqp_Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), self._osqp_Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N*nx)])
        ueq = leq
        # - input and state constraints
        Aineq = sparse.eye((N+1)*nx + N*nu)
        lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq]).tocsc()
        self._osqp_l = np.hstack([leq, lineq])
        self._osqp_u = np.hstack([ueq, uineq])

        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.prob.setup(P, q, A, self._osqp_l, self._osqp_u, warm_start=True)
        self.first = True