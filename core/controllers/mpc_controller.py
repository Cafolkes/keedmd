

from numpy import zeros
from numpy.linalg import eigvals

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as sparse
import osqp
import matplotlib.pyplot as plt

from .controller import Controller
from core.dynamics.linear_system_dynamics import LinearSystemDynamics
from core.learning_keedmd import BasisFunctions, Edmd


class MPCController(Controller):
    """Class for controllers MPC.

    MPC are solved using osqp.

    Use lifting=True to solve MPC in the lifted space
    """
    def __init__(self, linear_dynamics, N, dt, umin, umax, xmin, xmax, Q, R, QN, xr, plotMPC=False, plotMPC_filename="",lifting=False, edmd_object=Edmd()):
        """Create an MPC Controller object.

        Sizes:
        - N: number of timesteps for predictions
        - Nqd: number of timesteps of the desired trajectory
        - nx: number of states (original or lifted)
        - ns: number or original states
        - nu: number of control inputs

        Inputs:
        - initilized linear_dynamics, LinearSystemDynamics object. It takes Ac and Bc from it.
        - number of timesteps, N, integer
        - time step, dt, float
        - minimum control,  umin, numpy 1d array [nu,]
        - maximum control,  umax, numpy 1d array [nu,]
        - minimum state,    xmin, numpy 1d array [ns,]
        - maximum state,    xmax, numpy 1d array [ns,]
        - state cost matrix    Q, sparse numpy 2d array [ns,ns]. In practice it is always diagonal. 
        - control cost matrix, R, sparse numpy 2d array [nu,nu]. In practice it is always diagonal. 
        - final state cost matrix,  QN, sparse numpy 2d array [ns,ns]. In practice it is always diagonal. 
        - reference state trajectory, xr, numpy 2d array [ns,Nqd] OR numpy 1d array [ns,]
        (Optional)
        - flag to plot MPC thoughts, plotMPC=False, boolean
        - filename to save the previosu plot, plotMPC_filename="", string
        - flag to use or not lifting, lifting=False, boolean
        - object to store the eDMD data, edmd_object=Edmd(). It has been initialized. s
        """

        Controller.__init__(self, linear_dynamics)

        # Load arguments
        Ac, Bc = linear_dynamics.linear_system()
        [nx, nu] = Bc.shape
        self.dt = dt
        self._osqp_Ad = sparse.eye(nx)+Ac*self.dt
        self._osqp_Bd = Bc*self.dt
        self.plotMPC = plotMPC
        self.plotMPC_filename = plotMPC_filename
        self.q_d = xr
        
        self.ns = xr.shape[0]

        self.Q = Q
        self.lifting = lifting

        self.nu = nu
        self.nx = nx

        # Total desired path
        if self.q_d.ndim==2:
            self.Nqd = self.q_d.shape[1]
            xr = self.q_d[:,:N+1]

        # Prediction horizon
        self.N = N
        x0 = np.zeros(nx)
        
        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        if (self.lifting):
            # Load eDMD objects
            self.C = edmd_object.C
            self.edmd_object = edmd_object

            # - quadratic objective
            CQC  = sparse.csc_matrix(np.transpose(edmd_object.C).dot(Q.dot(edmd_object.C)))
            CQNC = sparse.csc_matrix(np.transpose(edmd_object.C).dot(QN.dot(edmd_object.C)))
            P = sparse.block_diag([sparse.kron(sparse.eye(N), CQC), CQNC,
                                sparse.kron(sparse.eye(N), R)]).tocsc()

            # - linear objective
            QCT = np.transpose(Q.dot(edmd_object.C))
            QNCT = np.transpose(QN.dot(edmd_object.C))
            if (xr.ndim==1):
                q = np.hstack([np.kron(np.ones(N), -QCT.dot(xr)), -QNCT.dot(xr), np.zeros(N*nu)])
            elif (xr.ndim==2):
                q = np.hstack([np.reshape(-QCT.dot(xr),((N+1)*nx,)), np.zeros(N*nu)])

            # - input and state constraints
                Aineq = sparse.block_diag([edmd_object.C for i in range(N+1)]+[np.eye(N*nu)]).shape

        else:
            # - quadratic objective
            P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                                sparse.kron(sparse.eye(N), R)]).tocsc()
            # - linear objective
            if (xr.ndim==1):
                q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr), np.zeros(N*nu)])
            elif (xr.ndim==2):
                q = np.hstack([np.reshape(-Q.dot(xr),((N+1)*nx,)), np.zeros(N*nu)])

            # - input and state constraints
            Aineq = sparse.eye((N+1)*nx + N*nu)

        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), self._osqp_Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), self._osqp_Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N*nx)])

        lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])

        ueq = leq
        self._osqp_q = q

        A = sparse.vstack([Aeq, Aineq]).tocsc()
        self._osqp_l = np.hstack([leq, lineq])
        self._osqp_u = np.hstack([ueq, uineq])

        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.prob.setup(P, q, A, self._osqp_l, self._osqp_u, warm_start=True, verbose=False)

        if self.plotMPC:
            # Figure to plot MPC thoughts
            self.ff = plt.figure()
            plt.xlabel('Time(s)')
            plt.ylabel('Position(m)')
            plt.grid()


    def eval(self, x, t):
        '''
        Inputs:
        - state, x, numpy 1d array [ns,]
        - time, t, float
        '''

        N = self.N
        nu = self.nu
        nx = self.nx

        tindex = int(t/self.dt)
        
        ## Update inequalities
        if self.q_d.ndim==2: 
            
            # Update the local reference trajectory
            if (tindex+N) < self.Nqd: # if we haven't reach the end of q_d yet
                xr = self.q_d[:,tindex:tindex+N+1]
            else: # we fill xr with copies of the last q_d
                xr = np.hstack( [self.q_d[:,tindex:],np.transpose(np.tile(self.q_d[:,-1],(N+1-self.Nqd+tindex,1)))])

            # Construct the new _osqp_q objects
            if (self.lifting):
                QCT = np.transpose(self.Q.dot(self.C))                        
                self._osqp_q = np.hstack([np.reshape(-QCT.dot(xr),((N+1)*nx,)), np.zeros(N*nu)])                    
            else:
                self._osqp_q = np.hstack([np.reshape(-self.Q.dot(xr),((N+1)*nx,)), np.zeros(N*nu)])

        elif self.q_d.ndim==1:
            # Update the local reference trajectory
            xr = np.transpose(np.tile(self.q_d,N+1))

        # Lift the current state if necessary
        if (self.lifting): 
            x = np.transpose(self.edmd_object.lift(x.reshape((x.shape[0],1)),xr[:,0].reshape((xr.shape[0],1))))[:,0]
        
        self._osqp_l[:self.nx] = -x
        self._osqp_u[:self.nx] = -x

        self.prob.update(q=self._osqp_q, l=self._osqp_l, u=self._osqp_u)

        ## Solve MPC Instance
        self._osqp_result = self.prob.solve()

        # Check solver status
        if self._osqp_result.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        if self.plotMPC:
            self.plot_MPC(t, xr)
        return  self._osqp_result.x[-N*nu:-(N-1)*nu]

    def parse_result(self):
        return  np.transpose(np.reshape( self._osqp_result.x[:(self.N+1)*self.nx], (self.N+1,self.nx)))


    def plot_MPC(self, current_time, xr):
        '''
        Inputs:
        - current_time, t, float
        - local MPC reference trajectory, xr, numpy 2d array [ns,N]
        '''

        # Unpack OSQP results
        nu = self.nu
        nx = self.nx
        N = self.N

        osqp_sim_state = np.transpose(np.reshape( self._osqp_result.x[:(N+1)*nx], (N+1,nx)))
        #osqp_sim_forces = np.reshape( self._osqp_result.x[-N*nu:], (N,nu))

        if self.lifting:
            osqp_sim_state = np.dot(self.C,osqp_sim_state)

        # Plot
        pos = current_time/(self.Nqd*self.dt) # position along the trajectory
        time = np.linspace(current_time,current_time+N*self.dt,num=N+1)
        plt.plot(time,osqp_sim_state[0,:],color=[0,1-pos,pos])
        #plt.savefig(self.plotMPC_filename)


        """
        plt.plot(range(N),osqp_sim_forces)
        #plt.plot(range(nsim),np.ones(nsim)*umin[1],label='U_{min}',linestyle='dashed', linewidth=1.5, color='black')
        #plt.plot(range(nsim),np.ones(nsim)*umax[1],label='U_{max}',linestyle='dashed', linewidth=1.5, color='black')
        plt.xlabel('Time(s)')
        plt.grid()
        plt.legend(['fx','fy','fz'])
        plt.show()
        plt.savefig('mpc_debugging_fz.png')
        """



