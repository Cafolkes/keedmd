from core.learning_keedmd.basis_functions import BasisFunctions
from numpy import array, linalg, transpose, math, diag, dot, ones, zeros, reshape, unique, power, prod, exp, log, divide
from numpy import concatenate as npconcatenate
from itertools import combinations_with_replacement, permutations
from core.learning import differentiate
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Concatenate, concatenate
from keras import backend as K
from core.dynamics.linear_system_dynamics import LinearSystemDynamics
from core.controllers.constant_controller import ConstantController


class KoopmanEigenfunctions(BasisFunctions):
    """
    Class for construction and lifting using Koopman eigenfunctions
    """
    def __init__(self, n, max_power, A_cl):
        self.n = n
        self.max_power = max_power
        self.A_cl = A_cl
        self.Nlift = None
        self.Lambda = None
        self.basis = None
        self.eigfuncs_lin = None  #Eigenfunctinos for linearized autonomous dynamics xdot = A_cl*x
        self.scale_func = None  #Scaling function scaling relevant state space into unit cube
        self.diffeomorphism_model = None
        self.diffeomorphism = lambda q: q  #Diffeomorphism between linearized and true dynamics (to be learned)


    def construct_basis(self, ub=None, lb=None):
        self.eigfunc_lin = self.construct_linear_eigfuncs()
        self.scale_func = self.construct_scaling_function(ub,lb)
        self.basis = lambda q: self.eigfunc_lin(self.scale_func(self.diffeomorphism(q)))
        #print('Dimensional test: ', self.lift(ones((self.n,2))).shape)

    def construct_linear_eigfuncs(self):

        lambd, v = linalg.eig(self.A_cl)
        _, w = linalg.eig(transpose(self.A_cl))

        p = array([ii for ii in range(self.max_power+1)])
        combinations = array(list(combinations_with_replacement(p, self.n)))
        powers = array([list(permutations(c,self.n)) for c in combinations]) # Find all permutations of powers
        powers = unique(powers.reshape((powers.shape[0] * powers.shape[1], powers.shape[2])),axis=0)  # Remove duplicates

        linfunc = lambda q: dot(transpose(w), q)  # Define principal eigenfunctions of the linearized system
        eigfunc_lin = lambda q: prod(power(linfunc(q), transpose(powers)), axis=0)  # Create desired number of eigenfunctions
        self.Nlift = eigfunc_lin(ones((self.n,1))).shape[0]
        self.Lambda = log(prod(power(exp(lambd).reshape((self.n,1)), transpose(powers)), axis=0))  # Calculate corresponding eigenvalues


        return eigfunc_lin

    def construct_scaling_function(self,ub,lb):
        scale_factor = (ub-lb).reshape((self.n,1))
        scale_func = lambda q: divide(q, scale_factor)

        return scale_func

    def build_diffeomorphism_model(self, layer_width=100):
        def diffeomorphism_loss(input_tensor, output_tensor, Acl_tensor):
            def loss(y_true, y_pred):
                h_pred = y_pred[:,:4]
                xdot = y_pred[:,4:]
                y_true_red = y_true[:,:4]

                h_grad = K.gradients(output_tensor,input_tensor)[0]
                h_dot = K.dot(K.transpose(h_grad), xdot)
                y_pred_sum = h_dot - K.transpose(K.dot(Acl_tensor, K.transpose(h_pred)))
                return K.square(y_pred_sum - y_true_red)
            return loss

        # NN for h(x):
        x_input = Input(shape=(self.n,), dtype='float32', name='x_input')  #Input to h(x)
        h = Dense(units=layer_width, activation='relu')(x_input)  # Hidden layer #1
        h = Dense(units=layer_width, activation='relu')(h)  #Hidden layer #2
        h_output = Dense(units=self.n)(h) #Output layer

        # Pass xdot through network as well to allow desired loss function:
        xdot_input = Input(shape=(self.n,), dtype='float32', name='xdot_input')  #h_dot(x) = grad_x(h(x))*xdot

        # Concatenate into single output
        concatenated_output = concatenate([h_output, xdot_input])

        model = Model(inputs=[x_input, xdot_input], outputs=concatenated_output)
        model.compile(optimizer='adam', loss=diffeomorphism_loss(x_input, h_output, K.variable(self.A_cl)))
        self.diffeomorphism_model = model

    def fit_diffeomorphism_model(self, X, t, X_d):
        X, X_dot = self.process(X=X, t=t, X_d=X_d)
        #X_fit = npconcatenate((X,X_dot),axis=1) #TODO: Remove (?)
        y_target = (X_dot - (dot(self.A_cl,X.transpose())).transpose())
        y_fit = npconcatenate((y_target, zeros(y_target.shape)),axis=1)
        self.diffeomorphism_model.fit([X, X_dot], y_fit, epochs=1, batch_size=1)

        self.plot_eigenfunction_evolution(X[:,-200:])


    def process(self, X, t, X_d):
        # Shift dynamics to make origin a fixed point
        X_f = X_d[:,:,-1]
        X_shift = array([X[ii] - X_f[:,ii] for ii in range(len(X))])

        # Calculate numerical derivatives
        X_dot = array([differentiate(X_shift[ii,:,:],t) for ii in range(len(X))])
        clip = int((X_shift.shape[1]-X_dot.shape[1])/2)
        X_shift = X_shift[:,clip:-clip,:]
        assert(X_shift.shape == X_dot.shape)

        # Reshape to have input-output data
        X_shift = X_shift.reshape((X_shift.shape[0]*X_shift.shape[1],X_shift.shape[2]))
        X_dot = X_dot.reshape((X_dot.shape[0] * X_dot.shape[1], X_dot.shape[2]))

        return X_shift, X_dot

    def plot_eigenfunction_evolution(self, X):
        print(self.Lambda)
        print(diag(self.Lambda).shape)
        eigval_system = LinearSystemDynamics(A=diag(self.Lambda),B=zeros((self.Lambda.shape[0],1)))
        eigval_sim = ConstantController(eigval_system,0.)
        print(X.shape)
        x0 = X[:,0]


    def lift(self, q):
        return array([self.basis(q[:,ii].reshape((self.n,1))) for ii in range(q.shape[1])]).transpose()
