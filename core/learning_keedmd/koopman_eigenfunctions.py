from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title
from numpy import array, linalg, transpose, math, diag, dot, ones, zeros, reshape, unique, power, prod, exp, log, divide, linspace, square
from numpy import concatenate as npconcatenate
from itertools import combinations_with_replacement, permutations
from core.learning import differentiate
#from keras.models import Model
#from keras.layers import Input, Dense, Concatenate, concatenate
#from keras.regularizers import l1_l2
#from keras import backend as K
#from tensorflow import Session
from core.learning_keedmd.basis_functions import BasisFunctions
from core.dynamics.linear_system_dynamics import LinearSystemDynamics
from core.controllers.constant_controller import ConstantController
from torch import nn, cuda, optim, from_numpy, manual_seed, no_grad, mean, autograd, transpose as t_transpose, mm, matmul, zeros as t_zeros, sum, eye
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
from torch.autograd.gradcheck import zero_gradients
from torchviz import make_dot


class KoopmanEigenfunctions(BasisFunctions):
    """
    Class for construction and lifting using Koopman eigenfunctions
    """
    def __init__(self, n, max_power, A_cl, BK):
        self.n = n
        self.max_power = max_power
        self.A_cl = A_cl
        self.BK = BK
        self.Nlift = None
        self.Lambda = None
        self.basis = None
        self.eigfuncs_lin = None  #Eigenfunctinos for linearized autonomous dynamics xdot = A_cl*x
        self.scale_func = None  #Scaling function scaling relevant state space into unit cube
        self.diffeomorphism_model = None

    def construct_basis(self, ub=None, lb=None):
        self.eigfunc_lin = self.construct_linear_eigfuncs()
        self.scale_func = self.construct_scaling_function(ub,lb)
        self.basis = lambda q, t: self.eigfunc_lin(self.scale_func(self.diffeomorphism(q, t)))
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

    def diffeomorphism(self, q, t):
        q = q.transpose()
        self.diffeomorphism_model.eval()
        input = npconcatenate((q,t.reshape((1,1))),axis=1)
        diff_pred = self.diffeomorphism_model(from_numpy(input)).detach().numpy()
        return (q + diff_pred).transpose()


    def diffeomorphism_keras(self, q, t):
        q = q.transpose()
        diff_pred = self.diffeomorphism_model.predict([q, zeros(q.shape), t])
        return (q + diff_pred[:,:4]).transpose()

    def build_diffeomorphism_model(self, n_hidden_layers = 4, layer_width=20, l1=0., l2=0., batch_size = 32):
        # Set up model architecture for h(x,t):
        N, d_h_in, H, d_h_out = batch_size, self.n + 1, layer_width, self.n
        self.diffeomorphism_model= nn.Sequential(
            nn.Linear(d_h_in,H),
            nn.ReLU()
        )
        for ii in range(n_hidden_layers):
            self.diffeomorphism_model.add_module('dense_' + str(ii+1), nn.Linear(H,H))
            self.diffeomorphism_model.add_module('relu_' + str(ii + 1), nn.ReLU())
        self.diffeomorphism_model.add_module('output', nn.Linear(H,d_h_out))

        self.diffeomorphism_model = self.diffeomorphism_model.double()
        #TODO: Add l1 and l2 regularization
        #TODO: Add dropout

    def fit_diffeomorphism_model(self, X, t, X_d, learning_rate=1e-3, n_epochs=100, train_frac=0.8, l2=1e1):
        X, X_dot, X_d, t = self.process(X=X, t=t, X_d=X_d)
        y_target = X_dot - dot(self.A_cl, X.transpose()).transpose() - dot(self.BK, X_d.transpose()).transpose()
        y_fit = npconcatenate((y_target, zeros(y_target.shape)), axis=1)

        device = 'cuda' if cuda.is_available() else 'cpu'

        # Prepare data for pytorch:
        manual_seed(42)  # Fix seed for reproducibility
        X_tensor = from_numpy(npconcatenate((X,t.reshape((t.shape[0],1)),X_dot),axis=1)) #[x (1,4), t (1,1), x_dot (1,4)]
        y_tensor = from_numpy(y_target)
        X_tensor.requires_grad_(True)


        # Builds dataset with all data
        dataset = TensorDataset(X_tensor, y_tensor)
        # Splits randomly into train and validation datasets
        n_train = int(train_frac*X.shape[0])
        n_val = X.shape[0]-n_train
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
        # Builds a loader for each dataset to perform mini-batch gradient descent
        train_loader = DataLoader(dataset=train_dataset, batch_size=32)
        val_loader = DataLoader(dataset=val_dataset, batch_size=32)

        def diffeomorphism_loss(h_dot, zero_jacobian, y_true, y_pred, is_training):
            h_sum_pred = h_dot - t_transpose(mm(self.A_cl, t_transpose(y_pred, 1, 0)), 1, 0)
            if is_training:
                loss = mean((y_true-h_sum_pred)**2) + 100*mean((zero_jacobian**2))
            else:
                loss = mean((y_true-h_sum_pred)**2)
            return loss

        # Set up optimizer and learning rate scheduler:
        optimizer = optim.Adam(self.diffeomorphism_model.parameters(),lr=learning_rate,weight_decay=l2)
        lambda1 = lambda epoch: 0.95 ** epoch
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        self.A_cl = from_numpy(self.A_cl)

        def calc_gradients(xt, xdot, yhat, zero_input, yzero, is_training):
            xt.retain_grad()
            if is_training:
                zero_input.retain_grad()
                zero_jacobian = compute_jacobian(zero_input, yzero)[:,:,:self.n]
                #print('xt: ', xt)
                #print('zt: ', zero_input)
                #print('z_jac: ', zero_jacobian)
                #zero_jac_prod = eye(self.n).double()
                #zero_jac_prod.requires_grad_(True)
                #print(zero_jacobian.shape, zero_jac_prod.shape)
                #zero_jacobian = matmul(zero_jacobian, zero_jac_prod)
                #print(zero_jacobian.shape)
                #print(zero_jacobian)
                zero_jacobian.requires_grad_(True)
                zero_jacobian = zero_jacobian.squeeze()
            else:
                zero_jacobian = None

            jacobian = compute_jacobian(xt, yhat)[:,:,:self.n]
            #print('jac: ', jacobian)
            h_dot = matmul(jacobian, xdot.reshape((xdot.shape[0],xdot.shape[1],1)))
            h_dot = h_dot.squeeze()

            return h_dot, zero_jacobian

        def compute_jacobian(inputs, output):
            """
            :param inputs: Batch X Size (e.g. Depth X Width X Height)
            :param output: Batch X Classes
            :return: jacobian: Batch X Classes X Size
            """
            assert inputs.requires_grad

            num_classes = output.size()[1]

            jacobian = t_zeros((num_classes, *inputs.size())).double()
            grad_output = t_zeros((*output.size(),)).double()
            if inputs.is_cuda:
                grad_output = grad_output.cuda()
                jacobian = jacobian.cuda()

            for i in range(num_classes):
                zero_gradients(inputs)
                grad_output.zero_()
                grad_output[:, i] = 1
                output.backward(grad_output, retain_graph=True)
                jacobian[i] = inputs.grad

            return t_transpose(jacobian, dim0=0, dim1=1)

        def make_train_step(model, loss_fn, optimizer):
            def train_step(xt, xdot, y):
                model.train() # Set model to training mode
                yhat = model(xt)
                zero_input = t_zeros(xt.shape).double()
                zero_input[:,self.n:] = xt[:,self.n:]
                zero_input.requires_grad_(True)
                y_zero = model(zero_input)

                # Do necessary calculations for loss formulation and regularization:
                h_dot, zero_jacobian = calc_gradients(xt, xdot, yhat, zero_input, y_zero, model.training)
                optimizer.zero_grad()
                loss = loss_fn(h_dot, zero_jacobian, y, yhat, model.training)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                return loss.item()
            return train_step

        losses = []
        val_losses = []
        train_step = make_train_step(self.diffeomorphism_model, diffeomorphism_loss, optimizer)

        # Training loop
        for i in range(n_epochs):
            # Uses loader to fetch one mini-batch for training
            for x_batch, y_batch in train_loader:
                # Send mini batch data to same location as model:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Train based on current batch:
                xt = x_batch[:,:self.n+1]  # [x, t]
                xdot = x_batch[:,self.n+1:]  # [xdot]
                loss = train_step(xt, xdot, y_batch)
                losses.append(loss)

            #with no_grad():
                # Uses loader to fetch one mini-batch for validation
            for x_val, y_val in val_loader:
                # Sends data to same device as model
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                self.diffeomorphism_model.eval() # Change model model to evaluation
                xt_val = x_val[:, :self.n + 1]  # [x, t]
                xdot_val = x_val[:, self.n + 1:]  # [xdot]
                yhat = self.diffeomorphism_model(xt_val)  # Predict
                #y_z = t_zeros(yhat.shape)
                #input_z = t_zeros(xt_val.shape)
                jacobian_xdot_val, zero_jacobian_val = calc_gradients(xt_val, xdot_val, yhat, None, None, self.diffeomorphism_model.training)
                val_loss = diffeomorphism_loss(jacobian_xdot_val, zero_jacobian_val, y_val, yhat, self.diffeomorphism_model.training)  # Compute validation loss
                val_losses.append(val_loss.item())  # Save validation loss

            scheduler.step(i)
            print('Epoch: ',i,' Training loss:', format(losses[-1], '08f'), ' Validation loss:', format(val_losses[-1], '08f'))


    def build_diffeomorphism_model_keras(self, layer_width=100, l1=0., l2=0.):
        def diffeomorphism_loss(input_tensor, output_tensor, Acl_tensor):
            def loss(y_true, y_pred):
                h_pred = y_pred[:,:self.n]
                xdot = y_pred[:,self.n:]
                y_true_red = y_true[:,:self.n]


                h_grad = [K.gradients(output_tensor[:,ii],input_tensor[0])[0] for ii in range(self.n)]
                h_dot = K.stack([K.dot(h_grad[ii], K.transpose(xdot)) for ii in range(self.n)], axis=0)

                y_pred_sum = h_dot - K.transpose(K.dot(Acl_tensor, K.transpose(h_pred)))

                h_jacobian = K.stack(h_grad)
                #print(h_grad)
                #print(h_jacobian)
                #jacobian_fn = K.function([input_tensor[0]], [h_jacobian])
                #print(jacobian_fn, jacobian_fn.inputs, jacobian_fn.outputs)
                #n_t = 10
                #t = linspace(0.,2.,n_t).reshape(n_t,1)
                #with K.get_session() as sess:
               #     zero_gradient_sum = sum(sum(sum(square(h_jacobian.eval(feed_dict={'x_input:0': zeros((n_t,self.n)), 'xdot_input:0': zeros((n_t,self.n)), 't_input:0': t})))))
                    #print(jacobian_fn([zeros((1,self.n))]))
                #sess = K.get_session()
                #print(t.eval(session=sess))
                #print('Getting t: ',sess.run(y_pred[:,2], feed_dict={'x_input:0': h_pred.eval(session=sess), 'xdot_input:0': xdot.eval(session=sess), 't_input:0': t.eval(session=sess)}))
                #n_timesteps = 10
                #zero_jacobian = sess.run([h_jacobian], feed_dict={'x_input:0': zeros((n_timesteps,self.n)), 'xdot_input:0': zeros((n_timesteps,self.n)), 't_input:0': linspace(0.,2.,n_timesteps).reshape((n_timesteps,1))})
                scale = K.sum(K.square(x_input)) < 0.1
                scale = K.cast(scale,dtype='float32')
                return K.mean(K.square(y_pred_sum - y_true_red)) + 100*scale*K.sum(K.square(h_jacobian))
            return loss

        K.clear_session()
        # NN for h(x,t):
        reg = l1_l2(l1=l1, l2=l2)
        t_input = Input(shape=(1,), dtype='float32', name='t_input') # Input to h(x,t)
        x_input = Input(shape=(self.n,), dtype='float32', name='x_input')  #Input to h(x,t)
        h_input = concatenate([x_input, t_input])
        h = Dense(units=layer_width, activation='relu', kernel_regularizer=reg)(h_input)  # Hidden layer #1
        h = Dense(units=layer_width, activation='relu', kernel_regularizer=reg)(h)  #Hidden layer #2
        #h = Dense(units=layer_width, activation='relu')(h)  # Hidden layer #3
        h_output = Dense(units=self.n)(h) #Output layer

        # Pass xdot through network as well to allow desired loss function:
        xdot_input = Input(shape=(self.n,), dtype='float32', name='xdot_input')  #h_dot(x) = grad_x(h(x))*xdot

        # Concatenate into single output
        concatenated_output = concatenate([h_output, xdot_input])

        model = Model(inputs=[x_input, xdot_input, t_input], outputs=concatenated_output)
        model.compile(optimizer='adam', loss=diffeomorphism_loss(model.input, model.output, K.variable(self.A_cl)))
        self.diffeomorphism_model = model
        print(model.summary())

    def fit_diffeomorphism_model_keras(self, X, t, X_d):
        X, X_dot, X_d, t = self.process(X=X, t=t, X_d=X_d)

        y_target = X_dot - dot(self.A_cl,X.transpose()).transpose() - dot(self.BK, X_d.transpose()).transpose()
        y_fit = npconcatenate((y_target, zeros(y_target.shape)),axis=1)
        self.diffeomorphism_model.fit([X, X_dot, t], y_fit, epochs=20, batch_size=1)

    def process(self, X, t, X_d):
        # Shift dynamics to make origin a fixed point
        X_f = X_d[:,:,-1]
        X_shift = array([X[ii] - X_f[:,ii] for ii in range(len(X))])
        X_d = array([X_d[:,ii,:].reshape((X_d.shape[2],X_d.shape[0])) - X_f[:, ii] for ii in range(len(X))])

        # Calculate numerical derivatives
        X_dot = array([differentiate(X_shift[ii,:,:],t) for ii in range(len(X))])
        t = array([t for _ in range(len(X))])
        clip = int((X_shift.shape[1]-X_dot.shape[1])/2)
        X_shift = X_shift[:,clip:-clip,:]
        X_d = X_d[:, clip:-clip, :]
        t = t[:,clip:-clip]
        assert(X_shift.shape == X_dot.shape)
        assert(X_d.shape == X_dot.shape)
        assert(t.shape == X_shift[:,:,0].shape)

        # Reshape to have input-output data
        X_shift = X_shift.reshape((X_shift.shape[0]*X_shift.shape[1],X_shift.shape[2]))
        X_dot = X_dot.reshape((X_dot.shape[0] * X_dot.shape[1], X_dot.shape[2]))
        X_d = X_d.reshape((X_d.shape[0] * X_d.shape[1], X_d.shape[2]))
        t = t.reshape((t.shape[0] * t.shape[1],))
        return X_shift, X_dot, X_d, t

    def plot_eigenfunction_evolution(self, X, t):
        X = X.transpose()
        eigval_system = LinearSystemDynamics(A=diag(self.Lambda),B=zeros((self.Lambda.shape[0],1)))
        eigval_ctrl = ConstantController(eigval_system,0.)
        x0 = X[:,:1]
        t0 = array([[0.]])
        z0 = self.lift(x0, array([[0.]]))
        eigval_evo, us = eigval_system.simulate(z0.flatten(), eigval_ctrl, t)
        eigval_evo = eigval_evo.transpose()
        eigfunc_evo = self.lift(X, t.reshape((t.shape[0],1)))


        figure()
        for ii in range(1,13):
            subplot(4, 3, ii)
            plot(t, eigval_evo[ii,:], linewidth=2, label='$eigval evo$')
            plot(t, eigfunc_evo[ii,:], linewidth=2, label='$eigfunc evo$')
            title('Eigenvalue VS eigenfunction evolution')
            grid()
        legend(fontsize=12)
        show()  # TODO: Create plot of all collected trajectories (subplot with one plot for each state), not mission critical




    def lift(self, q, t):
        return array([self.basis(q[:,ii].reshape((self.n,1)), t[ii]) for ii in range(q.shape[1])]).transpose()
