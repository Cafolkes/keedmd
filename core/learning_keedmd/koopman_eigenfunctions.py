from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title
from numpy import array, linalg, transpose, diag, dot, ones, zeros, unique, power, prod, exp, log, divide, real, iscomplex, any
from numpy import concatenate as npconcatenate
from itertools import combinations_with_replacement, permutations
from .utils import differentiate_vec
from .basis_functions import BasisFunctions
from ..dynamics.linear_system_dynamics import LinearSystemDynamics
from ..controllers.constant_controller import ConstantController
from torch import nn, cuda, optim, from_numpy, manual_seed, mean, transpose as t_transpose, mm, matmul, zeros as t_zeros, save, load
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
from torch.autograd.gradcheck import zero_gradients

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

        if any(iscomplex(lambd)) or any(iscomplex(w)):
            Warning("Complex eigenvalues and/or eigenvalues. Complex part supressed.")
            lambd = real(lambd)
            w = real(w)

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

    def diffeomorphism(self, q, q_d):
        q = q.transpose()
        q_d = q_d.transpose()
        self.diffeomorphism_model.eval()
        input = npconcatenate((q,q_d),axis=1)
        diff_pred = self.diffeomorphism_model(from_numpy(input)).detach().numpy()
        return (q + diff_pred).transpose()

    def build_diffeomorphism_model(self, n_hidden_layers = 2, layer_width=50, batch_size = 64, dropout_prob=0.1):
        

        # Set up model architecture for h(x,t):
        N, d_h_in, H, d_h_out = batch_size, 2*self.n, layer_width, self.n
        self.diffeomorphism_model= nn.Sequential(
            nn.Linear(d_h_in,H),
            nn.ReLU()
        )
        for ii in range(n_hidden_layers):
            self.diffeomorphism_model.add_module('dense_' + str(ii+1), nn.Linear(H,H))
            self.diffeomorphism_model.add_module('relu_' + str(ii + 1), nn.ReLU())
            if ii < n_hidden_layers-1:
                self.diffeomorphism_model.add_module('dropout_' + str(ii+1), nn.Dropout(p=dropout_prob))
        self.diffeomorphism_model.add_module('output', nn.Linear(H,d_h_out))

        self.diffeomorphism_model = self.diffeomorphism_model.double()
        self.A_cl = from_numpy(self.A_cl)

    def fit_diffeomorphism_model(self, X, t, X_d, learning_rate=1e-2, learning_decay=0.95, n_epochs=50, train_frac=0.8, l2=1e1, jacobian_penalty=1., batch_size=64, initialize=True, verbose=True):
        X, X_dot, X_d, t = self.process(X=X, t=t, X_d=X_d)
        y_target = X_dot - dot(self.A_cl, X.transpose()).transpose()# - dot(self.BK, X_d.transpose()).transpose()

        device = 'cpu' if cuda.is_available() else 'cpu'

        # Prepare data for pytorch:
        manual_seed(42)  # Fix seed for reproducibility
        X_tensor = from_numpy(npconcatenate((X, X_d, X_dot),axis=1)) #[x (1,4), t (1,1), x_dot (1,4)]
        y_tensor = from_numpy(y_target)
        X_tensor.requires_grad_(True)


        # Builds dataset with all data
        dataset = TensorDataset(X_tensor, y_tensor)
        # Splits randomly into train and validation datasets
        n_train = int(train_frac*X.shape[0])
        n_val = X.shape[0]-n_train
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
        # Builds a loader for each dataset to perform mini-batch gradient descent
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

        def diffeomorphism_loss(h_dot, zero_jacobian, y_true, y_pred, is_training):
            h_sum_pred = h_dot - t_transpose(mm(self.A_cl, t_transpose(y_pred, 1, 0)), 1, 0)
            if is_training:
                loss = mean((y_true-h_sum_pred)**2) + jacobian_penalty*mean((zero_jacobian**2))
            else:
                loss = mean((y_true-h_sum_pred)**2)
            return loss

        # Set up optimizer and learning rate scheduler:
        optimizer = optim.Adam(self.diffeomorphism_model.parameters(),lr=learning_rate,weight_decay=l2)
        lambda1 = lambda epoch: learning_decay ** epoch
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        def calc_gradients(xt, xdot, yhat, zero_input, yzero, is_training):
            xt.retain_grad()
            if is_training:
                zero_input.retain_grad()
                zero_jacobian = compute_jacobian(zero_input, yzero)[:,:,:self.n]
                zero_jacobian.requires_grad_(True)
                zero_jacobian = zero_jacobian.squeeze()
                optimizer.zero_grad()
            else:
                zero_jacobian = None

            jacobian = compute_jacobian(xt, yhat)[:,:,:self.n]
            optimizer.zero_grad()
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

            jacobian = t_zeros((int(num_classes), *inputs.size())).double()
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
                #optimizer.zero_grad() #TODO: Remove(?)
                loss = loss_fn(h_dot, zero_jacobian, y, yhat, model.training)
                loss.backward()
                optimizer.step()
                #optimizer.zero_grad() #TODO: Remove(?)
                return loss.item()
            return train_step

        batch_loss = []
        losses = []
        batch_val_loss = []
        val_losses = []
        train_step = make_train_step(self.diffeomorphism_model, diffeomorphism_loss, optimizer)

        # Initialize model weights:
        def init_normal(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

        if initialize:
            self.diffeomorphism_model.apply(init_normal)

        # Training loop
        for i in range(n_epochs):
            # Uses loader to fetch one mini-batch for training
            for x_batch, y_batch in train_loader:
                # Send mini batch data to same location as model:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Train based on current batch:
                xt = x_batch[:,:2*self.n]  # [x, x_d]
                xdot = x_batch[:,2*self.n:]  # [xdot]
                batch_loss.append(train_step(xt, xdot, y_batch))
            losses.append(sum(batch_loss)/len(batch_loss))
            batch_loss = []

            #with no_grad():
                # Uses loader to fetch one mini-batch for validation
            for x_val, y_val in val_loader:
                # Sends data to same device as model
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                self.diffeomorphism_model.eval() # Change model model to evaluation
                xt_val = x_val[:, :2*self.n]  # [x, t]
                xdot_val = x_val[:, 2*self.n:]  # [xdot]
                yhat = self.diffeomorphism_model(xt_val)  # Predict
                #y_z = t_zeros(yhat.shape)
                #input_z = t_zeros(xt_val.shape)
                jacobian_xdot_val, zero_jacobian_val = calc_gradients(xt_val, xdot_val, yhat, None, None, self.diffeomorphism_model.training)
                batch_val_loss.append(diffeomorphism_loss(jacobian_xdot_val, zero_jacobian_val, y_val, yhat, self.diffeomorphism_model.training).item()) # Compute validation loss
                optimizer.zero_grad()
            val_losses.append(sum(batch_val_loss)/len(batch_val_loss))  # Save validation loss
            batch_val_loss = []

            scheduler.step(i)
            if verbose:
                print('Epoch: ',i,' Training loss:', format(losses[-1], '08f'), ' Validation loss:', format(val_losses[-1], '08f'))

    def process(self, X, t, X_d):
        # Shift dynamics to make origin a fixed point
        X_f = X_d[:,-1,:]
        X_shift = array([X[ii,:,:] - X_f[ii,:] for ii in range(len(X))])
        X_d = array([X_d[ii,:,:].reshape((X_d.shape[1],X_d.shape[2])) - X_f[ii,:] for ii in range(len(X))])

        # Calculate numerical derivatives
        X_dot = array([differentiate_vec(X_shift[ii,:,:],t) for ii in range(X_shift.shape[0])])
        t = array([t for _ in range(len(X))])
        #clip = int((X_shift.shape[1]-X_dot.shape[1])/2)
        #X_shift = X_shift[:,clip:-clip,:]
        #X_d = X_d[:, clip:-clip, :]
        #t = t[:,clip:-clip]
        assert(X_shift.shape == X_dot.shape)
        assert(X_d.shape == X_dot.shape)
        assert(t.shape == X_shift[:,:,0].shape)

        # Reshape to have input-output data
        X_shift = X_shift.reshape((X_shift.shape[0]*X_shift.shape[1],X_shift.shape[2]))
        X_dot = X_dot.reshape((X_dot.shape[0] * X_dot.shape[1], X_dot.shape[2]))
        X_d = X_d.reshape((X_d.shape[0] * X_d.shape[1], X_d.shape[2]))
        t = t.reshape((t.shape[0] * t.shape[1],))

        return X_shift, X_dot, X_d, t

    def save_diffeomorphism_model(self, filename):
        save(self.diffeomorphism_model.state_dict(), filename)

    def load_diffeomorphism_model(self, filename):
        self.diffeomorphism_model.load_state_dict(load(filename))

    def plot_eigenfunction_evolution(self, X, X_d, t):
        X = X.transpose()
        X_d = X_d.transpose()
        eigval_system = LinearSystemDynamics(A=diag(self.Lambda),B=zeros((self.Lambda.shape[0],1)))
        eigval_ctrl = ConstantController(eigval_system,0.)
        x0 = X[:,:1]
        x0_d = X_d[:,:1]
        z0 = self.lift(x0, x0_d)
        eigval_evo, us = eigval_system.simulate(z0.flatten(), eigval_ctrl, t)
        eigval_evo = eigval_evo.transpose()
        eigfunc_evo = self.lift(X, X_d).transpose()


        figure()
        for ii in range(1,17):
            subplot(4, 4, ii)
            plot(t, eigval_evo[ii,:], linewidth=2, label='$eigval evo$')
            plot(t, eigfunc_evo[ii,:], linewidth=2, label='$eigfunc evo$')
            title('Eigenvalue VS eigenfunction evolution')
            grid()
        legend(fontsize=12)
        show()  # TODO: Create plot of all collected trajectories (subplot with one plot for each state), not mission critical

    def lift(self, q, q_d):
        return array([self.basis(q[:,ii].reshape((self.n,1)), q_d[:,ii].reshape((self.n,1))) for ii in range(q.shape[1])])
