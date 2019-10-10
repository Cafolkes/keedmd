import torch.nn.functional as F
from torch import nn, cuda, optim, from_numpy, manual_seed, mean, transpose as t_transpose, mm, bmm, matmul, cat


class DiffeomorphismNet(nn.Module):

    def __init__(self, n, A_cl, jacobian_penalty = 1e-2, n_hidden_layers = 2, layer_width=50, batch_size = 64, dropout_prob=0.1):
        super(DiffeomorphismNet, self).__init__()
        self.n = n
        self.n_hidden_layers = n_hidden_layers
        self.layer_width = layer_width
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.jacobian_penalty = jacobian_penalty

        N, d_h_in, H, d_h_out = batch_size, 2 * self.n, layer_width, self.n

        device = 'cuda' if cuda.is_available() else 'cpu'
        self.A_cl = A_cl.to(device)
        self.fc_in = nn.Linear(d_h_in, H).double().to(device)
        self.fc_hidden = []
        for _ in range(self.n_hidden_layers):
            self.fc_hidden.append(nn.Linear(H, H).double().to(device))
        self.fc_out = nn.Linear(H, d_h_out).double().to(device)

    def forward(self, x):
        xt = x[:, :2 * self.n]  # [x, x_d]
        xdot = x[:, 2 * self.n:3 * self.n]  # [xdot]
        x_zero = cat((x[:, 3 * self.n:], x[:, self.n:2 * self.n]), 1)
        cur_batch_size = x.shape[0]

        # Define diffeomorphism model:
        h = []
        h.append(self.fc_in(xt))
        for ii in range(self.n_hidden_layers):
            h.append(F.relu(self.fc_hidden[ii](h[-1])))
        h_out = self.fc_out(h[-1])

        #import torch
        #test_data = 0
        #test_dim = 0
        #print('Numerical jacobian: ', torch.autograd.grad(h_out[test_data,test_dim], xt, retain_graph=True, allow_unused=True)[0][0,:self.n])

        # Define diffeomorphism Jacobian model:
        h_grad = self.fc_in.weight
        h_grad = mm(self.fc_hidden[0].weight, h_grad)
        h_grad = h_grad.unsqueeze(0).expand(cur_batch_size, self.layer_width, 2 * self.n)
        delta = F.relu(self.fc_hidden[0](h[1])).sign().unsqueeze_(-1).expand(cur_batch_size, self.layer_width,2*self.n)
        h_grad = delta * h_grad
        for ii in range(1, self.n_hidden_layers):
            h_grad = bmm(self.fc_hidden[ii].weight.unsqueeze(0).expand(cur_batch_size, self.layer_width, self.layer_width), h_grad)
            delta = F.relu(self.fc_hidden[ii](h[ii+1])).sign().unsqueeze_(-1).expand(cur_batch_size,self.layer_width,2*self.n)
            h_grad = delta * h_grad

        h_grad = bmm(self.fc_out.weight.unsqueeze(0).expand(cur_batch_size,self.n,self.layer_width), h_grad)
        #print('Gradient network: ', h_grad[test_dim,test_dim,:self.n])
        #print('In network: ', h_grad.shape, xdot.unsqueeze(-1).shape)
        h_dot = bmm(h_grad[:,:,:self.n], xdot.unsqueeze(-1))
        h_dot = h_dot.squeeze()

        # Calculate zero Jacobian:
        h_zero = []
        h_zero.append(self.fc_in(x_zero))
        for ii in range(self.n_hidden_layers):
            h_zero.append(F.relu(self.fc_hidden[ii](h_zero[-1])))
        h_zero_out = self.fc_out(h_zero[-1])

        h_zerograd = self.fc_in.weight
        h_zerograd = mm(self.fc_hidden[0].weight, h_zerograd)
        h_zerograd = h_zerograd.unsqueeze(0).expand(cur_batch_size, self.layer_width, 2 * self.n)
        delta = F.relu(self.fc_hidden[0](h_zero[1])).sign().unsqueeze_(-1).expand(cur_batch_size, self.layer_width,2*self.n)
        h_zerograd = delta * h_zerograd
        for ii in range(1, self.n_hidden_layers):
            h_zerograd = bmm(self.fc_hidden[ii].weight.unsqueeze(0).expand(cur_batch_size, self.layer_width, self.layer_width), h_zerograd)
            delta = F.relu(self.fc_hidden[ii](h_zero[ii+1])).sign().unsqueeze_(-1).expand(cur_batch_size, self.layer_width, 2*self.n)
            h_zerograd = delta * h_zerograd
        h_zerograd = bmm(self.fc_out.weight.unsqueeze(0).expand(cur_batch_size, self.n, self.layer_width), h_zerograd)
        h_zerograd = h_zerograd[:, :, :self.n]

        y_pred = cat([h_out, h_dot, h_zerograd.norm(p=2,dim=2)], 1)  # [h, h_dot, norm(zerograd)]

        return y_pred

    def diffeomorphism_loss(self, y_true, y_pred):
        h = y_pred[:,:self.n]
        h_dot = y_pred[:,self.n:2*self.n]
        zerograd = y_pred[:,2*self.n:]
        cur_batch_size = y_pred.shape[0]
        A_cl_batch = self.A_cl.unsqueeze(0).expand(cur_batch_size, self.n, self.n)

        return mean((y_true - (h_dot - bmm(A_cl_batch, h.unsqueeze(-1)).squeeze()))**2) + self.jacobian_penalty*mean(zerograd**2)

    def predict(self, x):

        h = self.fc_in(x)
        for ii in range(self.n_hidden_layers):
            h = F.relu(self.fc_hidden[ii](h))
        h = self.fc_out(h)

        return h.detach().numpy()