import torch
import torch.nn as nn
import torch.autograd as autograd
from torchdiffeq import odeint_adjoint


class ResBlock(nn.Module):
    """
    Simple residual block used to construct ResNet.
    """
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_channels, n_inner_channels, kernel_size, padding='same', bias=False)
        self.conv2 = nn.Conv1d(n_inner_channels, n_channels, kernel_size, padding='same', bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.norm1(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.norm2(y)
        y += x
        return y


class ConcatConv1d(nn.Module):
    """
    1d convolution concatenated with time for usage in ODENet.
    """
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=0, bias=True, transpose=False):
        super(ConcatConv1d, self).__init__()
        module = nn.ConvTranspose1d if transpose else nn.Conv1d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=kernel_size, stride=stride, padding=padding,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):
    """
    Network architecture for ODENet.
    """
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super(ODEfunc, self).__init__()
        self.conv1 = ConcatConv1d(n_channels, n_inner_channels, kernel_size, padding='same', bias=False)
        self.conv2 = ConcatConv1d(n_inner_channels, n_channels, kernel_size, padding='same', bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.nfe = 0 

    def forward(self, t, x):
        self.nfe += 1
        y = self.conv1(t, x)
        y = self.relu(y)
        y = self.norm1(y)
        y = self.conv2(t, y)
        y = self.relu(y)
        y = self.norm2(y)
        return y


class ODENet(nn.Module):
    """
    Neural ODE.

    Uses ODE solver (dopri5 by default) to yield model output.
    Backpropagation is done with the adjoint method as described in
    https://arxiv.org/abs/1806.07366.

    Parameters
    ----------
    odefunc : nn.Module
        network architecture
    rtol : float
        relative tolerance of ODE solver
    atol : float
        absolute tolerance of ODE solver
    """
    def __init__(self, odefunc, rtol=1e-3, atol=1e-3):
        super(ODENet, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.rtol = rtol
        self.atol = atol

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint_adjoint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol)
        return out[1]

    # Update number of function evaluations (nfe)
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class DEQfunc(nn.Module):
    """
    Network architecture for DEQ.
    """
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super(DEQfunc, self).__init__()
        self.conv1 = nn.Conv1d(n_channels, n_inner_channels, kernel_size, padding='same', bias=False)
        self.conv2 = nn.Conv1d(n_inner_channels, n_channels, kernel_size, padding='same', bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        self.relu = nn.ReLU(inplace=True)
        self.nfe = 0
        
    def forward(self, z, x):
        self.nfe += 1
        y = self.conv1(z)
        y = self.relu(y)
        y = self.norm1(y)
        y = self.conv2(y)
        y += x
        y = self.norm2(y)
        y += z
        y = self.relu(y)
        y = self.norm3(y)
        return y


class DEQNet(nn.Module):
    """
    Deep Equilibrium Model.
    
    Backpropagation through the equilibrium point is done with implicit differentiation as described in
    https://arxiv.org/abs/1909.01377.

    Parameters
    ----------
    deqfunc : nn.Module
        network architecture
    solver : function
        fixed point solver
    """
    def __init__(self, deqfunc, solver, **kwargs):
        super(DEQNet, self).__init__()
        self.deqfunc = deqfunc
        self.solver = solver
        self.kwargs = kwargs
        
    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z : self.deqfunc(z, x), torch.zeros_like(x), **self.kwargs)
        z = self.deqfunc(z,x)
        
        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.deqfunc(z0,x)
        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad, grad, **self.kwargs)
            return g
                
        z.register_hook(backward_hook)
        return z

    # Update number of function evaluations (nfe)
    @property
    def nfe(self):
        return self.deqfunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.deqfunc.nfe = value