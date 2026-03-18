import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def G1(x):
    """
    Evaluate the metric G(x) = (1 + |x|^2) I for torch.tensor x of shape Nx2.
    The result has shape Nx2x2
    """
    N, D = x.shape
    I = torch.eye(D, D).reshape(1, D, D).repeat(N, 1, 1)  # NxDxD
    alpha = 1 + torch.sum(x**2, dim=1)  # N
    G = alpha.reshape(N, 1, 1) * I  # NxDxD
    return G

def G2(x, data, sigma=0.1):
    """
    Evaluate the metric 1/p(x) * I, where p(x) is a Gaussian kernel density estimate.
    The result has shape Nx2x2
    """
    N, D = x.shape
    M, D = data.shape
    sigma2 = sigma**2
    normalization = (2 * 3.14159)**(D/2) * sigma**D  # scalar
    I = torch.eye(D, D).reshape(1, D, D)  # 1xDxD
    Gs = []
    for n in range(N):
        xn = x[n].reshape(1, D)  # 1xD
        delta = (xn - data)  # MxD
        K = torch.exp(-0.5 * torch.sum(delta**2, dim=1) / sigma2) / normalization  # M
        pn = K.sum() / M  # scalar
        Gs.append(I / (pn + 1e-4))  # list containing 1xDxD matrices
    G = torch.concatenate(Gs, dim=0)  # NxDxD
    return G

def plot_metric(metric, grid_range):
    X, Y = torch.meshgrid(grid_range, grid_range, indexing='ij')
    XY = torch.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=1)
    G = metric(XY)  # NxDxD
    trG = G[:, 0, 0] + G[:, 1, 1]  # N
    plt.imshow(
        trG.reshape(X.shape).detach().numpy().T,
        extent=(grid_range[0], grid_range[-1], grid_range[0], grid_range[-1]),
        origin="lower"
    )
    
class PLcurve:
    def __init__(self, x0, x1, N):
        """
        Represent the piecewise linear curve connecting x0 to x1 using a
        total of N nodes (including end-points)
        """
        super().__init__()
        self.x0 = x0.reshape(1, -1)  # 1xD
        self.x1 = x1.reshape(1, -1)  # 1xD
        self.N = N

        t = torch.linspace(0, 1, N).reshape(N, 1)  # Nx1
        c = (1 - t) @ self.x0 + t @ self.x1  # NxD
        
        # We optimize the intermediate points
        self.params = c[1:-1].clone().detach()
        self.params.requires_grad = True

    def points(self):
        c = torch.cat((self.x0, self.params, self.x1), dim=0)  # NxD
        return c

    def plot(self):
        c = self.points().detach().numpy()
        plt.plot(c[:, 0], c[:, 1])

def curve_energy(metric, curve):
    G = metric(curve[:-1])  # (N-1)xDxD
    delta = curve[1:] - curve[:-1]  # (N-1)xD
    # Batch matrix multiplication: (N-1, D, D) x (N-1, D, 1) -> (N-1, D)
    tmp = torch.bmm(G, delta.unsqueeze(-1)).squeeze(-1)
    energy = torch.sum(delta * tmp)
    return energy

def connecting_geodesic(metric, curve):
    opt = optim.LBFGS([curve.params], lr=0.5)
    
    def closure():
        opt.zero_grad()
        energy = curve_energy(metric, curve.points())
        energy.backward()
        return energy

    max_iter = 1000
    for i in range(max_iter):
        opt.step(closure)

# metric = 'quadratic'
metric = 'density'

if metric == 'quadratic':
    r = 5
    plot_metric(G1, torch.linspace(-r, r, 100))
    N = 20
    for _ in range(5):
        # Generate two random points in the range [-r, r]
        x0 = 2 * r * (torch.rand(2) - 0.5)
        x1 = 2 * r * (torch.rand(2) - 0.5)
        
        c = PLcurve(x0, x1, N)
        c.plot()
        
        print('Energy before optimization is {}'.format(curve_energy(G1, c.points()).item()))
        connecting_geodesic(G1, c)
        print('Energy after optimization is {}'.format(curve_energy(G1, c.points()).item()))
        c.plot()

    plt.axis((-r, r, -r, r))
    plt.show()

elif metric == 'density':
    # Ensure 'toybanana.npy' is in your local directory
    data = torch.from_numpy(np.load('toybanana.npy')).to(torch.float32)  # 992x2
    r = 3
    G = lambda x: G2(x, data)
    
    plot_metric(G, torch.linspace(-r, r, 100))
    plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
    
    T = 20
    for _ in range(10):
        # Pick two random points from the existing dataset
        idx = torch.randint(data.shape[0], (2,))
        c = PLcurve(data[idx[0]], data[idx[1]], T)
        
        print('Energy before optimization is {}'.format(curve_energy(G, c.points()).item()))
        connecting_geodesic(G, c)
        print('Energy after optimization is {}'.format(curve_energy(G, c.points()).item()))
        c.plot()

    plt.axis((-r, r, -r, r))
    plt.show()