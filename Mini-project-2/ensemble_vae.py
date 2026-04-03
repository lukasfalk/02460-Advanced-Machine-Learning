# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)

class EnsembleVAE(VAE):
    def __init__(self, prior, decoders, encoder):
        super().__init__(prior, decoders[0], encoder)
        self.decoders = nn.ModuleList(decoders)

    def elbo(self, x):
        q = self.encoder(x)
        z = q.rsample()
        decoder = self.decoders[torch.randint(len(self.decoders), (1,)).item()] # Randomly sample a single encoder from the ensemble
        elbo = torch.mean(
            decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

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

        t = torch.linspace(0, 1, N).reshape(N, 1).to(x0.device)  # Nx1
        c = (1 - t) @ self.x0 + t @ self.x1  # NxD
        
        # We optimize the intermediate points
        self.params = c[1:-1].clone().detach()
        self.params.requires_grad = True

    def points(self):
        c = torch.cat((self.x0, self.params, self.x1), dim=0)  # NxD
        return c

    def plot(self):
        c = self.points().detach().cpu().numpy()
        plt.plot(c[:, 0], c[:, 1])

def curve_energy(metric, curve):
    '''
    Default function from week 7
    '''
    G = metric(curve[:-1])  # (N-1)xDxD
    delta = curve[1:] - curve[:-1]  # (N-1)xD
    # Batch matrix multiplication: (N-1, D, D) x (N-1, D, 1) -> (N-1, D)
    tmp = torch.bmm(G, delta.unsqueeze(-1)).squeeze(-1)
    energy = torch.sum(delta * tmp)
    return energy

def connecting_geodesic(metric, curve):
    '''
    Default function from week 7
    '''
    opt = optim.LBFGS([curve.params], lr=0.5)
    
    def closure():
        opt.zero_grad()
        energy = curve_energy(metric, curve.points())
        energy.backward()
        return energy

    max_iter = 100
    for i in range(max_iter):
        opt.step(closure)

def pullback_metric(z, decoder):
    """
    Compute the pull-back metric G(z) = J_f(z).T @ J_f(z) for latent points.
    
    z: Tensor (N, 2). Inputs and the specified 2 dimensions.
    G: Tensor (N, 2, 2)
    """
    def f(z_single):
        return decoder(z_single.unsqueeze(0)).flatten()  # (784,)

    J = torch.vmap(torch.func.jacrev(f))(z)  # (N, 784, 2)
    G = torch.bmm(J.transpose(1, 2), J)      # (N, 2, 2)
    return G

def ensemble_curve_energy(decoders, curve, n_samples=10):
    '''
    Model-average curve energy. Eq. 1 from project description
    '''
    energy = 0
    M = len(decoders)
    for _ in range(n_samples):
        l = torch.randint(M, (1,)).item()
        k = torch.randint(M, (1,)).item()
        f_l = decoders[l].decoder_net(curve[:-1])  # (N-1, 1, 28, 28)
        f_k = decoders[k].decoder_net(curve[1:])   # (N-1, 1, 28, 28)
        diff = (f_l - f_k).flatten(1)              # (N-1, 784)
        energy = energy + torch.sum(diff**2)
    return energy / n_samples

def connecting_ensemble_geodesic(decoders, curve, n_samples=5):
    opt = optim.LBFGS([curve.params], lr=0.001, max_iter=20,
                      tolerance_grad=1e-3, tolerance_change=1e-4,
                      line_search_fn='strong_wolfe')
    
    def closure():
        opt.zero_grad()
        energy = ensemble_curve_energy(decoders, curve.points(), n_samples)
        energy.backward()
        torch.nn.utils.clip_grad_norm_([curve.params], max_norm=1.0)
        return energy

    for _ in range(20):
        opt.step(closure)

def CoV(dist_key, all_distances):
    '''
    CoV from Eq, 2 in project description
    '''
    covs = []
    for decoder_number in range(1, 4):
        d = all_distances[decoder_number][dist_key]  # (reruns, curves)
        cov_per_pair = []
        for k in range(d.shape[1]):
            col = d[:, k]
            col = col[~torch.isnan(col)]  # drop NaN reruns for this pair
            if len(col) > 1:
                cov_per_pair.append((col.std() / col.mean()).item())
        covs.append(sum(cov_per_pair) / len(cov_per_pair))
    return covs

def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics", "train_ensemble", "geodesics_ensemble"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim
    seed = 42

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(dim=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(dim=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(dim=1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(dim=1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(dim=1),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":
        print("Plotting geodesics in the latent space")
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        print("Encoding test data")
        all_z      = []
        all_labels = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                z = model.encoder(x).mean
                all_z.append(z)
                all_labels.append(y)
        all_z = torch.cat(all_z, dim=0) # (N, 2)
        all_labels = torch.cat(all_labels, dim=0)

        # Define the metric as a closure over the trained decoder
        print("Defining pull-back metric")
        decoder_net = model.decoder.decoder_net
        metric = lambda z: pullback_metric(z, decoder_net)

        # Plot latent space
        print("Plotting latent space and geodesics")
        plt.figure(figsize=(8, 8))
        for class_label in range(num_classes):
            mask = all_labels == class_label
            plt.scatter(all_z[mask, 0].cpu(), all_z[mask, 1].cpu(), s=5, label=str(class_label))

        # Compute and plot geodesics between random pairs
        print("Computing geodesics")
        torch.manual_seed(seed)
        indices = torch.randperm(len(all_z))[:args.num_curves * 2]
        for k in tqdm(range(args.num_curves), desc="Computing geodesics"):
            z0 = all_z[indices[2*k]].detach()
            z1 = all_z[indices[2*k + 1]].detach()
            curve = PLcurve(z0, z1, args.num_t)
            connecting_geodesic(metric, curve)
            curve.plot()

        plt.legend()
        plt.title("Latent space, pull-back geodesics")
        plt.savefig("geodesics.png", dpi=150)
        plt.show()

    elif args.mode == "train_ensemble":
        all_distances = {}  # keyed by num_decoders
        
        # Same test point pairs across all reruns
        test_batch = next(iter(mnist_test_loader))[0].to(device)
        torch.manual_seed(0)
        pair_indices = torch.randperm(len(test_batch))[:args.num_curves * 2]
        
        for decoder_number in range(1, 4):  # [1, 2, 3]
            geodesic_dists  = []
            euclidean_dists = []
            
            for rerun in range(args.num_reruns):
                model = EnsembleVAE(
                    GaussianPrior(M),
                    [GaussianDecoder(new_decoder()) for _ in range(decoder_number)],
                    GaussianEncoder(new_encoder()),
                ).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                train(model, 
                      optimizer, 
                      mnist_train_loader, 
                      args.epochs_per_decoder, 
                      device
                    )
                model.eval()
                
                # Get latent means for the fixed test pairs
                with torch.no_grad():
                    z = model.encoder(test_batch).mean  # (N, 2)
                
                run_geo_dists = []
                run_euc_dists = []
                for k in range(args.num_curves):
                    z0 = z[pair_indices[2*k]].detach()
                    z1 = z[pair_indices[2*k + 1]].detach()
                    
                    # Euclidean distance
                    euc = torch.norm(z0 - z1).item()
                    run_euc_dists.append(euc)
                    
                    # Geodesic distance
                    curve = PLcurve(z0, z1, args.num_t)
                    energy = lambda c: ensemble_curve_energy(
                        model.decoders, 
                        c, 
                        n_samples=5
                    )
                    connecting_ensemble_geodesic(model.decoders, curve)
                    geodesic_dist = torch.sqrt(
                        ensemble_curve_energy(
                            model.decoders, 
                            curve.points().detach(), 
                            n_samples=20
                        )
                    ).item()
                    run_geo_dists.append(geodesic_dist)
                
                geodesic_dists.append(run_geo_dists)
                euclidean_dists.append(run_euc_dists)
                print(f"num_dec={decoder_number}, rerun={rerun} done")
            
            all_distances[decoder_number] = {
                'geodesic': torch.tensor(geodesic_dists),   # (reruns, curves)
                'euclidean': torch.tensor(euclidean_dists),
            }
        
        # Plot CoV
        plt.figure()
        for label, dist_key in [('Geodesic', 'geodesic'), 
                                ('Euclidean', 'euclidean')]:
            covs = CoV(dist_key, all_distances)
            plt.plot([1, 2, 3], covs, marker='o', label=label)
        plt.xlabel('Number of ensemble decoders')
        plt.ylabel('Mean CoV')
        plt.legend()
        plt.savefig('cov.png', dpi=150)

    elif args.mode == "geodesics_ensemble":
        os.makedirs(args.experiment_folder, exist_ok=True)
        model = EnsembleVAE(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)],
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, 
              optimizer, 
              mnist_train_loader, 
              args.epochs_per_decoder, 
              device
            )
        model.eval()

        # Save models
        torch.save(model.encoder.state_dict(),
                f"{args.experiment_folder}/ensemble_encoder.pt")
        for i, dec in enumerate(model.decoders):
            torch.save(dec.state_dict(),
                    f"{args.experiment_folder}/ensemble_decoder_{i}.pt")

        # Encode test data
        all_z       = []
        all_labels  = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                z = model.encoder(x.to(device)).mean
                all_z.append(z)
                all_labels.append(y)
        all_z = torch.cat(all_z, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Plot latent space
        plt.figure(figsize=(8, 8))
        for class_label in range(num_classes):
            mask = all_labels == class_label
            plt.scatter(all_z[mask, 0].cpu(), 
                        all_z[mask, 1].cpu(), 
                        s=5, 
                        label=str(class_label))

        # Compute and plot geodesics
        torch.manual_seed(seed)
        indices = torch.randperm(len(all_z))[:args.num_curves * 2]
        for k in tqdm(range(args.num_curves), desc="Computing geodesics"):
            z0 = all_z[indices[2*k]].detach()
            z1 = all_z[indices[2*k + 1]].detach()
            curve = PLcurve(z0, z1, args.num_t)
            connecting_ensemble_geodesic(model.decoders, curve)
            curve.plot()

        plt.legend()
        plt.title("Latent space, ensemble geodesics")
        plt.savefig("geodesics_ensemble.png", dpi=150)
        plt.show()