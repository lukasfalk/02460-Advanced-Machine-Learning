# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from flow_ex_2_5 import GaussianBase, MaskedCouplingLayer, Flow
import numpy as np

class GaussianPrior(nn.Module):
    def __init__(self, M: int):
        """
        Gaussian prior distribution with zero mean and unit variance.
        M: Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.prior_name = 'gaussian'
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Exercise 1.6
        Return the prior distribution.
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)
    
    def sample(self, sample_shape):
        return self.forward().sample(sample_shape)

class MoGPrior(nn.Module):
    def __init__(self, M: int, K: int):
        """
        Define a Mixture of Gaussians prior distribution.
        M: Dimension of the latent space.
        K: Number of mixture components.
        """
        super(MoGPrior, self).__init__()
        self.prior_name = 'MoG'
        self.M = M
        self.K = K
        self.means = nn.Parameter(torch.zeros(self.K, self.M), requires_grad=True)
        self.stds = nn.Parameter(torch.ones(self.K, self.M), requires_grad=True)
        self.logits = nn.Parameter(torch.zeros(self.K), requires_grad=True)
        self.log_stds = nn.Parameter(torch.zeros(self.K, self.M))

    def forward(self):
        """
        prior: [torch.distributions.Distribution]
        Return: prior distribution.
        """
        component_dist = td.Independent(td.Normal(self.means, torch.exp(self.log_stds)), 1)
        mixture_dist = td.Categorical(logits=self.logits)
        return td.MixtureSameFamily(mixture_dist, component_dist)
    
    def sample(self, sample_shape):
        return self.forward().sample(sample_shape)
    
class FlowPrior(nn.Module):
    def __init__(self, M, num_transformations=4, num_hidden=128):
        super(FlowPrior, self).__init__()
        self.prior_name = 'flow'
        base = GaussianBase(M)
        transformations = []
        for i in range(num_transformations):
            mask = torch.zeros(M)
            mask[M//2:] = 1
            if i % 2 == 0:
                mask = 1 - mask
            scale_net = nn.Sequential(nn.Linear(M, num_hidden), nn.ReLU(), nn.Linear(num_hidden, M), nn.Tanh())
            translation_net = nn.Sequential(nn.Linear(M, num_hidden), nn.ReLU(), nn.Linear(num_hidden, M))
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
        self.flow = Flow(base, transformations)

    def log_prob(self, z):
        return self.flow.log_prob(z)

    def sample(self, sample_shape):
        return self.flow.sample(sample_shape)

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Gaussian encoder distribution based on encoder network.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        x: Tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        Return: Gaussian distribution over the latent space.
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        z: Tensor of M-dimensional latent space `(batch_size, M)`.
        Return a Bernoulli distribution over the data space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    def __init__(self, prior, decoder, encoder):
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
    
    @property
    def prior_name(self):
        return self.prior.prior_name

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        match self.prior.prior_name:
            case 'gaussian':
                q = self.encoder(x)
                z = q.rsample()
                elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
                return elbo
            case 'MoG':
                return self.elbo_GoM(x)
            case 'flow':
                return self.elbo_flow(x)
            case _:
                raise ValueError(f"Unknown prior type: {self.prior.prior_name}")
            
    def elbo_GoM(self, x: torch.Tensor) -> torch.Tensor:
        """
        Exercise 1.6
        Compute the ELBO for the given batch of data using a Mixture of Gaussians prior.
        """
        q = self.encoder(x) # approximate posterior distribution q(z|x)
        z = q.rsample() # reparameterization trick
        log_px_z = self.decoder(z).log_prob(x) # reconstruction term log p(x|z)
        log_pz = self.prior().log_prob(z) # MoG prior log p(z)
        log_qz_x = q.log_prob(z) # entropy term log q(z|x)
        elbo = torch.mean(log_px_z + log_pz - log_qz_x, dim=0)
        return elbo
    
    def elbo_flow(self, x: torch.Tensor) -> torch.Tensor:
        q = self.encoder(x)
        z = q.rsample()
        log_px_z = self.decoder(z).log_prob(x)
        log_pz = self.prior.log_prob(z)
        log_qz_x = q.log_prob(z)
        return torch.mean(log_px_z + log_pz - log_qz_x, dim=0)

    def sample(self, n_samples=1):
        """
        Sample from the model.
        Number of samples to generate.
        """
        z = self.prior.sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.
        x: Tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

def evaluate_elbo(model, data_loader, device):
    """
    Exercise 1.5
    Evaluate the ELBO on the binarised MNIST test set.
    """
    model.eval()
    total_elbo = 0
    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            elbo = model.elbo(x)
            total_elbo += elbo.item()
    
    avg_elbo = total_elbo / len(data_loader)
    print(f"ELBO evaluation - {model.prior_name}: {avg_elbo:.4f}")

def plot_samples(model, data_loader, device, M):
    """
    Exercise 1.5
    """
    model.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            q = model.encoder(x) # approximate posterior p(z|x)
            z = q.rsample() # sample from the approximate posterior q_ø(z|x)
            latents.append(z.cpu())
            labels.append(y)

    latents = torch.cat(latents, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Apply PCA if dimensionality M > 2
    if M > 2:
        pca = PCA(n_components=2)
        latents = pca.fit_transform(latents)
        print(f"Projected {M}D latent space to 2D using PCA.")

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, alpha=0.5, cmap='tab10', s=2)
    plt.colorbar(scatter, label='Digit Class')
    plt.title(f'Aggregate Posterior (M={M})' + (' - PCA Projection' if M > 2 else ''))
    plt.xlabel('z1' if M == 2 else 'PC1')
    plt.ylabel('z2' if M == 2 else 'PC2')
    plt.savefig(f'figures/aggregate_posterior_{model.prior_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_prior_and_posterior(model, data_loader, device, M, n_samples=10000):
    model.eval()
    latents, labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            z = model.encoder(x).rsample()
            latents.append(z.cpu())
            labels.append(y)
        prior_samples = model.prior.sample(torch.Size([n_samples])).cpu().numpy()

    latents = torch.cat(latents, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    if M > 2:
        pca = PCA(n_components=2)
        pca.fit(latents)
        latents = pca.transform(latents)
        prior_samples = pca.transform(prior_samples)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Prior plot
    ax1.scatter(prior_samples[:, 0], prior_samples[:, 1], s=2, alpha=0.3, c='steelblue')
    ax1.set_title(f'Prior ({model.prior_name})')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')

    # Aggregate posterior plot
    scatter = ax2.scatter(latents[:, 0], latents[:, 1], c=labels, s=2, alpha=0.5, cmap='tab10')
    plt.colorbar(scatter, ax=ax2, label='Digit Class')
    ax2.set_title(f'Aggregate Posterior ({model.prior_name})')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')

    # Overlay component means on both plots for MoG
    if model.prior_name == 'MoG':
        K = model.prior.K
        component_samples = []
        component_labels = []
        with torch.no_grad():
            for k in range(K):
                # Sample from component k directly
                mean = model.prior.means[k]
                std = torch.exp(model.prior.log_stds[k])
                samples_k = td.Independent(td.Normal(mean, std), 1).sample((n_samples // K,))
                component_samples.append(samples_k.cpu().numpy())
                component_labels.extend([k] * (n_samples // K))
        
        component_samples = np.concatenate(component_samples)
        component_samples_2d = pca.transform(component_samples)
        
        scatter2 = ax1.scatter(component_samples_2d[:, 0], component_samples_2d[:, 1],
                            c=component_labels, cmap='tab10', s=2, alpha=0.4)
        plt.colorbar(scatter2, ax=ax1, label='Component')

    plt.suptitle(f'Prior vs Aggregate Posterior — {model.prior_name} (M={M})', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'./figures/prior_vs_posterior_{model.prior_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim

    prior_gaussian = GaussianPrior(M)
    prior_MoG = MoGPrior(M, K=10)
    prior_flow = FlowPrior(M)

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE models
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model_gaussian = VAE(prior_gaussian, decoder, encoder).to(device)
    model_MoG = VAE(prior_MoG, decoder, encoder).to(device)
    model_flow = VAE(prior_flow, decoder, encoder).to(device)

    models = {"gaussian": model_gaussian, "MoG": model_MoG, "flow": model_flow}

    # Choose mode to run
    if args.mode == 'train':
        for model in models:
            print(f"Training model with {model} prior...")
            # Define optimizer
            optimizer = torch.optim.Adam(models[model].parameters(), lr=1e-3)

            # Train model
            train(models[model], optimizer, mnist_train_loader, args.epochs, args.device)

            # Save model
            torch.save(models[model].state_dict(), f'models/model_{model}.pt')#args.samples)

    elif args.mode == 'sample':
        for model in models:
            models[model].load_state_dict(torch.load(f'models/model_{model}.pt', map_location=torch.device(args.device)))

            # Test model
            evaluate_elbo(models[model], mnist_test_loader, args.device)
            plot_samples(models[model], mnist_test_loader, args.device, M)
            plot_prior_and_posterior(models[model], mnist_test_loader, args.device, M)

            # Generate samples
            models[model].eval()
            with torch.no_grad():
                samples = (models[model].sample(64)).cpu() 
                save_image(samples.view(64, 1, 28, 28), f'sample_gen/model_{model}.png')#args.samples)
