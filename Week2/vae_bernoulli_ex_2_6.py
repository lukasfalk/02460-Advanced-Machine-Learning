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
        Exercise 1.6
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class MoGPrior(nn.Module):
    def __init__(self, M, K):
        """
        Define a Mixture of Gaussians prior distribution.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        K: [int]
           Number of mixture components.
        """
        super(MoGPrior, self).__init__()
        self.M = M
        self.K = K
        self.means = nn.Parameter(torch.zeros(self.K, self.M), requires_grad=True)
        self.stds = nn.Parameter(torch.ones(self.K, self.M), requires_grad=True)
        self.logits = nn.Parameter(torch.zeros(self.K), requires_grad=True)
        self.log_stds = nn.Parameter(torch.zeros(self.K, self.M))

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        component_dist = td.Independent(td.Normal(self.means, torch.exp(self.log_stds)), 1)
        mixture_dist = td.Categorical(logits=self.logits)
        return td.MixtureSameFamily(mixture_dist, component_dist)
    
class FlowPrior(nn.Module):
    def __init__(self, M, num_transformations=4, num_hidden=128):
        super(FlowPrior, self).__init__()
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


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


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
        elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        return elbo
    
    def elbo_GoM(self, x):
        """
        Exercise 1.6
        Compute the ELBO for the given batch of data using a Mixture of Gaussians prior.
        Parameters:
        x: [torch.Tensor]
        """
        # 1. Get the approximate posterior distribution q(z|x)
        q = self.encoder(x)
        
        # 2. Sample from the approximate posterior using the reparameterization trick
        z = q.rsample()
        
        # 3. Calculate log p(x|z) - The reconstruction term
        log_px_z = self.decoder(z).log_prob(x)
        
        # 4. Calculate log p(z) - The MoG prior term
        log_pz = self.prior().log_prob(z)
        
        # 5. Calculate log q(z|x) - The entropy term
        log_qz_x = q.log_prob(z)
        
        # Combined ELBO: log p(x|z) + log p(z) - log q(z|x)
        # We take the mean over the batch dimension
        elbo = torch.mean(log_px_z + log_pz - log_qz_x, dim=0)
        return elbo
    
    def elbo_flow(self, x):
        q = self.encoder(x)
        z = q.rsample()
        log_px_z = self.decoder(z).log_prob(x)
        log_pz = self.prior.log_prob(z)
        log_qz_x = q.log_prob(z)
        return torch.mean(log_px_z + log_pz - log_qz_x, dim=0)

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior.sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo_flow(x) # Change to elbo(x) if using Gaussian prior


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
    print(f"ELBO evaluation: {avg_elbo:.4f}")

def evaluate_elbo_GoM(model, data_loader, device):
    """
    Exercise 1.6
    Evaluate the ELBO on the binarised MNIST test set using a Mixture of Gaussians prior.
    """
    model.eval()
    total_elbo = 0
    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            elbo = model.elbo_GoM(x)
            total_elbo += elbo.item()
    
    avg_elbo = total_elbo / len(data_loader)
    print(f"ELBO evaluation: {avg_elbo:.4f}")

def evaluate_elbo_flow(model, data_loader, device):
    """
    Exercise 1.6
    Evaluate the ELBO on the binarised MNIST test set using a Flow prior.
    """
    model.eval()
    total_elbo = 0
    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            elbo = model.elbo_flow(x)
            total_elbo += elbo.item()
    
    avg_elbo = total_elbo / len(data_loader)
    print(f"ELBO evaluation: {avg_elbo:.4f}")

def plot_samples(model, data_loader, device, M):
    """
    Exercise 1.5
    Plot samples from the approximate posterior and colour them by their correct class
    label for each datapoint in the test set (i.e., samples from the aggregate posterior).
    Implement it such that you, for latent dimensions larger than two, M > 2, do
    PCA and project the sample onto the first two principal components (e.g., using
    scikit-learn).
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
    plt.show()

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
    # prior = GaussianPrior(M)
    prior = FlowPrior(M)

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

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Test model
        evaluate_elbo_flow(model, mnist_test_loader, args.device)

        plot_samples(model, mnist_test_loader, args.device, M)

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)
