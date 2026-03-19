# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Extension to compute curve lengths by Søren Hauberg, 2024
# Version 1.0 (2024-02-19)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm

class Poly2(nn.Module):
    def __init__(self, a, b, c):
        """
        Represent the second-order polynomial
          t --> a*t**2 + b**t + c
        
        a, b, c: [torch.Tensor]
           Polynomial coefficients. These must have identical dimensionality
        """
        super(Poly2, self).__init__()
        self.a = a.reshape(1, -1)  # 1xD
        self.b = b.reshape(1, -1)  # 1xD
        self.c = c.reshape(1, -1)  # 1xD

    def forward(self, t):
        _t = t.reshape(-1, 1)  # Tx1
        retval = ( _t**2) * self.a + _t * self.b + self.c.repeat(_t.numel(), 1)  # DxT
        return retval


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


class MoGPrior(nn.Module):
    def __init__(self, M, K):
        super(MoGPrior, self).__init__()
        self.M = M
        self.K = K

        self.weights = nn.Parameter(torch.zeros(K), requires_grad=True)
        self.means = nn.Parameter(torch.randn(self.K, self.M), requires_grad=True)
        self.stds = nn.Parameter(torch.ones(self.K, self.M), requires_grad=True)

    def forward(self):
        mixture = td.Categorical(logits=self.weights)
        components = td.Independent(td.Normal(self.means, self.stds), 1)
        return td.MixtureSameFamily(mixture, components)


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
        return td.Independent(td.Bernoulli(logits=logits), 3)


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
        #elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior.distribuion), dim=0)
        elbo = torch.mean(self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z), dim=0)
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

    def curve_length(self, curve, T):
        """
        Compute the length of a curve when passed through the decoder mean.

        Parameters:
        curve: [Poly2]
           A callable curve parametrization
        T: [torch.Tensor]
           Time indices where the curve should be evaluated, e.g. torch.linspace(0, 1, 100)
        """
        C = curve(T)  # |T|x(latent_dim)
        ambient_C = self.decoder(C).mean  # |T|x(data_shape)
        delta = (ambient_C[1:] - ambient_C[:-1])  # (|T|-1)x(data_shape)
        #retval = torch.sum(torch.sum(torch.sum(delta.reshape(T.numel()-1, -1)**2), dim=1).sqrt())  # scalar
        retval = torch.sum(delta.reshape(T.numel()-1, -1)**2, dim=1).sqrt().sum()
        return retval


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
    num_steps = len(data_loader)*epochs
    epoch = 0

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            x = next(iter(data_loader))[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Report
            if step % 5 ==0 :
                loss = loss.detach().cpu()
                pbar.set_description(f"epoch={epoch}, step={step}, loss={loss:.1f}")

            if (step+1) % len(data_loader) == 0:
                epoch += 1


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval', 'plot', 'curves'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--plot', type=str, default='plot.png', help='file to save latent plot in (default: %(default)s)')
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
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    #prior = GaussianPrior(M)
    prior = MoGPrior(M, 20)

    # Define encoder and decoder networks
    # encoder_net = nn.Sequential(
    #     nn.Flatten(),
    #     nn.Linear(784, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, M*2),
    # )

    # decoder_net = nn.Sequential(
    #     nn.Linear(M, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, 784),
    #     nn.Unflatten(-1, (28, 28))
    # )

    encoder_net = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(512, 2*M),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.Unflatten(-1, (32, 4, 4)),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
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

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)

    elif args.mode == 'eval':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()
        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == 'plot':
        import matplotlib.pyplot as plt
        import sklearn.decomposition

        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        samples_z = []
        samples_y = []

        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                z = model.encoder(x).sample()
                samples_z.append(z)
                samples_y.append(y)

            samples_z = torch.cat(samples_z, dim=0).numpy()
            samples_y = torch.cat(samples_y, dim=0).numpy()

        # Plot latent space
        pca = sklearn.decomposition.PCA(n_components=2)
        pca.fit(samples_z)
        samples_z_pca = pca.transform(samples_z)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(samples_z_pca[:, 0], samples_z_pca[:, 1], c=samples_y, cmap='tab10')
        plt.savefig(args.plot)

    elif args.mode == 'curves':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        curve = Poly2(torch.randn(M),
                      torch.randn(M),
                      torch.randn(M))
        T = torch.linspace(0, 1, 100)

        curve_len = model.curve_length(curve, T)
        print('The random curve has length {}'.format(curve_len))
