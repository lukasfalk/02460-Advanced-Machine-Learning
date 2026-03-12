# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import json

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
from vae_part_A import GaussianPrior, VAE, GaussianEncoder

class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        mean = self.decoder_net(z)
        return td.Independent(td.Normal(loc=mean, scale=0.1*torch.ones_like(mean)), 2)

class BetaVAE(VAE):
    def __init__(self, encoder, decoder, prior, beta=1.0):
        """
        Initialize a Beta-VAE model.

        Parameters:
        encoder: [nn.Module]
            The encoder network to use for the VAE.
        decoder: [nn.Module]
            The decoder network to use for the VAE.
        prior: [GaussianPrior]
            The prior distribution to use for the VAE.
        beta: [float]
            The weight of the KL divergence term in the ELBO (default: 1.0).
        """
        super(BetaVAE, self).__init__(prior, decoder, encoder)
        self.beta = beta

    def loss(self, x):
        """
        Evaluate the Beta-VAE loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        q = self.encoder(x)
        z = q.rsample()
        log_px_z = self.decoder(z).log_prob(x.view(-1, 28, 28))
        kl = td.kl_divergence(q, self.prior())
        return -torch.mean(log_px_z - self.beta * kl)

class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False) # alpha_bar in the lecture notes
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### Implement Algorithm 1 here ###
        t = torch.distributions.Categorical(logits=torch.zeros(self.T)).sample((x.shape[0],)).to(x.device)
        epsilon = torch.randn_like(x)
        alpha_t = self.alpha_cumprod[t].view(-1, 1)
        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * epsilon
        neg_elbo = torch.sum((epsilon - self.network(z_t, t.unsqueeze(1)/self.T))**2, dim=1)

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1):
            ### Implement the remaining of Algorithm 2 here ###
            eta = torch.randn_like(x_t) if t > 0 else 0
            alpha_t = self.alpha[t].view(-1, 1)
            alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1)
            beta_t = self.beta[t].view(-1, 1)
            epsilon_theta = self.network(x_t, torch.full((shape[0], 1), t/self.T, device=x_t.device))
            x_t = (1/torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * epsilon_theta) + torch.sqrt(beta_t) * eta

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
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
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), 
                                     nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), 
                                     nn.ReLU(), 
                                     nn.Linear(num_hidden, input_dim))

    def forward(self, x, t):
        """"
        Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import time
    import json
    # import ToyData as ToyData
    from unet import Unet

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Generate the data
    n_data = 10000000
    # toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
    # transform = lambda x: (x-0.5)*2.0
    # train_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
                                    transforms.Lambda(lambda x: (x-0.5)*2.0),
                                    transforms.Lambda(lambda x: x.flatten())]
                                    )
    train_data = datasets.MNIST('data/', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('data/', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # Get the dimension of the dataset
    D = 784 # next(iter(train_loader))[0].shape[1]
    M = 10 # Latent dimension

    # Define the network
    Fc_network = FcNetwork(M, num_hidden=256)

    # U-Net
    unet_network = Unet()

    # Beta-VAE
    prior_gaussian = GaussianPrior(M)

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
    # decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    decoder = GaussianDecoder(decoder_net)
    model_gaussian = VAE(prior_gaussian, decoder, encoder).to(device)

    # Set the number of steps in the diffusion process
    T = 1000

    # Define model
    model_unet = DDPM(unet_network, T=T).to(args.device)
    model_BetaVAE = BetaVAE(encoder, decoder, prior_gaussian).to(args.device)
    model_DDPM_BVAE = DDPM(Fc_network, T=T).to(args.device)
    

    models = {'unet': model_unet, 'beta_vae': model_BetaVAE, 'latent': model_DDPM_BVAE}
    beta_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

    if args.mode == 'train':
        # Train U-Net DDPM on images directly
        optimizer = torch.optim.Adam(model_unet.parameters(), lr=args.lr)
        train(model_unet, optimizer, train_loader, args.epochs, args.device)
        torch.save(model_unet.state_dict(), 'models/PartB/model_unet.pt')

        # Train β-VAE
        for beta in beta_values:
            print(f"Training β-VAE with β={beta}")
            model_BetaVAE.beta = beta
            optimizer = torch.optim.Adam(model_BetaVAE.parameters(), lr=args.lr)
            train(model_BetaVAE, optimizer, train_loader, args.epochs, args.device)
            torch.save(model_BetaVAE.state_dict(), f'models/PartB/model_beta_{beta}_vae.pt')

            # Train latent DDPM in β-VAE latent space
            model_BetaVAE.eval()
            
            optimizer = torch.optim.Adam(model_DDPM_BVAE.parameters(), lr=args.lr)

            for epoch in range(args.epochs):
                for x, _ in train_loader:
                    x = x.to(args.device)
                    with torch.no_grad():
                        z = model_BetaVAE.encoder(x).mean  # encode to latent space
                    optimizer.zero_grad()
                    loss = model_DDPM_BVAE.loss(z)
                    loss.backward()
                    optimizer.step()

            torch.save(model_DDPM_BVAE.state_dict(), f'models/PartB/model_latent_ddpm_{beta}.pt')

    elif args.mode == 'sample':
        model_unet.load_state_dict(torch.load('models/PartB/model_unet.pt', map_location=args.device))
        model_unet.eval()
        results = {}

        with torch.no_grad():
            # Sample from U-Net DDPM
            start = time.time()
            samples_unet = model_unet.sample((64, D))
            torch.cuda.synchronize()
            results['unet'] = {'samples_per_sec': 64 / (time.time() - start)}
            save_image(samples_unet.cpu().view(64, 1, 28, 28) / 2 + 0.5, 'sample_gen/PartB/model_unet.png')

            for beta in beta_values:
                model_BetaVAE.load_state_dict(torch.load(f'models/PartB/model_beta_{beta}_vae.pt', map_location=args.device))
                model_DDPM_BVAE.load_state_dict(torch.load(f'models/PartB/model_latent_ddpm_{beta}.pt', map_location=args.device))
                model_BetaVAE.eval()
                model_DDPM_BVAE.eval()

                # Beta-VAE sampling + timing
                start = time.time()
                samples_vae = model_BetaVAE.sample(64)
                torch.cuda.synchronize()
                results[f'beta_vae_{beta}'] = {'samples_per_sec': 64 / (time.time() - start)}
                save_image(samples_vae.cpu().clamp(0, 1).view(64, 1, 28, 28), f'sample_gen/PartB/model_beta_{beta}_vae.png')

                # Latent DDPM sampling + timing
                start = time.time()
                z = model_DDPM_BVAE.sample((64, M))
                samples_latent = model_BetaVAE.decoder(z).mean
                torch.cuda.synchronize()
                results[f'latent_ddpm_{beta}'] = {'samples_per_sec': 64 / (time.time() - start)}
                save_image(samples_latent.cpu().clamp(0, 1).view(64, 1, 28, 28), f'sample_gen/PartB/model_latent_ddpm_{beta}.png')

        with open('results/sampling_times.json', 'w') as f:
            json.dump(results, f, indent=2)