# %%
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch.distributions as td
import networkx as nx
from torch_geometric.utils import to_dense_adj, to_dense_batch

# %% Interactive plots
plt.ion() # Enable interactive plotting
def drawnow():
    """Force draw the current plot."""
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()

# %% Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %% Load the MUTAG dataset
# Load data
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
node_feature_dim = 7

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

# Create dataloader for training and validation
train_loader = DataLoader(train_dataset, batch_size=100)
validation_loader = DataLoader(validation_dataset, batch_size=44)
test_loader = DataLoader(test_dataset, batch_size=44)

### Question C.1
# data_batch = next(iter(train_loader))
# print(data_batch.x)
# print(data_batch.x.shape)
# print(data_batch.edge_index)
# print(data_batch.edge_index.shape)
# print(data_batch.batch)
# print(data_batch.batch.shape)
# exit() # exit() to avoid running the rest of the code before answering question C.1

# %% Define a simple GNN for graph classification
class SimpleGNN(torch.nn.Module):
    """Simple graph neural network for graph classification

    Keyword Arguments
    -----------------
        node_feature_dim : Dimension of the node features
        state_dim : Dimension of the node states
        num_message_passing_rounds : Number of message passing rounds
    """

    def __init__(self, node_feature_dim, state_dim, num_message_passing_rounds, latent_dim):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds
        self.latent_dim = latent_dim

        # Input network
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.node_feature_dim, self.state_dim),
            torch.nn.ReLU()
            )

        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])

        # Update network
        self.update_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)])

        # State output network
        self.output_net = torch.nn.Linear(self.state_dim, 2*self.latent_dim)

    def forward(self, x, edge_index, batch):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.

        Returns
        -------
        out : torch tensor (num_graphs)
            Neural network output for each graph.

        """
        # Extract number of nodes and graphs
        num_graphs = batch.max()+1
        num_nodes = batch.shape[0]

        # Initialize node state from node features
        state = self.input_net(x)
        # state = x.new_zeros([num_nodes, self.state_dim]) # Uncomment to disable the use of node features

        # Loop over message passing rounds
        for r in range(self.num_message_passing_rounds):
            # Compute outgoing messages
            message = self.message_net[r](state)

            # Aggregate: Sum messages
            aggregated = x.new_zeros((num_nodes, self.state_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])

            # Update states
            state = state + self.update_net[r](aggregated)

        # Aggretate: Sum node features
        graph_state = x.new_zeros((num_graphs, self.state_dim))
        graph_state = torch.index_add(graph_state, 0, batch, state)

        # Output
        out = self.output_net(graph_state)
        return out

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

    def forward(self, x, edge_index, batch):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x, edge_index, batch), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)

class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        mean = self.decoder_net(z)
        return td.Independent(td.Normal(loc=mean, scale=0.1*torch.ones_like(mean)), 2)

class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        logits = self.decoder_net(z).view(-1, N_max, N_max)
        return td.Independent(td.Bernoulli(logits=logits), 2)

class VAE(torch.nn.Module):
    def __init__(self, encoder_net, decoder_net, prior):
        super(VAE, self).__init__()
        self.encoder = GaussianEncoder(encoder_net)
        self.decoder = BernoulliDecoder(decoder_net)
        self.prior = prior

    def forward(self, x, edge_index, batch):
        q = self.encoder(x, edge_index, batch)
        z = q.rsample()
        p = self.decoder(z)
        return p, q, self.prior()
    
    def ELBO(self, x, edge_index, batch, A):
        p, q, prior = self.forward(x, edge_index, batch)
        log_pxz = p.log_prob(A)
        kl_qp = td.kl_divergence(q, prior)
        elbo = log_pxz - kl_qp
        return elbo.mean()

def erdos_renyi(train_dataset):
    # Baseline based on Erdos-Renyi model
    # 1. Sampling with empirical distribution
    node_counts = [graph.num_nodes for graph in train_dataset]
    N = random.choice(node_counts)

    # 2. Compute link probabilities
    graphs_with_N_nodes = [graph for graph in train_dataset if graph.num_nodes == N]
    edge_counts = sum([graph.num_edges for graph in graphs_with_N_nodes]) // 2
    total_possible_edges = N * (N - 1) // 2
    r = edge_counts / total_possible_edges

    # 3. Sample a random graph
    G = nx.erdos_renyi_graph(n=N, p=r)

    return G

# %% Set up the model, loss, and optimizer etc.
# Instantiate the model
state_dim = 16
num_message_passing_rounds = 4
latent_dim = 8
model = SimpleGNN(node_feature_dim, state_dim, num_message_passing_rounds, latent_dim).to(device)

hidden_dim = 64
N_max = max(graph.num_nodes for graph in dataset)
decoder_net = torch.nn.Sequential(
    torch.nn.Linear(latent_dim, hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, N_max*N_max)
)

vae = VAE(model, decoder_net, GaussianPrior(latent_dim)).to(device)

# Optimizer
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

# %% Lists to store accuracy and loss
train_accuracies = []
train_losses = []
validation_accuracies = []
validation_losses = []

losses = []

# %% Fit the model
# Number of epochs
epochs = 500

for epoch in range(epochs):
    # Loop over training batches
    vae.train()
    for data in train_loader:
        # out = model(data.x, data.edge_index, batch=data.batch)
        # loss = cross_entropy(out, data.y.float())
        num_graphs = data.batch.max()+1

        # Compute adjacency matrices and node features per graph
        A = to_dense_adj(data.edge_index, data.batch, max_num_nodes=N_max).float()
        # X, idx = to_dense_batch(data.x, data.batch)
        loss = -vae.ELBO(data.x, data.edge_index, batch=data.batch, A=A)
        losses.append(loss.item())
        # Gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # Learning rate scheduler step
    scheduler.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {losses[-1]:.3f}')
        plt.figure('Loss').clf()
        plt.plot(losses)
        plt.xlabel('Batch')
        plt.ylabel('ELBO')
        plt.tight_layout()
        drawnow()

# %% Save final predictions.
with torch.no_grad():
    data = next(iter(test_loader))
    out = vae.decoder(vae.encoder(data.x, data.edge_index, data.batch).rsample())
    torch.save(out, 'test_predictions.pt')