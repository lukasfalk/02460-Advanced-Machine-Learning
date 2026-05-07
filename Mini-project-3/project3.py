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
import numpy as np

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
    
    def ELBO(self, x, edge_index, batch, A, mask, kl_weight):
        p, q, prior = self.forward(x, edge_index, batch)
        log_pxz = (p.base_dist.log_prob(A) * mask).sum(dim=[-1,-2]) / mask.sum(dim=[-1,-2])
        kl_qp = td.kl_divergence(q, prior)
        elbo = log_pxz - kl_weight * kl_qp
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
latent_dim = 2
model = SimpleGNN(node_feature_dim, state_dim, num_message_passing_rounds, latent_dim).to(device)

hidden_dim = 64
N_max = max(graph.num_nodes for graph in dataset)
decoder_net = torch.nn.Sequential(
    torch.nn.Linear(latent_dim, hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, N_max*N_max)
)

vae = VAE(model, decoder_net, GaussianPrior(latent_dim)).to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

elbos = []

epochs = 1000

for epoch in range(epochs):
    # Loop over training batches
    vae.train()
    for data in train_loader:
        num_graphs = data.batch.max()+1

        # Compute adjacency matrices and node features per graph
        A = to_dense_adj(data.edge_index, data.batch, max_num_nodes=N_max).float()
        kl_weight = min(1.0, epoch/400)
        X, node_mask = to_dense_batch(data.x, data.batch, max_num_nodes=N_max)
        edge_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)

        loss = -vae.ELBO(data.x, data.edge_index, data.batch, A, edge_mask, kl_weight)
        elbos.append(loss.item())
        # Gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    # Plot the training
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, ELBO: {elbos[-1]:.3f}')
        plt.figure('ELBO').clf()
        plt.plot(elbos)
        plt.xlabel('Batch')
        plt.ylabel('ELBO')
        plt.tight_layout()
        drawnow()

plt.figure(figsize=(8, 4))
plt.plot(elbos)
plt.xlabel('Batch')
plt.ylabel('ELBO')
plt.tight_layout()
plt.savefig('elbos.png', dpi=150)
plt.clf()

def adj_to_nx(A):
    G = nx.from_numpy_array(A.cpu().numpy())
    return G

def compute_metrics(graphs):
    hashes = [nx.weisfeiler_lehman_graph_hash(graph) for graph in graphs]
    novel = [h not in train_hashes for h in hashes]
    unique = [hashes.count(h) == 1 for h in hashes]
    nov_and_uniq = [n and u for n, u in zip(novel, unique)]
    return sum(novel)/len(novel), sum(unique)/len(unique), sum(nov_and_uniq)/len(nov_and_uniq)

def get_graph_stats(graphs):
    degrees, clusterings, centralities = [], [], []
    for graph in graphs:
        if graph.number_of_nodes() == 0:
            continue
        degrees += [d for _, d in graph.degree()]
        clusterings += list(nx.clustering(graph).values())
        try:
            cent = nx.eigenvector_centrality(graph, max_iter=1000)
            centralities += list(cent.values())
        except:
            centralities += [0.0] * graph.number_of_nodes()
    return degrees, clusterings, centralities

def sample_vae_graph(vae, node_counts):
    N = random.choice(node_counts)
    z = vae.prior().sample((1,))
    A = vae.decoder(z).sample()[0, :N, :N]
    A = torch.triu(A, diagonal=1)
    A = A + A.T
    return adj_to_nx(A)

node_counts = [graph.num_nodes for graph in train_dataset]
with torch.no_grad():
    # Sampling
    baseline_samples = [erdos_renyi(train_dataset) for _ in range(1000)]
    vae_graphs = [sample_vae_graph(vae, node_counts) for _ in range(1000)]

    # Novelty and uniqueness
    train_hashes = set(
        nx.weisfeiler_lehman_graph_hash(adj_to_nx(
            to_dense_adj(g.edge_index)[0]
        )) for g in train_dataset
    )

    baseline_novel, baseline_unique, baseline_both = compute_metrics(baseline_samples)
    vae_novel, vae_unique, vae_both = compute_metrics(vae_graphs)

    print(f"Baseline:  Novel={baseline_novel:.1%}, Unique={baseline_unique:.1%}, Both={baseline_both:.1%}")
    print(f"VAE:       Novel={vae_novel:.1%}, Unique={vae_unique:.1%}, Both={vae_both:.1%}")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        nx.draw(vae_graphs[i], ax=ax, node_size=50, with_labels=False)
        ax.set_title(f'Graph {i}')
    plt.tight_layout()
    plt.savefig('vae_graphs.png', dpi=150)

    ## Plotting histograms
    # Train graphs for comparison
    train_graphs = [adj_to_nx(to_dense_adj(g.edge_index, max_num_nodes=N_max)[0]) for g in train_dataset]

    # Graph stats
    train_stats = get_graph_stats(train_graphs)
    baseline_stats = get_graph_stats(baseline_samples)
    vae_stats = get_graph_stats(vae_graphs)

    labels = ['Node Degree', 'Clustering Coefficient', 'Eigenvector Centrality']
    bins_list = [np.linspace(0, 28, 30), np.linspace(0, 1, 30), np.linspace(0, 1, 30)]
    row_labels = ['Train', 'Baseline', 'VAE']
    all_stats = [train_stats, baseline_stats, vae_stats]

    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    for col, (metric_label, bins) in enumerate(zip(labels, bins_list)):
        for row, (stats, row_label) in enumerate(zip(all_stats, row_labels)):
            ax = axes[row, col]
            ax.hist(stats[col], bins=bins, density=True, color='steelblue', edgecolor='white', linewidth=0.3)
            if row == 0:
                ax.set_title(metric_label, fontsize=11)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=11)
    plt.tight_layout()
    plt.savefig('histograms.png', dpi=150)
    plt.show()