import torch
import torch.nn as nn
import torch.nn.functional as F


######################## MLP model ########################
class MLPModel(nn.Module):
    def __init__(self, model_cat_unique_levels, n_cont, n_output=2):
        super().__init__()
        self.embedding_layers, self.n_emb = self._get_embedding_layers(model_cat_unique_levels)
        self.lin1 = nn.Linear(self.n_emb + n_cont, 16)
        self.lin2 = nn.Linear(16, 32)
        # if use BCEloss, number of output should be 1, i.e. the probability of getting category 1
        # else number of output should be as specified
        if n_output == 2:
            self.lin3 = nn.Linear(32, 1) 
        else:
            self.lin3 = nn.Linear(32, n_output)
        self.bn1 = nn.BatchNorm1d(n_cont)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

    def _get_embedding_layers(self, model_cat_unique_levels):
        # Ref: https://jovian.ml/aakanksha-ns/shelter-outcome
        # decide embedding sizes
        embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _, n_categories in model_cat_unique_levels.items()]
        embedding_layers = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in embedding_layers) # length of all embeddings combined
        # n_cont = dataset.x_cont.shape[1] # number of continuous variables

        return embedding_layers, n_emb
    
    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embedding_layers)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)
        return x


######################## GCN model ########################
#TODO: A is for all samples, for a batch N is different, information is incomplete

dummy_A = torch.randn(16, 16)
dummy_x = torch.randn(16, 4)
out = torch.matmul(dummy_A, dummy_x)
out.shape

from beowulf import load_uniform_statin
from beowulf.dgm import statin_dgm
import networkx as nx

n_nodes = 500
G, cat_vars, cont_vars, cat_unique_levels = load_uniform_statin(n=n_nodes, return_cat_cont_split=True)
H, cat_vars, cont_vars, cat_unique_levels = statin_dgm(network=G, restricted=False,
                                                        update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)

# Generate a fresh copy of the network with ascending node order
oid = "_original_id_"                                              # Name to save the original IDs
network = nx.convert_node_labels_to_integers(H,              # Copy of new network with new labels
                                             first_label=0,        # ... start at 0 for latent variance calc
                                             label_attribute=oid)  # ... saving the original ID labels

adj_matrix = nx.adjacency_matrix(network,   # Convert to adjacency matrix
                                 weight=None)    # TODO allow for weighted networks

# adj_matrix_tmp = adj_matrix.todense()
# adj_matrix_tmp = nx.to_numpy_array(network) # returns numpy array
adj_matrix_tmp = adj_matrix.toarray() # returns numpy array
adj_matrix_tmp = torch.from_numpy(adj_matrix_tmp) # convert to tensor
adj_matrix_tmp

dummy_indices = torch.tensor([0, 2, 9, 11, 267])
dummy_indices.shape

adj_subset = torch.index_select(adj_matrix_tmp, dim=0, index=dummy_indices) # select by rows
adj_subset = torch.index_select(adj_subset, dim=1, index=dummy_indices) # select by columns

adj_subset


class GCNLayer(nn.Module):
    def __init__(self, A, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.A = A # subgraph sampled from NetworkTMLE.adj_matrix, shape [batch_size, batch_size]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, F_in)
        :return: (batch_size, F_out)
        '''
        self.A = self.A.to(x.device) # send subgraph to device
        return F.relu(self.Theta(torch.matmul(self.A, x)))  # [batch_size, batch_size][batch_size, F_in] -> [batch_size, F_in] -> [batch_size, F_out]


class GCNModel(nn.Module):
    def __init__(self, adj_matrix, batched_nodes_indices, model_cat_unique_levels, n_cont, n_output=2):
        super(GCNModel, self).__init__()
        self.embedding_layers, self.n_emb = self._get_embedding_layers(model_cat_unique_levels)
        self.lin1 = nn.Linear(self.n_emb + n_cont, 16)
        self.gcn = GCNLayer(self._get_adj_subset(adj_matrix, batched_nodes_indices), 16, 32)
        if n_output == 2:
            self.lin3 = nn.Linear(32, 1) 
        else:
            self.lin3 = nn.Linear(32, n_output)
        self.bn1 = nn.BatchNorm1d(n_cont)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)
        

    def _get_embedding_layers(self, model_cat_unique_levels):
        # Ref: https://jovian.ml/aakanksha-ns/shelter-outcome
        # decide embedding sizes
        embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _, n_categories in model_cat_unique_levels.items()]
        embedding_layers = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in embedding_layers) # length of all embeddings combined
        # n_cont = dataset.x_cont.shape[1] # number of continuous variables

        return embedding_layers, n_emb

    def _get_adj_subset(self, adj_matrix, batched_nodes_indices):
        # conver SciPy sparse array to torch.tensor
        adj_matrix_tensor = torch.from_numpy(adj_matrix.toarray())
        # select by row and column, order does not matter
        adj_subset = torch.index_select(adj_matrix_tensor, dim=0, index=batched_nodes_indices)
        adj_subset = torch.index_select(adj_subset, dim=1, index=batched_nodes_indices)
        return adj_subset

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embedding_layers)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = self.gcn(x)
        x = self.bn3(x)
        x = self.lin3(x)
        return x


