import torch
import torch.nn as nn
import torch.nn.functional as F


######################## MLP model ########################
class MLPModel(nn.Module):
    def __init__(self, adj_matrix, model_cat_unique_levels, n_cont, n_output=2, _continuous_outcome=False):
        super().__init__()
        self.embedding_layers, self.n_emb = self._get_embedding_layers(model_cat_unique_levels)
        self.n_cont = n_cont

        self.lin1 = nn.Linear(self.n_emb + n_cont, 16)
        self.lin2 = nn.Linear(16, 32)
        # if use BCEloss, number of output should be 1, i.e. the probability of getting category 1
        # else number of output should be as specified
        if n_output == 2 or _continuous_outcome:
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
    
    def forward(self, x_cat, x_cont, batched_nodes_indices=None):
        if len(self.embedding_layers) > 0: # if there are categorical variables to be encoded
            x1 = [e(x_cat[:, i]) for i, e in enumerate(self.embedding_layers)]
            x1 = torch.cat(x1, 1)
            x1 = self.emb_drop(x1)

        if self.n_cont > 0: # if there are continuous variables to be encoded
            x2 = self.bn1(x_cont)
        
        if len(self.embedding_layers) > 0 and self.n_cont > 0: # if there are both categorical and continuous variables to be encoded 
            x = torch.cat([x1, x2], 1)
            x = F.relu(self.lin1(x))
            x = self.drops(x)       
            x = self.bn2(x)
            x = F.relu(self.lin2(x))
            x = self.drops(x)
            x = self.bn3(x)
            x = self.lin3(x)        
        elif len(self.embedding_layers) > 0 and self.n_cont == 0: 
            x = F.relu(self.lin1(x1))
            x = self.drops(x)       
            x = self.bn2(x)
            x = F.relu(self.lin2(x))
            x = self.drops(x)
            x = self.bn3(x)
            x = self.lin3(x)
        elif len(self.embedding_layers) == 0 and self.n_cont > 0:
            x = F.relu(self.lin1(x2))
            x = self.drops(x)       
            x = self.bn2(x)
            x = F.relu(self.lin2(x))
            x = self.drops(x)
            x = self.bn3(x)
            x = self.lin3(x)
        else:
            raise ValueError('No variables to be encoded')
    
        return x


######################## GCN model ########################
class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.A = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x, A=None):
        '''
        spatial graph convolution operation
        :param A: (batch_size, batch_size), subgraph sampled from NetworkTMLE.adj_matrix
        :param x: (batch_size, F_in)
        :return: (batch_size, F_out)
        '''
        # self.A = A.to(x.device) # send subgraph to device
        self.A = A
        # print(self.A)
        if self.A.sum().item() == 0.: # no connection in sampled subgraph, no need for graph propagation
            return F.relu(self.Theta(x))
        else:
            return F.relu(self.Theta(torch.matmul(self.A, x)))  # [batch_size, batch_size][batch_size, F_in] -> [batch_size, F_in] -> [batch_size, F_out]


class GCNModel(nn.Module):
    def __init__(self, adj_matrix, model_cat_unique_levels, n_cont, n_output=2, _continuous_outcome=False):
        super(GCNModel, self).__init__()
        self.adj_matrix = adj_matrix

        self.embedding_layers, self.n_emb = self._get_embedding_layers(model_cat_unique_levels)
        self.lin1 = nn.Linear(self.n_emb + n_cont, 16)
        self.gcn = GCNLayer(16, 32)
        if n_output == 2 or _continuous_outcome:
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
        adj_matrix_tensor = torch.from_numpy(adj_matrix.toarray()).float().to(batched_nodes_indices.device)
        # calierate indices for pooled samples, indices restart every num_nodes samples 
        caliberated_batches_nodes_indices = batched_nodes_indices%adj_matrix_tensor.shape[0]
        # select by row and column, order does not matter
        adj_subset = torch.index_select(adj_matrix_tensor, dim=0, index=caliberated_batches_nodes_indices)
        adj_subset = torch.index_select(adj_subset, dim=1, index=caliberated_batches_nodes_indices)
        return adj_subset

    def forward(self, x_cat, x_cont, batched_nodes_indices=None):
        if len(self.embedding_layers) > 0: # if there are categorical variables to be encoded
            x1 = [e(x_cat[:, i]) for i, e in enumerate(self.embedding_layers)]
            x1 = torch.cat(x1, 1)
            x1 = self.emb_drop(x1)
        if self.n_cont > 0: # if there are continuous variables to be encoded
            x2 = self.bn1(x_cont)

        if len(self.embedding_layers) > 0 and self.n_cont > 0: # if there are both categorical and continuous variables to be encoded 
            x = torch.cat([x1, x2], 1)
            x = F.relu(self.lin1(x))
            x = self.drops(x)
            x = self.bn2(x)
            adj_subset = self._get_adj_subset(self.adj_matrix, batched_nodes_indices)
            x = self.gcn(x, adj_subset)
            x = self.bn3(x)
            x = self.lin3(x)
        elif len(self.embedding_layers) > 0 and self.n_cont == 0: 
            x = F.relu(self.lin1(x1))
            x = self.drops(x)
            x = self.bn2(x)
            adj_subset = self._get_adj_subset(self.adj_matrix, batched_nodes_indices)
            x = self.gcn(x, adj_subset)
            x = self.bn3(x)
            x = self.lin3(x)
        elif len(self.embedding_layers) == 0 and self.n_cont > 0:
            x = F.relu(self.lin1(x2))
            x = self.drops(x)
            x = self.bn2(x)
            adj_subset = self._get_adj_subset(self.adj_matrix, batched_nodes_indices)
            x = self.gcn(x, adj_subset)
            x = self.bn3(x)
            x = self.lin3(x)
        else:
            raise ValueError('No variables to be encoded')
        return x
