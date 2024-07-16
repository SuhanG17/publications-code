import torch
import torch.nn as nn
import torch.nn.functional as F
from dl_layers import ReverseLayerF


######################## MLP model ########################
class MLPModel(nn.Module):
    def __init__(self, adj_matrix, model_cat_unique_levels, n_cont, n_output=2, _continuous_outcome=False):
        super(MLPModel, self).__init__()
        self.embedding_layers, self.n_emb = self._get_embedding_layers(model_cat_unique_levels)
        self.n_cont = n_cont

        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 16)
        self.lin2 = nn.Linear(16, 32)
        # if use BCEloss, number of output should be 1, i.e. the probability of getting category 1
        # else number of output should be as specified
        if n_output == 2 or _continuous_outcome:
            self.lin3 = nn.Linear(32, 1) 
        else:
            self.lin3 = nn.Linear(32, n_output)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

        self._init_weights()

    def _get_embedding_layers(self, model_cat_unique_levels):
        # Ref: https://jovian.ml/aakanksha-ns/shelter-outcome
        # decide embedding sizes
        embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _, n_categories in model_cat_unique_levels.items()]
        embedding_layers = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in embedding_layers) # length of all embeddings combined
        # n_cont = dataset.x_cont.shape[1] # number of continuous variables

        return embedding_layers, n_emb
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
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


class MLPModelTimeSeries(nn.Module):
    def __init__(self, adj_matrix_list, model_cat_unique_levels, n_cont, T_in=10, T_out=10,
                 n_output=2, _continuous_outcome=False):
        super(MLPModelTimeSeries, self).__init__()
        self.embedding_layers, self.n_emb = self._get_embedding_layers(model_cat_unique_levels)
        self.n_cont = n_cont

        # variable dimension feature extraction
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 16)
        self.lin2 = nn.Linear(16, 32)
        # if use BCEloss, number of output should be 1, i.e. the probability of getting category 1
        # else number of output should be as specified
        if n_output == 2 or _continuous_outcome:
            self.lin3 = nn.Linear(32, 1) 
        else:
            self.lin3 = nn.Linear(32, n_output)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

        # time dimension feature extract 
        self.ts_lin1 = nn.Linear(T_in, 16)
        self.ts_lin2 = nn.Linear(16, T_out)

        self._init_weights()

    def _get_embedding_layers(self, model_cat_unique_levels):
        # Ref: https://jovian.ml/aakanksha-ns/shelter-outcome
        # decide embedding sizes
        embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _, n_categories in model_cat_unique_levels.items()]
        embedding_layers = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in embedding_layers) # length of all embeddings combined
        # n_cont = dataset.x_cont.shape[1] # number of continuous variables

        return embedding_layers, n_emb

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 
    
    def forward(self, x_cat, x_cont, batched_nodes_indices=None):
        # x_cat: [batch_size, num_cat_vars, T]
        # x_cont: [batch_size, num_cont_vars, T]
        # batched_nodex_indices: [batch_size]

        x_cat_new = x_cat.permute(0, 2, 1)
        # x_cat_new: [batch_size, T, num_cat_vars]

        if len(self.embedding_layers) > 0: # if there are categorical variables to be encoded
            x1 = [e(x_cat_new[:, :, 1]) for i, e in enumerate(self.embedding_layers)]
            x1 = torch.cat(x1, -1) # [batch_size, T, n_emb]
            x1 = self.emb_drop(x1)

        if self.n_cont > 0: # if there are continuous variables to be encoded
            x2 = self.bn1(x_cont).permute(0, 2, 1) # [batch_size, T, n_cont]
        
        if len(self.embedding_layers) > 0 and self.n_cont > 0: # if there are both categorical and continuous variables to be encoded 
            x = torch.cat([x1, x2], -1) # [batch_size, T, n_emb + n_cont]
            # temporal perspective
            x = F.relu(self.ts_lin1(x.permute(0, 2, 1))).permute(0, 2, 1) 
            # [batch_size, T, n_emb + n_cont] -> [batch_size, n_emb + n_cont, T] 
            # -> [batch_size, n_emb + n_cont, 16] -> [batch_size, 16, n_emb + n_cont]

            # variable perspective
            x = F.relu(self.lin1(x)) # [batch_size, 16, n_emb + n_cont] -> [batch_size, 16, 16]
            x = self.drops(x)       
            x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1) 
            # [batch_size, 16(ts_c), 16(v_c)] -> [batch_size, 16(v_c), 16(ts_c)] ->  [batch_size, 16(ts_c), 16(v_c)]
            x = F.relu(self.lin2(x)) # [batch_size, 16, 16] -> [batch_size, 16, 32] 
            x = self.drops(x)
            x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
            # [batch_size, 16, 32] -> [batch_size, 32, 16] -> [batch_size, 16, 32
            x = self.lin3(x)         # [batch_size, 16, 32] -> [batch_size, 16, 1]

            # temporal perspective
            x = self.ts_lin2(x.permute(0, 2, 1))
            # [batch_size, 16, 1] -> [batch_size, 1, 16] -> [batch_size, 1, T] 

        elif len(self.embedding_layers) > 0 and self.n_cont == 0: 
            # temporal perspective
            x = self.ts_lin1(x1.permute(0, 2, 1)).permute(0, 2, 1)
            # variable perspective
            x = F.relu(self.lin1(x))
            x = self.drops(x)       
            x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = F.relu(self.lin2(x))
            x = self.drops(x)
            x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.lin3(x)
            # temporal perspective
            x = self.ts_lin2(x.permute(0, 2, 1))

        elif len(self.embedding_layers) == 0 and self.n_cont > 0:
            # temporal perspective
            x = self.ts_lin1(x2.permute(0, 2, 1)).permute(0, 2, 1)
            # variable perspective
            x = F.relu(self.lin1(x))
            x = self.drops(x)       
            x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = F.relu(self.lin2(x))
            x = self.drops(x)
            x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.lin3(x)
            # temporal perspective
            x = self.ts_lin2(x.permute(0, 2, 1))
        else:
            raise ValueError('No variables to be encoded')
    
        return x

class MLPModelTimeSeriesNumerical(nn.Module):
    def __init__(self, adj_matrix_list, model_cat_unique_levels, n_cont, T_in=10, T_out=10,
                 n_output=2, _continuous_outcome=False, lin_hidden=None, lin_hidden_temporal=None):
        super(MLPModelTimeSeriesNumerical, self).__init__()
        n_cat = len(model_cat_unique_levels)
        n_input = n_cat + n_cont
        # feature dim
        self.lin_input = nn.Linear(n_input, 32)
        if lin_hidden is not None:
            self.lin_hidden = lin_hidden
        else:
            self.lin_hidden = nn.ModuleList([nn.Linear(32, 128), nn.Linear(128, 512), 
                                            nn.Linear(512, 128), nn.Linear(128, 32)])
        if n_output == 2 or _continuous_outcome:
            self.lin_output = nn.Linear(32, 1) 
        else:
            self.lin_output = nn.Linear(32, n_output)
        
        # temporal dim
        if T_in > 1: # T_in > 1 and T_out >= 1
            self.lin_input_temporal = nn.Linear(T_in, 128)
            self.lin_output_temporal = nn.Linear(128, T_out)
            if lin_hidden_temporal is not None:
                self.lin_hidden_temporal = lin_hidden_temporal
            else:
                self.lin_hidden_temporal = None
        else: # T_in = 1 and T_out = 1
            self.lin_input_temporal = None
            self.lin_output_temporal = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 

    def forward(self, x_cat, x_cont, batched_nodes_indices=None):
        # x_cat: [batch_size, num_cat_vars, T_in]
        # x_cont: [batch_size, num_cont_vars, T_in]
        # batched_nodex_indices: [batch_size]

        x1, x2 = x_cat.permute(0, 2, 1), x_cont.permute(0, 2, 1) 
        x =  torch.cat([x1, x2], -1) # -> [batch_size, T_in, num_cat_vars + num_cont_vars]

        x = F.relu(self.lin_input(x))
        for layer in self.lin_hidden:
            x = F.relu(layer(x))
        x = self.lin_output(x) # -> [batch_size, T_in, 1]

        if self.lin_input_temporal is not None:
            x = F.relu(self.lin_input_temporal(x.permute(0, 2, 1))) # -> [batch_size, 1, 128]
            if self.lin_hidden_temporal is not None:
                for layer in self.lin_hidden_temporal:
                    x = F.relu(layer(x))
            x = self.lin_output_temporal(x) # -> [batch_size, 1, T_out]
            return x
        else:
            return x.permute(0, 2, 1) # -> [batch_size, 1, 1]


class MLPModelTimeSeriesNumericalUDA(nn.Module):
    def __init__(self, adj_matrix_list, model_cat_unique_levels, n_cont, T_in=10, T_out=10,
                 n_output=2, _continuous_outcome=False, 
                 lin_hidden=None, lin_hidden_temporal=None, domain_classifier=None):
        super(MLPModelTimeSeriesNumericalUDA, self).__init__()
        n_cat = len(model_cat_unique_levels)
        n_input = n_cat + n_cont
        # feature dim
        self.lin_input = nn.Linear(n_input, 32)
        if lin_hidden is not None:
            self.lin_hidden = lin_hidden
        else:
            self.lin_hidden = nn.ModuleList([nn.Linear(32, 128), nn.Linear(128, 512), 
                                             nn.Linear(512, 128), nn.Linear(128, 32)])
        if _continuous_outcome:
            self.lin_output = nn.Linear(32, 1) 
        else:
            self.lin_output = nn.Linear(32, n_output)
        
        # temporal dim
        if T_in > 1: # T_in > 1 and T_out >= 1
            self.lin_input_temporal = nn.Linear(T_in, 128)
            self.lin_output_temporal = nn.Linear(128, T_out)
            if lin_hidden_temporal is not None:
                self.lin_hidden_temporal = lin_hidden_temporal
            else:
                self.lin_hidden_temporal = None
        else: # T_in = 1 and T_out = 1
            self.lin_input_temporal = None
            self.lin_output_temporal = None

        # domain classifier
        if domain_classifier is not None:
            self.domain_classifier = domain_classifier
        else:
            if T_in > 1:
                self.domain_classifier = nn.Linear(128, 1)
            else:
                self.domain_classifier = nn.Linear(32, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 
    
    def forward(self, x_cat, x_cont, batched_nodes_indices=None, alpha=None):
        # x_cat: [batch_size, num_cat_vars, T_in]
        # x_cont: [batch_size, num_cont_vars, T_in]
        # batched_nodex_indices: [batch_size]

        x1, x2 = x_cat.permute(0, 2, 1), x_cont.permute(0, 2, 1) 
        x =  torch.cat([x1, x2], -1) # -> [batch_size, T_in, num_cat_vars + num_cont_vars]

        x = F.relu(self.lin_input(x))
        for layer in self.lin_hidden:
            x = F.relu(layer(x))
        class_output = self.lin_output(x) # -> [batch_size, T_in, 1]
        
        if self.lin_input_temporal is not None:
            x = F.relu(self.lin_input_temporal(class_output.permute(0, 2, 1))) # -> [batch_size, 1, 128]
            if self.lin_hidden_temporal is not None:
                for layer in self.lin_hidden_temporal:
                    x = F.relu(layer(x))
            
            class_output = self.lin_output_temporal(x) # -> [batch_size, 1, T_out]

            batch_size, feature_dim, temporal_dim = x.shape
            x = x.view(batch_size, feature_dim*temporal_dim)
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_x) # -> [batch_size, 1]
            return class_output, domain_output
        else:
            batch_size, feature_dim, temporal_dim = x.shape
            x = x.view(batch_size, feature_dim*temporal_dim) 
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_x) 
            return class_output.permute(0, 2, 1), domain_output # -> [batch_size, 1, 1], [batch_size, 1]



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

        self._init_weights()

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
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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


class GCNLayerTimeSeries(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayerTimeSeries, self).__init__()
        self.A = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x, A=None):
        '''
        spatial graph convolution operation
        :param A: (T, batch_size, batch_size), subgraph sampled from NetworkTMLE.adj_matrix
        :param x: (batch_size, T, F_in)
        :return: (batch_size, T, F_out)
        '''
        # self.A = A.to(x.device) # send subgraph to device
        self.A = A
        # print(self.A)
        if self.A.sum().item() == 0.: # no connection in sampled subgraph, no need for graph propagation
            return F.relu(self.Theta(x))
        else:
            return F.relu(self.Theta(torch.matmul(self.A, x.permute(1, 0, 2)))).permute(1, 0, 2)
            # [batch_size, T, F_in] -> [T, batch_size, F_in]
            # [batch_size, batch_size][T, batch_size, F_in] -> [T, batch_size, F_in] -> [batch_size, T, F_out]


class GCNModelTimeSeries(nn.Module):
    def __init__(self, adj_matrix_list, model_cat_unique_levels, n_cont, T=10,
                 n_output=2, _continuous_outcome=False):
        super(GCNModelTimeSeries, self).__init__()
        # self.adj_matrix_list = adj_matrix_list
        # len(adj_matrix_list) = T
        # for observed data: adj_matrix_list[i]: (num_nodes, num_nodes)
        # for pooled data: len(adj_matrix_list[i]) = pooled_samples, adj_matrix_list[i][j]: (num_nodes, num_nodes)

        # concat pooled adj_matrix_list[i] into a long array
        self.adj_matrix_list = []
        if isinstance(adj_matrix_list[0], list) and len(adj_matrix_list[0]) > 1:
            for i in range(len(adj_matrix_list)):
                tensor_list = [torch.from_numpy(adj_matrix.toarray()).float() for adj_matrix in adj_matrix_list[i]]
                tensor = torch.concat(tensor_list, dim=0)
                self.adj_matrix_list.append(tensor)
        else:
            self.adj_matrix_list = [torch.from_numpy(adj_matrix.toarray()).float() for adj_matrix in adj_matrix_list]
        
        self.adj_matrix = torch.stack(self.adj_matrix_list, dim=0) 
        # for observed data: [T, batch_size, batch_size]
        # for pooled data: [T, batch_size*pooled_samples, batch_size]
            
        self.embedding_layers, self.n_emb = self._get_embedding_layers(model_cat_unique_levels)
        self.n_cont = n_cont

        self.lin1 = nn.Linear(self.n_emb + n_cont, 16)
        self.gcn = GCNLayerTimeSeries(16, 32)
        if n_output == 2 or _continuous_outcome:
            self.lin3 = nn.Linear(32, 1) 
        else:
            self.lin3 = nn.Linear(32, n_output)
        self.bn1 = nn.BatchNorm1d(n_cont)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

        # time dimension feature extract 
        # self.ts_lin1 = nn.Linear(T, 16)
        # self.ts_lin2 = nn.Linear(16, T) 

        # ensure that the time dimension is consistent to pair up with the adj_matrix       
        self.ts_lin1 = nn.Linear(T, T)
        self.ts_lin2 = nn.Linear(T, T)

        self._init_weights()

    def _get_embedding_layers(self, model_cat_unique_levels):
        # Ref: https://jovian.ml/aakanksha-ns/shelter-outcome
        # decide embedding sizes
        embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _, n_categories in model_cat_unique_levels.items()]
        embedding_layers = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in embedding_layers) # length of all embeddings combined
        # n_cont = dataset.x_cont.shape[1] # number of continuous variables

        return embedding_layers, n_emb
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_adj_subset(self, adj_matrix, batched_nodes_indices):
        # bring adj_matrix to device
        adj_matrix_tensor = adj_matrix.to(batched_nodes_indices.device)
        T, samples, num_nodes = adj_matrix_tensor.shape

        # caliberate indices for pooled samples, indices restart every num_nodes samples 
        caliberated_batches_nodes_indices = batched_nodes_indices%adj_matrix_tensor.shape[0]

        if samples == num_nodes: # observed data
            # select by row and column, order does not matter
            adj_subset = torch.index_select(adj_matrix_tensor, dim=1, index=batched_nodes_indices)
            adj_subset = torch.index_select(adj_subset, dim=2, index=batched_nodes_indices)
        else: # pooled data
            # select by row and column, order does not matter
            adj_subset = torch.index_select(adj_matrix_tensor, dim=1, index=batched_nodes_indices)
            adj_subset = torch.index_select(adj_subset, dim=2, index=caliberated_batches_nodes_indices)
        return adj_subset # [T, batch_size, batch_size]

    def forward(self, x_cat, x_cont, batched_nodes_indices=None):
        # x_cat: [batch_size, num_cat_vars, T]
        # x_cont: [batch_size, num_cont_vars, T]
        # batched_nodex_indices: [batch_size]

        x_cat_new = x_cat.permute(0, 2, 1)
        # x_cat_new: [batch_size, T, num_cat_vars]

        if len(self.embedding_layers) > 0: # if there are categorical variables to be encoded
            x1 = [e(x_cat_new[:, :, i]) for i, e in enumerate(self.embedding_layers)]
            x1 = torch.cat(x1, -1) # [batch_size, T, n_emb]
            x1 = self.emb_drop(x1)

        if self.n_cont > 0: # if there are continuous variables to be encoded
            x2 = self.bn1(x_cont).permute(0, 2, 1) #[batch_size, T, n_cont]

        if len(self.embedding_layers) > 0 and self.n_cont > 0: # if there are both categorical and continuous variables to be encoded 
            x = torch.cat([x1, x2], -1) # [batch_size, T, n_emb + n_cont]
            # temporal perspective
            x = F.relu(self.ts_lin1(x.permute(0, 2, 1))).permute(0, 2, 1) 
            # [batch_size, T, n_emb + n_cont] -> [batch_size, n_emb + n_cont, T] 
            # -> [batch_size, n_emb + n_cont, 16] -> [batch_size, 16, n_emb + n_cont]

            # variable perspective
            x = F.relu(self.lin1(x)) # [batch_size, 16, n_emb + n_cont] -> [batch_size, 16, 16]
            x = self.drops(x)       
            x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1) 

            adj_subset = self._get_adj_subset(self.adj_matrix, batched_nodes_indices)
            x = self.gcn(x, adj_subset) # [batch_size, 16, 32]
            x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
            # [batch_size, 16, 32] -> [batch_size, 32, 16] -> [batch_size, 16, 32
            x = self.lin3(x)         # [batch_size, 16, 32] -> [batch_size, 16, 1]

            # temporal perspective
            x = self.ts_lin2(x.permute(0, 2, 1))
            # [batch_size, 16, 1] -> [batch_size, 1, 16] -> [batch_size, 1, T] 

        elif len(self.embedding_layers) > 0 and self.n_cont == 0: 
            # temporal perspective
            x = self.ts_lin1(x1.permute(0, 2, 1)).permute(0, 2, 1)
            # variable perspective
            x = F.relu(self.lin1(x))
            x = self.drops(x)       
            x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
            adj_subset = self._get_adj_subset(self.adj_matrix, batched_nodes_indices)
            x = self.gcn(x, adj_subset) # [batch_size, 16, 32]
            x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.lin3(x)
            # temporal perspective
            x = self.ts_lin2(x.permute(0, 2, 1))

        elif len(self.embedding_layers) == 0 and self.n_cont > 0:
            # temporal perspective
            x = self.ts_lin1(x2.permute(0, 2, 1)).permute(0, 2, 1)
            # variable perspective
            x = F.relu(self.lin1(x))
            x = self.drops(x)       
            x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
            adj_subset = self._get_adj_subset(self.adj_matrix, batched_nodes_indices)
            x = self.gcn(x, adj_subset) # [batch_size, 16, 32]
            x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.lin3(x)
            # temporal perspective
            x = self.ts_lin2(x.permute(0, 2, 1))
        else:
            raise ValueError('No variables to be encoded')
        return x


# class GCNLayerTimeSeries(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GCNLayerTimeSeries, self).__init__()
#         self.A = None
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.Theta = nn.Linear(in_channels, out_channels, bias=False)
    
#     def forward(self, x, A=None):
#         '''
#         spatial graph convolution operation
#         :param A: (batch_size, batch_size), subgraph sampled from NetworkTMLE.adj_matrix
#         :param x: (batch_size, T, F_in)
#         :return: (batch_size, T, F_out)
#         '''
#         # self.A = A.to(x.device) # send subgraph to device
#         self.A = A
#         # print(self.A)
#         if self.A.sum().item() == 0.: # no connection in sampled subgraph, no need for graph propagation
#             return F.relu(self.Theta(x))
#         else:
#             return F.relu(self.Theta(torch.matmul(self.A, x.permute(1, 0, 2)))).permute(1, 0, 2)
#             # [batch_size, T, F_in] -> [T, batch_size, F_in]
#             # [batch_size, batch_size][T, batch_size, F_in] -> [T, batch_size, F_in] -> [batch_size, T, F_out]


# class GCNModelTimeSeries(nn.Module):
#     def __init__(self, adj_matrix, model_cat_unique_levels, n_cont, T=10,
#                  n_output=2, _continuous_outcome=False):
#         super(GCNModelTimeSeries, self).__init__()
#         self.adj_matrix = adj_matrix

#         self.embedding_layers, self.n_emb = self._get_embedding_layers(model_cat_unique_levels)
#         self.n_cont = n_cont

#         self.lin1 = nn.Linear(self.n_emb + n_cont, 16)
#         self.gcn = GCNLayerTimeSeries(16, 32)
#         if n_output == 2 or _continuous_outcome:
#             self.lin3 = nn.Linear(32, 1) 
#         else:
#             self.lin3 = nn.Linear(32, n_output)
#         self.bn1 = nn.BatchNorm1d(n_cont)
#         self.bn2 = nn.BatchNorm1d(16)
#         self.bn3 = nn.BatchNorm1d(32)
#         self.emb_drop = nn.Dropout(0.6)
#         self.drops = nn.Dropout(0.3)

#         # time dimension feature extract 
#         self.ts_lin1 = nn.Linear(T, 16)
#         self.ts_lin2 = nn.Linear(16, T)


#     def _get_embedding_layers(self, model_cat_unique_levels):
#         # Ref: https://jovian.ml/aakanksha-ns/shelter-outcome
#         # decide embedding sizes
#         embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _, n_categories in model_cat_unique_levels.items()]
#         embedding_layers = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
#         n_emb = sum(e.embedding_dim for e in embedding_layers) # length of all embeddings combined
#         # n_cont = dataset.x_cont.shape[1] # number of continuous variables

#         return embedding_layers, n_emb

#     def _get_adj_subset(self, adj_matrix, batched_nodes_indices):
#         # conver SciPy sparse array to torch.tensor
#         adj_matrix_tensor = torch.from_numpy(adj_matrix.toarray()).float().to(batched_nodes_indices.device)
#         # calierate indices for pooled samples, indices restart every num_nodes samples 
#         caliberated_batches_nodes_indices = batched_nodes_indices%adj_matrix_tensor.shape[0]
#         # select by row and column, order does not matter
#         adj_subset = torch.index_select(adj_matrix_tensor, dim=0, index=caliberated_batches_nodes_indices)
#         adj_subset = torch.index_select(adj_subset, dim=1, index=caliberated_batches_nodes_indices)
#         return adj_subset

#     def forward(self, x_cat, x_cont, batched_nodes_indices=None):
#         # x_cat: [batch_size, num_cat_vars, T]
#         # x_cont: [batch_size, num_cont_vars, T]
#         # batched_nodex_indices: [batch_size]

#         x_cat_new = x_cat.permute(0, 2, 1)
#         # x_cat_new: [batch_size, T, num_cat_vars]

#         if len(self.embedding_layers) > 0: # if there are categorical variables to be encoded
#             x1 = [e(x_cat_new[:, :, i]) for i, e in enumerate(self.embedding_layers)]
#             x1 = torch.cat(x1, -1) # [batch_size, T, n_emb]
#             x1 = self.emb_drop(x1)

#         if self.n_cont > 0: # if there are continuous variables to be encoded
#             x2 = self.bn1(x_cont).permute(0, 2, 1) #[batch_size, T, n_cont]

#         if len(self.embedding_layers) > 0 and self.n_cont > 0: # if there are both categorical and continuous variables to be encoded 
#             x = torch.cat([x1, x2], -1) # [batch_size, T, n_emb + n_cont]
#             # temporal perspective
#             x = F.relu(self.ts_lin1(x.permute(0, 2, 1))).permute(0, 2, 1) 
#             # [batch_size, T, n_emb + n_cont] -> [batch_size, n_emb + n_cont, T] 
#             # -> [batch_size, n_emb + n_cont, 16] -> [batch_size, 16, n_emb + n_cont]

#             # variable perspective
#             x = F.relu(self.lin1(x)) # [batch_size, 16, n_emb + n_cont] -> [batch_size, 16, 16]
#             x = self.drops(x)       
#             x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1) 

#             adj_subset = self._get_adj_subset(self.adj_matrix, batched_nodes_indices)
#             x = self.gcn(x, adj_subset) # [batch_size, 16, 32]
#             x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
#             # [batch_size, 16, 32] -> [batch_size, 32, 16] -> [batch_size, 16, 32
#             x = self.lin3(x)         # [batch_size, 16, 32] -> [batch_size, 16, 1]

#             # temporal perspective
#             x = self.ts_lin2(x.permute(0, 2, 1))
#             # [batch_size, 16, 1] -> [batch_size, 1, 16] -> [batch_size, 1, T] 

#         elif len(self.embedding_layers) > 0 and self.n_cont == 0: 
#             # temporal perspective
#             x = self.ts_lin1(x1.permute(0, 2, 1)).permute(0, 2, 1)
#             # variable perspective
#             x = F.relu(self.lin1(x))
#             x = self.drops(x)       
#             x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
#             adj_subset = self._get_adj_subset(self.adj_matrix, batched_nodes_indices)
#             x = self.gcn(x, adj_subset) # [batch_size, 16, 32]
#             x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
#             x = self.lin3(x)
#             # temporal perspective
#             x = self.ts_lin2(x.permute(0, 2, 1))

#         elif len(self.embedding_layers) == 0 and self.n_cont > 0:
#             # temporal perspective
#             x = self.ts_lin1(x2.permute(0, 2, 1)).permute(0, 2, 1)
#             # variable perspective
#             x = F.relu(self.lin1(x))
#             x = self.drops(x)       
#             x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
#             adj_subset = self._get_adj_subset(self.adj_matrix, batched_nodes_indices)
#             x = self.gcn(x, adj_subset) # [batch_size, 16, 32]
#             x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
#             x = self.lin3(x)
#             # temporal perspective
#             x = self.ts_lin2(x.permute(0, 2, 1))
#         else:
#             raise ValueError('No variables to be encoded')
#         return x


######################## CNN model ########################
class CNNModelTimeSeries(nn.Module):
    def __init__(self, adj_matrix_list, model_cat_unique_levels, n_cont, T=10,
                 n_output=2, _continuous_outcome=False):
        super(CNNModelTimeSeries, self).__init__()
        self.embedding_layers, self.n_emb = self._get_embedding_layers(model_cat_unique_levels)
        self.n_cont = n_cont

        # conv layers
        self.conv1 = nn.Conv1d(in_channels=self.n_emb + self.n_cont, out_channels=16, 
                               kernel_size=5, padding='same')
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, 
                               kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, 
                               kernel_size=5, padding='same')  

        # bn layers
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)

        # dropout layers
        self.emb_drop = nn.Dropout(0.6)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)

        self._init_weights()

    def _get_embedding_layers(self, model_cat_unique_levels):
        # Ref: https://jovian.ml/aakanksha-ns/shelter-outcome
        # decide embedding sizes
        embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _, n_categories in model_cat_unique_levels.items()]
        embedding_layers = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in embedding_layers) # length of all embeddings combined
        # n_cont = dataset.x_cont.shape[1] # number of continuous variables

        return embedding_layers, n_emb
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x_cat, x_cont, batched_nodes_indices=None):
        # x_cat: [batch_size, num_cat_vars, T]
        # x_cont: [batch_size, num_cont_vars, T]
        # batched_nodex_indices: [batch_size]

        x_cat_new = x_cat.permute(0, 2, 1)
        # x_cat_new: [batch_size, T, num_cat_vars]

        if len(self.embedding_layers) > 0: # if there are categorical variables to be encoded
            x1 = [e(x_cat_new[:, :, 1]) for i, e in enumerate(self.embedding_layers)]
            x1 = torch.cat(x1, -1) # [batch_size, T, n_emb]
            x1 = self.emb_drop(x1)

        if self.n_cont > 0: # if there are continuous variables to be encoded
            x2 = self.bn1(x_cont).permute(0, 2, 1) # [batch_size, T, n_cont]
        
        if len(self.embedding_layers) > 0 and self.n_cont > 0: # if there are both categorical and continuous variables to be encoded 
            x = torch.cat([x1, x2], -1).permute(0, 2, 1) # [batch_size, T, n_emb + n_cont] -> [bathc_size, n_emb + n_cont, T]
            x = F.relu(self.bn2(self.conv1(x))) # [batch_size, 16, T] 
            x = self.drop1(x)
            x = F.relu(self.bn3(self.conv2(x))) # [batch_size, 32, T]
            x = self.drop2(x)
            x = self.conv3(x) # [batch_size, 1, T]

        elif len(self.embedding_layers) > 0 and self.n_cont == 0: 
            x = F.relu(self.bn2(self.conv1(x1.permute(0, 2, 1)))) # [batch_size, T, n_emb] -> [batch_size, n_emb, T] -> [batch_size, 16, T] 
            x = self.drop1(x)
            x = F.relu(self.bn3(self.conv2(x))) # [batch_size, 32, T]
            x = self.drop2(x)
            x = self.conv3(x) # [batch_size, 1, T]

        elif len(self.embedding_layers) == 0 and self.n_cont > 0:
            x = F.relu(self.bn2(self.conv1(x2.permute(0, 2, 1))))  # [batch_size, T, n_cont] -> [batch_size, n_cont, T] -> [batch_size, 16, T] 
            x = self.drop1(x)
            x = F.relu(self.bn3(self.conv2(x))) # [batch_size, 32, T]
            x = self.drop2(x)
            x = self.conv3(x) # [batch_size, 1, T]
        else:
            raise ValueError('No variables to be encoded')
    
        return x