import torch
import torch.nn as nn
import torch.nn.functional as F
from dl_layers import ReverseLayerF

class MLPModelTimeSeriesNumericalUDA(nn.Module):
    def __init__(self, adj_matrix_list, model_cat_unique_levels, n_cont, T_in=10, T_out=10,
                 n_output=2, _continuous_outcome=False, 
                 lin_hidden=None, lin_hidden_temporal=None, class_classifier=None, domain_classifier=None):
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
        lin_hidden_out_features = self.lin_hidden[-1].out_features
        
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
        
        # class classifier
        if class_classifier is not None:
            self.class_classifier = class_classifier
        else:
            if T_in > 1:
                self.class_classifier = nn.Linear(lin_hidden_out_features, n_output) 
            else:
                if _continuous_outcome:
                    self.class_classifier = nn.Linear(lin_hidden_out_features, 1) 
                else:
                    self.class_classifier = nn.Linear(lin_hidden_out_features, n_output) 
        
        # domain classifier
        if domain_classifier is not None:
            self.domain_classifier = domain_classifier
        else:
            if T_in > 1:
                self.domain_classifier = nn.Linear(T_out*lin_hidden_out_features, 2) 
            else:
                self.domain_classifier = nn.Linear(T_in*lin_hidden_out_features, 2) 

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

        # feature dim
        x = F.relu(self.lin_input(x))
        for layer in self.lin_hidden:
            x = F.relu(layer(x)) # -> [batch_size, T_in, lin_hidden[-1]]
        
        if self.lin_input_temporal is not None: # temporal dim
            x = F.relu(self.lin_input_temporal(x.permute(0, 2, 1))) # -> [batch_size, lin_hidden[-1], 128]
            if self.lin_hidden_temporal is not None:
                for layer in self.lin_hidden_temporal:
                    x = F.relu(layer(x))
            x = self.lin_output_temporal(x).permute(0, 2, 1) # -> [batch_size, T_out, lin_hidden[-1]]
            # class classifier
            class_output = self.class_classifier(x).permute(0, 2, 1) # -> [batch_size, n_output, T_out]
            # domain classifier
            batch_size, temporal_dim, feature_dim = x.shape
            # print(f'batch_size: {batch_size}, temporal_dim: {temporal_dim}, feature_dim: {feature_dim}')
            x = x.reshape(batch_size, temporal_dim*feature_dim)
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_x) # -> [batch_size, 2]
            return class_output, domain_output
        else:
            # class classifier
            class_output = self.class_classifier(x).permute(0, 2, 1) # -> [batch_size, n_output, T_in]
            # domain classifier
            batch_size, temporal_dim, feature_dim = x.shape
            # print(f'batch_size: {batch_size}, temporal_dim: {temporal_dim}, feature_dim: {feature_dim}')
            x = x.reshape(batch_size, temporal_dim*feature_dim) 
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_x) # -> [batch_size, 2] 
            return class_output, domain_output 
