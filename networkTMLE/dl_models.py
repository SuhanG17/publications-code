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
