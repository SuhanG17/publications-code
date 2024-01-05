import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
import pandas as pd

# TODO
# 1. add variables: from model string or from raw data
#   - how to define categorical variables in bins
#   - how to give raw variables
#   - when to use GNN?
#   - how do we know if model is better? better bias?    
#   - how to intepret 95% CI?
# 2. rewrite ml_part in tmle.py 
# 3. cross validation: https://machinelearningmastery.com/k-fold-cross-validation/
# 4. make the abstract ml fit to the mossspider version or vice versa


######################## define abstract ml_funtion ########################
class AbstractML:
    def __init__(self, split_ratio, batch_size, shuffle, 
                 epochs, print_every, device='cpu', save_path='./'):
        self.epochs = epochs
        self.best_model = None
        self.best_loss = np.inf
        self.save_path = save_path

        self.split_ratio, self.batch_size, self.shuffle = split_ratio, batch_size, shuffle
    
        self.print_every = print_every
        self.device = device

        self.model = self._build_model().to(self.device)
        self.optimizer = self._optimizer()
        self.criterion = self._loss_fn()

    def fit(self, df, model_string, target):
        # target is exposure for nuisance models, outcome for outcome model
        self.train_loader, self.valid_loader, self.test_loader = self._data_preprocess(df, model_string, target, 
                                                                                       fit=True,
                                                                                       split_ratio=self.split_ratio, 
                                                                                       batch_size=self.batch_size, 
                                                                                       shuffle=self.shuffle)
        for epoch in range(self.epochs):
            print(f'============================= Epoch {epoch + 1}: Training =============================')
            loss_train, metrics_train = self.train_epoch(epoch)
            print(f'============================= Epoch {epoch + 1}: Validation =============================')
            loss_valid, metrics_valid = self.valid_epoch(epoch)
            print(f'============================= Epoch {epoch + 1}: Testing =============================')
            loss_test, metrics_test = self.test_epoch(epoch, return_pred=False)

            # update best loss
            if loss_valid < self.best_loss:
                self._save_model()
                self.best_loss = loss_valid
                self.best_model = self.model
            
        return self.best_model

    def predict(self, df, model_string, target):
        _, _, self.test_loader = self._data_preprocess(df, model_string, target, 
                                                       fit=False, 
                                                       split_ratio=self.split_ratio, 
                                                       batch_size=self.batch_size, 
                                                       shuffle=self.shuffle)
        self._load_model()
        pred = self.test_epoch(epoch=0, return_pred=True) # pred should probabilities, one for binary
        return pred
    
    # def predict_proba(self, x):
    #     return np.zeros(x.shape[0])

    def train_epoch(self, epoch):
        self.model.train() # turn on train-mode

        # record loss and metrics for every print_every mini-batches
        running_loss = 0.0 
        running_metrics = 0.0
        # record loss and metrics for the whole epoch
        cumu_loss = 0.0 
        cumu_metrics = 0.0

        for i, (x_cat, x_cont, y) in enumerate(self.train_loader):
            # send to device
            x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device) 

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(x_cat, x_cont)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            # metrics
            metrics = self._metrics(outputs, y)

            # print statistics
            running_loss += loss.item()
            cumu_loss += loss.item()
            
            running_metrics += metrics
            cumu_metrics += metrics

            if i % self.print_every == self.print_every - 1:    # print every mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] | loss: {running_loss / self.print_every:.3f} | acc: {running_metrics / self.print_every:.3f}')
                running_loss = 0.0
                running_metrics = 0.0              

                # for metric_name, metric_value in running_metrics.items():
                #     print(f'[{epoch + 1}, {i + 1:5d}] {metric_name}: {metric_value / self.print_every:.3f}')
                # running_metrics = {}
        
        return cumu_loss / len(self.train_loader), cumu_metrics / len(self.train_loader)

    def valid_epoch(self, epoch): 
        self.model.eval() # turn on eval mode

        # record loss and metrics for the whole epoch
        cumu_loss = 0.0 
        cumu_metrics = 0.0

        with torch.no_grad():
            for i, (x_cat, x_cont, y) in enumerate(self.valid_loader):
                # send to device
                x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device) 
                outputs = self.model(x_cat, x_cont)
                loss = self.criterion(outputs, y)
                metrics = self._metrics(outputs, y)

                # print statistics
                cumu_loss += loss.item()
                cumu_metrics += metrics

            print(f'[{epoch + 1}, {i + 1:5d}] | loss: {cumu_loss / len(self.valid_loader):.3f} | acc: {cumu_metrics / len(self.valid_loader):.3f}')
            # for metric_name, metric_value in cumu_metrics.items():
            #     print(f'[{epoch + 1}, {i + 1:5d}] {metric_name}: {metric_value / len(self.valid_loader):.3f}')

        return cumu_loss / len(self.valid_loader), cumu_metrics / len(self.valid_loader)


    def test_epoch(self, epoch, return_pred=False):
        self.model.eval() # turn on eval mode

        if return_pred:
            pred_list = []
        
        # record loss and metrics for the whole epoch
        cumu_loss = 0.0 
        cumu_metrics = 0.0

        with torch.no_grad():
            for i, (x_cat, x_cont, y) in enumerate(self.test_loader):
                # send to device
                x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device) 
                output = self.model(x_cat, x_cont)
                if return_pred:
                    pred_list.append(torch.sigmoid(output).detach().to('cpu').numpy())

                loss = self.criterion(output, y)
                metrics = self._metrics(output, y)

                # print statistics
                cumu_loss += loss.item()
                cumu_metrics += metrics

            if not return_pred: # real label not available for predict()
                print(f'[{epoch + 1}, {i + 1:5d}] | loss: {cumu_loss / len(self.test_loader):.3f} | acc: {cumu_metrics / len(self.test_loader):.3f}')
                # for metric_name, metric_value in cumu_metrics.items():
                #     print(f'[{epoch + 1}, {i + 1:5d}] {metric_name}: {metric_value / len(self.test_loader):.3f}')

        if return_pred:
            return pred_list
        else:
            return cumu_loss / len(self.test_loader), cumu_metrics / len(self.test_loader)

    def _build_model(self):
        pass 

    def _data_preprocess(self, df, model_string, target=None, fit=True):
        dset = DfDataset(df, model_string, target=target, fit=fit)
        train_loader, valid_loader, test_loader = get_dataloaders(dset,
                                                                  split_ratio=self.split_ratio, 
                                                                  batch_size=self.batch_size,
                                                                  shuffle=self.shuffle)           
        return train_loader, valid_loader, test_loader


    def _optimizer(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # return optim.Adam(self.model.parameters(), lr=0.001)

    def _loss_fn(self):
        return nn.BCEWithLogitsLoss() # no need for sigmoid, require 2 output for binary classfication
        # return nn.CrossEntropyLoss() # no need for softmax, require 2 output for binary classification

    def _metrics(self, outputs, labels):
        pred = torch.sigmoid(outputs) # get binary probability
        pred_binary = torch.round(pred) # get binary prediction
        return (pred_binary == labels).sum().item()/labels.shape[0] # num samples correctly classified / num_samples
        
        # # the class with the highest energy is what we choose as prediction, if output 2 categories for binary classificaiton
        # _, predicted = torch.max(outputs.data, 1)
        # return (predicted == labels).sum().item()

    def _save_model(self):
        torch.save(self.model.state_dict(), self.save_path)

    def _load_model(self):
        self.model.load_state_dict(torch.load(self.save_path))

######################## define dataset ########################
class DfDataset(Dataset):
    def __init__(self, df, model, target=None, fit=True):
        ''' Retrieve train,label and pred data from Dataframe directly
        Args:  
            df: pd.DataFrame, data, i.e., df_restricted
            model: str, model formula, i.e., _gi_model
            target: str, target variable, i.e., exposure/outcome
            fit: bool, whether the dataset is for fitting or prediction

        if fit is set to true, df should be data_to_fit; else, df should be data_to_predict
        '''
        self.df = df
        self.model = model
        self.target = target
        self.fit = fit

        self.x_cat, self.x_cont, self.cat_unique_levels = self._split_cat_cont() 
        if self.fit:
            self.y = self._get_labels()
        else:
            self.y = np.empty((self.x_cat.shape[0], 1))
            self.y.fill(-1) # create dummy target for pdata
    
    def _split_cat_cont(self):
        # get variables from model string
        vars = self.model.split(' + ')

        cat_vars = []
        cont_vars = []
        cat_unique_levels = {}
        for var in vars:
            if var in self.df.columns:
                if self.df[var].dtype == 'int64':
                    
                    cat_vars.append(var)
                    if len(pd.unique(self.df[var])) == 2:
                        cat_unique_levels[var] = len(pd.unique(self.df[var])) # record number of levels
                    else:
                        cat_unique_levels[var] = self.df['A'].max() + 1 # record number of levels for 'A_30', temporary strategy
                else:
                    cont_vars.append(var)
        return self.df[cat_vars].to_numpy(), self.df[cont_vars].to_numpy(), cat_unique_levels
    
    def _get_labels(self):
        return np.asarray(self.df[self.target])[:, np.newaxis] #[num_samples, ] -> [num_samples, 1] 
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.x_cat[idx]).int(), torch.from_numpy(self.x_cont[idx]).float(), torch.from_numpy(self.y[idx]).float()
        # shape: [num_cat_vars], [num_cont_vars], [1]

    def __len__(self):
        return self.y.shape[0]

# class CatContDataset(Dataset):
#     def __init__(self, xdata, ydata=None, cat_vars_index=None, cont_vars_index=None):
#         if ydata is not None:
#             self.x_cat = xdata[:, cat_vars_index]
#             self.x_cont = xdata[:, cont_vars_index]
#             self.y = ydata[:, np.newaxis] #[num_samples, ] -> [num_samples, 1]
#         else:
#             self.x_cat = xdata[:, cat_vars_index]
#             self.x_cont = xdata[:, cont_vars_index]
#             self.y = np.empty(xdata.shape[0]).fill(-1) # create dummy target for pdata
    
#     def __getitem__(self, idx):
#         # return torch.from_numpy(np.asarray(self.y[idx]))
#         return torch.from_numpy(self.x_cat[idx]), torch.from_numpy(self.x_cont[idx]), torch.from_numpy(self.y[idx])

#     def __len__(self):
#         return self.y.shape[0]
    
######################## split dataset and define loader ########################
def get_dataloaders(dataset, split_ratio=[0.7, 0.1, 0.2], batch_size=16, shuffle=True):
    torch.manual_seed(17) # random split with reproducibility

    train_size = int(split_ratio[0] * len(dataset))
    test_size = int(split_ratio[-1] * len(dataset))
    valid_size = len(dataset) - train_size - test_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

######################## define models ########################
class SimpleModel(nn.Module):
    def __init__(self, cat_unique_levels, n_cont):
        super().__init__()
        self.embedding_layers, self.n_emb, self.n_cont = self._get_embedding_layers(cat_unique_levels, n_cont)
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 16)
        self.lin2 = nn.Linear(16, 32)
        self.lin3 = nn.Linear(32, 1) # use BCEloss, so output 1
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

    def _get_embedding_layers(self, cat_unique_levels, n_cont):
        # Ref: https://jovian.ml/aakanksha-ns/shelter-outcome
        # decide embedding sizes
        embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _, n_categories in cat_unique_levels.items()]
        embedding_layers = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in embedding_layers) # length of all embeddings combined
        # n_cont = dataset.x_cont.shape[1] # number of continuous variables

        return embedding_layers, n_emb, n_cont
    
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

######################## ml training ########################
class MLP(AbstractML):
    def __init__(self, df, model_string, exposure, epochs, print_every, device='cpu', save_path='./'):
        super().__init__(df, model_string, exposure, epochs, print_every, device, save_path)

    def _build_model(self):
        return SimpleModel(cat_unique_levels, n_cont) #TODO
    
# mlp_learner = MLP(df_restricted, _gi_model, exposure, epochs=10, print_every=50, device='cpu', save_path='./tmp.pth')
# mlp_learner.fit(split_ratio=[0.6, 0.2, 0.2], batch_size=2, shuffle=True) 
# pred = mlp_learner.predict(split_ratio=[0.6, 0.2, 0.2], batch_size=2, shuffle=False)
