import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold # kfold cross validation

# TODO
# 1. add variables: from model string or from raw data
#   - how to define categorical variables in bins: DONE
#   - how to give raw variables
#   - when to use GNN?
#   - how do we know if model is better? better bias?    
#   - how to intepret 95% CI?
# 2. rewrite ml_part in tmle.py 
# 3. cross validation: https://machinelearningmastery.com/k-fold-cross-validation/
# 4. make the abstract ml fit to the mossspider version or vice versa


######################## define abstract ml_funtion ########################
class AbstractML:
    def __init__(self, split_ratio, batch_size, shuffle, n_splits, predict_all,
                 epochs, print_every, device='cpu', save_path='./'):
        self.epochs = epochs
        self.best_model = None
        self.best_loss = np.inf
        self.save_path = save_path

        self.split_ratio, self.batch_size, self.shuffle, self.n_splits, self.predict_all = split_ratio, batch_size, shuffle, n_splits, predict_all
    
        self.print_every = print_every
        self.device = device

        # self.model = self._build_model().to(self.device) # instantiation requires df, model_string and target, move insides fit() and predict()
        # self.optimizer = self._optimizer()
        self.criterion = self._loss_fn()

    def fit(self, df, target, model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}, custom_path=None):
        # instantiate model
        self.model = self._build_model(model_cat_vars, model_cont_vars, model_cat_unique_levels).to(self.device)
        self.optimizer = self._optimizer()

        # target is exposure for nuisance models, outcome for outcome model
        fold_record = {'train_loss': [], 'val_loss': [],'train_acc':[],'val_acc':[]}

        if self.n_splits > 1: # Kfold cross validation is used
            splits, dset = self._data_preprocess(df, target, fit=True,
                                                 model_cat_vars=model_cat_vars, 
                                                 model_cont_vars=model_cont_vars, 
                                                 model_cat_unique_levels=model_cat_unique_levels)
            for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dset)))):
                print('Fold {}'.format(fold + 1))
                self.train_loader, self.valid_loader = get_kfold_dataloaders(dset, train_idx, val_idx, 
                                                                             batch_size=self.batch_size,
                                                                             shuffle=self.shuffle)
                for epoch in range(self.epochs):
                    print(f'============================= Epoch {epoch + 1}: Training =============================')
                    loss_train, metrics_train = self.train_epoch(epoch)
                    print(f'============================= Epoch {epoch + 1}: Validation =============================')
                    loss_valid, metrics_valid = self.valid_epoch(epoch)

                    fold_record['train_loss'].append(loss_train)
                    fold_record['val_loss'].append(loss_valid)
                    fold_record['train_acc'].append(metrics_train)
                    fold_record['val_acc'].append(metrics_valid)

                    # update best loss
                    if loss_valid < self.best_loss:
                        self._save_model(custom_path)
                        self.best_loss = loss_valid
                        self.best_model = self.model
                        print('Best model updated')
            
            avg_train_loss = np.mean(fold_record['train_loss'])
            avg_val_loss = np.mean(fold_record['val_loss'])
            avg_train_acc = np.mean(fold_record['train_acc'])
            avg_val_acc = np.mean(fold_record['val_acc'])

            print(f'Performance of {self.n_splits} fold cross validation')
            print(f'Average Training Loss: {avg_train_loss:.4f} \t Average Val Loss: {avg_val_loss:.4f} \t Average Training Acc: {avg_train_acc:.3f} \t Average Test Acc: {avg_val_acc:.3f}')  
        else:
            self.train_loader, self.valid_loader, self.test_loader = self._data_preprocess(df, target, fit=True,
                                                                                           model_cat_vars=model_cat_vars, 
                                                                                           model_cont_vars=model_cont_vars, 
                                                                                           model_cat_unique_levels=model_cat_unique_levels)
            for epoch in range(self.epochs):
                print(f'============================= Epoch {epoch + 1}: Training =============================')
                loss_train, metrics_train = self.train_epoch(epoch)
                print(f'============================= Epoch {epoch + 1}: Validation =============================')
                loss_valid, metrics_valid = self.valid_epoch(epoch)
                print(f'============================= Epoch {epoch + 1}: Testing =============================')
                loss_test, metrics_test = self.test_epoch(epoch, return_pred=False)

                # update best loss
                if loss_valid < self.best_loss:
                    self._save_model(custom_path)
                    self.best_loss = loss_valid
                    self.best_model = self.model
                    print('Best model updated')
            
        return self.save_path

    def predict(self, df, target, model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}, custom_path=None):
        # instantiate model
        self.model = self._build_model(model_cat_vars, model_cont_vars, model_cat_unique_levels).to(self.device)
        self._load_model(custom_path)

        if self.predict_all:
            dset = DfDataset(df, target=target, fit=False, 
                             model_cat_vars=model_cat_vars, 
                             model_cont_vars=model_cont_vars, 
                             model_cat_unique_levels=model_cat_unique_levels)
            self.test_loader = get_predict_loader(dset, self.batch_size)
        else:
            _, _, self.test_loader = self._data_preprocess(df, target, fit=False,
                                                           model_cat_vars=model_cat_vars, 
                                                           model_cont_vars=model_cont_vars, 
                                                           model_cat_unique_levels=model_cat_unique_levels)
        
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

    def _data_preprocess(self, df, target=None, fit=True,
                         model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}):
        dset = DfDataset(df, target=target, fit=fit,
                         model_cat_vars=model_cat_vars, 
                         model_cont_vars=model_cont_vars, 
                         model_cat_unique_levels=model_cat_unique_levels)

        if self.n_splits > 1: # Kfold cross validation is used
            return get_kfold_split(n_splits=self.n_splits, shuffle=self.shuffle), dset
        else:
            train_loader, valid_loader, test_loader = get_dataloaders(dset,
                                                                    split_ratio=self.split_ratio, 
                                                                    batch_size=self.batch_size,
                                                                    shuffle=self.shuffle)           
            return train_loader, valid_loader, test_loader


    def _optimizer(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # return optim.Adam(self.model.parameters(), lr=0.001)

    def _loss_fn(self):
        return nn.BCEWithLogitsLoss() # no need for sigmoid, require 1 output for binary classfication
        # return nn.CrossEntropyLoss() # no need for softmax, require 2 output for binary classification

    def _metrics(self, outputs, labels):
        pred = torch.sigmoid(outputs) # get binary probability
        pred_binary = torch.round(pred) # get binary prediction
        return (pred_binary == labels).sum().item()/labels.shape[0] # num samples correctly classified / num_samples
        
        # # the class with the highest energy is what we choose as prediction, if output 2 categories for binary classificaiton
        # _, predicted = torch.max(outputs.data, 1)
        # return (predicted == labels).sum().item()

    def _save_model(self, custom_path=None):
        if custom_path is None:
            custom_path = self.save_path
        torch.save(self.model.state_dict(), custom_path)

    def _load_model(self, custom_path=None):
        if custom_path is None:
            custom_path = self.save_path
        self.model.load_state_dict(torch.load(custom_path))


# class AbstractML:
#     def __init__(self, split_ratio, batch_size, shuffle, n_splits, predict_all,
#                  epochs, print_every, device='cpu', save_path='./'):
#         self.epochs = epochs
#         self.best_model = None
#         self.best_loss = np.inf
#         self.save_path = save_path

#         self.split_ratio, self.batch_size, self.shuffle, self.n_splits, self.predict_all = split_ratio, batch_size, shuffle, n_splits, predict_all
    
#         self.print_every = print_every
#         self.device = device

#         # self.model = self._build_model().to(self.device) # instantiation requires df, model_string and target, move insides fit() and predict()
#         # self.optimizer = self._optimizer()
#         self.criterion = self._loss_fn()

#     def fit(self, df, model_string, target, cat_vars=[], cont_vars=[], cat_unique_levels={}):
#         # instantiate model
#         self.model = self._build_model(df, model_string, target, cat_vars, cont_vars, cat_unique_levels).to(self.device)
#         self.optimizer = self._optimizer()

#         # target is exposure for nuisance models, outcome for outcome model
#         fold_record = {'train_loss': [], 'val_loss': [],'train_acc':[],'val_acc':[]}

#         if self.n_splits > 1: # Kfold cross validation is used
#             splits, dset = self._data_preprocess(df, model_string, target, fit=True,
#                                                  cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)
#             for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dset)))):
#                 print('Fold {}'.format(fold + 1))
#                 self.train_loader, self.valid_loader = get_kfold_dataloaders(dset, train_idx, val_idx, 
#                                                                              batch_size=self.batch_size,
#                                                                              shuffle=self.shuffle)
#                 for epoch in range(self.epochs):
#                     print(f'============================= Epoch {epoch + 1}: Training =============================')
#                     loss_train, metrics_train = self.train_epoch(epoch)
#                     print(f'============================= Epoch {epoch + 1}: Validation =============================')
#                     loss_valid, metrics_valid = self.valid_epoch(epoch)

#                     fold_record['train_loss'].append(loss_train)
#                     fold_record['val_loss'].append(loss_valid)
#                     fold_record['train_acc'].append(metrics_train)
#                     fold_record['val_acc'].append(metrics_valid)

#                     # update best loss
#                     if loss_valid < self.best_loss:
#                         self._save_model()
#                         self.best_loss = loss_valid
#                         self.best_model = self.model
#                         print('Best model updated')
            
#             avg_train_loss = np.mean(fold_record['train_loss'])
#             avg_val_loss = np.mean(fold_record['val_loss'])
#             avg_train_acc = np.mean(fold_record['train_acc'])
#             avg_val_acc = np.mean(fold_record['val_acc'])

#             print(f'Performance of {self.n_splits} fold cross validation')
#             print(f'Average Training Loss: {avg_train_loss:.4f} \t Average Val Loss: {avg_val_loss:.4f} \t Average Training Acc: {avg_train_acc:.3f} \t Average Test Acc: {avg_val_acc:.3f}')  
#         else:
#             self.train_loader, self.valid_loader, self.test_loader = self._data_preprocess(df, model_string, target, fit=True,
#                                                                                            cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)
#             for epoch in range(self.epochs):
#                 print(f'============================= Epoch {epoch + 1}: Training =============================')
#                 loss_train, metrics_train = self.train_epoch(epoch)
#                 print(f'============================= Epoch {epoch + 1}: Validation =============================')
#                 loss_valid, metrics_valid = self.valid_epoch(epoch)
#                 print(f'============================= Epoch {epoch + 1}: Testing =============================')
#                 loss_test, metrics_test = self.test_epoch(epoch, return_pred=False)

#                 # update best loss
#                 if loss_valid < self.best_loss:
#                     self._save_model()
#                     self.best_loss = loss_valid
#                     self.best_model = self.model
#                     print('Best model updated')
            
#         return self.best_model

#     def predict(self, df, model_string, target, cat_vars=[], cont_vars=[], cat_unique_levels={}):
#         # instantiate model
#         self.model = self._build_model(df, model_string, target, cat_vars, cont_vars, cat_unique_levels).to(self.device)

#         if self.predict_all:
#             dset = DfDataset(df, model_string, target=target, fit=False, 
#                              cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)
#             self.test_loader = get_predict_loader(dset, self.batch_size)
#         else:
#             _, _, self.test_loader = self._data_preprocess(df, model_string, target, fit=False,
#                                                            cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)
#         self._load_model()
#         pred = self.test_epoch(epoch=0, return_pred=True) # pred should probabilities, one for binary
#         return pred
    
#     # def predict_proba(self, x):
#     #     return np.zeros(x.shape[0])

#     def train_epoch(self, epoch):
#         self.model.train() # turn on train-mode

#         # record loss and metrics for every print_every mini-batches
#         running_loss = 0.0 
#         running_metrics = 0.0
#         # record loss and metrics for the whole epoch
#         cumu_loss = 0.0 
#         cumu_metrics = 0.0

#         for i, (x_cat, x_cont, y) in enumerate(self.train_loader):
#             # send to device
#             x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device) 

#             # zero the parameter gradients
#             self.optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = self.model(x_cat, x_cont)
#             loss = self.criterion(outputs, y)
#             loss.backward()
#             self.optimizer.step()

#             # metrics
#             metrics = self._metrics(outputs, y)

#             # print statistics
#             running_loss += loss.item()
#             cumu_loss += loss.item()
            
#             running_metrics += metrics
#             cumu_metrics += metrics

#             if i % self.print_every == self.print_every - 1:    # print every mini-batches
#                 print(f'[{epoch + 1}, {i + 1:5d}] | loss: {running_loss / self.print_every:.3f} | acc: {running_metrics / self.print_every:.3f}')
#                 running_loss = 0.0
#                 running_metrics = 0.0              

#                 # for metric_name, metric_value in running_metrics.items():
#                 #     print(f'[{epoch + 1}, {i + 1:5d}] {metric_name}: {metric_value / self.print_every:.3f}')
#                 # running_metrics = {}
        
#         return cumu_loss / len(self.train_loader), cumu_metrics / len(self.train_loader)

#     def valid_epoch(self, epoch): 
#         self.model.eval() # turn on eval mode

#         # record loss and metrics for the whole epoch
#         cumu_loss = 0.0 
#         cumu_metrics = 0.0

#         with torch.no_grad():
#             for i, (x_cat, x_cont, y) in enumerate(self.valid_loader):
#                 # send to device
#                 x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device) 
#                 outputs = self.model(x_cat, x_cont)
#                 loss = self.criterion(outputs, y)
#                 metrics = self._metrics(outputs, y)

#                 # print statistics
#                 cumu_loss += loss.item()
#                 cumu_metrics += metrics

#             print(f'[{epoch + 1}, {i + 1:5d}] | loss: {cumu_loss / len(self.valid_loader):.3f} | acc: {cumu_metrics / len(self.valid_loader):.3f}')
#             # for metric_name, metric_value in cumu_metrics.items():
#             #     print(f'[{epoch + 1}, {i + 1:5d}] {metric_name}: {metric_value / len(self.valid_loader):.3f}')

#         return cumu_loss / len(self.valid_loader), cumu_metrics / len(self.valid_loader)


#     def test_epoch(self, epoch, return_pred=False):
#         self.model.eval() # turn on eval mode

#         if return_pred:
#             pred_list = []
        
#         # record loss and metrics for the whole epoch
#         cumu_loss = 0.0 
#         cumu_metrics = 0.0

#         with torch.no_grad():
#             for i, (x_cat, x_cont, y) in enumerate(self.test_loader):
#                 # send to device
#                 x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device) 
#                 output = self.model(x_cat, x_cont)
#                 if return_pred:
#                     pred_list.append(torch.sigmoid(output).detach().to('cpu').numpy())

#                 loss = self.criterion(output, y)
#                 metrics = self._metrics(output, y)

#                 # print statistics
#                 cumu_loss += loss.item()
#                 cumu_metrics += metrics

#             if not return_pred: # real label not available for predict()
#                 print(f'[{epoch + 1}, {i + 1:5d}] | loss: {cumu_loss / len(self.test_loader):.3f} | acc: {cumu_metrics / len(self.test_loader):.3f}')
#                 # for metric_name, metric_value in cumu_metrics.items():
#                 #     print(f'[{epoch + 1}, {i + 1:5d}] {metric_name}: {metric_value / len(self.test_loader):.3f}')

#         if return_pred:
#             return pred_list
#         else:
#             return cumu_loss / len(self.test_loader), cumu_metrics / len(self.test_loader)

#     def _build_model(self):
#         pass 

#     def _data_preprocess(self, df, model_string, target=None, fit=True,
#                          cat_vars=[], cont_vars=[], cat_unique_levels={}):
#         dset = DfDataset(df, model_string, target=target, fit=fit,
#                          cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)

#         if self.n_splits > 1: # Kfold cross validation is used
#             return get_kfold_split(n_splits=self.n_splits, shuffle=self.shuffle), dset
#         else:
#             train_loader, valid_loader, test_loader = get_dataloaders(dset,
#                                                                     split_ratio=self.split_ratio, 
#                                                                     batch_size=self.batch_size,
#                                                                     shuffle=self.shuffle)           
#             return train_loader, valid_loader, test_loader


#     def _optimizer(self):
#         return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
#         # return optim.Adam(self.model.parameters(), lr=0.001)

#     def _loss_fn(self):
#         return nn.BCEWithLogitsLoss() # no need for sigmoid, require 1 output for binary classfication
#         # return nn.CrossEntropyLoss() # no need for softmax, require 2 output for binary classification

#     def _metrics(self, outputs, labels):
#         pred = torch.sigmoid(outputs) # get binary probability
#         pred_binary = torch.round(pred) # get binary prediction
#         return (pred_binary == labels).sum().item()/labels.shape[0] # num samples correctly classified / num_samples
        
#         # # the class with the highest energy is what we choose as prediction, if output 2 categories for binary classificaiton
#         # _, predicted = torch.max(outputs.data, 1)
#         # return (predicted == labels).sum().item()

#     def _save_model(self):
#         torch.save(self.model.state_dict(), self.save_path)

#     def _load_model(self):
#         self.model.load_state_dict(torch.load(self.save_path))

######################## define dataset ########################
class DfDataset(Dataset):
    def __init__(self, patsy_matrix_dataframe, target=None, fit=True, 
                 model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}):
        ''' Retrieve train,label and pred data from Dataframe directly
        Args:  
            patsy_matrix_dataframe: pd.DataFrame, data, i.e., dataframe created from patsy.dmatrix()
            model: str, model formula, i.e., _gi_model
            target: str, target variable, i.e., exposure/outcome
            fit: bool, whether the dataset is for fitting or prediction
            model_cat_vars: list, categorical variables in patsy_matrix_dataframe, subset of cat_vars
            model_cont_vars: list, continuous variables in patsy_matrix_dataframe, subset of cont_vars
            model_cat_unique_levels: dict, number of unique levels for each categorical variable of patsy_matrix_dataframe, subset of cat_unique_levels

        if fit is set to true, df should be data_to_fit; else, df should be data_to_predict
        '''
        self.df = patsy_matrix_dataframe 
        self.target = target
        self.fit = fit
        self.model_cat_vars, self.model_cont_vars, self.model_cat_unique_levels = model_cat_vars, model_cont_vars, model_cat_unique_levels

        self.x_cat, self.x_cont = self._split_cat_cont() 
        if self.fit:
            self.y = self._get_labels()
        else:
            self.y = np.empty((self.x_cat.shape[0], 1))
            self.y.fill(-1) # create dummy target for pdata
    
    def _split_cat_cont(self):
        return self.df[self.model_cat_vars].to_numpy(), self.df[self.model_cont_vars].to_numpy()
    
    def _get_labels(self):
        return np.asarray(self.df[self.target])[:, np.newaxis] #[num_samples, ] -> [num_samples, 1] 
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.x_cat[idx]).int(), torch.from_numpy(self.x_cont[idx]).float(), torch.from_numpy(self.y[idx]).float()
        # shape: [num_cat_vars], [num_cont_vars], [1]

    def __len__(self):
        return self.y.shape[0]


# class DfDataset(Dataset):
#     def __init__(self, df, model, target=None, fit=True, cat_vars=[], cont_vars=[], cat_unique_levels={}):
#         ''' Retrieve train,label and pred data from Dataframe directly
#         Args:  
#             df: pd.DataFrame, data, i.e., df_restricted
#             model: str, model formula, i.e., _gi_model
#             target: str, target variable, i.e., exposure/outcome
#             fit: bool, whether the dataset is for fitting or prediction
#             cat_vars: list, categorical variables in df
#             cont_vars: list, continuous variables in df
#             cat_unique_levels: dict, number of unique levels for each categorical variable of df

#         if fit is set to true, df should be data_to_fit; else, df should be data_to_predict
#         '''
#         self.df = df
#         self.model = model
#         self.target = target
#         self.fit = fit
#         self.cat_vars, self.cont_vars, self.cat_unique_levels = cat_vars, cont_vars, cat_unique_levels

#         self.x_cat, self.x_cont, self.model_cat_unique_levels = self._split_cat_cont() 
#         if self.fit:
#             self.y = self._get_labels()
#         else:
#             self.y = np.empty((self.x_cat.shape[0], 1))
#             self.y.fill(-1) # create dummy target for pdata
    
#     def _split_cat_cont(self):
#         # get variables from model string
#         vars = self.model.split(' + ')

#         model_cat_vars = []
#         model_cont_vars = []
#         model_cat_unique_levels = {}
#         for var in vars:
#             if var in self.cat_vars:
#                 model_cat_vars.append(var)
#                 model_cat_unique_levels[var] = self.cat_unique_levels[var]
#             elif var in self.cont_vars:
#                 model_cont_vars.append(var)
#             else:
#                 raise ValueError('Variable in model string not in cat_vars or cont_vars')
        
#         return self.df[model_cat_vars].to_numpy(), self.df[model_cont_vars].to_numpy(), model_cat_unique_levels 
    
#     def _get_labels(self):
#         return np.asarray(self.df[self.target])[:, np.newaxis] #[num_samples, ] -> [num_samples, 1] 
    
#     def __getitem__(self, idx):
#         return torch.from_numpy(self.x_cat[idx]).int(), torch.from_numpy(self.x_cont[idx]).float(), torch.from_numpy(self.y[idx]).float()
#         # shape: [num_cat_vars], [num_cont_vars], [1]

#     def __len__(self):
#         return self.y.shape[0]

# dset = DfDataset(df_restricted, _gi_model, target=target, fit=False, 
#                  cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)

# class DfDataset(Dataset):
#     def __init__(self, df, model, target=None, fit=True):
#         ''' Retrieve train,label and pred data from Dataframe directly
#         Args:  
#             df: pd.DataFrame, data, i.e., df_restricted
#             model: str, model formula, i.e., _gi_model
#             target: str, target variable, i.e., exposure/outcome
#             fit: bool, whether the dataset is for fitting or prediction

#         if fit is set to true, df should be data_to_fit; else, df should be data_to_predict
#         '''
#         self.df = df
#         self.model = model
#         self.target = target
#         self.fit = fit

#         self.x_cat, self.x_cont, self.cat_unique_levels = self._split_cat_cont() 
#         if self.fit:
#             self.y = self._get_labels()
#         else:
#             self.y = np.empty((self.x_cat.shape[0], 1))
#             self.y.fill(-1) # create dummy target for pdata
    
#     def _split_cat_cont(self):
#         # get variables from model string
#         vars = self.model.split(' + ')

#         cat_vars = []
#         cont_vars = []
#         cat_unique_levels = {}
#         for var in vars:
#             if var in self.df.columns:
#                 if self.df[var].dtype == 'int64':
                    
#                     cat_vars.append(var)
#                     if len(pd.unique(self.df[var])) == 2:
#                         cat_unique_levels[var] = len(pd.unique(self.df[var])) # record number of levels
#                     else:
#                         cat_unique_levels[var] = self.df['A'].max() + 1 # record number of levels for 'A_30', temporary strategy
#                 else:
#                     cont_vars.append(var)
#         return self.df[cat_vars].to_numpy(), self.df[cont_vars].to_numpy(), cat_unique_levels
    
#     def _get_labels(self):
#         return np.asarray(self.df[self.target])[:, np.newaxis] #[num_samples, ] -> [num_samples, 1] 
    
#     def __getitem__(self, idx):
#         return torch.from_numpy(self.x_cat[idx]).int(), torch.from_numpy(self.x_cont[idx]).float(), torch.from_numpy(self.y[idx]).float()
#         # shape: [num_cat_vars], [num_cont_vars], [1]

#     def __len__(self):
#         return self.y.shape[0]

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

def get_kfold_split(n_splits=5, shuffle=True):
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=17)

def get_kfold_dataloaders(dataset, train_index, val_index, batch_size=16, shuffle=True):
    train_dataset = Subset(dataset, train_index)
    val_dataset = Subset(dataset, val_index)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader

def get_predict_loader(dataset, batch_size=16):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

######################## define models ########################
class SimpleModel(nn.Module):
    def __init__(self, model_cat_unique_levels, n_cont):
        super().__init__()
        self.embedding_layers, self.n_emb, self.n_cont = self._get_embedding_layers(model_cat_unique_levels, n_cont)
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 16)
        self.lin2 = nn.Linear(16, 32)
        self.lin3 = nn.Linear(32, 1) # use BCEloss, so output 1
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

    def _get_embedding_layers(self, model_cat_unique_levels, n_cont):
        # Ref: https://jovian.ml/aakanksha-ns/shelter-outcome
        # decide embedding sizes
        embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _, n_categories in model_cat_unique_levels.items()]
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
    def __init__(self, split_ratio, batch_size, shuffle, n_splits, predict_all,
                 epochs, print_every, device='cpu', save_path='./'):
        super().__init__(split_ratio, batch_size, shuffle, n_splits, predict_all,
                         epochs, print_every, device, save_path)

    def _build_model(self, model_cat_vars, model_cont_vars, model_cat_unique_levels):
        n_cont = len(model_cont_vars)
        return SimpleModel(model_cat_unique_levels, n_cont) 

# params
_gi_model = "L + A_30 + R_1 + R_2 + R_3"
target = 'statin'

# test run
# no cross validation
mlp_learner = MLP(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=1, predict_all=False,
                  epochs=10, print_every=5, device='cpu', save_path='./tmp.pth')
# 5 fold cross validation 
mlp_learner = MLP(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
                  epochs=10, print_every=5, device='cpu', save_path='./tmp.pth')

# best_model = mlp_learner.fit(df_restricted, _gi_model, target, cat_vars, cont_vars, cat_unique_levels)
# pred = mlp_learner.predict(df_restricted, _gi_model, target, cat_vars, cont_vars, cat_unique_levels)
# len(pred)
# #16*31+4

best_model_address = mlp_learner.fit(tmp_data, target, model_cat_vars, model_cont_vars, model_cat_unique_levels)
pred = mlp_learner.predict(tmp_data, target, model_cat_vars, model_cont_vars, model_cat_unique_levels)
len(pred)

######################## get df_restricted ########################
from beowulf import load_uniform_statin
from beowulf.dgm import statin_dgm

# SG modified
# G = load_uniform_statin()
G, cat_vars, cont_vars, cat_unique_levels = load_uniform_statin(n=500, return_cat_cont_split=True)
# Simulation single instance of exposure and outcome
H, cat_vars, cont_vars, cat_unique_levels = statin_dgm(network=G, restricted=False,
                                                       update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)


import numpy as np
import pandas as pd
import networkx as nx
from EXP_mossspider_estimators_utils import network_to_df, exp_map_individual, tmle_unit_bounds, fast_exp_map
# params
network = H
exposure = 'statin'
outcome = 'cvd'
degree_restrict=None
alpha=0.05
continuous_bound=0.0005
verbose=False

######################## NetworkTMLE._check_degree_restrictions_ ########################
def _check_degree_restrictions_(bounds):
    """Checks degree restrictions are valid (and won't cause a later error).

    Parameters
    ----------
    bounds : list, set, array
        Specified degree bounds
    """
    if type(bounds) is not list and type(bounds) is not tuple:
        raise ValueError("`degree_restrict` should be a list/tuple of the upper and lower bounds")
    if len(bounds) != 2:
        raise ValueError("`degree_restrict` should only have two values")
    if bounds[0] > bounds[1]:
        raise ValueError("Degree restrictions must be specified in ascending order")

######################## NetworkTMLE._degree_restrictions_ ########################
def _degree_restrictions_(degree_dist, bounds):
    """Bounds the degree by the specified levels

    Parameters
    ----------
    degree_dist : array
        Degree values for the observations
    bounds : list, set, array
        Upper and lower bounds to use for the degree restriction
    """
    restrict = np.where(degree_dist < bounds[0], 1, 0)            # Apply the lower restriction on degree
    restrict = np.where(degree_dist > bounds[1], 1, restrict)     # Apply the upper restriction on degree
    return restrict



######################## NetworkTMLE.__init__() ########################
# Checking for some common problems that should provide errors
if not all([isinstance(x, int) for x in list(network.nodes())]):   # Check if all node IDs are integers
    raise ValueError("NetworkTMLE requires that "                  # ... possibly not needed?
                        "all node IDs must be integers")

if nx.number_of_selfloops(network) > 0:                            # Check for any self-loops in the network
    raise ValueError("NetworkTMLE does not support networks "      # ... self-loops don't make sense in this
                        "with self-loops")                            # ... setting

# Checking for a specified degree restriction
if degree_restrict is not None:                                    # not-None means apply a restriction
    _check_degree_restrictions_(bounds=degree_restrict)       # ... checks if valid degree restriction
    _max_degree_ = degree_restrict[1]                         # ... extract max degree as upper bound
else:                                                              # otherwise if no restriction(s)
    if nx.is_directed(network):                                    # ... directed max degree is max out-degree
        _max_degree_ = np.max([d for n, d in network.out_degree])
    else:                                                          # ... undirected max degree is max degree
        _max_degree_ = np.max([d for n, d in network.degree])

# Generate a fresh copy of the network with ascending node order
oid = "_original_id_"                                              # Name to save the original IDs
network = nx.convert_node_labels_to_integers(network,              # Copy of new network with new labels
                                             first_label=0,        # ... start at 0 for latent variance calc
                                             label_attribute=oid)  # ... saving the original ID labels

# Saving processed data copies
network = network                       # Network with correct re-labeling
exposure = exposure                     # Exposure column / attribute name
outcome = outcome                       # Outcome column / attribute name

# Background processing to convert network attribute data to pandas DataFrame
adj_matrix = nx.adjacency_matrix(network,   # Convert to adjacency matrix
                                 weight=None)    # TODO allow for weighted networks
df = network_to_df(network)                      # Convert node attributes to pandas DataFrame

# Error checking for exposure types
if not df[exposure].value_counts().index.isin([0, 1]).all():        # Only binary exposures allowed currently
    raise ValueError("NetworkTMLE only supports binary exposures "
                        "currently")

# Manage outcome data based on variable type
if df[outcome].dropna().value_counts().index.isin([0, 1]).all():    # Binary outcomes
    _continuous_outcome = False                                # ... mark as binary outcome
    _cb_ = 0.0                                                 # ... set continuous bound to be zero
    _continuous_min_ = 0.0                                     # ... saving binary min bound
    _continuous_max_ = 1.0                                     # ... saving binary max bound
else:                                                               # Continuous outcomes
    _continuous_outcome = True                                 # ... mark as continuous outcome
    _cb_ = continuous_bound                                    # ... save continuous bound value
    _continuous_min_ = np.min(df[outcome]) - _cb_         # ... determine min (with bound)
    _continuous_max_ = np.max(df[outcome]) + _cb_         # ... determine max (with bound)
    df[outcome] = tmle_unit_bounds(y=df[outcome],              # ... bound the outcomes to be (0,1)
                                    mini=_continuous_min_,
                                    maxi=_continuous_max_)

# Creating summary measure mappings for all variables in the network
summary_types = ['sum', 'mean', 'var', 'mean_dist', 'var_dist']           # Default summary measures available
handle_isolates = ['mean', 'var', 'mean_dist', 'var_dist']                # Whether isolates produce nan's
for v in [var for var in list(df.columns) if var not in [oid, outcome]]:  # All cols besides ID and outcome
    v_vector = np.asarray(df[v])                                          # ... extract array of column
    for summary_measure in summary_types:                                 # ... for each summary measure
        df[v+'_'+summary_measure] = fast_exp_map(adj_matrix,         # ... calculate corresponding measure
                                                 v_vector,
                                                 measure=summary_measure)
        if summary_measure in handle_isolates:                            # ... set isolates from nan to 0
            df[v+'_'+summary_measure] = df[v+'_'+summary_measure].fillna(0)
        cont_vars.append(v+'_'+summary_measure) #SG modified

# Creating summary measure mappings for non-parametric exposure_map_model()
exp_map_cols = exp_map_individual(network=network,               # Generate columns of indicator
                                  variable=exposure,             # ... for the exposure
                                  max_degree=_max_degree_)  # ... up to the maximum degree
_nonparam_cols_ = list(exp_map_cols.columns)                # Save column list for estimation procedure
df = pd.merge(df,                                                # Merge these columns into main data
              exp_map_cols.fillna(0),                            # set nan to 0 to keep same dimension across i
              how='left', left_index=True, right_index=True)     # Merge on index to left

#SG modified
if exposure in cat_vars:
    # print('categorical')
    cat_vars.extend(_nonparam_cols_) # add all mappings to categorical variables
    for col in _nonparam_cols_:
        cat_unique_levels[col] = pd.unique(df[col].astype('int')).max() + 1
elif exposure in cont_vars:
    # print('continuous')
    cont_vars.extend(_nonparam_cols_)
else:
    raise ValueError('exposure is neither assigned to categorical or continuous variables')

# Calculating degree for all the nodes
if nx.is_directed(network):                                         # For directed networks...
    degree_data = pd.DataFrame.from_dict(dict(network.out_degree),  # ... use the out-degree
                                            orient='index').rename(columns={0: 'degree'})
else:                                                               # For undirected networks...
    degree_data = pd.DataFrame.from_dict(dict(network.degree),      # ... use the regular degree
                                            orient='index').rename(columns={0: 'degree'})
df = pd.merge(df,                                              # Merge main data
                    degree_data,                                     # ...with degree data
                    how='left', left_index=True, right_index=True)   # ...based on index

#SG modified
cat_vars.append('degree')
cat_unique_levels['degree'] = pd.unique(df['degree'].astype('int')).max() + 1

# Apply degree restriction to data
if degree_restrict is not None:                                     # If restriction provided,
    df['__degree_flag__'] = _degree_restrictions_(degree_dist=df['degree'],
                                                            bounds=degree_restrict)
    _exclude_ids_degree_ = np.asarray(df.loc[df['__degree_flag__'] == 1].index)
else:                                                               # Else all observations are used
    df['__degree_flag__'] = 0                                  # Mark all as zeroes
    _exclude_ids_degree_ = None                                # No excluded IDs

# Marking data set restricted by degree (same as df if no restriction)
df_restricted = df.loc[df['__degree_flag__'] == 0].copy()





import patsy
tmp_data = patsy.dmatrix(_gi_model + ' -1', df_restricted, return_type="dataframe")


qn_model = "statin + statin_sum + L + I(R**2)"

data = patsy.dmatrix(qn_model + ' -1',
                     df_restricted)

tmp = patsy.dmatrix(qn_model + ' -1',
                     df_restricted, return_type="dataframe")
tmp

vars = tmp.columns
vars

for var in vars:
    if var in cat_vars:
        pass
    elif var in cont_vars:
        pass
    else:
        print(var)
        if '**' in var: # quadratic term
            pass
            # cont_vars.append(var)
        elif 'C()' in var: # categorical term
            pass
            # cat_vars.append(var)
            # cat_unique_levels[var] = pd.unique(data[var]).max() + 1
        elif '_t' in var: # threshold term, treated as categorical
            pass
            cat_vars.append(var)
            cat_unique_levels[var] = pd.unique(data[var]).max() + 1
        elif ':' in var: # interaction term, treated as continuous even between two categorical variables
            pass
            # cont_vars.appen(var)

def get_model_cat_cont_split_patsy_matrix(patsy_matrix_dataframe, cat_vars, cont_vars, cat_unique_levels):
    '''initiate model_car_vars, model_cont_vars, and cat_unique_levles, and
    update cat_vars, cont_vars, cat_unique_levels based on patsy matrix dataframe'''

    vars = patsy_matrix_dataframe.columns # all variables in patsy matrix

    model_cat_vars = []
    model_cont_vars = []
    model_cat_unique_levels = {}

    for var in vars:
        if var in cat_vars:
            model_cat_vars.append(var)
            model_cat_unique_levels[var] = cat_unique_levels[var]
        elif var in cont_vars:
            model_cont_vars.append(var)
        else:
            # update both model_{} and universal cat_vars, cont_vars adn cat_unique_levels to keep track of all variables
            if '**' in var: # quadratic term, treated as continuous
                model_cont_vars.append(var)
                cont_vars.append(var)
            elif 'C()' in var: # categorical term
                model_cat_vars.append(var)
                model_cat_unique_levels[var] = pd.unique(patsy_matrix_dataframe[var]).max() + 1
                cat_vars.append(var)
                cat_unique_levels[var] = pd.unique(patsy_matrix_dataframe[var]).max() + 1
            elif '_t' in var: # threshold term, treated as categorical
                model_cat_vars.append(var)
                model_cat_unique_levels[var] = pd.unique(patsy_matrix_dataframe[var]).max() + 1 
                cat_vars.append(var)
                cat_unique_levels[var] = pd.unique(patsy_matrix_dataframe[var]).max() + 1
            elif ':' in var: # interaction term, treated as continuous even between two categorical variables
                model_cont_vars.append(var)
                cont_vars.appen(var)
            else:
                raise ValueError(f'{var} is a unseen type of variable, cannot be assigned to categorical or continuous')
    return model_cat_vars, model_cont_vars, model_cat_unique_levels, cat_vars, cont_vars, cat_unique_levels


model_cat_vars, model_cont_vars, model_cat_unique_levels, cat_vars, cont_vars, cat_unique_levels = get_model_cat_cont_split_patsy_matrix(tmp_data, 
                                                                               cat_vars, cont_vars, cat_unique_levels)

cat_vars
cont_vars

target
df_restricted[target]



def append_target_to_df(df, patsy_matrix_dataframe, target):
    patsy_matrix_dataframe[target] = df[target]
    return patsy_matrix_dataframe

tmp_data = append_target_to_df(df_restricted, tmp_data, target)



# define categories and thresholds
from tmle_utils import create_categorical

def define_category(df_restricted, variable, bins, labels=False):
    """Function arbitrarily allows for multiple different defined thresholds

    Parameters
    ----------
    variable : str
        Variable to generate categories for
    bins : list, set, array
        Bin cutpoints to generate the categorical variable for. Uses ``pandas.cut(..., include_lowest=True)`` to
        create the binned variables.
    labels : list, set, array
        Specified labels. Can be given custom labels, but generally recommend to keep set as False
    """
    # self._categorical_any_ = True                   # Update logic to understand at least one category exists
    # self._categorical_variables_.append(variable)   # Add the variable to the list of category-generations
    # self._categorical_.append(bins)                 # Add the cut-points for the bins to the list of bins
    # self._categorical_def_.append(labels)           # Add the specified labels for the bins to the label list
    create_categorical(data=df_restricted,     # Create the desired category variable
                        variables=[variable],        # ... for the specified variable
                        bins=[bins],                 # ... for the specified bins
                        labels=[labels],             # ... with the specified labels
                        verbose=True)                # ... warns user if NaN's are being generated


define_category(df_restricted, variable='R_1_sum', bins=[0, 1, 5], labels=False)
pd.unique(df_restricted['R_1_sum_c'])
define_category(df_restricted, variable='R_2_sum', bins=[0, 1, 5], labels=False)
pd.unique(df_restricted['R_2_sum_c'])
define_category(df_restricted, variable='R_3_sum', bins=[0, 2], labels=False)
pd.unique(df_restricted['R_3_sum_c'])

gin_model = "L + A_30 + R_1 + R_2 + R_3 + C(R_1_sum_c) + C(R_2_sum_c) + C(R_3_sum_c) + A_mean_dist + L_mean_dist"

xdata = patsy.dmatrix(gin_model + ' - 1', df_restricted)       # Extract via patsy the data
xdata

xdata_tmp = patsy.dmatrix(gin_model + ' - 1', df_restricted, return_type='dataframe')       # Extract via patsy the data
xdata_tmp

pd.unique(xdata_tmp['C(R_1_sum_c)[0.0]'].astype('int')).max() + 1

# TODO: categorical variable should be int/long tensor, either do it here or in dataset
for col in xdata_tmp.columns:
    print(xdata_tmp[col].dtype)

# R_1_sum_c = 0
np.asarray(xdata)[:, 0]
df_restricted['R_1_sum_c']
np.array_equal(np.asarray(xdata)[:, 0], df_restricted['R_1_sum_c']==0)
# R_1_sum_c = 1
np.array_equal(np.asarray(xdata)[:, 1], df_restricted['R_1_sum_c']==1)
# R_2_sum_c = 0
np.asarray(xdata)[:, 2]
np.array_equal(np.asarray(xdata)[:, 2], df_restricted['R_2_sum_c'])
# R_3_sum_c only has 1 level


np.asarray(xdata)[:, 3]
df_restricted['L']

# order matters
patsy.dmatrix('C(R_1_sum_c) + C(R_2_sum_c) + C(R_3_sum_c) - 1', df_restricted)
patsy.dmatrix('C(R_2_sum_c) + C(R_1_sum_c) + C(R_3_sum_c) - 1', df_restricted)