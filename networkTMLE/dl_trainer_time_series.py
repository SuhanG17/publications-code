'''
CAVEAT

None of the independent variables involved in the forecasting task has changed over time.
Only variables that are time-varying are D, I and R.  
But, D is the dependent variable, and I, R should only be known from simulation not real epidemic.
Time Series Version should not be used, because the input is the same along the time-axis.

TODO: add a time-varing variable to the network.
'''


import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

from dl_dataset import TimeSeriesDataset, get_dataloaders, get_kfold_split, get_kfold_dataloaders, get_predict_loader
from dl_models import MLPModelTimeSeries, GCNModelTimeSeries, CNNModelTimeSeries

######################## ml abstract trainer (Parent) ########################
class AbstractMLTS:
    def __init__(self, split_ratio, batch_size, shuffle, n_splits, predict_all,
                 epochs, print_every, device='cpu', save_path='./'):
        self.epochs = epochs
        self.save_path = save_path

        self.split_ratio, self.batch_size, self.shuffle, self.n_splits, self.predict_all = split_ratio, batch_size, shuffle, n_splits, predict_all
    
        self.print_every = print_every
        self.print_per_time_slice_metrics = False # use to check if all time slices have similar performance
        self.device = device

    def fit(self, df_list, target, use_all_time_slices=True, T=10,
            adj_matrix=None, model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}, 
            n_output=2, _continuous_outcome=False, custom_path=None):
        # initiate best model
        self.best_model = None
        self.best_loss = np.inf        

        # instantiate model
        self.n_output = n_output
        self._continuous_outcome = _continuous_outcome
        self.model = self._build_model(adj_matrix, model_cat_vars, model_cont_vars, model_cat_unique_levels, T,
                                       n_output, _continuous_outcome).to(self.device)
        self.optimizer = self._optimizer()
        self.criterion = self._loss_fn()
        self._save_model(custom_path) # save the untrained model to custom_path

        # target is exposure for nuisance models, outcome for outcome model
        if self._continuous_outcome:
            fold_record = {'train_loss': [], 'val_loss': [],
                           'train_mae':  [], 'val_mae':  [],
                           'train_mse':  [], 'val_mse':  [],
                           'train_rmse': [], 'val_rmse': []}
        else:
            fold_record = {'train_loss': [], 'val_loss': [],'train_acc':[],'val_acc':[]}

        if self.n_splits > 1: # Kfold cross validation is used
            splits, dset = self._data_preprocess(df_list, target, use_all_time_slices,
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

                    if self._continuous_outcome:
                        fold_record['train_loss'].append(loss_train)
                        fold_record['val_loss'].append(loss_valid)
                        for (metric_name_train, metric_value_train), (metric_name_val, metirc_value_val) in zip(metrics_train.items(), metrics_valid.items()):
                            fold_record['train_' + metric_name_train].append(metric_value_train)
                            fold_record['val_' + metric_name_val].append(metirc_value_val)
                    else:
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
            
            if self._continuous_outcome:
                avg_train_loss = np.mean(fold_record['train_loss'])
                avg_val_loss = np.mean(fold_record['val_loss'])
                avg_metrics_train = {}
                avg_metrics_val = {}
                for metric_name in ['mae', 'mse', 'rmse']:
                    avg_metrics_train['train_' + metric_name] = np.mean(fold_record['train_' + metric_name])
                    avg_metrics_val['val_' + metric_name] = np.mean(fold_record['val_' + metric_name])
                
                print(f'Performance of {self.n_splits} fold cross validation')
                print(f'Average Training Loss: {avg_train_loss:.4f} \t Average Val Loss: {avg_val_loss:.4f}')
                for metric_name in ['mae', 'mse', 'rmse']:
                    print(f'Average Training {metric_name}: {avg_metrics_train["train_" + metric_name]:.3f} \t Average Val {metric_name}: {avg_metrics_val["val_" + metric_name]:.3f}')
            else:
                avg_train_loss = np.mean(fold_record['train_loss'])
                avg_val_loss = np.mean(fold_record['val_loss'])
                avg_train_acc = np.mean(fold_record['train_acc'])
                avg_val_acc = np.mean(fold_record['val_acc'])

                print(f'Performance of {self.n_splits} fold cross validation')
                print(f'Average Training Loss: {avg_train_loss:.4f} \t Average Val Loss: {avg_val_loss:.4f} \t Average Training Acc: {avg_train_acc:.3f} \t Average Test Acc: {avg_val_acc:.3f}')  
        else:
            self.train_loader, self.valid_loader, self.test_loader = self._data_preprocess(df_list, target, use_all_time_slices,
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

        if custom_path is None: 
            return self.save_path
        else:
            return custom_path

    def predict(self, df_list, target, use_all_time_slices=True, T=10,
                adj_matrix=None, model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}, 
                n_output=2, _continuous_outcome=False, custom_path=None):
        print(f'============================= Predicting =============================')
        # instantiate model
        self.n_output = n_output
        self._continuous_outcome = _continuous_outcome

        self.model = self._build_model(adj_matrix, model_cat_vars, model_cont_vars, model_cat_unique_levels, T, 
                                       n_output, _continuous_outcome).to(self.device)
        self._load_model(custom_path)
        self.criterion = self._loss_fn()

        if self.predict_all:
            dset = TimeSeriesDataset(df_list, target=target, use_all_time_slices=use_all_time_slices,
                                     model_cat_vars=model_cat_vars, 
                                     model_cont_vars=model_cont_vars, 
                                     model_cat_unique_levels=model_cat_unique_levels)
            self.test_loader = get_predict_loader(dset, self.batch_size)
        else:
            _, _, self.test_loader = self._data_preprocess(df_list, target, use_all_time_slices=use_all_time_slices,
                                                           model_cat_vars=model_cat_vars, 
                                                           model_cont_vars=model_cont_vars, 
                                                           model_cat_unique_levels=model_cat_unique_levels)
        
        pred = self.test_epoch(epoch=0, return_pred=True) # pred should probabilities, one for binary
        return pred
    
    def train_epoch(self, epoch):
        self.model.train() # turn on train-mode

        # record loss and metrics for every print_every mini-batches
        running_loss = 0.0 
        if self._continuous_outcome:
            running_metrics = {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0}
        else:
            running_metrics = 0.0
        # record loss and metrics for the whole epoch
        cumu_loss = 0.0 
        if self._continuous_outcome:
            cumu_metrics = {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0}
        else:
            cumu_metrics = 0.0

        for i, (x_cat, x_cont, y, sample_idx) in enumerate(self.train_loader):
            # send to device
            x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device) 
            sample_idx = sample_idx.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(x_cat, x_cont, sample_idx) # shape [batch_size, num_classes, T] 

            if self._continuous_outcome:
                y = y.float().unsqueeze(1) # shape [batch_size, T] -> [batch_size, 1, T] for l1_loss
            else:
                if self.n_output == 2: # binary classification
                    # BCEWithLogitsLoss requires target as float, same size as outputs
                    y = y.float().unsqueeze(1) # shape [batch_size, T] -> [batch_size, 1, T]
                else:
                    # CrossEntropyLoss requires target (class indicies) as int
                    y = y.long() # shape [batch_size, T]
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            # metrics
            if self.print_per_time_slice_metrics and i % self.print_every == self.print_every - 1:
                print(f'metrics for batch {i + 1}: train')
                # metrics_list = []
                for i in range(x_cat.shape[-1]):
                    # metrics_list.append(self._metrics(outputs[:, :, i], y[..., i]))
                    print(f'time slice {i}')
                    print(self._metrics(outputs[:, :, i], y[..., i]))        
                print()        

            metrics = self._metrics_ts(outputs, y)

            # print statistics
            running_loss += loss.item()
            cumu_loss += loss.item()
            
            if self._continuous_outcome:
                for metric_name, metric_value in metrics.items():
                    running_metrics[metric_name] += metric_value
                    cumu_metrics[metric_name] += metric_value
            else:
                running_metrics += metrics
                cumu_metrics += metrics

            if i % self.print_every == self.print_every - 1:    # print every mini-batches
                if self._continuous_outcome:
                    report_string = f'[{epoch + 1}, {i + 1:5d}] | loss: {running_loss / self.print_every:.3f}' 
                    for metric_name, metric_value in running_metrics.items():
                        report_string += f' | {metric_name}: {metric_value / self.print_every:.3f}'
                    print(report_string)

                    running_loss = 0.0
                    running_metrics = {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0}
                else:
                    print(f'[{epoch + 1}, {i + 1:5d}] | loss: {running_loss / self.print_every:.3f} | acc: {running_metrics / self.print_every:.3f}')
                    running_loss = 0.0
                    running_metrics = 0.0           

        if self._continuous_outcome:
            for metric_name, metric_value in cumu_metrics.items():
                cumu_metrics[metric_name] = metric_value / len(self.train_loader)
            return cumu_loss / len(self.train_loader), cumu_metrics
        else:
            return cumu_loss / len(self.train_loader), cumu_metrics / len(self.train_loader)

    def valid_epoch(self, epoch): 
        self.model.eval() # turn on eval mode

        # record loss and metrics for the whole epoch
        cumu_loss = 0.0 
        if self._continuous_outcome:
            cumu_metrics = {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0}
        else:
            cumu_metrics = 0.0

        with torch.no_grad():
            for i, (x_cat, x_cont, y, sample_idx) in enumerate(self.valid_loader):
                # send to device
                x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device) 
                sample_idx = sample_idx.to(self.device)
                outputs = self.model(x_cat, x_cont, sample_idx) # shape [batch_size, num_classes, T] 
                if self._continuous_outcome:
                    y = y.float().unsqueeze(1) # shape [batch_size, T] -> [batch_size, 1, T] for l1_loss
                else:
                    if self.n_output == 2: # binary classification
                        # BCEWithLogitsLoss requires target as float, same size as outputs
                        y = y.float().unsqueeze(1) # shape [batch_size, T] -> [batch_size, 1, T] 
                    else:
                        # CrossEntropyLoss requires target (class indicies) as int, shape [batch_size]
                        y = y.long() # shape [batch_size, T] 
                loss = self.criterion(outputs, y)

                # metrics
                if self.print_per_time_slice_metrics and i == len(self.valid_loader) - 1:
                    print(f'metrics for batch {i + 1}: validation')
                    # metrics_list = []
                    for i in range(x_cat.shape[-1]):
                        # metrics_list.append(self._metrics(outputs[:, :, i], y[..., i]))
                        print(f'time slice {i}')
                        print(self._metrics(outputs[:, :, i], y[..., i]))        
                    print()        

                metrics = self._metrics_ts(outputs, y)

                # print statistics
                cumu_loss += loss.item()

                if self._continuous_outcome:
                    for metric_name, metric_value in metrics.items():
                        cumu_metrics[metric_name] += metric_value
                else:
                    cumu_metrics += metrics

            if self._continuous_outcome:
                report_string = f'[{epoch + 1}, {i + 1:5d}] | loss: {cumu_loss / len(self.valid_loader):.3f}'
                for metric_name, metric_value in cumu_metrics.items():
                    report_string += f' | {metric_name}: {metric_value / len(self.valid_loader):.3f}'
                print(report_string) 
            else:
                print(f'[{epoch + 1}, {i + 1:5d}] | loss: {cumu_loss / len(self.valid_loader):.3f} | acc: {cumu_metrics / len(self.valid_loader):.3f}')
            
        if self._continuous_outcome:
            for metric_name, metric_value in cumu_metrics.items():
                cumu_metrics[metric_name] = metric_value / len(self.valid_loader)
            return cumu_loss / len(self.valid_loader), cumu_metrics
        else:
            return cumu_loss / len(self.valid_loader), cumu_metrics / len(self.valid_loader)

    def test_epoch(self, epoch, return_pred=False):
        self.model.eval() # turn on eval mode

        if return_pred:
            pred_list = []
        
        # record loss and metrics for the whole epoch
        cumu_loss = 0.0 
        if self._continuous_outcome:
            cumu_metrics = {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0}
        else:
            cumu_metrics = 0.0

        with torch.no_grad():
            for i, (x_cat, x_cont, y, sample_idx) in enumerate(self.test_loader):
                # send to device
                x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device) 
                sample_idx = sample_idx.to(self.device) 
                outputs = self.model(x_cat, x_cont, sample_idx) # shape [batch_size, num_classes, T] 
                if return_pred:
                    if self._continuous_outcome:
                        pred_list.append(outputs.detach().to('cpu').numpy())
                    else:
                        if self.n_output == 2:
                            pred_list.append(torch.sigmoid(outputs).detach().to('cpu').numpy())
                        else:
                            pred_list.append(torch.softmax(outputs, dim=1).detach().to('cpu').numpy())
                
                if self._continuous_outcome:
                    y = y.float().unsqueeze(1) # shape [batch_size, T] -> [batch_size, 1, T] for l1_loss
                else:
                    if self.n_output == 2: # binary classification
                        # BCEWithLogitsLoss requires target as float, same size as outputs
                        y = y.float().unsqueeze(1) # shape [batch_size, T] -> [batch_size, 1, T]
                    else:
                        # CrossEntropyLoss requires target (class indicies) as int, shape [batch_size]
                        y = y.long() # shape [batch_size, T]
                loss = self.criterion(outputs, y)

                # metrics
                if self.print_per_time_slice_metrics and i == len(self.test_loader) - 1:
                    print(f'metrics for batch {i + 1}: test')
                    # metrics_list = []
                    for i in range(x_cat.shape[-1]):
                        # metrics_list.append(self._metrics(outputs[:, :, i], y[..., i]))
                        print(f'time slice {i}')
                        print(self._metrics(outputs[:, :, i], y[..., i]))        
                    print()        

                metrics = self._metrics_ts(outputs, y)

                # print statistics
                cumu_loss += loss.item()

                if self._continuous_outcome:
                    for metric_name, metric_value in metrics.items():
                        cumu_metrics[metric_name] += metric_value
                else:
                    cumu_metrics += metrics

            if return_pred: 
                if self._continuous_outcome:
                    report_string = f'[data_to_predict averaged] | loss: {cumu_loss / len(self.test_loader):.3f}'
                    for metric_name, metric_value in cumu_metrics.items():
                        report_string += f' | {metric_name}: {metric_value / len(self.test_loader):.3f}'
                    print(report_string) 
                else:
                    print(f'[data_to_predict averaged] | loss: {cumu_loss / len(self.test_loader):.3f} | acc: {cumu_metrics / len(self.test_loader):.3f}') 
            else:
                if self._continuous_outcome:
                    report_string = f'[{epoch + 1}, {i + 1:5d}] | loss: {cumu_loss / len(self.test_loader):.3f}'
                    for metric_name, metric_value in cumu_metrics.items():
                        report_string += f' | {metric_name}: {metric_value / len(self.test_loader):.3f}'
                    print(report_string) 
                else:
                    print(f'[{epoch + 1}, {i + 1:5d}] | loss: {cumu_loss / len(self.test_loader):.3f} | acc: {cumu_metrics / len(self.test_loader):.3f}')
                
        if return_pred:
            return pred_list
        else:
            if self._continuous_outcome:
                for metric_name, metric_value in cumu_metrics.items():
                    cumu_metrics[metric_name] = metric_value / len(self.test_loader)
                return cumu_loss / len(self.test_loader), cumu_metrics
            else:
                return cumu_loss / len(self.test_loader), cumu_metrics / len(self.test_loader)

    def _build_model(self):
        pass 

    def _data_preprocess(self, df, target=None, use_all_time_slices=True,
                         model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}):
        ts_dset = TimeSeriesDataset(df, target=target, use_all_time_slices=use_all_time_slices,
                                    model_cat_vars=model_cat_vars, 
                                    model_cont_vars=model_cont_vars, 
                                    model_cat_unique_levels=model_cat_unique_levels)

        if self.n_splits > 1: # Kfold cross validation is used
            return get_kfold_split(n_splits=self.n_splits, shuffle=self.shuffle), ts_dset
        else:
            train_loader, valid_loader, test_loader = get_dataloaders(ts_dset,
                                                                      split_ratio=self.split_ratio, 
                                                                      batch_size=self.batch_size,
                                                                      shuffle=self.shuffle)           
            return train_loader, valid_loader, test_loader


    def _optimizer(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # return optim.Adam(self.model.parameters(), lr=0.001)

    def _loss_fn(self):
        if self._continuous_outcome:
            return nn.L1Loss() # mae
            # return nn.MSELoss() # mse
        else:
            if self.n_output == 2: # binary classification
                return nn.BCEWithLogitsLoss() # no need for sigmoid, require 1 output for binary classfication
            else:
                return nn.CrossEntropyLoss() # no need for softmax, require [n_output] output for classification

    def _metrics(self, outputs, labels):
        ''' calculate metrics for each time slice:
        _continuous_outcome:
            outputs: shape [batch_size, 1]
            labels:  shape [batch_size, 1]
        else:
            if n_output == 2:
                outputs: shape [batch_size, 1]
                labels:  shape [batch_size, 1]
            else:
                outputs: shape [batch_size, num_classes]
                labels:  shape [batch_size]
        '''
        if self._continuous_outcome:
            mae_error = torch.abs(outputs - labels).mean()
            mse_error = torch.pow(outputs - labels, 2).mean()
            rmse_error = torch.sqrt(mse_error)
            return {'mae':mae_error.item(), 'mse':mse_error.item(), 'rmse':rmse_error.item()}
        else:
            if self.n_output == 2:
                pred = torch.sigmoid(outputs) # get binary probability
                pred_binary = torch.round(pred) # get binary prediction
                return (pred_binary == labels).sum().item()/labels.shape[0] # num samples correctly classified / num_samples
            else:
                # the class with the highest energy is what we choose as prediction, if output 2 categories for binary classificaiton
                _, predicted = torch.max(outputs.data, 1) # [batch_size, num_classes] -> [batch_size]
                # print(f'shape: pred {predicted.shape}, label {labels.shape}')
                # print(f'device: pred {predicted.device}, label {labels.device}')
                return (predicted == labels).sum().item()/labels.shape[0]

    def _metrics_ts(self, outputs, labels):
        ''' calculate metrics for ALL time slice:
        _continuous_outcome:
            outputs: shape [batch_size, 1, T]
            labels:  shape [batch_size, 1, T]
        else:
            if n_output == 2:
                outputs: shape [batch_size, 1, T]
                labels:  shape [batch_size, 1, T]
            else:
                outputs: shape [batch_size, num_classes, T]
                labels:  shape [batch_size, T]
        '''
        if self._continuous_outcome:
            mae_error = torch.abs(outputs - labels).mean()
            mse_error = torch.pow(outputs - labels, 2).mean()
            rmse_error = torch.sqrt(mse_error)
            return {'mae':mae_error.item(), 'mse':mse_error.item(), 'rmse':rmse_error.item()}
        else:
            if self.n_output == 2:
                pred = torch.sigmoid(outputs) # get binary probability
                pred_binary = torch.round(pred) # get binary prediction
                return (pred_binary == labels).sum().item()/labels.numel() # num samples correctly classified / num_samples
            else:
                # the class with the highest energy is what we choose as prediction, if output 2 categories for binary classificaiton
                _, predicted = torch.max(outputs.data, 1) # [batch_size, num_classes] -> [batch_size]
                # print(f'shape: pred {predicted.shape}, label {labels.shape}')
                # print(f'device: pred {predicted.device}, label {labels.device}')
                return (predicted == labels).sum().item()/labels.numel()

    def _save_model(self, custom_path=None):
        if custom_path is None:
            custom_path = self.save_path
        torch.save(self.model.state_dict(), custom_path)

    def _load_model(self, custom_path=None):
        if custom_path is None:
            custom_path = self.save_path
        self.model.load_state_dict(torch.load(custom_path))


######################## MLP model trainer (Child) ########################
class MLPTS(AbstractMLTS):
    def __init__(self, split_ratio, batch_size, shuffle, n_splits, predict_all,
                 epochs, print_every, device='cpu', save_path='./'):
        super(MLPTS, self).__init__(split_ratio, batch_size, shuffle, n_splits, predict_all,
                         epochs, print_every, device, save_path)
    
    def _optimizer(self):
        # return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        return optim.Adam(self.model.parameters(), lr=0.0001)

    def _build_model(self, adj_matrix, 
                    model_cat_vars, model_cont_vars, model_cat_unique_levels, T,
                    n_output, _continuous_outcome):
        n_cont = len(model_cont_vars)
        net = MLPModelTimeSeries(adj_matrix, model_cat_unique_levels, n_cont, T, 
                                 n_output, _continuous_outcome)
        if (self.device != 'cpu') and (torch.cuda.device_count() > 1):
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
            net = net.to(self.device)
        return net


######################## GCN model trainer (Child) ########################
class GCNTS(AbstractMLTS):
    def __init__(self, split_ratio, batch_size, shuffle, n_splits, predict_all,
                 epochs, print_every, device='cpu', save_path='./'):
        super(GCNTS, self).__init__(split_ratio, batch_size, shuffle, n_splits, predict_all,
                         epochs, print_every, device, save_path)

    def _optimizer(self):
        # return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        return optim.Adam(self.model.parameters(), lr=0.0001)

    def _build_model(self, adj_matrix, 
                     model_cat_vars, model_cont_vars, model_cat_unique_levels, T, 
                     n_output, _continuous_outcome):
        n_cont = len(model_cont_vars)
        net = GCNModelTimeSeries(adj_matrix, model_cat_unique_levels, n_cont, T, 
                                 n_output, _continuous_outcome) 
        if (self.device != 'cpu') and (torch.cuda.device_count() > 1):
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
            net = net.to(self.device)
        return net

######################## CNN model trainer (Child) ########################
class CNNTS(AbstractMLTS):
    def __init__(self, split_ratio, batch_size, shuffle, n_splits, predict_all,
                 epochs, print_every, device='cpu', save_path='./'):
        super(CNNTS, self).__init__(split_ratio, batch_size, shuffle, n_splits, predict_all,
                         epochs, print_every, device, save_path)

    def _optimizer(self):
        # return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        return optim.Adam(self.model.parameters(), lr=0.0001)

    def _build_model(self, adj_matrix, 
                     model_cat_vars, model_cont_vars, model_cat_unique_levels, T, 
                     n_output, _continuous_outcome):
        n_cont = len(model_cont_vars)
        net = CNNModelTimeSeries(adj_matrix, model_cat_unique_levels, n_cont, T, 
                                 n_output, _continuous_outcome) 
        if (self.device != 'cpu') and (torch.cuda.device_count() > 1):
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
            net = net.to(self.device)
        return net


if __name__ == '__main__':

    # from tmle_dl import NetworkTMLE  # modfied library
    from tmle_dl_time_series import NetworkTMLETimeSeries
    # from amonhen import NetworkTMLE   # internal version, recommended to use library above instead
    from beowulf import load_random_vaccine
    # from beowulf.dgm import statin_dgm, naloxone_dgm, diet_dgm, vaccine_dgm
    # from Beowulf.beowulf.dgm.vaccine_with_cat_cont_split import vaccine_dgm_time_series
    from beowulf.dgm import vaccine_dgm_time_series
   
    # random network with reproducibility
    torch.manual_seed(17) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    ######################################### Vaccine-Infection -- DGM Time Series: test run #########################################
    # loading uniform network with diet W
    G, cat_vars, cont_vars, cat_unique_levels = load_random_vaccine(n=500, return_cat_cont_split=True)
    # Simulation single instance of exposure and outcome
    H, network_list, cat_vars, cont_vars, cat_unique_levels = vaccine_dgm_time_series(network=G, restricted=False, 
                                                                                      update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)
    
    ## network-TMLE applies to generated data
    # tmle = NetworkTMLE(H, exposure='vaccine', outcome='D', verbose=False, degree_restrict=(0, 18),
    #                    cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
    #                    use_deep_learner_A_i=True)
    # tmle = NetworkTMLE(H, exposure='vaccine', outcome='D', verbose=False, degree_restrict=(0, 18),
    #                    cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
    #                    use_deep_learner_A_i_s=True)

    # tmle = NetworkTMLETimeSeries(network_list, exposure='vaccine', outcome='D', verbose=False, degree_restrict=(0, 18),
    #                              cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
    #                              use_deep_learner_outcome=True) 
    
    # tmle = NetworkTMLETimeSeries(network_list, exposure='vaccine', outcome='D', verbose=False, degree_restrict=(0, 18),
    #                              cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
    #                              use_deep_learner_A_i=True, use_deep_learner_outcome=True) 
    
    tmle = NetworkTMLETimeSeries(network_list, exposure='vaccine', outcome='D', verbose=False, degree_restrict=(0, 18),
                                 cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
                                 use_deep_learner_A_i=True, use_deep_learner_A_i_s=True, use_deep_learner_outcome=True) 

    # instantiation of deep learning model
    # 5 fold cross validation 
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    # # MLP model
    # deep_learner_a_i = MLPTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
    #                          epochs=10, print_every=5, device=device, save_path='./tmp.pth')
    # deep_learner_a_i_s = MLPTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
    #                              epochs=10, print_every=5, device=device, save_path='./tmp.pth')    
    # deep_learner_outcome = MLPTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
    #                              epochs=10, print_every=5, device=device, save_path='./tmp.pth')                                    

    # GCN model
    # deep_learner_a_i = GCNTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
    #                          epochs=10, print_every=5, device=device, save_path='./tmp.pth')
    # deep_learner_a_i_s = GCNTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
    #                            epochs=10, print_every=5, device=device, save_path='./tmp.pth')
    # deep_learner_outcome = GCNTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
    #                              epochs=10, print_every=5, device=device, save_path='./tmp.pth')
    
    # # CNN model
    deep_learner_a_i = CNNTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
                             epochs=10, print_every=5, device=device, save_path='./tmp.pth')
    deep_learner_a_i_s = CNNTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
                               epochs=10, print_every=5, device=device, save_path='./tmp.pth')
    deep_learner_outcome = CNNTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
                                 epochs=10, print_every=5, device=device, save_path='./tmp.pth')
    
    # tmle.exposure_model("A + H + H_mean + degree")
    tmle.exposure_model("A + H + H_mean + degree", custom_model=deep_learner_a_i) # use_deep_learner_A_i=True
    # tmle.exposure_map_model("vaccine + A + H + H_mean + degree",
                            # measure='sum', distribution='poisson')
    tmle.exposure_map_model("vaccine + A + H + H_mean + degree",
                            measure='sum', distribution='poisson', custom_model=deep_learner_a_i_s)  # use_deep_learner_A_i_s=True
    # tmle.outcome_model("vaccine + vaccine_mean + A + H + A_mean + H_mean + degree")
    tmle.outcome_model("vaccine + vaccine_mean + A + H + A_mean + H_mean + degree", custom_model=deep_learner_outcome) # use_deep_learner_outcome=True
    tmle.fit(p=0.55, bound=0.01)
    tmle.summary()
    