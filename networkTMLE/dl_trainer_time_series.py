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
    from Beowulf.beowulf.dgm.vaccine_with_cat_cont_split import vaccine_dgm_time_series
   
    # random network with reproducibility
    torch.manual_seed(17) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    ######################################### Vaccine-Infection -- DGM Time Series: test run #########################################
    # loading uniform network with diet W
    G, cat_vars, cont_vars, cat_unique_levels = load_random_vaccine(n=500, return_cat_cont_split=True)
    # Simulation single instance of exposure and outcome
    H, cat_vars, cont_vars, cat_unique_levels, network_list = vaccine_dgm_time_series(network=G, restricted=False, 
                                                                                      update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)
    
    ## network-TMLE applies to generated data
    # tmle = NetworkTMLE(H, exposure='vaccine', outcome='D', verbose=False, degree_restrict=(0, 18),
    #                    cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
    #                    use_deep_learner_A_i=True)
    # tmle = NetworkTMLE(H, exposure='vaccine', outcome='D', verbose=False, degree_restrict=(0, 18),
    #                    cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
    #                    use_deep_learner_A_i_s=True)
    tmle = NetworkTMLETimeSeries(network_list, exposure='vaccine', outcome='D', verbose=False, degree_restrict=(0, 18),
                                 cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
                                 use_deep_learner_outcome=True) 

    # instantiation of deep learning model
    # 5 fold cross validation 
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    deep_learner = MLPTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
                         epochs=10, print_every=5, device=device, save_path='./tmp.pth')
    # deep_learner = GCN(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
    #                   epochs=10, print_every=5, device=device, save_path='./tmp.pth')
    
    tmle.exposure_model("A + H + H_mean + degree")
    # tmle.exposure_model("A + H + H_mean + degree", custom_model=deep_learner) # use_deep_learner_A_i=True
    tmle.exposure_map_model("vaccine + A + H + H_mean + degree",
                            measure='sum', distribution='poisson')
    # tmle.exposure_map_model("vaccine + A + H + H_mean + degree",
    #                         measure='sum', distribution='poisson', custom_model=deep_learner)  # use_deep_learner_A_i_s=True
    # tmle.outcome_model("vaccine + vaccine_mean + A + H + A_mean + H_mean + degree")
    tmle.outcome_model("vaccine + vaccine_mean + A + H + A_mean + H_mean + degree", custom_model=deep_learner) # use_deep_learner_outcome=True
    tmle.fit(p=0.55, bound=0.01)
    tmle.summary()
    
    # # ############################# scratch #################################
    # tmle.df_restricted.columns
    # tmle.df_restricted['diet_t3']
    # tmle.df_restricted['diet_t3'].value_counts()
    # tmle.df_restricted['bmi']

    # cat_vars
    # cont_vars
    # cat_unique_levels

    
    # import patsy
    # # exposure A_i
    # # data_to_fit = tmle.df_restricted.copy()
    # # data_to_predict = tmle.df_restricted.copy()

    # # xdata = patsy.dmatrix(tmle._gi_model + ' - 1', data_to_fit, return_type="dataframe")       # Extract via patsy the data
    # # ydata = data_to_fit[tmle.exposure] 
    # # n_output = pd.unique(ydata).shape[0]
    # # print(f'gi_model: n_output = {n_output} for target variable {tmle.exposure}')

    # # pdata = patsy.dmatrix(tmle._gi_model + ' - 1', data_to_predict, return_type="dataframe")   # Extract via patsy the data
    # # pdata_y = data_to_predict[tmle.exposure]
    # # custom_path = 'denom_' + 'A_i_' + tmle.exposure  + '.pth'

    # # outcome
    # xdata = patsy.dmatrix("diet + diet_t3 + B + G + E + E_sum + B_mean_dist" + ' - 1', tmle.df_restricted, return_type="dataframe")
    # ydata = tmle.df_restricted[tmle.outcome] 
    # n_output = pd.unique(ydata).shape[0]

    # from tmle_utils import get_model_cat_cont_split_patsy_matrix, append_target_to_df 
    # # exposure A_i
    # # model_cat_vars, model_cont_vars, model_cat_unique_levels, cat_vars, cont_vars, cat_unique_levels = get_model_cat_cont_split_patsy_matrix(xdata, 
    # #                                                                                                                                          cat_vars, cont_vars, cat_unique_levels)
    # # fit_df = append_target_to_df(ydata, xdata, tmle.exposure)

    # # outcome
    # model_cat_vars, model_cont_vars, model_cat_unique_levels, cat_vars, cont_vars, cat_unique_levels = get_model_cat_cont_split_patsy_matrix(xdata, 
    #                                                                                                                                          cat_vars, cont_vars, cat_unique_levels)
    # fit_df = append_target_to_df(ydata, xdata, tmle.outcome)

    # fit_df
    # model_cat_vars
    # model_cont_vars
    # model_cat_unique_levels
    # fit_df['B_30'].value_counts()
    # pd.unique(fit_df['B_30'])
    # pd.unique(tmle.df_restricted['B'])

    # tmle._continuous_outcome
    # model = deep_learner._build_model(None, model_cat_vars, model_cont_vars, model_cat_unique_levels, n_output, tmle._continuous_outcome)
    # embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _, n_categories in model_cat_unique_levels.items()]
    # embedding_sizes
    # model_cat_unique_levels

    # tmle.df_restricted['diet_t3']
    # pd.unique(tmle.df_restricted['diet_t3'])

    # cat_unique_levels

    # pd.unique(tmle.df_restricted['diet_t3']).max() + 1 
    
    # # poisson model
    # import statsmodels.api as sm
    # import statsmodels.formula.api as smf
    # from scipy.stats import poisson, norm

    # data_to_fit = tmle.df_restricted.copy()
    # data_to_predict = tmle.df_restricted.copy()
   
    # gs_model = tmle._gs_measure_ + ' ~ ' + tmle._gs_model     # Setup the model form
    # # if self._gs_custom_ is None:                              # If no custom model provided
    # f = sm.families.family.Poisson()                      # ... GLM with Poisson family
    # treat_s_model = smf.glm(gs_model,                     # Estimate model
    #                         data_to_fit,                  # ... with data to fit
    #                         family=f).fit()               # ... and Poisson distribution
    # # if store_model:                                       # If estimating denominator
    # #     self._treatment_models.append(treat_s_model)      # ... store the model
    # pred = treat_s_model.predict(data_to_predict)         # Predicted values with data to predict

    # # # If verbose requested, provide model output
    # # if self._verbose_:
    # #     print('==============================================================================')
    # #     print(verbose_label+': '+self._gs_measure_)
    # #     print(treat_s_model.summary())

    # pr_s = poisson.pmf(data_to_predict[tmle._gs_measure_], pred)


    # import patsy
    # data_to_fit = tmle.df_restricted.copy()
    # xdata = patsy.dmatrix(tmle._gs_model + ' - 1', 
    #                       data_to_fit, return_type="dataframe")       # Extract via patsy the data
    # ydata = data_to_fit[tmle._gs_measure_]
    # n_output = pd.unique(ydata).shape[0] 
    # # print(f'gs_model: n_output = {n_output} for target variable {self._gs_measure_}')

    # # pdata = patsy.dmatrix(self._gs_model + ' - 1', 
    # #                         data_to_predict, return_type="dataframe")   # Extract via patsy the data
    # # pdata_y = data_to_predict[self._gs_measure_]

    # from tmle_utils import get_model_cat_cont_split_patsy_matrix, append_target_to_df
    # model_cat_vars, model_cont_vars, model_cat_unique_levels, cat_vars, cont_vars, cat_unique_levels = get_model_cat_cont_split_patsy_matrix(xdata, 
    #                                                                                                                                          cat_vars, cont_vars, cat_unique_levels)
    # fit_df = append_target_to_df(ydata, xdata, tmle._gs_measure_)  

    # # initiate best model
    # # self.best_model = None
    # # self.best_loss = np.inf        

    # # instantiate model
    # # self.n_output = n_output
    

    # model_cat_vars_new = ['A_30']
    # model_cont_vars_new = ['L', 'statin', 'R_1', 'R_2', 'R_3']
    # model_cat_unique_levels_new = {'A_30':31}
    # mlp_learner.model = mlp_learner._build_model(model_cat_vars_new, 
    #                                              model_cont_vars_new, 
    #                                              model_cat_unique_levels_new, n_output)

    # # mlp_learner.model = mlp_learner._build_model(model_cat_vars, model_cont_vars, model_cat_unique_levels, n_output)
    # mlp_learner.optimizer = mlp_learner._optimizer()
    # mlp_learner.n_output = n_output
    # mlp_learner.criterion = mlp_learner._loss_fn()

    # # target is exposure for nuisance models, outcome for outcome model
    # fold_record = {'train_loss': [], 'val_loss': [],'train_acc':[],'val_acc':[]}


    # # splits, dset = mlp_learner._data_preprocess(fit_df, tmle._gs_measure_,
    # #                                             model_cat_vars=model_cat_vars, 
    # #                                             model_cont_vars=model_cont_vars, 
    # #                                             model_cat_unique_levels=model_cat_unique_levels)
    # splits, dset = mlp_learner._data_preprocess(fit_df, tmle._gs_measure_,
    #                                             model_cat_vars=model_cat_vars_new, 
    #                                             model_cont_vars=model_cont_vars_new, 
    #                                             model_cat_unique_levels=model_cat_unique_levels_new)

    # for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dset)))):
    #     print('Fold {}'.format(fold + 1))
    #     train_loader, valid_loader = get_kfold_dataloaders(dset, train_idx, val_idx, 
    #                                                        batch_size=16,
    #                                                        shuffle=True)
    #     break

    # epochs = 10
    # mlp_learner.train_loader = train_loader
    # mlp_learner.valid_loader = valid_loader
    # for epoch in range(epochs):
    #     print(f'============================= Epoch {epoch + 1}: Training =============================')
    #     loss_train, metrics_train = mlp_learner.train_epoch(epoch)
    #     print(f'============================= Epoch {epoch + 1}: Validation =============================')
    #     loss_valid, metrics_valid = mlp_learner.valid_epoch(epoch)

    # for i, (x_cat, x_cont, y, idx) in enumerate(train_loader):
    #     # send to device
    #     x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device) 


    #     # zero the parameter gradients
    #     mlp_learner.optimizer.zero_grad()

    #     # forward + backward + optimize
    #     outputs = mlp_learner.model(x_cat, x_cont)
    #     if mlp_learner.n_output == 2: # binary classification
    #         # BCEWithLogitsLoss requires target as float, same size as outputs
    #         y = y.float() 
    #     else:
    #         # CrossEntropyLoss requires target (class indicies) as int, shape [batch_size]
    #         y = y.long().squeeze(-1) 
    #     loss = mlp_learner.criterion(outputs, y)
    #     loss.backward()
    #     mlp_learner.optimizer.step()

    #     # metrics
    #     metrics = mlp_learner._metrics(outputs, y)

    #     print(loss)
    #     print(metrics)

    # mlp_learner.model
    # mlp_learner.model.module.embedding_layers[0].weight

    # from tmle_utils import get_probability_from_multilevel_prediction
    # pred = mlp_learner.predict(fit_df, tmle._gs_measure_, 
    #                            model_cat_vars, model_cont_vars, model_cat_unique_levels, n_output)
    # pred = np.concatenate(pred, 0)
    # pred = get_probability_from_multilevel_prediction(pred, ydata) 

    # p=0.35
    # samples=100
    # seed=None
    # pooled_df = tmle._generate_pooled_sample(p=p, samples=samples, seed=seed)
    # pooled_data_restricted = pooled_df.loc[pooled_df['__degree_flag__'] == 0].copy()

    # pooled_data_restricted.columns
    # tmle.df_restricted.columns

    # import patsy
    # xdata = patsy.dmatrix(tmle._gs_model + ' - 1', 
    #                       pooled_data_restricted, return_type="dataframe")       # Extract via patsy the data
    # pdata = patsy.dmatrix(tmle._gs_model + ' - 1', 
    #                       tmle.df_restricted, return_type="dataframe")   # Extract via patsy the data
    
    # xdata
    # pdata

    # pd.unique(pooled_data_restricted[tmle._gs_measure_]).shape[0]
    # tmle._gs_measure_

    # pd.unique(pooled_data_restricted[tmle.exposure])

    # from tmle_utils import get_model_cat_cont_split_patsy_matrix
    # model_cat_vars, model_cont_vars, model_cat_unique_levels, cat_vars, cont_vars, cat_unique_levels = get_model_cat_cont_split_patsy_matrix(xdata, 
    #                                                                                                                                          tmle.cat_vars, tmle.cont_vars, tmle.cat_unique_levels)

    # model_cat_unique_levels

    # import statsmodels.api as sm
    # import statsmodels.formula.api as smf
    # data_to_fit = pooled_data_restricted.copy()
    # data_to_predict = tmle.df_restricted.copy()

    # gs_model = tmle._gs_measure_ + ' ~ ' + tmle._gs_model     # Setup the model form
    # # if self._gs_custom_ is None:                              # If no custom model provided
    # f = sm.families.family.Poisson()                      # ... GLM with Poisson family
    # treat_s_model = smf.glm(gs_model,                     # Estimate model
    #                         data_to_fit,                  # ... with data to fit
    #                         family=f).fit()               # ... and Poisson distribution
    # # if store_model:                                       # If estimating denominator
    # #     self._treatment_models.append(treat_s_model)      # ... store the model
    # pred = treat_s_model.predict(data_to_predict)         # Predicted values with data to predict

    # # # If verbose requested, provide model output
    # # if self._verbose_:
    # #     print('==============================================================================')
    # #     print(verbose_label+': '+self._gs_measure_)
    # #     print(treat_s_model.summary())

    # from scipy.stats import poisson, norm
    # pr_s = poisson.pmf(data_to_predict[tmle._gs_measure_], pred)

    # data_to_predict[tmle._gs_measure_]

    # # Example of target with class indices
    # loss = nn.CrossEntropyLoss()
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.empty(3, dtype=torch.long).random_(5)
    # output = loss(input, target)
    # output.backward()
    # # Example of target with class probabilities
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.randn(3, 5).softmax(dim=1)
    # output = loss(input, target)
    # output.backward()

    # input[0, 4]

    # tmp = target.unsqueeze(-1) #[num_samples, 1]
    # input[tmp]

    # ydata = pooled_data_restricted[tmle._gs_measure_] 

    # torch.index_select(input, 1, target)

    # a = torch.arange(12).view(3,4)
    # a
    # idx = torch.tensor([[1,3],[0,1],[2,3]])
    # a[torch.arange(a.size(0)).unsqueeze(1), idx]

    # torch.arange(a.size(0)).unsqueeze(1)

    # input[torch.arange(input.size(0)).unsqueeze(1), tmp]


    # loss = nn.BCEWithLogitsLoss()
    # input = torch.randn(3, requires_grad=True)
    # target = torch.empty(3).random_(2)
    # output = loss(input, target)
    # output.backward()

    # # BCEwithlogits requires target as float, but softmax requires class indicies as int/long

    # m = nn.Softmax(dim=-1)
    # input = torch.randn(2, 3)
    # output = m(input)
    # output

    
    # dummy_target = pooled_data_restricted[tmle._gs_measure_].to_numpy(dtype='int') 
    # dummy_target.shape
    # dummy_pred = np.random.randn(dummy_target.shape[0], 7)
    # dummy_pred.shape
    # pred_correct = dummy_pred[np.arange(dummy_pred.shape[0])[:, np.newaxis], dummy_target[:, np.newaxis]]
    # pred_correct.shape

    # type(pooled_data_restricted[tmle._gs_measure_])
    # type(dummy_target)

    import random
    import numpy as np
    import pandas as pd
    import networkx as nx
    from scipy.stats import logistic

    from beowulf.dgm.utils import (network_to_df, fast_exp_map, exposure_restrictions,
                                odds_to_probability, probability_to_odds)
    
    from beowulf import load_uniform_vaccine, load_random_vaccine

    n=500

    # uniform
    G, cat_vars, cont_vars, cat_unique_levels = load_uniform_vaccine(n=n, return_cat_cont_split=True)


    # params
    network = G
    restricted = False
    time_limit = 10
    inf_duration = 5
    update_split = True


    graph = network.copy()
    data = network_to_df(graph)

    adj_matrix = nx.adjacency_matrix(graph, weight=None)
    data['A_sum'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='sum')
    data['A_mean'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='mean')
    data['H_sum'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='sum')
    data = pd.merge(data, pd.DataFrame.from_dict(dict(network.degree),
                                                 orient='index').rename(columns={0: 'F'}),
                    how='left', left_index=True, right_index=True)

    # Running Data Generating Mechanism for A
    pr_a = logistic.cdf(-1.9 + 1.75*data['A'] + 1.*data['H']
                        + 1.*data['H_sum'] + 1.3*data['A_sum'] - 0.65*data['F'])
    vaccine = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['vaccine'] = vaccine
    if restricted:  # if we are in the restricted scenarios
        attrs = exposure_restrictions(network=network.graph['label'], exposure='vaccine',
                                      n=nx.number_of_nodes(graph))
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['vaccine']))

    # print("Pr(V):", np.mean(vaccine))
    for n in graph.nodes():
        graph.nodes[n]['vaccine'] = int(data.loc[data.index == n, 'vaccine'].values)
    
    # inside outbreak
    duration = inf_duration
    limit = time_limit
    # Adding node attributes
    for n, d in graph.nodes(data=True):
        d['D'] = 0
        d['R'] = 0
        d['t'] = 0

    # Selecting initial infections
    all_ids = [n for n in graph.nodes()]
    # infected = random.sample(all_ids, 5)
    if len(all_ids) <= 500:
        infected = [4, 36, 256, 305, 443]
    elif len(all_ids) == 1000:
        infected = [4, 36, 256, 305, 443, 552, 741, 803, 825, 946]
    elif len(all_ids) == 2000:
        infected = [4, 36, 256, 305, 443, 552, 741, 803, 825, 946,
                    1112, 1204, 1243, 1253, 1283, 1339, 1352, 1376, 1558, 1702]
    else:
        raise ValueError("Invalid network IDs")

    # Running through infection cycle
    graph_by_time = []
    time = 0
    while time < limit:  # Simulate outbreaks until time-step limit is reached
        time += 1
        for inf in sorted(infected, key=lambda _: random.random()):
            # Book-keeping for infected nodes
            graph.nodes[inf]['D'] = 1
            graph.nodes[inf]['t'] += 1
            if graph.nodes[inf]['t'] > duration:
                graph.nodes[inf]['I'] = 0         # Node is no longer infectious after this loop
                graph.nodes[inf]['R'] = 1         # Node switches to Recovered
                infected.remove(inf)

            # Attempt infections of neighbors
            for contact in nx.neighbors(graph, inf):
                if graph.nodes[contact]["D"] == 1:
                    pass
                else:
                    pr_y = logistic.cdf(- 2.5
                                        - 1.0*graph.nodes[contact]['vaccine']
                                        - 0.2*graph.nodes[inf]['vaccine']
                                        + 1.0*graph.nodes[contact]['A']
                                        - 0.2*graph.nodes[contact]['H'])
                    if np.random.binomial(n=1, p=pr_y, size=1):
                        graph.nodes[contact]['I'] = 1
                        graph.nodes[contact]["D"] = 1
                        infected.append(contact)
        graph_by_time.append(graph.copy()) # save all variables and nodes for each time point
    
    len(graph_by_time)
    graph_by_time[0]

    data_0 = network_to_df(graph_by_time[0])
    data_0
    data_1 = network_to_df(graph_by_time[1])
    data_1

    for i in range(len(graph_by_time)):
        print(network_to_df(graph_by_time[i]))

    tmp = network_to_df(graph_by_time[-1])
    tmp['I'].fillna(2., inplace=True)
    tmp

    pd.unique(network_to_df(graph_by_time[-1])['I'])
    pd.unique(network_to_df(graph_by_time[-1])['t'])
    pd.unique(network_to_df(graph_by_time[-1])['D'])
    pd.unique(network_to_df(graph_by_time[-1])['R'])
    pd.unique(network_to_df(graph_by_time[-1])['vaccine'])
    pd.unique(network_to_df(graph_by_time[-1])['A'])
    pd.unique(network_to_df(graph_by_time[-1])['H'])

    ''' 
    'D' represents during the time_limit, have this individual ever been infected;
    for 'I', representing infectious ability, where 1. means can infect others, 0. means lose the ability to infect others,
    Nan means never been infected and thus unknown, should be recoded to 2.
    '''