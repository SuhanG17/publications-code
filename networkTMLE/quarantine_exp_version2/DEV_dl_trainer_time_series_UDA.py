'''
Deep Learning trainer with UDA mechanism
'''
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

from dl_dataset import TimeSeriesDataset, get_dataloaders, get_kfold_split, get_kfold_dataloaders, get_predict_loader
from dl_dataset import TimeSeriesDatasetSeparate, TimeSeriesDatasetSeparateNormalize, get_dataloaders, get_kfold_split, get_kfold_dataloaders, get_predict_loader
# from dl_models import MLPModelTimeSeries, GCNModelTimeSeries, CNNModelTimeSeries, MLPModelTimeSeriesNumerical
# from dl_trainer_time_series_UDA import AbstractMLTS
# from dl_models import MLPModelTimeSeriesNumericalUDA

# for model
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
            x = x.view(batch_size, temporal_dim*feature_dim)
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_x) # -> [batch_size, 2]
            return class_output, domain_output
        else:
            # class classifier
            class_output = self.class_classifier(x).permute(0, 2, 1) # -> [batch_size, n_output, T_in]
            # domain classifier
            batch_size, temporal_dim, feature_dim = x.shape
            # print(f'batch_size: {batch_size}, temporal_dim: {temporal_dim}, feature_dim: {feature_dim}')
            x = x.view(batch_size, temporal_dim*feature_dim) 
            reverse_x = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_x) # -> [batch_size, 2] 
            return class_output, domain_output 

batch_size = 8
num_cat_vars = 4
num_cont_vars = 5
T_in = 4
T_out = 1
dummy_x_cat = torch.randn(batch_size, num_cat_vars, T_in)
dummy_x_cont = torch.randn(batch_size, num_cont_vars, T_in)

tmp_net = MLPModelTimeSeriesNumericalUDA(adj_matrix_list=None, model_cat_unique_levels={key:1 for key in range(4)},
                                         n_cont=5, T_in=T_in, T_out=T_out, n_output=2, _continuous_outcome=False,
                                         lin_hidden=None,
                                         lin_hidden_temporal=None,
                                         class_classifier=None,
                                         domain_classifier=None)

class_outputs, domain_outputs = tmp_net(dummy_x_cat, dummy_x_cont, alpha=0.5)

class_outputs.shape
domain_outputs.shape



class MLPTS_UDA:
    def __init__(self, split_ratio, batch_size, shuffle, n_splits, predict_all,
                 epochs, print_every, device='cpu', save_path='./',
                 lin_hidden=None, lin_hidden_temporal=None, class_classifier=None, domain_classifier=None):
        super(MLPTS_UDA, self).__init__()
        # set up epochs and save path
        self.epochs = epochs
        self.save_path = save_path
        # set up dataset related params
        self.split_ratio, self.batch_size, self.shuffle, self.n_splits, self.predict_all = split_ratio, batch_size, shuffle, n_splits, predict_all
        # set up log params
        self.print_every = print_every
        self.print_per_time_slice_metrics = False # use to check if all time slices have similar performance
        # set up device
        self.device = device
        # set up model params
        self.lin_hidden = lin_hidden
        self.lin_hidden_temporal = lin_hidden_temporal
        self.class_classifier = class_classifier
        self.domain_classifier = domain_classifier

    def fit(self, src_xy_list, trg_xy_list, T_in_id=[*range(10)], T_out_id=[*range(10)], class_weight=None,
            adj_matrix_list=None, model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}, 
            n_output=2, _continuous_outcome=False, custom_path=None):
        
        # calculate T_in and T_out for _build_model()
        self.T_in = len(T_in_id)
        self.T_out = len(T_out_id)

        # initiate best model
        self.best_model = None
        self.best_loss = np.inf        

        # instantiate model
        self.n_output = n_output
        self._continuous_outcome = _continuous_outcome
        self.model = self._build_model(adj_matrix_list, model_cat_vars, model_cont_vars, model_cat_unique_levels,
                                       n_output, _continuous_outcome).to(self.device)
        self.optimizer = self._optimizer()
        self.criterion_class = self._loss_fn(_continuous_outcome=_continuous_outcome, class_weight=class_weight)
        self.criterion_domain = self._loss_fn(_continuous_outcome=False, class_weight=None)
        
        if self.n_splits > 1:
            raise NotImplementedError('Kfold cross validation is not implemented yet.')
        else:
            src_train_loader, src_valid_loader, src_test_loader = self._data_preprocess(src_xy_list,
                                                                                        model_cat_vars=model_cat_vars, 
                                                                                        model_cont_vars=model_cont_vars, 
                                                                                        model_cat_unique_levels=model_cat_unique_levels,
                                                                                        normalize=True,
                                                                                        drop_duplicates=False,
                                                                                        T_in_id=T_in_id, T_out_id=T_out_id)
            trg_train_loader, trg_valid_loader, trg_test_loader = self._data_preprocess(trg_xy_list,
                                                                                        model_cat_vars=model_cat_vars,
                                                                                        model_cont_vars=model_cont_vars,
                                                                                        model_cat_unique_levels=model_cat_unique_levels,
                                                                                        normalize=True,
                                                                                        drop_duplicates=False,
                                                                                        T_in_id=T_in_id, T_out_id=T_out_id)
            best_loss = np.inf
            for epoch in range(self.epochs):
                print(f'============================= Epoch {epoch + 1}: Training =============================')
                len_dataloader = min(len(src_train_loader), len(trg_train_loader))
                data_source_iter = iter(src_train_loader)
                data_target_iter = iter(trg_train_loader)
                train_trg_label_loss = self.train_epoch(epoch, len_dataloader, data_source_iter, data_target_iter)
                print()
                
                if src_valid_loader is not None:
                    print(f'============================= Epoch {epoch + 1}: Validation =============================')
                    len_dataloader = min(len(src_valid_loader), len(trg_valid_loader))
                    data_source_iter = iter(src_valid_loader)
                    data_target_iter = iter(trg_valid_loader)
                    self.valid_epoch(epoch, len_dataloader, data_source_iter, data_target_iter)
                    print()
            
                print(f'============================= Epoch {epoch + 1}: Test =============================')
                len_dataloader = min(len(src_test_loader), len(trg_test_loader))
                data_source_iter = iter(src_test_loader)
                data_target_iter = iter(trg_test_loader)
                test_trg_label_loss = self.test_epoch(epoch, len_dataloader, data_source_iter, data_target_iter)
                print()

                print(f'Epoch {epoch + 1}: train_trg_label_loss: {train_trg_label_loss:.3f}, test_trg_label_loss: {test_trg_label_loss:.3f}')

                if test_trg_label_loss < best_loss:
                    best_loss = test_trg_label_loss
                    # self._save_model(custom_path=custom_path)
                    # print('Best model updated')
                
                self._save_model(custom_path=custom_path)
                print('Best model updated')
        if custom_path is None:
            return self.save_path
        else:
            return custom_path
                    

    def predict(self, xy_list, T_in_id=[*range(10)], T_out_id=[*range(10)], class_weight=None,
                adj_matrix_list=None, model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}, 
                n_output=2, _continuous_outcome=False, custom_path=None):
        print(f'============================= Predicting =============================')
        # initiate weights for class imbalance for loss functions
        self.pos_weight = pos_weight
        self.class_weight = class_weight 

        # calculate T_in and T_out for _build_model()
        self.T_in = len(T_in_id)
        self.T_out = len(T_out_id)

        # instantiate model
        self.n_output = n_output
        self._continuous_outcome = _continuous_outcome

        self.model = self._build_model(adj_matrix_list, model_cat_vars, model_cont_vars, model_cat_unique_levels,
                                       n_output, _continuous_outcome).to(self.device)
        self._load_model(custom_path)
        self.criterion_class = self._loss_fn(_continuous_outcome=_continuous_outcome, class_weight=class_weight)
        # self.criterion_domain = self._loss_fn(_continuous_outcome=False, class_weight=None)

        if self.predict_all:
            dset = TimeSeriesDatasetSeparateNormalize(xy_list,
                                                      model_cat_vars=model_cat_vars, 
                                                      model_cont_vars=model_cont_vars, 
                                                      model_cat_unique_levels=model_cat_unique_levels,
                                                      normalize=True,
                                                      drop_duplicates=False,
                                                      T_in_id=T_in_id, T_out_id=T_out_id)
            test_loader = get_predict_loader(dset, self.batch_size)
        else:
            _, _, test_loader = self._data_preprocess(xy_list,
                                                      model_cat_vars=model_cat_vars, 
                                                      model_cont_vars=model_cont_vars, 
                                                      model_cat_unique_levels=model_cat_unique_levels,
                                                      normalize=True,
                                                      drop_duplicates=False,
                                                      T_in_id=T_in_id, T_out_id=T_out_id)

        self.model.eval()
        alpha = 0
        running_acc = 0.
        pred_list = []
        test_iter = iter(test_loader) 
        with torch.no_grad():
            for i in range(len(test_loader)):
                data = test_iter.next()
                x_cat, x_cont, y, sample_idx = data
                x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device)
                sample_idx = sample_idx.to(self.device)
                class_outputs, _ = self.model(x_cat, x_cont, sample_idx, alpha)
                y = self._reshape_target(y)
                loss = self.criterion_class(class_outputs, y)
                metrics = self._metrics_ts(class_outputs, y)
                running_acc += metrics
                print(f'predict iter: [{i}/{len(test_loader)}], label loss: {loss:.3f}, label acc: {metrics:.3f}', flush=True)

                if self._continuous_outcome:
                    pred_list.append(class_outputs.detach().to('cpu').numpy())
                else:
                    pred_list.append(torch.softmax(class_outputs, dim=1).detach().to('cpu').numpy())    
            print(f'overall acc: {running_acc/len(test_loader):.3f}', flush=True)    
        return pred_list

    def train_epoch(self, epoch, len_dataloader, data_source_iter, data_target_iter):
        self.model.train() # turn on train-mode

        trg_label_loss = 0.
        
        for i in range(len_dataloader):

            p = float(i + epoch * len_dataloader) / self.epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            data_source = data_source_iter.next()
            src_x_cat, src_x_cont, src_y, src_sample_idx = data_source
            # send to device
            src_x_cat, src_x_cont, src_y = src_x_cat.to(self.device), src_x_cont.to(self.device), src_y.to(self.device)
            src_sample_idx = src_sample_idx.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # set up domain label
            batch_size = src_y.shape[0]
            domain_label = torch.zeros(batch_size).long()
            domain_label = domain_label.to(self.device)
            
            # forward_pass(1/2)
            class_outputs, domain_outputs = self.model(src_x_cat, src_x_cont, src_sample_idx, alpha) # shape [batch_size, num_classes, T_out] 
            src_y = self._reshape_target(src_y)

            src_loss_class = self.criterion_class(class_outputs, src_y)
            src_loss_domain = self.criterion_domain(domain_outputs, domain_label)

            # calculate domain metrics (1/2)
            domain_metrics_0 = self._metrics(domain_outputs, domain_label, _continuous_outcome=False)

            # training model using target data
            data_target = data_target_iter.next()
            trg_x_cat, trg_x_cont, trg_y, trg_sample_x = data_target
            # send to device
            trg_x_cat, trg_x_cont, trg_y = trg_x_cat.to(self.device), trg_x_cont.to(self.device), trg_y.to(self.device)
            trg_sample_x = trg_sample_x.to(self.device)

            # set up domain label
            domain_label = torch.ones(batch_size).long()
            domain_label = domain_label.to(self.device)
            
            # forward_pass(2/2)
            # _, domain_outputs = self.model(trg_x_cat, trg_x_cont, trg_sample_x, alpha)
            # trg_loss_domain = self.criterion_domain(domain_outputs, domain_label)
            # loss = src_loss_class + src_loss_domain + trg_loss_domain

            trg_class_outputs, domain_outputs = self.model(trg_x_cat, trg_x_cont, trg_sample_x, alpha)
            trg_y = self._reshape_target(trg_y)
            trg_loss_class = self.criterion_class(trg_class_outputs, trg_y)
            trg_loss_domain = self.criterion_domain(domain_outputs, domain_label)
            trg_label_loss += trg_loss_class # TODO

            loss = src_loss_class + src_loss_domain + trg_loss_domain

            # backward + optimize
            loss.backward()
            self.optimizer.step()

            # calculate domain metrics (2/2)
            domain_metrics_1 = self._metrics(domain_outputs, domain_label, _continuous_outcome=False)

            # calculate metrics
            metrics = self._metrics_ts(class_outputs, src_y)
            metrics_trg = self._metrics_ts(trg_class_outputs, trg_y)                

            print(f'train epoch: [{epoch}], iter: [{i}/{len_dataloader}], alpha: {alpha:.3f}')
            print(f'src_loss_class: {src_loss_class.item():.3f}, src_loss_domain: {src_loss_domain.item():.3f}, trg_loss_domain: {trg_loss_domain.item():.3f}')   
            print(f'domain=0 acc: {domain_metrics_0:.3f}, domain=1 acc: {domain_metrics_1:.3f}, src_label acc: {metrics:.3f}')
            print(f'trg_label acc: {metrics_trg:.3f}, trg_loss_class: {trg_loss_class.item():.3f}')
        return trg_label_loss / len_dataloader
            

    def valid_epoch(self, epoch, len_dataloader, data_source_iter, data_target_iter):
        self.model.eval() # turn on eval-mode
        alpha = 0
        
        with torch.no_grad():
            for i in range(len_dataloader):
                # training model using source data
                data_source = data_source_iter.next()
                src_x_cat, src_x_cont, src_y, src_sample_idx = data_source
                # send to device
                src_x_cat, src_x_cont, src_y = src_x_cat.to(self.device), src_x_cont.to(self.device), src_y.to(self.device)
                src_sample_idx = src_sample_idx.to(self.device)

                # set up domain label
                batch_size = src_y.shape[0]
                domain_label = torch.zeros(batch_size).long()
                domain_label = domain_label.to(self.device)

                # forward_pass(1/2)
                class_outputs, domain_outputs = self.model(src_x_cat, src_x_cont, src_sample_idx, alpha) # shape [batch_size, num_classes, T_out] 
                src_y = self._reshape_target(src_y)

                src_loss_class = self.criterion_class(class_outputs, src_y)
                src_loss_domain = self.criterion_domain(domain_outputs, domain_label)

                # calculate domain metrics (1/2)
                domain_metrics_0 = self._metrics(domain_outputs, domain_label, _continuous_outcome=False)

                # training model using target data
                data_target = data_target_iter.next()
                trg_x_cat, trg_x_cont, trg_y, trg_sample_x = data_target
                # send to device
                trg_x_cat, trg_x_cont, trg_y = trg_x_cat.to(self.device), trg_x_cont.to(self.device), trg_y.to(self.device)
                trg_sample_x = trg_sample_x.to(self.device)

                # set up domain label
                domain_label = torch.ones(batch_size).long()
                domain_label = domain_label.to(self.device)
                
                # forward_pass(2/2)
                # _, domain_outputs = self.model(trg_x_cat, trg_x_cont, trg_sample_x, alpha)
                # trg_loss_domain = self.criterion_domain(domain_outputs, domain_label)
                # loss = src_loss_class + src_loss_domain + trg_loss_domain

                trg_class_outputs, domain_outputs = self.model(trg_x_cat, trg_x_cont, trg_sample_x, alpha)
                trg_y = self._reshape_target(trg_y)

                trg_loss_class = self.criterion_class(trg_class_outputs, trg_y)
                trg_loss_domain = self.criterion_domain(domain_outputs, domain_label)
                loss = src_loss_class + src_loss_domain + trg_loss_domain

                # calculate domain metrics (2/2)
                domain_metrics_1 = self._metrics(domain_outputs, domain_label, _continuous_outcome=False)

                # calculate metrics
                metrics = self._metrics_ts(class_outputs, src_y)                
                metrics_trg = self._metrics_ts(trg_class_outputs, trg_y)

                print(f'val epoch: [{epoch}], iter: [{i}/{len_dataloader}]')
                print(f'src_loss_class: {src_loss_class.item():.3f}, src_loss_domain: {src_loss_domain.item():.3f}, trg_loss_domain: {trg_loss_domain.item():.3f}')
                print(f'domain=0 acc: {domain_metrics_0:.3f}, domain=1 acc: {domain_metrics_1:.3f}, src label acc: {metrics:.3f}')
                print(f'trg_label acc: {metrics_trg:.3f}, trg_loss_class: {trg_loss_class.item():.3f}')

    def test_epoch(self, epoch, len_dataloader, data_source_iter, data_target_iter):
        self.model.eval() # turn on eval-mode
        alpha = 0

        trg_label_loss = 0.
        
        with torch.no_grad():
            for i in range(len_dataloader):
                # training model using source data
                data_source = data_source_iter.next()
                src_x_cat, src_x_cont, src_y, src_sample_idx = data_source
                # send to device
                src_x_cat, src_x_cont, src_y = src_x_cat.to(self.device), src_x_cont.to(self.device), src_y.to(self.device)
                src_sample_idx = src_sample_idx.to(self.device)

                # set up domain label
                batch_size = src_y.shape[0]
                domain_label = torch.zeros(batch_size).long()
                domain_label = domain_label.to(self.device)

                # forward_pass(1/2)
                class_outputs, domain_outputs = self.model(src_x_cat, src_x_cont, src_sample_idx, alpha) # shape [batch_size, num_classes, T_out] 
                src_y = self._reshape_target(src_y)

                src_loss_class = self.criterion_class(class_outputs, src_y)
                src_loss_domain = self.criterion_domain(domain_outputs, domain_label)

                # calculate domain metrics (1/2)
                domain_metrics_0 = self._metrics(domain_outputs, domain_label, _continuous_outcome=False)

                # training model using target data
                data_target = data_target_iter.next()
                trg_x_cat, trg_x_cont, trg_y, trg_sample_x = data_target
                # send to device
                trg_x_cat, trg_x_cont, trg_y = trg_x_cat.to(self.device), trg_x_cont.to(self.device), trg_y.to(self.device)
                trg_sample_x = trg_sample_x.to(self.device)

                # set up domain label
                domain_label = torch.ones(batch_size).long()
                domain_label = domain_label.to(self.device)
                
                # forward_pass(2/2)
                trg_class_outputs, domain_outputs = self.model(trg_x_cat, trg_x_cont, trg_sample_x, alpha)
                trg_y = self._reshape_target(trg_y)

                trg_loss_class = self.criterion_class(trg_class_outputs, trg_y)
                trg_loss_domain = self.criterion_domain(domain_outputs, domain_label)

                trg_label_loss += trg_loss_class # TODO

                loss = src_loss_class + src_loss_domain + trg_loss_domain

                # calculate domain metrics (2/2)
                domain_metrics_1 = self._metrics(domain_outputs, domain_label, _continuous_outcome=False)

                # calculate metrics
                metrics = self._metrics_ts(class_outputs, src_y)
                metrics_trg = self._metrics_ts(trg_class_outputs, trg_y)                

                print(f'test epoch: [{epoch}], iter: [{i}/{len_dataloader}]')
                print(f'src_loss_class: {src_loss_class.item():.3f}, src_loss_domain: {src_loss_domain.item():.3f}, trg_loss_domain: {trg_loss_domain.item():.3f}')
                print(f'domain=0 acc: {domain_metrics_0:.3f}, domain=1 acc: {domain_metrics_1:.3f}, src_label acc: {metrics:.3f}')
                print(f'trg_label acc: {metrics_trg:.3f}, trg_loss_class: {trg_loss_class.item():.3f}')
        return trg_label_loss / len_dataloader

    def _build_model(self, adj_matrix_list, model_cat_vars, model_cont_vars, model_cat_unique_levels,
                     n_output, _continuous_outcome):
        n_cont = len(model_cont_vars)
        net = MLPModelTimeSeriesNumericalUDA(adj_matrix_list, model_cat_unique_levels, n_cont, self.T_in, self.T_out,
                                             n_output, _continuous_outcome,
                                             self.lin_hidden, self.lin_hidden_temporal, 
                                             self.class_classifier, self.domain_classifier)
        if (self.device != 'cpu') and (torch.cuda.device_count() > 1):
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
            net = net.to(self.device)
        return net

    def _data_preprocess(self, xy_list, model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={},
                        normalize=True, drop_duplicates=True, T_in_id=[*range(10)], T_out_id=[*range(10)]):
        ts_dset = TimeSeriesDatasetSeparateNormalize(xy_list=xy_list,
                                                     model_cat_vars=model_cat_vars, 
                                                     model_cont_vars=model_cont_vars, 
                                                     model_cat_unique_levels=model_cat_unique_levels,
                                                     normalize=normalize,
                                                     drop_duplicates=drop_duplicates,
                                                     T_in_id=T_in_id, T_out_id=T_out_id)

        if self.n_splits > 1: # Kfold cross validation is used
            return get_kfold_split(n_splits=self.n_splits, shuffle=self.shuffle), ts_dset
        else:
            train_loader, valid_loader, test_loader = get_dataloaders(ts_dset,
                                                                      split_ratio=self.split_ratio, 
                                                                      batch_size=self.batch_size,
                                                                      shuffle=self.shuffle)           
            return train_loader, valid_loader, test_loader

    def _optimizer(self):
        # return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        return optim.Adam(self.model.parameters(), lr=0.0001)

    def _reshape_target(self, y):
        if self._continuous_outcome:
            if y.dim() == 2:
                y = y.float().unsqueeze(1) # shape [batch_size, T_out] -> [batch_size, 1, T_out] for l1_loss
            elif y.dim() == 1:
                y = y.float().unsqueeze(-1).unsqueeze(-1) # shape [batch_size] -> [batch_size, 1, 1] for l1_loss
        else:
            # CrossEntropyLoss requires target (class indicies) as int
            if y.dim() == 2:
                y = y.long() # shape [batch_size, T_out]
            elif y.dim() == 1:
                y = y.long().unsqueeze(-1) # shape [batch_size] -> [batch_size, 1]
        return y
    
    def _loss_fn(self, _continuous_outcome=False, class_weight=None):
        if _continuous_outcome:
            return nn.L1Loss() # mae
            # return nn.MSELoss() # mse
        else:
            # multi-class classification: no need for softmax layer, require [n_output] output for classification
            if class_weight is not None:
                return nn.CrossEntropyLoss(weight=torch.from_numpy(self.class_weight).to(self.device)) # weight to correct for class-imbalance
            else:
                return nn.CrossEntropyLoss()

    def _metrics(self, outputs, labels, _continuous_outcome=False):
        ''' calculate metrics for each time slice:
        _continuous_outcome:
            outputs: shape [batch_size, 1]
            labels:  shape [batch_size, 1]
        else:
            outputs: shape [batch_size, num_classes]
            labels:  shape [batch_size]
        '''
        if _continuous_outcome:
            mae_error = torch.abs(outputs - labels).mean()
            mse_error = torch.pow(outputs - labels, 2).mean()
            rmse_error = torch.sqrt(mse_error)
            return {'mae':mae_error.item(), 'mse':mse_error.item(), 'rmse':rmse_error.item()}
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
            outputs: shape [batch_size, num_classes, T]
            labels:  shape [batch_size, T]
        '''
        if self._continuous_outcome:
            mae_error = torch.abs(outputs - labels).mean()
            mse_error = torch.pow(outputs - labels, 2).mean()
            rmse_error = torch.sqrt(mse_error)
            return {'mae':mae_error.item(), 'mse':mse_error.item(), 'rmse':rmse_error.item()}
        else:
            # the class with the highest energy is what we choose as prediction, if output 2 categories for binary classificaiton
            _, predicted = torch.max(outputs.data, 1) # [batch_size, num_classes, T] -> [batch_size, T]
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


                         
from tmle_utils import get_patsy_for_model_w_C, get_final_model_cat_cont_split, get_model_cat_cont_split_patsy_matrix, all_equal, get_imbalance_weights
import patsy

def get_pred_df(model_string, model_outcome, 
                df_list, T_in_id, T_out_id, cat_vars, cont_vars, cat_unique_levels):
    
    # slice data
    xdata_list = []
    ydata_list = []
    n_output_list = []
    for df_restricted in df_list:
        if 'C(' in model_string:
            xdata_list.append(get_patsy_for_model_w_C(model_string, df_restricted))
        else:
            xdata_list.append(patsy.dmatrix(model_string + ' - 1', df_restricted, return_type="dataframe"))
        ydata_list.append(df_restricted[model_outcome])
        n_output_list.append(pd.unique(df_restricted[model_outcome]).shape[0])
    
    # slicing xdata and ydata list
    xdata_list, ydata_list = [xdata_list[i] for i in T_in_id], [ydata_list[i] for i in T_out_id]
    # T_in, T_out = len(xdata_list), len(ydata_list)

    # Re-arrange data
    model_cat_vars_list = []
    model_cont_vars_list = []
    model_cat_unique_levels_list = []

    cat_vars_list = []
    cont_vars_list = []
    cat_unique_levels_list = []

    # deep_learner_df_list = []
    ydata_array_list = []

    for xdata in xdata_list:
        model_cat_vars, model_cont_vars, model_cat_unique_levels, cat_vars, cont_vars, cat_unique_levels = get_model_cat_cont_split_patsy_matrix(xdata, 
                                                                                                                                                 cat_vars, cont_vars, cat_unique_levels)
        model_cat_vars_list.append(model_cat_vars)
        model_cont_vars_list.append(model_cont_vars)
        model_cat_unique_levels_list.append(model_cat_unique_levels)

        cat_vars_list.append(cat_vars)
        cont_vars_list.append(cont_vars)
        cat_unique_levels_list.append(cat_unique_levels)

    for ydata in ydata_list:
        ydata_array_list.append(ydata.to_numpy()) # convert pd.series to np.array

    model_cat_vars_final, model_cont_vars_final, model_cat_unique_levels_final = get_final_model_cat_cont_split(model_cat_vars_list, model_cont_vars_list, model_cat_unique_levels_list)

    ## check if n_output is consistent 
    if not all_equal(n_output_list):
        raise ValueError("n_output are not identical throughout time slices")
    else:
        n_output_final = n_output_list[-1]    

    ## set weight against class imbalance
    dummy_n_output_final = 3 # set up to use class weights
    pos_weight, class_weight = get_imbalance_weights(dummy_n_output_final, ydata_array_list, use_last_time_slice=True,
                                                     imbalance_threshold=3., imbalance_upper_bound=3.2, default_lock=False)
    # pos_weight, class_weight = None, None

    return [xdata_list, ydata_array_list], [model_cat_vars_final, model_cont_vars_final, model_cat_unique_levels_final], [pos_weight, class_weight, n_output_final]
    

    
# model string
qn_model = "quarantine + quarantine_sum + I_ratio + I_ratio_sum + A + H + A_sum + H_sum + degree"
outcome = 'D'

# T_in and T_out id
T_in_id = [6, 7, 8, 9]
# T_in_id = [8, 9]
T_out_id = [9]

# Raw Data
df_restricted_list = torch.load('tmp_pt/'+'df_restricted_list.pt')
pooled_data_restricted_list = torch.load('tmp_pt/'+'pooled_data_restricted_list.pt')

# Raw dict
cat_vars = torch.load('tmp_pt/'+'cat_vars.pt')
cont_vars = torch.load('tmp_pt/'+'cont_vars.pt')
cat_unique_levels = torch.load('tmp_pt/'+'cat_unique_levels.pt')

aa, bb, cc = get_pred_df(qn_model, outcome, df_restricted_list, T_in_id, T_out_id, cat_vars, cont_vars, cat_unique_levels)
src_xdata_list, src_ydata_list = aa
model_cat_vars_final, model_cont_vars_final, model_cat_unique_levels_final = bb
pos_weight, class_weight, n_output_final = cc

aa, bb, cc = get_pred_df(qn_model, outcome, pooled_data_restricted_list, T_in_id, T_out_id, cat_vars, cont_vars, cat_unique_levels)
trg_xdata_list, trg_ydata_list = aa


# initiate deep learning model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)

split_ratio = [0.8, 0.2]
deep_learner = MLPTS_UDA(split_ratio=split_ratio, batch_size=16, shuffle=True, n_splits=1, predict_all=True,
                         epochs=10, print_every=5, device=device, save_path='./tmp.pth',
                         lin_hidden=None, 
                         lin_hidden_temporal=nn.ModuleList([nn.Linear(128, 256), nn.Linear(256, 128)]), 
                         class_classifier=nn.Sequential(nn.Linear(32, 16), nn.ReLU(),
                                                        nn.Linear(16, 8), nn.ReLU(),
                                                        nn.Linear(8, 2)),
                         domain_classifier=nn.Sequential(nn.Linear(1*32, 16), nn.ReLU(), 
                                                         nn.Linear(16, 8), nn.ReLU(), 
                                                         nn.Linear(8, 2)))

# deep_learner = MLPTS_UDA(split_ratio=split_ratio, batch_size=16, shuffle=True, n_splits=1, predict_all=True,
#                          epochs=25, print_every=5, device=device, save_path='./tmp.pth',
#                          lin_hidden=None, 
#                          lin_hidden_temporal=None, 
#                          class_classifier=None,
#                          domain_classifier=None)

path_to_model = deep_learner.fit(src_xy_list=[src_xdata_list, src_ydata_list], 
                                trg_xy_list=[trg_xdata_list, trg_ydata_list], 
                                T_in_id=T_in_id, T_out_id=T_out_id, class_weight=None,
                                adj_matrix_list=None, 
                                model_cat_vars=model_cat_vars_final, model_cont_vars=model_cont_vars_final, model_cat_unique_levels=model_cat_unique_levels_final, 
                                n_output=2, _continuous_outcome=False, custom_path=None)

pred = deep_learner.predict(xy_list=[src_xdata_list, src_ydata_list],
                            T_in_id=T_in_id, T_out_id=T_out_id, class_weight=None,
                            adj_matrix_list=None, 
                            model_cat_vars=model_cat_vars_final, model_cont_vars=model_cont_vars_final, model_cat_unique_levels=model_cat_unique_levels_final, 
                            n_output=2, _continuous_outcome=False, custom_path=None)

pred = deep_learner.predict(xy_list=[trg_xdata_list, trg_ydata_list],
                            T_in_id=T_in_id, T_out_id=T_out_id, class_weight=None,
                            adj_matrix_list=None, 
                            model_cat_vars=model_cat_vars_final, model_cont_vars=model_cont_vars_final, model_cat_unique_levels=model_cat_unique_levels_final, 
                            n_output=2, _continuous_outcome=False, custom_path=None)

deep_learner.model

epochs = 10
len_dataloader = 25
for epoch in range(epochs):
    for i in range(len_dataloader):
        p = float(i + epoch * len_dataloader) / epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        print(f'epoch: {epoch}, iter: {i}, p: {p:.3f}, alpha: {alpha:.3f}')

