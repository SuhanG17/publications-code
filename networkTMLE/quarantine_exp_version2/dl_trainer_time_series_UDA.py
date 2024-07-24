'''
Deep Learning trainer with UDA mechanism
'''
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

from dl_dataset import get_dataloaders, get_kfold_split, get_predict_loader, TimeSeriesDatasetSeparateNormalize
from dl_model import MLPModelTimeSeriesNumericalUDA


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
            best_trg_label_loss = np.inf
            best_trg_label_acc = 0.
            for epoch in range(self.epochs):
                print(f'============================= Epoch {epoch + 1}: Training =============================')
                len_dataloader = min(len(src_train_loader), len(trg_train_loader))
                data_source_iter = iter(src_train_loader)
                data_target_iter = iter(trg_train_loader)
                train_src_recorder, train_trg_recorder = self.train_epoch(epoch, len_dataloader, data_source_iter, data_target_iter)
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
                test_src_recorder, test_trg_recorder = self.test_epoch(epoch, len_dataloader, data_source_iter, data_target_iter)
                print()

                print(f'Epoch {epoch + 1}:')
                print('train_src_label_acc: {:.3f}, test_src_label_acc: {:.3f}'.format(train_src_recorder['src_label_acc'], test_src_recorder['src_label_acc']))
                print('train_trg_label_acc: {:.3f}, test_trg_label_acc: {:.3f}'.format(train_trg_recorder['trg_label_acc'], test_trg_recorder['trg_label_acc']))
                print('train_src_domain_acc: {:.3f}, test_src_domain_acc: {:.3f}'.format(train_src_recorder['src_domain_acc'], test_src_recorder['src_domain_acc']))
                print('train_trg_domain_acc: {:.3f}, test_trg_domain_acc: {:.3f}'.format(train_trg_recorder['trg_domain_acc'], test_trg_recorder['trg_domain_acc']))

                if test_trg_recorder['trg_label_acc'] > best_trg_label_acc:
                    best_trg_label_acc = test_trg_recorder['trg_label_acc']
                    self._save_model(custom_path=custom_path)
                    print('Best model updated')

                # # naive save per epoch 
                # self._save_model(custom_path=custom_path)
                # print('Best model updated')
        if custom_path is None:
            return self.save_path
        else:
            return custom_path
                    

    def predict(self, xy_list, T_in_id=[*range(10)], T_out_id=[*range(10)], class_weight=None,
                adj_matrix_list=None, model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}, 
                n_output=2, _continuous_outcome=False, custom_path=None):
        print(f'============================= Predicting =============================')
        # initiate weights for class imbalance for loss functions
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

        src_recorder = {'src_label_loss': 0., 'src_label_acc': 0., 'src_domain_loss': 0., 'src_domain_acc': 0.}
        trg_recorder = {'trg_label_loss': 0., 'trg_label_acc': 0., 'trg_domain_loss': 0., 'trg_domain_acc': 0.}
        
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

            loss = src_loss_class + src_loss_domain + trg_loss_domain

            # backward + optimize
            loss.backward()
            self.optimizer.step()

            # calculate domain metrics (2/2)
            domain_metrics_1 = self._metrics(domain_outputs, domain_label, _continuous_outcome=False)

            # calculate metrics
            metrics = self._metrics_ts(class_outputs, src_y)
            metrics_trg = self._metrics_ts(trg_class_outputs, trg_y)                

            if i % self.print_every == 0:
                print(f'train epoch: [{epoch}], iter: [{i}/{len_dataloader}], alpha: {alpha:.3f}')
                print(f'src_loss_class: {src_loss_class.item():.3f}, src_loss_domain: {src_loss_domain.item():.3f}, trg_loss_domain: {trg_loss_domain.item():.3f}')   
                print(f'domain=0 acc: {domain_metrics_0:.3f}, domain=1 acc: {domain_metrics_1:.3f}, src_label acc: {metrics:.3f}')
                print(f'trg_label acc: {metrics_trg:.3f}, trg_loss_class: {trg_loss_class.item():.3f}')
            
            # update loss and metrics
            src_recorder['src_label_loss'] += src_loss_class.item()
            src_recorder['src_label_acc'] += metrics
            src_recorder['src_domain_loss'] += src_loss_domain.item()
            src_recorder['src_domain_acc'] += domain_metrics_0
            trg_recorder['trg_label_loss'] += trg_loss_class.item()
            trg_recorder['trg_label_acc'] += metrics_trg
            trg_recorder['trg_domain_loss'] += trg_loss_domain.item()
            trg_recorder['trg_domain_acc'] += domain_metrics_1
        
        src_recorder = {k: v / len_dataloader for k, v in src_recorder.items()}
        trg_recorder = {k: v / len_dataloader for k, v in trg_recorder.items()}

        return src_recorder, trg_recorder
            

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

        src_recorder = {'src_label_loss': 0., 'src_label_acc': 0., 'src_domain_loss': 0., 'src_domain_acc': 0.}
        trg_recorder = {'trg_label_loss': 0., 'trg_label_acc': 0., 'trg_domain_loss': 0., 'trg_domain_acc': 0.}
        
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

                loss = src_loss_class + src_loss_domain + trg_loss_domain

                # calculate domain metrics (2/2)
                domain_metrics_1 = self._metrics(domain_outputs, domain_label, _continuous_outcome=False)

                # calculate metrics
                metrics = self._metrics_ts(class_outputs, src_y)
                metrics_trg = self._metrics_ts(trg_class_outputs, trg_y)                

                if i % self.print_every == 0:
                    print(f'test epoch: [{epoch}], iter: [{i}/{len_dataloader}], alpha: {alpha:.3f}')
                    print(f'src_loss_class: {src_loss_class.item():.3f}, src_loss_domain: {src_loss_domain.item():.3f}, trg_loss_domain: {trg_loss_domain.item():.3f}')
                    print(f'domain=0 acc: {domain_metrics_0:.3f}, domain=1 acc: {domain_metrics_1:.3f}, src_label acc: {metrics:.3f}')
                    print(f'trg_label acc: {metrics_trg:.3f}, trg_loss_class: {trg_loss_class.item():.3f}')
                
                # update loss and metrics
                src_recorder['src_label_loss'] += src_loss_class.item()
                src_recorder['src_label_acc'] += metrics
                src_recorder['src_domain_loss'] += src_loss_domain.item()
                src_recorder['src_domain_acc'] += domain_metrics_0
                trg_recorder['trg_label_loss'] += trg_loss_class.item()
                trg_recorder['trg_label_acc'] += metrics_trg
                trg_recorder['trg_domain_loss'] += trg_loss_domain.item()
                trg_recorder['trg_domain_acc'] += domain_metrics_1
        
        src_recorder = {k: v / len_dataloader for k, v in src_recorder.items()}
        trg_recorder = {k: v / len_dataloader for k, v in trg_recorder.items()}
        return src_recorder, trg_recorder

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