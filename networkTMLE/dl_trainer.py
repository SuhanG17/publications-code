import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

from dl_dataset import DfDataset, get_dataloaders, get_kfold_split, get_kfold_dataloaders, get_predict_loader
from dl_models import MLPModel

######################## ml abstract trainer (Parent) ########################
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

        # instantiation requires df and target, move insides fit() and predict()
        # self.model = self._build_model().to(self.device) 
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


######################## MLP model trainer (Child) ########################
class MLP(AbstractML):
    def __init__(self, split_ratio, batch_size, shuffle, n_splits, predict_all,
                 epochs, print_every, device='cpu', save_path='./'):
        super().__init__(split_ratio, batch_size, shuffle, n_splits, predict_all,
                         epochs, print_every, device, save_path)

    def _build_model(self, model_cat_vars, model_cont_vars, model_cat_unique_levels):
        n_cont = len(model_cont_vars)
        net = MLPModel(model_cat_unique_levels, n_cont) 
        if torch.cuda.device_count()>1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
            net = net.to(self.device)
        return net


if __name__ == '__main__':

    from tmle_dl import NetworkTMLE  # modfied library
    # from amonhen import NetworkTMLE   # internal version, recommended to use library above instead
    from beowulf import (sofrygin_observational, generate_sofrygin_network,
                        load_uniform_statin, load_random_naloxone, load_uniform_diet, load_random_vaccine)
    from beowulf.dgm import statin_dgm, naloxone_dgm, diet_dgm, vaccine_dgm
   
    # random network with reproducibility
    torch.manual_seed(17) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Statin-ASCVD -- DGM: test run

    # Loading uniform network with statin W
    G, cat_vars, cont_vars, cat_unique_levels = load_uniform_statin(n=500, return_cat_cont_split=True)
    # Simulation single instance of exposure and outcome
    H, cat_vars, cont_vars, cat_unique_levels = statin_dgm(network=G, restricted=False,
                                                           update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)

    # network-TMLE applies to generated data
    tmle = NetworkTMLE(H, exposure='statin', outcome='cvd',
                       cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
                       use_deep_learner_A_i=True) 
    
    # instantiation of MLP model
    # 5 fold cross validation 
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    mlp_learner = MLP(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
                      epochs=10, print_every=5, device=device, save_path='./tmp.pth')

    tmle.exposure_model("L + A_30 + R_1 + R_2 + R_3", custom_model=mlp_learner)
    # tmle.exposure_model("L + A_30 + R_1 + R_2 + R_3")
    tmle.exposure_map_model("statin + L + A_30 + R_1 + R_2 + R_3",
                            measure='sum', distribution='poisson')  # Applying a Poisson model
    tmle.outcome_model("statin + statin_sum + A_sqrt + R + L")
    tmle.fit(p=0.35, bound=0.01)
    tmle.summary()