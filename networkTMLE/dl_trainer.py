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
        self.save_path = save_path

        self.split_ratio, self.batch_size, self.shuffle, self.n_splits, self.predict_all = split_ratio, batch_size, shuffle, n_splits, predict_all
    
        self.print_every = print_every
        self.device = device

    def fit(self, df, target, model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}, n_output=2, custom_path=None):
        # initiate best model
        self.best_model = None
        self.best_loss = np.inf        

        # instantiate model
        self.n_output = n_output
        self.model = self._build_model(model_cat_vars, model_cont_vars, model_cat_unique_levels, n_output).to(self.device)
        self.optimizer = self._optimizer()
        self.criterion = self._loss_fn()

        # target is exposure for nuisance models, outcome for outcome model
        fold_record = {'train_loss': [], 'val_loss': [],'train_acc':[],'val_acc':[]}

        if self.n_splits > 1: # Kfold cross validation is used
            splits, dset = self._data_preprocess(df, target,
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
            self.train_loader, self.valid_loader, self.test_loader = self._data_preprocess(df, target,
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

    def predict(self, df, target, model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}, n_output=2, custom_path=None):
        print(f'============================= Predicting =============================')
        # instantiate model
        self.n_output = n_output
        self.model = self._build_model(model_cat_vars, model_cont_vars, model_cat_unique_levels, n_output).to(self.device)
        self._load_model(custom_path)
        self.criterion = self._loss_fn()

        if self.predict_all:
            dset = DfDataset(df, target=target,
                             model_cat_vars=model_cat_vars, 
                             model_cont_vars=model_cont_vars, 
                             model_cat_unique_levels=model_cat_unique_levels)
            self.test_loader = get_predict_loader(dset, self.batch_size)
        else:
            _, _, self.test_loader = self._data_preprocess(df, target,
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
            if self.n_output == 2: # binary classification
                # BCEWithLogitsLoss requires target as float, same size as outputs
                y = y.float() 
            else:
                # CrossEntropyLoss requires target (class indicies) as int, shape [batch_size]
                y = y.long().squeeze(-1) 
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
                if self.n_output == 2: # binary classification
                    # BCEWithLogitsLoss requires target as float, same size as outputs
                    y = y.float() 
                else:
                    # CrossEntropyLoss requires target (class indicies) as int, shape [batch_size]
                    y = y.long().squeeze(-1) 
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
                outputs = self.model(x_cat, x_cont)
                if return_pred:
                    if self.n_output == 2:
                        pred_list.append(torch.sigmoid(outputs).detach().to('cpu').numpy())
                    else:
                        pred_list.append(torch.softmax(outputs, dim=-1).detach().to('cpu').numpy())
                
                if self.n_output == 2: # binary classification
                    # BCEWithLogitsLoss requires target as float, same size as outputs
                    y = y.float() 
                else:
                    # CrossEntropyLoss requires target (class indicies) as int, shape [batch_size]
                    y = y.long().squeeze(-1) 
                loss = self.criterion(outputs, y)
                metrics = self._metrics(outputs, y)

                # print statistics
                cumu_loss += loss.item()
                cumu_metrics += metrics

            if return_pred: 
                print(f'[data_to_predict averaged] | loss: {cumu_loss / len(self.test_loader):.3f} | acc: {cumu_metrics / len(self.test_loader):.3f}') 
            else:
                print(f'[{epoch + 1}, {i + 1:5d}] | loss: {cumu_loss / len(self.test_loader):.3f} | acc: {cumu_metrics / len(self.test_loader):.3f}')
                
            # for metric_name, metric_value in cumu_metrics.items():
            #     print(f'[{epoch + 1}, {i + 1:5d}] {metric_name}: {metric_value / len(self.test_loader):.3f}')

        if return_pred:
            return pred_list
        else:
            return cumu_loss / len(self.test_loader), cumu_metrics / len(self.test_loader)

    def _build_model(self):
        pass 

    def _data_preprocess(self, df, target=None,
                         model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}):
        dset = DfDataset(df, target=target,
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
        if self.n_output == 2: # binary classification
            return nn.BCEWithLogitsLoss() # no need for sigmoid, require 1 output for binary classfication
        else:
            return nn.CrossEntropyLoss() # no need for softmax, require [n_output] output for classification

    def _metrics(self, outputs, labels):
        if self.n_output == 2:
            pred = torch.sigmoid(outputs) # get binary probability
            pred_binary = torch.round(pred) # get binary prediction
            return (pred_binary == labels).sum().item()/labels.shape[0] # num samples correctly classified / num_samples
        else:
            # the class with the highest energy is what we choose as prediction, if output 2 categories for binary classificaiton
            _, predicted = torch.max(outputs.data, 1)
            # print(f'shape: pred {predicted.shape}, label {labels.shape}')
            # print(f'device: pred {predicted.device}, label {labels.device}')
            return (predicted == labels).sum().item()/labels.shape[0]

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
    
    def _optimizer(self):
        # return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        return optim.Adam(self.model.parameters(), lr=0.0001)

    def _build_model(self, model_cat_vars, model_cont_vars, model_cat_unique_levels, n_output):
        n_cont = len(model_cont_vars)
        net = MLPModel(model_cat_unique_levels, n_cont, n_output) 
        if (self.device != 'cpu') and (torch.cuda.device_count() > 1):
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
    # tmle = NetworkTMLE(H, exposure='statin', outcome='cvd',
    #                    cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
    #                    use_deep_learner_A_i=True) 
    tmle = NetworkTMLE(H, exposure='statin', outcome='cvd',
                    cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
                    use_deep_learner_A_i_s=True) 
    
    # instantiation of MLP model
    # 5 fold cross validation 
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    mlp_learner = MLP(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
                      epochs=10, print_every=5, device=device, save_path='./tmp.pth')
    
    tmle.exposure_model("L + A_30 + R_1 + R_2 + R_3")
    # tmle.exposure_model("L + A_30 + R_1 + R_2 + R_3", custom_model=mlp_learner) # use_deep_learner_A_i=True
    # tmle.exposure_map_model("statin + L + A_30 + R_1 + R_2 + R_3",
    #                         measure='sum', distribution='poisson')  # Applying a Poisson model
    tmle.exposure_map_model("statin + L + A_30 + R_1 + R_2 + R_3",
                        measure='sum', distribution='poisson', custom_model=mlp_learner)  # use_deep_learner_A_i_s=True
    tmle.outcome_model("statin + statin_sum + A_sqrt + R + L")
    tmle.fit(p=0.35, bound=0.01)
    tmle.summary()


    # ############################# scratch #################################
    import patsy
    data_to_fit = tmle.df_restricted.copy()
    xdata = patsy.dmatrix(tmle._gs_model + ' - 1', 
                          data_to_fit, return_type="dataframe")       # Extract via patsy the data
    ydata = data_to_fit[tmle._gs_measure_]
    n_output = pd.unique(ydata).shape[0] 
    # print(f'gs_model: n_output = {n_output} for target variable {self._gs_measure_}')

    # pdata = patsy.dmatrix(self._gs_model + ' - 1', 
    #                         data_to_predict, return_type="dataframe")   # Extract via patsy the data
    # pdata_y = data_to_predict[self._gs_measure_]

    from tmle_utils import get_model_cat_cont_split_patsy_matrix, append_target_to_df
    model_cat_vars, model_cont_vars, model_cat_unique_levels, cat_vars, cont_vars, cat_unique_levels = get_model_cat_cont_split_patsy_matrix(xdata, 
                                                                                                                                             cat_vars, cont_vars, cat_unique_levels)
    fit_df = append_target_to_df(ydata, xdata, tmle._gs_measure_)  

    # initiate best model
    # self.best_model = None
    # self.best_loss = np.inf        

    # instantiate model
    # self.n_output = n_output
    

    model_cat_vars_new = ['A_30']
    model_cont_vars_new = ['L', 'statin', 'R_1', 'R_2', 'R_3']
    model_cat_unique_levels_new = {'A_30':31}
    mlp_learner.model = mlp_learner._build_model(model_cat_vars_new, 
                                                 model_cont_vars_new, 
                                                 model_cat_unique_levels_new, n_output)

    # mlp_learner.model = mlp_learner._build_model(model_cat_vars, model_cont_vars, model_cat_unique_levels, n_output)
    mlp_learner.optimizer = mlp_learner._optimizer()
    mlp_learner.n_output = n_output
    mlp_learner.criterion = mlp_learner._loss_fn()

    # target is exposure for nuisance models, outcome for outcome model
    fold_record = {'train_loss': [], 'val_loss': [],'train_acc':[],'val_acc':[]}


    # splits, dset = mlp_learner._data_preprocess(fit_df, tmle._gs_measure_,
    #                                             model_cat_vars=model_cat_vars, 
    #                                             model_cont_vars=model_cont_vars, 
    #                                             model_cat_unique_levels=model_cat_unique_levels)
    splits, dset = mlp_learner._data_preprocess(fit_df, tmle._gs_measure_,
                                                model_cat_vars=model_cat_vars_new, 
                                                model_cont_vars=model_cont_vars_new, 
                                                model_cat_unique_levels=model_cat_unique_levels_new)

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dset)))):
        print('Fold {}'.format(fold + 1))
        train_loader, valid_loader = get_kfold_dataloaders(dset, train_idx, val_idx, 
                                                           batch_size=16,
                                                           shuffle=True)
        break

    epochs = 20
    mlp_learner.train_loader = train_loader
    mlp_learner.valid_loader = valid_loader
    for epoch in range(epochs):
        print(f'============================= Epoch {epoch + 1}: Training =============================')
        loss_train, metrics_train = mlp_learner.train_epoch(epoch)
        print(f'============================= Epoch {epoch + 1}: Validation =============================')
        loss_valid, metrics_valid = mlp_learner.valid_epoch(epoch)

    for i, (x_cat, x_cont, y) in enumerate(train_loader):
        # send to device
        x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device) 

        # zero the parameter gradients
        mlp_learner.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = mlp_learner.model(x_cat, x_cont)
        if mlp_learner.n_output == 2: # binary classification
            # BCEWithLogitsLoss requires target as float, same size as outputs
            y = y.float() 
        else:
            # CrossEntropyLoss requires target (class indicies) as int, shape [batch_size]
            y = y.long().squeeze(-1) 
        loss = mlp_learner.criterion(outputs, y)
        loss.backward()
        mlp_learner.optimizer.step()

        # metrics
        metrics = mlp_learner._metrics(outputs, y)

        print(loss)
        print(metrics)

    mlp_learner.model
    mlp_learner.model.module.embedding_layers[0].weight

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