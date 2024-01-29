import numpy as np
from sklearn.model_selection import KFold # kfold cross validation

import torch
from torch.utils.data import Dataset, DataLoader, Subset

######################## define dataset ########################
class DfDataset(Dataset):
    def __init__(self, patsy_matrix_dataframe, target=None, 
                 model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}):
        ''' Retrieve train,label and pred data from Dataframe directly
        Args:  
            patsy_matrix_dataframe: pd.DataFrame, data, i.e., dataframe created from patsy.dmatrix()
            target: str, target variable, i.e., exposure/outcome
            model_cat_vars: list, categorical variables in patsy_matrix_dataframe, subset of cat_vars
            model_cont_vars: list, continuous variables in patsy_matrix_dataframe, subset of cont_vars
            model_cat_unique_levels: dict, number of unique levels for each categorical variable of patsy_matrix_dataframe, subset of cat_unique_levels
        '''
        
        self.df = patsy_matrix_dataframe 
        self.target = target
        self.model_cat_vars, self.model_cont_vars, self.model_cat_unique_levels = model_cat_vars, model_cont_vars, model_cat_unique_levels

        self.x_cat, self.x_cont = self._split_cat_cont() 
        self.y = self._get_labels()
        # if no target is available
        # self.y = np.empty((self.x_cat.shape[0], 1))
        # self.y.fill(-1) # create dummy target for pdata
    
    def _split_cat_cont(self):
        return self.df[self.model_cat_vars].to_numpy(), self.df[self.model_cont_vars].to_numpy()
    
    def _get_labels(self):
        return np.asarray(self.df[self.target])[:, np.newaxis] #[num_samples, ] -> [num_samples, 1] 
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.x_cat[idx]).int(), torch.from_numpy(self.x_cont[idx]).float(), torch.from_numpy(self.y[idx]).float(), idx
        # shape: [num_cat_vars], [num_cont_vars], [1], [1]

    def __len__(self):
        return self.y.shape[0]

class TimeSeriesDataset(Dataset):
    def __init__(self, patsy_matrix_dataframe_list, target=None, use_all_time_slices=True, 
                 model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}):
        ''' Retrieve train, label and pred data from list of Dataframe directly
        Args:  
            patsy_matrix_dataframe_list: list, containing pd.DataFrame data, i.e., dataframe created from patsy.dmatrix()
            target: str, target variable, i.e., exposure/outcome
            use_all_time_slices: bool, use label data from all time slices, or only the last time slice, default: True
            model_cat_vars: list, categorical variables in patsy_matrix_dataframe, subset of cat_vars
            model_cont_vars: list, continuous variables in patsy_matrix_dataframe, subset of cont_vars
            model_cat_unique_levels: dict, number of unique levels for each categorical variable of patsy_matrix_dataframe, subset of cat_unique_levels
        '''
        
        # use numerical index to avoid looping inside _getitem_()
        self.cat_col_index = self._column_name_to_index(patsy_matrix_dataframe_list[-1], model_cat_vars)
        self.cont_col_index = self._column_name_to_index(patsy_matrix_dataframe_list[-1], model_cont_vars)
        self.target_col_index = self._column_name_to_index(patsy_matrix_dataframe_list[-1], [target])

        self.data_array = np.stack([df.to_numpy() for df in patsy_matrix_dataframe_list], axis=-1) 
        # len(patsy_matrix_dataframe_list) = T
        # self.data_array: [num_samples, num_features, T]

        self.use_all_time_slices = use_all_time_slices

    def _column_name_to_index(self, dataframe, column_name):
        return dataframe.columns.get_indexer(column_name)
    
    def _get_labels(self):
        return self.data_array[:, self.target, :]

    def __getitem__(self, idx):
        cat_vars = torch.from_numpy(self.data_array[idx, self.cat_col_index, :]).int() # [num_cat_vars, T]
        cont_vars = torch.from_numpy(self.data_array[idx, self.cont_col_index, :]).float() # [num_cont_vars, T]
        labels = torch.from_numpy(self.data_array[idx, self.target_col_index, :]).float().squeeze(0) # [1, T] -> [T]

        if not self.use_all_time_slices:
            labels = labels[-1] # [] 
        
        return cat_vars, cont_vars, labels, idx # idx shape []

    def __len__(self):
        return self.data_array.shape[0]

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