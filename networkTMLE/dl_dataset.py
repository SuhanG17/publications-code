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

class TimeSeriesDatasetSeparate(Dataset):
    def __init__(self, xy_list, model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}):
        ''' Retrieve train, label and pred data from list of pd.Dataframe (input) and list of np.array (label) directly
        Args:  
            xy_list: list, containing pd.DataFrame data and np.array label, PS, len(xdata_list) does not necassarily equal to len(ydata_list)
            model_cat_vars: list, categorical variables in patsy_matrix_dataframe, subset of cat_vars
            model_cont_vars: list, continuous variables in patsy_matrix_dataframe, subset of cont_vars
            model_cat_unique_levels: dict, number of unique levels for each categorical variable of patsy_matrix_dataframe, subset of cat_unique_levels
        '''
        xdata_list, ydata_list = xy_list

        # use numerical index to avoid looping inside _getitem_()
        self.cat_col_index = self._column_name_to_index(xdata_list[-1], model_cat_vars)
        self.cont_col_index = self._column_name_to_index(xdata_list[-1], model_cont_vars)

        self.input_data_array = np.stack([df.to_numpy() for df in xdata_list], axis=-1) 
        # len(xdata_list) = T_in
        # self.input_data_array: [num_samples, num_features, T_in]
        self.label_data_array = np.stack(ydata_list, axis=-1)
        # len(ydata_list) = T_out
        # self.label_data_array: [num_samples, T_out]

    def _column_name_to_index(self, dataframe, column_name):
        return dataframe.columns.get_indexer(column_name)

    def __getitem__(self, idx):
        cat_vars = torch.from_numpy(self.input_data_array[idx, self.cat_col_index, :]).int() # [num_cat_vars, T_in]
        cont_vars = torch.from_numpy(self.input_data_array[idx, self.cont_col_index, :]).float() # [num_cont_vars, T_in]
        labels = torch.from_numpy(self.label_data_array[idx, :]).float().squeeze(0) # [1, T_out] -> [T_out]
        
        return cat_vars, cont_vars, labels, idx # idx shape []

    def __len__(self):
        return self.input_data_array.shape[0] # num_samples

class TimeSeriesDatasetSeparateNormalize(Dataset):
    def __init__(self, xy_list, model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={},
                 normalize=True, drop_duplicates=True, T_in_id=[*range(10)], T_out_id=[*range(10)]):
        ''' Retrieve train, label and pred data from list of pd.Dataframe (input) and list of np.array (label) directly,
            Treat categorical variables as numerical float, apply normalization to them and drop duplicates in the dataset
        Args:  
            xy_list: list, containing pd.DataFrame data and np.array label, PS, len(xdata_list) does not necassarily equal to len(ydata_list)
            model_cat_vars: list, categorical variables in patsy_matrix_dataframe, subset of cat_vars
            model_cont_vars: list, continuous variables in patsy_matrix_dataframe, subset of cont_vars
            model_cat_unique_levels: dict, number of unique levels for each categorical variable of patsy_matrix_dataframe, subset of cat_unique_levels
            normalize: normalize the dataframe, treat every column as numerical float
            drop_duplicates: drop duplicates in the dataset, save only the minimum of data
            T_in_id: list
                list of index showing the time slice used for the input data
            T_out_id: list
                list of index showing the time slice used for the outcome data
        '''
        xdata_list, ydata_list = xy_list

        xdata_list_reduce = []
        ydata_list_reduce = []
        if drop_duplicates:
            sample_size = []
            for i in T_in_id:
                # map to index after slicing
                x_id = T_in_id.index(i)
                check_duplicates = xdata_list[x_id].duplicated()
                if check_duplicates.sum() > 0:
                    xdata_list_reduce.append(xdata_list[x_id].drop_duplicates())
                    if i in T_out_id:
                        y_id = T_out_id.index(i)
                        ydata_list_reduce.append(ydata_list[y_id][~check_duplicates])
                    sample_size.append(xdata_list_reduce[x_id].shape[0])
                else:
                    xdata_list_reduce.append(xdata_list[x_id])
                    if i in T_out_id:
                        y_id = T_out_id.index(i) 
                        ydata_list_reduce.append(ydata_list[y_id])
                    sample_size.append(xdata_list_reduce[x_id].shape[0])
            # select the minimum sample size after dropping duplicates
            xdata_list_reduce = [xdata_list_reduce[i][:min(sample_size)] for i in range(len(xdata_list_reduce))]
            ydata_list_reduce = [ydata_list_reduce[i][:min(sample_size)] for i in range(len(ydata_list_reduce))]
            print(f'sample sizes: {sample_size}')

        if normalize:
            if drop_duplicates:
                xdata_list_norm = [self._normlize_dataframe(df) for df in xdata_list_reduce]
            else:
                xdata_list_norm = [self._normlize_dataframe(df) for df in xdata_list]

        # use numerical index to avoid looping inside _getitem_()
        self.cat_col_index = self._column_name_to_index(xdata_list[-1], model_cat_vars)
        self.cont_col_index = self._column_name_to_index(xdata_list[-1], model_cont_vars)

        if drop_duplicates and normalize:
            self.input_data_array = np.stack([df.to_numpy() for df in xdata_list_norm], axis=-1) 
            self.label_data_array = np.stack(ydata_list_reduce, axis=-1)
        elif drop_duplicates:
            self.input_data_array = np.stack([df.to_numpy() for df in xdata_list_reduce], axis=-1) 
            self.label_data_array = np.stack(ydata_list_reduce, axis=-1)
        elif normalize:
            self.input_data_array = np.stack([df.to_numpy() for df in xdata_list_norm], axis=-1) 
            self.label_data_array = np.stack(ydata_list, axis=-1)
        else:
            self.input_data_array = np.stack([df.to_numpy() for df in xdata_list], axis=-1) 
            # len(xdata_list) = T_in
            # self.input_data_array: [num_samples, num_features, T_in]
            self.label_data_array = np.stack(ydata_list, axis=-1)
            # len(ydata_list) = T_out
            # self.label_data_array: [num_samples, T_out]
    
    def _normlize_dataframe(self, dataframe, method='zscore'):
        if method == 'zscore':
            return (dataframe - dataframe.mean())/dataframe.std()
        elif method == 'minmax':
            return (dataframe - dataframe.min())/(dataframe.max() - dataframe.min())

    def _column_name_to_index(self, dataframe, column_name):
        return dataframe.columns.get_indexer(column_name)

    def __getitem__(self, idx):
        cat_vars = torch.from_numpy(self.input_data_array[idx, self.cat_col_index, :]).float() # [num_cat_vars, T_in]
        cont_vars = torch.from_numpy(self.input_data_array[idx, self.cont_col_index, :]).float() # [num_cont_vars, T_in]
        labels = torch.from_numpy(self.label_data_array[idx, :]).float().squeeze(0) # [1, T_out] -> [T_out]
        
        return cat_vars, cont_vars, labels, idx # idx shape []

    def __len__(self):
        return self.input_data_array.shape[0] # num_samples


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