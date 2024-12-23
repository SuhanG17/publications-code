import numpy as np
import pandas as pd

# TODO: fix the path to input/output files; add extract_acc to git
# dir_path = './'
dir_path = '../results_csv/logfiles/input_logs/'
model_type_path = 'LR/'
filename = '10010' + '_training_outcome' + 'LR.log'
# filename = '70040' + '_training_outcome' + 'LR_all.log'
print(dir_path + model_type_path + filename)

with open(dir_path + model_type_path + filename) as inp:
    data = list(inp) # or set(inp) if you really need a set

# START: check irregularity in the log file: which simulation have less than normal acc values
sim_slice = []
for i, line in enumerate(data):
    if 'simulation' in line:
        sim = int(line.split(' ')[-1].split('\n')[0])
        print(f'simulation {sim} starts at line {i}')
        sim_slice.append(i)

for i, line in enumerate(data):
    if "RESULTS" in line:
        print(i, line)
        sim_slice.append(i) # append the last line to include the last simulation

for sim_id, (L, U) in enumerate(zip(sim_slice,sim_slice[1:])):
    data_slice = data[L:U]
    counter = 0
    for i, line in enumerate(data_slice):
        if 'Outcome model accuracy in pooled data' in line:
            # print(i, line)
            acc = float(line.split(' ')[-1].split('\n')[0]) 
            # print(acc)
            # acc_ls.append(acc)
            counter += 1
    print(f'sim {sim_id}: number of acc: {counter}')
# END: check irregularity in the log file

# START: extract the acc values
acc_ls = [] 
for i, line in enumerate(data):
    if 'Outcome model accuracy in pooled data' in line:
        # print(i, line)
        acc = float(line.split(' ')[-1].split('\n')[0]) 
        print(acc)
        acc_ls.append(acc)

len(acc_ls) / 19

col_index = np.round(np.arange(0.05, 1, 0.05), 2)
col_index
col_index.shape

row_index = np.arange(0, 30, 1)
row_index

acc_array = np.array(acc_ls)
acc_array = acc_array.reshape(19, 30)
acc_array = acc_array.T

df = pd.DataFrame(acc_array, columns=col_index, index=row_index)
df

df.describe()
df.apply([np.mean, np.std], axis=0) # avg and std per column (over simulations)
df.apply([np.mean, np.std], axis=1) # avg and std per row (over p_{\omega} values)

df_avg_std = pd.concat([df, df.apply([np.mean, np.std], axis=0)], axis=0)
df_avg_std = pd.concat([df_avg_std, df.apply([np.mean, np.std], axis=1)], axis=1)
df_avg_std


def extract_acc(filepath):
    with open(filepath) as inp:
        data = list(inp) # or set(inp) if you really need a set

    acc_ls = [] 
    for i, line in enumerate(data):
        if 'Outcome model accuracy in pooled data' in line:
            acc = float(line.split(' ')[-1].split('\n')[0]) 
            acc_ls.append(acc)

    acc_array = np.array(acc_ls)
    acc_array = acc_array.reshape(19, 30)
    acc_array = acc_array.T

    df = pd.DataFrame(acc_array, columns=col_index, index=row_index)
    col_avg_std = df.apply([np.mean, np.std], axis=0) # avg and std per column (over simulations)
    row_avg_std = df.apply([np.mean, np.std], axis=1) # avg and std per row (over p_{\omega} values)
    df_avg_std = pd.concat([df, col_avg_std], axis=0)
    df_avg_std = pd.concat([df_avg_std, row_avg_std], axis=1)

    return df_avg_std, col_avg_std, row_avg_std

df_avg_std, col_avg_std, row_avg_std = extract_acc(dir_path + model_type_path + filename)
df_avg_std
col_avg_std
row_avg_std 

col_avg_std_T = col_avg_std.T
latex_version = np.round(col_avg_std_T['mean'], 3).astype('str')+'/pm'+np.round(col_avg_std_T['std'], 3).astype('str')
latex_version.shape


# deep learning extraction
model_type_path = 'DL/'
filename = '10010' + '_training_outcome' + 'DL.log'

with open(dir_path + model_type_path + filename) as inp:
    data = list(inp) # or set(inp) if you really need a set

acc_ls = [] 
for i, line in enumerate(data):
    if 'overall acc' in line:
        # print(i, line)
        acc = float(line.split(' ')[-1].split('\n')[0]) 
        print(acc)
        acc_ls.append(acc)

observed_acc = acc_ls[0::2]
pooled_acc = acc_ls[1::2]
len(pooled_acc) / 19



############################## START HERE ########################################

def extract_acc(filepath, num_sim=30, LR=True):
    # col: 19 p_{\omega} values
    # row: 30 simulations
    col_index = np.round(np.arange(0.05, 1, 0.05), 2)
    row_index = np.arange(0, 30, 1)
    with open(filepath) as inp:
        data = list(inp) # or set(inp) if you really need a set

    acc_ls = [] 
    if LR:
        for i, line in enumerate(data):
            if 'Outcome model accuracy in pooled data' in line:
                acc = float(line.split(' ')[-1].split('\n')[0]) 
                acc_ls.append(acc)
        pooled_acc = acc_ls
    else:
        for i, line in enumerate(data):
            if 'overall acc' in line:
                # print(i, line)
                acc = float(line.split(' ')[-1].split('\n')[0]) 
                acc_ls.append(acc)
        observed_acc = acc_ls[0::2]
        pooled_acc = acc_ls[1::2]
        

    acc_array = np.array(pooled_acc)
    acc_array = acc_array.reshape(19, num_sim)
    acc_array = acc_array.T

    df = pd.DataFrame(acc_array, columns=col_index, index=row_index)
    col_avg_std = df.apply([np.mean, np.std], axis=0) # avg and std per column (over simulations)
    row_avg_std = df.apply([np.mean, np.std], axis=1) # avg and std per row (over p_{\omega} values)
    df_avg_std = pd.concat([df, col_avg_std], axis=0)
    df_avg_std = pd.concat([df_avg_std, row_avg_std], axis=1)

    return df_avg_std, col_avg_std, row_avg_std

# df_avg_std, col_avg_std, row_avg_std = extract_acc(dir_path + filename, LR=False)
# col_avg_std

def transform_col_avg_std(col_avg_std, network_type='10010LR'):
    col_index = np.round(np.arange(0.05, 1, 0.05), 2)
    col_avg_std_T = col_avg_std.T
    latex_version = '$'+np.round(col_avg_std_T['mean'], 3).astype('str') +' \pm '+np.round(col_avg_std_T['std'], 3).astype('str')+'$'
    latex_df = pd.DataFrame(latex_version, columns=[network_type], index=col_index).T
    return latex_df

# latex_df = transform_col_avg_std(col_avg_std, network_type='10010DL')
# latex_df
# pd.concat([latex_df, transform_col_avg_std(col_avg_std, network_type='10010LR')], axis=0)

networktype = ['10010', '10020', '10030', '10040',
               '20010', '20020', '20030', '20040',
               '40010', '40020', '40030', '40040',
               '50010', '50020', '50030', '50040',
               '60010', '60020', '60030', '60040',
               '70010', '70020', '70030', '70040']
save_path = '../results_csv/logfiles/output_csv/'

latex_df_holder = []
for network in networktype:
    if '0040' not in network and '700' not in network: # the acc for LR *0040 and 700*0 is not saved
        num_sim = 30    
        # if '700' in network: # only 15 simulations for 700*0
        #     num_sim = 15
        # else:
        #     num_sim = 30

        dir_path = '../results_csv/logfiles/input_logs/LR/'
        df_avg_std_LR, col_avg_std_LR, row_avg_std_LR = extract_acc(dir_path + network + '_training_outcome' + 'LR.log', num_sim=num_sim, LR=True)
        df_avg_std_LR.to_csv(save_path + network + '_pooled_acc_' + 'LR.csv')
        latex_df_LR = transform_col_avg_std(col_avg_std_LR, network_type=network+'LR')
        latex_df_holder.append(latex_df_LR)
    
        dir_path = '../results_csv/logfiles/input_logs/DL/'
        if  '400' in network:
            dir_path = 'DL/re-run/'
        df_avg_std_DL, col_avg_std_DL, row_avg_std_DL = extract_acc(dir_path + network + '_training_outcome' + 'DL.log', num_sim=num_sim, LR=False)
        df_avg_std_DL.to_csv(save_path + network + '_pooled_acc_' + 'DL.csv')
        latex_df_DL = transform_col_avg_std(col_avg_std_DL, network_type=network+'DL')
        latex_df_holder.append(latex_df_DL)

latex_df_final = pd.concat(latex_df_holder, axis=0)
latex_df_final.to_csv(save_path + 'latex_acc.csv')