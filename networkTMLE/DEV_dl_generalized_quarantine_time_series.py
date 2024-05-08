# from sys import argv
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic
from sklearn.linear_model import LogisticRegression

# from amonhen import NetworkTMLE
# from amonhen.utils import probability_to_odds, odds_to_probability, fast_exp_map

from tmle_dl_time_series import NetworkTMLETimeSeries
from tmle_utils import probability_to_odds, odds_to_probability, fast_exp_map

from beowulf import load_uniform_vaccine, load_random_vaccine, truth_values, simulation_setup
from beowulf.dgm import quarantine_dgm_time_series
from beowulf.dgm.utils import network_to_df

import torch
import torch.nn as nn
from dl_trainer_time_series import MLPTS, GCNTS, CNNTS

############################################
# Setting simulation parameters
############################################
# n_mc = 500
n_mc = 30

exposure = "quarantine"
outcome = "D"

########################################
# Running through logic from .sh script
########################################
parser = argparse.ArgumentParser(description='DLnetworkTMLE')
parser.add_argument('--task_string', type=str, required=True, default='10010',
                        help='the slurm_setup id in string format')
args = parser.parse_args()

# script_name, slurm_setup = argv
# script_name, slurm_setup = 'some_script', '10010'  
script_name = 'some_script'
slurm_setup = args.task_string

network, n_nodes, degree_restrict, shift, model, save = simulation_setup(slurm_id_str=slurm_setup)
sim_id = slurm_setup[4]
seed_number = 18900567 + 10000000*int(sim_id)
np.random.seed(seed_number)

# random network with reproducibility
torch.manual_seed(seed_number) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# choode which what-if mechanism to test
mode = 'top'
percent_candidates = 0.3

# decide use deep learning in which nuisance model
use_deep_learner_A_i = False
use_deep_learner_A_i_s = False 
use_deep_learner_outcome = False 

# decide which model to use
deep_learner_type = 'mlp' # 'mlp' or 'gcn' or 'cnn'

# Loading correct  Network
if network == "uniform":
    # G = load_uniform_vaccine(n=n_nodes)
    G, cat_vars, cont_vars, cat_unique_levels = load_uniform_vaccine(n=n_nodes, return_cat_cont_split=True)
if network == "random":
    # G = load_random_vaccine(n=n_nodes)
    G, cat_vars, cont_vars, cat_unique_levels = load_random_vaccine(n=n_nodes, return_cat_cont_split=True)


# Marking if degree restriction is being applied
if degree_restrict is not None:
    restrict = True
else:
    restrict = False

# Setting up models
independent = False
distribution_gs = "poisson"
measure_gs = "sum"
q_estimator = None
if model == "cc":
    # gin_model = "A + H + A_sum + H_sum + degree"
    # gsn_model = "quarantine + A + H + A_sum + H_sum + degree"
    # # qn_model = "quarantine + quarantine_mean + A + H + A_sum + H_sum + degree"
    # qn_model = "A + H + A_sum + H_sum"

    #TODO
    gin_model = "A + H + A_sum + H_sum + I_ratio"
    gsn_model = "quarantine + A + H + A_sum + H_sum + I_ratio"
    qn_model = "A + H + A_sum + H_sum + degree"

elif model == "cw":
    # gin_model = "A + H + A_sum + H_sum + degree"
    # gsn_model = "quarantine + A + H + A_sum + H_sum + degree"
    # qn_model = "quarantine + quarantine_mean + A + H + H_t3 + degree"

    gin_model = "A + H + A_sum + H_sum + I_ratio"
    gsn_model = "quarantine + A + H + A_sum + H_sum + I_ratio"
    qn_model = "quarantine + quarantine_mean + A + H + H_t3 + degree" 
elif model == "wc":
    # gin_model = "A + H + H_t3 + degree"
    # gsn_model = "quarantine + A + H + H_t3 + degree"
    # qn_model = "quarantine + quarantine_mean + A + H + A_sum + H_sum + degree"

    gin_model = "A + H + H_t3 + I_ratio"
    gsn_model = "quarantine + A + H + H_t3 + I_ratio"
    qn_model = "A + H + A_sum + H_sum + degree" 
elif model == 'np':
    # gin_model = "A + H + C(A_sum_c) + C(H_sum_c) + degree"
    # gsn_model = "quarantine + A + H + C(A_sum_c) + C(H_sum_c) + degree"
    # qn_model = "quarantine + quarantine_mean + A + H + C(A_sum_c) + C(H_sum_c) + degree"
    # q_estimator = LogisticRegression(penalty='l2', max_iter=2000)
    
    gin_model = "A + H + C(A_sum_c) + C(H_sum_c) + I_ratio"
    gsn_model = "quarantine + A + H + C(A_sum_c) + C(H_sum_c) + I_ratio"
    qn_model = "quarantine + quarantine_mean + A + H + C(A_sum_c) + C(H_sum_c) + degree"
    if not use_deep_learner_outcome:
        q_estimator = LogisticRegression(penalty='l2', max_iter=2000)
elif model == 'ind':
    independent = True
    gi_model = "A + H"
    qi_model = "quarantine + A + H"

# Determining if shift or absolute
if shift:
    prop_treated = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]

    # Generating probabilities (true) to assign
    data = network_to_df(G)
    adj_matrix = nx.adjacency_matrix(G, weight=None)
    data['A_sum'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='sum')
    data['H_sum'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='sum')
    data = pd.merge(data, pd.DataFrame.from_dict(dict(G.degree),
                                                 orient='index').rename(columns={0: 'degree'}),
                    how='left', left_index=True, right_index=True)
    data['I_sum'] = fast_exp_map(adj_matrix, np.array(data['I']), measure='sum')
    data['I_ratio'] = data['I_sum'] / data['degree'] # ratio of infected neighbors
    prob = logistic.cdf(- 3.5 
                        + 1.0*data['A'] + 0.5*data['H']
                        + 0.3*data['H_sum'] + 0.1*data['A_sum'] 
                        + 3.0*data['I_ratio'])
    log_odds = np.log(probability_to_odds(prob))

else:
    prop_treated = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

truth = truth_values(network=network, dgm=exposure,
                     restricted_degree=restrict, shift=shift, n=n_nodes,
                     mode=mode, percent_candidates=percent_candidates)
print(truth)

print("#############################################")
print("Sim Script:", slurm_setup)
print("Seed:      ", seed_number)
print("=============================================")
print("Network:     ", network)
print("N-nodes:     ", nx.number_of_nodes(G))
print("DGM:         ", exposure, '-', outcome)
print("Independent: ", independent)
print("Shift:       ", shift)
print("Set-up:      ", model)
print("=============================================")
print("results/" + exposure + "_" + save + ".csv")
print("#############################################")

########################################
# Setting up storage
########################################
if independent:
    cols = ['inc_'+exposure, 'inc_'+outcome] + ['bias_' + str(p) for p in prop_treated] + \
           ['lcl_' + str(p) for p in prop_treated] + ['ucl_' + str(p) for p in prop_treated] + \
           ['var_' + str(p) for p in prop_treated]
else:
    cols = ['inc_'+exposure, 'inc_'+outcome] + ['bias_' + str(p) for p in prop_treated] + \
           ['lcl_' + str(p) for p in prop_treated] + ['ucl_' + str(p) for p in prop_treated] + \
           ['lcll_' + str(p) for p in prop_treated] + ['ucll_' + str(p) for p in prop_treated] + \
           ['var_' + str(p) for p in prop_treated] + ['varl_' + str(p) for p in prop_treated]

results = pd.DataFrame(index=range(n_mc), columns=cols)

########################################
# Running simulation
########################################
for i in range(n_mc):
# i=0
    ######## inside for loop ########
    # Generating Data
    # H = vaccine_dgm(network=G, restricted=restrict)
    H, network_list, cat_vars_i, cont_vars_i, cat_unique_levels_i  = quarantine_dgm_time_series(network=G, restricted=restrict, 
                                                                                                time_limit=10, inf_duration=5,
                                                                                                update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
                                                                                                random_seed=seed_number+i)
    df = network_to_df(H)
    results.loc[i, 'inc_'+exposure] = np.mean(df[exposure])
    results.loc[i, 'inc_'+outcome] = np.mean(df[outcome])

    # Network TMLE
    # use deep learner for given nuisance model
    ntmle = NetworkTMLETimeSeries(network_list, exposure='quarantine', outcome='D', verbose=False, degree_restrict=degree_restrict,
                                    cat_vars=cat_vars_i, cont_vars=cont_vars_i, cat_unique_levels=cat_unique_levels_i,
                                    use_deep_learner_A_i=use_deep_learner_A_i, 
                                    use_deep_learner_A_i_s=use_deep_learner_A_i_s, 
                                    use_deep_learner_outcome=use_deep_learner_outcome,
                                    use_all_time_slices=False) 

    if model in ["cw", "wc"]:
        ntmle.define_threshold(variable='H', threshold=3, definition='sum')
    if model == "np":
        if network == "uniform":
            if n_nodes == 500:
                ntmle.define_category(variable='A_sum', bins=[0, 1, 3], labels=False)
                ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 6], labels=False)
            elif n_nodes == 1000:
                ntmle.define_category(variable='A_sum', bins=[0, 1, 5], labels=False)
                ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 6], labels=False)
            else:
                ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 3, 6], labels=False)
                ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 6], labels=False)
        elif network == "random":
            if n_nodes == 500:
                ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 4, 10], labels=False)
                ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 4, 7, 18], labels=False)
            elif n_nodes == 1000:
                ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 10], labels=False)
                ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 8, 26], labels=False)
            else:
                ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 3, 4, 5, 10], labels=False)
                ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 6, 9, 26], labels=False)
        else:
            raise ValueError("Invalid model-network combo")
    # ntmle.exposure_model(gin_model)
    # ntmle.exposure_map_model(gsn_model, measure=measure_gs, distribution=distribution_gs)
    # ntmle.outcome_model(qn_model, custom_model=q_estimator)

    # use deep learner
    def get_deep_learner(deep_learner_type, device):
        '''Return deep learner model based on the type of deep learner'''
        if deep_learner_type == 'mlp':
            deep_learner = MLPTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
                                epochs=25, print_every=5, device=device, save_path='./tmp.pth',
                                lin_hidden=None,
                                lin_hidden_temporal=nn.ModuleList([nn.Linear(128, 256), nn.Linear(256, 128)]))
        elif deep_learner_type == 'gcn':
            deep_learner = GCNTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
                                    epochs=10, print_every=5, device=device, save_path='./tmp.pth')
        elif deep_learner_type == 'cnn':
            deep_learner = CNNTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
                                    epochs=10, print_every=5, device=device, save_path='./tmp.pth')
        else:
            raise NotImplementedError("Invalid deep learner type. Choose from 'mlp', 'gcn', 'cnn'")
        return deep_learner


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    if use_deep_learner_A_i:
        deep_learner_a_i = get_deep_learner(deep_learner_type, device)
    else:
        deep_learner_a_i = None
        print(f'No deep learner for A_i model')

    if use_deep_learner_A_i_s:
        deep_learner_a_i_s = get_deep_learner(deep_learner_type, device)
    else:
        deep_learner_a_i_s  = None
        print(f'No deep learner for A_i_s model')
            
    if use_deep_learner_outcome:
        deep_learner_outcome = get_deep_learner(deep_learner_type, device)
    else:
        deep_learner_outcome = None
        print(f'No deep learner for outcome model')

    ntmle.exposure_model(gin_model, custom_model=deep_learner_a_i) 
    ntmle.exposure_map_model(gsn_model, measure=measure_gs, distribution=distribution_gs, custom_model=deep_learner_a_i_s)
    if q_estimator is not None:
        ntmle.outcome_model(qn_model, custom_model=q_estimator) 
    else:
        ntmle.outcome_model(qn_model, custom_model=deep_learner_outcome, T_in_id=[6, 7, 8, 9], T_out_id=[9])

    for p in prop_treated:  # loops through all treatment plans
        try:
            if shift:
                z = odds_to_probability(np.exp(log_odds + p))
                ntmle.fit(p=z, bound=0.01, seed=seed_number+i, 
                        shift=shift, mode=mode, percent_candidates=percent_candidates,
                        T_in_id=[6, 7, 8, 9], T_out_id=[9])
            else:
                ntmle.fit(p=p, bound=0.01, seed=seed_number+i,
                        shift=shift, mode=mode, percent_candidates=percent_candidates,
                        T_in_id=[6, 7, 8, 9], T_out_id=[9])
            results.loc[i, 'bias_'+str(p)] = ntmle.marginal_outcome - truth[p]
            results.loc[i, 'var_'+str(p)] = ntmle.conditional_variance
            results.loc[i, 'lcl_'+str(p)] = ntmle.conditional_ci[0]
            results.loc[i, 'ucl_'+str(p)] = ntmle.conditional_ci[1]
            results.loc[i, 'varl_'+str(p)] = ntmle.conditional_latent_variance
            results.loc[i, 'lcll_'+str(p)] = ntmle.conditional_latent_ci[0]
            results.loc[i, 'ucll_'+str(p)] = ntmle.conditional_latent_ci[1]
        except:
            results.loc[i, 'bias_'+str(p)] = np.nan
            results.loc[i, 'var_'+str(p)] = np.nan
            results.loc[i, 'lcl_'+str(p)] = np.nan
            results.loc[i, 'ucl_'+str(p)] = np.nan
            results.loc[i, 'varl_'+str(p)] = np.nan
            results.loc[i, 'lcll_'+str(p)] = np.nan
            results.loc[i, 'ucll_'+str(p)] = np.nan
    ######## inside for loop ########

########################################
# Summarizing results
########################################
print("RESULTS\n")

avg_over_sims={'prop_treated':[], 'bias':[], 'ese':[], 'cover':[], 'coverl':[]}
for p in prop_treated:
    # Confidence Interval Coverage
    results['cover_'+str(p)] = np.where((results['lcl_'+str(p)] < truth[p]) &
                                        (truth[p] < results['ucl_'+str(p)]), 1, 0)
    # Confidence Limit Difference
    results['cld_'+str(p)] = results['ucl_'+str(p)] - results['lcl_'+str(p)]
    if not independent:
        results['coverl_' + str(p)] = np.where((results['lcll_' + str(p)] < truth[p]) &
                                              (truth[p] < results['ucll_' + str(p)]), 1, 0)
        results['cldl_'+str(p)] = results['ucll_'+str(p)] - results['lcll_'+str(p)]

    print("===========================")
    print(p)
    print("---------------------------")
    print("Bias:", np.mean(results['bias_'+str(p)]))
    print("ESE:", np.std(results['bias_'+str(p)], ddof=1))
    print("Cover:", np.mean(results['cover_'+str(p)]))
    if not independent:
        print("Cover-Latent:", np.mean(results['coverl_' + str(p)]))
    
    avg_over_sims['prop_treated'].append(p)
    avg_over_sims['bias'].append(np.mean(results['bias_'+str(p)]))
    avg_over_sims['ese'].append(np.std(results['bias_'+str(p)], ddof=1))
    avg_over_sims['cover'].append(np.mean(results['cover_'+str(p)]))
    if not independent:
        avg_over_sims['coverl'].append(np.mean(results['coverl_' + str(p)]))

print("===========================")
avg_df = pd.DataFrame.from_dict(avg_over_sims, orient='index')
avg_df.to_csv("avg_sims_results/" + exposure + str(sim_id) + "_" + save + ".csv", index=False)

########################################
# Saving results
########################################
results.to_csv("results/" + exposure + str(sim_id) + "_" + save + ".csv", index=False)
            
# ##################################### TEST CODE #####################################
# p=prop_treated[3]
# ntmle.fit(p=p, bound=0.01, seed=seed_number+i,
#         shift=shift, mode=mode, percent_candidates=percent_candidates,
#         T_in_id=[6, 7, 8, 9], T_out_id=[9])

# ntmle._q_custom_path_
# ntmle._q_custom_.model

# tmp_state_dict = torch.load(ntmle._q_custom_path_)
# for key, value in tmp_state_dict.items():
#     print(key)
#     print(value.shape)
#     print()

# results.loc[i, 'bias_'+str(p)] = ntmle.marginal_outcome - truth[p]
# results.loc[i, 'var_'+str(p)] = ntmle.conditional_variance
# results.loc[i, 'lcl_'+str(p)] = ntmle.conditional_ci[0]
# results.loc[i, 'ucl_'+str(p)] = ntmle.conditional_ci[1]
# results.loc[i, 'varl_'+str(p)] = ntmle.conditional_latent_variance
# results.loc[i, 'lcll_'+str(p)] = ntmle.conditional_latent_ci[0]
# results.loc[i, 'ucll_'+str(p)] = ntmle.conditional_latent_ci[1]
# results



# prop_treated

# # ntmle.exposure_model(gin_model, custom_model=deep_learner_a_i) 
# # ntmle.exposure_map_model(gsn_model, measure=measure_gs, distribution=distribution_gs, custom_model=deep_learner_a_i_s)
# # if q_estimator is not None:
# #     ntmle.outcome_model(qn_model, custom_model=q_estimator) 
# # else:
# #     ntmle.outcome_model(qn_model, custom_model=deep_learner_outcome)

# import patsy
# from tmle_utils import get_patsy_for_model_w_C, get_model_cat_cont_split_patsy_matrix, get_final_model_cat_cont_split, get_imbalance_weights

# xdata_list = []
# ydata_list = []
# n_output_list = []
# for df_restricted in ntmle.df_restricted_list:
#     if 'C(' in ntmle._q_model:
#         xdata_list.append(get_patsy_for_model_w_C(ntmle._q_model, df_restricted))
#     else:
#         xdata_list.append(patsy.dmatrix(ntmle._q_model + ' - 1', df_restricted, return_type="dataframe"))
#     ydata_list.append(df_restricted[ntmle.outcome])
#     n_output_list.append(pd.unique(df_restricted[ntmle.outcome]).shape[0])


# # T_in_id=[9]
# # T_out_id=[9]

# T_in_id=[*range(10)]
# T_out_id=[9]

# # slicing xdata and ydata list
# xdata_list, ydata_list = [xdata_list[i] for i in T_in_id], [ydata_list[i] for i in T_out_id]
# # T_in, T_out = len(xdata_list), len(ydata_list)

# # Re-arrange data
# model_cat_vars_list = []
# model_cont_vars_list = []
# model_cat_unique_levels_list = []

# cat_vars_list = []
# cont_vars_list = []
# cat_unique_levels_list = []

# # deep_learner_df_list = []
# ydata_array_list = []

# for xdata in xdata_list:
#     model_cat_vars, model_cont_vars, model_cat_unique_levels, cat_vars, cont_vars, cat_unique_levels = get_model_cat_cont_split_patsy_matrix(xdata, 
#                                                                                                                                                 cat_vars, cont_vars, cat_unique_levels)
#     model_cat_vars_list.append(model_cat_vars)
#     model_cont_vars_list.append(model_cont_vars)
#     model_cat_unique_levels_list.append(model_cat_unique_levels)

#     cat_vars_list.append(cat_vars)
#     cont_vars_list.append(cont_vars)
#     cat_unique_levels_list.append(cat_unique_levels)

# for ydata in ydata_list:
#     ydata_array_list.append(ydata.to_numpy()) # convert pd.series to np.array

# model_cat_vars_final, model_cont_vars_final, model_cat_unique_levels_final = get_final_model_cat_cont_split(model_cat_vars_list, model_cont_vars_list, model_cat_unique_levels_list)

# # ## check if n_output is consistent 
# # if not all_equal(n_output_list):
# #     raise ValueError("n_output are not identical throughout time slices")
# # else:
# n_output_final = n_output_list[-1]    

# ## set weight against class imbalance
# pos_weight, class_weight = get_imbalance_weights(n_output_final, ydata_array_list, use_last_time_slice=True,
#                                                     imbalance_threshold=3., imbalance_upper_bound=3.2, default_lock=False)
        
# # Fitting model
# adj_matrix_list = None
# _continuous_outcome = False
# custom_path = './tmp.pth'
# best_model_path = ntmle._q_custom_.fit([xdata_list, ydata_array_list], T_in_id, T_out_id, pos_weight, class_weight,
#                                         adj_matrix_list, model_cat_vars_final, model_cont_vars_final, model_cat_unique_levels_final, 
#                                         n_output_final, _continuous_outcome, custom_path=custom_path)

# from dl_dataset import TimeSeriesDatasetSeparateNormalize, get_predict_loader
# from torch.utils.data import Dataset, DataLoader, Subset
# dset = TimeSeriesDatasetSeparateNormalize([xdata_list, ydata_array_list],
#                                           model_cat_vars=model_cat_vars, 
#                                             model_cont_vars=model_cont_vars, 
#                                             model_cat_unique_levels=model_cat_unique_levels,
#                                             normalize=True,
#                                             drop_duplicates=False,
#                                             T_in_id=T_in_id, T_out_id=T_out_id)
# # test_loader = get_predict_loader(dset, 16)
# test_loader = DataLoader(dset, batch_size=16, shuffle=True)
# len(test_loader)
# criterion = nn.BCEWithLogitsLoss()
# model = ntmle._q_custom_.model
# model.load_state_dict(torch.load(custom_path)) 
# model.eval()
# for x_cat, x_cont, y, sample_idx in test_loader:
#     # send to device
#     x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device) 
#     sample_idx = sample_idx.to(device) 
#     outputs = model(x_cat, x_cont, sample_idx) # shape [batch_size, num_classes, T_out] 

#     y = y.unsqueeze(-1).unsqueeze(-1)
#     loss = criterion(outputs, y)
#     print(loss.item())


# xdata = xdata_list[-1]
# xdata.duplicated().sum()
# xdata.duplicated(keep='first').sum()
# xdata.duplicated(keep=False).sum() - xdata.duplicated(keep='first').sum() 

# aa = xdata.drop_duplicates(keep='first')
# bb = xdata.drop_duplicates(keep=False)

# aa.index
# bb.index

# len(aa.index)
# len(bb.index)

# cc = []
# for i in aa.index:
#     if i not in bb.index:
#         print(i)
#         cc.append(i)

# for i in cc:
#     print(xdata.iloc[[i]])

# xdata['label'] = ydata_array_list[-1]
# xdata.duplicated(keep='first').sum()

# tmp = xdata[xdata.duplicated(['quarantine', 'quarantine_mean', 'A', 'H','C(A_sum_c)', 'C(H_sum_c)', 'degree'], 
#                              keep=False)].sort_values(['quarantine', 'quarantine_mean', 'A', 'H',
#                                                  'C(A_sum_c)', 'C(H_sum_c)', 'degree'])

# tmp.shape

# tmp.to_csv('tmp.csv')

# # train_loader, valid_loader, test_loader = deep_learner_outcome._data_preprocess([xdata_list, ydata_array_list],
# #                                                                   model_cat_vars=model_cat_vars, 
# #                                                                 model_cont_vars=model_cont_vars, 
# #                                                                 model_cat_unique_levels=model_cat_unique_levels)


# from torch.utils.data import Dataset, DataLoader, Subset
# class TimeSeriesDatasetSeparateNormalize(Dataset):
#     def __init__(self, xy_list, model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={},
#                  normalize=True, drop_duplicates=True):
#         ''' Retrieve train, label and pred data from list of pd.Dataframe (input) and list of np.array (label) directly,
#             Treat categorical variables as numerical float, apply normalization to them and drop duplicates in the dataset
#         Args:  
#             xy_list: list, containing pd.DataFrame data and np.array label, PS, len(xdata_list) does not necassarily equal to len(ydata_list)
#             model_cat_vars: list, categorical variables in patsy_matrix_dataframe, subset of cat_vars
#             model_cont_vars: list, continuous variables in patsy_matrix_dataframe, subset of cont_vars
#             model_cat_unique_levels: dict, number of unique levels for each categorical variable of patsy_matrix_dataframe, subset of cat_unique_levels
#             normalize: normalize the dataframe, treat every column as numerical float
#             drop_duplicates: drop duplicates in the dataset, save only the minimum of data
#         '''
#         xdata_list, ydata_list = xy_list

#         if drop_duplicates:
#             sample_size = []
#             for i in range(len(xdata_list)):
#                 check_duplicates = xdata_list[i].duplicated()
#                 if check_duplicates.sum() > 0:
#                     xdata_list[i] = xdata_list[i].drop_duplicates()
#                     ydata_list[i] = ydata_list[i][~check_duplicates]
#                     sample_size.append(xdata_list[i].shape[0])
#             # select the minimum sample size after dropping duplicates
#             xdata_list = [xdata_list[i][:min(sample_size)] for i in range(len(xdata_list))]
#             ydata_list = [ydata_list[i][:min(sample_size)] for i in range(len(ydata_list))]

#         if normalize:
#             xdata_list = [self._normlize_dataframe(df) for df in xdata_list]

#         # use numerical index to avoid looping inside _getitem_()
#         self.cat_col_index = self._column_name_to_index(xdata_list[-1], model_cat_vars)
#         self.cont_col_index = self._column_name_to_index(xdata_list[-1], model_cont_vars)

#         self.input_data_array = np.stack([df.to_numpy() for df in xdata_list], axis=-1) 
#         # len(xdata_list) = T_in
#         # self.input_data_array: [num_samples, num_features, T_in]
#         self.label_data_array = np.stack(ydata_list, axis=-1)
#         # len(ydata_list) = T_out
#         # self.label_data_array: [num_samples, T_out]
    
#     def _normlize_dataframe(self, dataframe, method='zscore'):
#         if method == 'zscore':
#             return (dataframe - dataframe.mean())/dataframe.std()
#         elif method == 'minmax':
#             return (dataframe - dataframe.min())/(dataframe.max() - dataframe.min())

#     def _column_name_to_index(self, dataframe, column_name):
#         return dataframe.columns.get_indexer(column_name)

#     def __getitem__(self, idx):
#         cat_vars = torch.from_numpy(self.input_data_array[idx, self.cat_col_index, :]).float() # [num_cat_vars, T_in]
#         cont_vars = torch.from_numpy(self.input_data_array[idx, self.cont_col_index, :]).float() # [num_cont_vars, T_in]
#         labels = torch.from_numpy(self.label_data_array[idx, :]).float().squeeze(0) # [1, T_out] -> [T_out]
        
#         return cat_vars, cont_vars, labels, idx # idx shape []

#     def __len__(self):
#         return self.input_data_array.shape[0] # num_samples

# def get_dataloaders(dataset, split_ratio=[0.7, 0.1, 0.2], batch_size=16, shuffle=True):
#     torch.manual_seed(17) # random split with reproducibility

#     train_size = int(split_ratio[0] * len(dataset))
#     test_size = int(split_ratio[-1] * len(dataset))
#     valid_size = len(dataset) - train_size - test_size
#     train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
#     valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, valid_loader, test_loader

# dset = TimeSeriesDatasetSeparateNormalize([xdata_list, ydata_array_list], 
#                                           model_cat_vars_final, model_cont_vars_final, model_cat_unique_levels_final,
#                                           normalize=True, drop_duplicates=True)
# train_loader, valid_loader, test_loader = get_dataloaders(dset, split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True)


# for x_cat, x_cont, y, sample_idx in train_loader:
#     print(x_cat.shape, x_cont.shape, y.shape, sample_idx.shape)
#     break


# ########################### overfit on a instance ############################
# BATCH_SIZE = 32
# LR = 0.0001
# NUM_EPOCHS = 100
# HIDDEN_SIZE = 512
# # device = "cuda"

# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch.optim as optim

# class FullyConnected(nn.Module):
#     def __init__(self, hidden_size=HIDDEN_SIZE):
#         super().__init__()
#         self.fc1 = nn.Linear(in_features=16, out_features=hidden_size)
#         self.fc2 = nn.Linear(in_features=hidden_size, out_features=16)
        
#     def forward(self, x): 
#         out = self.fc1(x)
#         out = F.relu(out)
#         out = self.fc2(out)
#         return out
    
    
# model = FullyConnected().to(device)
# criterion = nn.BCEWithLogitsLoss()
# # criterion = nn.MSELoss()
# # optimizer = optim.Adam(deep_learner_outcome.model.parameters(), lr=LR)
# optimizer = optim.Adam(model.parameters(), lr=LR)
# model.train()

# # x, target = torch.randn(BATCH_SIZE, 16), torch.randn(BATCH_SIZE, 1)
# # x = x.to(device)
# # target = target.to(device)

# x_cat, x_cont, y, sample_idx = next(iter(train_loader))
# # x_cat = torch.randint(0, 6, (16, 6, 1))
# x_cont = torch.randn(16, 1, 1)
# y = torch.empty(y.shape).random_(2)
# x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)

# model.training

# # mean=torch.mean(x_cont)
# # std=torch.std(x_cont)

# # x_cont = torch.randn(16, requires_grad=True).to(device)
# # y = torch.empty(16).random_(2).to(device)


# losses = []
# for epoch in range(NUM_EPOCHS):
#     # x = torch.cat([x_cat, x_cont], dim=1)
#     # output = model(x)
#     # x_cont = (x_cont - mean) / std
#     output = model(x_cont.squeeze())
#     # output = deep_learner_outcome.model(x_cat, x_cont, sample_idx)
#     # loss = criterion(output, y.unsqueeze(-1).unsqueeze(-1))                                                                                                                                                                                 
#     loss = criterion(output, y)                                                                                                                                                                                 
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     print("epoch {}, loss {:.3f}".format(epoch, loss.item()))
#     losses.append(loss.item())
    
# plt.plot(losses)

# pred_binary = torch.round(torch.sigmoid(output)) # get binary prediction
# # acc = (pred_binary == y.unsqueeze(-1).unsqueeze(-1)).sum()/y.numel()
# acc = (pred_binary == y).sum()/y.numel()
# print(f'acc: {acc:.2f}')


# # Test on our model
# class MLPModelTimeSeries(nn.Module):
#     def __init__(self, adj_matrix_list, model_cat_unique_levels, n_cont, T_in=10, T_out=10,
#                  n_output=2, _continuous_outcome=False):
#         super(MLPModelTimeSeries, self).__init__()
#         self.embedding_layers, self.n_emb = self._get_embedding_layers(model_cat_unique_levels)
#         self.n_cont = n_cont

#         self.hidden_size_1 = 128
#         self.hidden_size_2 = 256

#         # variable dimension feature extraction
#         self.lin1 = nn.Linear(self.n_emb + self.n_cont, self.hidden_size_1)
#         self.lin2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
#         # if use BCEloss, number of output should be 1, i.e. the probability of getting category 1
#         # else number of output should be as specified
#         if n_output == 2 or _continuous_outcome:
#             self.lin3 = nn.Linear(self.hidden_size_2, 1) 
#         else:
#             self.lin3 = nn.Linear(self.hidden_size_2, n_output)
#         self.bn1 = nn.BatchNorm1d(self.n_cont)
#         self.bn2 = nn.BatchNorm1d(self.hidden_size_1)
#         self.bn3 = nn.BatchNorm1d(self.hidden_size_2)
#         # self.emb_drop = nn.Dropout(0.6)
#         self.emb_drop = nn.Dropout(0.1)
#         self.drops = nn.Dropout(0.1)

#         # time dimension feature extract 
#         # self.ts_lin1 = nn.Linear(T_in, 16)
#         # self.ts_lin2 = nn.Linear(16, T_out)

#         self._init_weights()

#     def _get_embedding_layers(self, model_cat_unique_levels):
#         # Ref: https://jovian.ml/aakanksha-ns/shelter-outcome
#         # decide embedding sizes
#         embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _, n_categories in model_cat_unique_levels.items()]
#         embedding_layers = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
#         n_emb = sum(e.embedding_dim for e in embedding_layers) # length of all embeddings combined
#         # n_cont = dataset.x_cont.shape[1] # number of continuous variables

#         return embedding_layers, n_emb

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 m.bias.data.fill_(0.01)
#             elif isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.Embedding):
#                 nn.init.uniform_(m.weight)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0) 
    
#     def forward(self, x_cat, x_cont, batched_nodes_indices=None):
#         # x_cat: [batch_size, num_cat_vars, T]
#         # x_cont: [batch_size, num_cont_vars, T]
#         # batched_nodex_indices: [batch_size]

#         x_cat_new = x_cat.permute(0, 2, 1)
#         # x_cat_new: [batch_size, T, num_cat_vars]

#         if len(self.embedding_layers) > 0: # if there are categorical variables to be encoded
#             x1 = [e(x_cat_new[:, :, 1]) for i, e in enumerate(self.embedding_layers)]
#             x1 = torch.cat(x1, -1) # [batch_size, T, n_emb]
#             x1 = self.emb_drop(x1)

#         if self.n_cont > 0: # if there are continuous variables to be encoded
#             x2 = self.bn1(x_cont).permute(0, 2, 1) # [batch_size, T, n_cont]
        
#         if len(self.embedding_layers) > 0 and self.n_cont > 0: # if there are both categorical and continuous variables to be encoded 
#             x = torch.cat([x1, x2], -1) # [batch_size, T, n_emb + n_cont]
#             # temporal perspective
#             # x = F.relu(self.ts_lin1(x.permute(0, 2, 1))).permute(0, 2, 1) 
#             # [batch_size, T, n_emb + n_cont] -> [batch_size, n_emb + n_cont, T] 
#             # -> [batch_size, n_emb + n_cont, 16] -> [batch_size, 16, n_emb + n_cont]

#             # variable perspective
#             x = F.relu(self.lin1(x)) # [batch_size, 16, n_emb + n_cont] -> [batch_size, 16, 16]
#             x = self.drops(x)       
#             x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1) 
#             # [batch_size, 16(ts_c), 16(v_c)] -> [batch_size, 16(v_c), 16(ts_c)] ->  [batch_size, 16(ts_c), 16(v_c)]
#             x = F.relu(self.lin2(x)) # [batch_size, 16, 16] -> [batch_size, 16, 32] 
#             x = self.drops(x)
#             x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
#             # [batch_size, 16, 32] -> [batch_size, 32, 16] -> [batch_size, 16, 32
#             x = self.lin3(x)         # [batch_size, 16, 32] -> [batch_size, 16, 1]

#             # temporal perspective
#             # x = self.ts_lin2(x.permute(0, 2, 1))
#             # [batch_size, 16, 1] -> [batch_size, 1, 16] -> [batch_size, 1, T] 

#         elif len(self.embedding_layers) > 0 and self.n_cont == 0: 
#             # temporal perspective
#             # x = self.ts_lin1(x1.permute(0, 2, 1)).permute(0, 2, 1)
#             x = x1
#             # variable perspective
#             x = F.relu(self.lin1(x))
#             x = self.drops(x)       
#             x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
#             x = F.relu(self.lin2(x))
#             x = self.drops(x)
#             x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
#             x = self.lin3(x)
#             # temporal perspective
#             # x = self.ts_lin2(x.permute(0, 2, 1))

#         elif len(self.embedding_layers) == 0 and self.n_cont > 0:
#             # temporal perspective
#             # x = self.ts_lin1(x2.permute(0, 2, 1)).permute(0, 2, 1)
#             x = x2
#             # variable perspective
#             x = F.relu(self.lin1(x))
#             x = self.drops(x)       
#             x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
#             x = F.relu(self.lin2(x))
#             x = self.drops(x)
#             x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
#             x = self.lin3(x)
#             # temporal perspective
#             # x = self.ts_lin2(x.permute(0, 2, 1))
#         else:
#             raise ValueError('No variables to be encoded')
    
#         return x
    

# class TmpModel(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         # self.embedding = nn.Embedding(41, 21)
#         # self.lin_1 = nn.Linear(21, 256)
#         # self.lin_2 = nn.Linear(256, 512)
#         # self.lin_3 = nn.Linear(512, 1024)
#         # self.lin_4 = nn.Linear(1024, 1)

#         # self.lin_1 = nn.Linear(16, 256)
#         self.lin_input = nn.Linear(7, 32)
#         # self.lin_hidden = nn.ModuleList([nn.Linear(32, 64), nn.Linear(64, 128), nn.Linear(128, 256), nn.Linear(256, 512), 
#         #                                  nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 64), nn.Linear(64, 32)])
#         # self.lin_hidden = nn.ModuleList([nn.Linear(32, 128), nn.Linear(128, 512), nn.Linear(512, 1024), 
#         #                                  nn.Linear(1024, 512), nn.Linear(512, 128), nn.Linear(128, 32)])
#         self.lin_hidden = nn.ModuleList([nn.Linear(32, 128), nn.Linear(128, 512), nn.Linear(512, 128), nn.Linear(128, 32)])
#         self.lin_output = nn.Linear(32, 1)

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 m.bias.data.fill_(0.01)
#             elif isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.Embedding):
#                 nn.init.uniform_(m.weight)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0) 

#     def forward(self, x_cat, x_cont, batched_nodes_indices=None, mean=0., std=1.):
#         # x = self.embedding(x_cat)
#         x1 = x_cat.permute(0, 2, 1).float()
#         x2 = x_cont.permute(0, 2, 1)
#         x =  torch.cat([x1, x2], -1) 
#         # x = (x - mean) / std
#         # x = x_cat.squeeze().float()
#         x = F.relu(self.lin_input(x))
#         for layer in self.lin_hidden:
#             x = F.relu(layer(x))
#         x = self.lin_output(x)
#         # return x.unsqueeze(-1)
#         # x = x * std + mean
#         return x.permute(0, 2, 1)


# class TmpModel(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         # feature dim
#         self.lin_input = nn.Linear(7, 32)
#         self.lin_hidden = nn.ModuleList([nn.Linear(32, 128), nn.Linear(128, 512), nn.Linear(512, 128), nn.Linear(128, 32)])
#         self.lin_output = nn.Linear(32, 1)
#         # temporal dim
#         self.lin_input_temporal = nn.Linear(10, 128)
#         self.lin_output_temporal = nn.Linear(128, 10)

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 m.bias.data.fill_(0.01)
#             elif isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.Embedding):
#                 nn.init.uniform_(m.weight)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0) 

#     def forward(self, x_cat, x_cont, batched_nodes_indices=None):
#         # x = self.embedding(x_cat)
#         x1 = x_cat.permute(0, 2, 1).float()
#         x2 = x_cont.permute(0, 2, 1)
#         x =  torch.cat([x1, x2], -1) 
#         # x = (x - mean) / std
#         # x = x_cat.squeeze().float()
#         x = F.relu(self.lin_input(x))
#         for layer in self.lin_hidden:
#             x = F.relu(layer(x))
#         x = self.lin_output(x)

#         x = F.relu(self.lin_input_temporal(x.permute(0, 2, 1)))
#         x = self.lin_output_temporal(x)
#         # return x.unsqueeze(-1)
#         # x = x * std + mean
#         return x


# # BATCH_SIZE = 32
# LR = 0.0001
# NUM_EPOCHS = 300
# # HIDDEN_SIZE = 512

# # model = MLPModelTimeSeries(None, model_cat_unique_levels_final, n_cont=1, T_in=10, T_out=10,
# #                            n_output=2, _continuous_outcome=False).to(device)
# model = TmpModel().to(device)


# criterion = nn.BCEWithLogitsLoss()
# # criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=LR)
# model.train()

# # x_cat, x_cont, y, sample_idx = next(iter(train_loader))
# # # x_cat = torch.randint(0, 6, (16, 6, 1))
# # # x_cont = torch.randn(16, 1, 1)
# # # y = torch.empty(y.shape).random_(2)
# # x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)
# # # x_cat_slice = x_cat[:, -1, :].squeeze()

# # x_mean_ls = []
# # x_std_ls = []
# # for x_cat, x_cont, y, sample_idx in train_loader:
# #     x1 = x_cat.permute(0, 2, 1).float()
# #     x2 = x_cont.permute(0, 2, 1)
# #     x =  torch.cat([x1, x2], -1)
# #     x_mean = torch.mean(x, dim=0)
# #     x_std = torch.std(x, dim=0)
# #     x_mean_ls.append(x_mean)
# #     x_std_ls.append(x_std)

# # mean = torch.sum(torch.concat(x_mean_ls, dim=0), dim=0) / len(x_mean_ls)
# # std = torch.sum(torch.concat(x_std_ls, dim=0), dim=0) / len(x_std_ls)
# # mean, std = mean.to(device), std.to(device)

# epoch_losses = []
# for epoch in range(NUM_EPOCHS):
#     losses = []
#     accs = []
#     for x_cat, x_cont, y, sample_idx in train_loader:
#         x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)
#         # output = model(x_cat, x_cont, sample_idx, mean, std)
#         output = model(x_cat, x_cont, sample_idx)
#         # output = model(x_cat)
#         # loss = criterion(output, y.unsqueeze(-1).unsqueeze(-1))      
#         loss = criterion(output, y.unsqueeze(1))      


#         # output = model(x_cat_slice)                                                                                                                                                                              
#         # loss = criterion(output, y.unsqueeze(-1))                   

#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         # print("epoch {}, loss {:.3f}".format(epoch, loss.item()))
#         losses.append(loss.item())  

#         pred_binary = torch.round(torch.sigmoid(output)) # get binary prediction
#         # acc = (pred_binary == y.unsqueeze(-1).unsqueeze(-1)).sum()/y.numel()
#         acc = (pred_binary == y.unsqueeze(1)).sum()/y.numel()
#         accs.append(acc.item())
#         # acc = (pred_binary.squeeze() == y).sum()/y.numel()
#         # print(f'acc: {acc:.2f}')
#         epoch_losses.extend(losses)
#     print(f'epoch: {epoch}, loss: {np.mean(losses)}, acc: {np.mean(accs):.2f}')
    
# plt.plot(epoch_losses)





# ntmle.fit(p=0.7, bound=0.01, seed=seed_number,
#           shift=shift, mode=mode, percent_candidates=percent_candidates,
#           T_in_id=[*range(10)], T_out_id=[9])

# ntmle._Qinit_.shape

# pred = torch.from_numpy(ntmle._Qinit_)
# torch.max(pred)
# torch.min(pred)
# # pred = torch.sigmoid(pred) # get binary probability
# pred_binary = torch.round(pred) # get binary prediction
# labels = ntmle.df_restricted_list[-1][ntmle.outcome]
# labels = torch.from_numpy(labels.to_numpy())
# acc = (pred_binary == labels).sum()/labels.numel()
# print(f'self._Qinit_ accuracy: {acc:.3f}')

# zero_ratio = 1 - labels.sum()/labels.numel()
# print(f'zero_ratio: {zero_ratio:.3f}')

# labels.sum()

# ''' Observed Data
# uniform i=4
# q_estimator accuracy: 0.820
# DL accuracy: 0.182

# random i=0
# q_estimator accuracy: 0.6
# DL accuracy: 0.616
# '''

# for i in range(10):
#     labels = ntmle.df_restricted_list[i][ntmle.outcome]
#     labels = torch.from_numpy(labels.to_numpy())
#     print(labels)
#     print()


# def get_accuracy(pred:np.ndarray, label:pd.core.series.Series):
#     pred = torch.from_numpy(pred)
#     pred_binary = torch.round(pred)
#     label = torch.from_numpy(label.to_numpy())
#     acc = (pred_binary == label).sum()/label.numel()
#     print(f'acc: {acc:.2f}')
#     return pred_binary, label, acc

# ############### fit() ################
# from tmle_utils import targeting_step, get_patsy_for_model_w_C, outcome_deep_learner_ts, outcome_learner_predict, tmle_unit_unbound
# import patsy
# p=0.7
# samples = 500
# bound = 0.01
# seed = seed_number


# # Step 1) Estimate the weights
# # Also generates pooled_data for use in the Monte Carlo integration procedure
# ntmle._resamples_ = samples                                                              # Saving info on number of resamples
# h_iptw, pooled_data_restricted_list, pooled_adj_matrix_list = ntmle._estimate_iptw_ts_(p=p,                      # Generate pooled & estiamte weights
#                                                                                        samples=samples,           # ... for some number of samples
#                                                                                        bound=bound,               # ... with applied probability bounds
#                                                                                        seed=seed,
#                                                                                        shift=shift,
#                                                                                        mode=mode,
#                                                                                        percent_candidates=percent_candidates)                 # ... and with a random seed given

# # Saving some information for diagnostic procedures
# if ntmle._gs_measure_ is None:                                  # If no summary measure, use the A_sum
#     ntmle._for_diagnostics_ = pooled_data_restricted_list[-1][[ntmle.exposure, ntmle.exposure+"_sum"]].copy()
# else:                                                          # Otherwise, use the provided summary measure
#     ntmle._for_diagnostics_ = pooled_data_restricted_list[-1][[ntmle.exposure, ntmle._gs_measure_]].copy()

# # Step 2) Estimate from Q-model
# # process completed in .outcome_model() function and stored in self._Qinit_
# # so nothing to do here

# # Step 3) Target the parameter
# epsilon = targeting_step(y=ntmle.df_restricted_list[-1][ntmle.outcome],   # Estimate the targeting model given observed Y
#                             q_init=ntmle._Qinit_,                           # ... predicted values of Y under observed A
#                             ipw=h_iptw,                                    # ... weighted by IPW
#                             verbose=ntmle._verbose_)                        # ... with option for verbose info

# # Step 4) Monte Carlo integration (old code did in loop but faster as vector)
# #
# # Generating outcome predictions under the policies (via pooled data sets)
# if ntmle._q_custom_ is None:                                                     # If given a parametric default model
#     y_star = ntmle._outcome_model.predict(pooled_data_restricted_list[-1])       # ... predict using statsmodels syntax
# else:  # Custom input model by user
#     if ntmle.use_deep_learner_outcome:
#         xdata_list = []
#         ydata_list = []
#         n_output_list = []
#         for pooled_data_restricted in pooled_data_restricted_list:
#             if 'C(' in ntmle._q_model:
#                 xdata_list.append(get_patsy_for_model_w_C(ntmle._q_model, pooled_data_restricted))
#             else:
#                 xdata_list.append(patsy.dmatrix(ntmle._q_model + ' - 1', pooled_data_restricted, return_type="dataframe"))
#             ydata_list.append(pooled_data_restricted[ntmle.outcome])
#             n_output_list.append(pd.unique(pooled_data_restricted[ntmle.outcome]).shape[0])

#         y_star = outcome_deep_learner_ts(ntmle._q_custom_, 
#                                             xdata_list, ydata_list, [*range(10)], [9],
#                                             pooled_adj_matrix_list, ntmle.cat_vars, ntmle.cont_vars, ntmle.cat_unique_levels, n_output_list, ntmle._continuous_outcome_list_[-1],
#                                             predict_with_best=True, custom_path=ntmle._q_custom_path_)
#     else:
#         d = patsy.dmatrix(ntmle._q_model + ' - 1', pooled_data_restricted_list[-1])  # ... extract data via patsy
#         y_star = outcome_learner_predict(ml_model_fit=ntmle._q_custom_,              # ... predict using custom function
#                                         xdata=np.asarray(d))                        # ... for the extracted data


# pred_binary, labels, acc = get_accuracy(y_star, pooled_data_restricted_list[-1][ntmle.outcome])
# pred_binary.sum()
# labels.sum()
# labels.numel()
# 1-labels.sum()/labels.numel()
    
# # Ensure all predicted values are bounded properly for continuous
# # SG modified: continous outcome is already normalized, should compare with 0,1, not with _continuous_min/max_
# if ntmle._continuous_outcome_list_[-1]:
#     y_star = np.where(y_star < 0., 0. + ntmle._cb_list_[-1], y_star)
#     y_star = np.where(y_star > 1., 1. - ntmle._cb_list_[-1], y_star)     

# # if self._continuous_outcome_list_[-1]:
# #     y_star = np.where(y_star < self._continuous_min_list_[-1], self._continuous_min_list_[-1], y_star)
# #     y_star = np.where(y_star > self._continuous_max_list_[-1], self._continuous_max_list_[-1], y_star)

# # Updating predictions via intercept from targeting step
# logit_qstar = np.log(probability_to_odds(y_star)) + epsilon                         # NOTE: needs to be logit(Y^*) + e
# q_star = odds_to_probability(np.exp(logit_qstar))                                   # Back converting from odds
# pooled_data_restricted_list[-1]['__pred_q_star__'] = q_star                         # Storing predictions as column

# # Taking the mean, grouped-by the pooled sample IDs (fast)
# ntmle.marginals_vector = np.asarray(pooled_data_restricted_list[-1].groupby('_sample_id_')['__pred_q_star__'].mean())

# # If continuous outcome, need to unbound the means
# if ntmle._continuous_outcome_list_[-1]:
#     ntmle.marginals_vector = tmle_unit_unbound(ntmle.marginals_vector,                    # Take vector of MC results
#                                                 mini=ntmle._continuous_min_list_[-1],      # ... unbound using min
#                                                 maxi=ntmle._continuous_max_list_[-1])      # ... and max values

# # Calculating estimate for the policy
# ntmle.marginal_outcome = np.mean(ntmle.marginals_vector)                                  # Mean of Monte Carlo results
# ntmle._specified_p_ = p                                                                  # Save what the policy was

# # Prep for variance
# if ntmle._continuous_outcome_list_[-1]:                                                  # Continuous needs bounds...
#     y_ = np.array(tmle_unit_unbound(ntmle.df_restricted_list[-1][ntmle.outcome],          # Unbound observed outcomes for Var
#                                     mini=ntmle._continuous_min_list_[-1],                # ... using min
#                                     maxi=ntmle._continuous_max_list_[-1]))               # ... and max values
#     yq0_ = tmle_unit_unbound(ntmle._Qinit_,                                              # Unbound g-comp predictions
#                                 mini=ntmle._continuous_min_list_[-1],                       # ... using min
#                                 maxi=ntmle._continuous_max_list_[-1])                       # ... and max values
# else:                                                                                   # Otherwise nothing special...
#     y_ = np.array(ntmle.df_restricted_list[-1][ntmle.outcome])                            # Observed outcome for Var
#     yq0_ = ntmle._Qinit_                                                                 # Predicted outcome for Var

# print(f'{ntmle.marginal_outcome} - {truth[p]} = {ntmle.marginal_outcome-truth[p]}')

# ''' Pooled Data
# p=0.7

# uniform
# i=0
# q_estimator acc = 0.91
# bias = 0.10550313948472169 - 0.126 = -0.020496860515278312
# DL acc = 0.84
# bias = 0.103775404 - 0.126 = -0.022224595606327058

# i=4
# q_estimator acc = 0.82
# bias = 0.173819749295999 - 0.126 = 0.047819749295999
# DL acc = 0.19
# bias = 0.18217044 - 0.126 = 0.05617043578624725

# random
# i=0
# q_estimator acc = 0.58
# bias = 0.5226838016094336 - 0.512 = 0.010683801609433607
# DL acc = 0.33
# bias = 0.539951741695404 - 0.512 = 0.027951741695404042

# '''
