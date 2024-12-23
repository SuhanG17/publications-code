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
from dl_trainer_time_series_UDA import MLPTS_UDA

# Excel sheet: no shift_mode=all_t=9
############################################
# Setting simulation parameters
############################################
# n_mc = 500
n_mc = 15
# n_mc = 1

exposure = "quarantine"
outcome = "D"
parallel_id = 0

########################################
# Running through logic from .sh script
########################################
parser = argparse.ArgumentParser(description='DLnetworkTMLE')
parser.add_argument('--task_string', type=str, required=True, default='10010',
                    help='the slurm_setup id in string format')
parser.add_argument('--use_deep_learner_outcome', action='store_true', 
                    help='default to false, if given, use deep learner for outcome model')
args = parser.parse_args()
# # test run with dummy args
# class Args(object):
#     def __init__(self):
#         self.task_string = '20040'
#         self.use_deep_learner_outcome = False
# args = Args()


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

# choose which what-if mechanism to test
mode = 'all'
percent_candidates = 0.5
quarantine_period = 2
inf_duration = 5

# decide use deep learning in which nuisance model
use_deep_learner_A_i = False
use_deep_learner_A_i_s = False 
use_deep_learner_outcome = args.use_deep_learner_outcome 
# dl observed acc: 0.344
# dl pooled acc: 0.934
# dl bias: -0.019093970626592634

# lr observed acc: 0.842
# lr pooled acc: 0.346
# lr bias: 0.5108185131009413


# T_in_id = [6, 7, 8, 9]
T_in_id = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# T_in_id = [8, 9]
T_out_id = [9]


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
    qn_model = "quarantine + quarantine_sum + I_ratio + I_ratio_sum + A + H + A_sum + H_sum + degree"

elif model == "cw":
    # gin_model = "A + H + A_sum + H_sum + degree"
    # gsn_model = "quarantine + A + H + A_sum + H_sum + degree"
    # qn_model = "quarantine + quarantine_mean + A + H + H_t3 + degree"

    gin_model = "A + H + A_sum + H_sum + I_ratio"
    gsn_model = "quarantine + A + H + A_sum + H_sum + I_ratio"
    # qn_model = "quarantine + quarantine_mean + I_ratio + I_ratio_mean + A + H + H_t3 + degree" 
    qn_model = "quarantine + quarantine_mean + I_ratio + I_ratio_mean + A + H + H_t2 + degree" 
elif model == "wc":
    # gin_model = "A + H + H_t3 + degree"
    # gsn_model = "quarantine + A + H + H_t3 + degree"
    # qn_model = "quarantine + quarantine_mean + A + H + A_sum + H_sum + degree"

    # gin_model = "A + H + H_t3 + I_ratio"
    gin_model = "A + H + H_t2 + I_ratio"
    # gsn_model = "quarantine + A + H + H_t3 + I_ratio"
    gsn_model = "quarantine + A + H + H_t2 + I_ratio"
    qn_model = "quarantine + quarantine_sum + I_ratio + I_ratio_sum + A + H + A_sum + H_sum + degree" 
elif model == 'np':
    # gin_model = "A + H + C(A_sum_c) + C(H_sum_c) + degree"
    # gsn_model = "quarantine + A + H + C(A_sum_c) + C(H_sum_c) + degree"
    # qn_model = "quarantine + quarantine_mean + A + H + C(A_sum_c) + C(H_sum_c) + degree"
    # q_estimator = LogisticRegression(penalty='l2', max_iter=2000)
    
    gin_model = "A + H + C(A_sum_c) + C(H_sum_c) + I_ratio"
    gsn_model = "quarantine + A + H + C(A_sum_c) + C(H_sum_c) + I_ratio"
    qn_model = "quarantine + quarantine_sum + I_ratio + I_ratio_sum + A + H + C(A_sum_c) + C(H_sum_c) + degree"
    if not use_deep_learner_outcome:
        q_estimator = LogisticRegression(penalty='l2', max_iter=2000)
elif model == 'ind':
    independent = True
    gi_model = "A + H"
    qi_model = "quarantine + A + H"

# Determining if shift or absolute
if shift:
    prop_treated = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]

    # # Generating probabilities (true) to assign
    # data = network_to_df(G)
    # adj_matrix = nx.adjacency_matrix(G, weight=None)
    # data['A_sum'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='sum')
    # data['H_sum'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='sum')
    # data = pd.merge(data, pd.DataFrame.from_dict(dict(G.degree),
    #                                              orient='index').rename(columns={0: 'degree'}),
    #                 how='left', left_index=True, right_index=True)
    # data['I_sum'] = fast_exp_map(adj_matrix, np.array(data['I']), measure='sum')
    # data['I_ratio'] = data['I_sum'] / data['degree'] # ratio of infected neighbors
    # prob = logistic.cdf(- 4.5 
    #                     + 1.2*data['A'] + 0.8*data['H']
    #                     + 0.5*data['H_sum'] + 0.3*data['A_sum'] 
    #                     + 1.2*data['I_ratio'])
    # log_odds = np.log(probability_to_odds(prob))

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
    # i=1
    print(f'simulation {i}')
    ######## inside for loop ########
    # Generating Data
    # H = vaccine_dgm(network=G, restricted=restrict)
    H, network_list, cat_vars_i, cont_vars_i, cat_unique_levels_i  = quarantine_dgm_time_series(network=G, restricted=restrict, 
                                                                                                time_limit=10, inf_duration=5, quarantine_period=quarantine_period,
                                                                                                update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
                                                                                                random_seed=seed_number+i)
    df = network_to_df(H)
    results.loc[i, 'inc_'+exposure] = np.mean(df[exposure])
    results.loc[i, 'inc_'+outcome] = np.mean(df[outcome])

    if shift:
        # Generating probabilities (true) to assign
        adj_matrix = nx.adjacency_matrix(H, weight=None)
        df['A_sum'] = fast_exp_map(adj_matrix, np.array(df['A']), measure='sum')
        df['H_sum'] = fast_exp_map(adj_matrix, np.array(df['H']), measure='sum')
        df = pd.merge(df, pd.DataFrame.from_dict(dict(G.degree),
                                                 orient='index').rename(columns={0: 'degree'}),
                        how='left', left_index=True, right_index=True)
        df['I_sum'] = fast_exp_map(adj_matrix, np.array(df['I']), measure='sum')
        df['I_ratio'] = df['I_sum'] / df['degree'] # ratio of infected neighbors
        prob = logistic.cdf(- 4.5 
                            + 1.2*df['A'] + 0.8*df['H']
                            + 0.5*df['H_sum'] + 0.3*df['A_sum'] 
                            + 1.2*df['I_ratio'])
        log_odds = np.log(probability_to_odds(prob))

    # Network TMLE
    # use deep learner for given nuisance model
    ntmle = NetworkTMLETimeSeries(network_list, exposure='quarantine', outcome='D', verbose=False, degree_restrict=degree_restrict, _gs_measure_=measure_gs,
                                    task_string=args.task_string, parallel_id=parallel_id,
                                    cat_vars=cat_vars_i, cont_vars=cont_vars_i, cat_unique_levels=cat_unique_levels_i,
                                    use_deep_learner_A_i=use_deep_learner_A_i, 
                                    use_deep_learner_A_i_s=use_deep_learner_A_i_s, 
                                    use_deep_learner_outcome=use_deep_learner_outcome,
                                    use_all_time_slices=False) 

    if model in ["cw", "wc"]:
        ntmle.define_threshold(variable='H', threshold=2, definition='sum')
    if model == "np":
        if mode == 'all':
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
                    # ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 6], labels=False)
                    # ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 6], labels=False)
            elif network == "random":
                if n_nodes == 500:
                    # ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 4, 10], labels=False)
                    # ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 4, 7, 18], labels=False)
                    ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 10], labels=False)
                    ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 4, 18], labels=False)
                elif n_nodes == 1000:
                    # ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 10], labels=False)
                    # ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 8, 26], labels=False)
                    ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 10], labels=False)
                    ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 26], labels=False) 
                else:
                    # ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 3, 4, 5, 10], labels=False)
                    # ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 6, 9, 26], labels=False)
                    ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 3, 10], labels=False)
                    ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 6, 26], labels=False)
            else:
                raise ValueError("Invalid model-network combo")
        elif mode == 'top':
            if network == "uniform":
                if n_nodes == 500:
                    ntmle.define_category(variable='A_sum', bins=[0, 1, 3], labels=False)
                    ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 6], labels=False)
                elif n_nodes == 1000:
                    ntmle.define_category(variable='A_sum', bins=[0, 1, 5], labels=False)
                    ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 6], labels=False)
                else:
                    # ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 3, 6], labels=False)
                    # ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 6], labels=False)
                    ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 6], labels=False)
                    ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 6], labels=False)
            elif network == "random":
                if n_nodes == 500:
                    # ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 4, 10], labels=False)
                    # ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 4, 7, 18], labels=False)
                    ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 10], labels=False)
                    ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 4, 7, 18], labels=False)
                elif n_nodes == 1000:
                    ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 10], labels=False)
                    ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 8, 26], labels=False)
                    # ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 10], labels=False)
                    # ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 26], labels=False)
                else:
                    # ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 3, 4, 5, 10], labels=False)
                    # ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 6, 9, 26], labels=False)
                    ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 3, 10], labels=False)
                    ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 6, 26], labels=False)
            else:
                raise ValueError("Invalid model-network combo")
        elif mode == 'bottom':
            if network == "uniform":
                if n_nodes == 500:
                    ntmle.define_category(variable='A_sum', bins=[0, 1, 3], labels=False)
                    ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 6], labels=False)
                elif n_nodes == 1000:
                    ntmle.define_category(variable='A_sum', bins=[0, 1, 5], labels=False)
                    ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 6], labels=False)
                else:
                    # ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 3, 6], labels=False)
                    # ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 6], labels=False)
                    ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 6], labels=False)
                    ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 6], labels=False)
            elif network == "random":
                if n_nodes == 500:
                    # ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 4, 10], labels=False)
                    ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 10], labels=False)
                    ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 4, 7, 18], labels=False)
                elif n_nodes == 1000:
                    # ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 10], labels=False)
                    # ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 8, 26], labels=False)
                    ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 10], labels=False)
                    ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 26], labels=False)
                else:
                    # ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 3, 4, 5, 10], labels=False)
                    # ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 6, 9, 26], labels=False)
                    # ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 3, 10], labels=False)
                    # ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 4, 6, 26], labels=False)
                    ntmle.define_category(variable='A_sum', bins=[0, 1, 2, 10], labels=False)
                    ntmle.define_category(variable='H_sum', bins=[0, 1, 2, 3, 26], labels=False)
            else:
                raise ValueError("Invalid model-network combo")
        else:
            raise ValueError("Invalid mode")
            
    # ntmle.exposure_model(gin_model)
    # ntmle.exposure_map_model(gsn_model, measure=measure_gs, distribution=distribution_gs)
    # ntmle.outcome_model(qn_model, custom_model=q_estimator)

    # use deep learner
    def get_deep_learner(deep_learner_type, device):
        '''Return deep learner model based on the type of deep learner'''
        if deep_learner_type == 'mlp':
            split_ratio = [0.8, 0.2]
            deep_learner = MLPTS_UDA(split_ratio=split_ratio, batch_size=16, shuffle=True, n_splits=1, predict_all=True,
                                     epochs=20, print_every=5, device=device, save_path='./tmp.pth',
                                     lin_hidden=None, 
                                     lin_hidden_temporal=nn.ModuleList([nn.Linear(128, 256), nn.Linear(256, 128)]), 
                                     class_classifier=nn.Sequential(nn.Linear(32, 16), nn.ReLU(),
                                                                    nn.Linear(16, 8), nn.ReLU(),
                                                                    nn.Linear(8, 2)),
                                     domain_classifier=nn.Sequential(nn.Linear(1*32, 16), nn.ReLU(), 
                                                                     nn.Linear(16, 8), nn.ReLU(), 
                                                                     nn.Linear(8, 2)))
        elif deep_learner_type == 'gcn':
            pass
        elif deep_learner_type == 'cnn':
            pass
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
        ntmle.outcome_model(qn_model, custom_model=deep_learner_outcome)
    
    # Report outcome model accuracy for Linear Regression model
    # LogsticRegression model does not set _outcome_model but _q_custom_
    if not use_deep_learner_outcome and q_estimator is None:
        ntmle._outcome_model.summary()

    # p=0.05

    for p in prop_treated:  # loops through all treatment plans
        print(f'p={p}')
        # try:
        if shift:
            z = odds_to_probability(np.exp(log_odds + p))
            ntmle.fit(p=z, bound=0.01, seed=seed_number+i, 
                        shift=shift, mode=mode, percent_candidates=percent_candidates, quarantine_period=quarantine_period, inf_duration=inf_duration, 
                        T_in_id=T_in_id, T_out_id=T_out_id)
        else:
            ntmle.fit(p=p, samples=10, bound=0.01, seed=seed_number+i,
                        shift=shift, mode=mode, percent_candidates=percent_candidates, quarantine_period=quarantine_period, inf_duration=inf_duration,
                        T_in_id=T_in_id, T_out_id=T_out_id)
        results.loc[i, 'bias_'+str(p)] = ntmle.marginal_outcome - truth[p]
        results.loc[i, 'var_'+str(p)] = ntmle.conditional_variance
        results.loc[i, 'lcl_'+str(p)] = ntmle.conditional_ci[0]
        results.loc[i, 'ucl_'+str(p)] = ntmle.conditional_ci[1]
        results.loc[i, 'varl_'+str(p)] = ntmle.conditional_latent_variance
        results.loc[i, 'lcll_'+str(p)] = ntmle.conditional_latent_ci[0]
        results.loc[i, 'ucll_'+str(p)] = ntmle.conditional_latent_ci[1]
        # except:
        #     results.loc[i, 'bias_'+str(p)] = np.nan
        #     results.loc[i, 'var_'+str(p)] = np.nan
        #     results.loc[i, 'lcl_'+str(p)] = np.nan
        #     results.loc[i, 'ucl_'+str(p)] = np.nan
        #     results.loc[i, 'varl_'+str(p)] = np.nan
        #     results.loc[i, 'lcll_'+str(p)] = np.nan
        #     results.loc[i, 'ucll_'+str(p)] = np.nan
    ####### inside for loop ########
    print()

    # ########################################
    # # Saving results
    # ########################################
    # if i == 0:
    #     if use_deep_learner_outcome:
    #         results.to_csv("../results/" + exposure + str(sim_id) + "_" + save + "_DL_" +  args.task_string + ".csv", mode='w', index=False)
    #     else:
    #         results.to_csv("../results/" + exposure + str(sim_id) + "_" + save + "_LR_" +  args.task_string + ".csv", mode='w', index=False)
    # else:
    #     if use_deep_learner_outcome:
    #         results.to_csv("../results/" + exposure + str(sim_id) + "_" + save + "_DL_" +  args.task_string + ".csv", mode='a', index=False)
    #     else:
    #         results.to_csv("../results/" + exposure + str(sim_id) + "_" + save + "_LR_" +  args.task_string + ".csv", mode='a', index=False)

# # TEST BEGIN
# samples = 1
# seed = seed_number+i
# pooled_data_restricted_list, pooled_adj_matrix_list = ntmle._generate_pooled_samples(samples=samples,
#                                                                                     seed=seed,
#                                                                                     shift=shift,
#                                                                                     mode=mode,
#                                                                                     percent_candidates=percent_candidates,
#                                                                                     pr_a=p,
#                                                                                     quarantine_period=quarantine_period,
#                                                                                     inf_duration=inf_duration)


# import patsy
# obs_d = patsy.dmatrix(ntmle._q_model + ' - 1', ntmle.df_restricted_list[-1], return_type='dataframe') 
# obs_d.columns
# obs_d_array = np.array(obs_d)
# obs_d_array.shape

# d = patsy.dmatrix(ntmle._q_model + ' - 1', ntmle.pooled_data_restricted_list[-1], return_type='dataframe')
# d.columns
# d_array = np.array(d)
# d_array.shape

# data = ntmle.df_restricted_list[-1]
# data = ntmle.pooled_data_restricted_list[-1] 
# # # # data = pooled_data_restricted_list[-1] 

# var_name = 'H_sum'
# data[var_name].unique()
# data[var_name].value_counts()

# variables = [var_name]
# # bins = [[0, 1, 3]]
# # bins = [[0, 1, 2, 6]]
# bins = [[0, 1, 2, 4, 10]]
# bins = [[0, 1, 2, 10]]
# bins = [[0, 1, 2, 4, 7, 18]]
# bins = [[0, 1, 2, 4, 18]]
# # bins = [[0, 1, 5]]
# # bins = [[0, 1, 2, 3, 6]]
# # bins = [[0, 1, 2, 10]]
# # bins = [[0, 1, 2, 3, 4, 8, 26]]
# # bins = [[0, 1, 2, 3, 6]]
# # bins = [[0, 1, 2, 6]]
# # bins = [[0, 1, 2, 3, 4, 6]]
# # bins = [[0, 1, 2, 3, 4, 5, 10]]
# bins = [[0, 1, 2, 3, 10]]
# # bins = [[0, 1, 2, 3, 4, 6, 9, 26]]
# bins = [[0, 1, 2, 3, 4, 6, 26]]

# labels = [False]
# for v, b, l in zip(variables, bins, labels):
#     print(v)
#     print(b)
#     print(l)
#     # col_label = v + '_c'
#     # data[col_label] = pd.cut(data[v],
#     #                          bins=b,
#     #                          labels=l,
#     #                          include_lowest=True).astype(float)

# out = pd.cut(data[v],
#              bins=b,
#              labels=l,
#              include_lowest=True).astype(float)
# out.unique()
# out.value_counts()


# import statsmodels.api as sm

# y=ntmle.df_restricted_list[-1][ntmle.outcome]
# q_init=ntmle._Qinit_DL_
# ipw=ntmle.h_iptw
# verbose=False

# (q_init <= 0.005).sum()
# (q_init <= 0.995).sum()

# ipw[q_init <= 0.005].mean()
# ipw[q_init >= 0.995].mean()

# # ntmle._Qinit_.min()
# # ntmle._Qinit_.max()
# # q_init = ntmle._Qinit_
# q_init = np.clip(q_init, 0.2, 0.85)
# q_init
# # ipw

# np.log(probability_to_odds(0.05))
# np.log(probability_to_odds(0.95))
# np.exp(6)


# f = sm.families.family.Binomial()
# log = sm.GLM(y,  # Outcome / dependent variable
#              np.repeat(1, y.shape[0]),  # Generating intercept only model
#              offset=np.log(probability_to_odds(q_init)),  # Offset by g-formula predictions
#              freq_weights=ipw,  # Weighted by calculated IPW
#              family=f).fit(maxiter=500)

# epsilon = log.params[0]
# epsilon = -0.33
# print(f'epsilon: {epsilon}')

# np.exp(0)
# q_star = (ntmle.y_star * np.exp(epsilon)) / (1 - ntmle.y_star + ntmle.y_star * np.exp(epsilon))
# ntmle.pooled_data_restricted_list[-1]['__pred_q_star__'] = q_star 
# marginals_vector = np.asarray(ntmle.pooled_data_restricted_list[-1].groupby('_sample_id_')['__pred_q_star__'].mean())
# marginal_outcome = np.mean(marginals_vector) 
# print(f'marginal_outcome: {marginal_outcome}')
# bias = marginal_outcome - truth[p]
# print(f'bias: {bias}')

# log.params
# log.summary()

# TEST END




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
# parallel_id
if use_deep_learner_outcome:
    avg_df.to_csv("../avg_sims_results/" + exposure + str(sim_id) + "_" + save + "_DL_" +  args.task_string + "_" + str(parallel_id) + ".csv", index=False)
else:
    avg_df.to_csv("../avg_sims_results/" + exposure + str(sim_id) + "_" + save + "_LR_" +  args.task_string + "_" + str(parallel_id) + ".csv", index=False)

#######################################
# Saving results
#######################################
if use_deep_learner_outcome:
    results.to_csv("../results/" + exposure + str(sim_id) + "_" + save + "_DL_" +  args.task_string + "_" + str(parallel_id) + ".csv", index=False)
else:
    results.to_csv("../results/" + exposure + str(sim_id) + "_" + save + "_LR_" +  args.task_string + "_" + str(parallel_id) + ".csv", index=False)
