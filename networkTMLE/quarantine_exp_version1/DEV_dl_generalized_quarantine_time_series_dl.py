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
# n_mc = 30
n_mc = 1

exposure = "quarantine"
outcome = "D"

########################################
# Running through logic from .sh script
########################################
# parser = argparse.ArgumentParser(description='DLnetworkTMLE')
# parser.add_argument('--task_string', type=str, required=True, default='10010',
#                         help='the slurm_setup id in string format')
# args = parser.parse_args()
class Args(object):
    def __init__(self):
        self.task_string = '10010'
args = Args()


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
mode = 'all'
percent_candidates = 0.3
quarantine_period = 2
inf_duration = 5
finetune = False

# decide use deep learning in which nuisance model
use_deep_learner_A_i = False
use_deep_learner_A_i_s = False 
use_deep_learner_outcome = True 
# dl no ft bias: 0.8157051067352294
# dl ft with truth bias: 0.10792516279220582
# dl ft with LR pred bias: 0.6595336060523986 
# lr bias: 0.5108185131009413


T_in_id = [6, 7, 8, 9]
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
    ntmle = NetworkTMLETimeSeries(network_list, exposure='quarantine', outcome='D', verbose=False, degree_restrict=degree_restrict,
                                    task_string=args.task_string,
                                    cat_vars=cat_vars_i, cont_vars=cont_vars_i, cat_unique_levels=cat_unique_levels_i,
                                    use_deep_learner_A_i=use_deep_learner_A_i, 
                                    use_deep_learner_A_i_s=use_deep_learner_A_i_s, 
                                    use_deep_learner_outcome=use_deep_learner_outcome,
                                    use_all_time_slices=False) 

    if model in ["cw", "wc"]:
        ntmle.define_threshold(variable='H', threshold=2, definition='sum')
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
            deep_learner = MLPTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=1, predict_all=True,
                                epochs=25, print_every=5, device=device, save_path='./tmp.pth',
                                lin_hidden=None,
                                lin_hidden_temporal=nn.ModuleList([nn.Linear(128, 256), nn.Linear(256, 128)]))
            # deep_learner = MLPTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=1, predict_all=True,
            #                     epochs=25, print_every=5, device=device, save_path='./tmp.pth',
            #                     lin_hidden=nn.ModuleList([nn.Linear(32, 128), nn.Linear(128, 512), nn.Linear(512, 1024),
            #                                               nn.Linear(1024, 512), nn.Linear(512, 128), nn.Linear(128, 32)]),
            #                     lin_hidden_temporal=nn.ModuleList([nn.Linear(128, 256), nn.Linear(256, 128)]))
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
        ntmle.outcome_model(qn_model, custom_model=deep_learner_outcome, finetune=finetune, T_in_id=T_in_id, T_out_id=T_out_id)
    
    # Report outcome model accuracy
    if not use_deep_learner_outcome:
        ntmle._outcome_model.summary()

    label = ntmle.df_restricted_list[-1][ntmle.outcome]
    pred_binary = np.round(ntmle._Qinit_)
    acc = (pred_binary == label).sum().item()/label.shape[0]
    print(f'Outcome model accuracy in observed data: {acc}')
    # from sklearn.metrics import confusion_matrix
    # n = confusion_matrix(label, pred_binary)
    # n

    p=0.75

    for p in prop_treated:  # loops through all treatment plans
        print(f'p={p}')
        try:
            if shift:
                z = odds_to_probability(np.exp(log_odds + p))
                ntmle.fit(p=z, bound=0.01, seed=seed_number+i, 
                          shift=shift, mode=mode, percent_candidates=percent_candidates, quarantine_period=quarantine_period, inf_duration=inf_duration, finetune=finetune,
                          T_in_id=T_in_id, T_out_id=T_out_id)
            else:
                ntmle.fit(p=p, samples=1, bound=0.01, seed=seed_number+i,
                          shift=shift, mode=mode, percent_candidates=percent_candidates, quarantine_period=quarantine_period, inf_duration=inf_duration, finetune=finetune,
                          T_in_id=T_in_id, T_out_id=T_out_id)
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
    ####### inside for loop ########
    print()

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
avg_df.to_csv("avg_sims_results/" + exposure + str(sim_id) + "_" + save + "_DL_" +  args.task_string + ".csv", index=False)

########################################
# Saving results
########################################
results.to_csv("results/" + exposure + str(sim_id) + "_" + save + "_DL_" +  args.task_string + ".csv", index=False)


# ########### TEST CODE START #############
torch.save(ntmle.cat_vars, 'cat_vars.pt')
torch.save(ntmle.cont_vars, 'cont_vars.pt')
torch.save(ntmle.cat_unique_levels, 'cat_unique_levels.pt')

torch.save(ntmle.df_restricted_list, 'df_restricted_list.pt')
torch.save(ntmle.pooled_data_restricted_list, 'pooled_data_restricted_list.pt')
torch.save(ntmle.pooled_adj_matrix_list, 'pooled_adj_matrix_list.pt')


ntmle.df_restricted_list[-1]['D'].dtype

pred_binary.astype('int64')


# adding noise to the data, does not improve performance of the dl model
tmp = network_to_df(network_list[-1])
# Choose how many index include for random selection
chosen_idx = np.random.choice(tmp.shape[0], replace=True, size=int(np.ceil(tmp.shape[0]*0.25)))
tmp.loc[chosen_idx, 'A'] = 1 - tmp.loc[chosen_idx, 'A'] 
tmp.loc[chosen_idx, 'A']

def add_noise_to_binary(df, col_name, ratio_flipped, seed):
    np.random.seed(seed)
    chosen_idx = np.random.choice(df.shape[0], replace=False, size=int(np.ceil(df.shape[0]*ratio_flipped)))
    df.loc[chosen_idx, col_name] = 1 - df.loc[chosen_idx, col_name]
    return df

def add_noise_to_float(df, col_name, seed):
    np.random.seed(seed)
    noise = np.random.normal(0, 1, df.shape[0])
    # 0 is the mean of the normal distribution you are choosing from
    # 1 is the standard deviation of the normal distribution
    # df.shape[0] is the number of elements you get in array noise
    df[col_name] = df[col_name] + noise
    return df


tmp = add_noise_to_binary(tmp, 'A', 0.25, seed_number)
tmp = add_noise_to_binary(tmp, 'H', 0.25, seed_number)
tmp = add_noise_to_binary(tmp, 'I', 0.25, seed_number)

adj_matrix = nx.adjacency_matrix(H, weight=None)
tmp = pd.merge(tmp, pd.DataFrame.from_dict(dict(network_list[-1].degree),
                                              orient='index').rename(columns={0: 'degree'}),
                        how='left', left_index=True, right_index=True)
tmp['I_sum'] = fast_exp_map(adj_matrix, np.array(tmp['I']), measure='sum')
tmp['I_ratio'] = tmp['I_sum'] / tmp['degree'] # ratio of infected neighbors

nx.set_node_attributes(network_list[-1], dict(tmp['A']), 'A')
nx.set_node_attributes(network_list[-1], dict(tmp['H']), 'H')
nx.set_node_attributes(network_list[-1], dict(tmp['I']), 'I')
nx.set_node_attributes(network_list[-1], dict(tmp['I_ratio']), 'I_ratio')


from beowulf.dgm.quaratine_with_cat_cont_split import apply_quarantine_action, simulate_infection_of_immediate_neighbors, update_summary_measures, update_edge_in_graph, update_pr_a



def quarantine_dgm_truth_pool(network, pr_a, shift=False, restricted=False,
                         time_limit=10, inf_duration=5, quarantine_period=2,
                         percent_candidates=0.3, mode='top',
                         random_seed=100):
    '''Get ground truth for quarantine action
    percent_candidates: proportions of nodes to be selected as social distancing cluster center nodes candidates
    mode: 'top' or 'bottom' or 'all', 
          'top': select the top [percent_candidates] nodes with highest I_ratio (percentage of infected neighbors)
          'bottom': select the bottom [percent_candidates] nodes with lowest I_ratio
          'all': select all nodes

    P.S. pr_a is used to select the actual cluster center nodes from candidates'''
    # set up random generator
    rng = np.random.default_rng(seed=random_seed)

    graph = network.copy()
    data = network_to_df(graph)

    for n, d in graph.nodes(data=True):
        d['D'] = 0
        d['R'] = 0
        d['t'] = 0
        d['I'] = 0
        d['quarantine'] = 0

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
    graph_saved_by_time = [] # save graph slice for each time point
    edge_recorder = {key:[] for key in range(1, time_limit+1, quarantine_period)} # record edge_to_remove to be add back after the quarantine period has passed
    # starts with 1 because the timer below starts with 1
    time = 0
    while time < time_limit:  # Simulate outbreaks until time-step limit is reached
        time += 1
        print(f'time {time}')
        # for inf in sorted(infected, key=lambda _: random.random()):
        for inf in sorted(infected, key=lambda _: rng.random()):
            # Book-keeping for infected nodes
            graph.nodes[inf]['I'] = 1
            graph.nodes[inf]['D'] = 1
            graph.nodes[inf]['t'] += 1

            if graph.nodes[inf]['t'] > inf_duration:
                graph.nodes[inf]['I'] = 0         # Node is no longer infectious after this loop
                graph.nodes[inf]['R'] = 1         # Node switches to Recovered
                infected.remove(inf)
            
            # Calculate summary measure
            data, graph = update_summary_measures(data, graph)

            # Apply quarantine actions
            if quarantine_period == 1:
                data, graph, edge_recorder = apply_quarantine_action(data, graph, shift, restricted, edge_recorder, time, rng,
                                                                     pr_a, percent_candidates, mode)
                # print(f'time {time}: inf {inf} from {infected}')
            else:
                if time % quarantine_period == 1:
                    data, graph, edge_recorder = apply_quarantine_action(data, graph, shift, restricted, edge_recorder, time, rng,
                                                                         pr_a, percent_candidates, mode)
                    # print(f'time {time}: inf {inf} from {infected}')  
            
            # Simulate infections of immediate neighbors
            graph, infected = simulate_infection_of_immediate_neighbors(graph, inf, infected, rng)

        # save graph at current time step
        graph_saved_by_time.append(graph.copy())
        # update graph for next time step
        if quarantine_period == 1:
            graph = update_edge_in_graph(graph, edge_recorder, time, quarantine_period)
        else:
            if time % quarantine_period == 1:
                graph = update_edge_in_graph(graph, edge_recorder, time, quarantine_period)
    
    dis_save_by_time = []
    for g in graph_saved_by_time:
        dis = [] 
        for nod, d in g.nodes(data=True):
            dis.append(d['D'])
        dis_save_by_time.append(np.mean(dis))

    # return np.mean(dis), dis_save_by_time, graph_saved_by_time # save last time point and the whole time series
    return np.mean(dis), dis_save_by_time, graph_saved_by_time # save last time point and the whole time series

     
_, _, graph_saved_by_time = quarantine_dgm_truth_pool(network=G, pr_a=0.75, shift=shift, restricted=degree_restrict,
                                                     time_limit=10, inf_duration=5, quarantine_period=2,
                                                     percent_candidates=0.3, mode='top',
                                                     random_seed=seed_number+i)

ntmle = NetworkTMLETimeSeries(graph_saved_by_time, exposure='quarantine', outcome='D', verbose=False, degree_restrict=degree_restrict,
                                task_string=args.task_string,
                                cat_vars=cat_vars_i, cont_vars=cont_vars_i, cat_unique_levels=cat_unique_levels_i,
                                use_deep_learner_A_i=use_deep_learner_A_i, 
                                use_deep_learner_A_i_s=use_deep_learner_A_i_s, 
                                use_deep_learner_outcome=use_deep_learner_outcome,
                                use_all_time_slices=False) 

def add_summary_measures(graph_saved_by_time, ntmle):
    # Generate a fresh copy of the network with ascending node order
    oid = "_original_id_"                                                                   # Name to save the original IDs
    labeled_network_list = []
    for network in graph_saved_by_time:
        network = nx.convert_node_labels_to_integers(network,                               # Copy of new network with new labels
                                                    first_label=0,                          #  ... start at 0 for latent variance calc
                                                    label_attribute=oid)                    # ... saving the original ID labels
        labeled_network_list.append(network)                                                # ... saving to list

    adj_matrix_list = [nx.adjacency_matrix(network, weight=None) for network in labeled_network_list]
    df_list = [network_to_df(network) for network in labeled_network_list]
    _max_degree_list_ = [np.max([d for n, d in network.degree]) for network in labeled_network_list]

    # Creating summary measure mappings for all variables in the network
    summary_types = ['sum', 'mean', 'var', 'mean_dist', 'var_dist']                             # Default summary measures available
    handle_isolates = ['mean', 'var', 'mean_dist', 'var_dist']                                  # Whether isolates produce nan's
    for i in range(len(df_list)):         
        for v in [var for var in list(df_list[i].columns) if var not in [oid, outcome]]:        # All cols besides ID and outcome
            v_vector = np.asarray(df_list[i][v])                                                # ... extract array of column
            for summary_measure in summary_types:                                               # ... for each summary measure
                df_list[i][v+'_'+summary_measure] = fast_exp_map(adj_matrix_list[i],       # ... calculate corresponding measure
                                                                v_vector,
                                                                measure=summary_measure)
                if summary_measure in handle_isolates:                                          # ... set isolates from nan to 0
                    df_list[i][v+'_'+summary_measure] = df_list[i][v+'_'+summary_measure].fillna(0)
                
                if v+'_'+summary_measure not in ntmle.cont_vars:
                    ntmle.cont_vars.append(v+'_'+summary_measure)  

    # Creating summary measure mappings for non-parametric exposure_map_model()
    from tmle_utils import exp_map_individual
    _nonparam_cols_ = []
    for i in range(len(labeled_network_list)):
        exp_map_cols = exp_map_individual(network=labeled_network_list[i],                         # Generate columns of indicator
                                        variable=exposure,                                    # ... for the exposure
                                        max_degree=_max_degree_list_[i])                 # ... up to the maximum degree
        _nonparam_cols_.append(list(exp_map_cols.columns))                                 # Save column list for estimation procedure
        df_list[i] = pd.merge(df_list[i],                                                       # Merge these columns into main data
                            exp_map_cols.fillna(0),                                           # set nan to 0 to keep same dimension across i
                            how='left', left_index=True, right_index=True)                    # Merge on index to left

    # Calculating degree for all the nodes
    pool_list = [None] * len(df_list) # init self.df_list
    for i in range(len(labeled_network_list)):
        if nx.is_directed(labeled_network_list[i]):                                                    # For directed networks...
            degree_data = pd.DataFrame.from_dict(dict(labeled_network_list[i].out_degree),             # ... use the out-degree
                                                    orient='index').rename(columns={0: 'degree'})
        else:                                                                                       # For undirected networks...
            degree_data = pd.DataFrame.from_dict(dict(labeled_network_list[i].degree),                 # ... use the regular degree
                                                    orient='index').rename(columns={0: 'degree'})
        pool_list[i] = pd.merge(df_list[i],                                                      # Merge main data
                            degree_data,                                                     # ...with degree data
                            how='left', left_index=True, right_index=True)   

    return pool_list

# # Apply degree restriction to data
# for i in range(len(df_list)):
#     if degree_restrict is not None:                                                                                 # If restriction provided,
#         df_list[i]['__degree_flag__'] = ntmle._degree_restrictions_(degree_dist=df_list[i]['degree'],
#                                                                     bounds=degree_restrict)
#         self._exclude_ids_degree_ = np.asarray(self.df_list[i].loc[self.df_list[i]['__degree_flag__'] == 1].index)
#     else:                                                                                                           # Else all observations are used
#         self.df_list[i]['__degree_flag__'] = 0                                                                      # Mark all as zeroes
#         self._exclude_ids_degree_ = None      

# params
i=0
# p=prop_treated[0]
p=0.75
bound=0.01
seed=seed_number+i
# shift=shift
# mode=mode
# percent_candidates=percent_candidates
# quarantine_period=quarantine_period
# T_in_id=[6, 7, 8, 9]
# T_out_id=[9]
samples=1

# Step 1) Estimate the weights
# Also generates pooled_data for use in the Monte Carlo integration procedure
ntmle._resamples_ = samples                                                              # Saving info on number of resamples
h_iptw, pooled_data_restricted_list, pooled_adj_matrix_list = ntmle._estimate_iptw_ts_(p=p,                      # Generate pooled & estiamte weights
                                                                                       samples=samples,           # ... for some number of samples
                                                                                       bound=bound,               # ... with applied probability bounds
                                                                                       seed=seed,                 # ... and with a random seed given
                                                                                       shift=shift,
                                                                                       mode=mode,
                                                                                       percent_candidates=percent_candidates,
                                                                                       quarantine_period=quarantine_period,
                                                                                       inf_duration=inf_duration,
                                                                                       T_in_id=T_in_id, T_out_id=T_out_id)                 

# ntmle.cat_vars
# pooled_data_restricted_list[9].isnull().sum()



def update_graph_features(data, graph):
    for col in data.columns:
        if nx.is_directed(graph):
            raise NotImplementedError("Directed graph is not supported yet")
        else: 
            nx.set_node_attributes(graph, dict(data[col]), col)
    return graph


def update_summary_measures(data, graph):
    ''' Update summary measures for the graph data
    because I_sum has to be updated, summary data is updated for every infected node''' 
    data = network_to_df(graph)

    # get degree for all the nodes
    if nx.is_directed(graph):
        degree_data = pd.DataFrame.from_dict(dict(graph.out_degree),             
                                                orient='index').rename(columns={0: 'degree'}) 
    else:
        degree_data = pd.DataFrame.from_dict(dict(graph.degree),                 
                                                orient='index').rename(columns={0: 'degree'})
    data['degree'] = degree_data
    
    # get actions from current Infected individual, document as "quarantine"
    # apply quarantine if the nodes is a contact of inf, implemented in the neigbors loop below
    adj_matrix = nx.adjacency_matrix(graph, weight=None)
    data['I_sum'] = fast_exp_map(adj_matrix, np.array(data['I']), measure='sum')
    data['I_ratio'] = data['I_sum'] / data['degree']  # ratio of infected neighbors
    data['I_ratio'] = data['I_ratio'].fillna(0)  # fill in 0 for nodes with no neighbors

    # # add I_ratio to graph data
    # if nx.is_directed(graph):
    #     raise NotImplementedError("Directed graph is not supported yet")
    # else:
    #     nx.set_node_attributes(graph, dict(data['I_ratio']), 'I_ratio')
    # # for-loop version
    # for n in graph.nodes():
    #     graph.nodes[n]['I_ratio'] = float(data.loc[data.index == n, 'I_ratio'].values)    

    # update other summary measures in data and graph
    # graph has to be updated because data for each time step is synchronized from the graph
    summary_types = ['sum', 'mean', 'var', 'mean_dist', 'var_dist']                # Default summary measures available
    handle_isolates = ['mean', 'var', 'mean_dist', 'var_dist'] 
    # input_df = network_to_df(self.network_list[0])                                 # Extract the variables from the original data
    input_df = network_to_df(ntmle.network_list[0])                                 # Extract the variables from the original data
    # for v in [var for var in list(input_df.columns) if var not in [self.oid, self.outcome]]: # summary measure is generated for each original varible
    for v in [var for var in list(input_df.columns) if var not in [ntmle.oid, ntmle.outcome]]: # summary measure is generated for each original varible
        v_vector = np.asarray(data[v])                                                # ... extract array of column
        for summary_measure in summary_types:                                      # ... for each summary measure
            data[v+'_'+summary_measure] = fast_exp_map(adj_matrix,                    # ... calculate corresponding measure
                                                       v_vector,
                                                       measure=summary_measure)
            if summary_measure in handle_isolates:                                 # ... set isolates from nan to 0
                data[v+'_'+summary_measure] = data[v+'_'+summary_measure].fillna(0)
            # # add summary measure to graph data
            # if nx.is_directed(graph):
            #     raise NotImplementedError("Directed graph is not supported yet")
            # else: 
            #     nx.set_node_attributes(graph, dict(data[v+'_'+summary_measure]), v+'_'+summary_measure)
    
    # update I_ratio and summary measures to graph 
    graph = update_graph_features(data, graph)
    
    return data, graph
    
import math
def apply_quarantine_action(data, graph, shift, edge_recorder, time_step, rng, pr_a, percent_candidates, mode):
    ''' Update the probability of quarantine action for every quarantine_period '''
    if shift:
        probs = rng.binomial(n=1,                              # Flip a coin to generate A_i
                             p=pr_a,                               # ... based on policy-assigned probabilities
                             size=data.shape[0])                   # ... for the N units
    else:
        # select candidates for quarantine based on the degree
        if mode == 'top':
            num_candidates = math.ceil(data.shape[0] * percent_candidates)
            candidates_nodes = data.nlargest(num_candidates, 'F').index # super-spreader
        elif mode == 'bottom':
            num_candidates = math.ceil(data.shape[0] * percent_candidates)
            candidates_nodes = data.nsmallest(num_candidates, 'F').index # super-defender
        elif mode == 'all':
            num_candidates = data.shape[0]
            candidates_nodes = data.index
        
        quarantine_piror = rng.binomial(n=1, p=pr_a, size=num_candidates)
        quarantine_nodes = candidates_nodes[quarantine_piror==1]
        probs = np.zeros(data.shape[0])
        probs[quarantine_nodes] = 1 
        
    # data[self.exposure] = np.where(data['__degree_flag__'] == 1,  # Restrict to appropriate degåree
    #                                data[self.exposure], probs)    # ... keeps restricted nodes as observed A_i
    
    data['quarantine'] = np.where(data['__degree_flag__'] == 1,  # Restrict to appropriate degåree
                                  data['quarantine'], probs)    # ... keeps restricted nodes as observed A_i

    # apply quarantine to all selected nodes: remove immediate neighbors for the next time step
    # Function version
    if nx.is_directed(graph):
        raise NotImplementedError("Directed graph is not supported yet")
    else:
        nx.set_node_attributes(graph, dict(data['quarantine']), 'quarantine')
        edge_recorder[time_step].extend(list(nx.edges(graph, data[data['quarantine']==1].index)))
    # # for-loop version
    # for n in graph.nodes():
    #     graph.nodes[n]['quarantine'] = int(data.loc[data.index == n, 'quarantine'].values)
    #     if n in data[data['quarantine']==1].index: # remove all immediate neighbors
    #         for neighbor in graph.neighbors(n):
    #             edge_recorder[time_step].append((neighbor, n))
    return data, graph, edge_recorder

def update_edge_in_graph(graph, edge_recorder, time_step, quarantine_period):
    ''' Update the graph for every quarantine_period
    for current update, if it is not the first one, 
    add back previous quarantine edges after the quarantine period has passed;
    and then remove currently quarantined edges
    '''
    if time_step - quarantine_period >= quarantine_period: 
        graph.add_edges_from(edge_recorder[time_step-quarantine_period]) 
    graph.remove_edges_from(edge_recorder[time_step]) # remove current quarantined edges
    return graph

def simulate_infection_of_immediate_neighbors(graph, inf, infected, rng):
    '''Simulate infections of immediate neighbors'''
    for contact in nx.neighbors(graph, inf):
        if graph.nodes[contact]["D"] == 1:
            pass
        else:
            # probability of infection associated with quarantine and I_ratio directly
            pr_y = logistic.cdf(- 1.2
                                + 0.5*graph.nodes[contact]['I_ratio']
                                + 0.5*graph.nodes[inf]['I_ratio']
                                - 1.2*graph.nodes[contact]['quarantine']
                                - 1.2*graph.nodes[inf]['quarantine']
                                + 1.5*graph.nodes[contact]['A']
                                - 0.1*graph.nodes[contact]['H'])
            # print(pr_y)
            if rng.binomial(n=1, p=pr_y, size=1):
                graph.nodes[contact]['I'] = 1
                graph.nodes[contact]["D"] = 1
                infected.append(contact)
    return graph, infected

def update_adj_matrix(graph):
    adj_matrix = nx.adjacency_matrix(graph, weight=None) 

    if ntmle.degree_restrict is not None:                                                         
        ntmle._check_degree_restrictions_(bounds=ntmle.degree_restrict)                           
        _max_degree_ = ntmle.degree_restrict[1]                    
    else:                                                                                   
        if nx.is_directed(graph):
            _max_degree_ = np.max([d for n, d in graph.out_degree])                                                         
        else:                  
            _max_degree_ = np.max([d for n, d in graph.degree])
    
    return adj_matrix, _max_degree_


samples=2
inf_duration = 5
pr_a = 0.75


# initiate lists to store pooled data, network, adj_matrix, _max_degree_ for all samples
pooled_all_samples = []
graph_all_samples = []
adj_matrix_all_samples = []
_max_degree_all_samples = []

# for each sample drawn, the process of generating time_limit steps of data is recorded
for s in range(samples): 
    # initiate rng seed for bootstraps
    rng = np.random.default_rng(seed+s)
    # initiate dataframe for the first time step
    data = ntmle.df_list[0].copy()
    # initiate network, adj_matrix, _max_degree_  for the first time step
    graph_init = [update_graph_features(data, ntmle.network_list[0].copy())]
    adj_matrix_init = [ntmle.adj_matrix_list[0].copy()]
    _max_degree_init = [ntmle._max_degree_list_[0].copy()]
    # initiate edge_recorders to record edges to remove/add back for each quarantine_period
    edge_recorder = {key:[] for key in range(quarantine_period, len(ntmle.df_list), quarantine_period)} # record edge_to_remove to be add back after the quarantine period has passed
    # starts with quarantine_period because the initial action is already applied 
    # initiate infected list
    infected = list(ntmle.df_list[0][ntmle.df_list[0]['I'] == 1].index) 

    # intiate pooled sample list
    data['_sample_id_'] = s                # Setting sample ID
    pooled_per_sample = [data.copy()]
    

    for time_step in range(1, len(ntmle.df_list)): # start from 1 because the first time step is already initiated                
        graph_current = graph_init[time_step-1].copy()
        adj_matrix_current = adj_matrix_init[time_step-1].copy()
        _max_degree_current = _max_degree_init[time_step-1].copy()
        
        for inf in sorted(infected, key=lambda _: rng.random()):
            # Book-keeping for infected nodes
            graph_current.nodes[inf]['I'] = 1
            graph_current.nodes[inf]['D'] = 1
            graph_current.nodes[inf]['t'] += 1

            if graph_current.nodes[inf]['t'] > inf_duration:
                graph_current.nodes[inf]['I'] = 0         # Node is no longer infectious after this loop
                graph_current.nodes[inf]['R'] = 1         # Node switches to Recovered
                infected.remove(inf)
            
            # Calculate summary measure
            data, graph_current = update_summary_measures(data, graph_current)
            
            # update quarantine measure, summary measure for all vars for current time step,
            # update graph and adj_matrix for next time step
            if time_step % quarantine_period == 0:
                data, graph_current, edge_recorder = apply_quarantine_action(data, graph_current, shift, edge_recorder, time_step, rng, pr_a, percent_candidates, mode)            

            # Simulate infections of immediate neighbors
            graph_current, infected = simulate_infection_of_immediate_neighbors(graph_current, inf, infected, rng)
            
        # Update graph, adj_matrix and _max_degree_ for the next time step
        if time_step % quarantine_period == 0:
            graph_current = update_edge_in_graph(graph_current, edge_recorder, time_step, quarantine_period)
        
        adj_matrix_current, _max_degree_current = update_adj_matrix(graph_current)
        graph_init.append(graph_current)
        adj_matrix_init.append(adj_matrix_current)
        _max_degree_init.append(_max_degree_current)

        #     # Logic if no summary measure was specified (uses the complete factor approach)
        # if ntmle._gs_measure_ is None:
        #     network_tmp = network_init[time_step].copy()
        #     a = np.array(g[ntmle.exposure])                              # Transform A_i into array
        #     for n in network_tmp.nodes():                               # For each node,
        #         network_tmp.nodes[n][ntmle.exposure] = a[n]              # ...assign the new A_i*
        #     df = exp_map_individual(network_tmp,                        # Now do the individual exposure maps with new
        #                             variable=self.exposure,
        #                             max_degree=_max_degree_init[time_step]).fillna(0)
        #     for c in self._nonparam_cols_[-1]:                          # Adding back these np columns
        #         g[c] = df[c] 
            
        # # Re-creating any threshold variables in the pooled sample data
        # if self._thresholds_any_:
        #     create_threshold(data=g,
        #                         variables=self._thresholds_variables_,
        #                         thresholds=self._thresholds_,
        #                         definitions=self._thresholds_def_)

        # # Re-creating any categorical variables in the pooled sample data
        # if self._categorical_any_:
        #     create_categorical(data=g,
        #                         variables=self._categorical_variables_,
        #                         bins=self._categorical_,
        #                         labels=self._categorical_def_,
        #                         verbose=False)

        data['_sample_id_'] = s                # Setting sample ID
        pooled_per_sample.append(data.copy())      # Adding to list (for later concatenate)

    pooled_all_samples.append(pooled_per_sample) # append pooled sample for each sample
    graph_all_samples.append(graph_init) # append network for each sample
    adj_matrix_all_samples.append(adj_matrix_init) # append adj_matrix for each sample
    _max_degree_all_samples.append(_max_degree_init) # append _max_degree_ for each sample

infected

pooled_per_sample[-1].equals(g)
pooled_per_sample[-1]

len(pooled_per_sample)
len(pooled_all_samples)

g
g['sample_id']
pooled_per_sample[0]

pooled_all_samples[0][-1]
pooled_all_samples[1][-1]

for i in range(10):
    print(pooled_all_samples[0][i].equals(pooled_all_samples[1][i]))  

len(pooled_all_samples[0][-1])
pooled_all_samples[0][-1].shape

# re-arrange list of lists: each elemental list should contain all samples and the number of sublists equals to time_limit
pooled_all_samples_rearrange = [list(i) for i in zip(*pooled_all_samples)]
pooled_adj_matrix_list = [list(i) for i in zip(*adj_matrix_all_samples)]

len(pooled_all_samples_rearrange[0])
pooled_all_samples_rearrange[0][0]

for pooled_sample in pooled_all_samples:
    print(len(pooled_sample))
    for df in pooled_sample:
        print(df.shape)
    print()
    print(pooled_sample[-1].shape)


tmp = list(map(list, zip(*pooled_all_samples)))
len(tmp)
len(tmp[0])
tmp[0][0].shape


for pooled_sample in pooled_all_samples_rearrange:
    print(len(pooled_sample))
    print(pooled_sample[0].equals(pooled_sample[1]))

pooled_sample[0]['_sample_id_']
pooled_sample[1]['_sample_id_']

pooled_all_samples_rearrange[1][0].shape


pooled_data_restricted_list = []
for pooled_sample_for_one_time_step in pooled_all_samples_rearrange:
    df_one_step = pd.concat(pooled_sample_for_one_time_step, axis=0, ignore_index=True)
    pooled_data_restricted = df_one_step.loc[df_one_step['__degree_flag__'] == 0].copy()
    pooled_data_restricted_list.append(pooled_data_restricted)

len(pooled_data_restricted_list)
pooled_data_restricted_list[0].shape
pooled_data_restricted_list[-1].shape

df_one_step.shape
pooled_data_restricted.shape

for i in range(10):
    id_0_mask = pooled_data_restricted_list[i]['_sample_id_'] == 0
    id_1_mask = pooled_data_restricted_list[i]['_sample_id_'] == 1 
    print(pooled_data_restricted_list[i][id_0_mask]['A'].equals(pooled_data_restricted_list[i][id_1_mask]['A']))
    break


pooled_data_restricted_list[i][id_0_mask]['degree'].shape
pooled_data_restricted_list[i][id_1_mask]['degree'].shape

(pooled_data_restricted_list[9][id_1_mask]['I_ratio'].to_numpy() > 0).sum()
(pooled_data_restricted_list[9][id_1_mask]['degree'].to_numpy() > 1).sum()
(pooled_data_restricted_list[9][id_0_mask]['I_ratio_sum'].to_numpy() == pooled_data_restricted_list[9][id_1_mask]['I_ratio_sum'].to_numpy()).sum()


# Step 3) Target the parameter
from tmle_utils import targeting_step
epsilon = targeting_step(y=ntmle.df_restricted_list[-1][ntmle.outcome],   # Estimate the targeting model given observed Y
                         q_init=ntmle._Qinit_,                           # ... predicted values of Y under observed A
                         ipw=h_iptw,                                    # ... weighted by IPW
                         verbose=ntmle._verbose_)                        # ... with option for verbose info
epsilon

import statsmodels.api as sm
y = ntmle.df_restricted_list[-1][ntmle.outcome]   # Estimate the targeting model given observed Y
q_init = ntmle._Qinit_
ipw=h_iptw
# verbose

q_init.max()
q_init.min()
np.log(probability_to_odds(q_init))

y = y[q_init != 1]
ipw = ipw[q_init != 1]
q_init = q_init[q_init != 1]


f = sm.families.family.Binomial()
log = sm.GLM(y,  # Outcome / dependent variable
             np.repeat(1, y.shape[0]),  # Generating intercept only model
             offset=np.log(probability_to_odds(q_init)),  # Offset by g-formula predictions
             freq_weights=ipw,  # Weighted by calculated IPW
             family=f).fit(maxiter=500)

print('==============================================================================')
print('Targeting Model')
print(log.summary())
print(log.params[0])

tmp = probability_to_odds(q_init)
np.log(tmp)

pred_binary = np.round(q_init)
(pred_binary == y).sum().item()/y.shape[0]
q_init

q_init[q_init == 1]
ipw.shape
q_init.s

len(pooled_data_restricted_list)
pooled_data_restricted_list[9].shape

# Step 4) Monte Carlo integration (old code did in loop but faster as vector)
#
from tmle_utils import outcome_learner_predict, outcome_deep_learner_ts, get_patsy_for_model_w_C
import patsy
# Generating outcome predictions under the policies (via pooled data sets)
if ntmle._q_custom_ is None:                                                     # If given a parametric default model
    y_star = ntmle._outcome_model.predict(pooled_data_restricted_list[-1])       # ... predict using statsmodels syntax
else:  # Custom input model by user
    if ntmle.use_deep_learner_outcome:
        xdata_list = []
        ydata_list = []
        n_output_list = []
        for pooled_data_restricted in pooled_data_restricted_list:
        # for pooled_data_restricted in pool_list: 
        # for pooled_data_restricted in ntmle.df_list:
            if 'C(' in ntmle._q_model:
                xdata_list.append(get_patsy_for_model_w_C(ntmle._q_model, pooled_data_restricted))
            else:
                xdata_list.append(patsy.dmatrix(ntmle._q_model + ' - 1', pooled_data_restricted, return_type="dataframe"))
            ydata_list.append(pooled_data_restricted[ntmle.outcome])
            n_output_list.append(pd.unique(pooled_data_restricted[ntmle.outcome]).shape[0])

        y_star = outcome_deep_learner_ts(ntmle._q_custom_, 
                                            xdata_list, ydata_list, T_in_id, T_out_id,
                                            pooled_adj_matrix_list, ntmle.cat_vars, ntmle.cont_vars, ntmle.cat_unique_levels, n_output_list, ntmle._continuous_outcome_list_[-1],
                                            predict_with_best=True, custom_path=ntmle._q_custom_path_)
    else:
        d = patsy.dmatrix(ntmle._q_model + ' - 1', pooled_data_restricted_list[-1])  # ... extract data via patsy
        y_star = outcome_learner_predict(ml_model_fit=ntmle._q_custom_,              # ... predict using custom function
                                        xdata=np.asarray(d))                        # ... for the extracted data


for i in range(len(xdata_list)):
    tmp = xdata_list[i]
    # tmp.isnull().sum().sum()

    count = np.isinf(tmp).values.sum() 
    print("It contains " + str(count) + " infinite values") 

ydata_list[0].isnull().sum().sum()
np.isinf(ydata_list[0]).sum()

# CAUTION: label is not the right answer for Monte Carlo integration
# pred_binary = np.round(y_star)
# label = pooled_data_restricted_list[-1][ntmle.outcome]
# (pred_binary == label).sum().item()/label.shape[0]

# pred_binary.sum()
# label.sum()
# label.sum() / label.shape[0]

q_star = (y_star * np.exp(epsilon)) / (1 - y_star + y_star * np.exp(epsilon))
pooled_data_restricted_list[-1]['__pred_q_star__'] = q_star 

# Taking the mean, grouped-by the pooled sample IDs (fast)
ntmle.marginals_vector = np.asarray(pooled_data_restricted_list[-1].groupby('_sample_id_')['__pred_q_star__'].mean())

# Calculating estimate for the policy
ntmle.marginal_outcome = np.mean(ntmle.marginals_vector)                                  # Mean of Monte Carlo results
# self._specified_p_ = p  

bias = ntmle.marginal_outcome - truth[p]
bias

# -0.0392913349866867

ntmle.marginal_outcome
truth[p]

y_star
ntmle.df_list[0][ntmle.df_list[0]['D'] == 1].index

ntmle._outcome_model.summary()
ntmle._Qinit_
label = ntmle.df_restricted_list[-1][ntmle.outcome]
pred_binary = np.round(ntmle._Qinit_)
(pred_binary == label).sum().item()/label.shape[0]
from sklearn.metrics import confusion_matrix
n = confusion_matrix(label, pred_binary)
n

# acc: 0.822
# confusion:
# [[139,  41],
#  [ 48, 272]]

# input [9]
# accDL 0.81
# [[143,  37],
#  [ 58, 262]]

# input [8, 9]
# accDL 0.893
# [[167,  13],
#  [ 42, 278]]



for network in network_list:
    data = network_to_df(network)
    adj_matrix = nx.adjacency_matrix(network, weight=None)
    data['A_sum'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='sum')
    # data['A_mean'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='mean')
    data['H_sum'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='sum')

    pr_a = pr_a = logistic.cdf(- 4.5 
                        + 1.2*data['A'] + 0.8*data['H']
                        + 0.5*data['H_sum'] + 0.3*data['A_sum'] 
                        + 1.2*data['I_ratio'])
    print(f'pr_a min:{pr_a.min()}')
    print(f'pr_a max:{pr_a.max()}')


for i in range(10):
    if i > 0:
        print(pooled_data_restricted_list[i]['A_sum'].equals(pooled_data_restricted_list[i-1]['A_sum']))

from sklearn.metrics import confusion_matrix
import pandas as pd
test_labels = ntmle.df_restricted_list[-1][ntmle.outcome]
predictions = np.round(ntmle._Qinit_)

n = confusion_matrix(test_labels, predictions)
n
plot_confusion_matrix(n, classes = ['Dead cat', 'Alive cat'], 
					  title = 'Confusion Matrix')


### DOES MODEL STRUCTURE matter
import patsy
xdata_list = []
ydata_list = []     
n_output_list = []   
# for df_restricted in ntmle.df_restricted_list:
for df_restricted in pooled_data_restricted_list:
    if 'C(' in model:
        xdata_list.append(get_patsy_for_model_w_C(qn_model, df_restricted))
    else:
        xdata_list.append(patsy.dmatrix(qn_model + ' - 1', df_restricted, return_type="dataframe"))
    ydata_list.append(df_restricted[ntmle.outcome])
    n_output_list.append(pd.unique(df_restricted[ntmle.outcome]).shape[0])


from tmle_utils import get_model_cat_cont_split_patsy_matrix, get_final_model_cat_cont_split, get_imbalance_weights
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
pos_weight, class_weight = get_imbalance_weights(n_output_final, ydata_array_list, use_last_time_slice=True,
                                                    imbalance_threshold=3., imbalance_upper_bound=3.2, default_lock=False)
        
pos_weight
class_weight

model_cat_vars_final
model_cont_vars_final
xdata_list[0]['quarantine_sum'].sum()
(xdata_list[0]['I_ratio_sum'] == 0).sum()
xdata_list[9].std()
xdata_list[9].mean()

tmp = (xdata_list[8] - xdata_list[8].mean())/xdata_list[8].std()
tmp.isnull().sum().sum()
tmp.fillna(0, inplace=True)


train_loader, valid_loader, test_loader = ntmle._q_custom_._data_preprocess([xdata_list, ydata_list],
                                                                                model_cat_vars=model_cat_vars, 
                                                                                model_cont_vars=model_cont_vars, 
                                                                                model_cat_unique_levels=model_cat_unique_levels,
                                                                                normalize=True,
                                                                                drop_duplicates=False,
                                                                                T_in_id=T_in_id, T_out_id=T_out_id)

for i, (x_cat, x_cont, y, sample_idx) in enumerate(test_loader):
    print(x_cat)
    print(x_cont)
    print(y)

len(train_loader)
len(valid_loader)
len(test_loader)
# ########### TEST CODE END ##############

# from tmle_utils import get_patsy_for_model_w_C
# import patsy

# if ntmle.use_deep_learner_outcome:
#     xdata_list = []
#     ydata_list = []
#     n_output_list = []
#     for df_restricted in ntmle.df_restricted_list:
#         if 'C(' in model:
#             xdata_list.append(get_patsy_for_model_w_C(qn_model, df_restricted))
#         else:
#             xdata_list.append(patsy.dmatrix(qn_model + ' - 1', df_restricted, return_type="dataframe"))
#         ydata_list.append(df_restricted[ntmle.outcome])
#         n_output_list.append(pd.unique(df_restricted[ntmle.outcome]).shape[0])

# ydata_array_list = []
# T_in_id=[6, 7, 8, 9]
# T_out_id=[9]
# xdata_list, ydata_list = [xdata_list[i] for i in T_in_id], [ydata_list[i] for i in T_out_id]
# for ydata in ydata_list:
#     ydata_array_list.append(ydata.to_numpy()) # convert pd.series to np.array

# xdata_list
# ydata_array_list

# vars = xdata_list[0].columns
# vars

# for var in vars:
#     print(var)
#     if '_t' in var:
#         print(var)

# pd.unique(xdata_list[0][var].astype('int')).max() + 1 

# xdata_list[0][var][xdata_list[0][var] > 1]

# ntmle.cont_vars

# pd.unique(ntmle.df_restricted_list[0]['H_t3'])
# pd.unique(xdata_list[3]['H_t3'])

# len(xdata_list)

# for i in range(10):
#     tmp = ntmle.df_restricted_list[i]['H_sum'].max()
#     print(f'{i}: {tmp}')

# # self._resamples_ = samples 
# p = prop_treated[0]
# samples=100
# bound=0.01
# seed=seed_number+i
# T_in_id=[6, 7, 8, 9]
# T_out_id=[9]


# h_iptw, pooled_data_restricted_list, pooled_adj_matrix_list = ntmle._estimate_iptw_ts_(p=p,                      # Generate pooled & estiamte weights
#                                                                                        samples=samples,           # ... for some number of samples
#                                                                                        bound=bound,               # ... with applied probability bounds
#                                                                                        seed=seed,                 # ... and with a random seed given
#                                                                                        shift=shift,
#                                                                                        mode=mode,
#                                                                                        percent_candidates=percent_candidates,
#                                                                                        quarantine_period=quarantine_period,
#                                                                                        T_in_id=T_in_id, T_out_id=T_out_id)                 

# from tmle_utils import targeting_step
# epsilon = targeting_step(y=ntmle.df_restricted_list[-1][ntmle.outcome],   # Estimate the targeting model given observed Y
#                                  q_init=ntmle._Qinit_,                           # ... predicted values of Y under observed A
#                                  ipw=h_iptw,                                    # ... weighted by IPW
#                                  verbose=ntmle._verbose_) 
# epsilon

# from tmle_utils import outcome_learner_predict, outcome_deep_learner_ts, get_patsy_for_model_w_C
# import patsy
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
#                                             xdata_list, ydata_list, T_in_id, T_out_id,
#                                             pooled_adj_matrix_list, ntmle.cat_vars, ntmle.cont_vars, ntmle.cat_unique_levels, n_output_list, ntmle._continuous_outcome_list_[-1],
#                                             predict_with_best=True, custom_path=ntmle._q_custom_path_)
#     else:
#         d = patsy.dmatrix(ntmle._q_model + ' - 1', pooled_data_restricted_list[-1])  # ... extract data via patsy
#         y_star = outcome_learner_predict(ml_model_fit=ntmle._q_custom_,              # ... predict using custom function
#                                         xdata=np.asarray(d))                        # ... for the extracted data
        
# # Updating predictions via intercept from targeting step
# logit_qstar = np.log(probability_to_odds(y_star)) + epsilon                         # NOTE: needs to be logit(Y^*) + e
# q_star = odds_to_probability(np.exp(logit_qstar))                                   # Back converting from odds
# # pooled_data_restricted_list[-1]['__pred_q_star__'] = q_star 

# q_star

# tmp = y_star * np.exp(epsilon) / (1 - y_star + y_star * np.exp(epsilon))


# pooled_data_restricted_list[0]

# from tmle_utils import get_patsy_for_model_w_C
# import patsy
# if ntmle.use_deep_learner_outcome:
#     xdata_list = []
#     ydata_list = []
#     n_output_list = []
#     for pooled_data_restricted in pooled_data_restricted_list:
#         if 'C(' in ntmle._q_model:
#             xdata_list.append(get_patsy_for_model_w_C(ntmle._q_model, pooled_data_restricted))
#         else:
#             xdata_list.append(patsy.dmatrix(ntmle._q_model + ' - 1', pooled_data_restricted, return_type="dataframe"))
#         ydata_list.append(pooled_data_restricted[ntmle.outcome])
#         n_output_list.append(pd.unique(pooled_data_restricted[ntmle.outcome]).shape[0])

# xdata_list, ydata_list = [xdata_list[i] for i in T_in_id], [ydata_list[i] for i in T_out_id]

# len(xdata_list)

# ##### compare between p 
# pooled_data_restricted_list_list = []
# xdata_list_list = []
# ydata_list_list = []

# for index, p in enumerate(prop_treated[:4]):
#     h_iptw, pooled_data_restricted_list, pooled_adj_matrix_list = ntmle._estimate_iptw_ts_(p=p,                      # Generate pooled & estiamte weights
#                                                                                     samples=samples,           # ... for some number of samples
#                                                                                     bound=bound,               # ... with applied probability bounds
#                                                                                     seed=seed,                 # ... and with a random seed given
#                                                                                     shift=shift,
#                                                                                     mode=mode,
#                                                                                     percent_candidates=percent_candidates,
#                                                                                     quarantine_period=quarantine_period,
#                                                                                     T_in_id=T_in_id, T_out_id=T_out_id)                 
#     pooled_data_restricted_list_list.append(pooled_data_restricted_list)
#     xdata_list = []
#     ydata_list = []
#     n_output_list = []
#     for pooled_data_restricted in pooled_data_restricted_list:
#         if 'C(' in ntmle._q_model:
#             xdata_list.append(get_patsy_for_model_w_C(ntmle._q_model, pooled_data_restricted))
#         else:
#             xdata_list.append(patsy.dmatrix(ntmle._q_model + ' - 1', pooled_data_restricted, return_type="dataframe"))
#         ydata_list.append(pooled_data_restricted[ntmle.outcome])
#         n_output_list.append(pd.unique(pooled_data_restricted[ntmle.outcome]).shape[0])

#     xdata_list, ydata_list = [xdata_list[i] for i in T_in_id], [ydata_list[i] for i in T_out_id]
#     xdata_list_list.append(xdata_list)
#     ydata_list_list.append(ydata_list)

# ntmle._q_model
# ntmle.exposure

# time_step = 8
# pooled_data_restricted_list_list[0][time_step].equals(pooled_data_restricted_list_list[1][time_step])
# pooled_data_restricted_list_list[0][time_step].equals(pooled_data_restricted_list_list[2][time_step])
# pooled_data_restricted_list_list[0][time_step].equals(pooled_data_restricted_list_list[3][time_step])


# for i in range(len(xdata_list_list[0])):
#     p0_vs_p1 = xdata_list_list[0][i].equals(xdata_list_list[1][i])
#     p0_vs_p2 = xdata_list_list[0][i].equals(xdata_list_list[2][i])
#     p0_vs_p3 = xdata_list_list[0][i].equals(xdata_list_list[3][i])
#     p1_vs_p2 = xdata_list_list[1][i].equals(xdata_list_list[2][i])
#     p1_vs_p3 = xdata_list_list[1][i].equals(xdata_list_list[3][i])
#     p2_vs_p3 = xdata_list_list[2][i].equals(xdata_list_list[3][i])
#     print(f'p0 vs p1: {p0_vs_p1}')
#     print(f'p0 vs p2: {p0_vs_p2}')
#     print(f'p0 vs p3: {p0_vs_p3}')
#     print(f'p1 vs p2: {p1_vs_p2}')
#     print(f'p1 vs p3: {p1_vs_p3}')
#     print(f'p2 vs p3: {p2_vs_p3}')

# # Prep for pooled data set creation
# rng = np.random.default_rng(seed+8)  # Setting the seed for bootstraps
# pooled_sample = []
# # TODO one way to potentially speed up code is to run this using Pool. Easy for parallel
# # this is also the best target for optimization since it takes about ~85% of current run times
# df = ntmle.df_list[8]
# g = df.copy()   
# p = prop_treated[0]
# probs = ntmle.select_candiate_nodes(data=g, 
#                                     mode=mode, percent_candidates=percent_candidates, 
#                                     pr_a=p, rng=rng)
# probs

# g[ntmle.exposure] = np.where(g['__degree_flag__'] == 1,  # Restrict to appropriate degåree
#                                         g[ntmle.exposure], probs)


# g[ntmle.exposure+'_sum'] = fast_exp_map(ntmle.adj_matrix_list[0], np.array(g[ntmle.exposure]), measure='sum')
# g[ntmle.exposure + '_mean'] = fast_exp_map(ntmle.adj_matrix_list[0], np.array(g[ntmle.exposure]), measure='mean')
# g[ntmle.exposure + '_mean'] = g[ntmle.exposure + '_mean'].fillna(0)            # isolates should have mean=0
# g[ntmle.exposure + '_var'] = fast_exp_map(ntmle.adj_matrix_list[0], np.array(g[ntmle.exposure]), measure='var')
# g[ntmle.exposure + '_var'] = g[ntmle.exposure + '_var'].fillna(0)              # isolates should have mean=0
# g[ntmle.exposure + '_mean_dist'] = fast_exp_map(ntmle.adj_matrix_list[0],
#                                                 np.array(g[ntmle.exposure]), measure='mean_dist')
# g[ntmle.exposure + '_mean_dist'] = g[ntmle.exposure + '_mean_dist'].fillna(0)  # isolates should have mean=0
# g[ntmle.exposure + '_var_dist'] = fast_exp_map(ntmle.adj_matrix_list[0],
#                                                 np.array(g[ntmle.exposure]), measure='var_dist')
# g[ntmle.exposure + '_mean_dist'] = g[ntmle.exposure + '_mean_dist'].fillna(0)  # isolates should have mean=0

# # update network and adj_matrix based on the exposure policy for next time step
# # update g for current time step 
# edges_to_remove, network_current, g = ntmle.get_edges_to_remove_and_update_exposure(g, network_list[0], ntmle.exposure)

# ###################
# g1 = df.copy()   
# p = prop_treated[1]
# probs1 = ntmle.select_candiate_nodes(data=g1, 
#                                     mode=mode, percent_candidates=percent_candidates, 
#                                     pr_a=p, rng=rng)
# probs1

# g1[ntmle.exposure] = np.where(g1['__degree_flag__'] == 1,  # Restrict to appropriate degåree
#                                         g1[ntmle.exposure], probs1)


# g1[ntmle.exposure+'_sum'] = fast_exp_map(ntmle.adj_matrix_list[0], np.array(g1[ntmle.exposure]), measure='sum')
# g1[ntmle.exposure + '_mean'] = fast_exp_map(ntmle.adj_matrix_list[0], np.array(g1[ntmle.exposure]), measure='mean')
# g1[ntmle.exposure + '_mean'] = g1[ntmle.exposure + '_mean'].fillna(0)            # isolates should have mean=0
# g1[ntmle.exposure + '_var'] = fast_exp_map(ntmle.adj_matrix_list[0], np.array(g1[ntmle.exposure]), measure='var')
# g1[ntmle.exposure + '_var'] = g1[ntmle.exposure + '_var'].fillna(0)              # isolates should have mean=0
# g1[ntmle.exposure + '_mean_dist'] = fast_exp_map(ntmle.adj_matrix_list[0],
#                                                 np.array(g1[ntmle.exposure]), measure='mean_dist')
# g1[ntmle.exposure + '_mean_dist'] = g1[ntmle.exposure + '_mean_dist'].fillna(0)  # isolates should have mean=0
# g1[ntmle.exposure + '_var_dist'] = fast_exp_map(ntmle.adj_matrix_list[0],
#                                                 np.array(g1[ntmle.exposure]), measure='var_dist')
# g1[ntmle.exposure + '_mean_dist'] = g1[ntmle.exposure + '_mean_dist'].fillna(0)  # isolates should have mean=0

# # update network and adj_matrix based on the exposure policy for next time step
# # update g for current time step 
# edges_to_remove1, network_current1, g1 = ntmle.get_edges_to_remove_and_update_exposure(g1, network_list[0], ntmle.exposure)



# g['degree'].equals(g1['degree'])
# g[ntmle.exposure].equals(g1[ntmle.exposure])

# ntmle._q_model
# g["A"].equals(g1["A"])
# g["H"].equals(g1["H"])
# g["A_sum"].equals(g1["A_sum"])
# g["H_sum"].equals(g1["H_sum"])


# g.columns

# updated_network_list = []
# updated_adj_matrix_list = []
# updated_max_degree_list = []
# for s in range(samples):                                    # For each of the *m* samples
#     g = df.copy()                                           # Create a copy of the data
#     if shift:
#         probs = rng.binomial(n=1,                               # Flip a coin to generate A_i
#                                 p=p,                               # ... based on policy-assigned probabilities
#                                 size=g.shape[0])                   # ... for the N units
#     else:   
#         # New mechanism to apply quarantine
#         # print(f'Apply New Mechanism for Quarantine, mode {mode}, percent_candidates {percent_candidates}')
#         probs = self.select_candiate_nodes(data=g, 
#                                             mode=mode, percent_candidates=percent_candidates, 
#                                             pr_a=p, rng=rng)



# ########### TEST CODE END ################