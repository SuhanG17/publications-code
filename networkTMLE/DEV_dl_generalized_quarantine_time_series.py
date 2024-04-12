# from sys import argv
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
# script_name, slurm_setup = argv
# script_name, slurm_setup = 'some_script', '10010'  
script_name, slurm_setup = 'some_script', '20040'  
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
use_deep_learner_outcome = True 

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
                                 epochs=30, print_every=5, device=device, save_path='./tmp.pth')
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
        ntmle.outcome_model(qn_model, custom_model=deep_learner_outcome)
    


    for p in prop_treated:  # loops through all treatment plans
        try:
            if shift:
                z = odds_to_probability(np.exp(log_odds + p))
                ntmle.fit(p=z, bound=0.01, seed=seed_number+i, 
                          shift=shift, mode=mode, percent_candidates=percent_candidates)
            else:
                ntmle.fit(p=p, bound=0.01, seed=seed_number+i,
                          shift=shift, mode=mode, percent_candidates=percent_candidates)
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

##################################### TEST CODE #####################################
ntmle.exposure_model(gin_model, custom_model=deep_learner_a_i) 
ntmle.exposure_map_model(gsn_model, measure=measure_gs, distribution=distribution_gs, custom_model=deep_learner_a_i_s)
if q_estimator is not None:
    ntmle.outcome_model(qn_model, custom_model=q_estimator) 
else:
    ntmle.outcome_model(qn_model, custom_model=deep_learner_outcome)


ntmle._Qinit_

pred = torch.from_numpy(ntmle._Qinit_)
torch.max(pred)
torch.min(pred)
# pred = torch.sigmoid(pred) # get binary probability
pred_binary = torch.round(pred) # get binary prediction
labels = ntmle.df_restricted_list[-1][ntmle.outcome]
labels = torch.from_numpy(labels.to_numpy())
acc = (pred_binary == labels).sum()/labels.numel()
print(f'self._Qinit_ accuracy: {acc:.3f}')

zero_ratio = 1 - labels.sum()/labels.numel()
print(f'zero_ratio: {zero_ratio:.3f}')

labels.sum()

''' Observed Data
uniform i=4
q_estimator accuracy: 0.820
DL accuracy: 0.182

random i=0
q_estimator accuracy: 0.6
DL accuracy: 0.484
'''

for i in range(10):
    labels = ntmle.df_restricted_list[i][ntmle.outcome]
    labels = torch.from_numpy(labels.to_numpy())
    print(labels)
    print()


def get_accuracy(pred:np.ndarray, label:pd.core.series.Series):
    pred = torch.from_numpy(pred)
    pred_binary = torch.round(pred)
    label = torch.from_numpy(label.to_numpy())
    acc = (pred_binary == label).sum()/label.numel()
    print(f'acc: {acc:.2f}')
    return pred_binary, label, acc

############### fit() ################
from tmle_utils import targeting_step, get_patsy_for_model_w_C, outcome_deep_learner_ts, outcome_learner_predict, tmle_unit_unbound
import patsy
p=0.7
samples = 500
bound = 0.01
seed = seed_number


# Step 1) Estimate the weights
# Also generates pooled_data for use in the Monte Carlo integration procedure
ntmle._resamples_ = samples                                                              # Saving info on number of resamples
h_iptw, pooled_data_restricted_list, pooled_adj_matrix_list = ntmle._estimate_iptw_ts_(p=p,                      # Generate pooled & estiamte weights
                                                                                       samples=samples,           # ... for some number of samples
                                                                                       bound=bound,               # ... with applied probability bounds
                                                                                       seed=seed,
                                                                                       shift=shift,
                                                                                       mode=mode,
                                                                                       percent_candidates=percent_candidates)                 # ... and with a random seed given

# Saving some information for diagnostic procedures
if ntmle._gs_measure_ is None:                                  # If no summary measure, use the A_sum
    ntmle._for_diagnostics_ = pooled_data_restricted_list[-1][[ntmle.exposure, ntmle.exposure+"_sum"]].copy()
else:                                                          # Otherwise, use the provided summary measure
    ntmle._for_diagnostics_ = pooled_data_restricted_list[-1][[ntmle.exposure, ntmle._gs_measure_]].copy()

# Step 2) Estimate from Q-model
# process completed in .outcome_model() function and stored in self._Qinit_
# so nothing to do here

# Step 3) Target the parameter
epsilon = targeting_step(y=ntmle.df_restricted_list[-1][ntmle.outcome],   # Estimate the targeting model given observed Y
                            q_init=ntmle._Qinit_,                           # ... predicted values of Y under observed A
                            ipw=h_iptw,                                    # ... weighted by IPW
                            verbose=ntmle._verbose_)                        # ... with option for verbose info

# Step 4) Monte Carlo integration (old code did in loop but faster as vector)
#
# Generating outcome predictions under the policies (via pooled data sets)
if ntmle._q_custom_ is None:                                                     # If given a parametric default model
    y_star = ntmle._outcome_model.predict(pooled_data_restricted_list[-1])       # ... predict using statsmodels syntax
else:  # Custom input model by user
    if ntmle.use_deep_learner_outcome:
        xdata_list = []
        ydata_list = []
        n_output_list = []
        for pooled_data_restricted in pooled_data_restricted_list:
            if 'C(' in ntmle._q_model:
                xdata_list.append(get_patsy_for_model_w_C(ntmle._q_model, pooled_data_restricted))
            else:
                xdata_list.append(patsy.dmatrix(ntmle._q_model + ' - 1', pooled_data_restricted, return_type="dataframe"))
            ydata_list.append(pooled_data_restricted[ntmle.outcome])
            n_output_list.append(pd.unique(pooled_data_restricted[ntmle.outcome]).shape[0])

        y_star = outcome_deep_learner_ts(ntmle._q_custom_, 
                                            xdata_list, ydata_list, ntmle.outcome, ntmle.use_all_time_slices,
                                            pooled_adj_matrix_list, ntmle.cat_vars, ntmle.cont_vars, ntmle.cat_unique_levels, n_output_list, ntmle._continuous_outcome_list_[-1],
                                            predict_with_best=True, custom_path=ntmle._q_custom_path_)
    else:
        d = patsy.dmatrix(ntmle._q_model + ' - 1', pooled_data_restricted_list[-1])  # ... extract data via patsy
        y_star = outcome_learner_predict(ml_model_fit=ntmle._q_custom_,              # ... predict using custom function
                                        xdata=np.asarray(d))                        # ... for the extracted data


pred_binary, labels, acc = get_accuracy(y_star, pooled_data_restricted_list[-1][ntmle.outcome])
pred_binary.sum()
labels.sum()
labels.numel()
1-labels.sum()/labels.numel()
    
# Ensure all predicted values are bounded properly for continuous
# SG modified: continous outcome is already normalized, should compare with 0,1, not with _continuous_min/max_
if ntmle._continuous_outcome_list_[-1]:
    y_star = np.where(y_star < 0., 0. + ntmle._cb_list_[-1], y_star)
    y_star = np.where(y_star > 1., 1. - ntmle._cb_list_[-1], y_star)     

# if self._continuous_outcome_list_[-1]:
#     y_star = np.where(y_star < self._continuous_min_list_[-1], self._continuous_min_list_[-1], y_star)
#     y_star = np.where(y_star > self._continuous_max_list_[-1], self._continuous_max_list_[-1], y_star)

# Updating predictions via intercept from targeting step
logit_qstar = np.log(probability_to_odds(y_star)) + epsilon                         # NOTE: needs to be logit(Y^*) + e
q_star = odds_to_probability(np.exp(logit_qstar))                                   # Back converting from odds
pooled_data_restricted_list[-1]['__pred_q_star__'] = q_star                         # Storing predictions as column

# Taking the mean, grouped-by the pooled sample IDs (fast)
ntmle.marginals_vector = np.asarray(pooled_data_restricted_list[-1].groupby('_sample_id_')['__pred_q_star__'].mean())

# If continuous outcome, need to unbound the means
if ntmle._continuous_outcome_list_[-1]:
    ntmle.marginals_vector = tmle_unit_unbound(ntmle.marginals_vector,                    # Take vector of MC results
                                                mini=ntmle._continuous_min_list_[-1],      # ... unbound using min
                                                maxi=ntmle._continuous_max_list_[-1])      # ... and max values

# Calculating estimate for the policy
ntmle.marginal_outcome = np.mean(ntmle.marginals_vector)                                  # Mean of Monte Carlo results
ntmle._specified_p_ = p                                                                  # Save what the policy was

# Prep for variance
if ntmle._continuous_outcome_list_[-1]:                                                  # Continuous needs bounds...
    y_ = np.array(tmle_unit_unbound(ntmle.df_restricted_list[-1][ntmle.outcome],          # Unbound observed outcomes for Var
                                    mini=ntmle._continuous_min_list_[-1],                # ... using min
                                    maxi=ntmle._continuous_max_list_[-1]))               # ... and max values
    yq0_ = tmle_unit_unbound(ntmle._Qinit_,                                              # Unbound g-comp predictions
                                mini=ntmle._continuous_min_list_[-1],                       # ... using min
                                maxi=ntmle._continuous_max_list_[-1])                       # ... and max values
else:                                                                                   # Otherwise nothing special...
    y_ = np.array(ntmle.df_restricted_list[-1][ntmle.outcome])                            # Observed outcome for Var
    yq0_ = ntmle._Qinit_                                                                 # Predicted outcome for Var

print(f'{ntmle.marginal_outcome} - {truth[p]} = {ntmle.marginal_outcome-truth[p]}')

''' Pooled Data
p=0.7

uniform
i=0
q_estimator acc = 0.91
bias = 0.10550313948472169 - 0.126 = -0.020496860515278312
DL acc = 0.84
bias = 0.103775404 - 0.126 = -0.022224595606327058

i=4
q_estimator acc = 0.82
bias = 0.173819749295999 - 0.126 = 0.047819749295999
DL acc = 0.19
bias = 0.18217044 - 0.126 = 0.05617043578624725

random
i=0
q_estimator acc = 0.58
bias = 0.5226838016094336 - 0.512 = 0.010683801609433607
DL acc = 0.48
bias = 0.6852731704711914 - 0.512 = 0.1732731704711914

'''
########################################
# Summarizing results
########################################
print("RESULTS\n")

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

print("===========================")

########################################
# Saving results
########################################
results.to_csv("results/" + exposure + str(sim_id) + "_" + save + ".csv", index=False)

