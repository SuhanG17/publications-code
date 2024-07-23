# from sys import argv
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic
from sklearn.linear_model import LogisticRegression

# from amonhen import NetworkTMLE
# from amonhen.utils import probability_to_odds, odds_to_probability, fast_exp_map

from tmle_dl import NetworkTMLE
from tmle_utils import probability_to_odds, odds_to_probability, fast_exp_map

from beowulf import load_uniform_vaccine, load_random_vaccine, truth_values, simulation_setup
from beowulf.dgm import vaccine_dgm
from beowulf.dgm.utils import network_to_df

import torch
from dl_trainer import MLP, GCN

############################################
# Setting simulation parameters
############################################
# n_mc = 500
n_mc = 2

exposure = "vaccine"
outcome = "D"

########################################
# Running through logic from .sh script
########################################
# script_name, slurm_setup = argv
script_name, slurm_setup = 'some_script', '10010'
network, n_nodes, degree_restrict, shift, model, save = simulation_setup(slurm_id_str=slurm_setup)
sim_id = slurm_setup[4]
seed_number = 18900567 + 10000000*int(sim_id)
np.random.seed(seed_number)

# random network with reproducibility
torch.manual_seed(17) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# decide use deep learning in which nuisance model
use_deep_learner_A_i = False
use_deep_learner_A_i_s = False 
use_deep_learner_outcome = True 

# decide which model to use
deep_learner_type = 'mlp' # 'mlp' or 'gcn'

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
    gin_model = "A + H + A_sum + H_sum + degree"
    gsn_model = "vaccine + A + H + A_sum + H_sum + degree"
    qn_model = "vaccine + vaccine_sum + A + H + A_sum + H_sum + degree"
elif model == "cw":
    gin_model = "A + H + A_sum + H_sum + degree"
    gsn_model = "vaccine + A + H + A_sum + H_sum + degree"
    qn_model = "vaccine + vaccine_sum + A + H + H_t3 + degree"
elif model == "wc":
    gin_model = "A + H + H_t3 + degree"
    gsn_model = "vaccine + A + H + H_t3 + degree"
    qn_model = "vaccine + vaccine_sum + A + H + A_sum + H_sum + degree"
elif model == 'np':
    gin_model = "A + H + C(A_sum_c) + C(H_sum_c) + degree"
    gsn_model = "vaccine + A + H + C(A_sum_c) + C(H_sum_c) + degree"
    qn_model = "vaccine + vaccine_sum + A + H + C(A_sum_c) + C(H_sum_c) + degree"
    q_estimator = LogisticRegression(penalty='l2', max_iter=2000)
elif model == 'ind':
    independent = True
    gi_model = "A + H"
    qi_model = "vaccine + A + H"

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
    prob = logistic.cdf(-1.9 + 1.75*data['A'] + 1.*data['H'] + 1.*data['H_sum'] + 1.3*data['A_sum']
                        - 0.65*data['degree'])
    log_odds = np.log(probability_to_odds(prob))

else:
    prop_treated = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

truth = truth_values(network=network, dgm=exposure,
                     restricted_degree=restrict, shift=shift, n=n_nodes)
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
    H, cat_vars, cont_vars, cat_unique_levels = vaccine_dgm(network=G, restricted=restrict, 
                                                            update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)
    df = network_to_df(H)
    results.loc[i, 'inc_'+exposure] = np.mean(df[exposure])
    results.loc[i, 'inc_'+outcome] = np.mean(df[outcome])

    # Network TMLE
    # use deep learner for given nuisance model
    if use_deep_learner_A_i:
        ntmle = NetworkTMLE(H, exposure=exposure, outcome=outcome, degree_restrict=degree_restrict,
                            cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
                            use_deep_learner_A_i=True)
    elif use_deep_learner_A_i_s:
        ntmle = NetworkTMLE(H, exposure=exposure, outcome=outcome, degree_restrict=degree_restrict,
                            cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
                            use_deep_learner_A_i_s=True)
    elif use_deep_learner_outcome:
        ntmle = NetworkTMLE(H, exposure=exposure, outcome=outcome, degree_restrict=degree_restrict,
                            cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
                            use_deep_learner_outcome=True)
    else: # DO NOT use deep learner
        ntmle = NetworkTMLE(H, exposure=exposure, outcome=outcome, degree_restrict=degree_restrict,
                            cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)


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
    if use_deep_learner_A_i or use_deep_learner_A_i_s or use_deep_learner_outcome:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # device = 'cpu'
        print(device)

        if deep_learner_type == 'mlp':
            deep_learner = MLP(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
                            epochs=10, print_every=5, device=device, save_path='./tmp.pth')
        elif deep_learner_type == 'gcn':
            deep_learner = GCN(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
                            epochs=10, print_every=5, device=device, save_path='./tmp.pth')
        else:
            raise NotImplementedError("Deep learner type not implemented")
        
        if use_deep_learner_A_i:
            ntmle.exposure_model(gin_model, custom_model=deep_learner) 
            ntmle.exposure_map_model(gsn_model, measure=measure_gs, distribution=distribution_gs)
            ntmle.outcome_model(qn_model, custom_model=q_estimator)
        elif use_deep_learner_A_i_s:
            ntmle.exposure_model(gin_model) 
            ntmle.exposure_map_model(gsn_model, measure=measure_gs, distribution=distribution_gs, custom_model=deep_learner) 
            ntmle.outcome_model(qn_model, custom_model=q_estimator) 
        elif use_deep_learner_outcome:
            ntmle.exposure_model(gin_model)
            ntmle.exposure_map_model(gsn_model, measure=measure_gs, distribution=distribution_gs) 
            ntmle.outcome_model(qn_model, custom_model=deep_learner) 
        else:
            raise ValueError("Deep learner should be used in given nuisance model, but not")
    else: # DO NOT use deep learner
        ntmle.exposure_model(gin_model)
        ntmle.exposure_map_model(gsn_model, measure=measure_gs, distribution=distribution_gs)
        ntmle.outcome_model(qn_model, custom_model=q_estimator)


    for p in prop_treated:  # loops through all treatment plans
        try:
            if shift:
                z = odds_to_probability(np.exp(log_odds + p))
                ntmle.fit(p=z, bound=0.01)
            else:
                ntmle.fit(p=p, bound=0.01)
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
