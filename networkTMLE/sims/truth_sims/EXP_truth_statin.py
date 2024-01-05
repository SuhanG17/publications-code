# from sys import argv
import numpy as np
import pandas as pd

from beowulf.dgm import statin_dgm_truth
from beowulf import load_uniform_statin, load_random_statin, simulation_setup

n_sims_truth = 10000
np.random.seed(20220109)

########################################
# Running through logic from .sh script
########################################
# script_name, slurm_setup = argv
script_name, slurm_setup = 'some_string', '2111'
network, n_nodes, degree_restrict, shift, model, save = simulation_setup(slurm_id_str=slurm_setup)

# Loading correct  Network
if network == "uniform":
    G = load_uniform_statin(n=n_nodes)
if network == "random":
    G = load_random_statin(n=n_nodes)

# Marking if degree restriction is being applied
if degree_restrict is not None:
    restrict = True
else:
    restrict = False

if shift:
    treat_plan = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]
else:
    treat_plan = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


print("#############################################")
print("Truth Sim Script:", slurm_setup)
print("=============================================")
print("Network:     ", network)
print("N-nodes:     ", n_nodes)
print("DGM:         ", 'statin')
print("Restricted:  ", restrict)
print("Shift:       ", shift)
print("#############################################")

########################################
# Running Truth
########################################

truth = {}
for t in treat_plan:
    ans = []
    for i in range(n_sims_truth):
        ans.append(statin_dgm_truth(network=G, pr_a=t, shift=shift, restricted=restrict))

    truth[t] = np.mean(ans)
    print(truth)

print("#############################################")
print(truth)
print("#############################################")



############################## Inside statin_dgm_truth ########################################
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic

from beowulf.dgm.utils import (network_to_df, fast_exp_map, exposure_restrictions,
                               odds_to_probability, probability_to_odds)

# params
network = G
pr_a = treat_plan[0]
# shift = shift
restricted = restrict

# code
graph = network.copy()
data = network_to_df(graph)

# Running Data Generating Mechanism for A
if shift:  # If a shift in the Odds distribution is instead specified
    prob = logistic.cdf(-5.3 + 0.2 * data['L'] + 0.15 * (data['A'] - 30)
                        + 0.4 * np.where(data['R_1'] == 1, 1, 0)
                        + 0.9 * np.where(data['R_2'] == 2, 1, 0)
                        + 1.5 * np.where(data['R_3'] == 3, 1, 0))
    odds = probability_to_odds(prob)
    pr_a = odds_to_probability(np.exp(np.log(odds) + pr_a))

statin = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
data['statin'] = statin

if restricted:  # removing other observations from the restricted set
    attrs = exposure_restrictions(network=network.graph['label'], exposure='statin',
                                    n=nx.number_of_nodes(graph))
    exclude = list(attrs.keys())
    data = data.loc[~data.index.isin(exclude)].copy()

# Running Data Generating Mechanism for Y
pr_y = logistic.cdf(-5.05 - 0.8*data['statin'] + 0.37*(np.sqrt(data['A']-39.9))
                    + 0.75*data['R'] + 0.75*data['L'])
cvd = np.random.binomial(n=1, p=pr_y, size=data.shape[0])

if restricted:
    data['cvd'] = cvd
    data = data.loc[~data.index.isin(exclude)].copy()
    cvd = np.array(data['cvd'])

np.mean(cvd)


def statin_dgm_truth(network, pr_a, shift=False, restricted=False):
    graph = network.copy()
    data = network_to_df(graph)

    # Running Data Generating Mechanism for A
    if shift:  # If a shift in the Odds distribution is instead specified
        prob = logistic.cdf(-5.3 + 0.2 * data['L'] + 0.15 * (data['A'] - 30)
                            + 0.4 * np.where(data['R_1'] == 1, 1, 0)
                            + 0.9 * np.where(data['R_2'] == 2, 1, 0)
                            + 1.5 * np.where(data['R_3'] == 3, 1, 0))
        odds = probability_to_odds(prob)
        pr_a = odds_to_probability(np.exp(np.log(odds) + pr_a))

    statin = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['statin'] = statin

    if restricted:  # removing other observations from the restricted set
        attrs = exposure_restrictions(network=network.graph['label'], exposure='statin',
                                      n=nx.number_of_nodes(graph))
        exclude = list(attrs.keys())
        data = data.loc[~data.index.isin(exclude)].copy()

    # Running Data Generating Mechanism for Y
    pr_y = logistic.cdf(-5.05 - 0.8*data['statin'] + 0.37*(np.sqrt(data['A']-39.9))
                        + 0.75*data['R'] + 0.75*data['L'])
    cvd = np.random.binomial(n=1, p=pr_y, size=data.shape[0])

    if restricted:
        data['cvd'] = cvd
        data = data.loc[~data.index.isin(exclude)].copy()
        cvd = np.array(data['cvd'])

    return np.mean(cvd)
