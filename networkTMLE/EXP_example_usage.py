############################################################################################
# Example of MossSpider with simulated data generating mechanisms
#   This file demonstrates basic usage of the NetworkTMLE class with some of the available
#       data generating mechanisms
############################################################################################

from mossspider import NetworkTMLE  # current library
# from tmle import NetworkTMLE         # modified version

# from amonhen import NetworkTMLE   # internal version, recommended to use library above instead
from beowulf import (sofrygin_observational, generate_sofrygin_network,
                     load_uniform_statin, load_random_naloxone, load_uniform_diet, load_random_vaccine)
from beowulf.dgm import statin_dgm, naloxone_dgm, diet_dgm, vaccine_dgm


###########################################
# Sofrygin & van der Laan (2017) Example

# generating data
sample_size = 1000
G = generate_sofrygin_network(n=sample_size, max_degree=2, seed=20200115)
H = sofrygin_observational(G)

# network-TMLE applied to generated data
tmle = NetworkTMLE(network=H, exposure='A', outcome='Y', verbose=False)
tmle.exposure_model('W + W_sum')
tmle.exposure_map_model('A + W + W_sum')  # by default a series of logistic models is used
tmle.outcome_model('A + A_sum + W + W_sum')
# Policy of setting everyone's probability of exposure to 0.35
tmle.fit(p=0.35, samples=100)
tmle.summary()
# Policy of setting everyone's probability of exposure to 0.65
tmle.fit(p=0.65, samples=100)
tmle.summary()

###########################################
# Statin-ASCVD -- DGM

# ######################### beowulf/load_networks.py  #########################
# import numpy as np
# import pandas as pd
# import networkx as nx
# import os

# def network_generator(edgelist, source, target, label):
#     """Reads in a NetworkX graph object from an edgelist

#     IDs need to be sequential from 0 to n_max for this function to behave as expected (and add nodes that have no edges)
#     """
#     graph = nx.Graph(label=label)

#     # adding edges
#     for i, j in zip(edgelist[source], edgelist[target]):
#         graph.add_edge(i, j)

#     return graph


# def load_uniform_network(n=500, dir_path='.'):
#     # file path to uniform network.
#     if n == 500:
#         edgelist = pd.read_csv(os.path.join(dir_path, 'data_files/network-uniform.csv'), index_col=False)
#     elif n == 1000:
#         edgelist = pd.read_csv(os.path.join(dir_path, 'data_files/network-uniform-1k.csv'), index_col=False)
#     elif n == 2000:
#         edgelist = pd.read_csv(os.path.join(dir_path, 'data_files/network-uniform-2k.csv'), index_col=False)
#     else:
#         raise ValueError("Invalid N for the network")

#     # call network_generator function
#     graph = network_generator(edgelist, source='source', target='target', label='uniform')
#     return graph


# def load_uniform_statin(n=500, dir_path='.'):
#     graph = load_uniform_network(n=n, dir_path=dir_path)

#     # adding attributes to the network
#     if n == 500:
#         attrs = pd.read_csv(os.path.join(dir_path, 'data_files/dgm-statin-uniform.csv'), index_col=False)
#     elif n == 1000:
#         attrs = pd.read_csv(os.path.join(dir_path, 'data_files/dgm-statin-uniform-1k.csv'), index_col=False)
#     elif n == 2000:
#         attrs = pd.read_csv(os.path.join(dir_path, 'data_files/dgm-statin-uniform-2k.csv'), index_col=False)
#     else:
#         raise ValueError("Invalid N for the network")

#     attrs['R_1'] = np.where((attrs['R'] >= .05) & (attrs['R'] < .075), 1, 0)
#     attrs['R_2'] = np.where((attrs['R'] >= .075) & (attrs['R'] < .2), 1, 0)
#     attrs['R_3'] = np.where(attrs['R'] >= .2, 1, 0)
#     attrs['A_30'] = attrs['A'] - 30
#     attrs['A_sqrt'] = np.sqrt(attrs['A']-39.9)

#     for n in graph.nodes():
#         graph.nodes[n]['A'] = int(attrs.loc[attrs['id'] == n, 'A'].values)
#         graph.nodes[n]['L'] = float(attrs.loc[attrs['id'] == n, 'L'].values)
#         graph.nodes[n]['R'] = float(attrs.loc[attrs['id'] == n, 'R'].values)
#         graph.nodes[n]['R_1'] = int(attrs.loc[attrs['id'] == n, 'R_1'].values)
#         graph.nodes[n]['R_2'] = int(attrs.loc[attrs['id'] == n, 'R_2'].values)
#         graph.nodes[n]['R_3'] = int(attrs.loc[attrs['id'] == n, 'R_3'].values)
#         graph.nodes[n]['A_30'] = int(attrs.loc[attrs['id'] == n, 'A_30'].values)
#         graph.nodes[n]['A_sqrt'] = float(attrs.loc[attrs['id'] == n, 'A_sqrt'].values)

#     return nx.convert_node_labels_to_integers(graph)


# # Loading uniform network with statin W
# dir_path = '/root/autodl-tmp/causal_inference/publications-code/networkTMLE/Beowulf/beowulf'
# G = load_uniform_statin(dir_path=dir_path)

G = load_uniform_statin()
# Simulation single instance of exposure and outcome
H = statin_dgm(network=G)

# network-TMLE applies to generated data
tmle = NetworkTMLE(H, exposure='statin', outcome='cvd')
tmle.exposure_model("L + A_30 + R_1 + R_2 + R_3")
tmle.exposure_map_model("statin + L + A_30 + R_1 + R_2 + R_3",
                        measure='sum', distribution='poisson')  # Applying a Poisson model
tmle.outcome_model("statin + statin_sum + A_sqrt + R + L")
tmle.fit(p=0.35, bound=0.01)
tmle.summary()

############## details in mossspider estimator tmle.py #########################
import re
import warnings
import patsy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
from scipy.stats.kde import gaussian_kde

from EXP_mossspider_estimators_utils import (network_to_df, fast_exp_map, exp_map_individual, tmle_unit_bounds, tmle_unit_unbound,
                                             probability_to_odds, odds_to_probability, bounding,
                                             outcome_learner_fitting, outcome_learner_predict, exposure_machine_learner,
                                             targeting_step, create_threshold, create_categorical)

network = H
exposure = 'statin'
outcome = 'cvd'
degree_restrict=None
alpha=0.05
continuous_bound=0.0005
verbose=False

######################## NetworkTMLE._check_degree_restrictions_ ########################
def _check_degree_restrictions_(bounds):
    """Checks degree restrictions are valid (and won't cause a later error).

    Parameters
    ----------
    bounds : list, set, array
        Specified degree bounds
    """
    if type(bounds) is not list and type(bounds) is not tuple:
        raise ValueError("`degree_restrict` should be a list/tuple of the upper and lower bounds")
    if len(bounds) != 2:
        raise ValueError("`degree_restrict` should only have two values")
    if bounds[0] > bounds[1]:
        raise ValueError("Degree restrictions must be specified in ascending order")

######################## NetworkTMLE._degree_restrictions_ ########################
def _degree_restrictions_(degree_dist, bounds):
    """Bounds the degree by the specified levels

    Parameters
    ----------
    degree_dist : array
        Degree values for the observations
    bounds : list, set, array
        Upper and lower bounds to use for the degree restriction
    """
    restrict = np.where(degree_dist < bounds[0], 1, 0)            # Apply the lower restriction on degree
    restrict = np.where(degree_dist > bounds[1], 1, restrict)     # Apply the upper restriction on degree
    return restrict

######################## NetworkTMLE._estimate_exposure_nuisance_ ########################
def _estimate_exposure_nuisance_(data_to_fit, data_to_predict, distribution, verbose_label, store_model,
                                 _gi_custom_, _gi_model, _gs_custom_, _gs_model, _gs_measure_, _nonparam_cols_,
                                 exposure, _verbose_, _treatment_models):
    """Unified function to estimate the numerator and denominator of the weights.

    Parameters
    ----------
    data_to_fit : dataframe
        Data to estimate the parameters of the nuisance model with. For the numerator, this is the pooled data set
        and for the denominator this is the observed data set. Both are restricted by degree, when applicable.
    data_to_predict : dataframe
        Data to generate predictions for from the fitted nuisance models. Will always be the observed data set,
        restricted by degree when applicable.
    distribution : None, str
        Distribution to use for the A^s model
    verbose_label : str
        When verbose is used, this gives an informative label for the model being estimated (i.e., it identifies
        whether it is the numerator or denominator model being fit)
    store_model : bool
        Identifies whether the nuisance models should be stored. This is to be set to True for the denominator
        models and False for the numerator models

    Returns
    -------
    array
        Predicted probability for the corresponding piece of the weights.
    """
    ##################################
    # Model: A_i
    if _gi_custom_ is None:                                          # If no A_i custom_model is provided
        f = sm.families.family.Binomial()                                 # ... use a logit model
        treat_i_model = smf.glm(exposure + ' ~ ' + _gi_model,   # Specified model from exposure_...()
                                data_to_fit,                              # ... using the restricted data
                                family=f).fit()                           # ... for logit family
        # Verbose model results if requested
        if _verbose_:
            print('==============================================================================')
            print(verbose_label+': A')
            print(treat_i_model.summary())

        pred = treat_i_model.predict(data_to_predict)                     # Save predicted probability
        if store_model:                                                   # If denominator,
            _treatment_models.append(treat_i_model)                  # save model to list (so can be extracted)

    else:                                                                 # Otherwise use the custom_model
        xdata = patsy.dmatrix(_gi_model + ' - 1', data_to_fit)       # Extract via patsy the data
        pdata = patsy.dmatrix(_gi_model + ' - 1', data_to_predict)   # Extract via patsy the data
        pred = exposure_machine_learner(ml_model=_gi_custom_,        # Custom model application and preds
                                        xdata=np.asarray(xdata),          # ... with data to fit
                                        ydata=np.asarray(data_to_fit[exposure]),
                                        pdata=np.asarray(pdata))          # ... and data to predict

    # Assigning probability given observed
    pr_i = np.where(data_to_predict[exposure] == 1,                  # If A_i = 1
                    pred,                                                 # ... get Pr(A_i=1 | ...)
                    1 - pred)                                             # ... otherwise get Pr(A_i=0 | ...)

    ##################################
    # Model: A_i^s
    if distribution is None:                                              # When no distribution is provided
        if _gs_custom_ is None:                                      # and no custom_model is given
            f = sm.families.family.Binomial()                             # ... use a logit model
            cond_vars = patsy.dmatrix(_gs_model,                     # Extract initial set of covs
                                      data_to_fit,                        # ... from the data to fit
                                      return_type='matrix')               # ... as a NumPy matrix
            pred_vars = patsy.dmatrix(_gs_model,                     # Extract initial set of covs
                                      data_to_predict,                    # ... from the data to predict
                                      return_type='matrix')               # ... as a NumPy matrix
            pr_s = np.array([1.] * data_to_predict.shape[0])              # Setup vector of 1's as the probability

            for c in _nonparam_cols_:                                # For each of the NP columns
                treat_s_model = sm.GLM(data_to_fit[c], cond_vars,         # Estimate using the pooled data
                                        family=f).fit()                    # ... with logistic model
                if store_model:                                           # If estimating denominator
                    _treatment_models.append(treat_s_model)          # Save estimated model for checking

                # If verbose requested, provide model output
                if _verbose_:
                    print('==============================================================================')
                    print(verbose_label+': ' + c)
                    print(treat_s_model.summary())

                # Generating predicted probabilities
                pred = treat_s_model.predict(pred_vars)                   # Generate the prediction for observed
                pr_s *= np.where(data_to_predict[c] == 1,                 # cumulative probability for each step
                                 pred, 1 - pred)                          # ...logic for correct probability value

                # Stacking vector to the end of the array for next cycle
                cond_vars = np.c_[cond_vars, np.array(data_to_fit[c])]       # Update the covariates
                pred_vars = np.c_[pred_vars, np.array(data_to_predict[c])]   # Update the covariates

        else:                                                     # Placeholder for conditional density SL
            # TODO fill in the super-learner conditional density approach here when possible...
            raise ValueError("Not available currently...")

    elif distribution == 'normal':                                # If a normal distribution
        gs_model = _gs_measure_ + ' ~ ' + _gs_model     # Setup the model form
        treat_s_model = smf.ols(gs_model,                         # Estimate via OLS
                                data_to_fit).fit()                # ... with data to fit
        _treatment_models.append(treat_s_model)              # Store estimated model to check
        pred = treat_s_model.predict(data_to_predict)             # Generate predicted values for data to predict
        pr_s = norm.pdf(data_to_predict[_gs_measure_],       # Get f(A_i^s | ...) for measure
                        pred,                                     # ... given predicted value
                        np.sqrt(treat_s_model.mse_resid))         # ... and sqrt of model residual

        # If verbose requested, provide model output
        if _verbose_:
            print('==============================================================================')
            print(verbose_label+': '+_gs_measure_)
            print(treat_s_model.summary())

    elif distribution == 'poisson':                               # If a poisson distribution
        gs_model = _gs_measure_ + ' ~ ' + _gs_model     # Setup the model form
        if _gs_custom_ is None:                              # If no custom model provided
            f = sm.families.family.Poisson()                      # ... GLM with Poisson family
            treat_s_model = smf.glm(gs_model,                     # Estimate model
                                    data_to_fit,                  # ... with data to fit
                                    family=f).fit()               # ... and Poisson distribution
            if store_model:                                       # If estimating denominator
                _treatment_models.append(treat_s_model)      # ... store the model
            pred = treat_s_model.predict(data_to_predict)         # Predicted values with data to predict

            # If verbose requested, provide model output
            if _verbose_:
                print('==============================================================================')
                print(verbose_label+': '+_gs_measure_)
                print(treat_s_model.summary())

        else:                                                           # Custom model for Poisson
            xdata = patsy.dmatrix(_gs_model + ' - 1',              # ... extract data given relevant model
                                  data_to_fit)                          # ... from degree restricted
            pdata = patsy.dmatrix(_gs_model + ' - 1',              # ... extract data given relevant model
                                  data_to_predict)                      # ... from degree restricted
            pred = exposure_machine_learner(ml_model=_gs_custom_,  # Custom ML model
                                            xdata=np.asarray(xdata),    # ... with data to fit
                                            ydata=np.asarray(data_to_fit[_gs_measure_]),
                                            pdata=np.asarray(pdata))    # ... and data to predict

        pr_s = poisson.pmf(data_to_predict[_gs_measure_], pred)    # Get f(A_i^s | ...) for measure

    elif distribution == 'multinomial':                              # If multinomial distribution
        gs_model = _gs_measure_ + ' ~ ' + _gs_model        # Setup the model form
        treat_s_model = smf.mnlogit(gs_model,                        # Estimate multinomial model
                                    data_to_fit).fit(disp=False)     # ... with data to fit
        if store_model:                                              # If estimating denominator
            _treatment_models.append(treat_s_model)             # ... add fitted model to list of models

        pred = treat_s_model.predict(data_to_predict)                # predict probabilities for each category
        values = pd.get_dummies(data_to_predict[_gs_measure_])  # transform to dummy variables for processing
        pr_s = np.array([0.0] * data_to_predict.shape[0])            # generate blank array of probabilities
        for i in data_to_predict[_gs_measure_].unique():        # for each unique value in the multinomial
            try:                                                     # ... try-except skips if unique not occur
                pr_s += pred[i] * values[i]                          # ... update probability
            except KeyError:                                         # ... logic to skip the KeyError's
                pass

        # If verbose requested, provide model output
        if _verbose_:
            print('==============================================================================')
            print(verbose_label+': '+_gs_measure_)
            print(treat_s_model.summary())

    elif distribution == 'binomial':                                 # If binomial distribution
        gs_model = _gs_measure_ + ' ~ ' + _gs_model        # setup the model form
        f = sm.families.family.Binomial()                            # specify the logistic family option
        treat_s_model = smf.glm(gs_model,                            # Estimate the model
                                data_to_fit,                         # ... with data to fit
                                family=f).fit()                      # ... and logistic model
        if store_model:                                              # If estimating denominator
            _treatment_models.append(treat_s_model)             # ... add fitted model to list of models
        pr_s = treat_s_model.predict(data_to_predict)                # generate predicted probabilities of As=1

        # If verbose requested, provide model output
        if _verbose_:
            print('==============================================================================')
            print(verbose_label+': '+_gs_measure_)
            print(treat_s_model.summary())

    elif distribution == 'threshold':                                # If distribution is a threshold
        gs_model = _gs_measure_ + ' ~ ' + _gs_model        # setup the model form
        if _gs_custom_ is None:                                 # if no custom model is given
            f = sm.families.family.Binomial()                        # ... logistic model
            treat_s_model = smf.glm(gs_model,                        # Estimate the model
                                    data_to_fit,                     # ... with data to fit
                                    family=f).fit()                  # ... and logistic model
            if store_model:                                          # If estimating the denominator
                _treatment_models.append(treat_s_model)         # ... add fitted model to list of models
            pred = treat_s_model.predict(data_to_predict)            # Generate predicted values of As=threshold

            # If verbose requested, provide model output
            if _verbose_:
                print('==============================================================================')
                print('g-model: '+_gs_measure_)
                print(treat_s_model.summary())
        else:                                                                 # Else custom model for threshold
            xdata = patsy.dmatrix(_gs_model + ' - 1', data_to_fit)       # Processing data to be fit
            pdata = patsy.dmatrix(_gs_model + ' - 1', data_to_predict)   # Processing data to be fit
            pred = exposure_machine_learner(ml_model=_gs_custom_,        # Estimating the ML
                                            xdata=np.asarray(xdata),          # ... with data to fit
                                            ydata=np.asarray(data_to_fit[_gs_measure_]),
                                            pdata=np.asarray(xdata))          # ... and data to predict
        pr_s = np.where(data_to_predict[_gs_measure_] == 1,              # Getting predicted values
                        pred,
                        1 - pred)

    else:
        raise ValueError("Invalid distribution choice")

    ##################################
    # Creating estimated Pr(A,A^s|W,W^s)
    return pr_i * pr_s    # Multiplying the factored probabilities back together

# # params
# data_to_fit=df_restricted.copy()
# data_to_predict=df_restricted.copy()
# distribution=_map_dist_
# verbose_label='Weight - Denominator'
# store_model=True

# # code
# # Model: A_i
# if _gi_custom_ is None:                                          # If no A_i custom_model is provided
#     f = sm.families.family.Binomial()                                 # ... use a logit model
#     treat_i_model = smf.glm(exposure + ' ~ ' + _gi_model,   # Specified model from exposure_...()
#                             data_to_fit,                              # ... using the restricted data
#                             family=f).fit()                           # ... for logit family
#     # Verbose model results if requested
#     if _verbose_:
#         print('==============================================================================')
#         print(verbose_label+': A')
#         print(treat_i_model.summary())

#     pred = treat_i_model.predict(data_to_predict)                     # Save predicted probability
#     if store_model:                                                   # If denominator,
#         _treatment_models.append(treat_i_model)                  # save model to list (so can be extracted)

# else:                                                                 # Otherwise use the custom_model
#     xdata = patsy.dmatrix(_gi_model + ' - 1', data_to_fit)       # Extract via patsy the data
#     pdata = patsy.dmatrix(_gi_model + ' - 1', data_to_predict)   # Extract via patsy the data
#     pred = exposure_machine_learner(ml_model=_gi_custom_,        # Custom model application and preds
#                                     xdata=np.asarray(xdata),          # ... with data to fit
#                                     ydata=np.asarray(data_to_fit[exposure]),
#                                     pdata=np.asarray(pdata))          # ... and data to predict

# # Assigning probability given observed
# pr_i = np.where(data_to_predict[exposure] == 1,                  # If A_i = 1
#                 pred,                                                 # ... get Pr(A_i=1 | ...)
#                 1 - pred)                                             # ... otherwise get Pr(A_i=0 | ...)

# # Model: A_i^s
# # elif distribution == 'poisson':                               # If a poisson distribution
# gs_model = _gs_measure_ + ' ~ ' + _gs_model     # Setup the model form
# if _gs_custom_ is None:                              # If no custom model provided
#     f = sm.families.family.Poisson()                      # ... GLM with Poisson family
#     treat_s_model = smf.glm(gs_model,                     # Estimate model
#                             data_to_fit,                  # ... with data to fit
#                             family=f).fit()               # ... and Poisson distribution
#     if store_model:                                       # If estimating denominator
#         _treatment_models.append(treat_s_model)      # ... store the model
#     pred = treat_s_model.predict(data_to_predict)         # Predicted values with data to predict

#     # If verbose requested, provide model output
#     if _verbose_:
#         print('==============================================================================')
#         print(verbose_label+': '+_gs_measure_)
#         print(treat_s_model.summary())

# else:                                                           # Custom model for Poisson
#     xdata = patsy.dmatrix(_gs_model + ' - 1',              # ... extract data given relevant model
#                             data_to_fit)                          # ... from degree restricted
#     pdata = patsy.dmatrix(_gs_model + ' - 1',              # ... extract data given relevant model
#                             data_to_predict)                      # ... from degree restricted
#     pred = exposure_machine_learner(ml_model=_gs_custom_,  # Custom ML model
#                                     xdata=np.asarray(xdata),    # ... with data to fit
#                                     ydata=np.asarray(data_to_fit[_gs_measure_]),
#                                     pdata=np.asarray(pdata))    # ... and data to predict

# pr_s = poisson.pmf(data_to_predict[_gs_measure_], pred)    # Get f(A_i^s | ...) for measure




######################## NetworkTMLE._generate_pooled_sample ########################
def _generate_pooled_sample(p, samples, seed,
                            df, exposure, adj_matrix, network, _max_degree_, _gs_measure_, _nonparam_cols_,
                            _thresholds_any_, _thresholds_variables_, _thresholds_, _thresholds_def_,
                            _categorical_any_, _categorical_variables_, _categorical_, _categorical_def_):
    """

    Note
    ----
    Vectorization doesn't work, since the matrix manipulations get extremely large (even when using
    scipy.sparse.block_diag()). So here the loop is more efficient due to how the summary measures are being
    calculated via matrix multiplication.

    Parameters
    ----------
    p : float, array
        Probability of A_i as assigned by the policy
    samples : int
        Number of sampled data sets to generate
    seed : None, int
        Seed for pooled data set creation

    Returns
    -------
    dataframe
        Pooled data set under applications of the policy omega
    """
    # Prep for pooled data set creation
    rng = np.random.default_rng(seed)  # Setting the seed for bootstraps
    pooled_sample = []
    # TODO one way to potentially speed up code is to run this using Pool. Easy for parallel
    # this is also the best target for optimization since it takes about ~85% of current run times

    for s in range(samples):                                    # For each of the *m* samples
        g = df.copy()                                      # Create a copy of the data
        probs = rng.binomial(n=1,                               # Flip a coin to generate A_i
                             p=p,                               # ... based on policy-assigned probabilities
                             size=g.shape[0])                   # ... for the N units
        g[exposure] = np.where(g['__degree_flag__'] == 1,  # Restrict to appropriate degree
                               g[exposure], probs)    # ... keeps restricted nodes as observed A_i

        # Generating all summary measures based on the new exposure (could maybe avoid for all?)
        g[exposure+'_sum'] = fast_exp_map(adj_matrix, np.array(g[exposure]), measure='sum')
        g[exposure + '_mean'] = fast_exp_map(adj_matrix, np.array(g[exposure]), measure='mean')
        g[exposure + '_mean'] = g[exposure + '_mean'].fillna(0)            # isolates should have mean=0
        g[exposure + '_var'] = fast_exp_map(adj_matrix, np.array(g[exposure]), measure='var')
        g[exposure + '_var'] = g[exposure + '_var'].fillna(0)              # isolates should have mean=0
        g[exposure + '_mean_dist'] = fast_exp_map(adj_matrix,
                                                        np.array(g[exposure]), measure='mean_dist')
        g[exposure + '_mean_dist'] = g[exposure + '_mean_dist'].fillna(0)  # isolates should have mean=0
        g[exposure + '_var_dist'] = fast_exp_map(adj_matrix,
                                                        np.array(g[exposure]), measure='var_dist')
        g[exposure + '_mean_dist'] = g[exposure + '_mean_dist'].fillna(0)  # isolates should have mean=0

        # Logic if no summary measure was specified (uses the complete factor approach)
        if _gs_measure_ is None:
            network = network.copy()                           # Copy the network
            a = np.array(g[exposure])                          # Transform A_i into array
            for n in network.nodes():                               # For each node,
                network.nodes[n][exposure] = a[n]              # ...assign the new A_i*
            df = exp_map_individual(network,                        # Now do the individual exposure maps with new
                                    variable=exposure,
                                    max_degree=_max_degree_).fillna(0)
            for c in _nonparam_cols_:                          # Adding back these np columns
                g[c] = df[c]

        # Re-creating any threshold variables in the pooled sample data
        if _thresholds_any_:
            create_threshold(data=g,
                                variables=_thresholds_variables_,
                                thresholds=_thresholds_,
                                definitions=_thresholds_def_)

        # Re-creating any categorical variables in the pooled sample data
        if _categorical_any_:
            create_categorical(data=g,
                                variables=_categorical_variables_,
                                bins=_categorical_,
                                labels=_categorical_def_,
                                verbose=False)

        g['_sample_id_'] = s         # Setting sample ID
        pooled_sample.append(g)      # Adding to list (for later concatenate)

    # Returning the pooled data set
    return pd.concat(pooled_sample, axis=0, ignore_index=True)

# # params
# p=0.35
# samples=100
# seed=None

# # code
# # Prep for pooled data set creation
# rng = np.random.default_rng(seed)  # Setting the seed for bootstraps
# pooled_sample = []

# for s in range(samples):                                    # For each of the *m* samples
#     g = df.copy()                                      # Create a copy of the data
#     probs = rng.binomial(n=1,                               # Flip a coin to generate A_i
#                          p=p,                               # ... based on policy-assigned probabilities
#                          size=g.shape[0])                   # ... for the N units
#     g[exposure] = np.where(g['__degree_flag__'] == 1,  # Restrict to appropriate degree
#                            g[exposure], probs)    # ... keeps restricted nodes as observed A_i

#     # Generating all summary measures based on the new exposure (could maybe avoid for all?)
#     g[exposure+'_sum'] = fast_exp_map(adj_matrix, np.array(g[exposure]), measure='sum')
#     g[exposure + '_mean'] = fast_exp_map(adj_matrix, np.array(g[exposure]), measure='mean')
#     g[exposure + '_mean'] = g[exposure + '_mean'].fillna(0)            # isolates should have mean=0
#     g[exposure + '_var'] = fast_exp_map(adj_matrix, np.array(g[exposure]), measure='var')
#     g[exposure + '_var'] = g[exposure + '_var'].fillna(0)              # isolates should have mean=0
#     g[exposure + '_mean_dist'] = fast_exp_map(adj_matrix,
#                                               np.array(g[exposure]), measure='mean_dist')
#     g[exposure + '_mean_dist'] = g[exposure + '_mean_dist'].fillna(0)  # isolates should have mean=0
#     g[exposure + '_var_dist'] = fast_exp_map(adj_matrix,
#                                              np.array(g[exposure]), measure='var_dist')
#     g[exposure + '_mean_dist'] = g[exposure + '_mean_dist'].fillna(0)  # isolates should have mean=0

#     # Logic if no summary measure was specified (uses the complete factor approach)
#     if _gs_measure_ is None:
#         network = network.copy()                           # Copy the network
#         a = np.array(g[exposure])                          # Transform A_i into array
#         for n in network.nodes():                               # For each node,
#             network.nodes[n][exposure] = a[n]              # ...assign the new A_i*
#         df = exp_map_individual(network,                        # Now do the individual exposure maps with new
#                                 variable=exposure,
#                                 max_degree=_max_degree_).fillna(0)
#         for c in _nonparam_cols_:                          # Adding back these np columns
#             g[c] = df[c]

#     # Re-creating any threshold variables in the pooled sample data
#     if _thresholds_any_:
#         create_threshold(data=g,
#                          variables=_thresholds_variables_,
#                          thresholds=_thresholds_,
#                          definitions=_thresholds_def_)

#     # Re-creating any categorical variables in the pooled sample data
#     if _categorical_any_:
#         create_categorical(data=g,
#                            variables=_categorical_variables_,
#                            bins=_categorical_,
#                            labels=_categorical_def_,
#                            verbose=False)

#     g['_sample_id_'] = s         # Setting sample ID
#     pooled_sample.append(g)      # Adding to list (for later concatenate)


######################## NetworkTMLE._estimate_iptw_ ########################
def _estimate_iptw_(p, samples, bound, seed,
                    _denominator_estimated_, _denominator_, df_restricted, _map_dist_, 
                    _gi_custom_, _gi_model, _gs_custom_, _gs_model, _gs_measure_, _nonparam_cols_,
                    exposure, _verbose_, _treatment_models,
                    df, adj_matrix, network, _max_degree_, 
                    _thresholds_any_, _thresholds_variables_, _thresholds_, _thresholds_def_,
                    _categorical_any_, _categorical_variables_, _categorical_, _categorical_def_):
    """Background function to estimate the IPTW based on the algorithm described in Sofrygin & van der Laan Journal
    of Causal Inference 2017

    IPTW are estimated using the following process.

    For the observed data, models are fit to estimate the Pr(A=a) for individual i (treating as IID data) and then
    the Pr(A=a) for their contacts (treated as IID data). These probabilities are then multiplied together to
    generate the denominator.

    To calculate the numerator, the input data set is replicated `samples` times. To each of the data set copies,
    the treatment plan is repeatedly applied. From this large set of observations under the stochastic treatment
    plan of interest, models are again fit to the data, same as the prior procedure. The corresponding probabilities
    are then multiplied together to generate the numerator.

    Note: not implemented but the `deterministic` argument will use the following procedure. When a deterministic
    treatment plan (like all-treat)vis input, only a single data set under the treatment plan is generated. This
    saves computation time since all the replicate data sets would be equivalent. The deterministic part will be
    resolved in an earlier procedure

    Parameters
    ----------
    p : float, array
        Probability of A_i as assigned by the policy
    samples : int
        Number of sampled data sets to generate
    bound : None, int, float
        Bounds to truncate calculate weights with
    seed : None, int
        Seed for pooled data set creation
    """
    # Estimate the denominator if not previously estimated
    if not _denominator_estimated_:
        _denominator_ = _estimate_exposure_nuisance_(data_to_fit=df_restricted.copy(),
                                                     data_to_predict=df_restricted.copy(),
                                                     distribution=_map_dist_,
                                                     verbose_label='Weight - Denominator',
                                                     store_model=True,
                                                     _gi_custom_=_gi_custom_, _gi_model=_gi_model, 
                                                     _gs_custom_=_gs_custom_, _gs_model=_gs_model, 
                                                     _gs_measure_=_gs_measure_, _nonparam_cols_=_nonparam_cols_,
                                                     exposure=exposure, _verbose_=_verbose_, _treatment_models=_treatment_models)
        _denominator_estimated_ = True  # Updates flag for denominator

    # Creating pooled sample to estimate weights
    pooled_df = _generate_pooled_sample(p=p,                                      # Generate data under policy
                                        samples=samples,                          # ... for m samples
                                        seed=seed,
                                        df=df, exposure=exposure, adj_matrix=adj_matrix, network=network, _max_degree_=_max_degree_,
                                        _gs_measure_=_gs_measure_, _nonparam_cols_=_nonparam_cols_,
                                        _thresholds_any_=_thresholds_any_, _thresholds_variables_=_thresholds_variables_, _thresholds_=_thresholds_, _thresholds_def_=_thresholds_def_,
                                        _categorical_any_=_categorical_any_, _categorical_variables_=_categorical_variables_, _categorical_=_categorical_, _categorical_def_=_categorical_def_)                                # ... with a provided seed
    pooled_data_restricted = pooled_df.loc[pooled_df['__degree_flag__'] == 0].copy()   # Restricting pooled sample

    # Estimate the numerator using the pooled data
    numerator = _estimate_exposure_nuisance_(data_to_fit=pooled_data_restricted.copy(),
                                             data_to_predict=df_restricted.copy(),
                                             distribution=_map_dist_,
                                             verbose_label='Weight - Numerator',
                                             store_model=False,
                                             _gi_custom_=_gi_custom_, _gi_model=_gi_model,
                                             _gs_custom_=_gs_custom_, _gs_model=_gs_model,
                                             _gs_measure_=_gs_measure_, _nonparam_cols_=_nonparam_cols_,
                                             exposure=exposure, _verbose_=_verbose_, _treatment_models=_treatment_models)

    # Calculating weight: H = Pr*(A,A^s | W,W^s) / Pr(A,A^s | W,W^s)
    iptw = numerator / _denominator_           # Divide numerator by denominator
    if bound is not None:                           # If weight bound provided
        iptw = bounding(ipw=iptw, bound=bound)      # ... apply the bound

    # Return both the array of estimated weights and the generated pooled data set
    return iptw, pooled_data_restricted


######################## NetworkTMLE._est_variance_conditional_ ########################
def _est_variance_conditional_(iptw, obs_y, pred_y):
    """Variance estimator from Sofrygin & van der Laan 2017; section 6.3

    Parameters
    ----------
    iptw : array
        Estimated weights
    obs_y : array
        Observed outcomes
    pred_y : array
        Predicted outcomes under observed values of A

    Returns
    -------
    float
        Estimated variance
    """
    return np.mean((iptw * (obs_y - pred_y))**2) / iptw.shape[0]

######################## NetworkTMLE._est_variance_latent_conditional_ ########################
def _est_variance_latent_conditional_(iptw, obs_y, pred_y, adj_matrix, excluded_ids=None):
    """Variance estimator from Sofrygin & van der Laan 2017; section 6.3 adapted for latent dependence as
    described in Ogburn et al. (2017).

    Parameters
    ----------
    iptw : array
        Estimated weights
    obs_y : array
        Observed outcomes
    pred_y : array
        Predicted outcomes under observed values of A
    adj_matrix : array
        Adjacency matrix for the full network
    excluded_ids : None, array, optional
        Used to expand the pseudo-Y array at the right places with zeroes to align the predictions and the network
        together for the quick version of this variance calculation.

    Returns
    -------
    float
        Estimated variance
    """
    # Matrix of all possible pairs of pseudo-outcomes
    pseudo_y = iptw * (obs_y - pred_y)                # Calculates the pseudo Y values
    pseudo_y_matrix = np.outer(pseudo_y, pseudo_y)    # Create matrix of outer product of pseudo Y values

    # Fill in the diagonal with 1's (ensures i's self-pair contributes to the variance)
    pair_matrix = adj_matrix.toarray()                # create a paired matrix from the adjacency
    np.fill_diagonal(pair_matrix, val=1)              # everyone becomes a friend with themselves for the variance

    # If there is a degree restriction, the following code is needed. Briefly, it fills-in Zeros in the pseudo-Y
    #   matrix, so it is the same dimensions as the adjacency matrix. This is better than a for loop for all steps
    if excluded_ids is not None:
        for eids in sorted(excluded_ids):  # need to for loop over so all are inserted at correct spot!!
            pseudo_y_matrix = np.insert(pseudo_y_matrix, obj=eids, values=0, axis=1)
            pseudo_y_matrix = np.insert(pseudo_y_matrix, obj=eids, values=0, axis=0)

    # var is sum over element-wise multiplied matrices
    return np.sum(pair_matrix * pseudo_y_matrix) / (iptw.shape[0]**2)  # squared here since np.sum replaces np.mean

######################## NetworkTMLE._check_distribution_measure_ ########################
def _check_distribution_measure_(distribution, measure):
    """Checks whether the distribution and measure specified are compatible

    Parameters
    ----------
    distribution : str, None
        Distribution to use for the exposure map nuisance model
    measure : str
        Summary measure or mapping to use for A^s
    """
    if distribution is None:
        if measure is not None:
            raise ValueError("The distribution `None` and `"+str(measure)+"` are not compatible")
    elif distribution.lower() == 'normal':
        if measure not in ['sum', 'mean', 'var', 'mean_dist', 'var_dist']:
            raise ValueError("The distribution `"+str(distribution)+"` and `"+str(measure)+"` are not compatible")
    elif distribution.lower() == 'poisson':
        if measure not in ['sum', 'mean']:
            raise ValueError("The distribution `"+str(distribution)+"` and `"+str(measure)+"` are not compatible")
    elif distribution.lower() == 'multinomial':
        if measure not in ['sum', 'sum_c', 'mean_c', 'var_c', 'mean_dist_c', 'var_dist_c']:
            raise ValueError("The distribution `"+str(distribution)+"` and `"+str(measure)+"` are not compatible")
    elif distribution.lower() == 'binomial':
        if measure not in ['mean']:
            raise ValueError("The distribution `"+str(distribution)+"` and `"+str(measure)+"` are not compatible")
    elif distribution.lower() == 'threshold':
        # if re.search(r"^t[0-9]+$", measure) is None:
        if re.search(r"^t\d", measure) is None and re.search(r"^tp\d", measure) is None:
            raise ValueError("The distribution `"+str(distribution)+"` and `"+str(measure)+"` are not compatible")
    else:
        raise ValueError("The distribution `"+str(distribution)+"` is not currently implemented")



######################## NetworkTMLE.__init__() ########################
# Checking for some common problems that should provide errors
if not all([isinstance(x, int) for x in list(network.nodes())]):   # Check if all node IDs are integers
    raise ValueError("NetworkTMLE requires that "                  # ... possibly not needed?
                        "all node IDs must be integers")

if nx.number_of_selfloops(network) > 0:                            # Check for any self-loops in the network
    raise ValueError("NetworkTMLE does not support networks "      # ... self-loops don't make sense in this
                        "with self-loops")                            # ... setting

# Checking for a specified degree restriction
if degree_restrict is not None:                                    # not-None means apply a restriction
    _check_degree_restrictions_(bounds=degree_restrict)       # ... checks if valid degree restriction
    _max_degree_ = degree_restrict[1]                         # ... extract max degree as upper bound
else:                                                              # otherwise if no restriction(s)
    if nx.is_directed(network):                                    # ... directed max degree is max out-degree
        _max_degree_ = np.max([d for n, d in network.out_degree])
    else:                                                          # ... undirected max degree is max degree
        _max_degree_ = np.max([d for n, d in network.degree])

# Generate a fresh copy of the network with ascending node order
oid = "_original_id_"                                              # Name to save the original IDs
network = nx.convert_node_labels_to_integers(network,              # Copy of new network with new labels
                                             first_label=0,        # ... start at 0 for latent variance calc
                                             label_attribute=oid)  # ... saving the original ID labels

# Saving processed data copies
network = network                       # Network with correct re-labeling
exposure = exposure                     # Exposure column / attribute name
outcome = outcome                       # Outcome column / attribute name

# Background processing to convert network attribute data to pandas DataFrame
adj_matrix = nx.adjacency_matrix(network,   # Convert to adjacency matrix
                                 weight=None)    # TODO allow for weighted networks
df = network_to_df(network)                      # Convert node attributes to pandas DataFrame

# Error checking for exposure types
if not df[exposure].value_counts().index.isin([0, 1]).all():        # Only binary exposures allowed currently
    raise ValueError("NetworkTMLE only supports binary exposures "
                        "currently")

# Manage outcome data based on variable type
if df[outcome].dropna().value_counts().index.isin([0, 1]).all():    # Binary outcomes
    _continuous_outcome = False                                # ... mark as binary outcome
    _cb_ = 0.0                                                 # ... set continuous bound to be zero
    _continuous_min_ = 0.0                                     # ... saving binary min bound
    _continuous_max_ = 1.0                                     # ... saving binary max bound
else:                                                               # Continuous outcomes
    _continuous_outcome = True                                 # ... mark as continuous outcome
    _cb_ = continuous_bound                                    # ... save continuous bound value
    _continuous_min_ = np.min(df[outcome]) - _cb_         # ... determine min (with bound)
    _continuous_max_ = np.max(df[outcome]) + _cb_         # ... determine max (with bound)
    df[outcome] = tmle_unit_bounds(y=df[outcome],              # ... bound the outcomes to be (0,1)
                                    mini=_continuous_min_,
                                    maxi=_continuous_max_)

# Creating summary measure mappings for all variables in the network
summary_types = ['sum', 'mean', 'var', 'mean_dist', 'var_dist']           # Default summary measures available
handle_isolates = ['mean', 'var', 'mean_dist', 'var_dist']                # Whether isolates produce nan's
for v in [var for var in list(df.columns) if var not in [oid, outcome]]:  # All cols besides ID and outcome
    v_vector = np.asarray(df[v])                                          # ... extract array of column
    for summary_measure in summary_types:                                 # ... for each summary measure
        df[v+'_'+summary_measure] = fast_exp_map(adj_matrix,         # ... calculate corresponding measure
                                                 v_vector,
                                                 measure=summary_measure)
        if summary_measure in handle_isolates:                            # ... set isolates from nan to 0
            df[v+'_'+summary_measure] = df[v+'_'+summary_measure].fillna(0)

# Creating summary measure mappings for non-parametric exposure_map_model()
exp_map_cols = exp_map_individual(network=network,               # Generate columns of indicator
                                  variable=exposure,             # ... for the exposure
                                  max_degree=_max_degree_)  # ... up to the maximum degree
_nonparam_cols_ = list(exp_map_cols.columns)                # Save column list for estimation procedure
df = pd.merge(df,                                                # Merge these columns into main data
              exp_map_cols.fillna(0),                            # set nan to 0 to keep same dimension across i
              how='left', left_index=True, right_index=True)     # Merge on index to left

# Calculating degree for all the nodes
if nx.is_directed(network):                                         # For directed networks...
    degree_data = pd.DataFrame.from_dict(dict(network.out_degree),  # ... use the out-degree
                                            orient='index').rename(columns={0: 'degree'})
else:                                                               # For undirected networks...
    degree_data = pd.DataFrame.from_dict(dict(network.degree),      # ... use the regular degree
                                            orient='index').rename(columns={0: 'degree'})
df = pd.merge(df,                                              # Merge main data
                    degree_data,                                     # ...with degree data
                    how='left', left_index=True, right_index=True)   # ...based on index

# Apply degree restriction to data
if degree_restrict is not None:                                     # If restriction provided,
    df['__degree_flag__'] = _degree_restrictions_(degree_dist=df['degree'],
                                                            bounds=degree_restrict)
    _exclude_ids_degree_ = np.asarray(df.loc[df['__degree_flag__'] == 1].index)
else:                                                               # Else all observations are used
    df['__degree_flag__'] = 0                                  # Mark all as zeroes
    _exclude_ids_degree_ = None                                # No excluded IDs

# Marking data set restricted by degree (same as df if no restriction)
df_restricted = df.loc[df['__degree_flag__'] == 0].copy()

# Output attributes
marginals_vector, marginal_outcome = None, None
conditional_variance, conditional_latent_variance = None, None
conditional_ci, conditional_latent_ci = None, None
alpha = alpha

# Storage for items for estimation procedures
_outcome_model, _q_model, _Qinit_ = None, None, None

_treatment_models = []
_gi_model, _gs_model = None, None
_gs_measure_, _map_dist_ = None, None
_exposure_measure_ = None
_denominator_, _denominator_estimated_ = None, False

# Threshold or category processing
_thresholds_, _thresholds_variables_, _thresholds_def_ = [], [], []
_thresholds_any_ = False
_categorical_, _categorical_variables_, _categorical_def_ = [], [], []
_categorical_any_ = False

# Custom model / machine learner storage
_gi_custom_, _gi_custom_sim_ = None, None
_gs_custom_, _gs_custom_sim_ = None, None
_q_custom_ = None

# Storage items for summary formatting
_specified_p_, _specified_bound_, _resamples_ = None, None, None
_verbose_ = verbose


######################## NetworkTMLE.exposure_model() ########################
# params
model = "L + A_30 + R_1 + R_2 + R_3"
custom_model = None
custom_model_sim = None

# code
# Clearing memory of previous / old models
_gi_model = model                       # Exposure model for A_i
_treatment_models = []                  # Clearing out possible old stored models
_denominator_estimated_ = False         # Mark the denominator as not having been estimated yet

# Storing user-specified model
_gi_custom_ = custom_model              # Custom model fitter being stored
if custom_model_sim is None:                 # Custom model fitter for simulated data
    _gi_custom_sim_ = custom_model      # ... same as actual data if not specified
else:                                        # Otherwise
    _gi_custom_sim_ = custom_model_sim  # ... use specified model


######################## NetworkTMLE.exposure_map_model() ########################
# params
model = "statin + L + A_30 + R_1 + R_2 + R_3"
measure='sum'
distribution='poisson'
custom_model=None
custom_model_sim=None

# code
# Checking that distribution and measure are compatible
_check_distribution_measure_(distribution=distribution,   # Check distribution for model
                             measure=measure)             # ... and specified measure

# Clearing memory of previous / old models
_gs_model = model                                  # Exposure model for A_i^s
_treatment_models = []                             # Clearing out possible old stored models
_denominator_estimated_ = False                    # Mark denominator as not having been estimated yet

# Getting distribution for parametric models. Ignored if custom_model is not None
if distribution is None:                                # If None is the distribution
    _map_dist_ = distribution                      # ... keeping as None for later logic
else:                                                   # Otherwise
    _map_dist_ = distribution.lower()              # ... making lower-case to avoid processing errors

if measure is not None:                                 # If the specified measure is not NoneType
    _exposure_measure_ = measure                   # ... save the measures name
    _gs_measure_ = exposure + '_' + measure   # ... pick out relevant column

# Storing user-specified model(s)
if custom_model is not None:                            # Logic if provided custom_model
    if distribution in ["poisson", "threshold"]:        # ...need that distribution to be Poisson or Berno
        pass                                            # ...good to move on
    elif distribution is None:                          # if distribution is not specified
        # TODO see implementation in TL for Data Science, Chapter 14.4
        raise ValueError("...generalized conditional "  # ... raise ValueError since not implemented yet
                            "distribution super-learner "
                            "to be added...")
    else:                                               # Otherwise mark as incompatible options currently
        raise ValueError("Incompatible `distribution` for implemented "
                            "`custom_model` forms. Select from: "
                            "poisson, threshold")
_gs_custom_ = custom_model                         # Saving custom model for later fitting
if custom_model_sim is None:                            # Checking if another model for the simulations is given
    _gs_custom_sim_ = custom_model                 # ... if not, use previous
else:                                                   # Otherwise
    # TODO this should have some error-checking
    _gs_custom_sim_ = custom_model_sim             # ... store alternative model

######################## NetworkTMLE.outcome_model() ########################
# params
model="statin + statin_sum + A_sqrt + R + L"
custom_model=None
distribution='normal'    


# code
# Storing model specification
_q_model = model

# Running through logic for custom models
if custom_model is None:                                           # If no custom model
    if not _continuous_outcome:                               # and not continuous
        f = sm.families.family.Binomial()                          # ... use logit regression
    elif distribution.lower() == 'normal':                         # or use distribution normal
        f = sm.families.family.Gaussian()                          # ... use OLS
    elif distribution.lower() == 'poisson':                        # or use distribution poisson
        f = sm.families.family.Poisson()                           # ... use Poisson regression
    else:                                                          # Otherwise error for not available
        raise ValueError("Distribution" +
                            str(distribution) +
                            " is not currently supported")

    # Estimate outcome model and predicted Y with the observed network data
    _outcome_model = smf.glm(outcome + ' ~ ' + _q_model,   # Specified model form
                             df_restricted,                     # ... fit to restricted data
                             family=f).fit()                         # ... for given GLM family
    _Qinit_ = _outcome_model.predict(df_restricted)        # Predict outcome values

    # If verbose is requested, output the relevant information
    if _verbose_:
        print('==============================================================================')
        print('Outcome model')
        print(_outcome_model.summary())

# Logic if custom_model is provided
else:
    # Extract data using the model
    data = patsy.dmatrix(model + ' - 1',                      # Specified model WITHOUT an intercept
                         df_restricted)                  # ... using the degree restricted data

    # Estimating custom_model
    _q_custom_ = outcome_learner_fitting(ml_model=custom_model,       # User-specified model
                                         xdata=np.asarray(data),      # Extracted X data
                                         ydata=np.asarray(df_restricted[outcome]))

    # Generating predictions
    _Qinit_ = outcome_learner_predict(ml_model_fit=_q_custom_,   # Fit custom_model
                                      data=np.asarray(data))         # Observed X data

# Ensures all predicted values are bounded
if _continuous_outcome:
    _Qinit_ = np.where(_Qinit_ < _continuous_min_,          # When lower than lower bound
                            _continuous_min_,                         # ... set to lower bound
                            _Qinit_)                                  # ... otherwise keep
    _Qinit_ = np.where(_Qinit_ > _continuous_max_,          # When above the upper bound
                            _continuous_max_,                         # ... set to upper bound
                            _Qinit_)         


######################## NetworkTMLE.fit() ########################
# params
p=0.35
samples=100
bound=0.01
seed=None


# code
# Error checking for function order called correctly
if _gi_model is None:                                               # A_i model must be specified
    raise ValueError("exposure_model() must be specified before fit()")
if _gs_model is None:                                               # A_i^s model must be specified
    raise ValueError("exposure_map_model() must be specified before fit()")
if _q_model is None:                                                # Y model must be specified
    raise ValueError("outcome_model() must be specified before fit()")

# Error checking for policies
if type(p) is int:                                                       # Check if an integer is provided
    raise ValueError("Input `p` must be float or container of floats")

if type(p) != float:                                                     # Check if not a float
    if len(p) != df.shape[0]:                                       # ... check length matches data shape
        raise ValueError("Vector of `p` must be same length as input data")
    if np.all(np.asarray(p) == 0) or np.all(np.asarray(p) == 1):         # ... check if deterministic plan
        raise ValueError("Deterministic treatment plans not supported")
    if np.any(np.asarray(p) < 0) or np.any(np.asarray(p) > 1):           # ... check if outside of prob bounds
        raise ValueError("Probabilities for treatment must be between 0 and 1")
else:                                                                    # If it is a float
    if p == 0 or p == 1:                                                 # ... check if deterministic plan
        raise ValueError("Deterministic treatment plans not supported")
    if p < 0 or p > 1:                                                   # ... check if outside of prob bounds
        raise ValueError("Probabilities for treatment must be between 0 and 1")

# Step 1) Estimate the weights
# Also generates pooled_data for use in the Monte Carlo integration procedure
_resamples_ = samples                                                # Saving info on number of resamples
h_iptw, pooled_data_restricted = _estimate_iptw_(p=p,                # Generate pooled & estiamte weights
                                                 samples=samples,    # ... for some number of samples
                                                 bound=bound,        # ... with applied probability bounds
                                                 seed=seed,          # ... and with a random seed given
                                                 _denominator_estimated_=_denominator_estimated_,  _denominator_=_denominator_, df_restricted=df_restricted, _map_dist_=_map_dist_,
                                                 _gi_custom_=_gi_custom_, _gi_model=_gi_model, _gs_custom_=_gs_custom_, _gs_model=_gs_model, _gs_measure_=_gs_measure_, _nonparam_cols_=_nonparam_cols_,
                                                 exposure=exposure, _verbose_=_verbose_, _treatment_models=_treatment_models,
                                                 df=df, adj_matrix=adj_matrix, network=network, _max_degree_=_max_degree_,
                                                 _thresholds_any_=_thresholds_any_, _thresholds_variables_=_thresholds_variables_, _thresholds_=_thresholds_, _thresholds_def_=_thresholds_def_,
                                                 _categorical_any_=_categorical_any_, _categorical_variables_=_categorical_variables_, _categorical_=_categorical_, _categorical_def_=_categorical_def_)          

# Saving some information for diagnostic procedures
if _gs_measure_ is None:                                  # If no summary measure, use the A_sum
    _for_diagnostics_ = pooled_data_restricted[[exposure, exposure+"_sum"]].copy()
else:                                                     # Otherwise, use the provided summary measure
    _for_diagnostics_ = pooled_data_restricted[[exposure, _gs_measure_]].copy()

# Step 2) Estimate from Q-model
# process completed in .outcome_model() function and stored in self._Qinit_
# so nothing to do here

# Step 3) Target the parameter
epsilon = targeting_step(y=df_restricted[outcome],   # Estimate the targeting model given observed Y
                         q_init=_Qinit_,                  # ... predicted values of Y under observed A
                         ipw=h_iptw,                           # ... weighted by IPW
                         verbose=_verbose_)               # ... with option for verbose info

# Step 4) Monte Carlo integration (old code did in loop but faster as vector)
#
# Generating outcome predictions under the policies (via pooled data sets)
if _q_custom_ is None:                                            # If given a parametric default model
    y_star = _outcome_model.predict(pooled_data_restricted)       # ... predict using statsmodels syntax
else:  # Custom input model by user
    d = patsy.dmatrix(_q_model + ' - 1', pooled_data_restricted)  # ... extract data via patsy
    y_star = outcome_learner_predict(ml_model_fit=_q_custom_,     # ... predict using custom function
                                        xdata=np.asarray(d))      # ... for the extracted data

# Ensure all predicted values are bounded properly for continuous
if _continuous_outcome:
    y_star = np.where(y_star < _continuous_min_, _continuous_min_, y_star)
    y_star = np.where(y_star > _continuous_max_, _continuous_max_, y_star)

# Updating predictions via intercept from targeting step
logit_qstar = np.log(probability_to_odds(y_star)) + epsilon            # NOTE: needs to be logit(Y^*) + e
q_star = odds_to_probability(np.exp(logit_qstar))                      # Back converting from odds
pooled_data_restricted['__pred_q_star__'] = q_star                     # Storing predictions as column

# Taking the mean, grouped-by the pooled sample IDs (fast)
marginals_vector = np.asarray(pooled_data_restricted.groupby('_sample_id_')['__pred_q_star__'].mean())

# If continuous outcome, need to unbound the means
if _continuous_outcome:
    marginals_vector = tmle_unit_unbound(marginals_vector,          # Take vector of MC results
                                         mini=_continuous_min_,     # ... unbound using min
                                         maxi=_continuous_max_)     # ... and max values

# Calculating estimate for the policy
marginal_outcome = np.mean(marginals_vector)                        # Mean of Monte Carlo results
_specified_p_ = p                                                   # Save what the policy was

# Prep for variance
if _continuous_outcome:                                                 # Continuous needs bounds...
    y_ = np.array(tmle_unit_unbound(df_restricted[outcome],        # Unbound observed outcomes for Var
                                    mini=_continuous_min_,              # ... using min
                                    maxi=_continuous_max_))             # ... and max values
    yq0_ = tmle_unit_unbound(_Qinit_,                                   # Unbound g-comp predictions
                             mini=_continuous_min_,                     # ... using min
                             maxi=_continuous_max_)                     # ... and max values
else:                                                                        # Otherwise nothing special...
    y_ = np.array(df_restricted[outcome])                          # Observed outcome for Var
    yq0_ = _Qinit_                                                      # Predicted outcome for Var

# Step 5) Variance estimation
zalpha = norm.ppf(1 - alpha / 2, loc=0, scale=1)      # Get corresponding Z-value based on desired alpha

# Variance: direct-only, conditional on W variance
var_cond = _est_variance_conditional_(iptw=h_iptw,                  # Estimate direct-only variance
                                      obs_y=y_,                     # ... observed value of Y
                                      pred_y=yq0_)                  # ... predicted value of Y
conditional_variance = var_cond                                     # Store the var estimate and CIs
conditional_ci = [marginal_outcome - zalpha*np.sqrt(var_cond),
                  marginal_outcome + zalpha*np.sqrt(var_cond)]

# Variance: direct and latent, conditional on W variance
var_lcond = _est_variance_latent_conditional_(iptw=h_iptw,          # Estimate latent variance
                                              obs_y=y_,             # ... observed value of Y
                                              pred_y=yq0_,          # ... predicted value of Y
                                              adj_matrix=adj_matrix,
                                              excluded_ids=_exclude_ids_degree_)
conditional_latent_variance = var_lcond                             # Store variance estimate and CIs
conditional_latent_ci = [marginal_outcome - zalpha*np.sqrt(var_lcond),
                         marginal_outcome + zalpha*np.sqrt(var_lcond)]


######################## NetworkTMLE.summary() ########################
# params
decimal=3


# code
# Check to make sure there is an answer to actually report
if marginal_outcome is None:
    raise ValueError('The fit() statement must be ran before summary()')

# Summary information
print('======================================================================')
print('            Network Targeted Maximum Likelihood Estimator             ')
print('======================================================================')
fmt = 'Treatment:        {:<15} No. Observations:     {:<20}'
print(fmt.format(exposure, df_restricted.shape[0]))
fmt = 'Outcome:          {:<15} No. Background Nodes: {:<20}'
print(fmt.format(outcome, df.shape[0] - df_restricted.shape[0]))
fmt = 'Q-Model:          {:<15} No. IPW Truncated:    {:<20}'
if _specified_bound_ is None:
    b = 0
else:
    b = _specified_bound_
if _q_custom_ is None:
    qm = 'Logistic'
else:
    qm = 'Custom'
print(fmt.format(qm, b))

fmt = 'g-Model:          {:<15} No. Resamples:        {:<20}'
if _gi_custom_ is None:
    gim = 'Logistic'
else:
    gim = 'Custom'
print(fmt.format(gim, _resamples_))

fmt = 'gs-Model:         {:<15} g-Distribution:       {:<20}'
if _gs_custom_ is None:
    if _map_dist_ is None:
        gsm = 'Logitistic'
    else:
        gsm = _map_dist_.capitalize()
else:
    gsm = 'Custom'
if _map_dist_ is None:
    gs = 'Nonparametric'
else:
    gs = _map_dist_.capitalize()
print(fmt.format(gsm, gs))

print('======================================================================')
print('Mean under policy:      ', np.round(marginal_outcome, decimals=decimal))
print('----------------------------------------------------------------------')
print('Variance Estimates')
print('----------------------------------------------------------------------')
print('Conditional: Direct-Only')
print("SE      :     ", np.round(conditional_variance**0.5, decimals=decimal))
print(str(round(100 * (1 - alpha), 0)) + '% CL:    ',
        np.round(conditional_ci, decimals=decimal))
print('Conditional: Direct & Latent')
print("SE      :     ", np.round(conditional_latent_variance**0.5, decimals=decimal))
print(str(round(100 * (1 - alpha), 0)) + '% CL:    ',
        np.round(conditional_latent_ci, decimals=decimal))
print('======================================================================')

######################### tmle.define_category() #########################
def define_category(variable, bins, labels=False):
    """Function arbitrarily allows for multiple different defined thresholds

    Parameters
    ----------
    variable : str
        Variable to generate categories for
    bins : list, set, array
        Bin cutpoints to generate the categorical variable for. Uses ``pandas.cut(..., include_lowest=True)`` to
        create the binned variables.
    labels : list, set, array
        Specified labels. Can be given custom labels, but generally recommend to keep set as False
    """
    _categorical_any_ = True                   # Update logic to understand at least one category exists
    _categorical_variables_.append(variable)   # Add the variable to the list of category-generations
    _categorical_.append(bins)                 # Add the cut-points for the bins to the list of bins
    _categorical_def_.append(labels)           # Add the specified labels for the bins to the label list
    create_categorical(data=df_restricted,     # Create the desired category variable
                       variables=[variable],        # ... for the specified variable
                       bins=[bins],                 # ... for the specified bins
                       labels=[labels],             # ... with the specified labels
                       verbose=True)                # ... warns user if NaN's are being generated

define_category(variable='R_1_sum', bins=[0, 1, 5], labels=False)

# for v, b, l in zip(['R_1_sum'], [[0, 1, 5]], [False]):
#     print(v, b, l)

# oid = "_original_id_"                                              # Name to save the original IDs
# network = nx.convert_node_labels_to_integers(H,              # Copy of new network with new labels
#                                              first_label=0,        # ... start at 0 for latent variance calc
#                                              label_attribute=oid)  # ... saving the original ID labels
# tmp = network_to_df(network)


######################## ml_funtion fit ########################
data_to_fit=df_restricted.copy()
data_to_predict=df_restricted.copy()

xdata = patsy.dmatrix(_gi_model + ' - 1', data_to_fit)       # Extract via patsy the data
pdata = patsy.dmatrix(_gi_model + ' - 1', data_to_predict)   # Extract via patsy the data

xdata = np.asarray(xdata)                                   # Convert to numpy array
ydata = np.asarray(data_to_fit[exposure])                   # Extract the exposure data
pdata = np.asarray(pdata)                                   # Convert to numpy array


pred = exposure_machine_learner(ml_model=_gi_custom_,        # Custom model application and preds
                                xdata=np.asarray(xdata),          # ... with data to fit
                                ydata=np.asarray(data_to_fit[exposure]),
                                pdata=np.asarray(pdata))          # ... and data to predict

######################## define ml_funtion ########################
import torch
import torch.nn as nn
import torch.optim as optim

class AbstractML:
    def __init__(self, df, model_string, exposure, 
                 epochs, print_every, device='cpu', save_path='./'):
        self.epochs = epochs
        self.best_model = None
        self.best_loss = np.inf
        self.save_path = save_path

        self.df, self.model_string, self.exposure = df, model_string, exposure
    
        self.print_every = print_every
        self.device = device

        self.dataset = self._data_preprocess(df, model_string, exposure, fit=True, return_dataset=True)
        self.model = self._build_model().to(self.device)
        self.optimizer = self._optimizer()
        self.criterion = self._loss_fn()

    def fit(self, split_ratio, batch_size, shuffle):
        self.train_loader, self.valid_loader, self.test_loader = self._data_preprocess(self.df, self.model_string, self.exposure, 
                                                                                       fit=True, return_dataset=False,
                                                                                       split_ratio=split_ratio, 
                                                                                       batch_size=batch_size, 
                                                                                       shuffle=shuffle)
        for epoch in range(self.epochs):
            print(f'============================= Epoch {epoch + 1}: Training =============================')
            loss_train, metrics_train = self.train_epoch(epoch)
            print(f'============================= Epoch {epoch + 1}: Validation =============================')
            loss_valid, metrics_valid = self.valid_epoch(epoch)
            print(f'============================= Epoch {epoch + 1}: Testing =============================')
            loss_test, metrics_test = self.test_epoch(epoch, return_pred=False)

            # update best loss
            if loss_valid < self.best_loss:
                self._save_model()
                self.best_loss = loss_valid
                self.best_model = self.model
            
        return self.best_model

    def predict(self, split_ratio, batch_size, shuffle):
        _, _, self.test_loader = self._data_preprocess(self.df, self.model_string, self.exposure, 
                                                       fit=False, return_dataset=False,
                                                       split_ratio=split_ratio, 
                                                       batch_size=batch_size, 
                                                       shuffle=shuffle)
        self._load_model()
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
                output = self.model(x_cat, x_cont)
                if return_pred:
                    pred_list.append(torch.sigmoid(output).detach().to('cpu').numpy())

                loss = self.criterion(output, y)
                metrics = self._metrics(output, y)

                # print statistics
                cumu_loss += loss.item()
                cumu_metrics += metrics

            if not return_pred: # real label not available for predict()
                print(f'[{epoch + 1}, {i + 1:5d}] | loss: {cumu_loss / len(self.test_loader):.3f} | acc: {cumu_metrics / len(self.test_loader):.3f}')
                # for metric_name, metric_value in cumu_metrics.items():
                #     print(f'[{epoch + 1}, {i + 1:5d}] {metric_name}: {metric_value / len(self.test_loader):.3f}')

        if return_pred:
            return pred_list
        else:
            return cumu_loss / len(self.test_loader), cumu_metrics / len(self.test_loader)

    def _build_model(self):
        pass 

    def _data_preprocess(self, df, model_string, exposure=None, fit=True, return_dataset=False,
                         split_ratio=[0.6, 0.2, 0.2], batch_size=2, shuffle=True):
        dset = DfDataset(df, model_string, exposure=exposure, fit=fit)

        if return_dataset:
            return dset
        else:
            train_loader, valid_loader, test_loader = get_dataloaders(dset, fit=fit,
                                                                      split_ratio=split_ratio, 
                                                                      batch_size=batch_size,
                                                                      shuffle=shuffle)           
            return train_loader, valid_loader, test_loader


    def _optimizer(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # return optim.Adam(self.model.parameters(), lr=0.001)

    def _loss_fn(self):
        return nn.BCEWithLogitsLoss() # no need for sigmoid, require 2 output for binary classfication
        # return nn.CrossEntropyLoss() # no need for softmax, require 2 output for binary classification

    def _metrics(self, outputs, labels):
        pred = torch.sigmoid(outputs) # get binary probability
        pred_binary = torch.round(pred) # get binary prediction
        return (pred_binary == labels).sum().item()/labels.shape[0] # num samples correctly classified / num_samples
        
        # # the class with the highest energy is what we choose as prediction, if output 2 categories for binary classificaiton
        # _, predicted = torch.max(outputs.data, 1)
        # return (predicted == labels).sum().item()

    def _save_model(self):
        torch.save(self.model.state_dict(), self.save_path)

    def _load_model(self):
        self.model.load_state_dict(torch.load(self.save_path))
    
    # @staticmethod
    # def _update_metrics_dict(cumu_metrics, metrics):
    #     pass
    
    # @staticmethod
    # def _dict_division(cumu_metrics, denominator):
    #     pass



abs_ml = AbstractML()

hasattr(abs_ml, 'predict_proba')
hasattr(abs_ml, 'predict')


######################## ml training ########################
class MLP(AbstractML):
    def __init__(self, df, model_string, exposure, epochs, print_every, device='cpu', save_path='./'):
        super().__init__(df, model_string, exposure, epochs, print_every, device, save_path)

    def _build_model(self):
        return SimpleModel(self.dataset)
    
mlp_learner = MLP(df_restricted, _gi_model, exposure, epochs=10, print_every=50, device='cpu', save_path='./tmp.pth')
mlp_learner.fit(split_ratio=[0.6, 0.2, 0.2], batch_size=2, shuffle=True) 
pred = mlp_learner.predict(split_ratio=[0.6, 0.2, 0.2], batch_size=2, shuffle=False)


################################# ml: use nn.embedding #################################
# Ref: https://jovian.ml/aakanksha-ns/shelter-outcome

# params
data_to_fit=df_restricted.copy()
data_to_predict=df_restricted.copy()
distribution=_map_dist_
verbose_label='Weight - Denominator'
store_model=True

# code
# Model: A_i
xdata = patsy.dmatrix(_gi_model + ' - 1', data_to_fit)       # Extract via patsy the data
pdata = patsy.dmatrix(_gi_model + ' - 1', data_to_predict)   # Extract via patsy the data

xdata = np.asarray(xdata)                                   # Convert to numpy array
ydata = np.asarray(data_to_fit[exposure])
pdata = np.asarray(pdata)                                   # Convert to numpy array

# pred = exposure_machine_learner(ml_model=_gi_custom_,        # Custom model application and preds
#                                 xdata=np.asarray(xdata),          # ... with data to fit
#                                 ydata=np.asarray(data_to_fit[exposure]),
#                                 pdata=np.asarray(pdata))          # ... and data to predict

# Model: A_i^s
# else:                                                           # Custom model for Poisson
xdata = patsy.dmatrix(_gs_model + ' - 1',              # ... extract data given relevant model
                        data_to_fit)                          # ... from degree restricted
pdata = patsy.dmatrix(_gs_model + ' - 1',              # ... extract data given relevant model
                        data_to_predict)                      # ... from degree restricted

xdata = np.asarray(xdata)                                   # Convert to numpy array
ydata = np.asarray(data_to_fit[_gs_measure_])               # Extract the exposure data
pdata = np.asarray(pdata)                                   # Convert to numpy array

# pred = exposure_machine_learner(ml_model=_gs_custom_,  # Custom ML model
#                                 xdata=np.asarray(xdata),    # ... with data to fit
#                                 ydata=np.asarray(data_to_fit[_gs_measure_]),
#                                 pdata=np.asarray(pdata))    # ... and data to predict


# determine cat and cont variables
def split_data(model, df):
    vars = model.split(' + ')

    cat_vars = {}
    cat_vars_index = []
    cont_vars_index = []
    for id, var in enumerate(vars):
        if var in df.columns:
            if df[var].dtype == 'int64':
                if len(pd.unique(df[var])) == 2:
                    cat_vars[var] = len(pd.unique(df[var])) # record number of levels
                else:
                    cat_vars[var] = df['A'].max() + 1 # record number of levels for 'A_30', temporary strategy
                cat_vars_index.append(id)
            else:
                cont_vars_index.append(id)
    return cat_vars, cat_vars_index, cont_vars_index

cat_vars, cat_vars_index, cont_vars_index = split_data(_gi_model, df_restricted)
cat_vars


#categorical embedding for categorical columns
embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _, n_categories in cat_vars.items()]
embedding_sizes

xdata[:, cat_vars_index].astype(np.int64)
xdata[:, cont_vars_index].shape
len(ydata)

import torch
from torch.utils.data import Dataset, DataLoader
class CatContDataset(Dataset):
    def __init__(self, xdata, ydata=None, cat_vars_index=None, cont_vars_index=None):
        if ydata is not None:
            self.x_cat = xdata[:, cat_vars_index]
            self.x_cont = xdata[:, cont_vars_index]
            self.y = ydata[:, np.newaxis] #[num_samples, ] -> [num_samples, 1]
        else:
            self.x_cat = xdata[:, cat_vars_index]
            self.x_cont = xdata[:, cont_vars_index]
            self.y = np.empty(xdata.shape[0]).fill(-1) # create dummy target for pdata
    
    def __getitem__(self, idx):
        # return torch.from_numpy(np.asarray(self.y[idx]))
        return torch.from_numpy(self.x_cat[idx]), torch.from_numpy(self.x_cont[idx]), torch.from_numpy(self.y[idx])

    def __len__(self):
        return self.y.shape[0]
    
dset = CatContDataset(xdata, ydata, cat_vars_index, cont_vars_index)
aa = dset.__getitem__(0)
aa, bb, cc = dset.__getitem__(0)
aa
bb.shape
cc.shape

class DfDataset(Dataset):
    def __init__(self, df, model, exposure=None, fit=True):
        ''' Retrieve train,label and pred data from Dataframe directly
        Args:  
            df: pd.DataFrame, data, i.e., df_restricted
            model: str, model formula, i.e., _gi_model
            exposure: str, exposure variable, i.e., exposure
            fit: bool, whether the dataset is for fitting or prediction

        if fit is set to true, df should be data_to_fit; else, df should be data_to_predict
        '''
        self.df = df
        self.model = model
        self.exposure = exposure
        self.fit = fit

        self.x_cat, self.x_cont, self.cat_unique_levels = self._split_cat_cont() 
        if self.fit:
            self.y = self._get_labels()
        else:
            self.y = np.empty((self.x_cat.shape[0], 1))
            self.y.fill(-1) # create dummy target for pdata
    
    def _split_cat_cont(self):
        # get variables from model string
        vars = self.model.split(' + ')

        cat_vars = []
        cont_vars = []
        cat_unique_levels = {}
        for var in vars:
            if var in self.df.columns:
                if self.df[var].dtype == 'int64':
                    
                    cat_vars.append(var)
                    if len(pd.unique(df[var])) == 2:
                        cat_unique_levels[var] = len(pd.unique(self.df[var])) # record number of levels
                    else:
                        cat_unique_levels[var] = self.df['A'].max() + 1 # record number of levels for 'A_30', temporary strategy
                else:
                    cont_vars.append(var)
        return df[cat_vars].to_numpy(), df[cont_vars].to_numpy(), cat_unique_levels
    
    def _get_labels(self):
        return np.asarray(self.df[self.exposure])[:, np.newaxis] #[num_samples, ] -> [num_samples, 1] 
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.x_cat[idx]).int(), torch.from_numpy(self.x_cont[idx]).float(), torch.from_numpy(self.y[idx]).float()
        # shape: [num_cat_vars], [num_cont_vars], [1]

    def __len__(self):
        return self.y.shape[0]


dset = DfDataset(df_restricted, _gi_model, exposure=exposure, fit=True)
dset = DfDataset(df_restricted, _gi_model, exposure=exposure, fit=False)
aa, bb, cc = dset.__getitem__(0)
aa.shape
bb.shape
cc.shape

# split dataset
def get_dataloaders(dataset, fit=True, split_ratio=[0.7, 0.1, 0.2], batch_size=16, shuffle=True):
    torch.manual_seed(17) # random split with reproducibility

    if fit:
        train_size = int(split_ratio[0] * len(dataset))
        test_size = int(split_ratio[-1] * len(dataset))
        valid_size = len(dataset) - train_size - test_size
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        return train_loader, valid_loader, test_loader
    else:
        return None, None, DataLoader(dataset, batch_size=batch_size, shuffle=False)

train_loader, valid_loader, test_loader = get_dataloaders(dset, fit=True, batch_size=2)
_, _, test_loader = get_dataloaders(dset, fit=False, batch_size=2)


import torch.nn.functional as F
class SimpleModel(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.embedding_layers, self.n_emb, self.n_cont = self._get_embedding_layers(dataset)
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 16)
        self.lin2 = nn.Linear(16, 32)
        self.lin3 = nn.Linear(32, 1) # use BCEloss, so output 1
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

    def _get_embedding_layers(self, dataset):
        # Ref: https://jovian.ml/aakanksha-ns/shelter-outcome
        # decide embedding sizes
        embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _, n_categories in dataset.cat_unique_levels.items()]
        embedding_layers = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in embedding_layers) # length of all embeddings combined
        n_cont = dataset.x_cont.shape[1] # number of continuous variables

        return embedding_layers, n_emb, n_cont
    
    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embedding_layers)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)
        return x

test_model = SimpleModel(dset)


for i, (x_cat, x_cont, y) in enumerate(train_loader):
    # print(x_cat.shape)
    # print(x_cont.shape)
    # print(y.shape)
    out = test_model(x_cat, x_cont)

pred = torch.sigmoid(out)
pred_binary = torch.round(pred)
y

###########################################
# Naloxone-Overdose -- DGM

# Loading clustered power-law network with naloxone W
G = load_random_naloxone()
# Simulation single instance of exposure and outcome
H = naloxone_dgm(network=G)

# network-TMLE applies to generated data
tmle = NetworkTMLE(H, exposure='naloxone', outcome='overdose',
                   degree_restrict=(0, 18))  # Applying restriction by degree
tmle.exposure_model("P + P:G + O_mean + G_mean")
tmle.exposure_map_model("naloxone + P + P:G + O_mean + G_mean",
                        measure='sum', distribution='poisson')  # Applying a Poisson model
tmle.outcome_model("naloxone_sum + P + G + O_mean + G_mean")
tmle.fit(p=0.35, bound=0.01)
tmle.summary()

###########################################
# Diet-BMI -- DGM

# Loading clustered power-law network with naloxone W
G = load_uniform_diet()
# Simulation single instance of exposure and outcome
H = diet_dgm(network=G)

# network-TMLE applies to generated data
tmle = NetworkTMLE(H, exposure='diet', outcome='bmi')
tmle.define_threshold(variable='diet', threshold=3,
                      definition='sum')  # Defining threshold measure of at least 3 contacts with a diet
tmle.exposure_model("B_30 + G:E + E_mean")
tmle.exposure_map_model("diet + B_30 + G:E + E_mean", measure='t3',
                        distribution='threshold')  # Logistic model for the threshold summary measure
tmle.outcome_model("diet + diet_t3 + B + G + E + E_sum + B_mean_dist")
tmle.fit(p=0.65, bound=0.01)
tmle.summary()

###########################################
# Vaccine-Infection -- DGM

# Loading clustered power-law network with naloxone W
G = load_random_vaccine()
# Simulation single instance of exposure and outcome
H = vaccine_dgm(network=G)

tmle = NetworkTMLE(H, exposure='vaccine', outcome='D', verbose=False, degree_restrict=(0, 18))
tmle.exposure_model("A + H + H_mean + degree")
tmle.exposure_map_model("vaccine + A + H + H_mean + degree",
                        measure='sum', distribution='poisson')
tmle.outcome_model("vaccine + vaccine_mean + A + H + A_mean + H_mean + degree")
tmle.fit(p=0.55, bound=0.01)
tmle.summary()
