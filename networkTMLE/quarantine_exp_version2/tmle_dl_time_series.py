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
import math
from scipy.stats import logistic

from tmle_utils import (network_to_df, fast_exp_map, exp_map_individual, tmle_unit_bounds, tmle_unit_unbound,
                        probability_to_odds, odds_to_probability, bounding,
                        outcome_learner_fitting, outcome_learner_predict, exposure_machine_learner, 
                        exposure_deep_learner_ts, outcome_deep_learner_ts,
                        targeting_step, create_threshold, create_categorical,
                        check_pooled_sample_levels, select_pooled_sample_with_observed_data,
                        get_patsy_for_model_w_C, get_model_cat_cont_split_patsy_matrix, append_target_to_df, get_probability_from_multilevel_prediction)

import torch

class NetworkTMLETimeSeries:
    r"""Implementation of the Targeted Maximum Likelihood Estimator (TMLE) for network dependent data. The following
    procedure estimates the expected incidence under a treatment plan of interest. For stochastic treatment plans, the
    expected incidence is obtained through Monte Carlo integration of a subsample of possible treatment allotments that
    correspond to the plan of interest.

    Note
    ----
    Network-TMLE makes the weak dependence assumption, such that only direct contacts' treatment can interfere with
    individual i's outcome.

    Parameters
    ----------
    network : NetworkX Graph
        NetworkX undirected network *without* self-loops. Additionally, all variables should be stored as attributes
        for each node. ``Targetula`` extracts the node data from the graph and creates a ``pandas.DataFrame`` object
        from that information. It is important that no nodes have missing data. Currently there is no procedure to
        handle missing data
    exposure : str
        String indicating the exposure variable of interest.
    outcome : str
        String indicating the outcome variable of interest.
    degree_restrict : None, list, tuple, optional
        Restriction on the minimum & maximum degree for nodes to be included in the estimand. Must be a list with a
        length of two, where the first value corresponds to the lower bound and the second is the upper bound for
        degree. Values are inclusive. All samples below the first value OR above the second level are considered as
        "background" features. Hence the intervention does not change their exposure.
    alpha : float, optional
        Alpha for confidence interval level. Default is 0.05
    continuous_bound : float, optional
        For continuous outcomes, TMLE needs to bound Y between 0 and 1. However, 0/1 cannot be included in these
        bounded values. This specification sets the bounds for the continuous outcomes. The default is 0.0005.
    verbose : bool, optional
        Whether to print all intermediary model results for the estimation process. When set to True, each of the
        model results are printed to the console. The default is False.

    Note
    ----
    ``mossspider`` calculates exposure mapping variables automatically with the input network. These variables are
    saved as variable-name_map. So for a variable `'A'`, the newly created exposure mapping variable calculated is
    `'A_map'`

    Note
    ----
    For directed networks, the direction of of influence goes from the target node to the source (i.e. opposite of the
    arrow direction). If `A --> B` then B's covariates will be part of the A's summary measures.

    Examples
    --------
    Setting up environment

    >>> from mossspider import NetworkTMLE
    >>> from mossspider.dgm import uniform_network, generate_observed

    Generating a generic network and some data

    >>> graph = generate_observed(uniform_network(n=500, degree=[1, 6]))

    Estimation with `NetworkTMLE` (nonparametric summary measure in exposure map model)

    >>> tmle = NetworkTMLE(network=graph, exposure='A', outcome='Y')
    >>> tmle.exposure_model('W + W_map')
    >>> tmle.exposure_map_model('A + W + W_map', distribution=None)
    >>> tmle.outcome_model('A + W + A_map + W_map', print_results=False)
    >>> tmle.fit(p=0.8, bound=10e5)
    >>> tmle.summary()

    Estimation with `NetworkTMLE` (parametric summary measure in exposure map model)

    >>> tmle = NetworkTMLE(network=graph, exposure='A', outcome='Y')
    >>> tmle.exposure_model('W + W_map')
    >>> tmle.exposure_map_model('A + W + W_map', measure='sum', distribution='poisson')
    >>> tmle.outcome_model('A + W + A_map + W_map', print_results=False)
    >>> tmle.fit(p=0.8, bound=10e5)
    >>> tmle.summary()

    Estimation with `NetworkTMLE` and restricting inference by degree

    >>> tmle = NetworkTMLE(network=graph, exposure='A', outcome='Y', degree_restrict=[0, 5])
    >>> tmle.exposure_model('W + W_map')
    >>> tmle.exposure_map_model('A + W + W_map', measure='sum', distribution='poisson')
    >>> tmle.outcome_model('A + W + A_map + W_map', print_results=False)
    >>> tmle.fit(p=0.8, bound=10e5)
    >>> tmle.summary()

    Diagnostic plot for support of policy of interest in observed data

    >>> import matplotlib.pyplot as plt
    >>> tmle.diagnostics()
    >>> plt.show()

    Generating a threshold measure based on a summary measure

    >>> tmle = NetworkTMLE(network=graph, exposure='A', outcome='Y')
    >>> tmle.define_threshold(variable='A_sum', threshold=3)  # A_sum_t3

    Generating a category measure based on a binned summary measure

    >>> tmle = NetworkTMLE(network=graph, exposure='A', outcome='Y')
    >>> tmle.define_category(variable='A_sum', bins=[0, 1, 2, 4, 6])  # A_sum_c

    References
    ----------
    van der Laan MJ. (2014). Causal inference for a population of causally connected units.
    *Journal of Causal Inference*, 2(1), 13-74.

    Sofrygin O & van der Laan MJ. (2017). Semi-parametric estimation and inference for the mean outcome of
    the single time-point intervention in a causally connected population. *Journal of Causal Inference*, 5(1).

    Ogburn EL, Sofrygin O, Diaz I, & van der Laan MJ. (2017). Causal inference for social network data.
    *arXiv preprint* arXiv:1705.08527.

    Sofrygin O, Ogburn EL, & van der Laan MJ. (2018). Single Time Point Interventions in Network-Dependent
    Data. In Targeted Learning in Data Science (pp. 373-396). Springer.
    """
    def __init__(self, network_list, exposure, outcome, degree_restrict=None, alpha=0.05,
                 continuous_bound=0.0005, verbose=False,
                 task_string='00000',
                 cat_vars=[], cont_vars=[], cat_unique_levels={},
                 use_deep_learner_A_i=False, use_deep_learner_A_i_s=False, use_deep_learner_outcome=False, use_all_time_slices=True):
        # initiate task_string to save model under the right name
        self.task_string = task_string
        # initiate cat_vars, cont_vars and cat_unique_levels (SG_modified)
        self.cat_vars, self.cont_vars, self.cat_unique_levels = cat_vars, cont_vars, cat_unique_levels

        # Checking for some common problems that should provide errors
        for network in network_list:
            if not all([isinstance(x, int) for x in list(network.nodes())]):                    # Check if all node IDs are integers
                raise ValueError("NetworkTMLE requires that "                                   # ... possibly not needed?
                                "all node IDs must be integers")
        for network in network_list:
            if nx.number_of_selfloops(network) > 0:                                             # Check for any self-loops in the network
                raise ValueError("NetworkTMLE does not support networks "                       # ... self-loops don't make sense in this
                                "with self-loops")                                              # ... setting

        # Checking for a specified degree restriction
        self.degree_restrict = degree_restrict                                                  # Saving degree restriction
        if degree_restrict is not None:                                                         # not-None means apply a restriction
            self._check_degree_restrictions_(bounds=degree_restrict)                            # ... checks if valid degree restriction
            self._max_degree_list_ = [degree_restrict[1]]*len(network_list)                     # ... extract max degree as upper bound
        else:                                                                                   # otherwise if no restriction(s)
            if nx.is_directed(network):                                                         # ... directed max degree is max out-degree
                self._max_degree_list_ = [np.max([d for n, d in network.out_degree]) for network in network_list]
            else:                                                                               # ... undirected max degree is max degree
                self._max_degree_list_ = [np.max([d for n, d in network.degree]) for network in network_list]

        # Generate a fresh copy of the network with ascending node order
        self.oid = "_original_id_"                                                                   # Name to save the original IDs
        labeled_network_list = []
        for network in network_list:
            network = nx.convert_node_labels_to_integers(network,                               # Copy of new network with new labels
                                                        first_label=0,                          #  ... start at 0 for latent variance calc
                                                        label_attribute=self.oid)                    # ... saving the original ID labels
            labeled_network_list.append(network)                                                # ... saving to list

        # Saving processed data copies
        # self.network = network                                                                # Network with correct re-labeling
        self.network_list = labeled_network_list                                                # List of networks with correct re-labeling
        self.exposure = exposure                                                                # Exposure column / attribute name
        self.outcome = outcome                                                                  # Outcome column / attribute name

        # Background processing to convert network attribute data to pandas DataFrame
        self.adj_matrix_list = [nx.adjacency_matrix(network, weight=None) for network in self.network_list]
        df_list = [network_to_df(network) for network in self.network_list]
        # self.adj_matrix = nx.adjacency_matrix(self.network,   # Convert to adjacency matrix
        #                                       weight=None)    # TODO allow for weighted networks
        # df = network_to_df(self.network)                      # Convert node attributes to pandas DataFrame

        # Error checking for exposure types
        for df in df_list:
            if not df[exposure].value_counts().index.isin([0, 1]).all():        # Only binary exposures allowed currently
                raise ValueError("NetworkTMLE only supports binary exposures "
                                "currently")

        # Manage outcome data based on variable type
        self._continuous_outcome_list_ = []                                  
        self._cb_list_ = []                                                 
        self._continuous_min_list_ = []                                     
        self._continuous_max_list_ = []                                    
        
        for i in range(len(df_list)):  
            if df_list[i][outcome].dropna().value_counts().index.isin([0, 1]).all():                # Binary outcomes
                self._continuous_outcome_list_.append(False)                                        # ... mark as binary outcome
                self._cb_list_.append(0.0)                                                          # ... set continuous bound to be zero
                self._continuous_min_list_.append(0.0)                                              # ... saving binary min bound
                self._continuous_max_list_.append(1.0)                                              # ... saving binary max bound
            else:                                                                                   # Continuous outcomes
                self._continuous_outcome_list_.append(True)                                         # ... mark as continuous outcome
                self._cb_list_.append(continuous_bound)                                             # ... save continuous bound value
                self._continuous_min_list_.append(np.min(df_list[i][outcome]) - self._cb_list_[i])  # ... determine min (with bound)
                self._continuous_max_list_.append(np.max(df_list[i][outcome]) + self._cb_list_[i])  # ... determine max (with bound)
                df_list[i][outcome] = tmle_unit_bounds(y=df_list[i][self.outcome],                  # ... bound the outcomes to be (0,1)
                                                       mini=self._continuous_min_list_[i],
                                                       maxi=self._continuous_max_list_[i])

        # Creating summary measure mappings for all variables in the network
        summary_types = ['sum', 'mean', 'var', 'mean_dist', 'var_dist']                             # Default summary measures available
        handle_isolates = ['mean', 'var', 'mean_dist', 'var_dist']                                  # Whether isolates produce nan's
        for i in range(len(df_list)):         
            for v in [var for var in list(df_list[i].columns) if var not in [self.oid, outcome]]:        # All cols besides ID and outcome
                v_vector = np.asarray(df_list[i][v])                                                # ... extract array of column
                for summary_measure in summary_types:                                               # ... for each summary measure
                    df_list[i][v+'_'+summary_measure] = fast_exp_map(self.adj_matrix_list[i],       # ... calculate corresponding measure
                                                                     v_vector,
                                                                     measure=summary_measure)
                    if summary_measure in handle_isolates:                                          # ... set isolates from nan to 0
                        df_list[i][v+'_'+summary_measure] = df_list[i][v+'_'+summary_measure].fillna(0)
                    
                    if v+'_'+summary_measure not in self.cont_vars:
                        self.cont_vars.append(v+'_'+summary_measure)                                # ... add to continuous variables (SG_modified)

        # Creating summary measure mappings for non-parametric exposure_map_model()
        self._nonparam_cols_ = []
        for i in range(len(self.network_list)):
            exp_map_cols = exp_map_individual(network=self.network_list[i],                         # Generate columns of indicator
                                              variable=exposure,                                    # ... for the exposure
                                              max_degree=self._max_degree_list_[i])                 # ... up to the maximum degree
            self._nonparam_cols_.append(list(exp_map_cols.columns))                                 # Save column list for estimation procedure
            df_list[i] = pd.merge(df_list[i],                                                       # Merge these columns into main data
                                  exp_map_cols.fillna(0),                                           # set nan to 0 to keep same dimension across i
                                  how='left', left_index=True, right_index=True)                    # Merge on index to left

        # Assign all mappings variables  (SG_modified)
        # summary measures are consistent throughout time, hence all var names can be added to self.cat_vars/cont_vars
        # but the mapping values from neighbors may not be consistent, choose the maximum degree mapping to ensure inclusiveness
        if exposure in cat_vars:
            # print('categorical')
            self.cat_vars.extend(self._nonparam_cols_[-1]) # add all mappings to categorical variables

            for i in range(len(self._nonparam_cols_)):
                if i == 0: # init with the first time slice values
                    for col in self._nonparam_cols_[i]:
                        self.cat_unique_levels[col] = pd.unique(df_list[i][col].astype('int')).max() + 1
                else: # update when bigger degree is encountered
                    for col in self._nonparam_cols_[i]: 
                        if pd.unique(df_list[i][col].astype('int')).max() + 1 > self.cat_unique_levels[col]:
                            self.cat_unique_levels[col] = pd.unique(df_list[i][col].astype('int')).max() + 1

        elif exposure in cont_vars:
            # print('continuous')
           self.cont_vars.extend(self._nonparam_cols_[-1])
        else:
            raise ValueError('exposure is neither assigned to categorical or continuous variables')

        # Calculating degree for all the nodes
        self.df_list = [None] * len(df_list) # init self.df_list
        for i in range(len(self.network_list)):
            if nx.is_directed(self.network_list[i]):                                                    # For directed networks...
                degree_data = pd.DataFrame.from_dict(dict(self.network_list[i].out_degree),             # ... use the out-degree
                                                     orient='index').rename(columns={0: 'degree'})
            else:                                                                                       # For undirected networks...
                degree_data = pd.DataFrame.from_dict(dict(self.network_list[i].degree),                 # ... use the regular degree
                                                     orient='index').rename(columns={0: 'degree'})
            self.df_list[i] = pd.merge(df_list[i],                                                      # Merge main data
                                       degree_data,                                                     # ...with degree data
                                       how='left', left_index=True, right_index=True)                   # ...based on index
        
        # Assign degree variables (SG_modified)
        self.cat_vars.append('degree')
        for i in range(len(self.df_list)):
            if i == 0:
                self.cat_unique_levels['degree'] = pd.unique(self.df_list[i]['degree'].astype('int')).max() + 1
            else: # update when bigger degree is encountered
                if pd.unique(self.df_list[i]['degree'].astype('int')).max() + 1 > self.cat_unique_levels['degree']:
                    self.cat_unique_levels['degree'] = pd.unique(self.df_list[i]['degree'].astype('int')).max() + 1 

        # Apply degree restriction to data
        for i in range(len(self.df_list)):
            if degree_restrict is not None:                                                                                 # If restriction provided,
                self.df_list[i]['__degree_flag__'] = self._degree_restrictions_(degree_dist=self.df_list[i]['degree'],
                                                                                bounds=degree_restrict)
                self._exclude_ids_degree_ = np.asarray(self.df_list[i].loc[self.df_list[i]['__degree_flag__'] == 1].index)
            else:                                                                                                           # Else all observations are used
                self.df_list[i]['__degree_flag__'] = 0                                                                      # Mark all as zeroes
                self._exclude_ids_degree_ = None                                                                            # No excluded IDs

        # Marking data set restricted by degree (same as df if no restriction)
        # self.df_restricted = self.df.loc[self.df['__degree_flag__'] == 0].copy()
        self.df_restricted_list = [df.loc[df['__degree_flag__'] == 0].copy() for df in self.df_list]

        # Output attributes
        self.marginals_vector, self.marginal_outcome = None, None
        self.conditional_variance, self.conditional_latent_variance = None, None
        self.conditional_ci, self.conditional_latent_ci = None, None
        self.alpha = alpha

        # Storage for items for estimation procedures
        self._outcome_model, self._q_model, self._Qinit_ = None, None, None

        self._treatment_models = []
        self._gi_model, self._gs_model = None, None
        self._gs_measure_, self._map_dist_ = None, None
        self._exposure_measure_ = None
        self._denominator_, self._denominator_estimated_ = None, False

        # Threshold or category processing
        self._thresholds_, self._thresholds_variables_, self._thresholds_def_ = [], [], []
        self._thresholds_any_ = False
        self._categorical_, self._categorical_variables_, self._categorical_def_ = [], [], []
        self._categorical_any_ = False

        # Custom model / machine learner storage
        self._gi_custom_, self._gi_custom_sim_ = None, None
        self._gs_custom_, self._gs_custom_sim_ = None, None
        self._q_custom_ = None

        # Storage items for summary formatting
        self._specified_p_, self._specified_bound_, self._resamples_ = None, None, None
        self._verbose_ = verbose

        # Use deep learner for exposure or outcome nuisance model (SG_modified)
        self.use_deep_learner_A_i, self.use_deep_learner_A_i_s = use_deep_learner_A_i, use_deep_learner_A_i_s
        self.use_deep_learner_outcome = use_deep_learner_outcome
        self.use_all_time_slices = use_all_time_slices

    def exposure_model(self, model, custom_model=None, custom_model_sim=None):
        """Exposure model for individual i.  Estimates Pr(A=a|W, W_map) using a logistic regression model.

        Note
        ----
        This function only saves the model specifications. IPTW are calculated later during the fit() procedure since
        the policy is needed.

        Parameters
        ----------
        model : str
            Exposure mapping model. Ideally would include treatment for individual i
        custom_model
            User-specified model
        custom_model_sim
            User-specified model. This allows the user to specify a different IPW model to be fit for the numerator.
            That model is fit to the simulated data, so some constraints may be added to speed up the estimation
            procedure. If None and custom_model is not None, copies over the custom_model used.
        """
        # Clearing memory of previous / old models
        self._gi_model = model                       # Exposure model for A_i
        self._treatment_models = []                  # Clearing out possible old stored models
        self._denominator_estimated_ = False         # Mark the denominator as not having been estimated yet

        # Storing user-specified model
        self._gi_custom_ = custom_model              # Custom model fitter being stored
        if custom_model_sim is None:                 # Custom model fitter for simulated data
            self._gi_custom_sim_ = custom_model      # ... same as actual data if not specified
        else:                                        # Otherwise
            self._gi_custom_sim_ = custom_model_sim  # ... use specified model

    def exposure_map_model(self, model, measure=None, distribution=None, custom_model=None, custom_model_sim=None):
        """Exposure summary measure model for individual i. Estimates Pr(A_map=a|A=a, W, W_map) using a logistic
        regression model.

        Note
        ----
        Only saves the model specifications. IPTW are calculated later during the fit() function

        There are several options for the distributions of the summary measure. One option is a non-parametric approach
        that estimates the probability for each individual contact (works best for uniform distributions). However, this
        approach may not always be possible to estimate. Instead, parametric distributional assumption can be used
        instead. Currently, implemented are normal and Poisson distributions.

        Parameters
        ----------
        model : str
            Exposure mapping model. Ideally would include treatment for individual i
        measure : None, str, optional
            Exposure mapping to use for the modeling statement. Options include 'mean' and 'sum'. Default is None
            which natively works with the `distribution=None` option
        distribution : None, str, optional
            Distribution to use for exposure mapping model. Options include: non-parametric (None), Normal ('normal'),
            Poisson ('poisson').
        custom_model : None, optional
            User-specified model
        custom_model_sim
            User-specified model. This allows the user to specify a different IPW model to be fit for the numerator.
            That model is fit to the simulated data, so some constraints may be added to speed up the estimation
            procedure. If None and custom_model is not None, copies over the custom_model used.
        """
        # Checking that distribution and measure are compatible
        self._check_distribution_measure_(distribution=distribution,   # Check distribution for model
                                          measure=measure)             # ... and specified measure

        # Clearing memory of previous / old models
        self._gs_model = model                                  # Exposure model for A_i^s
        self._treatment_models = []                             # Clearing out possible old stored models
        self._denominator_estimated_ = False                    # Mark denominator as not having been estimated yet

        # Getting distribution for parametric models. Ignored if custom_model is not None
        if distribution is None:                                # If None is the distribution
            self._map_dist_ = distribution                      # ... keeping as None for later logic
        else:                                                   # Otherwise
            self._map_dist_ = distribution.lower()              # ... making lower-case to avoid processing errors

        if measure is not None:                                 # If the specified measure is not NoneType
            self._exposure_measure_ = measure                   # ... save the measures name
            self._gs_measure_ = self.exposure + '_' + measure   # ... pick out relevant column

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
        self._gs_custom_ = custom_model                         # Saving custom model for later fitting
        if custom_model_sim is None:                            # Checking if another model for the simulations is given
            self._gs_custom_sim_ = custom_model                 # ... if not, use previous
        else:                                                   # Otherwise
            # TODO this should have some error-checking
            self._gs_custom_sim_ = custom_model_sim             # ... store alternative model

    def outcome_model(self, model, custom_model=None, distribution='normal'):
        """Estimation of the outcome model E(Y|A, A_map, W, W_map).

        Note
        ----
        Estimates the outcome model (g-formula) using the observed data and generates predictions under the observed
        distribution of the exposure.

        Parameters
        ----------
        model : str
            Specified Q-model
        custom_model :
            User-specified model
        distribution : optional, str
            For non-binary outcome variables, the distribution of Y must be specified. Default is 'normal'.
        """
        # Storing model specification
        self._q_model = model

        # Running through logic for custom models
        if custom_model is None:                                           # If no custom model
            if not self._continuous_outcome_list_[-1]:                     # and not continuous
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
            self._outcome_model = smf.glm(self.outcome + ' ~ ' + self._q_model,            # Specified model form
                                          self.df_restricted_list[-1],                     # ... fit to restricted data
                                          family=f).fit()                                  # ... for given GLM family
            self._Qinit_ = self._outcome_model.predict(self.df_restricted_list[-1])        # Predict outcome values

            label = self.df_restricted_list[-1][self.outcome]
            pred_binary = np.round(self._Qinit_)
            acc = (pred_binary == label).sum().item()/label.shape[0]
            print(f'LR outcome model accuracy in observed data: {acc}')

            # If verbose is requested, output the relevant information
            if self._verbose_:
                print('==============================================================================')
                print('Outcome model')
                print(self._outcome_model.summary())

        # Logic if custom_model is provided
        else:
            if self.use_deep_learner_outcome:
                self._q_custom_ = custom_model
                print(f'Deep Learner will be trained in fit()')

                # use LR model to find epsilon
                if not self._continuous_outcome_list_[-1]:                     # and not continuous
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
                self._outcome_model = smf.glm(self.outcome + ' ~ ' + self._q_model,            # Specified model form
                                            self.df_restricted_list[-1],                       # ... fit to restricted data
                                            family=f).fit()                                    # ... for given GLM family
                self._Qinit_ = self._outcome_model.predict(self.df_restricted_list[-1])        # Predict outcome values

                label = self.df_restricted_list[-1][self.outcome]
                pred_binary = np.round(self._Qinit_)
                acc = (pred_binary == label).sum().item()/label.shape[0]
                print(f'LR outcome model accuracy in observed data: {acc}')

            else:
                # Extract data using the model
                data = patsy.dmatrix(model + ' - 1',                              # Specified model WITHOUT an intercept
                                    self.df_restricted_list[-1])                  # ... using the degree restricted data

                # Estimating custom_model
                self._q_custom_ = outcome_learner_fitting(ml_model=custom_model,                                    # User-specified model
                                                          xdata=np.asarray(data),                                   # Extracted X data
                                                          ydata=np.asarray(self.df_restricted_list[-1][self.outcome]))

                # Generating predictions
                self._Qinit_ = outcome_learner_predict(ml_model_fit=self._q_custom_,   # Fit custom_model
                                                       xdata=np.asarray(data))         # Observed X data

        # Ensures all predicted values are bounded: 
        # SG modified: continous outcome is already normalized, should compare with 0,1, not with _continuous_min/max_
        if self._continuous_outcome_list_[-1]:
            self._Qinit_ = np.where(self._Qinit_ < 0.,                              # When lower than lower bound
                                    0 + self._cb_list_[-1],                         # ... set to lower bound
                                    self._Qinit_)                                   # ... otherwise keep
            self._Qinit_ = np.where(self._Qinit_ > 1.,                              # When above the upper bound
                                    1 - self._cb_list_[-1],                         # ... set to upper bound
                                    self._Qinit_)                                   # ... otherwise keep

    def outcome_model_UDA(self, model, custom_model=None, 
                          observed_data_list=None, pooled_data_list=None, 
                          T_in_id=[*range(10)], T_out_id=[*range(10)]):
        src_xdata_list = []
        src_ydata_list = []
        trg_xdata_list = []
        trg_ydata_list = []
        n_output_list = []
        for obs_df, pool_df in zip(observed_data_list, pooled_data_list):
            if 'C(' in model:
                src_xdata_list.append(get_patsy_for_model_w_C(model, obs_df))
                trg_xdata_list.append(get_patsy_for_model_w_C(model, pool_df))
            else:
                src_xdata_list.append(patsy.dmatrix(model + ' - 1', obs_df, return_type="dataframe"))
                trg_xdata_list.append(patsy.dmatrix(model + ' - 1', pool_df, return_type="dataframe"))
            src_ydata_list.append(obs_df[self.outcome])
            trg_ydata_list.append(pool_df[self.outcome])
            n_output_list.append(pd.unique(obs_df[self.outcome]).shape[0])

        # custom_path = 'outcome_' + self.outcome + '.pth'
        custom_path = 'outcome_' + self.outcome + '_' + self.task_string + '.pth'
        self._q_custom_ = custom_model
        self._q_custom_path_, self._Qinit_DL_, self.y_star = outcome_deep_learner_ts(custom_model, src_xdata_list, src_ydata_list, trg_xdata_list, trg_ydata_list,
                                                                                     T_in_id, T_out_id,
                                                                                     self.adj_matrix_list, self.cat_vars, self.cont_vars, self.cat_unique_levels, n_output_list, self._continuous_outcome_list_[-1],
                                                                                     predict_with_best=False, custom_path=custom_path)
        # self._q_custom_path_, self._Qinit_, self.y_star = outcome_deep_learner_ts(custom_model, src_xdata_list, src_ydata_list, trg_xdata_list, trg_ydata_list,
        #                                                                           T_in_id, T_out_id,
        #                                                                           self.adj_matrix_list, self.cat_vars, self.cont_vars, self.cat_unique_levels, n_output_list, self._continuous_outcome_list_[-1],
        #                                                                           predict_with_best=False, custom_path=custom_path)

        # # Ensures all predicted values are bounded: 
        # # SG modified: continous outcome is already normalized, should compare with 0,1, not with _continuous_min/max_
        # if self._continuous_outcome_list_[-1]:
        #     self._Qinit_ = np.where(self._Qinit_ < 0.,                              # When lower than lower bound
        #                             0 + self._cb_list_[-1],                         # ... set to lower bound
        #                             self._Qinit_)                                   # ... otherwise keep
        #     self._Qinit_ = np.where(self._Qinit_ > 1.,                              # When above the upper bound
        #                             1 - self._cb_list_[-1],                         # ... set to upper bound
        #                             self._Qinit_)                                   # ... otherwise keep

    
    def fit(self, p, samples=100, bound=None, seed=None,
            shift=False, mode='top', percent_candidates=0.3, quarantine_period=2, inf_duration=5,
            T_in_id=[*range(10)], T_out_id=[*range(10)]):
        """Estimation procedure under a specified treatment plan.

        This function estimates the IPTW for the treatment plan of interest, performs the target steps, and
        performs Monte Carlo integration with the targeted model, and calculates confidence intervals. Confidence
        intervals are obtained from influence curves.

        Parameters
        ----------
        p : float, int, list, set
            Percent of population to treat. For conditional treatment plans, a container object of floats. All values
            must be between 0 and 1
        samples : int
            Number of samples to generate to calculate numerator for weights and for the Monte Carlo integration
            procedure for stochastic treatment plans. For deterministic treatment plans (p==1 or p==0), samples is set
            to 1 to reduce computation burden. Deterministic treatment plan do not require the Monte Carlo integration
            procedure
        bound : None, int, float
            Bounds to truncate calculate weights by...
        seed : int, None
            Random seed for the Monte Carlo integration procedure
        """
        # Error checking for function order called correctly
        if self._gi_model is None:                                                      # A_i model must be specified
            raise ValueError("exposure_model() must be specified before fit()")
        if self._gs_model is None:                                                      # A_i^s model must be specified
            raise ValueError("exposure_map_model() must be specified before fit()")
        if self._q_model is None:                                                       # Y model must be specified
            raise ValueError("outcome_model() must be specified before fit()")

        # Error checking for policies
        if type(p) is int:                                                              # Check if an integer is provided
            raise ValueError("Input `p` must be float or container of floats")

        if type(p) != float:                                                            # Check if not a float
            if len(p) != self.df_list[-1].shape[0]:                                     # ... check length matches data shape
                raise ValueError("Vector of `p` must be same length as input data")
            if np.all(np.asarray(p) == 0) or np.all(np.asarray(p) == 1):                # ... check if deterministic plan
                raise ValueError("Deterministic treatment plans not supported")
            if np.any(np.asarray(p) < 0) or np.any(np.asarray(p) > 1):                  # ... check if outside of prob bounds
                raise ValueError("Probabilities for treatment must be between 0 and 1")
        else:                                                                           # If it is a float
            if p == 0 or p == 1:                                                        # ... check if deterministic plan
                raise ValueError("Deterministic treatment plans not supported")
            if p < 0 or p > 1:                                                          # ... check if outside of prob bounds
                raise ValueError("Probabilities for treatment must be between 0 and 1")

        # Step 1) Estimate the weights
        # Also generates pooled_data for use in the Monte Carlo integration procedure
        self._resamples_ = samples                                                              # Saving info on number of resamples
        h_iptw, pooled_data_restricted_list, pooled_adj_matrix_list = self._estimate_iptw_ts_(p=p,                      # Generate pooled & estiamte weights
                                                                                              samples=samples,           # ... for some number of samples
                                                                                              bound=bound,               # ... with applied probability bounds
                                                                                              seed=seed,                 # ... and with a random seed given
                                                                                              shift=shift,
                                                                                              mode=mode,
                                                                                              percent_candidates=percent_candidates,
                                                                                              quarantine_period=quarantine_period,
                                                                                              inf_duration=inf_duration,
                                                                                              T_in_id=T_in_id, T_out_id=T_out_id)      
        # save for model test run, can be commented after test run
        self.h_iptw = h_iptw
        self.pooled_data_restricted_list, self.pooled_adj_matrix_list = pooled_data_restricted_list, pooled_adj_matrix_list 

        # Saving some information for diagnostic procedures
        if self._gs_measure_ is None:                                  # If no summary measure, use the A_sum
            self._for_diagnostics_ = pooled_data_restricted_list[-1][[self.exposure, self.exposure+"_sum"]].copy()
        else:                                                          # Otherwise, use the provided summary measure
            self._for_diagnostics_ = pooled_data_restricted_list[-1][[self.exposure, self._gs_measure_]].copy()

        # Step 2) Estimate from Q-model
        # if not dl model, process completed in .outcome_model() function and stored in self._Qinit_
        # so nothing to do here
        # dl model 
        if self.use_deep_learner_outcome: # train and pred in one run
            self.outcome_model_UDA(model=self._q_model, custom_model=self._q_custom_, 
                                   observed_data_list=self.df_restricted_list, pooled_data_list=pooled_data_restricted_list, 
                                   T_in_id=T_in_id, T_out_id=T_out_id)
        
        # Step 3) Target the parameter
        # if self.use_deep_learner_outcome:  # use LR to find epsilon
        if self.use_deep_learner_outcome: # set DL epsilon with bounded q_init in targeting_step()
            # epsilon = 0.
            epsilon = targeting_step(y=self.df_restricted_list[-1][self.outcome],   # Estimate the targeting model given observed Y
                                     q_init=self._Qinit_DL_,                        # ... predicted values of Y under observed A
                                     ipw=h_iptw,                                    # ... weighted by IPW
                                     verbose=self._verbose_)                        # ... with option for verbose info
            print(f'epsilon: {epsilon}') 
            if np.abs(epsilon) > 10: # if epsilon is off-the-chart large, set it to 0
                print(f'epsilon: {epsilon} is off-the-chart large, set it to 0')
                epsilon = 0.         
        else:
            epsilon = targeting_step(y=self.df_restricted_list[-1][self.outcome],   # Estimate the targeting model given observed Y
                                     q_init=self._Qinit_,                           # ... predicted values of Y under observed A
                                     ipw=h_iptw,                                    # ... weighted by IPW
                                     verbose=self._verbose_)                        # ... with option for verbose info
            print(f'epsilon: {epsilon}')

        # Step 4) Monte Carlo integration (old code did in loop but faster as vector)
        #
        # Generating outcome predictions under the policies (via pooled data sets)
        if self._q_custom_ is None:                                                     # If given a parametric default model
            self.y_star = self._outcome_model.predict(pooled_data_restricted_list[-1])       # ... predict using statsmodels syntax
            label = pooled_data_restricted_list[-1][self.outcome]
            pred_binary = np.round(self.y_star)
            acc = (pred_binary == label).sum().item()/label.shape[0]
            print(f'Outcome model accuracy in pooled data: {acc}')
        else:  # Custom input model by user
            if self.use_deep_learner_outcome:
                pass
            else:
                d = patsy.dmatrix(self._q_model + ' - 1', pooled_data_restricted_list[-1])  # ... extract data via patsy
                self.y_star = outcome_learner_predict(ml_model_fit=self._q_custom_,              # ... predict using custom function
                                                xdata=np.asarray(d))                        # ... for the extracted data
                label = pooled_data_restricted_list[-1][self.outcome]
                pred_binary = np.round(self.y_star)
                acc = (pred_binary == label).sum().item()/label.shape[0]
                print(f'Outcome model accuracy in pooled data: {acc}')
        
            
        # Ensure all predicted values are bounded properly for continuous
        # SG modified: continous outcome is already normalized, should compare with 0,1, not with _continuous_min/max_
        if self._continuous_outcome_list_[-1]:
            self.y_star = np.where(self.y_star < 0., 0. + self._cb_list_[-1], self.y_star)
            self.y_star = np.where(self.y_star > 1., 1. - self._cb_list_[-1], self.y_star)     

        # if self._continuous_outcome_list_[-1]:
        #     self.y_star = np.where(self.y_star < self._continuous_min_list_[-1], self._continuous_min_list_[-1], self.y_star)
        #     self.y_star = np.where(self.y_star > self._continuous_max_list_[-1], self._continuous_max_list_[-1], self.y_star)

        # Updating predictions via intercept from targeting step
        # # Article equation version, which is the same as described in the article, 
        # # but probability_to_odd() and odds_to_probability() cannot handle zero division and create nans when self.y_star == 1
        # logit_qstar = np.log(probability_to_odds(self.y_star)) + epsilon                         # NOTE: needs to be logit(Y^*) + e
        # q_star = odds_to_probability(np.exp(logit_qstar))                                   # Back converting from odds
        # Stablized version: handles zero division problem and when self.y_star contains one, the q_star will report 1 correspondingly
        q_star = (self.y_star * np.exp(epsilon)) / (1 - self.y_star + self.y_star * np.exp(epsilon))
        # # if need to check the results of q_star equals, rename variable to q_star_eq and q_star_stable
        # where_they_differ = np.isclose(q_star_eq, q_star_stable).sum()
        # print(f'Two versions have {len(where_they_differ)} differences')
        # print(f'Equation version q_star: {q_star_eq[~np.isclose(q_star_eq, q_star_stable)]}')
        # print(f'Stable version q_star: {q_star_stable[~np.isclose(q_star_eq, q_star_stable)]}')
        pooled_data_restricted_list[-1]['__pred_q_star__'] = q_star                         # Storing predictions as column

        # Taking the mean, grouped-by the pooled sample IDs (fast)
        self.marginals_vector = np.asarray(pooled_data_restricted_list[-1].groupby('_sample_id_')['__pred_q_star__'].mean())

        # If continuous outcome, need to unbound the means
        if self._continuous_outcome_list_[-1]:
            self.marginals_vector = tmle_unit_unbound(self.marginals_vector,                    # Take vector of MC results
                                                      mini=self._continuous_min_list_[-1],      # ... unbound using min
                                                      maxi=self._continuous_max_list_[-1])      # ... and max values

        # Calculating estimate for the policy
        self.marginal_outcome = np.mean(self.marginals_vector)                                  # Mean of Monte Carlo results
        self._specified_p_ = p                                                                  # Save what the policy was

        # Prep for variance
        if self._continuous_outcome_list_[-1]:                                                  # Continuous needs bounds...
            y_ = np.array(tmle_unit_unbound(self.df_restricted_list[-1][self.outcome],          # Unbound observed outcomes for Var
                                            mini=self._continuous_min_list_[-1],                # ... using min
                                            maxi=self._continuous_max_list_[-1]))               # ... and max values
            yq0_ = tmle_unit_unbound(self._Qinit_,                                              # Unbound g-comp predictions
                                     mini=self._continuous_min_list_[-1],                       # ... using min
                                     maxi=self._continuous_max_list_[-1])                       # ... and max values
        else:                                                                                   # Otherwise nothing special...
            y_ = np.array(self.df_restricted_list[-1][self.outcome])                            # Observed outcome for Var
            if self.use_deep_learner_outcome:
                yq0_ = self._Qinit_DL_
            else:
                yq0_ = self._Qinit_                                                                 # Predicted outcome for Var

        # Step 5) Variance estimation
        zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)                                   # Get corresponding Z-value based on desired alpha

        # Variance: direct-only, conditional on W variance
        var_cond = self._est_variance_conditional_(iptw=h_iptw,                                 # Estimate direct-only variance
                                                   obs_y=y_,                                    # ... observed value of Y
                                                   pred_y=yq0_)                                 # ... predicted value of Y
        self.conditional_variance = var_cond                                                    # Store the var estimate and CIs
        self.conditional_ci = [self.marginal_outcome - zalpha*np.sqrt(var_cond),
                               self.marginal_outcome + zalpha*np.sqrt(var_cond)]

        # Variance: direct and latent, conditional on W variance
        var_lcond = self._est_variance_latent_conditional_(iptw=h_iptw,                         # Estimate latent variance
                                                           obs_y=y_,                            # ... observed value of Y
                                                           pred_y=yq0_,                         # ... predicted value of Y
                                                           adj_matrix=self.adj_matrix_list[-1],
                                                           excluded_ids=self._exclude_ids_degree_)
        self.conditional_latent_variance = var_lcond                                            # Store variance estimate and CIs
        self.conditional_latent_ci = [self.marginal_outcome - zalpha*np.sqrt(var_lcond),
                                      self.marginal_outcome + zalpha*np.sqrt(var_lcond)]

    def summary(self, decimal=3):
        """Prints summary results for the sample average treatment effect under the treatment plan specified in
        the fit procedure

        Parameters
        ----------
        decimal : int
            Number of decimal places to display

        Returns
        -------
        None
        """
        # Check to make sure there is an answer to actually report
        if self.marginal_outcome is None:
            raise ValueError('The fit() statement must be ran before summary()')

        # Summary information
        print('======================================================================')
        print('            Network Targeted Maximum Likelihood Estimator             ')
        print('======================================================================')
        fmt = 'Treatment:        {:<15} No. Observations:     {:<20}'
        print(fmt.format(self.exposure, self.df_restricted_list[-1].shape[0]))
        fmt = 'Outcome:          {:<15} No. Background Nodes: {:<20}'
        print(fmt.format(self.outcome, self.df_list[-1].shape[0] - self.df_restricted_list[-1].shape[0]))
        fmt = 'Q-Model:          {:<15} No. IPW Truncated:    {:<20}'
        if self._specified_bound_ is None:
            b = 0
        else:
            b = self._specified_bound_
        if self._q_custom_ is None:
            qm = 'Logistic'
        else:
            qm = 'Custom'
        print(fmt.format(qm, b))

        fmt = 'g-Model:          {:<15} No. Resamples:        {:<20}'
        if self._gi_custom_ is None:
            gim = 'Logistic'
        else:
            gim = 'Custom'
        print(fmt.format(gim, self._resamples_))

        fmt = 'gs-Model:         {:<15} g-Distribution:       {:<20}'
        if self._gs_custom_ is None:
            if self._map_dist_ is None:
                gsm = 'Logitistic'
            else:
                gsm = self._map_dist_.capitalize()
        else:
            gsm = 'Custom'
        if self._map_dist_ is None:
            gs = 'Nonparametric'
        else:
            gs = self._map_dist_.capitalize()
        print(fmt.format(gsm, gs))

        print('======================================================================')
        print('Mean under policy:      ', np.round(self.marginal_outcome, decimals=decimal))
        print('----------------------------------------------------------------------')
        print('Variance Estimates')
        print('----------------------------------------------------------------------')
        print('Conditional: Direct-Only')
        print("SE      :     ", np.round(self.conditional_variance**0.5, decimals=decimal))
        print(str(round(100 * (1 - self.alpha), 0)) + '% CL:    ',
              np.round(self.conditional_ci, decimals=decimal))
        print('Conditional: Direct & Latent')
        print("SE      :     ", np.round(self.conditional_latent_variance**0.5, decimals=decimal))
        print(str(round(100 * (1 - self.alpha), 0)) + '% CL:    ',
              np.round(self.conditional_latent_ci, decimals=decimal))
        print('======================================================================')

    def diagnostics(self, figsize=(6, 5), color_a1='blue', color_a0='red'):
        r"""Returns diagnostic plot for the specified network-TMLE. The currently available diagnostic presents plots of
        the designated summary measure for :math:`A^s` (stratified by :math:`A`) for the observed data, and the Monte
        Carlo simulated data. This diagnostic can be used to visually assess whether the designated policy is
        poorly-supported by the data.

        Note
        ----
        A policy that has little overlap with the observed data is indicative of the policy being poorly supported by
        the observed data. Poorly-supported policies may not be well estimated and thus considering other stochastic
        policies in recommended.

        Parameters
        ----------
        figsize : list, set, array, optional
            Determine the figure size (dimensions). Passes directly to ``plt.subplots(...figsize=figsize)``.
        color_a1 : str, optional
            Color for the A=1 group in the figure. Default is blue.
        color_a0 : str, optional
            Color for the A=0 group in the figure. Default is red.

        Returns
        -------
        Diagnostic plot for data support of policy.
        """
        # Extract the summary measures, stratified by A_i, from the observed and MC simulated data
        obs_a1 = self.df_restricted.loc[self.df_restricted[self.exposure] == 1, self._gs_measure_].copy()
        obs_a0 = self.df_restricted.loc[self.df_restricted[self.exposure] == 0, self._gs_measure_].copy()
        sim_a1 = self._for_diagnostics_.loc[self._for_diagnostics_[self.exposure] == 1, self._gs_measure_].copy()
        sim_a0 = self._for_diagnostics_.loc[self._for_diagnostics_[self.exposure] == 0, self._gs_measure_].copy()

        # Generic figure setup
        fig, ax = plt.subplots(nrows=2, figsize=figsize)

        # If provided a summary measure of None, sum, or threshold
        if self._exposure_measure_ in [None, "sum"] or self._exposure_measure_[0] == "t":
            min_x = 0                                                                 # Min is always zero here
            max_x = int(np.max([np.max(self.df_restricted[self._gs_measure_]),        # Look for max in two sets of data
                                np.max(self._for_diagnostics_[self._gs_measure_])]))

            # Plotting: Observed values
            pa1 = obs_a1.value_counts(normalize=True, dropna=True, ascending=True)
            ax[0].bar([x-0.2 for x in pa1.index], pa1, width=0.4, color=color_a1, label=r"$A=1$")
            pa0 = obs_a0.value_counts(normalize=True, dropna=True, ascending=True)
            ax[0].bar([x+0.2 for x in pa0.index], pa0, width=0.4, color=color_a0, label=r"$A=0$")
            ax[0].set_xticks([x for x in range(min_x, max_x+1)])
            ax[0].set_xticklabels(["" for x in range(min_x, max_x+1)])
            ax[0].set_xlim([min_x-1, max_x+1])
            ax[0].set_ylim([0, 1])
            ax[0].set_ylabel("Proportion")
            ax[0].set_title("Observed")
            ax[0].legend()

            # Plotting: Policy
            qa1 = sim_a1.value_counts(normalize=True, dropna=True, ascending=True)
            ax[1].bar([x-0.2 for x in qa1.index], qa1, width=0.4, color=color_a1)
            qa0 = sim_a0.value_counts(normalize=True, dropna=True, ascending=True)
            ax[1].bar([x+0.2 for x in qa0.index], qa0, width=0.4, color=color_a0)
            ax[1].set_xticks([x for x in range(min_x, max_x+1)])
            ax[1].set_xlim([min_x-1, max_x+1])
            ax[1].set_xlabel(r"$A^s$")
            ax[1].set_ylim([0, 1])
            ax[1].set_ylabel("Proportion")
            ax[1].set_title(r"Under $\omega$")

        # Otherwise, plot as a density plot for other summary measures
        else:
            if self._exposure_measure_ in ["mean"]:   # The mean summary measure has
                min_x = 0                             # ... min lower bound of zero
                max_x = 1                             # ... max upper bound of one
            else:                                     # Otherwise extract min/max from data itself
                min_x = np.min([np.min(self.df_restricted[self._gs_measure_]),
                                np.min(self._for_diagnostics_[self._gs_measure_])]) - 0.2
                max_x = np.max([np.max(self.df_restricted[self._gs_measure_]),
                                np.max(self._for_diagnostics_[self._gs_measure_])]) + 0.2

            xticks = np.linspace(min_x, max_x, num=11)   # Defining some (hopefully) useful tick marks
            xvals = np.linspace(min_x, max_x, num=200)   # Creating values to fill-in between

            # Plotting: Observed values
            pa1 = gaussian_kde(obs_a1.dropna())
            pa0 = gaussian_kde(obs_a0.dropna())
            ax[0].fill_between(xvals, pa1(xvals), color=color_a1, alpha=0.2, label=None)
            ax[0].fill_between(xvals, pa0(xvals), color=color_a0, alpha=0.2, label=None)
            ax[0].plot(xvals, pa1(xvals), color=color_a1, alpha=1, label=r'$A=1$')
            ax[0].plot(xvals, pa0(xvals), color=color_a0, alpha=1, label=r'$A=0$')
            ax[0].set_xticks(xticks)
            ax[0].set_xticklabels(["" for x in range(len(xticks))])
            ax[0].set_xlim([min_x, max_x])
            ax[0].set_ylabel("Density")
            ax[0].set_yticks([])
            ax[0].set_title("Observed")
            ax[0].legend()

            # Plotting: Policy values
            qa1 = gaussian_kde(sim_a1.dropna())
            qa0 = gaussian_kde(sim_a0.dropna())
            ax[1].fill_between(xvals, qa1(xvals), color=color_a1, alpha=0.2, label=None)
            ax[1].fill_between(xvals, qa0(xvals), color=color_a0, alpha=0.2, label=None)
            ax[1].plot(xvals, qa1(xvals), color=color_a1, alpha=1)
            ax[1].plot(xvals, qa0(xvals), color=color_a0, alpha=1)
            ax[1].set_xticks(xticks)
            ax[1].set_xlim([min_x, max_x])
            ax[1].set_xlabel(r"$A^s$")
            ax[1].set_ylabel("Density")
            ax[1].set_yticks([])
            ax[1].set_title(r"Under $\alpha$")

        return ax

    def define_threshold(self, variable, threshold, definition):
        """Function arbitrarily allows for multiple different defined thresholds

        Parameters
        ----------
        variable : str
            Variable to generate categories for
        threshold : int, float
            Threshold to use as the cutpoint.
        definition: str
            ??
        """
        self._thresholds_any_ = True                    # Update logic to understand at least one threshold exists
        self._thresholds_.append(threshold)             # Add the threshold to the list to look at
        self._thresholds_variables_.append(variable)    # Add the variable to the list to make thresholds for
        self._thresholds_def_.append(definition)    
        
        for i in range(len(self.df_restricted_list)):
            create_threshold(self.df_restricted_list[i],            # Create the desired threshold variable
                             variables=[variable],          # ... for the specified variable
                             thresholds=[threshold],        # ... at the desired threshold
                             definitions=[definition])

    def define_category(self, variable, bins, labels=False):
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
        self._categorical_any_ = True                   # Update logic to understand at least one category exists
        self._categorical_variables_.append(variable)   # Add the variable to the list of category-generations
        self._categorical_.append(bins)                 # Add the cut-points for the bins to the list of bins
        self._categorical_def_.append(labels)           # Add the specified labels for the bins to the label list

        for i in range(len(self.df_restricted_list)):
            create_categorical(data=self.df_restricted_list[i],     # Create the desired category variable
                               variables=[variable],        # ... for the specified variable
                               bins=[bins],                 # ... for the specified bins
                               labels=[labels],             # ... with the specified labels
                               verbose=True)                # ... warns user if NaN's are being generated

    def _estimate_iptw_ts_(self, p, samples, bound, seed, shift, mode, percent_candidates, quarantine_period, inf_duration, T_in_id, T_out_id):
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

        SG_modified: time series version to generate pooled data and weights: 
        If exposure nuisance model is NOT deep learner, then denominator and numerator is estimated via the last time slice dataframe.
        The pooled data is longitudinal, generated per time slice. But the generated prediction from model is still the last time slice, same as non-ts version.        

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
        
        Retunrs
        ----------
        iptw: array
            Weights for targeting
        pooled_data_restricted_list: list
            List of dataframes to pooled data, len() = T        
        """
        # Estimate the denominator if not previously estimated
        if not self._denominator_estimated_:
            self._denominator_ = self._estimate_exposure_nuisance_ts_(data_to_fit_list=self.df_restricted_list.copy(),
                                                                      data_to_predict_list=self.df_restricted_list.copy(),
                                                                      adj_matrix_list=self.adj_matrix_list.copy(),
                                                                      T_in_id=T_in_id, 
                                                                      T_out_id=T_out_id,
                                                                      distribution=self._map_dist_,
                                                                      verbose_label='Weight - Denominator',
                                                                      store_model=True,
                                                                      custom_path_prefix='denom_',
                                                                      print_every=5)
            self._denominator_estimated_ = True  # Updates flag for denominator

        # Creating pooled sample to estimate weights
        pooled_data_restricted_list, pooled_adj_matrix_list = self._generate_pooled_samples(samples=samples,
                                                                                            seed=seed,
                                                                                            shift=shift,
                                                                                            mode=mode,
                                                                                            percent_candidates=percent_candidates,
                                                                                            pr_a=p,
                                                                                            quarantine_period=quarantine_period,
                                                                                            inf_duration=inf_duration)


        # Estimate the numerator using the pooled data
        # remove the adj_matrix for the final time step, retain the adj_matrix for the first time step
        pooled_adj_matrix_list = pooled_adj_matrix_list[:-1]
        numerator = self._estimate_exposure_nuisance_ts_(data_to_fit_list=pooled_data_restricted_list.copy(),
                                                         data_to_predict_list=self.df_restricted_list.copy(),
                                                         adj_matrix_list=pooled_adj_matrix_list,
                                                         T_in_id=T_in_id,
                                                         T_out_id=T_out_id,
                                                         distribution=self._map_dist_,
                                                         verbose_label='Weight - Numerator',
                                                         store_model=False,
                                                         custom_path_prefix='num_',
                                                         # kwargs
                                                         batch_size=512,
                                                         print_every=15)

        # Calculating weight: H = Pr*(A,A^s | W,W^s) / Pr(A,A^s | W,W^s)
        iptw = numerator / self._denominator_           # Divide numerator by denominator
        if bound is not None:                           # If weight bound provided
            iptw = bounding(ipw=iptw, bound=bound)      # ... apply the bound

        # Return both the array of estimated weights and the generated pooled data set
        return iptw, pooled_data_restricted_list, pooled_adj_matrix_list

    def update_graph_features(self, data, graph):
        '''Add features to graph nodes using data columns as node attributes'''
        for col in data.columns:
            if nx.is_directed(graph):
                raise NotImplementedError("Directed graph is not supported yet")
            else: 
                nx.set_node_attributes(graph, dict(data[col]), col)
        return graph
    
    def update_I_ratio(self, data, graph):
        ''' Update I_ratio for the graph data
        NOTE: this function is separated from update_summary_measures() because I_ratio needs to be updated for each inf,
        but other summary measures are only updated for each time step
        '''
        # retreive updated graph connections
        adj_matrix = nx.adjacency_matrix(graph, weight=None)
        # calculate I_ratio
        if 'degree' not in data.columns:
            data = pd.merge(data, pd.DataFrame.from_dict(dict(graph.degree),
                                                        orient='index').rename(columns={0: 'degree'}),
                                                        how='left', left_index=True, right_index=True)
        else:
            data['degree'] = list(dict(graph.degree).values())
        data['I_sum'] = fast_exp_map(adj_matrix, np.array(data['I']), measure='sum')
        data['I_ratio'] = data['I_sum'] / data['degree']  # ratio of infected neighbors
        data['I_ratio'] = data['I_ratio'].fillna(0)  # fill in 0 for nodes with no neighbors

        # add I_ratio to graph data
        if nx.is_directed(graph):
            raise NotImplementedError("Directed graph is not supported yet")
        else:
            nx.set_node_attributes(graph, dict(data['I_ratio']), 'I_ratio')
        
        return data, graph
    
    def update_summary_measures(self, data, graph):
        ''' Update data based on graph, whose connection is modified in the previous time step.
        Synchronize the new I_ratio and summary measure to the graph for current step''' 
        data = network_to_df(graph)

        # update degree in data
        if nx.is_directed(graph):
            degree_data = pd.DataFrame.from_dict(dict(graph.out_degree),             
                                                 orient='index').rename(columns={0: 'degree'}) 
        else:
            degree_data = pd.DataFrame.from_dict(dict(graph.degree),                 
                                                 orient='index').rename(columns={0: 'degree'})
        data['degree'] = degree_data
        
        # update I_ratio in data
        data, graph = self.update_I_ratio(data, graph)
        
        # update other summary measures in data
        adj_matrix = nx.adjacency_matrix(graph, weight=None)
        summary_types = ['sum', 'mean', 'var', 'mean_dist', 'var_dist']                           # Default summary measures available
        handle_isolates = ['mean', 'var', 'mean_dist', 'var_dist'] 
        input_df = network_to_df(self.network_list[0])                                            # Extract the variables from the original data
        for v in [var for var in list(input_df.columns) if var not in [self.oid, self.outcome]]:  # summary measure is generated for each original varible
            v_vector = np.asarray(data[v])                                                        # ... extract array of column
            for summary_measure in summary_types:                                                 # ... for each summary measure
                data[v+'_'+summary_measure] = fast_exp_map(adj_matrix,                            # ... calculate corresponding measure
                                                           v_vector,
                                                           measure=summary_measure)
                if summary_measure in handle_isolates:                                            # ... set isolates from nan to 0
                    data[v+'_'+summary_measure] = data[v+'_'+summary_measure].fillna(0)
        
        # update I_ratio and summary measures to graph
        # graph has to be updated because data for each time step data is synchronized from the graph to retreive I and D
        graph = self.update_graph_features(data, graph)
        
        return data, graph
    
    def apply_quarantine_action(self, data, graph, shift, edge_recorder, time_step, rng, pr_a, percent_candidates, mode):
        ''' Update the probability of quarantine action for every quarantine_period 
        P.S. The pr_a predicted by the exposure function is replaced by the selected pr_a'''
        if shift:
            probs = rng.binomial(n=1,                                  # Flip a coin to generate A_i
                                 p=pr_a,                               # ... based on policy-assigned probabilities
                                 size=data.shape[0])                   # ... for the N units
        else:
            # select candidates for quarantine based on the degree
            if mode == 'top':
                num_candidates = math.ceil(data.shape[0] * percent_candidates)
                candidates_nodes = data.nlargest(num_candidates, 'degree').index # super-spreader
            elif mode == 'bottom':
                num_candidates = math.ceil(data.shape[0] * percent_candidates)
                candidates_nodes = data.nsmallest(num_candidates, 'degree').index # super-defender
            elif mode == 'all':
                num_candidates = data.shape[0]
                candidates_nodes = data.index
            
            quarantine_piror = rng.binomial(n=1, p=pr_a, size=num_candidates)
            quarantine_nodes = candidates_nodes[quarantine_piror==1]
            probs = np.zeros(data.shape[0])
            probs[quarantine_nodes] = 1 

        # update and apply restriction if given
        if self.degree_restrict is not None:
            data['__degree_flag__'] = self._degree_restrictions_(degree_dist=data['degree'],
                                                                bounds=self.degree_restrict)
        else:
            data['__degree_flag__'] = 0
        data[self.exposure] = np.where(data['__degree_flag__'] == 1,  # Restrict to appropriate degree
                                       data[self.exposure], probs)    # ... keeps restricted nodes as observed A_i
        
        # apply quarantine to all selected nodes: remove immediate neighbors for the next time step
        # Function version
        if nx.is_directed(graph):
            raise NotImplementedError("Directed graph is not supported yet")
        else:
            nx.set_node_attributes(graph, dict(data['quarantine']), 'quarantine')
            edge_recorder[time_step].extend(list(nx.edges(graph, data[data['quarantine']==1].index))) 
            # extend list because for each time_step the edge_recorder is accumulated over all infected nodes' neighbors

        return data, graph, edge_recorder

    def update_edge_in_graph(self, graph, edge_recorder, time_step, quarantine_period):
        ''' Update the graph for every quarantine_period
        for current update, if it is not the first one, 
        add back previous quarantine edges after the quarantine period has passed;
        and then remove currently quarantined edges
        '''
        if time_step - quarantine_period >= quarantine_period: 
            graph.add_edges_from(edge_recorder[time_step-quarantine_period]) 
        graph.remove_edges_from(edge_recorder[time_step]) # remove current quarantined edges
        return graph

    def simulate_infection_of_immediate_neighbors(self, graph, inf, infected, rng):
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

    def update_adj_matrix(self, graph):
        adj_matrix = nx.adjacency_matrix(graph, weight=None) 

        if self.degree_restrict is not None:                                                         
            self._check_degree_restrictions_(bounds=self.degree_restrict)                           
            _max_degree_ = self.degree_restrict[1]                    
        else:                                                                                   
            if nx.is_directed(graph):
                _max_degree_ = np.max([d for n, d in graph.out_degree])                                                         
            else:                  
                _max_degree_ = np.max([d for n, d in graph.degree])
        
        return adj_matrix, _max_degree_

    def _generate_pooled_samples(self, samples, seed, shift, mode, percent_candidates, pr_a,
                                 quarantine_period, inf_duration):
        # initiate lists to store pooled data, network, adj_matrix, _max_degree_ for all samples
        pooled_all_samples = []
        network_all_samples = []
        adj_matrix_all_samples = []
        # _max_degree_all_samples = []

        # for each sample drawn, the process of generating time_limit steps of data is recorded
        for s in range(samples):
            print(f'Generating sample: {s}...') 
            # initiate rng seed for bootstraps
            rng = np.random.default_rng(seed+s)
            # initiate dataframe for the first time step
            data = self.df_list[0].copy()
            # initiate network, adj_matrix, _max_degree_  for the first time step
            graph_init = [self.update_graph_features(data, self.network_list[0].copy())]
            adj_matrix_init = [self.adj_matrix_list[0].copy()]
            # _max_degree_init = [self._max_degree_list_[0].copy()]
            # initiate edge_recorders to record edges to remove/add back for each quarantine_period
            edge_recorder = {key:[] for key in range(quarantine_period, len(self.df_list), quarantine_period)} # record edge_to_remove to be add back after the quarantine period has passed
            # starts with quarantine_period because the initial action is already applied
            # initiate infected list
            infected = list(self.df_list[0][self.df_list[0]['I'] == 1].index) 

            # intiate pooled sample list
            data['_sample_id_'] = s                # Setting sample ID
            # Re-creating any threshold variables in the pooled sample data
            if self._thresholds_any_:
                create_threshold(data=data,
                                 variables=self._thresholds_variables_,
                                 thresholds=self._thresholds_,
                                 definitions=self._thresholds_def_,
                                 graph=graph_init[0])
            # Re-creating any categorical variables in the pooled sample data
            if self._categorical_any_:
                create_categorical(data=data,
                                   variables=self._categorical_variables_,
                                   bins=self._categorical_,
                                   labels=self._categorical_def_,
                                   verbose=False,
                                   graph=graph_init[0])            

            pooled_per_sample = [data.copy()]

            for time_step in range(1, len(self.df_list)): # start from 1 because the first time step is already initiated
                print(f'Generating time step: {time_step}...')                
                if time_step == 1:
                    graph_current = graph_init[0].copy()
                    _max_degree_current = self._max_degree_list_[0].copy()

                # 1. Update data from previous graph 
                # 2. Update I_ratio and summary measures in data based on previous graph
                # 3. Synchronize graph node features using the updated data for next iteration
                data, graph_current = self.update_summary_measures(data, graph_current)
                
                for inf in sorted(infected, key=lambda _: rng.random()):
                    # Book-keeping for infected nodes
                    graph_current.nodes[inf]['I'] = 1
                    graph_current.nodes[inf]['D'] = 1
                    graph_current.nodes[inf]['t'] += 1

                    if graph_current.nodes[inf]['t'] > inf_duration:
                        graph_current.nodes[inf]['I'] = 0         # Node is no longer infectious after this loop
                        graph_current.nodes[inf]['R'] = 1         # Node switches to Recovered
                        infected.remove(inf)                      # Remove node from infected list
                    
                    # Update I_ratio for the graph data
                    data, graph_current = self.update_I_ratio(data, graph_current)

                    # Simulate infections of immediate neighbors
                    graph_current, infected = self.simulate_infection_of_immediate_neighbors(graph_current, inf, infected, rng)
                    
                # Update quarantine measures, which will be activated at the beginning of next time step via modifed graph connections
                if time_step % quarantine_period == 0:
                    data, graph_current, edge_recorder = self.apply_quarantine_action(data, graph_current, shift, edge_recorder, time_step, rng, pr_a, percent_candidates, mode)            
                            
                # Logic if no summary measure was specified (uses the complete factor approach)
                if self._gs_measure_ is None:
                    network_tmp = graph_init[time_step].copy()
                    a = np.array(data[self.exposure])                           # Transform A_i into array
                    for n in network_tmp.nodes():                               # For each node,
                        network_tmp.nodes[n][self.exposure] = a[n]              # ...assign the new A_i*
                    df = exp_map_individual(network_tmp,                        # Now do the individual exposure maps with new
                                            variable=self.exposure,
                                            max_degree=_max_degree_current).fillna(0)
                    for c in self._nonparam_cols_[-1]:                          # Adding back these np columns
                        data[c] = df[c] 
                        if nx.is_directed(graph_current):
                            raise NotImplementedError("Directed graph is not supported yet")
                        else:
                            nx.set_node_attributes(graph_current, dict(data[c]), c)
                    
                # Re-creating any threshold variables in the pooled sample data
                if self._thresholds_any_:
                    create_threshold(data=data,
                                     variables=self._thresholds_variables_,
                                     thresholds=self._thresholds_,
                                     definitions=self._thresholds_def_,
                                     graph=graph_current)

                # Re-creating any categorical variables in the pooled sample data
                if self._categorical_any_:
                    create_categorical(data=data,
                                       variables=self._categorical_variables_,
                                       bins=self._categorical_,
                                       labels=self._categorical_def_,
                                       verbose=False,
                                       graph=graph_current)

                data['_sample_id_'] = s                    # Setting sample ID
                pooled_per_sample.append(data.copy())      # Adding to list (for later concatenate)

                # Save graph, adj_matrix and update _max_degree_
                adj_matrix_current, _max_degree_current = self.update_adj_matrix(graph_current)
                graph_init.append(graph_current)
                adj_matrix_init.append(adj_matrix_current)
                # _max_degree_init.append(_max_degree_current)
                
                # Update graph for the next time step
                if time_step % quarantine_period == 0:
                    graph_current = self.update_edge_in_graph(graph_current, edge_recorder, time_step, quarantine_period)
                
            pooled_all_samples.append(pooled_per_sample)     # append pooled sample for each sample
            network_all_samples.append(graph_init)           # append network for each sample
            adj_matrix_all_samples.append(adj_matrix_init)   # append adj_matrix for each sample
            # _max_degree_all_samples.append(_max_degree_init) # append _max_degree_ for each sample
                
        # re-arrange list of lists: each elemental list should contain all samples and the number of sublists equals to time_limit
        pooled_all_samples_rearrange = [list(i) for i in zip(*pooled_all_samples)]
        pooled_adj_matrix_list = [list(i) for i in zip(*adj_matrix_all_samples)]

        pooled_data_restricted_list = []
        for pooled_sample_for_one_time_step in pooled_all_samples_rearrange:
            df_one_step = pd.concat(pooled_sample_for_one_time_step, axis=0, ignore_index=True)
            pooled_data_restricted = df_one_step.loc[df_one_step['__degree_flag__'] == 0].copy()
            pooled_data_restricted_list.append(pooled_data_restricted)
            # print('code is reachable')

        return pooled_data_restricted_list, pooled_adj_matrix_list

    def _estimate_exposure_nuisance_ts_(self, data_to_fit_list, data_to_predict_list, adj_matrix_list, T_in_id, T_out_id,
                                        distribution, verbose_label, store_model, 
                                        custom_path_prefix=None, **kwargs):
        """Unified function to estimate the numerator and denominator of the weights.

        Parameters
        ----------
        data_to_fit_list : list of dataframe
            Contains dataframe of all time slices, len() = T;
            Data to estimate the parameters of the nuisance model with. For the numerator, this is the pooled data set
            and for the denominator this is the observed data set. Both are restricted by degree, when applicable.
        data_to_predict_list : list of dataframe
            Contains dataframe of all time slices, len() = T;
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
        custom_path_prefix: string
            a prefix add to model path to distinguish between fitted models for numerator and denominator
        kwargs: dict
            include all/partial init parameters for AbstractML model, key should be the same as the init parameter name


        Returns
        -------
        array
            Predicted probability for the corresponding piece of the weights.
        """
        ##################################
        # Model: A_i
        if self._gi_custom_ is None:                                                    # If no A_i custom_model is provided
            f = sm.families.family.Binomial()                                           # ... use a logit model
            treat_i_model = smf.glm(self.exposure + ' ~ ' + self._gi_model,             # Specified model from exposure_...()
                                    data_to_fit_list[-1],                               # ... using the restricted data
                                    family=f).fit()                                     # ... for logit family
            # Verbose model results if requested
            if self._verbose_:
                print('==============================================================================')
                print(verbose_label+': A')
                print(treat_i_model.summary())

            pred = treat_i_model.predict(data_to_predict_list[-1])                     # Save predicted probability
            if store_model:                                                            # If denominator,
                self._treatment_models.append(treat_i_model)                           # save model to list (so can be extracted)

        else:                                                                          # Otherwise use the custom_model
            if self.use_deep_learner_A_i:
                xdata_list = []
                ydata_list = []
                pdata_list = []
                pdata_y_list = []
                n_output_list =[]
                for i, (d2f, d2p) in enumerate(zip(data_to_fit_list, data_to_predict_list)):
                    if 'C(' in self._gi_model:
                        xdata_list.append(get_patsy_for_model_w_C(self._gi_model, d2f))
                    else:
                        xdata_list.append(patsy.dmatrix(self._gi_model + ' - 1', d2f, return_type="dataframe"))
                    ydata_list.append(d2f[self.exposure])
                    n_output_list.append(pd.unique(d2f[self.exposure]).shape[0])
                    print(f'T_slice: {i} | gi_model | n_output: {n_output_list[i]} | target variable: {self.exposure}')
                    if 'C(' in self._gi_model:
                        pdata_list.append(get_patsy_for_model_w_C(self._gi_model, d2p))
                    else: 
                        pdata_list.append(patsy.dmatrix(self._gi_model + ' - 1', d2p, return_type="dataframe"))
                    pdata_y_list.append(d2p[self.exposure])
                
                # print(f'n_output_list: {n_output_list}')
                custom_path = custom_path_prefix + 'A_i_' + self.exposure  + '_' + self.task_string + '.pth' 
                pred = exposure_deep_learner_ts(self._gi_custom_,
                                                xdata_list, ydata_list, pdata_list, pdata_y_list, 
                                                T_in_id, T_out_id,
                                                adj_matrix_list, self.cat_vars, self.cont_vars, self.cat_unique_levels, n_output_list,
                                                custom_path, **kwargs)
            else:
                xdata = patsy.dmatrix(self._gi_model + ' - 1', data_to_fit_list[-1])       # Extract via patsy the data
                pdata = patsy.dmatrix(self._gi_model + ' - 1', data_to_predict_list[-1])   # Extract via patsy the data
                pred = exposure_machine_learner(ml_model=self._gi_custom_,                 # Custom model application and preds
                                                xdata=np.asarray(xdata),                   # ... with data to fit
                                                ydata=np.asarray(data_to_fit_list[-1][self.exposure]),
                                                pdata=np.asarray(pdata))                   # ... and data to predict

        # Assigning probability given observed
        pr_i = np.where(data_to_predict_list[-1][self.exposure] == 1,                      # If A_i = 1
                        pred,                                                              # ... get Pr(A_i=1 | ...)
                        1 - pred)                                                          # ... otherwise get Pr(A_i=0 | ...)

        ##################################
        # Model: A_i^s
        if distribution is None:                                                           # When no distribution is provided
            if self._gs_custom_ is None:                                                   # and no custom_model is given
                f = sm.families.family.Binomial()                                          # ... use a logit model
                cond_vars = patsy.dmatrix(self._gs_model,                                  # Extract initial set of covs
                                          data_to_fit_list[-1],                            # ... from the data to fit
                                          return_type='matrix')                            # ... as a NumPy matrix
                pred_vars = patsy.dmatrix(self._gs_model,                                  # Extract initial set of covs
                                          data_to_predict_list[-1],                        # ... from the data to predict
                                          return_type='matrix')                            # ... as a NumPy matrix
                pr_s = np.array([1.] * data_to_predict_list[-1].shape[0])                  # Setup vector of 1's as the probability

                for c in self._nonparam_cols_:                                             # For each of the NP columns
                    treat_s_model = sm.GLM(data_to_fit_list[-1][c], cond_vars,             # Estimate using the pooled data
                                           family=f).fit()                                 # ... with logistic model
                    if store_model:                                                        # If estimating denominator
                        self._treatment_models.append(treat_s_model)                       # Save estimated model for checking

                    # If verbose requested, provide model output
                    if self._verbose_:
                        print('==============================================================================')
                        print(verbose_label+': ' + c)
                        print(treat_s_model.summary())

                    # Generating predicted probabilities
                    pred = treat_s_model.predict(pred_vars)                                # Generate the prediction for observed
                    pr_s *= np.where(data_to_predict_list[-1][c] == 1,                     # cumulative probability for each step
                                     pred, 1 - pred)                                       # ...logic for correct probability value

                    # Stacking vector to the end of the array for next cycle
                    cond_vars = np.c_[cond_vars, np.array(data_to_fit_list[-1][c])]        # Update the covariates
                    pred_vars = np.c_[pred_vars, np.array(data_to_predict_list[-1][c])]    # Update the covariates

            else:                                                                          # Placeholder for conditional density SL
                # TODO fill in the super-learner conditional density approach here when possible...
                raise ValueError("Not available currently...")

        elif distribution == 'normal':                                                     # If a normal distribution
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model                          # Setup the model form
            treat_s_model = smf.ols(gs_model,                                              # Estimate via OLS
                                    data_to_fit_list[-1]).fit()                            # ... with data to fit
            self._treatment_models.append(treat_s_model)                                   # Store estimated model to check
            pred = treat_s_model.predict(data_to_predict_list[-1])                         # Generate predicted values for data to predict
            pr_s = norm.pdf(data_to_predict_list[-1][self._gs_measure_],                   # Get f(A_i^s | ...) for measure
                            pred,                                                          # ... given predicted value
                            np.sqrt(treat_s_model.mse_resid))                              # ... and sqrt of model residual

            # If verbose requested, provide model output
            if self._verbose_:
                print('==============================================================================')
                print(verbose_label+': '+self._gs_measure_)
                print(treat_s_model.summary())

        elif distribution == 'poisson':                                                    # If a poisson distribution
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model                          # Setup the model form
            if self._gs_custom_ is None:                                                   # If no custom model provided
                f = sm.families.family.Poisson()                                           # ... GLM with Poisson family
                treat_s_model = smf.glm(gs_model,                                          # Estimate model
                                        data_to_fit_list[-1],                              # ... with data to fit
                                        family=f).fit()                                    # ... and Poisson distribution
                if store_model:                                                            # If estimating denominator
                    self._treatment_models.append(treat_s_model)                           # ... store the model
                pred = treat_s_model.predict(data_to_predict_list[-1])                     # Predicted values with data to predict

                # If verbose requested, provide model output
                if self._verbose_:
                    print('==============================================================================')
                    print(verbose_label+': '+self._gs_measure_)
                    print(treat_s_model.summary())

            else:                                                                          # Custom model for Poisson
                if self.use_deep_learner_A_i_s:
                    xdata_list = []
                    ydata_list = []
                    pdata_list = []
                    pdata_y_list = []
                    n_output_list =[]

                    data_to_fit_subset_list = []
                    num_samples_used_min = float('inf')
                    for i, (d2f, d2p) in enumerate(zip(data_to_fit_list, data_to_predict_list)): # number of samples used should be consistent throughout time, loop over pooled dataset to obtain the minimum number of samples at a given time slice
                        data_to_fit_subset = select_pooled_sample_with_observed_data(self._gs_measure_, d2f, d2p)
                        data_to_fit_subset_list.append(data_to_fit_subset)
                        num_samples_used_min = data_to_fit_subset.shape[0] if data_to_fit_subset.shape[0] < num_samples_used_min else num_samples_used_min
                        print(f'T_slice: {i} | gs_model | samples used: {data_to_fit_subset.shape[0]} | samples original: {d2f.shape[0]}')
                        if 'C(' in self._gs_model:
                            pdata_list.append(get_patsy_for_model_w_C(self._gs_model, d2p))
                        else:
                            pdata_list.append(patsy.dmatrix(self._gs_model + ' - 1', d2p, return_type="dataframe"))
                        pdata_y_list.append(d2p[self._gs_measure_])

                    for i, data_to_fit_subset in enumerate(data_to_fit_subset_list):
                        data_to_fit_subset = data_to_fit_subset[:num_samples_used_min]
                        if 'C(' in self._gs_model:
                            xdata_list.append(get_patsy_for_model_w_C(self._gs_model, data_to_fit_subset))
                        else:                        
                            xdata_list.append(patsy.dmatrix(self._gs_model + ' - 1', data_to_fit_subset, return_type="dataframe"))
                        ydata_list.append(data_to_fit_subset[self._gs_measure_])
                        n_output_list.append(pd.unique(data_to_fit_subset[self._gs_measure_]).shape[0])
                        print(f'T_slice: {i} | gs_model | n_output : {n_output_list[i]} | target variables: {self._gs_measure_}')
                        

                    custom_path = custom_path_prefix + 'A_i_s_' + self.exposure + '_' + self.task_string + '.pth'
                    pred = exposure_deep_learner_ts(self._gs_custom_,
                                                    xdata_list, ydata_list, pdata_list, pdata_y_list, T_in_id, T_out_id,
                                                    adj_matrix_list, self.cat_vars, self.cont_vars, self.cat_unique_levels, n_output_list,
                                                    custom_path, **kwargs)
                else:
                    xdata = patsy.dmatrix(self._gs_model + ' - 1',                          # ... extract data given relevant model
                                          data_to_fit_list[-1])                             # ... from degree restricted
                    pdata = patsy.dmatrix(self._gs_model + ' - 1',                          # ... extract data given relevant model
                                          data_to_predict_list[-1])                         # ... from degree restricted
                    pred = exposure_machine_learner(ml_model=self._gs_custom_,              # Custom ML model
                                                    xdata=np.asarray(xdata),                # ... with data to fit
                                                    ydata=np.asarray(data_to_fit_list[-1][self._gs_measure_]),
                                                    pdata=np.asarray(pdata))                # ... and data to predict
            
            if self.use_deep_learner_A_i_s: # deep learner output probability already, no need to transform
                pr_s = pred
            else:
                pr_s = poisson.pmf(data_to_predict_list[-1][self._gs_measure_], pred)       # Get f(A_i^s | ...) for measure

        elif distribution == 'multinomial':                                                 # If multinomial distribution
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model                           # Setup the model form
            treat_s_model = smf.mnlogit(gs_model,                                           # Estimate multinomial model
                                        data_to_fit_list[-1]).fit(disp=False)               # ... with data to fit
            if store_model:                                                                 # If estimating denominator
                self._treatment_models.append(treat_s_model)                                # ... add fitted model to list of models

            pred = treat_s_model.predict(data_to_predict_list[-1])                          # predict probabilities for each category
            values = pd.get_dummies(data_to_predict_list[-1][self._gs_measure_])            # transform to dummy variables for processing
            pr_s = np.array([0.0] * data_to_predict_list[-1].shape[0])                      # generate blank array of probabilities
            for i in data_to_predict_list[-1][self._gs_measure_].unique():                  # for each unique value in the multinomial
                try:                                                                        # ... try-except skips if unique not occur
                    pr_s += pred[i] * values[i]                                             # ... update probability
                except KeyError:                                                            # ... logic to skip the KeyError's
                    pass

            # If verbose requested, provide model output
            if self._verbose_:
                print('==============================================================================')
                print(verbose_label+': '+self._gs_measure_)
                print(treat_s_model.summary())

        elif distribution == 'binomial':                                                    # If binomial distribution
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model                           # setup the model form
            f = sm.families.family.Binomial()                                               # specify the logistic family option
            treat_s_model = smf.glm(gs_model,                                               # Estimate the model
                                    data_to_fit_list[-1],                                   # ... with data to fit
                                    family=f).fit()                                         # ... and logistic model
            if store_model:                                                                 # If estimating denominator
                self._treatment_models.append(treat_s_model)                                # ... add fitted model to list of models
            pr_s = treat_s_model.predict(data_to_predict_list[-1])                          # generate predicted probabilities of As=1

            # If verbose requested, provide model output
            if self._verbose_:
                print('==============================================================================')
                print(verbose_label+': '+self._gs_measure_)
                print(treat_s_model.summary())

        elif distribution == 'threshold':                                                   # If distribution is a threshold
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model                           # setup the model form
            if self._gs_custom_ is None:                                                    # if no custom model is given
                f = sm.families.family.Binomial()                                           # ... logistic model
                treat_s_model = smf.glm(gs_model,                                           # Estimate the model
                                        data_to_fit_list[-1],                               # ... with data to fit
                                        family=f).fit()                                     # ... and logistic model
                if store_model:                                                             # If estimating the denominator
                    self._treatment_models.append(treat_s_model)                            # ... add fitted model to list of models
                pred = treat_s_model.predict(data_to_predict_list[-1])                      # Generate predicted values of As=threshold

                # If verbose requested, provide model output
                if self._verbose_:
                    print('==============================================================================')
                    print('g-model: '+self._gs_measure_)
                    print(treat_s_model.summary())
            else:                                                                           # Else custom model for threshold
                if self.use_deep_learner_A_i_s:
                    xdata_list = []
                    ydata_list = []
                    pdata_list = []
                    pdata_y_list = []
                    n_output_list =[]

                    for i, (d2f, d2p) in enumerate(zip(data_to_fit_list, data_to_predict_list)):
                        data_to_fit_subset = select_pooled_sample_with_observed_data(self._gs_measure_, d2f, d2p)
                        print(f'T_slice: {i} | gs_model | samples used: {data_to_fit_subset.shape[0]} | samples original: {d2f.shape[0]}')
                        xdata_list.append(patsy.dmatrix(self._gs_model + ' - 1', data_to_fit_subset, return_type="dataframe"))
                        ydata_list.append(data_to_fit_subset[self._gs_measure_])
                        n_output_list.append(pd.unique(data_to_fit_subset[self._gs_measure_]).shape[0])
                        print(f'T_slice: {i} | gs_model | n_output : {n_output_list[i]} | target variables: {self._gs_measure_}')
                        pdata_list.append(patsy.dmatrix(self._gs_model + ' - 1', d2p, return_type="dataframe"))
                        pdata_y_list.append(d2p[self._gs_measure_])

                    custom_path = custom_path_prefix + 'A_i_s_' + self.exposure + '_' + self.task_string + '.pth'
                    pred = exposure_deep_learner_ts(self._gs_custom_,
                                                    xdata_list, ydata_list, pdata_list, pdata_y_list, T_in_id, T_out_id,
                                                    adj_matrix_list, self.cat_vars, self.cont_vars, self.cat_unique_levels, n_output_list,
                                                    custom_path, **kwargs)
                else:
                    xdata = patsy.dmatrix(self._gs_model + ' - 1', data_to_fit_list[-1])       # Processing data to be fit
                    pdata = patsy.dmatrix(self._gs_model + ' - 1', data_to_predict_list[-1])   # Processing data to be fit
                    pred = exposure_machine_learner(ml_model=self._gs_custom_,                 # Estimating the ML
                                                    xdata=np.asarray(xdata),                   # ... with data to fit
                                                    ydata=np.asarray(data_to_fit_list[-1][self._gs_measure_]),
                                                    pdata=np.asarray(xdata))                   # ... and data to predict
                    
            pr_s = np.where(data_to_predict_list[-1][self._gs_measure_] == 1,                  # Getting predicted values
                            pred,
                            1 - pred)

        else:
            raise ValueError("Invalid distribution choice")

        ##################################
        # Creating estimated Pr(A,A^s|W,W^s)
        return pr_i * pr_s    # Multiplying the factored probabilities back together

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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


if __name__ == '_main__':

    ############################ Test using quarantine ##########################
    from beowulf import load_uniform_vaccine
    from beowulf.dgm import quarantine_dgm_time_series
    import torch
    from dl_trainer_time_series import MLPTS, GCNTS


    n=500
    G, cat_vars, cont_vars, cat_unique_levels = load_uniform_vaccine(n=n, return_cat_cont_split=True)
    H, network_list, cat_vars, cont_vars, cat_unique_levels = quarantine_dgm_time_series(G, restricted=False, 
                                                                                        time_limit=10, inf_duration=5,
                                                                                        update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
                                                                                        random_seed=3407)
    print(f'quarantine uniform n={n}: {cat_unique_levels}')

    tmle = NetworkTMLETimeSeries(network_list, exposure='quarantine', outcome='D', verbose=False, degree_restrict=None,
                                cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
                                use_deep_learner_A_i=True, use_deep_learner_A_i_s=False, use_deep_learner_outcome=True)

    # tmle.exposure_model("A + H + H_mean + degree")
    # tmle.exposure_map_model("quarantine + A + H + H_mean + degree",
    #                             measure='sum', distribution='poisson')
    # tmle.outcome_model("quarantine + quarantine_mean + A + H + A_mean + H_mean + degree")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(device)

    # # MLP model
    # deep_learner_a_i = MLPTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
    #                          epochs=10, print_every=5, device=device, save_path='./tmp.pth')
    # # deep_learner_a_i_s = MLPTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
    # #                              epochs=10, print_every=5, device=device, save_path='./tmp.pth')    
    # deep_learner_outcome = MLPTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
    #                              epochs=10, print_every=5, device=device, save_path='./tmp.pth')  

    # GCN model
    deep_learner_a_i = GCNTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
                            epochs=10, print_every=5, device=device, save_path='./tmp.pth')
    # deep_learner_a_i_s = GCNTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
    #                            epochs=10, print_every=5, device=device, save_path='./tmp.pth')
    deep_learner_outcome = GCNTS(split_ratio=[0.6, 0.2, 0.2], batch_size=16, shuffle=True, n_splits=5, predict_all=True,
                                epochs=10, print_every=5, device=device, save_path='./tmp.pth')

    tmle.exposure_model("A + H + H_mean + degree", custom_model=deep_learner_a_i)
    tmle.exposure_map_model("quarantine + A + H + H_mean + degree",
                                measure='sum', distribution='poisson')
    tmle.outcome_model("quarantine + quarantine_mean + A + H + A_mean + H_mean + degree", custom_model=deep_learner_outcome)

    tmle.fit(p=0.55, bound=0.01, seed=3407)
    tmle.summary()
