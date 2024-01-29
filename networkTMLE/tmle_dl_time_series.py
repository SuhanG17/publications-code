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

from tmle_utils import (network_to_df, fast_exp_map, exp_map_individual, tmle_unit_bounds, tmle_unit_unbound,
                        probability_to_odds, odds_to_probability, bounding,
                        outcome_learner_fitting, outcome_learner_predict, exposure_machine_learner, exposure_deep_learner, outcome_deep_learner,
                        targeting_step, create_threshold, create_categorical,
                        check_pooled_sample_levels, select_pooled_sample_with_observed_data,
                        get_model_cat_cont_split_patsy_matrix, append_target_to_df, get_probability_from_multilevel_prediction)


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
                 cat_vars=[], cont_vars=[], cat_unique_levels={},
                 use_deep_learner_A_i=False, use_deep_learner_A_i_s=False, use_deep_learner_outcome=False):
        # initiate cat_vars, cont_vars and cat_unique_levels (SG_modified)
        self.cat_vars, self.cont_vars, self.cat_unique_levels = cat_vars, cont_vars, cat_unique_levels

        # Checking for some common problems that should provide errors
        for network in network_list:
            if not all([isinstance(x, int) for x in list(network.nodes())]):   # Check if all node IDs are integers
                raise ValueError("NetworkTMLE requires that "                  # ... possibly not needed?
                                "all node IDs must be integers")
        for network in network_list:
            if nx.number_of_selfloops(network) > 0:                            # Check for any self-loops in the network
                raise ValueError("NetworkTMLE does not support networks "      # ... self-loops don't make sense in this
                                "with self-loops")                             # ... setting

        # Checking for a specified degree restriction
        if degree_restrict is not None:                                    # not-None means apply a restriction
            self._check_degree_restrictions_(bounds=degree_restrict)       # ... checks if valid degree restriction
            self._max_degree_ = [degree_restrict[1]]*len(network_list)     # ... extract max degree as upper bound
        else:                                                              # otherwise if no restriction(s)
            if nx.is_directed(network):                                    # ... directed max degree is max out-degree
                self._max_degree_ = [np.max([d for n, d in network.out_degree]) for network in network_list]
            else:                                                          # ... undirected max degree is max degree
                self._max_degree_ = [np.max([d for n, d in network.degree]) for network in network_list]

        # Generate a fresh copy of the network with ascending node order
        oid = "_original_id_"                                              # Name to save the original IDs
        labeled_network_list = []
        for network in network_list:
            network = nx.convert_node_labels_to_integers(network,              # Copy of new network with new labels
                                                        first_label=0,        # ... start at 0 for latent variance calc
                                                        label_attribute=oid)  # ... saving the original ID labels
            labeled_network_list.append(network)                               # ... saving to list

        # Saving processed data copies
        # self.network = network                       # Network with correct re-labeling
        self.network_list = labeled_network_list     # List of networks with correct re-labeling
        self.exposure = exposure                     # Exposure column / attribute name
        self.outcome = outcome                       # Outcome column / attribute name

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
        self._continuous_outcome = []                                  
        self._cb_ = []                                                 
        self._continuous_min_ = []                                     
        self._continuous_max_ = []                                    
        
        for i in range(len(df_list)):  
            if df_list[i][outcome].dropna().value_counts().index.isin([0, 1]).all():  # Binary outcomes
                self._continuous_outcome.append(False)                                # ... mark as binary outcome
                self._cb_.append(0.0)                                                 # ... set continuous bound to be zero
                self._continuous_min_.append(0.0)                                     # ... saving binary min bound
                self._continuous_max_.append(1.0)                                     # ... saving binary max bound
            else:                                                                     # Continuous outcomes
                self._continuous_outcome.append(True)                                 # ... mark as continuous outcome
                self._cb_.append(continuous_bound)                                    # ... save continuous bound value
                self._continuous_min_.append(np.min(df_list[i][outcome]) - self._cb_) # ... determine min (with bound)
                self._continuous_max_.append(np.max(df_list[i][outcome]) + self._cb_) # ... determine max (with bound)
                df_list[i][outcome] = tmle_unit_bounds(y=df_list[i][self.outcome],    # ... bound the outcomes to be (0,1)
                                                       mini=self._continuous_min_,
                                                       maxi=self._continuous_max_)

        # Creating summary measure mappings for all variables in the network
        summary_types = ['sum', 'mean', 'var', 'mean_dist', 'var_dist']                         # Default summary measures available
        handle_isolates = ['mean', 'var', 'mean_dist', 'var_dist']                              # Whether isolates produce nan's
        for i in range(len(df_list)):         
            for v in [var for var in list(df_list[i].columns) if var not in [oid, outcome]]:    # All cols besides ID and outcome
                v_vector = np.asarray(df_list[i][v])                                            # ... extract array of column
                for summary_measure in summary_types:                                           # ... for each summary measure
                    df_list[i][v+'_'+summary_measure] = fast_exp_map(self.adj_matrix_list[i],   # ... calculate corresponding measure
                                                                     v_vector,
                                                                     measure=summary_measure)
                    if summary_measure in handle_isolates:                                      # ... set isolates from nan to 0
                        df_list[i][v+'_'+summary_measure] = df_list[i][v+'_'+summary_measure].fillna(0)
                    
                    if v+'_'+summary_measure not in self.cont_vars:
                        self.cont_vars.append(v+'_'+summary_measure)                            # ... add to continuous variables (SG_modified)

        # Creating summary measure mappings for non-parametric exposure_map_model()
        self._nonparam_cols_ = []
        for i in range(len(self.network_list)):
            exp_map_cols = exp_map_individual(network=self.network_list[i],               # Generate columns of indicator
                                              variable=exposure,             # ... for the exposure
                                              max_degree=self._max_degree_[i])  # ... up to the maximum degree
            self._nonparam_cols_.append(list(exp_map_cols.columns))                # Save column list for estimation procedure
            df_list[i] = pd.merge(df_list[i],                                                # Merge these columns into main data
                                  exp_map_cols.fillna(0),                            # set nan to 0 to keep same dimension across i
                                  how='left', left_index=True, right_index=True)     # Merge on index to left

        # Assign all mappings variables  (SG_modified)
        # summary measures are consistent throughout time, hence all var names can be added to self.cat_vars/cont_vars
        # but the mapping values from neighbors may not be consistent, choose the maximum degree mapping to ensure inclusiveness
        if exposure in cat_vars:
            # print('categorical')
            self.cat_vars.extend(self._nonparam_cols_[-1]) # add all mappings to categorical variables

            for i in range(len(self._nonparam_cols_)):
                if i == 0: # init with the first time point values
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
            self.df_list[i] = pd.merge(df_list[i],                                                         # Merge main data
                                       degree_data,                                                        # ...with degree data
                                       how='left', left_index=True, right_index=True)                      # ...based on index
        
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
            if not self._continuous_outcome:                               # and not continuous
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
            self._outcome_model = smf.glm(self.outcome + ' ~ ' + self._q_model,   # Specified model form
                                          self.df_restricted,                     # ... fit to restricted data
                                          family=f).fit()                         # ... for given GLM family
            self._Qinit_ = self._outcome_model.predict(self.df_restricted)        # Predict outcome values

            # If verbose is requested, output the relevant information
            if self._verbose_:
                print('==============================================================================')
                print('Outcome model')
                print(self._outcome_model.summary())

        # Logic if custom_model is provided
        else:
            if self.use_deep_learner_outcome:
                for df_restricted in self.df_restricted_list:
                    xdata.append(patsy.dmatrix(model + ' - 1', df_restricted, return_type="dataframe"))
                ydata.append(self.df_restricted_list[-1][self.outcome])
                n_output = pd.unique(ydata).shape[0]
                custom_path = 'outcome_' + self.outcome + '.pth'
                self._q_custom_ = custom_model

        

                xdata = patsy.dmatrix(model + ' - 1', self.df_restricted, return_type="dataframe")
                ydata = self.df_restricted[self.outcome] 
                n_output = pd.unique(ydata).shape[0]
                custom_path = 'outcome_' + self.outcome + '.pth'
                self._q_custom_ = custom_model
                self._q_custom_path_, self._Qinit_ = outcome_deep_learner(custom_model, 
                                                                          xdata, ydata, self.outcome,
                                                                          self.adj_matrix, self.cat_vars, self.cont_vars, self.cat_unique_levels, n_output, self._continuous_outcome,
                                                                          predict_with_best=False, custom_path=custom_path)
            else:
                # Extract data using the model
                data = patsy.dmatrix(model + ' - 1',                      # Specified model WITHOUT an intercept
                                    self.df_restricted)                  # ... using the degree restricted data

                # Estimating custom_model
                self._q_custom_ = outcome_learner_fitting(ml_model=custom_model,       # User-specified model
                                                        xdata=np.asarray(data),      # Extracted X data
                                                        ydata=np.asarray(self.df_restricted[self.outcome]))

                # Generating predictions
                self._Qinit_ = outcome_learner_predict(ml_model_fit=self._q_custom_,   # Fit custom_model
                                                    xdata=np.asarray(data))         # Observed X data

        # Ensures all predicted values are bounded: 
        # SG modified: continous outcome is already normalized, should compare with 0,1, not with _continuous_min/max_
        if self._continuous_outcome:
            self._Qinit_ = np.where(self._Qinit_ < 0.,          # When lower than lower bound
                                    0 + self._cb_,                         # ... set to lower bound
                                    self._Qinit_)                                  # ... otherwise keep
            self._Qinit_ = np.where(self._Qinit_ > 1.,          # When above the upper bound
                                    1 - self._cb_,                         # ... set to upper bound
                                    self._Qinit_)                                  # ... otherwise keep

        # if self._continuous_outcome:
        #     self._Qinit_ = np.where(self._Qinit_ < self._continuous_min_,          # When lower than lower bound
        #                             self._continuous_min_,                         # ... set to lower bound
        #                             self._Qinit_)                                  # ... otherwise keep
        #     self._Qinit_ = np.where(self._Qinit_ > self._continuous_max_,          # When above the upper bound
        #                             self._continuous_max_,                         # ... set to upper bound
        #                             self._Qinit_)                                  # ... otherwise keep

    def fit(self, p, samples=100, bound=None, seed=None):
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
        if self._gi_model is None:                                               # A_i model must be specified
            raise ValueError("exposure_model() must be specified before fit()")
        if self._gs_model is None:                                               # A_i^s model must be specified
            raise ValueError("exposure_map_model() must be specified before fit()")
        if self._q_model is None:                                                # Y model must be specified
            raise ValueError("outcome_model() must be specified before fit()")

        # Error checking for policies
        if type(p) is int:                                                       # Check if an integer is provided
            raise ValueError("Input `p` must be float or container of floats")

        if type(p) != float:                                                     # Check if not a float
            if len(p) != self.df.shape[0]:                                       # ... check length matches data shape
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
        self._resamples_ = samples                                                # Saving info on number of resamples
        h_iptw, pooled_data_restricted = self._estimate_iptw_(p=p,                # Generate pooled & estiamte weights
                                                              samples=samples,    # ... for some number of samples
                                                              bound=bound,        # ... with applied probability bounds
                                                              seed=seed)          # ... and with a random seed given

        # Saving some information for diagnostic procedures
        if self._gs_measure_ is None:                                  # If no summary measure, use the A_sum
            self._for_diagnostics_ = pooled_data_restricted[[self.exposure, self.exposure+"_sum"]].copy()
        else:                                                          # Otherwise, use the provided summary measure
            self._for_diagnostics_ = pooled_data_restricted[[self.exposure, self._gs_measure_]].copy()

        # Step 2) Estimate from Q-model
        # process completed in .outcome_model() function and stored in self._Qinit_
        # so nothing to do here

        # Step 3) Target the parameter
        epsilon = targeting_step(y=self.df_restricted[self.outcome],   # Estimate the targeting model given observed Y
                                 q_init=self._Qinit_,                  # ... predicted values of Y under observed A
                                 ipw=h_iptw,                           # ... weighted by IPW
                                 verbose=self._verbose_)               # ... with option for verbose info

        # Step 4) Monte Carlo integration (old code did in loop but faster as vector)
        #
        # Generating outcome predictions under the policies (via pooled data sets)
        if self._q_custom_ is None:                                            # If given a parametric default model
            y_star = self._outcome_model.predict(pooled_data_restricted)       # ... predict using statsmodels syntax
        else:  # Custom input model by user
            if self.use_deep_learner_outcome:
                xdata = patsy.dmatrix(self._q_model + ' - 1', pooled_data_restricted, return_type="dataframe")
                ydata = pooled_data_restricted[self.outcome] 
                n_output = pd.unique(ydata).shape[0]
                y_star = outcome_deep_learner(self._q_custom_, xdata, ydata, self.outcome, 
                                              self.adj_matrix, self.cat_vars, self.cont_vars, self.cat_unique_levels, n_output, self._continuous_outcome,
                                              predict_with_best=True, custom_path=self._q_custom_path_)
            else:
                d = patsy.dmatrix(self._q_model + ' - 1', pooled_data_restricted)  # ... extract data via patsy
                y_star = outcome_learner_predict(ml_model_fit=self._q_custom_,     # ... predict using custom function
                                                xdata=np.asarray(d))              # ... for the extracted data

        # Ensure all predicted values are bounded properly for continuous
        # SG modified: continous outcome is already normalized, should compare with 0,1, not with _continuous_min/max_
        if self._continuous_outcome:
            y_star = np.where(y_star < 0., 0. + self._cb_, y_star)
            y_star = np.where(y_star > 1., 1. - self._cb_, y_star)     

        # if self._continuous_outcome:
        #     y_star = np.where(y_star < self._continuous_min_, self._continuous_min_, y_star)
        #     y_star = np.where(y_star > self._continuous_max_, self._continuous_max_, y_star)

        # Updating predictions via intercept from targeting step
        logit_qstar = np.log(probability_to_odds(y_star)) + epsilon            # NOTE: needs to be logit(Y^*) + e
        q_star = odds_to_probability(np.exp(logit_qstar))                      # Back converting from odds
        pooled_data_restricted['__pred_q_star__'] = q_star                     # Storing predictions as column

        # Taking the mean, grouped-by the pooled sample IDs (fast)
        self.marginals_vector = np.asarray(pooled_data_restricted.groupby('_sample_id_')['__pred_q_star__'].mean())

        # If continuous outcome, need to unbound the means
        if self._continuous_outcome:
            self.marginals_vector = tmle_unit_unbound(self.marginals_vector,          # Take vector of MC results
                                                      mini=self._continuous_min_,     # ... unbound using min
                                                      maxi=self._continuous_max_)     # ... and max values

        # Calculating estimate for the policy
        self.marginal_outcome = np.mean(self.marginals_vector)                       # Mean of Monte Carlo results
        self._specified_p_ = p                                                       # Save what the policy was

        # Prep for variance
        if self._continuous_outcome:                                                 # Continuous needs bounds...
            y_ = np.array(tmle_unit_unbound(self.df_restricted[self.outcome],        # Unbound observed outcomes for Var
                                            mini=self._continuous_min_,              # ... using min
                                            maxi=self._continuous_max_))             # ... and max values
            yq0_ = tmle_unit_unbound(self._Qinit_,                                   # Unbound g-comp predictions
                                     mini=self._continuous_min_,                     # ... using min
                                     maxi=self._continuous_max_)                     # ... and max values
        else:                                                                        # Otherwise nothing special...
            y_ = np.array(self.df_restricted[self.outcome])                          # Observed outcome for Var
            yq0_ = self._Qinit_                                                      # Predicted outcome for Var

        # Step 5) Variance estimation
        zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)      # Get corresponding Z-value based on desired alpha

        # Variance: direct-only, conditional on W variance
        var_cond = self._est_variance_conditional_(iptw=h_iptw,                  # Estimate direct-only variance
                                                   obs_y=y_,                     # ... observed value of Y
                                                   pred_y=yq0_)                  # ... predicted value of Y
        self.conditional_variance = var_cond                                     # Store the var estimate and CIs
        self.conditional_ci = [self.marginal_outcome - zalpha*np.sqrt(var_cond),
                               self.marginal_outcome + zalpha*np.sqrt(var_cond)]

        # Variance: direct and latent, conditional on W variance
        var_lcond = self._est_variance_latent_conditional_(iptw=h_iptw,          # Estimate latent variance
                                                           obs_y=y_,             # ... observed value of Y
                                                           pred_y=yq0_,          # ... predicted value of Y
                                                           adj_matrix=self.adj_matrix,
                                                           excluded_ids=self._exclude_ids_degree_)
        self.conditional_latent_variance = var_lcond                             # Store variance estimate and CIs
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
        print(fmt.format(self.exposure, self.df_restricted.shape[0]))
        fmt = 'Outcome:          {:<15} No. Background Nodes: {:<20}'
        print(fmt.format(self.outcome, self.df.shape[0] - self.df_restricted.shape[0]))
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

    # def define_threshold(self, variable, threshold):
    #     """Function arbitrarily allows for multiple different defined thresholds

    #     Parameters
    #     ----------
    #     variable : str
    #         Variable to generate categories for
    #     threshold : int, float
    #         Threshold to use as the cutpoint.
    #     """
    #     self._thresholds_any_ = True                    # Update logic to understand at least one threshold exists
    #     self._thresholds_.append(threshold)             # Add the threshold to the list to look at
    #     self._thresholds_variables_.append(variable)    # Add the variable to the list to make thresholds for
    #     create_threshold(self.df_restricted,            # Create the desired threshold variable
    #                      variables=[variable],          # ... for the specified variable
    #                      thresholds=[threshold])        # ... at the desired threshold

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
        create_threshold(self.df_restricted,            # Create the desired threshold variable
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
        create_categorical(data=self.df_restricted,     # Create the desired category variable
                           variables=[variable],        # ... for the specified variable
                           bins=[bins],                 # ... for the specified bins
                           labels=[labels],             # ... with the specified labels
                           verbose=True)                # ... warns user if NaN's are being generated

    def _estimate_iptw_(self, p, samples, bound, seed):
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
        if not self._denominator_estimated_:
            self._denominator_ = self._estimate_exposure_nuisance_(data_to_fit=self.df_restricted.copy(),
                                                                   data_to_predict=self.df_restricted.copy(),
                                                                   distribution=self._map_dist_,
                                                                   verbose_label='Weight - Denominator',
                                                                   store_model=True,
                                                                   custom_path_prefix='denom_',
                                                                   print_every=5)
            self._denominator_estimated_ = True  # Updates flag for denominator

        # Creating pooled sample to estimate weights
        pooled_df = self._generate_pooled_sample(p=p,                                      # Generate data under policy
                                                 samples=samples,                          # ... for m samples
                                                 seed=seed)                                # ... with a provided seed
        pooled_data_restricted = pooled_df.loc[pooled_df['__degree_flag__'] == 0].copy()   # Restricting pooled sample

        # ensure pooled data contains all exposure levels in observed data
        if self.use_deep_learner_A_i_s:
            regenerate_flag = check_pooled_sample_levels(self._gs_measure_, pooled_data_restricted, self.df_restricted)
            print(f'before while loop regenerate_flag: {regenerate_flag}')
            while regenerate_flag:
                print(f'regenerating pooled sample for {self._gs_measure_}')
                pooled_df = self._generate_pooled_sample(p=p,                                      # Generate data under policy
                                                        samples=samples,                           # ... for m samples
                                                        seed=seed)                                 # ... with a provided seed
                pooled_data_restricted = pooled_df.loc[pooled_df['__degree_flag__'] == 0].copy()   # Restricting pooled sample
                regenerate_flag = check_pooled_sample_levels(self._gs_measure_, pooled_data_restricted, self.df_restricted)
                print(f'in while loop regenerate_flag: {regenerate_flag}')
                print()

        # Estimate the numerator using the pooled data
        numerator = self._estimate_exposure_nuisance_(data_to_fit=pooled_data_restricted.copy(),
                                                      data_to_predict=self.df_restricted.copy(),
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
        return iptw, pooled_data_restricted

    def _generate_pooled_sample(self, p, samples, seed):
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
            g = self.df.copy()                                      # Create a copy of the data
            probs = rng.binomial(n=1,                               # Flip a coin to generate A_i
                                 p=p,                               # ... based on policy-assigned probabilities
                                 size=g.shape[0])                   # ... for the N units
            g[self.exposure] = np.where(g['__degree_flag__'] == 1,  # Restrict to appropriate degree
                                        g[self.exposure], probs)    # ... keeps restricted nodes as observed A_i

            # Generating all summary measures based on the new exposure (could maybe avoid for all?)
            g[self.exposure+'_sum'] = fast_exp_map(self.adj_matrix, np.array(g[self.exposure]), measure='sum')
            g[self.exposure + '_mean'] = fast_exp_map(self.adj_matrix, np.array(g[self.exposure]), measure='mean')
            g[self.exposure + '_mean'] = g[self.exposure + '_mean'].fillna(0)            # isolates should have mean=0
            g[self.exposure + '_var'] = fast_exp_map(self.adj_matrix, np.array(g[self.exposure]), measure='var')
            g[self.exposure + '_var'] = g[self.exposure + '_var'].fillna(0)              # isolates should have mean=0
            g[self.exposure + '_mean_dist'] = fast_exp_map(self.adj_matrix,
                                                           np.array(g[self.exposure]), measure='mean_dist')
            g[self.exposure + '_mean_dist'] = g[self.exposure + '_mean_dist'].fillna(0)  # isolates should have mean=0
            g[self.exposure + '_var_dist'] = fast_exp_map(self.adj_matrix,
                                                          np.array(g[self.exposure]), measure='var_dist')
            g[self.exposure + '_mean_dist'] = g[self.exposure + '_mean_dist'].fillna(0)  # isolates should have mean=0

            # Logic if no summary measure was specified (uses the complete factor approach)
            if self._gs_measure_ is None:
                network = self.network.copy()                           # Copy the network
                a = np.array(g[self.exposure])                          # Transform A_i into array
                for n in network.nodes():                               # For each node,
                    network.nodes[n][self.exposure] = a[n]              # ...assign the new A_i*
                df = exp_map_individual(network,                        # Now do the individual exposure maps with new
                                        variable=self.exposure,
                                        max_degree=self._max_degree_).fillna(0)
                for c in self._nonparam_cols_:                          # Adding back these np columns
                    g[c] = df[c]

            # Re-creating any threshold variables in the pooled sample data
            if self._thresholds_any_:
                create_threshold(data=g,
                                 variables=self._thresholds_variables_,
                                 thresholds=self._thresholds_,
                                 definitions=self._thresholds_def_)

            # Re-creating any categorical variables in the pooled sample data
            if self._categorical_any_:
                create_categorical(data=g,
                                   variables=self._categorical_variables_,
                                   bins=self._categorical_,
                                   labels=self._categorical_def_,
                                   verbose=False)

            g['_sample_id_'] = s         # Setting sample ID
            pooled_sample.append(g)      # Adding to list (for later concatenate)

        # Returning the pooled data set
        return pd.concat(pooled_sample, axis=0, ignore_index=True)

    def _estimate_exposure_nuisance_(self, data_to_fit, data_to_predict, distribution, verbose_label, store_model, 
                                     custom_path_prefix=None, **kwargs):
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
        if self._gi_custom_ is None:                                          # If no A_i custom_model is provided
            f = sm.families.family.Binomial()                                 # ... use a logit model
            treat_i_model = smf.glm(self.exposure + ' ~ ' + self._gi_model,   # Specified model from exposure_...()
                                    data_to_fit,                              # ... using the restricted data
                                    family=f).fit()                           # ... for logit family
            # Verbose model results if requested
            if self._verbose_:
                print('==============================================================================')
                print(verbose_label+': A')
                print(treat_i_model.summary())

            pred = treat_i_model.predict(data_to_predict)                     # Save predicted probability
            if store_model:                                                   # If denominator,
                self._treatment_models.append(treat_i_model)                  # save model to list (so can be extracted)

        else:                                                                 # Otherwise use the custom_model
            if self.use_deep_learner_A_i:
                xdata = patsy.dmatrix(self._gi_model + ' - 1', data_to_fit, return_type="dataframe")       # Extract via patsy the data
                ydata = data_to_fit[self.exposure] 
                n_output = pd.unique(ydata).shape[0]
                print(f'gi_model: n_output = {n_output} for target variable {self.exposure}')

                pdata = patsy.dmatrix(self._gi_model + ' - 1', data_to_predict, return_type="dataframe")   # Extract via patsy the data
                pdata_y = data_to_predict[self.exposure]
                custom_path = custom_path_prefix + 'A_i_' + self.exposure  + '.pth'
                pred = exposure_deep_learner(self._gi_custom_, 
                                             xdata, ydata, pdata, pdata_y, self.exposure,
                                             self.adj_matrix, self.cat_vars, self.cont_vars, self.cat_unique_levels, n_output,
                                             custom_path, **kwargs)
            else:
                xdata = patsy.dmatrix(self._gi_model + ' - 1', data_to_fit)       # Extract via patsy the data
                pdata = patsy.dmatrix(self._gi_model + ' - 1', data_to_predict)   # Extract via patsy the data
                pred = exposure_machine_learner(ml_model=self._gi_custom_,        # Custom model application and preds
                                                xdata=np.asarray(xdata),          # ... with data to fit
                                                ydata=np.asarray(data_to_fit[self.exposure]),
                                                pdata=np.asarray(pdata))          # ... and data to predict

        # Assigning probability given observed
        pr_i = np.where(data_to_predict[self.exposure] == 1,                  # If A_i = 1
                        pred,                                                 # ... get Pr(A_i=1 | ...)
                        1 - pred)                                             # ... otherwise get Pr(A_i=0 | ...)

        ##################################
        # Model: A_i^s
        if distribution is None:                                              # When no distribution is provided
            if self._gs_custom_ is None:                                      # and no custom_model is given
                f = sm.families.family.Binomial()                             # ... use a logit model
                cond_vars = patsy.dmatrix(self._gs_model,                     # Extract initial set of covs
                                          data_to_fit,                        # ... from the data to fit
                                          return_type='matrix')               # ... as a NumPy matrix
                pred_vars = patsy.dmatrix(self._gs_model,                     # Extract initial set of covs
                                          data_to_predict,                    # ... from the data to predict
                                          return_type='matrix')               # ... as a NumPy matrix
                pr_s = np.array([1.] * data_to_predict.shape[0])              # Setup vector of 1's as the probability

                for c in self._nonparam_cols_:                                # For each of the NP columns
                    treat_s_model = sm.GLM(data_to_fit[c], cond_vars,         # Estimate using the pooled data
                                           family=f).fit()                    # ... with logistic model
                    if store_model:                                           # If estimating denominator
                        self._treatment_models.append(treat_s_model)          # Save estimated model for checking

                    # If verbose requested, provide model output
                    if self._verbose_:
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
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model     # Setup the model form
            treat_s_model = smf.ols(gs_model,                         # Estimate via OLS
                                    data_to_fit).fit()                # ... with data to fit
            self._treatment_models.append(treat_s_model)              # Store estimated model to check
            pred = treat_s_model.predict(data_to_predict)             # Generate predicted values for data to predict
            pr_s = norm.pdf(data_to_predict[self._gs_measure_],       # Get f(A_i^s | ...) for measure
                            pred,                                     # ... given predicted value
                            np.sqrt(treat_s_model.mse_resid))         # ... and sqrt of model residual

            # If verbose requested, provide model output
            if self._verbose_:
                print('==============================================================================')
                print(verbose_label+': '+self._gs_measure_)
                print(treat_s_model.summary())

        elif distribution == 'poisson':                               # If a poisson distribution
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model     # Setup the model form
            if self._gs_custom_ is None:                              # If no custom model provided
                f = sm.families.family.Poisson()                      # ... GLM with Poisson family
                treat_s_model = smf.glm(gs_model,                     # Estimate model
                                        data_to_fit,                  # ... with data to fit
                                        family=f).fit()               # ... and Poisson distribution
                if store_model:                                       # If estimating denominator
                    self._treatment_models.append(treat_s_model)      # ... store the model
                pred = treat_s_model.predict(data_to_predict)         # Predicted values with data to predict

                # If verbose requested, provide model output
                if self._verbose_:
                    print('==============================================================================')
                    print(verbose_label+': '+self._gs_measure_)
                    print(treat_s_model.summary())

            else:                                                           # Custom model for Poisson
                if self.use_deep_learner_A_i_s:
                    data_to_fit_subset = select_pooled_sample_with_observed_data(self._gs_measure_, data_to_fit, data_to_predict)
                    print(f'gs_model: use {data_to_fit_subset.shape[0]} samples from original {data_to_fit.shape[0]} to fit the model')
                    xdata = patsy.dmatrix(self._gs_model + ' - 1', 
                                          data_to_fit_subset, return_type="dataframe")       # Extract via patsy the data
                    ydata = data_to_fit_subset[self._gs_measure_]
                    n_output = pd.unique(ydata).shape[0] 
                    print(f'gs_model: n_output = {n_output} for target variable {self._gs_measure_}')

                    pdata = patsy.dmatrix(self._gs_model + ' - 1', 
                                          data_to_predict, return_type="dataframe")   # Extract via patsy the data
                    pdata_y = data_to_predict[self._gs_measure_]
                    custom_path = custom_path_prefix + 'A_i_s_' + self.exposure  + '.pth'
                    pred = exposure_deep_learner(self._gs_custom_, 
                                                 xdata, ydata, pdata, pdata_y, self._gs_measure_,
                                                 self.adj_matrix, self.cat_vars, self.cont_vars, self.cat_unique_levels, n_output,
                                                 custom_path, **kwargs)
                else:
                    xdata = patsy.dmatrix(self._gs_model + ' - 1',              # ... extract data given relevant model
                                        data_to_fit)                          # ... from degree restricted
                    pdata = patsy.dmatrix(self._gs_model + ' - 1',              # ... extract data given relevant model
                                        data_to_predict)                      # ... from degree restricted
                    pred = exposure_machine_learner(ml_model=self._gs_custom_,  # Custom ML model
                                                    xdata=np.asarray(xdata),    # ... with data to fit
                                                    ydata=np.asarray(data_to_fit[self._gs_measure_]),
                                                    pdata=np.asarray(pdata))    # ... and data to predict
            
            if self.use_deep_learner_A_i_s: # deep learner output probability already, no need to transform
                pr_s = pred
            else:
                pr_s = poisson.pmf(data_to_predict[self._gs_measure_], pred)    # Get f(A_i^s | ...) for measure

        elif distribution == 'multinomial':                              # If multinomial distribution
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model        # Setup the model form
            treat_s_model = smf.mnlogit(gs_model,                        # Estimate multinomial model
                                        data_to_fit).fit(disp=False)     # ... with data to fit
            if store_model:                                              # If estimating denominator
                self._treatment_models.append(treat_s_model)             # ... add fitted model to list of models

            pred = treat_s_model.predict(data_to_predict)                # predict probabilities for each category
            values = pd.get_dummies(data_to_predict[self._gs_measure_])  # transform to dummy variables for processing
            pr_s = np.array([0.0] * data_to_predict.shape[0])            # generate blank array of probabilities
            for i in data_to_predict[self._gs_measure_].unique():        # for each unique value in the multinomial
                try:                                                     # ... try-except skips if unique not occur
                    pr_s += pred[i] * values[i]                          # ... update probability
                except KeyError:                                         # ... logic to skip the KeyError's
                    pass

            # If verbose requested, provide model output
            if self._verbose_:
                print('==============================================================================')
                print(verbose_label+': '+self._gs_measure_)
                print(treat_s_model.summary())

        elif distribution == 'binomial':                                 # If binomial distribution
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model        # setup the model form
            f = sm.families.family.Binomial()                            # specify the logistic family option
            treat_s_model = smf.glm(gs_model,                            # Estimate the model
                                    data_to_fit,                         # ... with data to fit
                                    family=f).fit()                      # ... and logistic model
            if store_model:                                              # If estimating denominator
                self._treatment_models.append(treat_s_model)             # ... add fitted model to list of models
            pr_s = treat_s_model.predict(data_to_predict)                # generate predicted probabilities of As=1

            # If verbose requested, provide model output
            if self._verbose_:
                print('==============================================================================')
                print(verbose_label+': '+self._gs_measure_)
                print(treat_s_model.summary())

        elif distribution == 'threshold':                                # If distribution is a threshold
            gs_model = self._gs_measure_ + ' ~ ' + self._gs_model        # setup the model form
            if self._gs_custom_ is None:                                 # if no custom model is given
                f = sm.families.family.Binomial()                        # ... logistic model
                treat_s_model = smf.glm(gs_model,                        # Estimate the model
                                        data_to_fit,                     # ... with data to fit
                                        family=f).fit()                  # ... and logistic model
                if store_model:                                          # If estimating the denominator
                    self._treatment_models.append(treat_s_model)         # ... add fitted model to list of models
                pred = treat_s_model.predict(data_to_predict)            # Generate predicted values of As=threshold

                # If verbose requested, provide model output
                if self._verbose_:
                    print('==============================================================================')
                    print('g-model: '+self._gs_measure_)
                    print(treat_s_model.summary())
            else:                                                                 # Else custom model for threshold
                if self.use_deep_learner_A_i_s:
                    data_to_fit_subset = select_pooled_sample_with_observed_data(self._gs_measure_, data_to_fit, data_to_predict)
                    print(f'gs_model: use {data_to_fit_subset.shape[0]} samples from original {data_to_fit.shape[0]} to fit the model')
                    xdata = patsy.dmatrix(self._gs_model + ' - 1', 
                                          data_to_fit_subset, return_type="dataframe")       # Extract via patsy the data
                    ydata = data_to_fit_subset[self._gs_measure_]
                    n_output = pd.unique(ydata).shape[0] 
                    print(f'gs_model: n_output = {n_output} for target variable {self._gs_measure_}')

                    pdata = patsy.dmatrix(self._gs_model + ' - 1', 
                                          data_to_predict, return_type="dataframe")   # Extract via patsy the data
                    pdata_y = data_to_predict[self._gs_measure_]
                    custom_path = custom_path_prefix + 'A_i_s_' + self.exposure  + '.pth'
                    pred = exposure_deep_learner(self._gs_custom_, 
                                                 xdata, ydata, pdata, pdata_y, self._gs_measure_,
                                                 self.adj_matrix, self.cat_vars, self.cont_vars, self.cat_unique_levels, n_output,
                                                 custom_path, **kwargs)
                else:
                    xdata = patsy.dmatrix(self._gs_model + ' - 1', data_to_fit)       # Processing data to be fit
                    pdata = patsy.dmatrix(self._gs_model + ' - 1', data_to_predict)   # Processing data to be fit
                    pred = exposure_machine_learner(ml_model=self._gs_custom_,        # Estimating the ML
                                                    xdata=np.asarray(xdata),          # ... with data to fit
                                                    ydata=np.asarray(data_to_fit[self._gs_measure_]),
                                                    pdata=np.asarray(xdata))          # ... and data to predict
            pr_s = np.where(data_to_predict[self._gs_measure_] == 1,              # Getting predicted values
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


# TODO: 
# initiate NetworkTMLETimeSeries to check on if self.df_restricted_list works as expected
# use outcome model as a examples, re-define the dataset object, which kind of y to use, time series or single ?
# model: how to apply GCN? 
# consider how to modify AbstractML

# load uniform vaccine network
from beowulf import load_uniform_vaccine        

# load network_list
from Beowulf.beowulf.dgm.vaccine_with_cat_cont_split import vaccine_dgm_time_series

n_nodes = 500
restrict = False
exposure = "vaccine"
outcome = "D"
degree_restrict = None


G, cat_vars, cont_vars, cat_unique_levels = load_uniform_vaccine(n=n_nodes, return_cat_cont_split=True)
H, cat_vars, cont_vars, cat_unique_levels, network_list = vaccine_dgm_time_series(network=G, restricted=restrict, 
                                                            update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)

ntmle = NetworkTMLETimeSeries(network_list, exposure=exposure, outcome=outcome, degree_restrict=degree_restrict,
                              cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)

ntmle.df_restricted_list

################## inside outcome_model(): if self.use_deep_learner_outcome: ################## 
model = "vaccine + vaccine_sum + A + H + A_sum + H_sum + degree"
custom_model = None


xdata_list = []
ydata_list = []
n_output_list = []

for df_restricted in ntmle.df_restricted_list:
    xdata_list.append(patsy.dmatrix(model + ' - 1', df_restricted, return_type="dataframe"))
    ydata_list.append(df_restricted[ntmle.outcome])
    # ydata.append(self.df_restricted_list[-1][self.outcome])
    # n_output = pd.unique(ydata).shape[0]
    n_output_list.append(pd.unique(df_restricted[ntmle.outcome]).shape[0])
    custom_path = 'outcome_' + ntmle.outcome + '.pth'
    ntmle._q_custom_ = custom_model


from tmle_utils import get_model_cat_cont_split_patsy_matrix, append_target_to_df

aa = []
bb = []
cc = []
for xdata in xdata_list:
    model_cat_vars, model_cont_vars, model_cat_unique_levels, cat_vars, cont_vars, cat_unique_levels = get_model_cat_cont_split_patsy_matrix(xdata, 
                                                                                                                                            cat_vars, cont_vars, cat_unique_levels)
    aa.append(model_cat_vars)
    bb.append(model_cont_vars)
    cc.append(model_cat_unique_levels)

# def check_identical(list):
    
#     return len(set(list)) == 1

from itertools import groupby

def all_equal(iterable):
    '''check if all elements in a list are identical'''
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def update_cat_var_unique_levels(cat_var_unique_levels_list):
    model_cat_unique_levels_final = {} 
    for i, cat_var_level_dict in enumerate(cat_var_unique_levels_list):
        for var_name, num_levels in cat_var_level_dict.items():
            if i == 0:
                model_cat_unique_levels_final[var_name] = num_levels
            else:
                if num_levels > model_cat_unique_levels_final[var_name]:
                    model_cat_unique_levels_final[var_name] = num_levels 
    return model_cat_unique_levels_final


if not all_equal(aa):
    raise ValueError("cat_vars are not identical throughout time slices")
else:
    model_cat_vars_final = aa[-1]

if not all_equal(bb):
    raise ValueError("cont_vars are not identical throughout time slices")
else:
    model_cont_vars_final = bb[-1]

if not all_equal(cc):
    print(f'cat_vars levels are not identical througout time slices:')
    print(cc)
    print(f'adopt the maximum levels for each variable:')
    model_cat_unique_levels_final = update_cat_var_unique_levels(cc) 
    print(model_cat_unique_levels_final)
else:
    model_cat_unique_levels_final = cc[-1]




# model_cat_unique_levels_final = update_cat_var_unique_levels(cc)
# model_cat_unique_levels_final

deep_learner_df_list = []
for xdata, ydata in zip(xdata_list, ydata_list):
    deep_learner_df_list.append(append_target_to_df(ydata, xdata, ntmle.outcome))

len(deep_learner_df_list)


import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, patsy_matrix_dataframe_list, target=None, use_all_time_slices=True, 
                 model_cat_vars=[], model_cont_vars=[], model_cat_unique_levels={}):
        
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

ts_dset = TimeSeriesDataset(deep_learner_df_list, ntmle.outcome, use_all_time_slices=True,
                            model_cat_vars=model_cat_vars_final, 
                            model_cont_vars=model_cont_vars_final, 
                            model_cat_unique_levels=model_cat_unique_levels_final)

ts_dset = TimeSeriesDataset(deep_learner_df_list, ntmle.outcome, use_all_time_slices=False,
                            model_cat_vars=model_cat_vars_final, 
                            model_cont_vars=model_cont_vars_final, 
                            model_cat_unique_levels=model_cat_unique_levels_final)

cat_dummy, cont_dummy, label_dummy, idx_dummy = ts_dset.__getitem__(0)

len(ts_dset)

cat_dummy.shape
cont_dummy.shape
label_dummy.shape
label_dummy
idx_dummy.shape


loader = DataLoader(ts_dset, batch_size=4, shuffle=True)


for cat_vars, cont_vars, labels, idices in loader:
    print(cat_vars.shape)
    print(cont_vars.shape)
    print(labels.shape)
    print(idices.shape)
    break


from dl_models import MLPModel
mlp_model = MLPModel(None, model_cat_unique_levels_final, n_cont=len(model_cont_vars), 
                    n_output=2, _continuous_outcome=False)

import torch.nn as nn
import torch.nn.functional as F


class MLPModelTimeSeries(nn.Module):
    def __init__(self, adj_matrix, model_cat_unique_levels, n_cont, T=10,
                 n_output=2, _continuous_outcome=False):
        super().__init__()
        self.embedding_layers, self.n_emb = self._get_embedding_layers(model_cat_unique_levels)
        self.n_cont = n_cont

        # variable dimension feature extraction
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 16)
        self.lin2 = nn.Linear(16, 32)
        # if use BCEloss, number of output should be 1, i.e. the probability of getting category 1
        # else number of output should be as specified
        if n_output == 2 or _continuous_outcome:
            self.lin3 = nn.Linear(32, 1) 
        else:
            self.lin3 = nn.Linear(32, n_output)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

        # time dimension feature extract 
        self.ts_lin1 = nn.Linear(T, 16)
        self.ts_lin2 = nn.Linear(16, T)


    def _get_embedding_layers(self, model_cat_unique_levels):
        # Ref: https://jovian.ml/aakanksha-ns/shelter-outcome
        # decide embedding sizes
        embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _, n_categories in model_cat_unique_levels.items()]
        embedding_layers = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in embedding_layers) # length of all embeddings combined
        # n_cont = dataset.x_cont.shape[1] # number of continuous variables

        return embedding_layers, n_emb
    
    def forward(self, x_cat, x_cont, batched_nodes_indices=None):
        # x_cat: [batch_size, num_cat_vars, T]
        # x_cont: [batch_size, num_cont_vars, T]
        # batched_nodex_indices: [batch_size]

        x_cat_new = x_cat.permute(0, 2, 1)
        # x_cat_new: [batch_size, T, num_cat_vars]

        if len(self.embedding_layers) > 0: # if there are categorical variables to be encoded
            x1 = [e(x_cat_new[:, :, 1]) for i, e in enumerate(self.embedding_layers)]
            x1 = torch.cat(x1, -1) # [batch_size, T, n_emb]
            x1 = self.emb_drop(x1)

        if self.n_cont > 0: # if there are continuous variables to be encoded
            x2 = self.bn1(x_cont).permute(0, 2, 1) # [batch_size, T, n_cont]
        
        if len(self.embedding_layers) > 0 and self.n_cont > 0: # if there are both categorical and continuous variables to be encoded 
            x = torch.cat([x1, x2], -1) # [batch_size, T, n_emb + n_cont]
            # temporal perspective
            x = F.relu(self.ts_lin1(x.permute(0, 2, 1))).permute(0, 2, 1) 
            # [batch_size, T, n_emb + n_cont] -> [batch_size, n_emb + n_cont, T] 
            # -> [batch_size, n_emb + n_cont, 16] -> [batch_size, 16, n_emb + n_cont]

            # variable perspective
            x = F.relu(self.lin1(x)) # [batch_size, 16, n_emb + n_cont] -> [batch_size, 16, 16]
            x = self.drops(x)       
            x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1) 
            # [batch_size, 16(ts_c), 16(v_c)] -> [batch_size, 16(v_c), 16(ts_c)] ->  [batch_size, 16(ts_c), 16(v_c)]
            x = F.relu(self.lin2(x)) # [batch_size, 16, 16] -> [batch_size, 16, 32] 
            x = self.drops(x)
            x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
            # [batch_size, 16, 32] -> [batch_size, 32, 16] -> [batch_size, 16, 32
            x = self.lin3(x)         # [batch_size, 16, 32] -> [batch_size, 16, 1]

            # temporal perspective
            x = self.ts_lin2(x.permute(0, 2, 1))
            # [batch_size, 16, 1] -> [batch_size, 1, 16] -> [batch_size, 1, T] 

        elif len(self.embedding_layers) > 0 and self.n_cont == 0: 
            # temporal perspective
            x = self.ts_lin1(x1.permute(0, 2, 1)).permute(0, 2, 1)
            # variable perspective
            x = F.relu(self.lin1(x))
            x = self.drops(x)       
            x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = F.relu(self.lin2(x))
            x = self.drops(x)
            x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.lin3(x)
            # temporal perspective
            x = self.ts_lin2(x.permute(0, 2, 1))

        elif len(self.embedding_layers) == 0 and self.n_cont > 0:
            # temporal perspective
            x = self.ts_lin1(x2.permute(0, 2, 1)).permute(0, 2, 1)
            # variable perspective
            x = F.relu(self.lin1(x))
            x = self.drops(x)       
            x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = F.relu(self.lin2(x))
            x = self.drops(x)
            x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.lin3(x)
            # temporal perspective
            x = self.ts_lin2(x.permute(0, 2, 1))
        else:
            raise ValueError('No variables to be encoded')
    
        return x

mlp_model = MLPModelTimeSeries(None, model_cat_unique_levels_final, n_cont=len(model_cont_vars), 
                    n_output=2, _continuous_outcome=False)

class CNNModelTimeSeries(nn.Module):
    def __init__(self, adj_matrix, model_cat_unique_levels, n_cont, T=10,
                 n_output=2, _continuous_outcome=False):
        super().__init__()
        self.embedding_layers, self.n_emb = self._get_embedding_layers(model_cat_unique_levels)
        self.n_cont = n_cont

        # conv layers
        self.conv1 = nn.Conv1d(in_channels=self.n_emb + self.n_cont, out_channels=16, 
                               kernel_size=5, padding='same')
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, 
                               kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, 
                               kernel_size=5, padding='same')  

        # bn layers
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)

        # dropout layers
        self.emb_drop = nn.Dropout(0.6)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)

    def _get_embedding_layers(self, model_cat_unique_levels):
        # Ref: https://jovian.ml/aakanksha-ns/shelter-outcome
        # decide embedding sizes
        embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _, n_categories in model_cat_unique_levels.items()]
        embedding_layers = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in embedding_layers) # length of all embeddings combined
        # n_cont = dataset.x_cont.shape[1] # number of continuous variables

        return embedding_layers, n_emb
    
    def forward(self, x_cat, x_cont, batched_nodes_indices=None):
        # x_cat: [batch_size, num_cat_vars, T]
        # x_cont: [batch_size, num_cont_vars, T]
        # batched_nodex_indices: [batch_size]

        x_cat_new = x_cat.permute(0, 2, 1)
        # x_cat_new: [batch_size, T, num_cat_vars]

        if len(self.embedding_layers) > 0: # if there are categorical variables to be encoded
            x1 = [e(x_cat_new[:, :, 1]) for i, e in enumerate(self.embedding_layers)]
            x1 = torch.cat(x1, -1) # [batch_size, T, n_emb]
            x1 = self.emb_drop(x1)

        if self.n_cont > 0: # if there are continuous variables to be encoded
            x2 = self.bn1(x_cont).permute(0, 2, 1) # [batch_size, T, n_cont]
        
        if len(self.embedding_layers) > 0 and self.n_cont > 0: # if there are both categorical and continuous variables to be encoded 
            x = torch.cat([x1, x2], -1).permute(0, 2, 1) # [batch_size, T, n_emb + n_cont] -> [bathc_size, n_emb + n_cont, T]
            x = F.relu(self.bn2(self.conv1(x))) # [batch_size, 16, T] 
            x = self.drop1(x)
            x = F.relu(self.bn3(self.conv2(x))) # [batch_size, 32, T]
            x = self.drop2(x)
            x = self.conv3(x) # [batch_size, 1, T]

        elif len(self.embedding_layers) > 0 and self.n_cont == 0: 
            x = F.relu(self.bn2(self.conv1(x1.permute(0, 2, 1)))) # [batch_size, T, n_emb] -> [batch_size, n_emb, T] -> [batch_size, 16, T] 
            x = self.drop1(x)
            x = F.relu(self.bn3(self.conv2(x))) # [batch_size, 32, T]
            x = self.drop2(x)
            x = self.conv3(x) # [batch_size, 1, T]

        elif len(self.embedding_layers) == 0 and self.n_cont > 0:
            x = F.relu(self.bn2(self.conv1(x2.permute(0, 2, 1))))  # [batch_size, T, n_cont] -> [batch_size, n_cont, T] -> [batch_size, 16, T] 
            x = self.drop1(x)
            x = F.relu(self.bn3(self.conv2(x))) # [batch_size, 32, T]
            x = self.drop2(x)
            x = self.conv3(x) # [batch_size, 1, T]
        else:
            raise ValueError('No variables to be encoded')
    
        return x


cnn_model = CNNModelTimeSeries(None, model_cat_unique_levels_final, n_cont=len(model_cont_vars), 
                               n_output=2, _continuous_outcome=False)


for cat_vars, cont_vars, labels, indices in loader:
    # out = mlp_model(cat_vars, cont_vars)
    out = cnn_model(cat_vars, cont_vars)
    break

out[0].shape

out.shape


mlp_model.embedding_layers

for i, e in enumerate(mlp_model.embedding_layers):
    print(cat_vars[:, i, :].permute(0, 2, 1).shape)

dummy_A = torch.rand(4, 4)
dummy_x = torch.rand(4, 16, 10).permute(1, 0, 2)
out = torch.matmul(dummy_A, dummy_x)
out.shape