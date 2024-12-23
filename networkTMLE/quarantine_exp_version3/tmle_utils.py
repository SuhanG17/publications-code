import warnings
import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm
import patsy
from itertools import groupby
from sklearn.utils.class_weight import compute_class_weight

def probability_to_odds(prob):
    """Converts given probability (proportion) to odds

    Parameters
    ----------
    prob : float, array
        Probability or array of probabilities to convert to odds
    """
    return prob / (1 - prob)


def odds_to_probability(odds):
    """Converts given odds to probability

    Parameters
    ----------
    odds : float, array
        Odds or array of odds to convert to probabilities
    """
    return odds / (1 + odds)


def exp_map(graph, var):
    """Slow implementation of the exposure mapping functionality. Only supports the sum summary measure.
    Still used by the dgm files.

    Note
    ----
    Depreciated and no longer actively used by any functions.

    Parameters
    ----------
    graph : networkx.Graph
        Network to calculate the summary measure for.
    var : str
        Variable in the graph to calculate the summary measure for

    Returns
    -------
    array
        One dimensional array of calculated summary measure
    """
    # get adjacency matrix
    matrix = nx.adjacency_matrix(graph, weight=None)
    # get node attributes
    y_vector = np.array(list(nx.get_node_attributes(graph, name=var).values()))
    # multiply the weight matrix by node attributes
    wy_matrix = np.nan_to_num(matrix * y_vector.reshape((matrix.shape[0]), 1)).flatten()
    return np.asarray(wy_matrix).flatten()  # I hate converting between arrays and matrices...


def fast_exp_map(matrix, y_vector, measure):
    r"""Improved (computation-speed-wise) implementation of the exposure mapping functionality. Further supports a
    variety of summary measures. This is accomplished by using the adjacency matrix and vectors to efficiently
    calculate the summary measures (hence the function name). This is an improvement on previous iterations of this
    function.

    Available summary measures are

    Sum (``'sum'``) :

    .. math::

        X_i^s = \sum_{j=1}^n X_j \mathcal{G}_{ij}

    Mean (``'mean'``) :

    .. math::

        X_i^s = \sum_{j=1}^n X_j \mathcal{G}_{ij} / \sum_{j=1}^n \mathcal{G}_{ij}

    Variance (``'var'``):

    .. math::

        \bar{X}_j = \sum_{j=1}^n X_j \mathcal{G}_{ij} \\
        X_i^s = \sum_{j=1}^n (X_j - \bar{X}_j)^2 \mathcal{G}_{ij} / \sum_{j=1}^n \mathcal{G}_{ij}

    Mean distance (``'mean_dist'``) :

    .. math::

        X_i^s = \sum_{j=1}^n (X_i - X_j) \mathcal{G}_{ij} / \sum_{j=1}^n \mathcal{G}_{ij}

    Variance distance (``'var_dist'``) :

    .. math::

        \bar{X}_{ij} = \sum_{j=1}^n (X_i - X_j) \mathcal{G}_{ij} \\
        X_i^s = \sum_{j=1}^n ((X_j - X_j) - \bar{X}_{ij})^2 \mathcal{G}_{ij} / \sum_{j=1}^n \mathcal{G}_{ij}

    Note
    ----
    If you would like other summary measures to be added or made available, please reach out via GitHub.

    Parameters
    ----------
    matrix : array
        Adjacency matrix. Should be extract from a ``networkx.Graph`` via ``nx.adjacency_matrix(...)``
    y_vector : array
        Array of the variable to calculate the summary measure for. Should be in same order as ``matrix`` for
        calculation to work as intended.
    measure : str
        Summary measure to calculate. Options are provided above.

    Returns
    -------
    array
        One dimensional array of calculated summary measure
    """
    if measure.lower() == 'sum':
        # multiply the weight matrix by node attributes
        wy_matrix = np.nan_to_num(matrix * y_vector.reshape((matrix.shape[0]), 1)).flatten()
        return np.asarray(wy_matrix).flatten()         # converting between arrays and matrices...
    elif measure.lower() == 'mean':
        rowsum_vector = np.sum(matrix, axis=1)         # calculate row-sum (denominator / degree)
        with warnings.catch_warnings():                # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            weight_matrix = matrix / rowsum_vector.reshape((matrix.shape[0]), 1)  # calculate each nodes weight
        wy_matrix = weight_matrix * y_vector.reshape((matrix.shape[0]), 1)  # multiply matrix by node attributes
        return np.asarray(wy_matrix).flatten()         # converting between arrays and matrices...
    elif measure.lower() == 'var':
        a = matrix.toarray()                           # Convert matrix to array
        a = np.where(a == 0, np.nan, a)                # filling non-edges with NaN's
        with warnings.catch_warnings():                # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanvar(a * y_vector, axis=1)
    elif measure.lower() == 'mean_dist':
        a = matrix.toarray()                           # Convert matrix to array
        a = np.where(a == 0, np.nan, a)                # filling non-edges with NaN's
        c = (a * y_vector).transpose() - y_vector      # Calculates the distance metric (needs transpose)
        with warnings.catch_warnings():                # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmean(c.transpose(),           # back-transpose
                              axis=1)
    elif measure.lower() == 'var_dist':
        a = matrix.toarray()                           # Convert matrix to array
        a = np.where(a == 0, np.nan, a)                # filling non-edges with NaN's
        c = (a * y_vector).transpose() - y_vector      # Calculates the distance metric (needs transpose)
        with warnings.catch_warnings():                # ignores NumPy's RuntimeWarning for isolated nodes (divide by 0)
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanvar(c.transpose(),            # back-transpose
                             axis=1)
    else:
        raise ValueError("The summary measure mapping" + str(measure) + "is not available")


def exp_map_individual(network, variable, max_degree):
    """Summary measure calculate for the non-parametric mapping approach described in Sofrygin & van der Laan (2017).
    This approach works best for networks with uniform degree distributions. This summary measure generates a number
    of columns (a total of ``max_degree``). Each column is then an indicator variable for each observation. To keep
    all columns the same number of dimensions, zeroes are filled in for all degrees above unit i's observed degree.

    Parameters
    ----------
    network : networkx.Graph
        The NetworkX graph object to calculate the summary measure for.
    variable : str
        Variable to calculate the summary measure for (this will always be the exposure variable internally).
    max_degree : int
        Maximum degree in the network (defines the number of columns to generate).

    Returns
    -------
    dataframe
        Data set containing all generated columns
    """
    attrs = []
    for i in network.nodes:
        j_attrs = []
        for j in network.neighbors(i):
            j_attrs.append(network.nodes[j][variable])
        attrs.append(j_attrs[:max_degree])

    return pd.DataFrame(attrs,
                        columns=[variable+'_map'+str(x+1) for x in range(max_degree)])


def network_to_df(graph):
    """Take input network and converts all node attributes to a pandas DataFrame object. This dataframe is then used
    within ``NetworkTMLE`` internally.

    Parameters
    ----------
    graph : networkx.Graph
        Graph with node attributes to transform into data set

    Returns
    -------
    dataframe
        Data set containing all node attributes
    """
    return pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')


def bounding(ipw, bound):
    """Internal function to bound or truncate the estimated inverse probablity weights.

    Parameters
    ----------
    ipw : array
        Estimate inverse probability weights to truncate.
    bound : list, float, int, set, array
        Bounds to truncate weights by.

    Returns
    -------
    array
        Truncated inverse probability weights.
    """
    if type(bound) is float or type(bound) is int:  # Symmetric bounding
        if bound > 1:
            ipw = np.where(ipw > bound, bound, ipw)
            ipw = np.where(ipw < 1 / bound, 1 / bound, ipw)
        elif 0 < bound < 1:
            ipw = np.where(ipw < bound, bound, ipw)
            ipw = np.where(ipw > 1 / bound, 1 / bound, ipw)
        else:
            raise ValueError('Bound must be a positive value')
    elif type(bound) is str:  # Catching string inputs
        raise ValueError('Bounds must either be a float or integer, or a collection')
    else:  # Asymmetric bounds
        if bound[0] > bound[1]:
            raise ValueError('Bound thresholds must be listed in ascending order')
        if len(bound) > 2:
            warnings.warn('It looks like your specified bounds is more than two floats. Only the first two '
                          'specified bounds are used by the bound statement. So only ' +
                          str(bound[0:2]) + ' will be used', UserWarning)
        if type(bound[0]) is str or type(bound[1]) is str:
            raise ValueError('Bounds must be floats or integers')
        if bound[0] < 0 or bound[1] < 0:
            raise ValueError('Both bound values must be positive values')
        ipw = np.where(ipw < bound[0], bound[0], ipw)
        ipw = np.where(ipw > bound[1], bound[1], ipw)
    return ipw


def outcome_learner_fitting(ml_model, xdata, ydata):
    """Internal function to fit custom_models for the outcome nuisance model.

    Parameters
    ----------
    ml_model :
        Unfitted model to be fit.
    xdata : array
        Covariate data to fit the model with
    ydata : array
        Outcome data to fit the model with

    Returns
    -------
    Fitted user-specified model
    """
    try:
        fm = ml_model.fit(X=xdata, y=ydata)
    except TypeError:
        raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                        "covers both sklearn and supylearner. If there is a predictive model you would "
                        "like to use, please open an issue at https://github.com/pzivich/zepid and I "
                        "can work on adding support")
    return fm


def outcome_learner_predict(ml_model_fit, xdata):
    """Internal function to take a fitted custom_model for the outcome nuisance model and generate the predictions.

    Parameters
    ----------
    ml_model_fit :
        Fitted user-specified model
    xdata : array
        Covariate data to generate the predictions with.

    Returns
    -------
    array
        Predicted values for the outcome (probability if binary, and expected value otherwise)
    """
    if hasattr(ml_model_fit, 'predict_proba'):
        g = ml_model_fit.predict_proba(xdata)
        if g.ndim == 1:  # allows support for pygam.LogisticGAM
            return g
        else:
            return g[:, 1]
    elif hasattr(ml_model_fit, 'predict'):
        return ml_model_fit.predict(xdata)
    else:
        raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")


def exposure_machine_learner(ml_model, xdata, ydata, pdata):
    """Internal function to fit custom_models for the exposure nuisance model and generate the predictions.

    Parameters
    ----------
    ml_model :
        Unfitted model to be fit.
    xdata : array
        Covariate data to fit the model with
    ydata : array
        Outcome data to fit the model with
    pdata : array
        Covariate data to generate the predictions with.

    Returns
    -------
    array
        Predicted values for the outcome (probability if binary, and expected value otherwise)
    """
    # Fitting model
    try:
        fm = ml_model.fit(X=xdata, y=ydata)
    except TypeError:
        raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                        "covers both sklearn and supylearner. If there is a predictive model you would "
                        "like to use, please open an issue at https://github.com/pzivich/zepid and I "
                        "can work on adding support")

    # Generating predictions
    if hasattr(fm, 'predict_proba'):
        g = fm.predict_proba(pdata)
        if g.ndim == 1:  # allows support for pygam.LogisticGAM
            return g
        else:
            return g[:, 1]
    elif hasattr(fm, 'predict'):
        g = fm.predict(pdata)
        return g
    else:
        raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")


# def targeting_step(y, q_init, ipw, verbose):
#     r"""Estimate :math:`\eta` via the targeting model

#     Parameters
#     ----------
#     y : array
#         Observed outcome values.
#     q_init : array
#         Predicted outcome values under the observed values of exposure.
#     ipw : array
#         Estimated inverse probability weights.
#     verbose : bool
#         Whether to print the summary details of the targeting model.

#     Returns
#     -------
#     float
#         Estimated value to use to target the outcome model predictions
#     """
#     f = sm.families.family.Binomial()
#     log = sm.GLM(y,  # Outcome / dependent variable
#                  np.repeat(1, y.shape[0]),  # Generating intercept only model
#                  offset=np.log(probability_to_odds(q_init)),  # Offset by g-formula predictions
#                  freq_weights=ipw,  # Weighted by calculated IPW
#                  family=f).fit(maxiter=500)

#     if verbose:  # Optional argument to print each intermediary result
#         print('==============================================================================')
#         print('Targeting Model')
#         print(log.summary())

#     return log.params[0]  # Returns single-step estimated Epsilon term

def targeting_step(y, q_init, ipw, verbose):
    """Bound q_init values with (0.05, 0.95) to avoid zero division under probability_to_odds() """
    q_init = np.clip(q_init, 0.005, 0.995)
    
    f = sm.families.family.Binomial()
    log = sm.GLM(y,  # Outcome / dependent variable
                 np.repeat(1, y.shape[0]),  # Generating intercept only model
                 offset=np.log(probability_to_odds(q_init)),  # Offset by g-formula predictions
                 freq_weights=ipw,  # Weighted by calculated IPW
                 family=f).fit(maxiter=500)

    if verbose:  # Optional argument to print each intermediary result
        print('==============================================================================')
        print('Targeting Model')
        print(log.summary())

    return log.params[0]  # Returns single-step estimated Epsilon term


def tmle_unit_bounds(y, mini, maxi):
    """Bounding for continuous outcomes for TMLE.

    Parameters
    ----------
    y : array
        Observed outcome values
    mini : float
        Lower bound to apply
    maxi : float
        Upper bound to apply

    Returns
    -------
    array
        Bounded outcomes
    """
    return (y - mini) / (maxi - mini)


def tmle_unit_unbound(ystar, mini, maxi):
    """Unbound the bounded continuous outcomes for presentation of results.

    Parameters
    ----------
    ystar : array
        Bounded outcome values
    mini : float
        Lower bound to apply
    maxi : float
        Upper bound to apply

    Returns
    -------
    array
        Unbounded outcomes
    """
    return ystar*(maxi - mini) + mini


# def create_threshold(data, variables, thresholds):
#     """Internal function to create threshold variables given setup information.

#     Parameters
#     ----------
#     data : dataframe
#         Data set to calculate the measure for
#     variables : list, set
#         List of variable names to create the threshold variables for
#     thresholds : list, set
#         List of values (float or int) to create the thresholds at.

#     Returns
#     -------
#     None
#     """
#     for v, t in zip(variables, thresholds):
#         if type(t) is float:
#             label = v + '_t' + str(int(t * 100))
#         else:
#             label = v + '_t' + str(t)
#         data[label] = np.where(data[v] > t, 1, 0)

def create_threshold(data, variables, thresholds, definitions, graph=None):
    """Internal function to create threshold variables given setup information.

    Parameters
    ----------
    data : dataframe
        Data set to calculate the measure for
    variables : list, set
        List of variable names to create the threshold variables for
    thresholds : list, set
        List of values (float or int) to create the thresholds at.
    definitions: list, set
        list of summary definitions for each threshold variable

    Returns
    -------
    None
    """
    for v, t, d in zip(variables, thresholds, definitions):
        if type(t) is float:
            label = v + '_t' + str(int(t * 100))
        else:
            label = v + '_t' + str(t)
        data[label] = np.where(data[v + '_' + d] > t, 1, 0)
        if graph is not None:
            if nx.is_directed(graph):
                raise NotImplementedError("Directed graph is not supported yet")
            else:
                nx.set_node_attributes(graph, dict(data[label]), label)

def create_categorical(data, variables, bins, labels, verbose=False, graph=None):
    """

    Parameters
    ----------
    data : dataframe
        Data set to calculate the measure for
    variables : list, set
        List of variable names to create the threshold variables for
    bins : list, set
        List of lists of values (float or int) to create bins at.
    labels : list, set
        List of lists of labels (str) to apply as the new column names
    verbose : bool, optional
        Whether to warn the user if any NaN values occur (a result of bad or incompletely specified bins). Interally,
        this option is always set to be True (since important for user to recognize this issue).

    Returns
    -------
    None
    """
    for v, b, l in zip(variables, bins, labels):
        col_label = v + '_c'
        data[col_label] = pd.cut(data[v],
                                 bins=b,
                                 labels=l,
                                 include_lowest=True).astype(float)
        if graph is not None:
            if nx.is_directed(graph):
                raise NotImplementedError("Directed graph is not supported yet")
            else:
                nx.set_node_attributes(graph, dict(data[col_label]), col_label)
        if verbose:
            if np.any(data[col_label].isna()):
                warnings.warn("It looks like some of your categories have missing values when being generated on the "
                              "input data. Please check pandas.cut to make sure the `bins` and `labels` arguments are "
                              "being used correctly.", UserWarning)
                

######################################### SG_modified: Deep Learning Related Functions #########################################
def get_patsy_for_model_w_C(model_string, df_restricted_dataframe):
    ''' For model string with C() terms, return patsy matrix dataframe with C() terms
    The naive way of creating C() variables with patsy.dmatrix() will not work 
    because it will create one-hot columns for each level of C() var
    Hence, we re-define C() in dataframe for deep learners
    '''
    model_elem = model_string.split('+')
    model_exclude_c = []
    model_c = []
    model_c_id = []
    for id, item in enumerate(model_elem):
        if 'C(' not in item:
            model_exclude_c.append(item)
        else:
            # remove space in C() 
            # select only the var name in C()
            var_name = item.replace(" ", "").split('(')[1].split(')')[0] 
            model_c.append(var_name)
            model_c_id.append(id)

    model_exclude_c = ' + '.join(model_exclude_c) 
    patsy_matrix_dataframe = patsy.dmatrix(model_exclude_c + ' - 1', df_restricted_dataframe, return_type="dataframe") 
    for id, item in zip(model_c_id, model_c):
        patsy_matrix_dataframe.insert(id, 'C(' + item + ')', df_restricted_dataframe[item])
    return patsy_matrix_dataframe


# call with patsy_matrix_dataframe=xdata
def get_model_cat_cont_split_patsy_matrix(patsy_matrix_dataframe, cat_vars, cont_vars, cat_unique_levels):
    '''initiate model_car_vars, model_cont_vars, and cat_unique_levles, and
    update cat_vars, cont_vars, cat_unique_levels based on patsy matrix dataframe'''

    vars = patsy_matrix_dataframe.columns # all variables in patsy matrix
    # print(vars)

    model_cat_vars = []
    model_cont_vars = []
    model_cat_unique_levels = {}

    for var in vars:
        if var in cat_vars:
            model_cat_vars.append(var)
            model_cat_unique_levels[var] = cat_unique_levels[var]
        elif var in cont_vars:
            model_cont_vars.append(var)
        else:
            # update both model_{} and universal cat_vars, cont_vars adn cat_unique_levels to keep track of all variables
            if '**' in var: # quadratic term, treated as continuous
                model_cont_vars.append(var)
                cont_vars.append(var)
            elif 'C(' in var: # categorical term newly defined for model == 'np'
                model_cat_vars.append(var)
                model_cat_unique_levels[var] = pd.unique(patsy_matrix_dataframe[var].astype('int')).max() + 1
                cat_vars.append(var)
                cat_unique_levels[var] = pd.unique(patsy_matrix_dataframe[var].astype('int')).max() + 1
            elif '_t' in var: # threshold term, treated as categorical
                model_cat_vars.append(var)
                model_cat_unique_levels[var] = pd.unique(patsy_matrix_dataframe[var].astype('int')).max() + 1 
                cat_vars.append(var)
                cat_unique_levels[var] = pd.unique(patsy_matrix_dataframe[var].astype('int')).max() + 1
            elif ':' in var: # interaction term, treated as continuous even between two categorical variables
                model_cont_vars.append(var)
                cont_vars.append(var)
            else:
                raise ValueError(f'{var} is a unseen type of variable, cannot be assigned to categorical or continuous')
    return model_cat_vars, model_cont_vars, model_cat_unique_levels, cat_vars, cont_vars, cat_unique_levels

# call before feed dataframe to the deep learning fitter: xdata and ydata should be in one dataframe
def append_target_to_df(ydata, patsy_matrix_dataframe, target):
    patsy_matrix_dataframe[target] = ydata
    return patsy_matrix_dataframe

def get_probability_from_multilevel_prediction(pred:np.ndarray, target:pd.core.series.Series):
    ''' get the predicted probability for the target category '''
    sample_dim_indices = np.arange(pred.shape[0])[:, np.newaxis] #[num_samples] -> [num_samples, 1]
    class_indices = target.to_numpy(dtype='int')[:, np.newaxis] #pd.series -> np.array -> [num_samples, 1]
    pred = pred[sample_dim_indices, class_indices].squeeze(-1) #[num_samples, n_output] -> [num_samples, 1] -> [num_sample]
    return pred

# def get_n_output_for_summary_variable(gs_measure, max_degree, exposure_series:pd.core.series.Series):
#     ''' get the all possible categorical levels for A_i^s model, 
#     because summary measure levels may change at _generate_pooled_sample() '''

#     if 'sum' in gs_measure: # use sum as summary measure
#         max_level = pd.unique(exposure_series).max()
#         n_output = max_degree*max_level # maximum number of possible neighbors * maximum level = all possible categorical levels
#     return n_output

def check_pooled_sample_levels(target, pooled_data, observed_data):
    ''' check if pooled_data contains all levels of target in observed_data, 
    if not, return True regeneration flag; 
    if true, return False regeneartion flag '''

    pooled_data_levels = pd.unique(pooled_data[target])
    observed_data_levels = pd.unique(observed_data[target])
    print(f'pooled_data_levels: {pooled_data_levels}')
    print(f'observed_data_levels: {observed_data_levels}')
    if set(observed_data_levels).issubset(set(pooled_data_levels)):
        return False # regenerate_flag=False
    else:
        return True # regenerate_flag=True


def select_pooled_sample_with_observed_data(target, pooled_data, observed_data):
    ''' select the pooled sample with observed data target level'''

    pooled_subset = pooled_data.copy()
    observed_levels = observed_data[target]
    pooled_subset = pooled_subset.loc[pooled_subset[target].isin(observed_levels)]
    return pooled_subset


def exposure_deep_learner(deep_learner, xdata, ydata, pdata, pdata_y, exposure,
                          adj_matrix, cat_vars, cont_vars, cat_unique_levels, n_output, 
                          custom_path, **kwargs):
    """Internal function to fit custom_models for the exposure nuisance model and generate the predictions.

    Parameters
    ----------
    deep_learner :
        instance of the deep learning model
    xdata : pd.dataframe
        Covariate data to fit the model with
    ydata : pandas.core.series.Series
        Outcome data to fit the model with
    pdata : pd.dataframe
        Covariate data to generate the predictions with.
    pdata_y : pandas.core.series.Series
        Truth for predictions, used to evaluate model performance
    exposure: string
        Exposure patamerter to predict
    adj_matrix: SciPy sparse array
        adjacency matrix for GCN model
    cat_vars: list
        list of categorical variables for df_restricted, not xdata
    cont_vars: list
        list of continuous variables for df_restricted, not xdata
    cat_unique_levles: dict
        dictionary of categorical variables and their unique levels for df_restricted, not xdata
    n_output: int
        number of levels in output layer, 2 for binary, multilevel as specified
    custom_path: string
        path to saved best model, if different from model.save_path
    kwargs: dict
        include all/partial init parameters for AbstractML model, key should be the same as the init parameter name

    Returns
    -------
    array
        Predicted values for the outcome (probability if binary, and expected value otherwise)
    """
    # Re-arrange data
    model_cat_vars, model_cont_vars, model_cat_unique_levels, cat_vars, cont_vars, cat_unique_levels = get_model_cat_cont_split_patsy_matrix(xdata, 
                                                                                                                                             cat_vars, cont_vars, cat_unique_levels)
    fit_df = append_target_to_df(ydata, xdata, exposure)  

    # Fitting model
    ## update init parameters
    for param, value in kwargs.items():
        setattr(deep_learner, param, value)

    best_model_path = deep_learner.fit(fit_df, exposure, 
                                       adj_matrix, model_cat_vars, model_cont_vars, model_cat_unique_levels, 
                                       n_output, custom_path=custom_path)

    # Generating predictions
    pred_df = append_target_to_df(pdata_y, pdata, exposure)
    pred = deep_learner.predict(pred_df, exposure, 
                                adj_matrix, model_cat_vars, model_cont_vars, model_cat_unique_levels, n_output, custom_path=custom_path)
    pred = np.concatenate(pred, 0) # [[batch_size, n_output], [batch_size, n_output] ...] -> [sample_size, n_output]
    if n_output == 2: # binary classification with BCEloss
        pred = pred.squeeze(-1) # [sample_size, 1] -> [sample_size]
    else:
        pred = get_probability_from_multilevel_prediction(pred, pdata_y) 

    return pred

def outcome_deep_learner(deep_learner, xdata, ydata, outcome,
                         adj_matrix, cat_vars, cont_vars, cat_unique_levels, n_output, _continuous_outcome,
                         predict_with_best=False, custom_path=None):
    """Internal function to fit custom_models for the outcome nuisance model.

    Parameters
    ----------
    deep_learner :
        instance of the deep learning model
    xdata : pd.dataframe
        Covariate data to fit the model with
    ydata : pandas.core.series.Series
        Outcome data to fit the model with
    outcome: string
        outcome patamerter to predict
    adj_matrix: SciPy sparse array
        adjacency matrix for GCN model
    cat_vars: list
        list of categorical variables for df_restricted, not xdata
    cont_vars: list
        list of continuous variables for df_restricted, not xdata
    cat_unique_levles: dict
        dictionary of categorical variables and their unique levels for df_restricted, not xdata
    n_output: int
        number of levels in output layer, 2 for binary, multilevel as specified 
    _continuous_outcome: bool
        if the outcome is continuous, default is False
    predict_with_best: bool
        if use the best model to predict, default is False
    custom_path: string
        path to saved best model, if different from model.save_path

    Returns
    -------
    model_object
        best model fitted
    array
        Predicted values for the outcome (probability if binary, and expected value otherwise)
    """
    # Re-arrange data
    model_cat_vars, model_cont_vars, model_cat_unique_levels, cat_vars, cont_vars, cat_unique_levels = get_model_cat_cont_split_patsy_matrix(xdata, 
                                                                                                                                             cat_vars, cont_vars, cat_unique_levels)
    deep_learner_df = append_target_to_df(ydata, xdata, outcome)
    
    if not predict_with_best:
        # Fitting model
        best_model_path = deep_learner.fit(deep_learner_df, outcome, 
                                           adj_matrix, model_cat_vars, model_cont_vars, model_cat_unique_levels, 
                                           n_output, _continuous_outcome, custom_path=custom_path)

    # Generating predictions
    pred = deep_learner.predict(deep_learner_df, outcome, 
                                adj_matrix, model_cat_vars, model_cont_vars, model_cat_unique_levels, 
                                n_output=n_output, _continuous_outcome=_continuous_outcome, custom_path=custom_path)
    pred = np.concatenate(pred, 0) # [[batch_size, n_output], [batch_size, n_output] ...] -> [sample_size, n_output]
    if n_output == 2 or _continuous_outcome: # binary classification with BCEloss / continuous outcome
        pred = pred.squeeze(-1) # [sample_size, 1] -> [sample_size]
    else:
        pred = get_probability_from_multilevel_prediction(pred, ydata) 

    if not predict_with_best:
        return best_model_path, pred
    else:
        return pred


# for time series data
def all_equal(iterable):
    '''check if all elements in a list are identical'''
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def update_cat_var_unique_levels(cat_var_unique_levels_list):
    '''update cat_var_unique_level to maintain the maximum level throughout all time slices'''
    model_cat_unique_levels_final = {} 
    for i, cat_var_level_dict in enumerate(cat_var_unique_levels_list):
        for var_name, num_levels in cat_var_level_dict.items():
            if i == 0:
                model_cat_unique_levels_final[var_name] = num_levels
            else:
                if num_levels > model_cat_unique_levels_final[var_name]:
                    model_cat_unique_levels_final[var_name] = num_levels 
    return model_cat_unique_levels_final

    
def get_final_model_cat_cont_split(model_cat_vars_list, model_cont_vars_list, model_cat_unique_levels_list):
    '''1. check if categorical and continuous variables are consistent throughout time slices
       2. update the categorical variables unique levels to maintain the maximum level throughout all time slices
    '''
    if not all_equal(model_cat_vars_list):
        raise ValueError("cat_vars are not identical throughout time slices")
    else:
        model_cat_vars_final = model_cat_vars_list[-1]
    
    if not all_equal(model_cont_vars_list):
        raise ValueError("cont_vars are not identical throughout time slices")
    else:
        model_cont_vars_final = model_cont_vars_list[-1]

    if not all_equal(model_cat_unique_levels_list):
        print(f'cat_vars levels are not identical througout time slices:')
        print(model_cont_vars_final)
        print(f'adopt the maximum levels for each variable:')
        model_cat_unique_levels_final = update_cat_var_unique_levels(model_cat_unique_levels_list) 
        print(model_cat_unique_levels_final)
    else:
        model_cat_unique_levels_final = model_cat_unique_levels_list[-1]
    
    return model_cat_vars_final, model_cont_vars_final, model_cat_unique_levels_final

def get_imbalance_weights(n_output_final, ydata_array_list, use_last_time_slice=False,
                          imbalance_threshold=3., imbalance_upper_bound=5., default_lock=False):
    ''' set weight against class imbalance
    Parameters
    ----------
    n_output_final: int
        number of classes for the outcome/exposure variables
    ydata_array_list: list 
        a list containing ydata_array, it can contain all time slices or a single time slice, aka the last one
    use_last_time_slice: bool
        use the num_zeros/num_ones odds from the last time slice as the deciding factor
    imbalance_upper_bound: float
        the upper bound for pos_weight, in case the calculated imaalance_odds is too high
    imbalance_threshold: float
        the threshold to correct for imbalance
    default_lock: bool
        if True, return default pos_weight=1. and class_weight=None, which means no weight correction

    Ref:
    https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75
    
    Returns:
    --------
    pos_weight: float
        weight for positive class/negative class
    class_weight: array
        weight for each class if more than 2 classes
    '''
    pos_weight = 1.
    class_weight = None

    if default_lock:
        return pos_weight, class_weight
    
    if use_last_time_slice:
        ydata_array_concat = ydata_array_list[-1]
    else:
        ydata_array_concat = np.concatenate(ydata_array_list, 0)


    if n_output_final == 2: # binary classification with BCEloss
        imbalance_odds = (ydata_array_concat.shape[0] - ydata_array_concat.sum()) / ydata_array_concat.sum() # num_zeros / num_ones
        print(f'imbalance_odds: {imbalance_odds} = ({ydata_array_concat.shape[0]} - {ydata_array_concat.sum()}) / {ydata_array_concat.sum()}')
        if imbalance_odds > imbalance_threshold:
            if imbalance_odds > imbalance_upper_bound:
                pos_weight = imbalance_upper_bound
            else:
                pos_weight = imbalance_odds 
            print(f'pos_weight: {pos_weight}')
    else:
        class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(ydata_array_concat), y=ydata_array_concat)
        max_min_odds = np.max(class_weight) / np.min(class_weight)
        if max_min_odds < imbalance_threshold:
            class_weight = None # if class balance is not severe, do not use class_weight
        else:
            print(f'class_weight: {class_weight}')   
    
    return pos_weight, class_weight
    
def exposure_deep_learner_ts(deep_learner, xdata_list, ydata_list, pdata_list, pdata_y_list, 
                             T_in_id, T_out_id,
                             adj_matrix_list, cat_vars, cont_vars, cat_unique_levels, n_output_list, 
                             custom_path, **kwargs):
    """Internal function to fit custom_models for the exposure nuisance model and generate the predictions.

    Parameters
    ----------
    deep_learner :
        instance of the deep learning model
    xdata_list : list of pd.dataframe
        list of Covariate data to fit the model with, len() = T
    ydata : list of pandas.core.series.Series
        list of Outcome data to fit the model with, len() = T
    pdata_list : list of pd.dataframe
        list of Covariate data to generate the predictions with, len() = T
    pdata_y_list : list of pandas.core.series.Series
        lisf of Truth for predictions, used to evaluate model performance, len() = T
    T_in_id: list
        list of index showing the time slice used for the input data
    T_out_id: list
        list of index showing the time slice used for the outcome data
    adj_matrix_list: list of SciPy sparse array
        list of adjacency matrix for GCN model, len() = T, 
        adj_matrix_list[i] is a SciPy sparse array for observed data
                           is a list of Scipy sparse array for pooled data
        accomendations are made in GCNModelTimeSeries() model
    cat_vars: list
        list of categorical variables for df_restricted, not xdata
    cont_vars: list
        list of continuous variables for df_restricted, not xdata
    cat_unique_levles: dict
        dictionary of categorical variables and their unique levels for df_restricted, not xdata
    n_output_list: list of int
        list of number of levels in output layer, 2 for binary, multilevel as specified; len() = T
    custom_path: string
        path to saved best model, if different from model.save_path
    kwargs: dict
        include all/partial init parameters for AbstractML model, key should be the same as the init parameter name

    Returns
    -------
    array
        Predicted values for the outcome (probability if binary, and expected value otherwise)
    """
    # slicing xdata and ydata list
    xdata_list, ydata_list = [xdata_list[i] for i in T_in_id], [ydata_list[i] for i in T_out_id]
    pdata_list, pdata_y_list = [pdata_list[i] for i in T_in_id], [pdata_y_list[i] for i in T_out_id]
    T_in, T_out = len(xdata_list), len(ydata_list)


    # Re-arrange data
    model_cat_vars_list = []
    model_cont_vars_list = []
    model_cat_unique_levels_list = []

    cat_vars_list = []
    cont_vars_list = []
    cat_unique_levels_list = []

    # fit_df_list = []
    # pred_df_list = []
    ydata_array_list = []
    pdata_y_array_list = []

    for xdata, pdata in zip(xdata_list, pdata_list):
        model_cat_vars, model_cont_vars, model_cat_unique_levels, cat_vars, cont_vars, cat_unique_levels = get_model_cat_cont_split_patsy_matrix(xdata, 
                                                                                                                                                 cat_vars, cont_vars, cat_unique_levels)
        model_cat_vars_list.append(model_cat_vars)
        model_cont_vars_list.append(model_cont_vars)
        model_cat_unique_levels_list.append(model_cat_unique_levels)

        cat_vars_list.append(cat_vars)
        cont_vars_list.append(cont_vars)
        cat_unique_levels_list.append(cat_unique_levels)
    
    for ydata, pdata_y in zip(ydata_list, pdata_y_list):
        ydata_array_list.append(ydata.to_numpy()) # convert pd.series to np.array
        pdata_y_array_list.append(pdata_y.to_numpy()) # convert pd.series to np.array
        
    model_cat_vars_final, model_cont_vars_final, model_cat_unique_levels_final = get_final_model_cat_cont_split(model_cat_vars_list, model_cont_vars_list, model_cat_unique_levels_list)

    ## check if n_output is consistent 
    if not all_equal(n_output_list):
        raise ValueError("n_output are not identical throughout time slices")
    else:
        n_output_final = n_output_list[-1]    

    ## set weight against class imbalance
    pos_weight, class_weight = get_imbalance_weights(n_output_final, ydata_array_list, use_last_time_slice=True,
                                                     imbalance_threshold=3., imbalance_upper_bound=3.2, default_lock=False)
            

    # Fitting model
    ## update init parameters
    for param, value in kwargs.items():
        setattr(deep_learner, param, value)    

    best_model_path = deep_learner.fit([xdata_list, ydata_array_list], T_in, T_out, pos_weight, class_weight,
                                       adj_matrix_list, model_cat_vars_final, model_cont_vars_final, model_cat_unique_levels_final, 
                                       n_output_final, custom_path=custom_path)

    # Generating predictions
    pred = deep_learner.predict([pdata_list, pdata_y_array_list], T_in, T_out, pos_weight, class_weight,
                                adj_matrix_list, model_cat_vars_final, model_cont_vars_final, model_cat_unique_levels_final, 
                                n_output_final, custom_path=custom_path)
    pred = np.concatenate(pred, 0)
    # [[batch_size, n_output, T], [batch_size, n_output, T] ...] -> [sample_size, n_output, T]
    if n_output_final == 2: # binary classification with BCEloss
        pred = pred.squeeze(1) # [sample_size, 1, T] -> [sample_size, T]
        pred = pred[:, -1] # only return the last time slice 
    else:
        pred = get_probability_from_multilevel_prediction(pred[:, :, -1], pdata_y_list[-1]) # only return the last time slice

    return pred


def outcome_deep_learner_ts(deep_learner, src_xdata_list, src_ydata_list, trg_xdata_list, trg_ydata_list,
                            T_in_id, T_out_id,
                            adj_matrix_list, cat_vars, cont_vars, cat_unique_levels, n_output_list, _continuous_outcome,
                            predict_with_best=False, custom_path=None):
    """Internal function to fit custom_models for the outcome nuisance model.

    Parameters
    ----------
    deep_learner :
        instance of the deep learning model
    src_xdata_list : list of pd.dataframe
        list of Covariate data to fit the model with, src_data
    src_ydata_list : list of pandas.core.series.Series
        list of Outcome data to fit the model with, src_data
    trg_xdata_list : list of pd.dataframe
        list of Covariate data to fit the model with, trg_data
    trg_ydata_list : list of pandas.core.series.Series
        list of Outcome data to fit the model with, trg_data
    T_in_id: list
        list of index showing the time slice used for the input data
    T_out_id: list
        list of index showing the time slice used for the outcome data
    adj_matrix_list: list of SciPy sparse array
        list of adjacency matrix for GCN model, len() = T, 
        adj_matrix_list[i] is a SciPy sparse array for observed data
                           is a list of Scipy sparse array for pooled data
        accomendations are made in GCNModelTimeSeries() model
    cat_vars: list
        list of categorical variables for df_restricted, not xdata
    cont_vars: list
        list of continuous variables for df_restricted, not xdata
    cat_unique_levles: dict
        dictionary of categorical variables and their unique levels for df_restricted, not xdata
    n_output_list: list of int
        list of number of levels in output layer, 2 for binary, multilevel as specified; len() = T
    _continuous_outcome: bool
        if the outcome is continuous, default is False
    predict_with_best: bool
        if use the best model to predict, default is False
    custom_path: string
        path to saved best model, if different from model.save_path

    Returns
    -------
    model_object
        best model fitted
    array
        Predicted values for the outcome (probability if binary, and expected value otherwise)
    """
    # slicing xdata and ydata list
    # src_xdata_list, src_ydata_list = [src_xdata_list[i] for i in T_in_id], [src_ydata_list[i] for i in T_out_id]
    src_xdata_list_full, src_ydata_list_full = [src_xdata_list[i] for i in T_in_id], [src_ydata_list[i] for i in T_out_id]
    trg_xdata_list_full, trg_ydata_list_full = [trg_xdata_list[i] for i in T_in_id], [trg_ydata_list[i] for i in T_out_id]

    # slice pooled data / observed data to match the number of observations
    min_num_samples_in_src, max_num_samples_in_src = min([df.shape[0] for df in src_xdata_list]), max([df.shape[0] for df in src_xdata_list])
    min_num_samples_in_trg, max_num_samples_in_trg = min([df.shape[0] for df in trg_xdata_list]), max([df.shape[0] for df in trg_xdata_list])
    keep_samples = min(min_num_samples_in_src, min_num_samples_in_trg)
    print(f'min_num_samples_in_src: {min_num_samples_in_src}, min_num_samples_in_trg: {min_num_samples_in_trg}, keep_samples: {keep_samples}')
    if min_num_samples_in_src == max_num_samples_in_src and min_num_samples_in_trg == max_num_samples_in_trg and min_num_samples_in_src == min_num_samples_in_trg:
        src_xdata_list, src_ydata_list = src_xdata_list_full, src_ydata_list_full
        trg_xdata_list, trg_ydata_list = trg_xdata_list_full, trg_ydata_list_full
    else:
        src_xdata_list = [df[:keep_samples] for df in src_xdata_list_full]
        src_ydata_list = [df[:keep_samples] for df in src_ydata_list_full] 
        trg_xdata_list = [df[:keep_samples] for df in trg_xdata_list_full]
        trg_ydata_list = [df[:keep_samples] for df in trg_ydata_list_full]
        # retain as much trg data for prediction, not just the ones used for training
        pred_trg_xdata_list = [df[:min_num_samples_in_trg] for df in trg_xdata_list_full]
        pred_trg_ydata_list = [df[:min_num_samples_in_trg] for df in trg_ydata_list_full]


    # # slice pooled data to match the observed data: won't work if trg obs is less than src obs
    # num_samples_in_src = src_xdata_list[0].shape[0]
    # num_samples_in_trg = trg_xdata_list[0].shape[0]
    # print(f'num_samples_in_src: {num_samples_in_src}, num_samples_in_trg: {num_samples_in_trg}')
    # if num_samples_in_src != num_samples_in_trg:
    #     trg_xdata_list = [df[:num_samples_in_src] for df in trg_xdata_list_full]
    #     trg_ydata_list = [df[:num_samples_in_src] for df in trg_ydata_list_full]
    # else:
    #     trg_xdata_list, trg_ydata_list = trg_xdata_list_full, trg_ydata_list_full


    # Re-arrange data
    model_cat_vars_list = []
    model_cont_vars_list = []
    model_cat_unique_levels_list = []

    cat_vars_list = []
    cont_vars_list = []
    cat_unique_levels_list = []

    # deep_learner_df_list = []
    src_ydata_array_list = []
    trg_ydata_array_list = []

    for src_xdata in src_xdata_list:
        model_cat_vars, model_cont_vars, model_cat_unique_levels, cat_vars, cont_vars, cat_unique_levels = get_model_cat_cont_split_patsy_matrix(src_xdata, 
                                                                                                                                                 cat_vars, cont_vars, cat_unique_levels)
        model_cat_vars_list.append(model_cat_vars)
        model_cont_vars_list.append(model_cont_vars)
        model_cat_unique_levels_list.append(model_cat_unique_levels)

        cat_vars_list.append(cat_vars)
        cont_vars_list.append(cont_vars)
        cat_unique_levels_list.append(cat_unique_levels)

    for src_ydata, trg_ydata in zip(src_ydata_list, trg_ydata_list):
        src_ydata_array_list.append(src_ydata.to_numpy()) # convert pd.series to np.array
        trg_ydata_array_list.append(trg_ydata.to_numpy()) # convert pd.series to np.array
    # trg_ydata_array_list_full =[df.to_numpy() for df in trg_ydata_list_full]
    pred_trg_ydata_array_list =[df.to_numpy() for df in pred_trg_ydata_list]
    
    model_cat_vars_final, model_cont_vars_final, model_cat_unique_levels_final = get_final_model_cat_cont_split(model_cat_vars_list, model_cont_vars_list, model_cat_unique_levels_list)

    ## check if n_output is consistent 
    if not all_equal(n_output_list):
        raise ValueError("n_output are not identical throughout time slices")
    else:
        n_output_final = n_output_list[-1]    

    ## set weight against class imbalance
    pos_weight, class_weight = get_imbalance_weights(n_output_final, src_ydata_array_list, use_last_time_slice=True,
                                                     imbalance_threshold=3., imbalance_upper_bound=3.2, default_lock=False)
            
    if not predict_with_best:
        # Fitting model
        best_model_path = deep_learner.fit([src_xdata_list, src_ydata_array_list], [trg_xdata_list, trg_ydata_array_list],
                                           T_in_id, T_out_id, class_weight,
                                           adj_matrix_list, model_cat_vars_final, model_cont_vars_final, model_cat_unique_levels_final,
                                           n_output_final, _continuous_outcome, custom_path=custom_path)

    # Generating predictions
    pred_src = deep_learner.predict([src_xdata_list, src_ydata_array_list], T_in_id, T_out_id, class_weight,
                                     adj_matrix_list, model_cat_vars_final, model_cont_vars_final, model_cat_unique_levels_final, 
                                     n_output=n_output_final, _continuous_outcome=_continuous_outcome, custom_path=custom_path)
    # generate predictions for all trg data, not just the trained ones
    # pred_trg = deep_learner.predict([trg_xdata_list_full, trg_ydata_array_list_full], T_in_id, T_out_id, class_weight,
    #                                  adj_matrix_list, model_cat_vars_final, model_cont_vars_final, model_cat_unique_levels_final,
    #                                  n_output=n_output_final, _continuous_outcome=_continuous_outcome, custom_path=custom_path)
    pred_trg = deep_learner.predict([pred_trg_xdata_list, pred_trg_ydata_array_list], T_in_id, T_out_id, class_weight,
                                     adj_matrix_list, model_cat_vars_final, model_cont_vars_final, model_cat_unique_levels_final,
                                     n_output=n_output_final, _continuous_outcome=_continuous_outcome, custom_path=custom_path)
    pred_src, pred_trg = np.concatenate(pred_src, 0), np.concatenate(pred_trg, 0)
    # [[batch_size, n_output, T], [batch_size, n_output, T] ...] -> [sample_size, n_output, T]
    if _continuous_outcome: # continuous outcome
        pred_src, pred_trg = pred_src.squeeze(1), pred_trg.squeeze(1) # [sample_size, 1, T] -> [sample_size, T]
        pred_src, pred_trg = pred_src[:, -1], pred_trg[:, -1] # only return the last time slice
    elif n_output_final == 2: # binary classification with CrossEntropyLoss #TODO
        pred_src, pred_trg = pred_src[:, -1, -1], pred_trg[:, -1, -1]  
        # select the prob for class 1 and select the prob for last time slice
    else:
        NotImplementedError("Multilevel classification is not supported yet")

    if not predict_with_best:
        return best_model_path, pred_src, pred_trg
    else:
        return pred_src, pred_trg