# import random
import math
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic

from beowulf.dgm.utils import (network_to_df, fast_exp_map, exposure_restrictions,
                               odds_to_probability, probability_to_odds)
# from utils import (network_to_df, fast_exp_map, exposure_restrictions,
#                                odds_to_probability, probability_to_odds)
# import time

################ social distancing ####################
def get_removed_edges(data, graph, social_distancing, p_remove_connection=0.1, rng_generator=None):
    selected_nodes = np.arange(data.shape[0])[social_distancing==1]
    
    removed_edges = []
    # edges_counter = 0
    for a, b, attrs in graph.edges(data=True):
        if a in selected_nodes or b in selected_nodes:
            # edges_counter += 1
            # if random.random() < p_remove_connection:  
            if rng_generator.uniform() < p_remove_connection:  
                # print(f'Edge {a}-{b} is removed')
                removed_edges.append((a,b))
    # print(f'percent edges removed: {len(removed_edges)/edges_counter}')
    return removed_edges

def _outbreak_social_dist(true_graph, training_graph, duration, limit, rng_generator=None):
    """Outbreak simulation script in a single function
    true_graph: graph when edges picked using the social distancing parameter is removed
    training_graph: graph where all edges preserved, did not run for the infection cycle, 
                    merely document values calculated using the true graph, used for training
    """
    # Adding node attributes
    for (n_true, d_true), (n_train, d_train) in zip(true_graph.nodes(data=True), training_graph.nodes(data=True)):
        d_true['D'] = 0
        d_true['R'] = 0
        d_true['t'] = 0

        d_train['D'] = 0
        d_train['R'] = 0
        d_train['t'] = 0


    # Selecting initial infections
    all_ids = [n for n in true_graph.nodes()]
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
    time = 0
    while time < limit:  # Simulate outbreaks until time-step limit is reached
        time += 1
        for inf in sorted(infected, key=lambda _: rng_generator.random()):
        # for inf in sorted(infected, key=lambda _: random.random()):
            # Book-keeping for infected nodes
            true_graph.nodes[inf]['D'] = 1
            true_graph.nodes[inf]['t'] += 1

            training_graph.nodes[inf]['D'] = 1
            training_graph.nodes[inf]['t'] += 1

            if true_graph.nodes[inf]['t'] > duration:
                true_graph.nodes[inf]['I'] = 0         # Node is no longer infectious after this loop
                true_graph.nodes[inf]['R'] = 1         # Node switches to Recovered
                
                training_graph.nodes[inf]['I'] = 0        
                training_graph.nodes[inf]['R'] = 1         
                infected.remove(inf)

            # Attempt infections of neighbors
            for contact in nx.neighbors(true_graph, inf):
                if true_graph.nodes[contact]["D"] == 1:
                    pass
                else:
                    # pr_y = logistic.cdf(- 4.5
                    #                     + 1.0*true_graph.nodes[contact]['A']
                    #                     - 0.5*true_graph.nodes[contact]['H'])
                    pr_y = logistic.cdf(- 2.5
                                        + 1.0*true_graph.nodes[contact]['A']
                                        - 0.5*true_graph.nodes[contact]['H'])
                    # print(pr_y)
                    # if np.random.binomial(n=1, p=pr_y, size=1):
                    if rng_generator.binomial(n=1, p=pr_y, size=1):
                        true_graph.nodes[contact]['I'] = 1
                        true_graph.nodes[contact]["D"] = 1

                        training_graph.nodes[contact]['I'] = 1
                        training_graph.nodes[contact]["D"] = 1
                        infected.append(contact)

    return true_graph, training_graph

def social_dist_dgm(network, restricted=False,
                    time_limit=10, inf_duration=5,
                    update_split=False, cat_vars=[], cont_vars=[], cat_unique_levels={},
                    random_seed=100):
    """
    Parameters
    ----------
    network:
        input network
    restricted:
        whether to use the restricted treatment assignment
    time_limit:
        maximum time to let the outbreak go through
    inf_duration:
        duration of infection status in time-steps
    update_split: 
        whether to update the cat_vars, cont_vars, and cat_unique_levels
    cat_vars:
        list of categorial variable names
    cont_vars:
        list of continuous variables names
    cat_unique_levels:
        dict of categorical variable names and their number of unique levels
    """
    # set up random generator
    rng = np.random.default_rng(seed=random_seed)

    graph = network.copy()
    data = network_to_df(graph)

    adj_matrix = nx.adjacency_matrix(graph, weight=None)
    data['A_sum'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='sum')
    data['A_mean'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='mean')
    data['H_sum'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='sum')
    data = pd.merge(data, pd.DataFrame.from_dict(dict(network.degree),
                                                orient='index').rename(columns={0: 'F'}),
                    how='left', left_index=True, right_index=True)

    # Running Data Generating Mechanism for A
    pr_a = logistic.cdf(- 3.0 
                        + 1.5*data['A'] + 0.7*data['H']
                        + 0.75*data['H_sum'] + 0.95*data['A_sum'] 
                        - 0.3*data['F'])
    # pr_a = logistic.cdf(- 5.35 
    #                     + 1.0*data['A'] + 0.5*data['H']
    #                     + 0.3*data['H_sum'] + 0.1*data['A_sum'] 
    #                     + 1.5*data['F'])
    # social_distancing = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    social_distancing = rng.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['social_dist'] = social_distancing

    if restricted:  # if we are in the restricted scenarios
        # Use the restriction design for vaccine
        attrs = exposure_restrictions(network=network.graph['label'], exposure='social_dist',
                                      n=nx.number_of_nodes(graph))
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['social_dist']))
        social_distancing = data['social_dist'].to_numpy() # update social_distancing

    # print("Pr(V):", np.mean(vaccine))
    for n in graph.nodes():
        graph.nodes[n]['social_dist'] = int(data.loc[data.index == n, 'social_dist'].values)

    # Running outbreak simulation
    # Remove 10% of connections if apply social distancing
    true_graph = graph.copy()
    # true_graph.remove_edges_from(get_removed_edges(data, graph, social_distancing, p_remove_connection=0.1))
    true_graph.remove_edges_from(get_removed_edges(data, graph, social_distancing, p_remove_connection=0.1, rng_generator=rng))
    
    true_graph, graph = _outbreak_social_dist(true_graph, graph, duration=inf_duration, limit=time_limit, rng_generator=rng)

    if update_split:
        cat_vars.append('social_dist')
        cat_unique_levels['social_dist'] = pd.unique(data['social_dist'].astype('int')).max() + 1
        return true_graph, graph, cat_vars, cont_vars, cat_unique_levels
    else:
        return true_graph, graph

def social_dist_dgm_truth(network, pr_a, shift=False, restricted=False,
                          time_limit=10, inf_duration=5,
                          percent_candidates=0.3, mode='top',
                          random_seed=100):
    '''Get ground truth for social distancing action
    percent_candidates: proportions of nodes to be selected as social distancing cluster center nodes candidates
    mode: 'top' or 'bottom' or 'all', 
          'top': select the top [percent_candidates] nodes with highest degree
          'bottom': select the bottom [percent_candidates] nodes with lowest degree
          'all': select all nodes

    P.S. if shift = False, pr_a is used to select the actual cluster center nodes from candidates
    '''
    # set up random generator
    rng = np.random.default_rng(seed=random_seed)

    graph = network.copy()
    data = network_to_df(graph)

    # Running Data Generating Mechanism for A
    if shift:  # If a shift in the Odds distribution is instead specified
        adj_matrix = nx.adjacency_matrix(graph, weight=None)
        data['A_sum'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='sum')
        data['A_mean'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='mean')
        data['H_sum'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='sum')
        data = pd.merge(data, pd.DataFrame.from_dict(dict(network.degree),
                                                    orient='index').rename(columns={0: 'F'}),
                        how='left', left_index=True, right_index=True)
        # Running Data Generating Mechanism for A
        prob = logistic.cdf(- 3.0 
                            + 1.5*data['A'] + 0.7*data['H']
                            + 0.75*data['H_sum'] + 0.95*data['A_sum'] 
                            - 0.3*data['F'])
        # prob = logistic.cdf(- 5.35 
        #                     + 1.0*data['A'] + 0.5*data['H']
        #                     + 0.3*data['H_sum'] + 0.1*data['A_sum'] 
        #                     + 1.5*data['F'])
        # print(prob)
        odds = probability_to_odds(prob)
        # print(odds)
        pr_a = odds_to_probability(np.exp(np.log(odds) + pr_a))
        # print(pr_a)
        social_distancing = rng.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
        data['social_dist'] = social_distancing
    else:
        # calculate degree to select candidates for super-spreaders or super-defenders
        data = pd.merge(data, pd.DataFrame.from_dict(dict(network.degree),
                                                        orient='index').rename(columns={0: 'F'}),
                            how='left', left_index=True, right_index=True)

        if mode == 'top':
            num_candidates = math.ceil(data.shape[0] * percent_candidates)
            candidates_nodes = data.nlargest(num_candidates, 'F').index # super-spreader
        elif mode == 'bottom':
            num_candidates = math.ceil(data.shape[0] * percent_candidates)
            candidates_nodes = data.nsmallest(num_candidates, 'F').index # super-defender
        elif mode == 'all':
            num_candidates = data.shape[0]
            candidates_nodes = data.index
        
        # social_distancing_prior = np.random.binomial(n=1, p=pr_a, size=num_candidates)
        social_distancing_prior = rng.binomial(n=1, p=pr_a, size=num_candidates)
        social_distancing_nodes = candidates_nodes[social_distancing_prior==1]
        social_distancing = np.zeros(data.shape[0])
        social_distancing[social_distancing_nodes] = 1
        data['social_dist'] = social_distancing
    
    if restricted:  # if we are in the restricted scenarios
        # Use the restriction design for vaccine
        attrs = exposure_restrictions(network=network.graph['label'], exposure='social_dist',
                                    n=nx.number_of_nodes(graph))
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['social_dist']))
        social_distancing = data['social_dist'].to_numpy() # update social_distancing
    
    
    for n in graph.nodes():
        graph.nodes[n]['social_dist'] = int(data.loc[data.index == n, 'social_dist'].values)
    
    # Running outbreak simulation
    # Remove 10% of connections if apply social distancing
    true_graph = graph.copy()
    # removed_edges = get_removed_edges(data, graph, social_distancing, p_remove_connection=0.1, rng_generator=rng)
    # print(removed_edges) 
    true_graph.remove_edges_from(get_removed_edges(data, graph, social_distancing, p_remove_connection=0.1, rng_generator=rng))
    
    true_graph, graph = _outbreak_social_dist(true_graph, graph, duration=inf_duration, limit=time_limit, rng_generator=rng)
    dis = []
    for nod, d in graph.nodes(data=True):
        dis.append(d['D'])
    return np.mean(dis)
    

################# quarantine policy: time series ####################
def update_summary_measures(data, graph):
    ''' Update summary measures for the graph data
    because I_sum has to be updated, summary data is updated for every infected node''' 
    data = network_to_df(graph)

    adj_matrix = nx.adjacency_matrix(graph, weight=None)
    data['A_sum'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='sum')
    # data['A_mean'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='mean')
    data['H_sum'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='sum')
    data = pd.merge(data, pd.DataFrame.from_dict(dict(graph.degree),
                                                 orient='index').rename(columns={0: 'F'}),
                                                 how='left', left_index=True, right_index=True)
    
    # get actions from current Infected individual, document as "quarantine"
    # apply quarantine if the nodes is a contact of inf, implemented in the neigbors loop below
    data['I_sum'] = fast_exp_map(adj_matrix, np.array(data['I']), measure='sum')
    data['I_ratio'] = data['I_sum'] / data['F']  # ratio of infected neighbors
    data['I_ratio'] = data['I_ratio'].fillna(0)  # fill in 0 for nodes with no neighbors

    # add I_ratio to graph data
    if nx.is_directed(graph):
        raise NotImplementedError("Directed graph is not supported yet")
    else:
        nx.set_node_attributes(graph, dict(data['I_ratio']), 'I_ratio')
    # # for-loop version
    # for n in graph.nodes():
    #     graph.nodes[n]['I_ratio'] = float(data.loc[data.index == n, 'I_ratio'].values)
    
    return data, graph
    

def update_pr_a(data, graph, restricted, edge_recorder, time_step, rng):
    ''' Update the probability of quarantine action for every quarantine_period '''
    pr_a = logistic.cdf(- 4.5 
                        + 1.2*data['A'] + 0.8*data['H']
                        + 0.5*data['H_sum'] + 0.3*data['A_sum'] 
                        + 1.2*data['I_ratio'])
    quarantine = rng.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['quarantine'] = quarantine
    # print(pr_a)
    # print(quarantine)

    if restricted:  # if we are in the restricted scenarios
        # Use the restriction design for vaccine
        attrs = exposure_restrictions(network=graph.graph['label'], exposure='quarantine',
                                        n=nx.number_of_nodes(graph))
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['quarantine']))
        # quarantine = data['quarantine'].to_numpy() # update quarantine    

    # apply quarantine to all selected nodes: remove immediate neighbors for the next time step
    # Function version
    if nx.is_directed(graph):
        raise NotImplementedError("Directed graph is not supported yet")
    else:
        nx.set_node_attributes(graph, dict(data['quarantine']), 'quarantine')
        edge_recorder[time_step] = list(nx.edges(graph, data[data['quarantine']==1].index))
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
    if time_step - quarantine_period > 0: 
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

def quarantine_dgm_time_series(network, restricted=False,
                               time_limit=10, inf_duration=5, quarantine_period=2,
                               update_split=False, cat_vars=[], cont_vars=[], cat_unique_levels={},
                               random_seed=100):
    ''' Assume that we do not know who are the infected, that is the ID for inf is unknown.
    Because checking up which IDs are infected is hard/expensive to do, 
    we apply quarantine on most connected/leasted connected nodes' neighbors to reduce the spread of the disease.
    The action should also balance budgets, hence setting up the percent_candidates to 30% of nodes.

    '''
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
        # edge_to_remove = []
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
            
            # calculate summary measure
            data, graph = update_summary_measures(data, graph)
            
            # A quarantined case stays for quarantine_period days, and released after that.
            # Thus, pr_a should be re-calcualted for every quarantine_period days.            
            if quarantine_period == 1: # Since the first time step is 1, quarantine_period = 1 cannot be included in the % below
                data, graph, edge_recorder = update_pr_a(data, graph, restricted, edge_recorder, time, rng)
                # print(f'time {time}: inf {inf} from {infected}')
            else:
                if time % quarantine_period == 1:
                    data, graph, edge_recorder = update_pr_a(data, graph, restricted, edge_recorder, time, rng)
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
    
    if update_split:
        cat_vars.append('quarantine')
        cat_unique_levels['quarantine'] = pd.unique(data['quarantine'].astype('int')).max() + 1
        cont_vars.append('I_ratio')
        return graph, graph_saved_by_time, cat_vars, cont_vars, cat_unique_levels
    else:
        return graph, graph_saved_by_time

def apply_quarantine_action(data, graph, shift, restricted, edge_recorder, time_step, rng,
                            pr_a, percent_candidates, mode):
    # Running Data Generating Mechanism for A
    if shift: # If a shift in the Odds distribution is instead specified
        prob = logistic.cdf(- 4.5 
                            + 1.2*data['A'] + 0.8*data['H']
                            + 0.5*data['H_sum'] + 0.3*data['A_sum'] 
                            + 1.2*data['I_ratio'])
        # print(prob)
        odds = probability_to_odds(prob)
        pr_a = odds_to_probability(np.exp(np.log(odds) + pr_a))
        # print(pr_a)
        quarantine = rng.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
        # print(quarantine)
        data['quarantine'] = quarantine
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
        quarantine = np.zeros(data.shape[0])
        quarantine[quarantine_nodes] = 1 
        data['quarantine'] = quarantine
    
    if restricted:  # if we are in the restricted scenarios
        # Use the restriction design for vaccine
        attrs = exposure_restrictions(network=graph.graph['label'], exposure='quarantine',
                                        n=nx.number_of_nodes(graph))
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['quarantine']))
        # quarantine = data['quarantine'].to_numpy() # update quarantine
    
    # apply quarantine to all selected nodes: remove immediate neighbors for the next time step
    # Function version
    if nx.is_directed(graph):
        raise NotImplementedError("Directed graph is not supported yet")
    else:
        nx.set_node_attributes(graph, dict(data['quarantine']), 'quarantine')
        edge_recorder[time_step] = list(nx.edges(graph, data[data['quarantine']==1].index))
    # # for-loop version
    # for n in graph.nodes():
    #     graph.nodes[n]['quarantine'] = int(data.loc[data.index == n, 'quarantine'].values)
    #     if n in data[data['quarantine']==1].index: # remove all immediate neighbors
    #         for neighbor in graph.neighbors(n):
    #             edge_recorder[time_step].append((neighbor, n))
    return data, graph, edge_recorder
    


# def quarantine_dgm_truth(network, pr_a, shift=False, restricted=False,
#                          time_limit=10, inf_duration=5, quarantine_period=2,
#                          percent_candidates=0.3, mode='top',
#                          random_seed=100):
#     '''Get ground truth for quarantine action
#     percent_candidates: proportions of nodes to be selected as social distancing cluster center nodes candidates
#     mode: 'top' or 'bottom' or 'all', 
#           'top': select the top [percent_candidates] nodes with highest I_ratio (percentage of infected neighbors)
#           'bottom': select the bottom [percent_candidates] nodes with lowest I_ratio
#           'all': select all nodes

#     P.S. pr_a is used to select the actual cluster center nodes from candidates'''
#     # set up random generator
#     rng = np.random.default_rng(seed=random_seed)

#     graph = network.copy()
#     data = network_to_df(graph)

#     for n, d in graph.nodes(data=True):
#         d['D'] = 0
#         d['R'] = 0
#         d['t'] = 0
#         d['I'] = 0
#         d['quarantine'] = 0

#     # Selecting initial infections
#     all_ids = [n for n in graph.nodes()]
#     # infected = random.sample(all_ids, 5)
#     if len(all_ids) <= 500:
#         infected = [4, 36, 256, 305, 443]
#     elif len(all_ids) == 1000:
#         infected = [4, 36, 256, 305, 443, 552, 741, 803, 825, 946]
#     elif len(all_ids) == 2000:
#         infected = [4, 36, 256, 305, 443, 552, 741, 803, 825, 946,
#                     1112, 1204, 1243, 1253, 1283, 1339, 1352, 1376, 1558, 1702]
#     else:
#         raise ValueError("Invalid network IDs")
    
#     # Running through infection cycle
#     graph_saved_by_time = [] # save graph slice for each time point
#     edge_recorder = {key:[] for key in range(1, time_limit+1, quarantine_period)} # record edge_to_remove to be add back after the quarantine period has passed
#     # starts with 1 because the timer below starts with 1
#     timer = 0
#     while timer < time_limit:  # Simulate outbreaks until time-step limit is reached
#         timer += 1
#         print(f'time {timer}')
#         # for inf in sorted(infected, key=lambda _: random.random()):
#         for inf in sorted(infected, key=lambda _: rng.random()):
#             # Book-keeping for infected nodes
#             graph.nodes[inf]['I'] = 1
#             graph.nodes[inf]['D'] = 1
#             graph.nodes[inf]['t'] += 1

#             if graph.nodes[inf]['t'] > inf_duration:
#                 graph.nodes[inf]['I'] = 0         # Node is no longer infectious after this loop
#                 graph.nodes[inf]['R'] = 1         # Node switches to Recovered
#                 infected.remove(inf)
            
#             # Calculate summary measure
#             start_time = time.time()
#             data, graph = update_summary_measures(data, graph)
#             end_time = time.time()
#             print(f'update_summary_measures(): {end_time-start_time:.3f} sec')

#             # Apply quarantine actions
#             if quarantine_period == 1:
#                 start_time = time.time()
#                 data, graph, edge_recorder = apply_quarantine_action(data, graph, shift, restricted, edge_recorder, timer, rng,
#                                                                      pr_a, percent_candidates, mode)
#                 end_time = time.time()
#                 print(f'apply_quarantine_action(): {end_time-start_time:.3f} sec')
#                 # print(f'time {time}: inf {inf} from {infected}')
#             else:
#                 if timer % quarantine_period == 1:
#                     start_time = time.time()
#                     data, graph, edge_recorder = apply_quarantine_action(data, graph, shift, restricted, edge_recorder, timer, rng,
#                                                                          pr_a, percent_candidates, mode)
#                     end_time = time.time()
#                     print(f'apply_quarantine_action(): {end_time-start_time:.3f} sec')
#                     # print(f'time {time}: inf {inf} from {infected}')  
            
#             # Simulate infections of immediate neighbors
#             time_start = time.time()
#             graph, infected = simulate_infection_of_immediate_neighbors(graph, inf, infected, rng)
#             time_end = time.time()
#             print(f'simulate_infection_of_immediate_neighbors(): {time_end-time_start:.3f} sec')

#         # save graph at current time step
#         graph_saved_by_time.append(graph.copy())
#         # update graph for next time step
#         if quarantine_period == 1:
#             start_time = time.time()
#             graph = update_edge_in_graph(graph, edge_recorder, timer, quarantine_period)
#             end_time = time.time()
#             print(f'update_edge_in_graph(): {end_time-start_time:.3f} sec')
#         else:
#             if timer % quarantine_period == 1:
#                 start_time = time.time()
#                 graph = update_edge_in_graph(graph, edge_recorder, timer, quarantine_period)
#                 end_time = time.time()
#                 print(f'update_edge_in_graph(): {end_time-start_time:.3f} sec')
    
#     dis_save_by_time = []
#     for g in graph_saved_by_time:
#         dis = [] 
#         for nod, d in g.nodes(data=True):
#             dis.append(d['D'])
#         dis_save_by_time.append(np.mean(dis))

#     # return np.mean(dis), dis_save_by_time, graph_saved_by_time # save last time point and the whole time series
#     return np.mean(dis), dis_save_by_time # save last time point and the whole time series

def quarantine_dgm_truth(network, pr_a, shift=False, restricted=False,
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
    return np.mean(dis), dis_save_by_time # save last time point and the whole time series

     


if __name__ == '__main__':

    from beowulf import load_uniform_vaccine, load_random_vaccine
    import json 
    import time
    # import random
    # random.seed(17)
    # np.random.seed(17)

    # n=2000
    # restricted=False # only for random network

    # # proportion to be removed: if no node selection is made, the proportion is the same for all nodes
    # shift=True
    # if shift:
    #     prop_treated = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]
    # else:
    #     prop_treated = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    #                     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    # def _check_if_quaratine_is_applied_correctly(network_list, check_I_ratio, check_quarantine, check_infected):
    #     for i, graph in enumerate(network_list):
    #         print(f'graph: {i}')
    #         tmp_df = network_to_df(graph)
    #         if check_I_ratio:
    #             print('++++++++++++++++++++++++++++I_ratio+++++++++++++++++++++++++++++++++')
    #             print(pd.unique(tmp_df['I_ratio']))
    #             print((tmp_df['I_ratio'] > 0).sum())
    #             print(tmp_df['I_ratio'][tmp_df['I_ratio'] > 0].index)
    #         if check_quarantine:
    #             print('++++++++++++++++++++++++++++Quarantine+++++++++++++++++++++++++++++++++')
    #             print(pd.unique(tmp_df['quarantine']))
    #             print((tmp_df['quarantine'] == 1).sum())
    #             print(tmp_df['quarantine'][tmp_df['quarantine'] == 1].index)
    #         if check_infected:
    #             print('++++++++++++++++++++++++++++Infected+++++++++++++++++++++++++++++++++')
    #             print(pd.unique(tmp_df['D']))
    #             print((tmp_df['D'] == 1).sum())
    #             print(tmp_df['D'][tmp_df['D'] == 1].index)
    #         print()

    
    # # social distancing 
    # ## uniform
    # G, cat_vars, cont_vars, cat_unique_levels = load_uniform_vaccine(n=n, return_cat_cont_split=True)
    # H_true, H, cat_vars, cont_vars, cat_unique_levels = social_dist_dgm(G, restricted=False, 
    #                                                                     time_limit=10, inf_duration=5,
    #                                                                     update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
    #                                                                     random_seed=3407)                                                                        
    # print(f'social distancing uniform n={n}: {cat_unique_levels}')

    # ### ground truth
    # results = {}
    # for pr_a in prop_treated:
    #     ground_truth = social_dist_dgm_truth(G, pr_a, shift=shift, restricted=False,
    #                                          time_limit=10, inf_duration=5,
    #                                          percent_candidates=0.3, mode='top',
    #                                          random_seed=3407) 
    #     print(f'pr_a: {pr_a} ground truth: {ground_truth}')
    #     results[pr_a] = ground_truth
    # print(f'n:{n} | shift: {shift}')
    # print(results)

    # # random
    # G, cat_vars, cont_vars, cat_unique_levels = load_random_vaccine(n=n, return_cat_cont_split=True)
    # H_true, H, cat_vars, cont_vars, cat_unique_levels = social_dist_dgm(G, restricted=restricted, 
    #                                                                     time_limit=10, inf_duration=5,
    #                                                                     update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)
    # print(f'social distancing uniform n={n}: {cat_unique_levels}')

    # ### ground truth
    # results = {}
    # for pr_a in prop_treated:
    #     ground_truth = social_dist_dgm_truth(G, pr_a, shift=shift, restricted=restricted,
    #                                          time_limit=10, inf_duration=5,
    #                                          percent_candidates=0.3, mode='top',
    #                                          random_seed=3407) 
    #     print(f'pr_a: {pr_a} ground truth: {ground_truth}')
    #     results[pr_a] = ground_truth
    # print(f'n:{n} | restricted:{restricted} | shift: {shift}')
    # print(results)

    
    # # quarantine
    # # uniform
    # start_time = time.time()
    # G, cat_vars, cont_vars, cat_unique_levels = load_uniform_vaccine(n=n, return_cat_cont_split=True)
    # end_time = time.time()
    # print(f'load_uniform_vaccin(): {end_time - start_time:.2f} sec(s)')

    # start_time = time.time()
    # H, network_list, cat_vars, cont_vars, cat_unique_levels = quarantine_dgm_time_series(G, restricted=False, 
    #                                                                                      time_limit=10, inf_duration=5, quarantine_period=2,
    #                                                                                      update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
    #                                                                                      random_seed=3407)
    # end_time = time.time()
    # print(f'quarantine_dgm_time_series(): {end_time - start_time:.2f} sec(s)')
    # print(f'quarantine uniform n={n}: {cat_unique_levels}')

    # # check if the quarantine is applied correctly
    # _check_if_quaratine_is_applied_correctly(network_list, check_I_ratio=True, check_quarantine=True, check_infected=True)


    # ### ground truth
    # results = {}
    # for pr_a in prop_treated:
    #     start_time = time.time()
    #     ground_truth_last, ground_truth_all = quarantine_dgm_truth(G, pr_a, shift=shift, restricted=False,
    #                                                                time_limit=10, inf_duration=5, quarantine_period=2,
    #                                                                percent_candidates=0.3, mode='top',
    #                                                                random_seed=3407) 
    #     end_time = time.time()
    #     print(f'time elapes: {end_time - start_time} sec(s)')
    #     print(f'pr_a: {pr_a}')
    #     print(f'ground truth: {ground_truth_last}')
    #     print(f'grond truth all time point: {ground_truth_all}')
    #     results[pr_a] = ground_truth_last
    # print(f'n:{n} | shift: {shift}')
    # print(results)

    # shift=False
    # pr_a = 0.85
    # ground_truth_last, ground_truth_al, graph_all = quarantine_dgm_truth(G, pr_a, shift=shift, restricted=False,
    #                                                                      time_limit=10, inf_duration=5, quarantine_period=2,
    #                                                                      percent_candidates=0.3, mode='all',
    #                                                                      random_seed=3407) 
    # _check_if_quaratine_is_applied_correctly(graph_all, check_I_ratio=True, check_quarantine=True, check_infected=True)
 


    # random
    # G, cat_vars, cont_vars, cat_unique_levels = load_random_vaccine(n=n, return_cat_cont_split=True)
    # H, network_list, cat_vars, cont_vars, cat_unique_levels = quarantine_dgm_time_series(G, restricted=restricted, 
    #                                                                                      time_limit=10, inf_duration=5, quarantine_period=2,
    #                                                                                      update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)
    # print(f'quarantine uniform n={n}: {cat_unique_levels}')

    # _check_if_quaratine_is_applied_correctly(network_list, check_I_ratio=True, check_quarantine=True, check_infected=True)

    # for graph in network_list:
    #     tmp_df = network_to_df(graph)
    #     print(pd.unique(tmp_df['quarantine']))
    #     print((tmp_df['quarantine'] == 1).sum()) 

    # ### ground truth
    # results = {}
    # for pr_a in prop_treated:
    #     start = time.time()
    #     ground_truth_last, ground_truth_all = quarantine_dgm_truth(G, pr_a, shift=shift, restricted=restricted,
    #                                                                time_limit=10, inf_duration=5, quarantine_period=2,
    #                                                                percent_candidates=0.3, mode='all',
    #                                                                random_seed=3407) 
    #     end = time.time()
    #     print(f'time elapes: {end - start} sec')
    #     print(f'pr_a: {pr_a}')
    #     print(f'ground truth: {ground_truth_last}')
    #     print(f'grond truth all time point: {ground_truth_all}')
    #     results[pr_a] = ground_truth_last
    # print(f'n:{n} | restricted:{restricted} | shift: {shift} | mode: None')
    # print(results)
    # results['n'] = n
    # results['restricted'] = restricted
    # results['shift'] = shift
    # with open("sample.json", "w") as outfile: 
    #     json.dump(results, outfile)

    # # save results
    # results = {}
    # pr_a = 0.05
    # start = time.time()
    # ground_truth_last, ground_truth_all = quarantine_dgm_truth(G, pr_a, shift=shift, restricted=restricted,
    #                                                            time_limit=10, inf_duration=5,
    #                                                            percent_candidates=0.3, mode='bottom',
    #                                                            random_seed=3407) 
    # end = time.time()
    # print(end - start)
    # results['n'] = n
    # results['restricted'] = restricted
    # results['shift'] = shift
    # with open("sample.json", "w") as outfile: 
    #     json.dump(results, outfile)
    
    # # loop    
    # print('+++++++++++++++++++++++++++++++++++++++ uniform graph +++++++++++++++++++++++++++++++++++++++')
    # for n in [500, 1000, 2000]:
    #     for shift in [True, False]:
    #         if shift:
    #             prop_treated = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]

    #             G, cat_vars, cont_vars, cat_unique_levels = load_uniform_vaccine(n=n, return_cat_cont_split=True)

    #             results = {}
    #             for pr_a in prop_treated:
    #                 ground_truth_last, ground_truth_all = quarantine_dgm_truth(G, pr_a, shift=shift, restricted=False,
    #                                                                             time_limit=10, inf_duration=5, quarantine_period=2,
    #                                                                             percent_candidates=0.3, mode='bottom',
    #                                                                             random_seed=3407) 
    #                 print(f'pr_a: {pr_a}')
    #                 print(f'ground truth: {ground_truth_last}')
    #                 print(f'grond truth all time point: {ground_truth_all}')
    #                 results[pr_a] = ground_truth_last
    #             print(f'n:{n} | shift: {shift} | mode: None')
    #             print(results)
    #             print()
        
    #         else:
    #             prop_treated = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    #                             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] 
    #             for mode in ['top', 'bottom', 'all']:               
    #                 G, cat_vars, cont_vars, cat_unique_levels = load_uniform_vaccine(n=n, return_cat_cont_split=True)

    #                 results = {}
    #                 for pr_a in prop_treated:
    #                     ground_truth_last, ground_truth_all = quarantine_dgm_truth(G, pr_a, shift=shift, restricted=False,
    #                                                                                 time_limit=10, inf_duration=5, quarantine_period=2,
    #                                                                                 percent_candidates=0.3, mode=mode,
    #                                                                                 random_seed=3407) 
    #                     print(f'pr_a: {pr_a}')
    #                     print(f'ground truth: {ground_truth_last}')
    #                     print(f'grond truth all time point: {ground_truth_all}')
    #                     results[pr_a] = ground_truth_last
    #                 print(f'n:{n} | shift: {shift} | mode: {mode}')
    #                 print(results)
    #                 print()
    # print()
    # print('+++++++++++++++++++++++++++++++++++++++ random graph +++++++++++++++++++++++++++++++++++++++')
    # # for n in [500, 1000, 2000]:
    # for n in [1000, 2000]:
    # # for n in [2000]:
    #     for restricted in [True, False]:
    #         for shift in [True, False]:
    #         # for shift in [False]:
    #             if shift:
    #                 prop_treated = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]

    #                 G, cat_vars, cont_vars, cat_unique_levels = load_random_vaccine(n=n, return_cat_cont_split=True)

    #                 results = {}
    #                 for pr_a in prop_treated:
    #                     ground_truth_last, ground_truth_all = quarantine_dgm_truth(G, pr_a, shift=shift, restricted=restricted,
    #                                                                             time_limit=10, inf_duration=5, quarantine_period=2,
    #                                                                             percent_candidates=0.3, mode='bottom',
    #                                                                             random_seed=3407) 
    #                     print(f'pr_a: {pr_a}')
    #                     print(f'ground truth: {ground_truth_last}')
    #                     print(f'grond truth all time point: {ground_truth_all}')
    #                     results[pr_a] = ground_truth_last
    #                 print(f'n:{n} | restricted:{restricted} | shift: {shift} | mode: None')
    #                 print(results)
    #                 print()
            
    #             else:
    #                 prop_treated = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    #                                 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] 
    #                 for mode in ['top', 'bottom', 'all']:               
    #                 # for mode in ['bottom', 'all']:               
    #                     G, cat_vars, cont_vars, cat_unique_levels = load_random_vaccine(n=n, return_cat_cont_split=True)

    #                     results = {}
    #                     for pr_a in prop_treated:
    #                         ground_truth_last, ground_truth_all = quarantine_dgm_truth(G, pr_a, shift=shift, restricted=restricted,
    #                                                                                    time_limit=10, inf_duration=5, quarantine_period=2,
    #                                                                                    percent_candidates=0.3, mode=mode,
    #                                                                                    random_seed=3407) 
    #                         print(f'pr_a: {pr_a}')
    #                         print(f'ground truth: {ground_truth_last}')
    #                         print(f'grond truth all time point: {ground_truth_all}')
    #                         results[pr_a] = ground_truth_last
    #                     print(f'n:{n} | restricted:{restricted} | shift: {shift} | mode: {mode}')
    #                     print(results)
    #                     print()
