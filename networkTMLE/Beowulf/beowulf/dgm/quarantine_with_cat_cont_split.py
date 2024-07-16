# import random
import math
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic

from beowulf.dgm.utils import (network_to_df, fast_exp_map, exposure_restrictions,
                               odds_to_probability, probability_to_odds)


################# quarantine policy: time series ####################
def update_I_ratio(data, graph):
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

def update_summary_measures(data, graph):
    ''' Update I_ratio and summary measures for the graph and data object for each time step
    NOTE: this update is to keep up with the change in graph connections
    ''' 
    # retreive updated I, D, t, R from graph node features
    data = network_to_df(graph)

    # retreive updated graph connections
    adj_matrix = nx.adjacency_matrix(graph, weight=None)

    # update summary measures
    data['A_sum'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='sum')
    data['H_sum'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='sum')

    # update I_ratio
    data, graph = update_I_ratio(data, graph)
    
    return data, graph


# def update_summary_measures(data, graph):
#     ''' Update summary measures for the graph data
#     because I_sum has to be updated, summary data is updated for every infected node''' 
#     # retreive updated I, D, t, R from graph node features
#     data = network_to_df(graph)

#     # retreive updated graph connections
#     adj_matrix = nx.adjacency_matrix(graph, weight=None)

#     # update I ratio and summary measures
#     data['A_sum'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='sum')
#     data['H_sum'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='sum')
#     # calculate I_ratio
#     data = pd.merge(data, pd.DataFrame.from_dict(dict(graph.degree),
#                                                  orient='index').rename(columns={0: 'degree'}),
#                                                  how='left', left_index=True, right_index=True)
#     data['I_sum'] = fast_exp_map(adj_matrix, np.array(data['I']), measure='sum')
#     data['I_ratio'] = data['I_sum'] / data['degree']  # ratio of infected neighbors
#     data['I_ratio'] = data['I_ratio'].fillna(0)  # fill in 0 for nodes with no neighbors

#     # add I_ratio to graph data
#     if nx.is_directed(graph):
#         raise NotImplementedError("Directed graph is not supported yet")
#     else:
#         nx.set_node_attributes(graph, dict(data['I_ratio']), 'I_ratio')
    
#     return data, graph

def update_pr_a(data, graph, restricted, edge_recorder, time_step, rng):
    ''' Update the probability of quarantine action for every quarantine_period NOT for each inf'''
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
    if time_step - quarantine_period >= 0: 
        graph.add_edges_from(edge_recorder[time_step-quarantine_period]) 
    graph.remove_edges_from(edge_recorder[time_step]) # remove current quarantined edges
    # if an edge tuple is not in graph, it will be ignore silently. 
    # Match our purpose with duplicated edges that they can only be removed once.
    return graph

# def simulate_infection_of_immediate_neighbors(graph, inf, infected, rng):
#     '''Simulate infections of immediate neighbors'''
#     for contact in nx.neighbors(graph, inf):
#         if graph.nodes[contact]["D"] == 1:
#             pass
#         else:
#             # probability of infection associated with quarantine and I_ratio directly
#             pr_y = logistic.cdf(- 1.2
#                                 + 0.5*graph.nodes[contact]['I_ratio']
#                                 + 0.5*graph.nodes[inf]['I_ratio']
#                                 + 0.8*graph.nodes[contact]['I_ratio']*graph.nodes[inf]['I_ratio']
#                                 - 1.2*graph.nodes[contact]['quarantine']
#                                 - 1.2*graph.nodes[inf]['quarantine']
#                                 + 1.2*graph.nodes[contact]['A']
#                                 + 1.2*graph.nodes[contact]['A']**2
#                                 - 0.1*graph.nodes[contact]['H'])
#             # print(pr_y)
#             if rng.binomial(n=1, p=pr_y, size=1):
#                 graph.nodes[contact]['I'] = 1
#                 graph.nodes[contact]["D"] = 1
#                 infected.append(contact)
#     return graph, infected

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
    edge_recorder = {key:[] for key in range(0, time_limit, quarantine_period)} 
    # record edge_to_remove to be add back after the quarantine period has passed

    time = 0
    while time < time_limit:  # Simulate outbreaks until time-step limit is reached
        print(f'time {time+1}')
        # 1. Update data from previous graph 
        # 2. Update I_ratio and summary measures in data based on previous graph
        # 3. Synchronize graph node features using the updated data for next iteration
        data, graph = update_summary_measures(data, graph)

        for inf in sorted(infected, key=lambda _: rng.random()):
            # Book-keeping for infected nodes
            graph.nodes[inf]['I'] = 1
            graph.nodes[inf]['D'] = 1
            graph.nodes[inf]['t'] += 1

            if graph.nodes[inf]['t'] > inf_duration:
                graph.nodes[inf]['I'] = 0         # Node is no longer infectious after this loop
                graph.nodes[inf]['R'] = 1         # Node switches to Recovered
                infected.remove(inf)            
            
            # Update I_ratio for the graph data
            data, graph = update_I_ratio(data, graph)

            # Simulate infections of immediate neighbors
            graph, infected = simulate_infection_of_immediate_neighbors(graph, inf, infected, rng) 

        # A quarantined case stays for quarantine_period days, and released after that.
        # Thus, pr_a should be re-calcualted for every quarantine_period days. 
        if time % quarantine_period == 0:
            data, graph, edge_recorder = update_pr_a(data, graph, restricted, edge_recorder, time, rng)
        
        # Save graph at current time step
        graph_saved_by_time.append(graph.copy())        

        # Update graph for next time step
        if time % quarantine_period == 0:
            graph = update_edge_in_graph(graph, edge_recorder, time, quarantine_period)
        
        # Increment time step
        time += 1
        
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
            candidates_nodes = data.nlargest(num_candidates, 'degree').index # super-spreader
        elif mode == 'bottom':
            num_candidates = math.ceil(data.shape[0] * percent_candidates)
            candidates_nodes = data.nsmallest(num_candidates, 'degree').index # super-defender
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
        edge_recorder[time_step].extend(list(nx.edges(graph, data[data['quarantine']==1].index)))
    # # for-loop version
    # for n in graph.nodes():
    #     graph.nodes[n]['quarantine'] = int(data.loc[data.index == n, 'quarantine'].values)
    #     if n in data[data['quarantine']==1].index: # remove all immediate neighbors
    #         for neighbor in graph.neighbors(n):
    #             edge_recorder[time_step].append((neighbor, n))
    # return data, graph, edge_recorder, quarantine.sum()
    return data, graph, edge_recorder

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
    edge_recorder = {key:[] for key in range(0, time_limit, quarantine_period)} 
    # record edge_to_remove to be add back after the quarantine period has passed
    
    time = 0
    while time < time_limit:  # Simulate outbreaks until time-step limit is reached
        print(f'time {time}')

        # 1. Update data from previous graph 
        # 2. Update I_ratio and summary measures in data based on previous graph
        # 3. Synchronize graph node features using the updated data for next iteration
        data, graph = update_summary_measures(data, graph)

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

            # Update I_ratio for the graph data
            data, graph = update_I_ratio(data, graph)

            # Simulate infections of immediate neighbors
            graph, infected = simulate_infection_of_immediate_neighbors(graph, inf, infected, rng)
        
        # Apply quarantine actions
        if time % quarantine_period == 0:
            data, graph, edge_recorder = apply_quarantine_action(data, graph, shift, restricted, edge_recorder, time, rng,
                                                                 pr_a, percent_candidates, mode)
            # print(f'time {time}: inf {inf} from {infected}')
        
        # Save graph at current time step
        graph_saved_by_time.append(graph.copy())
        
        # Update graph for next time step
        if time % quarantine_period == 0:
            graph = update_edge_in_graph(graph, edge_recorder, time, quarantine_period)
                    
        # Increment time step
        time += 1

    dis_save_by_time = []
    for g in graph_saved_by_time:
        dis = [] 
        for nod, d in g.nodes(data=True):
            dis.append(d['D'])
        dis_save_by_time.append(np.mean(dis))

    # return np.mean(dis), dis_save_by_time, graph_saved_by_time, edge_recorder # save last time point and the whole time series
    return np.mean(dis), dis_save_by_time # save last time point and the whole time series

if __name__ == '__main__':

    from beowulf import load_uniform_vaccine, load_random_vaccine
    import json 
    import time

    # n=500
    # restricted=False # only for random network

    # # proportion to be removed: if no node selection is made, the proportion is the same for all nodes
    # shift=False
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

    # # quarantine
    # # uniform
    # # start_time = time.time()
    # G, cat_vars, cont_vars, cat_unique_levels = load_uniform_vaccine(n=n, return_cat_cont_split=True)
    # # end_time = time.time()
    # # print(f'load_uniform_vaccin(): {end_time - start_time:.2f} sec(s)')

    # # start_time = time.time()
    # H, network_list, cat_vars, cont_vars, cat_unique_levels = quarantine_dgm_time_series(G, restricted=False, 
    #                                                                                      time_limit=10, inf_duration=5, quarantine_period=2,
    #                                                                                      update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels,
    #                                                                                      random_seed=3407)
    # # end_time = time.time()
    # # print(f'quarantine_dgm_time_series(): {end_time - start_time:.2f} sec(s)')
    # print(f'quarantine uniform n={n}: {cat_unique_levels}')

    # ### ground truth
    # results = {}
    # for pr_a in prop_treated:
    #     start_time = time.time()
    #     ground_truth_last, ground_truth_all, ground_truth_list, edge_recorder = quarantine_dgm_truth(G, pr_a, shift=shift, restricted=False,
    #                                                                time_limit=10, inf_duration=5, quarantine_period=2,
    #                                                                percent_candidates=0.3, mode='top',
    #                                                                random_seed=3407) 
    #     end_time = time.time()
    #     print(f'time elapes: {end_time - start_time} sec(s)')
    #     print(f'pr_a: {pr_a}')
    #     print(f'ground truth: {ground_truth_last}')
    #     print(f'grond truth all time point: {ground_truth_all}')
    #     results[pr_a] = ground_truth_last
    #     # _check_if_quaratine_is_applied_correctly(ground_truth_list, check_I_ratio=True, check_quarantine=False, check_infected=False)
    # print(f'n:{n} | shift: {shift}')
    # print(results)


    # loop    
    print('+++++++++++++++++++++++++++++++++++++++ uniform graph +++++++++++++++++++++++++++++++++++++++')
    for n in [500, 1000, 2000]:
        for shift in [True, False]:
            if shift:
                prop_treated = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]

                G, cat_vars, cont_vars, cat_unique_levels = load_uniform_vaccine(n=n, return_cat_cont_split=True)

                results = {}
                for pr_a in prop_treated:
                    ground_truth_last, ground_truth_all = quarantine_dgm_truth(G, pr_a, shift=shift, restricted=False,
                                                                                time_limit=10, inf_duration=5, quarantine_period=2,
                                                                                percent_candidates=0.3, mode='bottom',
                                                                                random_seed=3407) 
                    print(f'pr_a: {pr_a}')
                    print(f'ground truth: {ground_truth_last}')
                    print(f'grond truth all time point: {ground_truth_all}')
                    results[pr_a] = ground_truth_last
                print(f'n:{n} | shift: {shift} | mode: None')
                print(results)
                print()
        
            else:
                prop_treated = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] 
                for mode in ['top', 'bottom', 'all']:               
                    G, cat_vars, cont_vars, cat_unique_levels = load_uniform_vaccine(n=n, return_cat_cont_split=True)

                    results = {}
                    for pr_a in prop_treated:
                        ground_truth_last, ground_truth_all = quarantine_dgm_truth(G, pr_a, shift=shift, restricted=False,
                                                                                    time_limit=10, inf_duration=5, quarantine_period=2,
                                                                                    percent_candidates=0.3, mode=mode,
                                                                                    random_seed=3407) 
                        print(f'pr_a: {pr_a}')
                        print(f'ground truth: {ground_truth_last}')
                        print(f'grond truth all time point: {ground_truth_all}')
                        results[pr_a] = ground_truth_last
                    print(f'n:{n} | shift: {shift} | mode: {mode}')
                    print(results)
                    print()
    print()
    print('+++++++++++++++++++++++++++++++++++++++ random graph +++++++++++++++++++++++++++++++++++++++')
    for n in [500, 1000, 2000]:
    # for n in [1000, 2000]:
    # for n in [2000]:
        for restricted in [True, False]:
            for shift in [True, False]:
            # for shift in [False]:
                if shift:
                    prop_treated = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]

                    G, cat_vars, cont_vars, cat_unique_levels = load_random_vaccine(n=n, return_cat_cont_split=True)

                    results = {}
                    for pr_a in prop_treated:
                        ground_truth_last, ground_truth_all = quarantine_dgm_truth(G, pr_a, shift=shift, restricted=restricted,
                                                                                time_limit=10, inf_duration=5, quarantine_period=2,
                                                                                percent_candidates=0.3, mode='bottom',
                                                                                random_seed=3407) 
                        print(f'pr_a: {pr_a}')
                        print(f'ground truth: {ground_truth_last}')
                        print(f'grond truth all time point: {ground_truth_all}')
                        results[pr_a] = ground_truth_last
                    print(f'n:{n} | restricted:{restricted} | shift: {shift} | mode: None')
                    print(results)
                    print()
            
                else:
                    prop_treated = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] 
                    for mode in ['top', 'bottom', 'all']:               
                    # for mode in ['bottom', 'all']:               
                        G, cat_vars, cont_vars, cat_unique_levels = load_random_vaccine(n=n, return_cat_cont_split=True)

                        results = {}
                        for pr_a in prop_treated:
                            ground_truth_last, ground_truth_all = quarantine_dgm_truth(G, pr_a, shift=shift, restricted=restricted,
                                                                                       time_limit=10, inf_duration=5, quarantine_period=2,
                                                                                       percent_candidates=0.3, mode=mode,
                                                                                       random_seed=3407) 
                            print(f'pr_a: {pr_a}')
                            print(f'ground truth: {ground_truth_last}')
                            print(f'grond truth all time point: {ground_truth_all}')
                            results[pr_a] = ground_truth_last
                        print(f'n:{n} | restricted:{restricted} | shift: {shift} | mode: {mode}')
                        print(results)
                        print()