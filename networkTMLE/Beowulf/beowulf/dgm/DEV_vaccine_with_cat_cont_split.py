import random
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import logistic

from beowulf.dgm.utils import (network_to_df, fast_exp_map, exposure_restrictions,
                               odds_to_probability, probability_to_odds)

def vaccine_dgm(network, restricted=False,
                time_limit=10, inf_duration=5,
                update_split=False, cat_vars=[], cont_vars=[], cat_unique_levels={}):
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
    pr_a = logistic.cdf(-1.9 + 1.75*data['A'] + 1.*data['H']
                        + 1.*data['H_sum'] + 1.3*data['A_sum'] - 0.65*data['F'])
    vaccine = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['vaccine'] = vaccine
    if restricted:  # if we are in the restricted scenarios
        attrs = exposure_restrictions(network=network.graph['label'], exposure='vaccine',
                                      n=nx.number_of_nodes(graph))
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['vaccine']))

    # print("Pr(V):", np.mean(vaccine))
    for n in graph.nodes():
        graph.nodes[n]['vaccine'] = int(data.loc[data.index == n, 'vaccine'].values)

    # Running outbreak simulation
    graph = _outbreak_(graph, duration=inf_duration, limit=time_limit)

    if update_split:
        cat_vars.append('vaccine')
        cat_unique_levels['vaccine'] = pd.unique(data['vaccine'].astype('int')).max() + 1
        return graph, cat_vars, cont_vars, cat_unique_levels
    else:
        return graph

def vaccine_dgm_time_series(network, restricted=False,
                            time_limit=10, inf_duration=5,
                            update_split=False, cat_vars=[], cont_vars=[], cat_unique_levels={}):
    """
    besides the final graph object,
    return a list of  graph objects for each time point in time_limit.
    
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
    pr_a = logistic.cdf(-1.9 + 1.75*data['A'] + 1.*data['H']
                        + 1.*data['H_sum'] + 1.3*data['A_sum'] - 0.65*data['F'])
    vaccine = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['vaccine'] = vaccine
    if restricted:  # if we are in the restricted scenarios
        attrs = exposure_restrictions(network=network.graph['label'], exposure='vaccine',
                                      n=nx.number_of_nodes(graph))
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['vaccine']))

    # print("Pr(V):", np.mean(vaccine))
    for n in graph.nodes():
        graph.nodes[n]['vaccine'] = int(data.loc[data.index == n, 'vaccine'].values)

    # Running outbreak simulation
    graph, graph_saved_by_time = _outbreak_time_series(graph, duration=inf_duration, limit=time_limit)

    if update_split:
        cat_vars.append('vaccine')
        cat_unique_levels['vaccine'] = pd.unique(data['vaccine'].astype('int')).max() + 1
        # cat_vars.append('D') # outcome variable D is generated using _outbreak_time_series()
        # cat_unique_levels['D'] = pd.unique(network_to_df(graph)['D'].astype('int')).max() + 1
        return graph, cat_vars, cont_vars, cat_unique_levels, graph_saved_by_time
    else:
        return graph, graph_saved_by_time

def vaccine_dgm_truth(network, pr_a, shift=False, restricted=False,
                      time_limit=10, inf_duration=5):
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
        prob = logistic.cdf(-1.9 + 1.75 * data['A'] + 1. * data['H']
                            + 1. * data['H_sum'] + 1.3 * data['A_sum'] - 0.65 * data['F'])
        odds = probability_to_odds(prob)
        pr_a = odds_to_probability(np.exp(np.log(odds) + pr_a))

    vaccine = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['vaccine'] = vaccine
    if restricted:  # if we are in the restricted scenarios
        attrs = exposure_restrictions(network=network.graph['label'], exposure='vaccine',
                                      n=nx.number_of_nodes(graph))
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['vaccine']))

    for n in graph.nodes():
        graph.nodes[n]['vaccine'] = int(data.loc[data.index == n, 'vaccine'].values)

    # Running Data Generating Mechanism for Y
    graph = _outbreak_(graph, duration=inf_duration, limit=time_limit)
    dis = []
    for nod, d in graph.nodes(data=True):
        dis.append(d['D'])
    return np.mean(dis)


def _outbreak_(graph, duration, limit):
    """Outbreak simulation script in a single function"""
    # Adding node attributes
    for n, d in graph.nodes(data=True):
        d['D'] = 0
        d['R'] = 0
        d['t'] = 0
        d['I'] = 0

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
    time = 0
    while time < limit:  # Simulate outbreaks until time-step limit is reached
        time += 1
        print(f'begin time {time}')
        for inf in sorted(infected, key=lambda _: random.random()):
            print(inf)
            # Book-keeping for infected nodes
            graph.nodes[inf]['I'] = 1
            graph.nodes[inf]['D'] = 1
            graph.nodes[inf]['t'] += 1
            if graph.nodes[inf]['t'] > duration:
                graph.nodes[inf]['I'] = 0         # Node is no longer infectious after this loop
                graph.nodes[inf]['R'] = 1         # Node switches to Recovered
                infected.remove(inf)

            # Attempt infections of neighbors
            for contact in nx.neighbors(graph, inf):
                if graph.nodes[contact]["D"] == 1:
                    pass
                else:
                    pr_y = logistic.cdf(- 2.5
                                        - 1.0*graph.nodes[contact]['vaccine']
                                        - 0.2*graph.nodes[inf]['vaccine']
                                        + 1.0*graph.nodes[contact]['A']
                                        - 0.2*graph.nodes[contact]['H'])
                    if np.random.binomial(n=1, p=pr_y, size=1):
                        graph.nodes[contact]['I'] = 1
                        graph.nodes[contact]["D"] = 1
                        infected.append(contact)
        print(f'end of time {time}: {infected}')
    return graph




def _outbreak_time_series(graph, duration, limit):
    """Outbreak simulation script in a single function"""
    # Adding node attributes
    for n, d in graph.nodes(data=True):
        d['D'] = 0
        d['R'] = 0
        d['t'] = 0

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
    time = 0
    while time < limit:  # Simulate outbreaks until time-step limit is reached
        time += 1
        for inf in sorted(infected, key=lambda _: random.random()):
            # Book-keeping for infected nodes
            graph.nodes[inf]['D'] = 1
            graph.nodes[inf]['t'] += 1
            if graph.nodes[inf]['t'] > duration:
                graph.nodes[inf]['I'] = 0         # Node is no longer infectious after this loop
                graph.nodes[inf]['R'] = 1         # Node switches to Recovered
                infected.remove(inf)

            # Attempt infections of neighbors
            for contact in nx.neighbors(graph, inf):
                if graph.nodes[contact]["D"] == 1:
                    pass
                else:
                    pr_y = logistic.cdf(- 2.5
                                        - 1.0*graph.nodes[contact]['vaccine']
                                        - 0.2*graph.nodes[inf]['vaccine']
                                        + 1.0*graph.nodes[contact]['A']
                                        - 0.2*graph.nodes[contact]['H'])
                    if np.random.binomial(n=1, p=pr_y, size=1):
                        graph.nodes[contact]['I'] = 1
                        graph.nodes[contact]["D"] = 1
                        infected.append(contact)
        graph_saved_by_time.append(graph.copy())

    return graph, graph_saved_by_time

if __name__ == '__main__':

    from beowulf import load_uniform_vaccine, load_random_vaccine

    n=500

    # uniform
    G, cat_vars, cont_vars, cat_unique_levels = load_uniform_vaccine(n=n, return_cat_cont_split=True)
    H, cat_vars, cont_vars, cat_unique_levels = vaccine_dgm(G, restricted=False, 
                                                           update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)
    print(f'Vaccine uniform n={n}: {cat_unique_levels}')

    # random
    G, cat_vars, cont_vars, cat_unique_levels = load_random_vaccine(n=n, return_cat_cont_split=True)
    H, cat_vars, cont_vars, cat_unique_levels = vaccine_dgm(G, restricted=False, 
                                                           update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)
    print(f'Vaccine random n={n}: {cat_unique_levels}')

    ################ social distancing ####################
    def get_removed_edges(data, graph, social_distancing, p_remove_connection=0.1):
        selected_nodes = np.arange(data.shape[0])[social_distancing==1]
        
        removed_edges = []
        for a, b, attrs in graph.edges(data=True):
            if a in selected_nodes or b in selected_nodes:
                if random.random() < p_remove_connection:  
                    # print(f'Edge {a}-{b} is removed')
                    removed_edges.append((a,b))
        return removed_edges

    def _outbreak_social_dist(true_graph, training_graph, duration, limit):
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
            for inf in sorted(infected, key=lambda _: random.random()):
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
                        pr_y = logistic.cdf(- 4.5
                                            + 1.0*true_graph.nodes[contact]['A']
                                            - 0.5*true_graph.nodes[contact]['H'])
                        # print(pr_y)
                        if np.random.binomial(n=1, p=pr_y, size=1):
                            true_graph.nodes[contact]['I'] = 1
                            true_graph.nodes[contact]["D"] = 1

                            training_graph.nodes[contact]['I'] = 1
                            training_graph.nodes[contact]["D"] = 1
                            infected.append(contact)

        return true_graph, training_graph
    
    def social_dist_dgm(network, restricted=False,
                        time_limit=10, inf_duration=5,
                        update_split=False, cat_vars=[], cont_vars=[], cat_unique_levels={}):
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
        pr_a = logistic.cdf(- 5.35 
                            + 1.0*data['A'] + 0.5*data['H']
                            + 0.3*data['H_sum'] + 0.1*data['A_sum'] 
                            + 1.5*data['F'])
        social_distancing = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
        data['social_dist'] = social_distancing

        if restricted:  # if we are in the restricted scenarios
            # Use the restriction design for vaccine
            attrs = exposure_restrictions(network=network.graph['label'], exposure='vaccine',
                                        n=nx.number_of_nodes(graph))
            data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['social_dist']))

        # print("Pr(V):", np.mean(vaccine))
        for n in graph.nodes():
            graph.nodes[n]['social_dist'] = int(data.loc[data.index == n, 'social_dist'].values)

        # Running outbreak simulation
        # Remove 10% of connections if apply social distancing
        true_graph = graph.copy()
        true_graph.remove_edges_from(get_removed_edges(data, graph, social_distancing, p_remove_connection=0.1))
        
        graph = _outbreak_social_dist(true_graph, graph, duration=inf_duration, limit=time_limit)

        if update_split:
            cat_vars.append('social_dist')
            cat_unique_levels['social_dist'] = pd.unique(data['social_dist'].astype('int')).max() + 1
            return true_graph, graph, cat_vars, cont_vars, cat_unique_levels
        else:
            return true_graph, graph

    true_graph, graph, cat_vars, cont_vars, cat_unique_levels = social_dist_dgm(G, restricted=False, 
                                        time_limit=10, inf_duration=5,
                                        update_split=True, cat_vars=cat_vars, cont_vars=cont_vars, cat_unique_levels=cat_unique_levels)
    
    cat_vars
    cat_unique_levels


    ################# quarantine policy: time series ####################
    # Params
    network=G
    restricted=False
    time_limit=10
    inf_duration=5
    update_split=False
    cat_vars=[]
    cont_vars=[]
    cat_unique_levels={}

    # graph = network.copy()
    # data = network_to_df(graph)

    # adj_matrix = nx.adjacency_matrix(graph, weight=None)
    # data['A_sum'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='sum')
    # data['A_mean'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='mean')
    # data['H_sum'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='sum')
    # data = pd.merge(data, pd.DataFrame.from_dict(dict(network.degree),
    #                                              orient='index').rename(columns={0: 'F'}),
    #                 how='left', left_index=True, right_index=True)

    # # Running Data Generating Mechanism for A
    # pr_a = logistic.cdf(-1.9 + 1.75*data['A'] + 1.*data['H']
    #                     + 1.*data['H_sum'] + 1.3*data['A_sum'] - 0.65*data['F'])
    # vaccine = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    # data['vaccine'] = vaccine
    # if restricted:  # if we are in the restricted scenarios
    #     attrs = exposure_restrictions(network=network.graph['label'], exposure='vaccine',
    #                                   n=nx.number_of_nodes(graph))
    #     data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['vaccine']))

    # # print("Pr(V):", np.mean(vaccine))
    # for n in graph.nodes():
    #     graph.nodes[n]['vaccine'] = int(data.loc[data.index == n, 'vaccine'].values)
    
    # graph = _outbreak_(graph, duration=inf_duration, limit=time_limit)
    # tmp_data = network_to_df(graph)
    # tmp_data['I'].sum()

    # tmp_data[tmp_data['I']==1].index
    # tmp_data[tmp_data['t']==6].index
    # tmp_data[tmp_data['D']==1].index
    

    # Code
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
    time = 0
    while time < limit:  # Simulate outbreaks until time-step limit is reached
        time += 1
        edge_to_remove = []
        for inf in sorted(infected, key=lambda _: random.random()):
            # Book-keeping for infected nodes
            graph.nodes[inf]['I'] = 1
            graph.nodes[inf]['D'] = 1
            graph.nodes[inf]['t'] += 1

            if graph.nodes[inf]['t'] > duration:
                graph.nodes[inf]['I'] = 0         # Node is no longer infectious after this loop
                graph.nodes[inf]['R'] = 1         # Node switches to Recovered
                infected.remove(inf)
            
            # calculate summary measure
            data = network_to_df(graph)

            adj_matrix = nx.adjacency_matrix(graph, weight=None)
            data['A_sum'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='sum')
            data['A_mean'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='mean')
            data['H_sum'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='sum')
            data = pd.merge(data, pd.DataFrame.from_dict(dict(network.degree),
                                                        orient='index').rename(columns={0: 'F'}),
                            how='left', left_index=True, right_index=True)
            
        
            # get actions from current Infected individual, document as "quarantine"
            data['I_sum'] = fast_exp_map(adj_matrix, np.array(data['I']), measure='sum')
            data['I_ratio'] = data['I_sum'] / data['F'] # ratio of infected neighbors
            
            pr_a = logistic.cdf(- 5.35 
                                + 1.0*data['A'] + 0.5*data['H']
                                + 0.3*data['H_sum'] + 0.1*data['A_sum'] 
                                + 2.1*data['I_ratio'])
            quarantine = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
            data['quarantine'] = quarantine
            # apply quarantine if the nodes is a contact of inf, implemented in the loop below


            # Attempt infections of neighbors
            for contact in nx.neighbors(graph, inf):
                if graph.nodes[contact]["D"] == 1:
                    pass
                else:
                    # apply quarantine
                    graph.nodes[contact]['quarantine'] = int(data.loc[data.index == contact, 'quarantine'].values)
                    if graph.nodes[contact]['quarantine'] == 1:
                        edge_to_remove.append((inf, contact))

                    # probability of infection is not associated with quarantine directly, 
                    # but through the change in graph edges in the next time step    
                    pr_y = logistic.cdf(- 4.5
                                        + 1.0*graph.nodes[contact]['A']
                                        - 0.5*graph.nodes[contact]['H'])
                    if np.random.binomial(n=1, p=pr_y, size=1):
                        graph.nodes[contact]['I'] = 1
                        graph.nodes[contact]["D"] = 1
                        infected.append(contact)

        graph_saved_by_time.append(graph.copy())
        graph.remove_edges_from(edge_to_remove) # remove quarantined edges

        









    adj_matrix = nx.adjacency_matrix(graph, weight=None)
    data['A_sum'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='sum')
    data['A_mean'] = fast_exp_map(adj_matrix, np.array(data['A']), measure='mean')
    data['H_sum'] = fast_exp_map(adj_matrix, np.array(data['H']), measure='sum')
    data = pd.merge(data, pd.DataFrame.from_dict(dict(network.degree),
                                                 orient='index').rename(columns={0: 'F'}),
                    how='left', left_index=True, right_index=True)

    # Running Data Generating Mechanism for A
    # pr_a = logistic.cdf(-1.9 + 1.75*data['A'] + 1.*data['H']
    #                     + 1.*data['H_sum'] + 1.3*data['A_sum'] - 0.65*data['F'])
    # vaccine = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))

    pr_a = logistic.cdf(- 5.35 
                        + 1.0*data['A'] + 0.5*data['H']
                        + 0.3*data['H_sum'] + 0.1*data['A_sum'] 
                        + 1.5*data['F'])
    social_distancing = np.random.binomial(n=1, p=pr_a, size=nx.number_of_nodes(graph))
    data['social_dist'] = social_distancing

    if restricted:  # if we are in the restricted scenarios
        attrs = exposure_restrictions(network=network.graph['label'], exposure='vaccine',
                                      n=nx.number_of_nodes(graph))
        data.update(pd.DataFrame(list(attrs.values()), index=list(attrs.keys()), columns=['vaccine']))

    # print("Pr(V):", np.mean(vaccine))
    for n in graph.nodes():
        graph.nodes[n]['social_dist'] = int(data.loc[data.index == n, 'social_dist'].values)

    # Running outbreak simulation
    # Remove 10% of connections if apply social distancing
    new_graph = graph.copy()
    new_graph.remove_edges_from(get_removed_edges(data, graph, social_distancing, p_remove_connection=0.1))
    true_graph, training_graph = _outbreak_(new_graph, graph, duration=inf_duration, limit=time_limit)

    true_edges = [(a, b) for a, b, attrs in true_graph.edges(data=True)]
    train_edges = [(a, b) for a, b, attrs in training_graph.edges(data=True)] 

    len(true_edges)
    len(train_edges)