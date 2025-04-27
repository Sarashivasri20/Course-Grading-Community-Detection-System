import networkx as nx
import igraph as ig
import leidenalg
import community.community_louvain as community_louvain
from networkx.algorithms.community import greedy_modularity_communities
from collections import defaultdict

def nx_subgraph_communities(G, communities):
    """Helper to extract subgraphs from node communities"""
    subgraphs = []
    for comm in communities:
        subgraphs.append(list(comm))
    return subgraphs

def validate_weights(G):
    """Ensure the graph has numeric weights; convert or assign default weight of 1.0."""
    for u, v, data in G.edges(data=True):
        if 'weight' not in data:
            G[u][v]['weight'] = 1.0
        else:
            try:
                G[u][v]['weight'] = float(data['weight'])
            except (ValueError, TypeError):
                print(f"⚠️ Non-numeric weight found for edge ({u}, {v}): {data['weight']}. Setting to 1.0.")
                G[u][v]['weight'] = 1.0
    return G

def hybrid_greedy_louvain(G):
    G = validate_weights(G)
    greedy_comms = greedy_modularity_communities(G, weight='weight')
    hybrid_communities = []
    for comm_nodes in nx_subgraph_communities(G, greedy_comms):
        if len(comm_nodes) <= 2:
            hybrid_communities.append(comm_nodes)
            continue
        subgraph = G.subgraph(comm_nodes).copy()
        if subgraph.number_of_edges() == 0:
            hybrid_communities.append(comm_nodes)
            continue
        try:
            partition = community_louvain.best_partition(subgraph, weight='weight')
            louvain_comms = defaultdict(list)
            for node, comm_id in partition.items():
                louvain_comms[comm_id].append(node)
            hybrid_communities.extend(louvain_comms.values())
        except Exception as e:
            print(f"⚠️ Error in Louvain clustering for subgraph: {e}")
            hybrid_communities.append(comm_nodes)
    return hybrid_communities

def hybrid_greedy_walktrap(G):
    G = validate_weights(G)
    greedy_comms = greedy_modularity_communities(G, weight='weight')
    hybrid_communities = []
    for comm_nodes in nx_subgraph_communities(G, greedy_comms):
        if len(comm_nodes) <= 2:
            hybrid_communities.append(comm_nodes)
            continue
        subgraph = G.subgraph(comm_nodes).copy()
        if subgraph.number_of_edges() == 0:
            hybrid_communities.append(comm_nodes)
            continue
        # Convert edges to iGraph with explicit numeric weights
        ig_sub = ig.Graph.TupleList(
            [(str(u), str(v), {'weight': float(d.get('weight', 1.0))}) for u, v, d in subgraph.edges(data=True)],
            edge_attrs=['weight'],
            directed=False
        )
        try:
            walktrap = ig_sub.community_walktrap(weights='weight').as_clustering()
            walktrap_comms = [list(map(str, ig_sub.vs[cluster]['name'])) for cluster in walktrap]
            hybrid_communities.extend(walktrap_comms)
        except Exception as e:
            print(f"⚠️ Error in Walktrap clustering for subgraph: {e}")
            hybrid_communities.append(comm_nodes)
    return hybrid_communities

def hybrid_greedy_leiden(G):
    G = validate_weights(G)
    greedy_comms = greedy_modularity_communities(G, weight='weight')
    hybrid_communities = []
    for comm_nodes in nx_subgraph_communities(G, greedy_comms):
        if len(comm_nodes) <= 2:
            hybrid_communities.append(comm_nodes)
            continue
        subgraph = G.subgraph(comm_nodes).copy()
        if subgraph.number_of_edges() == 0:
            hybrid_communities.append(comm_nodes)
            continue
        # Convert edges to iGraph with explicit numeric weights
        ig_sub = ig.Graph.TupleList(
            [(str(u), str(v), {'weight': float(d.get('weight', 1.0))}) for u, v, d in subgraph.edges(data=True)],
            edge_attrs=['weight'],
            directed=False
        )
        try:
            leiden_partition = leidenalg.find_partition(ig_sub, leidenalg.ModularityVertexPartition)
            leiden_comms = [list(map(str, ig_sub.vs[cluster]['name'])) for cluster in leiden_partition]
            hybrid_communities.extend(leiden_comms)
        except Exception as e:
            print(f"⚠️ Error in Leiden clustering for subgraph: {e}")
            hybrid_communities.append(comm_nodes)
    return hybrid_communities

def hybrid_louvain_walktrap(G):
    G = validate_weights(G)
    partition = community_louvain.best_partition(G, weight='weight')
    louvain_comms = defaultdict(list)
    for node, comm_id in partition.items():
        louvain_comms[comm_id].append(node)
    hybrid_communities = []
    for comm_nodes in louvain_comms.values():
        if len(comm_nodes) <= 2:
            hybrid_communities.append(comm_nodes)
            continue
        subgraph = G.subgraph(comm_nodes).copy()
        if subgraph.number_of_edges() == 0:
            hybrid_communities.append(comm_nodes)
            continue
        # Convert edges to iGraph with explicit numeric weights
        ig_sub = ig.Graph.TupleList(
            [(str(u), str(v), {'weight': float(d.get('weight', 1.0))}) for u, v, d in subgraph.edges(data=True)],
            edge_attrs=['weight'],
            directed=False
        )
        try:
            walktrap = ig_sub.community_walktrap(weights='weight').as_clustering()
            walktrap_comms = [list(map(str, ig_sub.vs[cluster]['name'])) for cluster in walktrap]
            hybrid_communities.extend(walktrap_comms)
        except Exception as e:
            print(f"⚠️ Error in Walktrap clustering for subgraph: {e}")
            hybrid_communities.append(comm_nodes)
    return hybrid_communities

def hybrid_walktrap_leiden(G):
    G = validate_weights(G)
    ig_graph = ig.Graph.TupleList(
        [(str(u), str(v), {'weight': float(d.get('weight', 1.0))}) for u, v, d in G.edges(data=True)],
        edge_attrs=['weight'],
        directed=False
    )
    try:
        walktrap = ig_graph.community_walktrap(weights='weight').as_clustering()
        walktrap_comms = [list(map(str, ig_graph.vs[cluster]['name'])) for cluster in walktrap]
    except Exception as e:
        print(f"⚠️ Error in Walktrap clustering: {e}")
        walktrap_comms = [list(G.nodes())]  # Fallback to single community

    hybrid_communities = []
    for comm_nodes in walktrap_comms:
        if len(comm_nodes) <= 2:
            hybrid_communities.append(comm_nodes)
            continue
        subgraph = G.subgraph(comm_nodes).copy()
        if subgraph.number_of_edges() == 0:
            hybrid_communities.append(comm_nodes)
            continue
        # Convert edges to iGraph with explicit numeric weights
        ig_sub = ig.Graph.TupleList(
            [(str(u), str(v), {'weight': float(d.get('weight', 1.0))}) for u, v, d in subgraph.edges(data=True)],
            edge_attrs=['weight'],
            directed=False
        )
        try:
            leiden_partition = leidenalg.find_partition(ig_sub, leidenalg.ModularityVertexPartition)
            leiden_comms = [list(map(str, ig_sub.vs[cluster]['name'])) for cluster in leiden_partition]
            hybrid_communities.extend(leiden_comms)
        except Exception as e:
            print(f"⚠️ Error in Leiden clustering for subgraph: {e}")
            hybrid_communities.append(comm_nodes)
    return hybrid_communities

def hybrid_louvain_leiden(G):
    G = validate_weights(G)
    partition = community_louvain.best_partition(G, weight='weight')
    louvain_comms = defaultdict(list)
    for node, comm_id in partition.items():
        louvain_comms[comm_id].append(node)
    hybrid_communities = []
    for comm_nodes in louvain_comms.values():
        if len(comm_nodes) <= 2:
            hybrid_communities.append(comm_nodes)
            continue
        subgraph = G.subgraph(comm_nodes).copy()
        if subgraph.number_of_edges() == 0:
            hybrid_communities.append(comm_nodes)
            continue
        # Convert edges to iGraph with explicit numeric weights
        ig_sub = ig.Graph.TupleList(
            [(str(u), str(v), {'weight': float(d.get('weight', 1.0))}) for u, v, d in subgraph.edges(data=True)],
            edge_attrs=['weight'],
            directed=False
        )
        try:
            leiden_partition = leidenalg.find_partition(ig_sub, leidenalg.ModularityVertexPartition)
            leiden_comms = [list(map(str, ig_sub.vs[cluster]['name'])) for cluster in leiden_partition]
            hybrid_communities.extend(leiden_comms)
        except Exception as e:
            print(f"⚠️ Error in Leiden clustering for subgraph: {e}")
            hybrid_communities.append(comm_nodes)
    return hybrid_communities