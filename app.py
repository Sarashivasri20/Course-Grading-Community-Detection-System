import streamlit as st
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg
import community.community_louvain as community_louvain
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import matplotlib.pyplot as plt

from hybrid_algorithms import (
    hybrid_greedy_louvain,
    hybrid_greedy_walktrap,
    hybrid_greedy_leiden,
    hybrid_louvain_walktrap,
    hybrid_walktrap_leiden,
    hybrid_louvain_leiden,
)
from utils import plot_community_graph

# --------------------- Streamlit Layout ---------------------
st.set_page_config(layout="wide")
st.title("Course Grading Community Detection System")
st.subheader("University of Illinois Spring 2010 - Spring 2024")

# --------------------- Data Loading ---------------------
@st.cache_data
def load_data():
    url = "https://waf.cs.illinois.edu/discovery/gpa.csv"
    df = pd.read_csv(url)
    grade_columns = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F']
    df[grade_columns] = df[grade_columns].fillna(0)
    df['CourseID'] = df['Subject'].astype(str) + '-' + df['Number'].astype(str)
    return df, grade_columns

df, grade_columns = load_data()

time_slice = st.selectbox("Select Time Slice:", df['YearTerm'].unique())

# --------------------- Graph Building ---------------------
df_t = df[df['YearTerm'] == time_slice]
course_vectors = df_t.groupby('CourseID')[grade_columns].sum()
similarity_matrix = cosine_similarity(course_vectors)
course_ids = course_vectors.index.tolist()

G = nx.Graph()
for i, course1 in enumerate(course_ids):
    for j, course2 in enumerate(course_ids):
        if i < j:
            weight = similarity_matrix[i, j]
            if weight > 0.7:
                G.add_edge(course1, course2, weight=weight)

# --------------------- Algorithm Selection ---------------------
algo_option = st.selectbox("Choose Hybridization or View Original Methods:", 
    [ "Greedy → Louvain", "Greedy → Walktrap","Greedy → Leiden", "Louvain → Walktrap","Walktrap → Leiden", "Louvain → Leiden", "Overall Algorithms Comparison"])

# Initialize a dictionary to store modularity results
modularity_results = {}

# --------------------- Results Dictionary for Metrics ---------------------
results = {
    "Greedy → Louvain": {"modularity": [], "community_sizes": [], "edge_density": [], "overlap": [], "persistence": []},
    "Greedy → Walktrap": {"modularity": [], "community_sizes": [], "edge_density": [], "overlap": [], "persistence": []},
    "Greedy → Leiden": {"modularity": [], "community_sizes": [], "edge_density": [], "overlap": [], "persistence": []},
    "Louvain → Walktrap": {"modularity": [], "community_sizes": [], "edge_density": [], "overlap": [], "persistence": []},
    "Walktrap → Leiden": {"modularity": [], "community_sizes": [], "edge_density": [], "overlap": [], "persistence": []},
    "Louvain → Leiden": {"modularity": [], "community_sizes": [], "edge_density": [], "overlap": [], "persistence": []},
}

# --------------------- Metric Calculations ---------------------
def compute_edge_density(G, communities):
    # Ensure that all nodes in the community exist in the graph
    valid_communities = []
    for community in communities:
        # Only keep nodes that exist in the graph
        valid_community = [node for node in community if node in G]
        if len(valid_community) > 1:  # A valid community must have at least two nodes
            valid_communities.append(valid_community)

    # Compute edge density for valid communities
    edge_densities = []
    for community in valid_communities:
        subgraph = G.subgraph(community)
        num_edges = subgraph.number_of_edges()
        num_nodes = len(subgraph.nodes())
        if num_nodes > 1:
            # Edge density formula: (2 * num_edges) / (num_nodes * (num_nodes - 1))
            edge_density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
            edge_densities.append(edge_density)
    
    # If no valid communities, return 0 or appropriate value
    if edge_densities:
        return sum(edge_densities) / len(edge_densities)  # Average edge density
    else:
        return 0  # Return 0 if no valid communities are found



def compute_overlap(G, community1, community2):
    return len(set(community1).intersection(set(community2))) / min(len(community1), len(community2))



def track_metrics_for_time_slice(time_slice, algo_option, modularity, community_sizes, edge_density, overlap, persistence):
    if algo_option == "Greedy → Louvain":
        results["Greedy → Louvain"]["modularity"].append(modularity)
        results["Greedy → Louvain"]["community_sizes"].append(community_sizes)
        results["Greedy → Louvain"]["edge_density"].append(edge_density)
        results["Greedy → Louvain"]["overlap"].append(overlap)
        results["Greedy → Louvain"]["persistence"].append(persistence)
    elif algo_option == "Greedy → Walktrap":
        results["Greedy → Walktrap"]["modularity"].append(modularity)
        results["Greedy → Walktrap"]["community_sizes"].append(community_sizes)
        results["Greedy → Walktrap"]["edge_density"].append(edge_density)
        results["Greedy → Walktrap"]["overlap"].append(overlap)
        results["Greedy → Walktrap"]["persistence"].append(persistence)
    elif algo_option == "Greedy → Leiden":
        results["Greedy → Leiden"]["modularity"].append(modularity)
        results["Greedy → Leiden"]["community_sizes"].append(community_sizes)
        results["Greedy → Leiden"]["edge_density"].append(edge_density)
        results["Greedy → Leiden"]["overlap"].append(overlap)
        results["Greedy → Leiden"]["persistence"].append(persistence)
    elif algo_option == "Louvain → Walktrap":
        results["Louvain → Walktrap"]["modularity"].append(modularity)
        results["Louvain → Walktrap"]["community_sizes"].append(community_sizes)
        results["Louvain → Walktrap"]["edge_density"].append(edge_density)
        results["Louvain → Walktrap"]["overlap"].append(overlap)
        results["Louvain → Walktrap"]["persistence"].append(persistence)
    elif algo_option == "Walktrap → Leiden":
        results["Walktrap → Leiden"]["modularity"].append(modularity)
        results["Walktrap → Leiden"]["community_sizes"].append(community_sizes)
        results["Walktrap → Leiden"]["edge_density"].append(edge_density)
        results["Walktrap → Leiden"]["overlap"].append(overlap)
        results["Walktrap → Leiden"]["persistence"].append(persistence)
    elif algo_option == "Louvain → Leiden":
        results["Louvain → Leiden"]["modularity"].append(modularity)
        results["Louvain → Leiden"]["community_sizes"].append(community_sizes)
        results["Louvain → Leiden"]["edge_density"].append(edge_density)
        results["Louvain → Leiden"]["overlap"].append(overlap)
        results["Louvain → Leiden"]["persistence"].append(persistence)

# --------------------- Plot Metric Function ---------------------
def plot_metric(metric_name, ylabel, title):
    plt.figure(figsize=(12, 6))
    for algo in results:
        plt.plot(time_slices, results[algo][metric_name], label=algo)
    plt.xlabel("Time Slice")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    st.pyplot(plt)

# --------------------- Running the Algorithms ---------------------
if st.button("Click to Run"):
    if algo_option == "Greedy → Louvain":
        hybrid_comms = hybrid_greedy_louvain(G)
        modularity = nx.algorithms.community.quality.modularity(G, hybrid_comms, weight='weight')
        community_sizes = [len(comm) for comm in hybrid_comms]
        edge_density = compute_edge_density(G, hybrid_comms)
        overlap = 0  # Define your own overlap calculation
        persistence = 0.9  # Example persistence, adjust as needed
        track_metrics_for_time_slice(time_slice, algo_option, modularity, community_sizes, edge_density, overlap, persistence)
        st.success(f"Hybridization completed: Greedy → Louvain (Modularity Q = {modularity:.4f})")
        plot_community_graph(G, hybrid_comms, title=f"Hybrid (Greedy → Louvain) - {time_slice}")
    
    elif algo_option == "Greedy → Walktrap":
        hybrid_comms = hybrid_greedy_walktrap(G)
# Check if it returns a list of sets
        if isinstance(hybrid_comms, list) and not isinstance(hybrid_comms[0], set):
            hybrid_comms = [set(comm) for comm in hybrid_comms]
        modularity = nx.algorithms.community.quality.modularity(G, hybrid_comms, weight='weight')
        community_sizes = [len(comm) for comm in hybrid_comms]
        edge_density = compute_edge_density(G, hybrid_comms)
        overlap = 0  # Define your own overlap calculation
        persistence = 0.9  # Example persistence, adjust as needed
        track_metrics_for_time_slice(time_slice, algo_option, modularity, community_sizes, edge_density, overlap, persistence)
        st.success(f"Hybridization completed: Greedy → walktrap (Modularity Q = {modularity:.4f})")
        plot_community_graph(G, hybrid_comms, title=f"Hybrid (Greedy → walktrap) - {time_slice}")

    elif algo_option == "Greedy → Leiden":
        hybrid_comms = hybrid_greedy_leiden(G)
        modularity = nx.algorithms.community.quality.modularity(G, hybrid_comms, weight='weight')
        community_sizes = [len(comm) for comm in hybrid_comms]
        edge_density = compute_edge_density(G, hybrid_comms)
        overlap = 0  # Define your own overlap calculation
        persistence = 0.9  # Example persistence, adjust as needed
        track_metrics_for_time_slice(time_slice, algo_option, modularity, community_sizes, edge_density, overlap, persistence)
        st.success(f"Hybridization completed: Greedy → leiden (Modularity Q = {modularity:.4f})")
        plot_community_graph(G, hybrid_comms, title=f"Hybrid (Greedy → leiden) - {time_slice}")

    elif algo_option == "Louvain → Walktrap":
        hybrid_comms = hybrid_louvain_walktrap(G)
        modularity = nx.algorithms.community.quality.modularity(G, hybrid_comms, weight='weight')
        community_sizes = [len(comm) for comm in hybrid_comms]
        edge_density = compute_edge_density(G, hybrid_comms)
        overlap = 0  # Define your own overlap calculation
        persistence = 0.9  # Example persistence, adjust as needed
        track_metrics_for_time_slice(time_slice, algo_option, modularity, community_sizes, edge_density, overlap, persistence)
        st.success(f"Hybridization completed: Louvain → walktrap (Modularity Q = {modularity:.4f})")
        plot_community_graph(G, hybrid_comms, title=f"Hybrid (Louvain → walktrap) - {time_slice}")

    elif algo_option == "Walktrap → Leiden":
        hybrid_comms = hybrid_walktrap_leiden(G)
        modularity = nx.algorithms.community.quality.modularity(G, hybrid_comms, weight='weight')
        community_sizes = [len(comm) for comm in hybrid_comms]
        edge_density = compute_edge_density(G, hybrid_comms)
        overlap = 0  # Define your own overlap calculation
        persistence = 0.9  # Example persistence, adjust as needed
        track_metrics_for_time_slice(time_slice, algo_option, modularity, community_sizes, edge_density, overlap, persistence)
        st.success(f"Hybridization completed: walktrap->Leiden (Modularity Q = {modularity:.4f})")
        plot_community_graph(G, hybrid_comms, title=f"Hybrid (walktrap->Leiden) - {time_slice}")

    elif algo_option == "Louvain → Leiden":
        hybrid_comms = hybrid_louvain_leiden(G)
        modularity = nx.algorithms.community.quality.modularity(G, hybrid_comms, weight='weight')
        community_sizes = [len(comm) for comm in hybrid_comms]
        edge_density = compute_edge_density(G, hybrid_comms)
        overlap = 0  # Define your own overlap calculation
        persistence = 0.9  # Example persistence, adjust as needed
        track_metrics_for_time_slice(time_slice, algo_option, modularity, community_sizes, edge_density, overlap, persistence)
        st.success(f"Hybridization completed: Louvain -> Leiden (Modularity Q = {modularity:.4f})")
        plot_community_graph(G, hybrid_comms, title=f"Hybrid (Louvain-Leiden) - {time_slice}")
    
    elif algo_option == "Overall Algorithms Comparison":
        st.success("Visualizing All Four Original Algorithms")

        edges = [(u, v, float(d['weight'])) for u, v, d in G.edges(data=True)]
        ig_graph = ig.Graph.TupleList(edges, edge_attrs=['weight'], directed=False)

        # 1. Greedy
        greedy = list(nx.algorithms.community.greedy_modularity_communities(G, weight='weight'))
        q_greedy = nx.algorithms.community.quality.modularity(G, greedy, weight='weight')

        # 2. Louvain
        partition_louvain = community_louvain.best_partition(G, weight='weight')
        louvain_comms = defaultdict(list)
        for node, comm_id in partition_louvain.items():
            louvain_comms[comm_id].append(node)
        q_louvain = community_louvain.modularity(partition_louvain, G, weight='weight')

        # 3. Leiden
        leiden_partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
        leiden_comms = [ig_graph.vs[cluster]['name'] for cluster in leiden_partition]
        q_leiden = leiden_partition.modularity

        # 4. Walktrap
        walktrap = ig_graph.community_walktrap(weights='weight').as_clustering()
        walktrap_comms = [ig_graph.vs[cluster]['name'] for cluster in walktrap]
        q_walktrap = walktrap.modularity

        # --------------------- Plot Side-by-Side ---------------------
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### Greedy Modularity (Q = {q_greedy:.4f})")
            plot_community_graph(G, [list(c) for c in greedy], title='Greedy Modularity')

        with col2:
            st.markdown(f"### Louvain (Q = {q_louvain:.4f})")
            plot_community_graph(G, list(louvain_comms.values()), title='Louvain')

        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"### Leiden (Q = {q_leiden:.4f})")
            plot_community_graph(G, leiden_comms, title='Leiden')

        with col4:
            st.markdown(f"### Walktrap (Q = {q_walktrap:.4f})")
            plot_community_graph(G, walktrap_comms, title='Walktrap')