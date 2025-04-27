import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

def plot_community_graph(G, communities, title="Community Graph"):
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(14, 10))
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=community,
            node_color=f"C{i % 10}",
            label=f'Community {i + 1}',
            node_size=300,
            alpha=0.8,
            ax=ax
        )
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', ax=ax)

    ax.set_title(title)
    ax.set_axis_off()
    ax.legend()
    st.pyplot(fig)
