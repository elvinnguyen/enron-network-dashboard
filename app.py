import matplotlib.pyplot as plt
import networkit as nk
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Enron Data Set Dashboard", layout="wide")


def giant_component_subgraph(G):
    """Return the subgraph of the largest connected component (undirected)."""
    if G.number_of_nodes() == 0:
        return G.copy()
    components = list(nx.connected_components(G))
    gcc_nodes = max(components, key=len)
    return G.subgraph(gcc_nodes).copy()


def graph_stats(G):
    """
    Compute basic stats:
    - n, m
    - avg degree
    - average clustering (nx.average_clustering)
    - average shortest path length on giant component (if >= 2 nodes)
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    avg_deg = (2 * m / n) if n > 0 else float("nan")
    C = nx.average_clustering(G) if n > 0 else float("nan")

    GCC = giant_component_subgraph(G)
    # if GCC.number_of_nodes() >= 2 and nx.is_connected(GCC):
    #    L = nx.average_shortest_path_length(GCC)
    # else:
    #    L = float('nan')

    # return {"n": n, "m": m, "avg_deg": avg_deg, "C": C, "L_gcc": L, "gcc_size": GCC.number_of_nodes()}
    return {
        "n": n,
        "m": m,
        "avg_deg": avg_deg,
        "C": C,
        "gcc_size": GCC.number_of_nodes(),
    }


def draw_graph(G, title="", layout="spring", node_size=40):
    """Draw graph with a chosen layout."""
    plt.figure(figsize=(6, 5))
    if layout == "spring":
        pos = nx.spring_layout(G, seed=SEED)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=SEED)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, alpha=0.85)
    nx.draw_networkx_edges(G, pos, width=0.6, alpha=0.35)
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_degree_hist(G, title="Degree distribution"):
    degrees = [d for _, d in G.degree()]
    plt.figure(figsize=(6, 4))
    plt.hist(degrees, bins=20)
    plt.title(title)
    plt.xlabel("Degree k")
    plt.ylabel("Count of nodes")
    st.pyplot(plt, use_container_width=True)


def plot_centrality_hist(G, title="Centrality distribution"):
    centrality = nx.degree_centrality(G)
    values = list(centrality.values())
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=20)
    plt.title(title)
    plt.xlabel("Centrality")
    plt.ylabel("Count of nodes")
    st.pyplot(plt, use_container_width=True)


st.title("Enron Data Set Dashboard")
st.write("Placeholder")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Centrality", "Community Detection", "Interpretation and Limitations"]
)

df = pd.read_csv(
    "email-Enron.txt", sep="\t", comment="#", names=["FromNodeId", "ToNodeId"]
)

G = nx.from_pandas_edgelist(
    df, source="FromNodeId", target="ToNodeId", create_using=nx.Graph()
)

G2 = nk.nxadapter.nx2nk(G)

with tab1:
    st.header("Overview")

    st.markdown(
        """
        ### Guiding Questions
        - What is the Enron data set?
        - What are the key features of the data?
        - What are some potential analyses we can perform with this data?
        """
    )

    st.markdown(
        """
        ### What is the Enron Data Set?
        The Enron data set is a collection of emails from the Enron Corporation, which was an American energy company that went bankrupt in 2001. 
        The data set contains around 500,000 emails from about 150 users, mostly senior management of Enron. The emails were made public during the 
        investigation of the company's collapse and have since been used for various research purposes, including natural language processing, social 
        network analysis, and machine learning.
        """
    )

    df["FromNodeId"].nunique()
    degree_distribution = df["FromNodeId"].value_counts()
    degree_distribution.rename("Degree", inplace=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Nodes", G.number_of_nodes())
    c2.metric("Edges", G.number_of_edges())
    c3.metric(
        "Density",
        round(
            2 * G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1)),
            6,
        ),
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Degree Distribution (Top 10)")
        st.dataframe(degree_distribution.head(10))
    with right:
        # Degree distribution plot
        plot_degree_hist(G)

    st.subheader("Enron Email Data Sample")
    st.write(df.head(100))

with tab2:
    st.header("Centrality")

    st.markdown(
        """
        ### Guiding Questions
        - What is centrality in the context of social networks?
        - How can we measure centrality in the Enron email network?
        - Who are the most central nodes in the Enron email network?
        """
    )

    st.markdown(
        """
        ### What is Centrality?
        Centrality is a measure of the importance or influence of a node within a network. 
        In the context of social networks, centrality can help us identify key individuals who may 
        have significant influence over others. There are several types of centrality measures, including 
        degree centrality, betweenness centrality, closeness centrality, and eigenvector centrality.
        """
    )

    left, right = st.columns(2)

    deg_centrality = nk.centrality.DegreeCentrality(G2)
    deg_centrality.run()
    deg_ranking = deg_centrality.ranking()[:10]

    betweeness_centrality = nk.centrality.EstimateBetweenness(G2, 1000)
    betweeness_centrality.run()
    betweeness_ranking = betweeness_centrality.ranking()[:10]

    with left:
        st.subheader("Top 10 Nodes by Degree Centrality")
        st.dataframe(
            pd.DataFrame(
                deg_ranking, columns=["Node", "Degree Centrality"]
            ).style.format({"Degree Centrality": "{:.0f}"})
        )
    with right:
        st.subheader("Top 10 Nodes by Betweenness Centrality")
        st.dataframe(
            pd.DataFrame(
                betweeness_ranking, columns=["Node", "Betweenness Centrality"]
            ).style.format({"Betweenness Centrality": "{:.0f}"})
        )

with tab3:
    st.header("Community Detection")
    st.markdown(
        """
        ### Guiding Questions
        - What is community detection in the context of social networks?
        - How can we detect communities in the Enron email network?
        - What are the characteristics of the detected communities?
        """
    )

    st.markdown(
        """
        ### What is Community Detection?
        Community detection is the process of identifying groups of nodes in a network that are more densely connected to each other than to the rest of the network. 
        In social networks, communities can represent groups of individuals who interact more frequently with each other than with those outside the group. 
        There are various algorithms for community detection, such as modularity-based methods, spectral clustering, and label propagation.
        """
    )

with tab4:
    st.header("Interpretation and Limitations")
    st.markdown(
        """
        ### Guiding Questions
        - What are the limitations of our analysis of the Enron email network?
        - How can we interpret the results of our centrality and community detection analyses?
        - What are some potential biases or confounding factors in the Enron data set?
        """
    )

    st.markdown(
        """
        ### Interpretation and Limitations
        While analyzing the Enron email network can provide insights into the structure and dynamics of communication within the company, there are several limitations to consider. 
        The data set may not be representative of all employees, as it primarily contains emails from senior management. Additionally, the data may be incomplete or contain errors, 
        which could affect the accuracy of our analyses. Furthermore, centrality measures and community detection algorithms have their own assumptions and limitations, which should be taken into account when interpreting results.
        """
    )
