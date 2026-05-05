import matplotlib.pyplot as plt
import networkit as nk
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Enron Data Set Dashboard", layout="wide")


@st.cache_data
def giant_component_subgraph(_G):
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
    return {
        "n": n,
        "m": m,
        "avg_deg": avg_deg,
        "C": C,
        "gcc_size": GCC.number_of_nodes(),
    }


def plot_degree_hist(G):
    degrees = [d for _, d in G.degree()]

    fig = go.Figure(
        data=[go.Histogram(x=degrees, nbinsx=30)],
        layout=go.Layout(
            title="Degree Distribution",
            xaxis_title="Degree",
            yaxis=dict(
                title="Count of nodes",
                type="log",
                tickvals=[1, 10, 100, 1000, 10000],
            ),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_deg_centrality_hist(G):
    centrality = nx.degree_centrality(G)
    values = list(centrality.values())

    fig = go.Figure(
        data=[go.Histogram(x=values, nbinsx=30)],
        layout=go.Layout(
            title="Degree Centrality distribution",
            xaxis=dict(
                title="Degree Centrality",
                tickformat=".3f",
            ),
            yaxis=dict(
                title="Count of nodes",
                type="log",
                tickvals=[1, 10, 100, 1000, 10000],
            ),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def compute_deg_centrality(G):
    deg_centrality = nk.centrality.DegreeCentrality(G)
    deg_centrality.run()
    return deg_centrality.ranking()


def compute_betweenness_centrality(G, samples=1000, seed=42):
    nk.setSeed(seed, False)

    betweenness_centrality = nk.centrality.EstimateBetweenness(G, samples)
    betweenness_centrality.run()
    return betweenness_centrality


def compute_community_detection(G):
    nk.setSeed(42, False)
    communities = nk.community.detectCommunities(G, algo=nk.community.PLM(G, True))

    return communities


def plot_communities(communities):
    sizes = communities.subsetSizes()
    fig = go.Figure(
        data=[go.Histogram(x=sizes, xbins=dict(start=0, end=max(sizes), size=100))],
        layout=go.Layout(
            title="Community Size Distribution",
            xaxis_title="Community Size",
            yaxis=dict(
                title="Count of communities",
                type="log",
                tickvals=[1, 10, 100, 1000],
                ticktext=["1", "10", "100", "1000"],
            ),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_betweenness_centrality_hist(betweenness):
    values = betweenness.scores()
    n = G.number_of_nodes()
    values = [v / (n * (n - 1) / 2) for v in values]

    fig = go.Figure(
        data=[go.Histogram(x=values, xbins=dict(start=0, end=max(values), size=0.002))],
        layout=go.Layout(
            title="Betweenness Centrality Distribution",
            xaxis_title="Betweenness Centrality",
            yaxis=dict(
                title="Count of nodes",
                type="log",
                tickvals=[1, 10, 100, 1000, 10000],
            ),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


st.title("Enron Data Set Dashboard")
st.write("Analysis and Understanding")

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

    st.markdown("""
        ### Overarching Guiding Questions
        - What is the enron data set?
        - Is the network highly centralized or fragmented?
        - Who are the most influential communicators in the network?
        - Are there meaningful communities in the network or clusters in the network?
        """)

    st.markdown("""
        ### What is the Enron Data Set?
        The Enron data set is a collection of emails from the Enron Corporation, which was an American energy company that went bankrupt in 2001. 
        The data set contains around 500,000 emails from about 150 users, mostly senior management of Enron. The emails were made public during the 
        investigation of the company's collapse and have since been used for various research purposes, including natural language processing, social 
        network analysis, and machine learning.
        
        **Note**: The original data set lists the total edges as 367,662. 
                After removing duplicate edges, we have 183,831 unique edges in our graph. 
        """)

    st.markdown("""
        ### Related Overarching Guided Question
        - Is the network highly centralized or fragmented?
        """)

    df["FromNodeId"].nunique()
    degree_distribution = df["FromNodeId"].value_counts()
    degree_distribution.rename("Degree", inplace=True)

    gcc = giant_component_subgraph(G)
    gcc_ratio = gcc.number_of_nodes() / G.number_of_nodes()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", G.number_of_nodes())
    c2.metric("Edges", G.number_of_edges())
    c3.metric(
        "Density",
        round(
            2 * G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1)),
            6,
        ),
    )
    c4.metric("GCC Ratio", round(gcc_ratio, 4))

    left, right = st.columns(2)
    with left:
        st.subheader("Degree Distribution (Top 10)")
        st.dataframe(degree_distribution.head(10))
        st.markdown("""
            This table highlights the top 10 nodes with the highest degree, 
            which represent the most active communicators in the Enron email network.
            """)
    with right:
        # Degree distribution plot
        plot_degree_hist(G)
        st.markdown("""
            The distribution skewed right, meaning that most nodes have few connections while
            a small number of nodes have high connections. This suggests that the network contains
            few highly connected hubs.
            """)

    st.subheader("Enron Email Data Sample")
    st.write(df.head(100))

    st.markdown("""
        ### Interpretation Related to Overarching Question
        The networking is most likely centralized.
        The gcc ratio is .9183, meaning that most nodes are in a connected component.
        Connections are also sparse with .00273 density, also the degree distribution would mean a few nodes the brokers.
        This means that within the component a few nodes act as the central hubs.
        The data set also mentioned that there were some emails which belonged to journalists.
        This would make perfect sense, the component represents the company, and the outside nodes are the journalists.
        """)

with tab2:
    st.header("Centrality")
    
    st.markdown("""
        ### Related Overarching Guiding Question
        - Who are the most influential communicators in the network?
        """)

    left, right = st.columns(2)

    deg_ranking = compute_deg_centrality(G2)

    betweenness_centrality = compute_betweenness_centrality(G2, samples=1000, seed=42)
    n = G2.numberOfNodes()
    betweenness_ranking = [
        (node, score / ((n - 1) * (n - 2)))
        for node, score in betweenness_centrality.ranking()
    ][:10]

    with left:
        st.subheader("Top 10 Nodes by Degree Centrality")
        st.dataframe(
            pd.DataFrame(
                deg_ranking, columns=["Node", "Degree Centrality"]
            ).style.format({"Degree Centrality": "{:.0f}"})
        )
    with right:
        plot_deg_centrality_hist(G)

    with left:
        st.subheader("Top 10 Nodes by Betweenness Centrality")
        st.dataframe(
            pd.DataFrame(
                betweenness_ranking, columns=["Node", "Betweenness Centrality"]
            ).style.format({"Betweenness Centrality": "{:.6f}"})
        )

    with right:
        plot_betweenness_centrality_hist(betweenness_centrality)
    st.markdown("""
        ### Interpretation Related to Overarching Question
        Nodes 197, 271, 80, 144, 148, 85, 92, 191 are the most central communicators.
        This is because these appear in the top 10 for both betweenness and degree centrality.
        So they are likely to be structurally important, they have the most connections(degree), and they are structural important(betweenness) because they link different parts together.
        """)


with tab3:
    st.header("Community Detection")
    st.markdown("""
        ### Overarching Guiding Question 
        - Are there meaningful communities in the network or clusters in the network?
        """)

    communities = compute_community_detection(G2)

    st.subheader("Community Detection Results")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Number of Communities Detected", communities.numberOfSubsets())
    with c2:
        avg_size = G2.numberOfNodes() / communities.numberOfSubsets()
        st.metric("Average community size:", f"{avg_size:.2f}")

    left, right = st.columns(2)
    with left:
        st.write("Community Sizes:")
        st.dataframe(
            pd.DataFrame(
                sorted(communities.subsetSizes(), reverse=True),
                columns=["Community Size"],
            )
        )
    with right:
        plot_communities(communities)
    st.markdown("""
            ### Interpretation Related to Overarching Question
            The network contains a large number of communities, with a highly skewed distribution size, having 
            few large communities and many small communities as shown by the distribution plot. This is common 
            in social networks, where most individuals belong to small groups while a few belong to larger, more 
            influential communities. The presence of larger communities may indicate that there are certain groups of
            individuals who communicate more frequently with each other than with the rest of the network. This suggests 
            that communication within the Enron network is highly fragmented, with many small groups and a limited number 
            of highly connected hubs.
            These could relate to different departments within the Enron corporation.
        """)


with tab4:
    st.header("Interpretation and Limitations")
    st.markdown("""
        ### Guiding Questions
        - What are the limitations of our analysis of the Enron email network?
        - How can we interpret the results of our centrality and community detection analyses?
        """)

    st.markdown("""
        ### Interpretation and Limitations
        While analyzing the Enron email network can provide insights into the structure and dynamics of communication within the company, there are several limitations to consider. 
        The data set may not be representative of all employees, as it primarily contains emails from senior management. Additionally, the data may be incomplete or contain errors, which could affect the accuracy of our analyses.
        Furthermore, centrality measures and community detection algorithms have their own assumptions and limitations, which should be taken into account when interpreting results.
        Another key limitation is that the network does not have any metadata.
        There is not any information about the nodes that is easily available.
        However, this information does exist online, but it is fairly large, and not feasible to analyze within the time allotted.
        An example of this would be the potential for certain accounts outside the giant connected component to belong to journalists. 
        """)
