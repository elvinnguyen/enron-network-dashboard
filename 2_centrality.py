import streamlit as st
import pandas as pd
import networkx as nx

st.set_page_config(page_title="Enron Data Set Dashboard", layout="wide")

df = pd.read_csv(
    "email-Enron.txt", sep="\t", comment="#", names=["FromNodeId", "ToNodeId"]
)

tab1, tab2 = st.tabs(["Overview", "Centrality"])
