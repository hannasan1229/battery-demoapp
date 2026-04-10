import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from demodata_cycle import generate_DoE_dataframes
from cycle_analysis import collect_data, process_batch

st.set_page_config(page_title="Battery DoE Tool", layout="centered")

st.title("🔋 Battery Cycle DoE Analysis")

# ----------------------------------
# User Input
# ----------------------------------

st.header("⚙️ DoE Setup")

n_mat = st.number_input("Number of materials", min_value=1, max_value=10, value=2)

materials = {}

for i in range(n_mat):

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input(f"Material {i+1} name", value=chr(65+i))

    with col2:
        n_cells = st.number_input(
            f"Cells for {name}", min_value=1, max_value=10, value=2, key=f"cells_{i}"
        )

    materials[name] = {
        "n_cells": n_cells,
        "direction": None
    }

# ----------------------------------
# Session State
# ----------------------------------

if "results" not in st.session_state:
    st.session_state.results = None

# ----------------------------------
# Run DoE
# ----------------------------------

if st.button("🚀 Run DoE Simulation"):

    with st.spinner("Generating data..."):

        DoE = generate_DoE_dataframes(materials)

        min_sums = collect_data(DoE)

        results = process_batch(min_sums)

        st.session_state.results = results

    st.success("Simulation complete!")

# ----------------------------------
# Plot Results
# ----------------------------------

if st.session_state.results is not None:

    st.header("📊 SoH vs Cycle")

    fig, ax = plt.subplots(figsize=(6, 5))

    cmap = plt.get_cmap("Set1")

    for i, (mat, df) in enumerate(st.session_state.results.items()):

        x = df.index
        y = df["ave"]
        e = df["std"]

        ax.plot(x, y, "--s", label=mat, color=cmap(i))
        ax.errorbar(x, y, e, capsize=4, color=cmap(i))

    ax.set_xlabel("Cycle")
    ax.set_ylabel("SoH [%]")
    ax.set_ylim(70, 101)
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)



# ----------------------------------
# Raw Data Preview (optional)
# ----------------------------------

with st.expander("Show raw processed data"):

    if st.session_state.results is not None:
        for mat, df in st.session_state.results.items():
            st.write(f"Material {mat}")
            st.dataframe(df.head())
