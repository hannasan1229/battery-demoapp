import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from demodata_cycle import generate_DoE_dataframes
from cycle_analysis import process_batch

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

if "full_results" not in st.session_state:
    st.session_state.full_results = None

if "capcheck_results" not in st.session_state:
    st.session_state.capcheck_results = None

# ----------------------------------
# Run DoE
# ----------------------------------

if st.button("🚀 Run DoE Simulation"):

    with st.spinner("Generating data..."):
        DoE = generate_DoE_dataframes(materials)
        full_results, capcheck_results = process_batch(DoE)
        
        st.session_state.full_results = full_results
        st.session_state.capcheck_results = capcheck_results

    st.success("Simulation complete!")

# ----------------------------------
# Plot Results
# ----------------------------------

if st.session_state.full_results is not None:

    st.header("📊 Battery Aging Analysis")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    cmap = plt.get_cmap("Set1")

    for i, mat in enumerate(st.session_state.full_results.keys()):

        full_df = st.session_state.full_results[mat]
        cap_df = st.session_state.capcheck_results[mat]

        color = cmap(i)

        if not full_df.empty:

            ax[0].plot(
                full_df["cycle"],
                full_df["ave"],
                "--s",
                label=mat,
                color=color
            )

            ax[0].errorbar(
                full_df["cycle"],
                full_df["ave"],
                full_df["std"],
                capsize=4,
                color=color
            )

        if not cap_df.empty:

            ax[1].plot(
                cap_df["cycle"],
                cap_df["ave"],
                "--s",
                label=mat,
                color=color
            )

            ax[1].errorbar(
                cap_df["cycle"],
                cap_df["ave"],
                cap_df["std"],
                capsize=4,
                color=color
            )

    ax[0].set_title("Full SoH vs Cycle")
    ax[0].set_xlabel("Cycle")
    ax[0].set_ylabel("SoH [%]")
    ax[0].set_ylim(70, 101)
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_title("Capacity Check Summary")
    ax[1].set_xlabel("Capacity Check Cycle")
    ax[1].set_ylabel("SoH [%]")
    ax[1].set_ylim(70, 101)
    ax[1].grid(True)
    ax[1].legend()

    st.pyplot(fig)


# ----------------------------------
# Raw Data Preview (optional)
# ----------------------------------

with st.expander("Show raw processed data"):

    if st.session_state.full_results is not None:

        for mat in st.session_state.full_results.keys():

            st.write(f"### {mat} – Full Aging")
            st.dataframe(
                st.session_state.full_results[mat].head()
            )

            st.write(f"### {mat} – Capacity Checks")
            st.dataframe(
                st.session_state.capcheck_results[mat].head()
            )