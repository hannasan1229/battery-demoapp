import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from demodata_cycle import generate_varM_dataframes
from cycle_analysis import process_batch

st.set_page_config(page_title="Battery Analysis Tool", layout="centered")

st.title("🔋 Battery Cycle Analysis Tool")

st.markdown("""
This demo showcases automated battery data analysis, including:
- Cycle detection  
- State-of-Health (SoH) evaluation  
- Capacity check extraction  
- Statistical aggregation across multiple cells  
""")

# ----------------------------------
# User Input
# ----------------------------------

st.header("⚙️ Variants Setup (VarM)")
st.caption("Define different material variants (VarM) for comparative testing.")

n_mat = st.number_input(
    "Number of variants",
    min_value=1,
    max_value=10,
    value=2
)

materials = {}

for i in range(n_mat):

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input(
            f"Variant {i+1} name",
            value=f"Material-{chr(65+i)}"
        )

    with col2:
        n_cells = st.number_input(
            f"Number of cells for {name}",
            min_value=1,
            max_value=10,
            value=2,
            key=f"cells_{i}"
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

if "raw_varM" not in st.session_state:
    st.session_state.raw_varM = None

# ----------------------------------
# Run Simulation
# ----------------------------------

if st.button("🚀 Run Analysis"):

    with st.spinner("Generating and analyzing data..."):

        varM = generate_varM_dataframes(materials)
        st.session_state.raw_varM = varM

        full_results, capcheck_results = process_batch(varM)

        st.session_state.full_results = full_results
        st.session_state.capcheck_results = capcheck_results

    st.success("Analysis complete!")

# ----------------------------------
# Plot Results
# ----------------------------------

if st.session_state.full_results is not None:

    st.header("📊 Aging & Performance Analysis")
    st.caption("Comparison of full degradation behavior and capacity check benchmarks.")

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    cmap = plt.get_cmap("Set1")

    # --------------------------------------------------
    # RAW DATA PLOTS
    # --------------------------------------------------
    if st.session_state.raw_varM is not None:

        first_mat = next(iter(st.session_state.raw_varM))
        raw_df = st.session_state.raw_varM[first_mat][0]

        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])

        ax1.plot(raw_df["timestamp"], raw_df["voltage_V"])
        ax1.set_title("Voltage Profile")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Voltage [V]")
        ax1.grid(True)

        ax2.plot(raw_df["timestamp"], raw_df["current_A"])
        ax2.set_title("Current Profile")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Current [A]")
        ax2.grid(True)

    # --------------------------------------------------
    # SoH PLOTS
    # --------------------------------------------------
    for i, mat in enumerate(st.session_state.full_results.keys()):

        full_df = st.session_state.full_results[mat]
        cap_df = st.session_state.capcheck_results[mat]

        color = cmap(i)

        if not full_df.empty:

            ax3.plot(
                full_df["cycle"],
                full_df["ave"],
                "--s",
                label=mat,
                color=color
            )

            ax3.errorbar(
                full_df["cycle"],
                full_df["ave"],
                full_df["std"],
                capsize=4,
                color=color
            )

        if not cap_df.empty:

            ax4.plot(
                cap_df["cycle"],
                cap_df["ave"],
                "--s",
                label=mat,
                color=color
            )

            ax4.errorbar(
                cap_df["cycle"],
                cap_df["ave"],
                cap_df["std"],
                capsize=4,
                color=color
            )

    ax3.set_title("Full Degradation (All Cycles)")
    ax3.set_xlabel("Cycle")
    ax3.set_ylabel("SoH [%]")
    ax3.set_ylim(70, 101)
    ax3.grid(True)
    ax3.legend()

    ax4.set_title("Capacity Check Summary")
    ax4.set_xlabel("Cycle")
    ax4.set_ylabel("SoH [%]")
    ax4.set_ylim(70, 101)
    ax4.grid(True)
    ax4.legend()

    fig.tight_layout()

    st.pyplot(fig)

# ----------------------------------
# Raw Data Preview
# ----------------------------------

with st.expander("🔍 Show processed results"):

    if st.session_state.full_results is not None:

        for mat in st.session_state.full_results.keys():

            st.write(f"### {mat} – Full Degradation")
            st.dataframe(st.session_state.full_results[mat].head())

            st.write(f"### {mat} – Capacity Checks")
            st.dataframe(st.session_state.capcheck_results[mat].head())
