import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from demodata_cycle import generate_varM_dataframes
from cycle_analysis import process_batch, compute_dqdv_split

st.set_page_config(page_title="Battery Analysis Tool", layout="centered")

st.title("🔋 Battery Cycle Analysis Tool")

st.markdown(
    """
This demo showcases automated battery data analysis, including:
- Cycle detection  
- State-of-Health (SoH) evaluation  
- Capacity check extraction  
- Statistical aggregation across multiple cells  
"""
)

# ----------------------------------
# User Input
# ----------------------------------

st.header("⚙️ Variants Setup (VarM)")
st.caption("Define different material variants (VarM) for comparative testing.")

n_mat = st.number_input("Number of variants", min_value=1, max_value=10, value=2)

# 🔥 HIER hinzufügen (global für alle Varianten!)
colA, colB = st.columns(2)

with colA:
    n_cycle_blocks = st.number_input(
        "Number of cycle blocks", min_value=1, max_value=20, value=3
    )

with colB:
    n_cycles = st.number_input("Cycles per block", min_value=1, max_value=100, value=10)

# optional nice UX
st.caption(f"Total cycles ≈ {n_cycle_blocks * n_cycles}")

# ----------------------------------

materials = {}

for i in range(n_mat):

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input(f"Variant {i+1} name", value=f"Material-{chr(65+i)}")

    with col2:
        n_cells = st.number_input(
            f"Number of cells for {name}",
            min_value=1,
            max_value=10,
            value=2,
            key=f"cells_{i}",
        )

    materials[name] = {"n_cells": n_cells, "direction": None}

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
    
    n_var = len(st.session_state.raw_varM)
    rows_needed = 3 + n_var

    fig = plt.figure(figsize=(14, 4 * rows_needed))
    gs = fig.add_gridspec(rows_needed, 2)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    dqdv_axes = []

    for i in range(n_var):
        ax = fig.add_subplot(gs[3 + i // 2, i % 2])
        dqdv_axes.append(ax)

    cmap = plt.get_cmap("Set1")

    # --------------------------------------------------
    # RAW DATA PLOTS
    # --------------------------------------------------
    if st.session_state.raw_varM is not None:

        for i, mat in enumerate(st.session_state.raw_varM.keys()):
            
            df = st.session_state.raw_varM[mat][0].copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["t0"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

            color = cmap(i)
            
            ax1.plot(df["t0"], df["voltage_V"], label=mat, color=color)
            ax2.plot(df["t0"], df["current_A"], label=mat, color=color)
        ax1.set_title("Voltage Profile")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Voltage [V]")
        ax1.grid(True)
        ax1.legend()

        ax2.set_title("Current Profile")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Current [A]")
        ax2.grid(True)
        ax2.legend()

    # --------------------------------------------------
    # SoH PLOTS
    # --------------------------------------------------
    for i, mat in enumerate(st.session_state.full_results.keys()):

        full_df = st.session_state.full_results[mat]
        cap_df = st.session_state.capcheck_results[mat]

        color = cmap(i)

        if not full_df.empty:

            ax3.plot(full_df["cycle"], full_df["ave"], "--s", label=mat, color=color)

            ax3.errorbar(
                full_df["cycle"], full_df["ave"], full_df["std"], capsize=4, color=color
            )

        if not cap_df.empty:

            ax4.plot(cap_df["cycle"], cap_df["ave"], "--s", label=mat, color=color)

            ax4.errorbar(
                cap_df["cycle"], cap_df["ave"], cap_df["std"], capsize=4, color=color
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

# --------------------------------------------------
# dQ/dV PLOTS (Charge / Discharge getrennt)
# --------------------------------------------------

for i, mat in enumerate(st.session_state.raw_varM.keys()):

    df = st.session_state.raw_varM[mat][0]

    ch_curves, dch_curves = compute_dqdv_split(df)

    ax_ch = fig.add_subplot(gs[3 + i, 0])
    ax_dch = fig.add_subplot(gs[3 + i, 1])

    cmap_ch = plt.cm.winter
    cmap_dch = plt.cm.summer

    # Charge
    for j, (V, dqdv) in enumerate(ch_curves):
        ax_ch.plot(V, dqdv, color=cmap_ch(j / max(len(ch_curves)-1, 1)), alpha=0.7)

    # Discharge
    for j, (V, dqdv) in enumerate(dch_curves):
        ax_dch.plot(V, dqdv, color=cmap_dch(j / max(len(dch_curves)-1, 1)), alpha=0.7)

    ax_ch.set_title(f"{mat} – Charge dQ/dV")
    ax_dch.set_title(f"{mat} – Discharge dQ/dV")

    for ax in [ax_ch, ax_dch]:
        ax.set_xlim(2.8, 4.2)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel("Voltage [V]")
        ax.set_ylabel("dQ/dV")
        ax.grid(True)

    # --------------------------------------------------

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
