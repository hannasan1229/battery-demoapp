import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from demodata_cycle import generate_varM_dataframes
from cycle_analysis import process_batch, extract_dqdv_cycles

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

def plot_dqdv(ax, dqdv_data, cmap_name="viridis"):

    if len(dqdv_data) == 0:
        return

    cycles = [d["cycle"] for d in dqdv_data]

    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=min(cycles), vmax=max(cycles))

    for d in dqdv_data:

        color = cmap(norm(d["cycle"]))

        ax.plot(d["V"], d["dqdv"], color=color, alpha=0.9)

    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("dQ/dV")
    ax.grid(True)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Cycle")
# ----------------------------------
# Plot Results
# ----------------------------------

if (
    st.session_state.full_results is not None
    and st.session_state.capcheck_results is not None
    and st.session_state.raw_varM is not None
):

    st.header("📊 Aging & Performance Analysis")

    n_var = len(st.session_state.raw_varM)
    rows_needed = 3 + n_var

    fig = plt.figure(figsize=(14, 3.5 * rows_needed))
    gs = fig.add_gridspec(
    rows_needed, 
    2, 
    width_ratios=[1, 1],   # gleich breit
    wspace=0.25            # Abstand zwischen Spalten
)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    # dQdV axes (Charge | Discharge pro Material)
    dqdv_axes = []
    for i in range(n_var):
        ax_c = fig.add_subplot(gs[3 + i, 0])
        ax_d = fig.add_subplot(gs[3 + i, 1])
        dqdv_axes.append((ax_c, ax_d))

    cmap = plt.get_cmap("Set1")

    # ---------------- RAW DATA ----------------
    for i, mat in enumerate(st.session_state.raw_varM.keys()):

        color = cmap(i)
        df = st.session_state.raw_varM[mat][0].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        ax1.plot(df["timestamp"], df["voltage_V"], label=mat, color=color)
        ax2.plot(df["timestamp"], df["current_A"], label=mat, color=color)

    ax1.set_title("Voltage Profile")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Current Profile")
    ax2.legend()
    ax2.grid(True)

    # ---------------- SoH ----------------
    for i, mat in enumerate(st.session_state.full_results.keys()):

        color = cmap(i)

        full_df = st.session_state.full_results[mat]
        cap_df = st.session_state.capcheck_results[mat]

        if not full_df.empty:
            ax3.plot(full_df["cycle"], full_df["ave"], "--s", color=color, label=mat)
            ax3.errorbar(full_df["cycle"], full_df["ave"], full_df["std"], color=color)

        if not cap_df.empty:
            ax4.plot(cap_df["cycle"], cap_df["ave"], "--s", color=color, label=mat)
            ax4.errorbar(cap_df["cycle"], cap_df["ave"], cap_df["std"], color=color)

    ax3.set_title("Full Degradation")
    ax3.legend()
    ax3.grid(True)

    ax4.set_title("Capacity Check")
    ax4.legend()
    ax4.grid(True)

    # ---------------- dQdV ----------------
    for i, mat in enumerate(st.session_state.raw_varM.keys()):

        ax_c, ax_d = dqdv_axes[i]

        df = st.session_state.raw_varM[mat][0].copy()

        dqdv_charge = extract_dqdv_cycles(df, mode="charge")
        dqdv_discharge = extract_dqdv_cycles(df, mode="discharge")

        # Charge (Summer)
        if dqdv_charge:
            cycles = [d["cycle"] for d in dqdv_charge]
            cmap_c = plt.get_cmap("summer")
            norm = plt.Normalize(min(cycles), max(cycles))

            for d in dqdv_charge:
                ax_c.plot(d["V"], d["dqdv"], color=cmap_c(norm(d["cycle"])))

            sm = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm)
            sm.set_array([])
            
            divider = make_axes_locatable(ax_c)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(sm, cax=cax)

        ax_c.set_title(f"{mat} – Charge")
        ax_c.grid(True)

        # Discharge (Winter)
        if dqdv_discharge:
            cycles = [d["cycle"] for d in dqdv_discharge]
            cmap_d = plt.get_cmap("winter")
            norm = plt.Normalize(min(cycles), max(cycles))

            for d in dqdv_discharge:
                ax_d.plot(d["V"], d["dqdv"], color=cmap_d(norm(d["cycle"])))

            sm = plt.cm.ScalarMappable(cmap=cmap_d, norm=norm)
            sm.set_array([])

            divider = make_axes_locatable(ax_d)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(sm, cax=cax)

        ax_d.set_title(f"{mat} – Discharge")
        ax_d.grid(True)

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
