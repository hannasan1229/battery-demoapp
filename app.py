import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from demodata_cycle import generate_varM_dataframes
from cycle_analysis import process_batch, compute_dqdv_curves

st.set_page_config(page_title="Battery Analysis Tool", layout="centered")

st.title("🔋 Battery Cycle Analysis Tool")

# ----------------------------------
# User Input
# ----------------------------------

n_mat = st.number_input("Number of variants", 1, 10, 2)

materials = {}

for i in range(n_mat):

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input(f"Variant {i+1}", value=f"Material-{chr(65+i)}")

    with col2:
        n_cells = st.number_input(f"Cells for {name}", 1, 10, 2, key=f"cells_{i}")

    materials[name] = {"n_cells": n_cells, "direction": None}

# ----------------------------------
# Run
# ----------------------------------

if st.button("Run Analysis"):

    varM = generate_varM_dataframes(materials)

    st.session_state.raw_varM = varM
    full, cap = process_batch(varM)

    st.session_state.full = full
    st.session_state.cap = cap

# ----------------------------------
# Plot
# ----------------------------------

if "full" in st.session_state:

    varM = st.session_state.raw_varM

    n_var = len(varM)
    rows = 3 + n_var

    fig = plt.figure(figsize=(14, 4 * rows))
    gs = fig.add_gridspec(rows, 2)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    cmap = plt.get_cmap("Set1")

    # RAW
    for i, mat in enumerate(varM):

        df = varM[mat][0].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["t"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

        ax1.plot(df["t"], df["voltage_V"], label=mat, color=cmap(i))
        ax2.plot(df["t"], df["current_A"], label=mat, color=cmap(i))

    ax1.set_title("Voltage")
    ax2.set_title("Current")
    ax1.legend()
    ax2.legend()

    # SoH
    for i, mat in enumerate(st.session_state.full):

        f = st.session_state.full[mat]
        c = st.session_state.cap[mat]

        if not f.empty:
            ax3.plot(f["cycle"], f["ave"], label=mat, color=cmap(i))

        if not c.empty:
            ax4.plot(c["cycle"], c["ave"], label=mat, color=cmap(i))

    ax3.set_title("SoH Full")
    ax4.set_title("SoH Capacity Check")
    ax3.legend()
    ax4.legend()

    # dQ/dV
    for i, mat in enumerate(varM):

        df = varM[mat][0]

        ch, dch = compute_dqdv_curves(df)

        ax_ch = fig.add_subplot(gs[3 + i, 0])
        ax_dch = fig.add_subplot(gs[3 + i, 1])

        for j, (V, dqdv) in enumerate(ch):
            ax_ch.plot(V, dqdv, color=plt.cm.winter(j / max(len(ch)-1, 1)))

        for j, (V, dqdv) in enumerate(dch):
            ax_dch.plot(V, dqdv, color=plt.cm.summer(j / max(len(dch)-1, 1)))

        ax_ch.set_title(f"{mat} Charge")
        ax_dch.set_title(f"{mat} Discharge")

    fig.tight_layout()
    st.pyplot(fig)
