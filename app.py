import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from demodata_cycle import generate_varM_dataframes
from cycle_analysis import process_batch, extract_dqdv_cycles


# -----------------------------
# Language Handling (NEU)
# -----------------------------
params = st.query_params
lang = params.get("lang", "en")

TEXTS = {
    "en": {
        "title": "🔋 Battery Cycle Analysis Tool",
        "intro": """This demo showcases automated battery data analysis, including:
- Cycle detection  
- State-of-Health (SoH) evaluation  
- Capacity check extraction  
- Statistical aggregation across multiple cells  
""",
        "setup": "⚙️ Variants Setup (VarM)",
        "setup_caption": "Define different material variants (VarM) for comparative testing.",
        "n_variants": "Number of variants",
        "n_blocks": "Number of cycle blocks",
        "cycles_per_block": "Cycles per block",
        "run": "🚀 Run Analysis",
        "running": "Generating and analyzing data...",
        "done": "Analysis complete!",
        "aging": "📊 Aging & Performance Analysis",
        "show_results": "🔍 Show processed results",
    },
    "de": {
        "title": "🔋 Batterie-Zyklenanalyse",
        "intro": """Diese Demo zeigt automatisierte Batteriedatenanalyse:
- Zyklenerkennung  
- SoH-Bewertung  
- Kapazitätsprüfung  
- Statistische Auswertung mehrerer Zellen  
""",
        "setup": "⚙️ Varianten Setup (VarM)",
        "setup_caption": "Materialvarianten für Vergleichstests definieren.",
        "n_variants": "Anzahl Varianten",
        "n_blocks": "Anzahl Zyklenblöcke",
        "cycles_per_block": "Zyklen pro Block",
        "run": "🚀 Analyse starten",
        "running": "Daten werden generiert und analysiert...",
        "done": "Analyse abgeschlossen!",
        "aging": "📊 Alterung & Performance",
        "show_results": "🔍 Ergebnisse anzeigen",
    },
    "ja": {
        "title": "🔋 バッテリーサイクル解析",
        "intro": """このデモではバッテリーデータの自動解析を行います：
- サイクル検出  
- SOH評価  
- 容量チェック  
- 統計解析  
""",
        "setup": "⚙️ バリアント設定",
        "setup_caption": "材料バリアントを定義します。",
        "n_variants": "バリアント数",
        "n_blocks": "サイクルブロック数",
        "cycles_per_block": "ブロックあたりのサイクル数",
        "run": "🚀 解析開始",
        "running": "解析中...",
        "done": "解析完了",
        "aging": "📊 劣化解析",
        "show_results": "🔍 結果表示",
    }
}

t = TEXTS.get(lang, TEXTS["en"])


# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Battery Analysis Tool", layout="centered")

st.title(t["title"])
st.markdown(t["intro"])


# ----------------------------------
# User Input
# ----------------------------------

st.header(t["setup"])
st.caption(t["setup_caption"])

n_mat = st.number_input(t["n_variants"], min_value=1, max_value=10, value=2)

colA, colB = st.columns(2)

with colA:
    n_cycle_blocks = st.number_input(
        t["n_blocks"], min_value=1, max_value=20, value=3
    )

with colB:
    n_cycles = st.number_input(
        t["cycles_per_block"], min_value=1, max_value=100, value=10
    )

st.caption(f"Total cycles ≈ {n_cycle_blocks * n_cycles}")


# ----------------------------------
# Materials
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

# 👉 detect changes in setup
current_config = (materials, n_cycle_blocks, n_cycles)

if "last_config" not in st.session_state:
    st.session_state.last_config = current_config

if st.session_state.last_config != current_config:
    st.session_state.raw_varM = None
    st.session_state.full_results = None
    st.session_state.capcheck_results = None
    st.session_state.last_config = current_config

# ----------------------------------
# Cached Functions
# ----------------------------------

@st.cache_data(show_spinner=False)
def cached_generate(materials, n_cycle_blocks, n_cycles):
    return generate_varM_dataframes(
        materials, n_cycle_blocks=n_cycle_blocks, n_cycles=n_cycles
    )


@st.cache_data(show_spinner=False)
def cached_process(varM):
    return process_batch(varM)


# ----------------------------------
# Run Simulation
# ----------------------------------

if st.button(t["run"]):

    with st.spinner(t["running"]):

        if st.session_state.raw_varM is None:

            st.write("⏳ Generating data...")

            varM = cached_generate(materials, n_cycle_blocks, n_cycles)
            st.session_state.raw_varM = varM

            st.write("⚙️ Processing...")

            full_results, capcheck_results = cached_process(varM)

            st.session_state.full_results = full_results
            st.session_state.capcheck_results = capcheck_results

        else:
            st.write("⚡ Using cached data")

    st.success(t["done"])


# ----------------------------------
# Plot Results
# ----------------------------------

if (
    st.session_state.full_results is not None
    and st.session_state.capcheck_results is not None
    and st.session_state.raw_varM is not None
):

    st.header(t["aging"])

    n_var = len(st.session_state.raw_varM)
    rows_needed = 3 + n_var

    fig = plt.figure(figsize=(14, 3.5 * rows_needed), constrained_layout=True)
    gs = fig.add_gridspec(rows_needed, 2)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

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

        # Charge
        if dqdv_charge:
            cycles = [d["cycle"] for d in dqdv_charge]
            cmap_c = plt.get_cmap("summer")
            norm = plt.Normalize(min(cycles), max(cycles))

            for d in dqdv_charge:
                ax_c.plot(d["V"], d["dqdv"], color=cmap_c(norm(d["cycle"])))

            sm = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm)
            divider = make_axes_locatable(ax_c)
            cax = divider.append_axes("right", size="4%", pad=0.05)
            fig.colorbar(sm, cax=cax)

        ax_c.set_title(f"{mat} – Charge")
        ax_c.grid(True)

        # Discharge
        if dqdv_discharge:
            cycles = [d["cycle"] for d in dqdv_discharge]
            cmap_d = plt.get_cmap("winter")
            norm = plt.Normalize(min(cycles), max(cycles))

            for d in dqdv_discharge:
                ax_d.plot(d["V"], d["dqdv"], color=cmap_d(norm(d["cycle"])))

            sm = plt.cm.ScalarMappable(cmap=cmap_d, norm=norm)
            divider = make_axes_locatable(ax_d)
            cax = divider.append_axes("right", size="4%", pad=0.05)
            fig.colorbar(sm, cax=cax)

        ax_d.set_title(f"{mat} – Discharge")
        ax_d.grid(True)

    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------
# Raw Data Preview
# ----------------------------------

with st.expander(t["show_results"]):

    if st.session_state.full_results is not None:

        for mat in st.session_state.full_results.keys():

            st.write(f"### {mat} – Full Degradation")
            st.dataframe(st.session_state.full_results[mat].head())

            st.write(f"### {mat} – Capacity Checks")
            st.dataframe(st.session_state.capcheck_results[mat].head())
