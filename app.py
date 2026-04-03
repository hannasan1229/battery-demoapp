import streamlit as st
import demodata_cycle
import pandas as pd

st.title("Battery Cycle Demo Tool")

if st.button("Generate Demo"):

    df = demodata_cycle.generate_dataset(
        output_folder=None,
        n_cycle_blocks=3
    )

    # einfache Cycle-Erkennung
    df["sign"] = df["current_A"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df["sign"] = df["sign"].replace(0, None).ffill()
    df["cycle"] = (df["sign"].diff() == -2).cumsum()

    cyc = df.groupby("cycle")["Q_Ah"].max()

    st.line_chart(cyc)