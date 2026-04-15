import numpy as np
import pandas as pd

# ------------------------------------------------
# Cycle preprocessing
# ------------------------------------------------

def preprocess_cycles(df, threshold_factor=0.6):

    df = df.copy()

    I_max = df["current_A"].abs().max()
    threshold = I_max * threshold_factor

    sign = np.sign(df["current_A"])
    sign = pd.Series(sign).replace(0, np.nan).ffill()

    active = df["current_A"].abs() > threshold

    cycle_start = (
        (sign < 0) &
        (sign.shift(1) >= 0) &
        active
    )

    df["cycle"] = cycle_start.cumsum()
    df["cycle"] = df["cycle"].ffill()

    return df, threshold


# ------------------------------------------------
# SoH
# ------------------------------------------------

def compute_soh(df):

    df, _ = preprocess_cycles(df)

    cap = df.groupby("cycle")["Q_Ah"].max()

    soh = cap / cap.iloc[0] * 100

    return pd.DataFrame({
        "cycle": cap.index,
        "SoH": soh.values
    })


def compute_capacitycheck_soh(df, threshold_factor=0.6):

    df, threshold = preprocess_cycles(df, threshold_factor)

    cap_df = df[
        (df["current_A"] < 0) &
        (df["current_A"].abs() < threshold)
    ]

    cap = cap_df.groupby("cycle")["Q_Ah"].max()

    soh = cap / cap.iloc[0] * 100

    return pd.DataFrame({
        "cycle": cap.index,
        "SoH": soh.values
    })


# ------------------------------------------------
# dQ/dV
# ------------------------------------------------

def compute_dqdv_curves(df, threshold_factor=0.6):

    df, threshold = preprocess_cycles(df, threshold_factor)

    cap_df = df[
        (df["current_A"].abs() < threshold) &
        (df["current_A"] != 0)
    ].copy()

    if cap_df.empty:
        return [], []

    charge_curves = []
    discharge_curves = []

    for sign_val in [1, -1]:

        sub = cap_df[np.sign(cap_df["current_A"]) == sign_val]

        if sub.empty:
            continue

        splits = np.where(np.diff(sub.index) > 1)[0] + 1
        groups = np.split(sub, splits)

        for g in groups:

            g = g.dropna(subset=["Q_Ah", "voltage_V"]).reset_index(drop=True)

            if len(g) < 20:
                continue

            try:
                dQ = np.gradient(g["Q_Ah"].values)
                dV = np.gradient(g["voltage_V"].values)

                dV[np.abs(dV) < 1e-6] = np.nan
                dqdv = dQ / dV

            except:
                continue

            if sign_val > 0:
                charge_curves.append((g["voltage_V"].values, dqdv))
            else:
                discharge_curves.append((g["voltage_V"].values, dqdv))

    return charge_curves, discharge_curves


# ------------------------------------------------
# Aggregation
# ------------------------------------------------

def zscore_check(df):

    for col in df.select_dtypes(include=np.number).columns:
        data = df[col]
        df[col] = data.mask((data - data.mean()).abs() > 2 * data.std())

    return df


def cyctab_rev(min_sums):

    if len(min_sums) == 0:
        return pd.DataFrame()

    ms = pd.concat(min_sums, axis=1)
    ms.columns = [f"cell_{i}" for i in range(len(ms.columns))]

    ms = zscore_check(ms)
    ms = ms.dropna(thresh=len(ms.columns) / 4)

    if ms.empty:
        return pd.DataFrame()

    ms["ave"] = ms.mean(axis=1)
    ms["std"] = ms.std(axis=1)

    ms = ms.reset_index().rename(columns={"index": "cycle"})

    return ms


# ------------------------------------------------
# Batch
# ------------------------------------------------

def process_batch(varM):

    full_results = {}
    capcheck_results = {}

    for mat, dfs in varM.items():

        full_data = []
        cap_data = []

        for df in dfs:

            full_soh = compute_soh(df)
            cap_soh = compute_capacitycheck_soh(df)

            if not full_soh.empty:
                full_data.append(full_soh.set_index("cycle")["SoH"])

            if not cap_soh.empty:
                cap_data.append(cap_soh.set_index("cycle")["SoH"])

        full_results[mat] = cyctab_rev(full_data)
        capcheck_results[mat] = cyctab_rev(cap_data)

    return full_results, capcheck_results
