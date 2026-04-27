import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# ------------------------------------------------
# compute SoH (DataFrame-based → Web + Desktop)
# ------------------------------------------------


def preprocess_cycles(df, threshold_factor=0.6):

    df = df.copy()

    I_max = df["current_A"].abs().max()
    threshold = I_max * threshold_factor

    sign = np.sign(df["current_A"])
    sign = pd.Series(sign).replace(0, np.nan).ffill()

    active = df["current_A"].abs() > threshold

    #cycle_start = (sign < 0) & (sign.shift(1) >= 0) & active
    cycle_start =(sign > 0) & (sign.shift(1) < 0) & active

    df["cycle"] = cycle_start.cumsum()
    df["cycle"] = df["cycle"].ffill()

    return df, threshold


def compute_soh(df):

    df, _ = preprocess_cycles(df)

    cap = df.groupby("cycle")["Q_Ah"].max()

    soh = cap / cap.iloc[0] * 100

    return pd.DataFrame({"cycle": cap.index, "SoH": soh.values})


# ------------------------------------------------
# dQ/dV calculation
# ------------------------------------------------


def compute_dqdv(df):

    df = df.sort_values("voltage_V")

    dQ = np.diff(df["Q_Ah"])
    dV = np.diff(df["voltage_V"])

    mask = np.abs(dV) > 1e-6

    dqdv = np.zeros_like(dQ)
    dqdv[mask] = dQ[mask] / dV[mask]

    V_mid = df["voltage_V"].values[:-1]

    return V_mid, dqdv


# ------------------------------------------------
# dQ/dV extraction per cycle
# ------------------------------------------------


def extract_dqdv_cycles(df, mode="charge"):

    df = df.copy()

    # 👉 SAFETY: cycle column prüfen
    if "cycle" not in df.columns:
        return []

    # 👉 nur echte Cycling-Daten
    df = df[df["test_type"] == "cycle"]

    if df.empty:
        return []

    # Charge / Discharge
    if mode == "charge":
        df = df[df["current_A"] > 0]
    else:
        df = df[df["current_A"] < 0]

    if df.empty:
        return []

    # Forward fill cycles
    df["cycle"] = df["cycle"].ffill()

    cycles = sorted(df["cycle"].dropna().unique())

    results = []

    for cycle in cycles:

        df_c = df[df["cycle"] == cycle]

        if len(df_c) < 10:
            continue

        V, dqdv = compute_dqdv(df_c)

        results.append({"cycle": cycle, "V": V, "dqdv": dqdv})

    return results


def compute_capacitycheck_soh(df, threshold_factor=0.6):

    # 🔥 EINMAL cycle korrekt berechnen
    df = df.copy()

    I_max = df["current_A"].abs().max()
    threshold = I_max * threshold_factor

    sign = np.sign(df["current_A"])
    sign = pd.Series(sign).replace(0, np.nan).ffill()

    active = df["current_A"].abs() > threshold

    cycle_start = (sign < 0) & (sign.shift(1) >= 0) & active

    df["cycle"] = cycle_start.cumsum()
    df["cycle"] = df["cycle"].ffill()

    # 🔥 Capacity Check = low current discharge
    cap_df = df[(df["current_A"] < 0) & (df["current_A"].abs() < threshold)]

    if cap_df.empty:
        return pd.DataFrame()

    # 🔥 pro cycle genau ein Punkt
    cap = cap_df.groupby("cycle")["Q_Ah"].max()

    if len(cap) == 0:
        return pd.DataFrame()

    soh = cap / cap.iloc[0] * 100

    return pd.DataFrame({"cycle": cap.index, "SoH": soh.values})


# ------------------------------------------------
# statistics functions
# ------------------------------------------------


def zscore_check(df):

    for col in df.select_dtypes(include=np.number).columns:
        data = df[col]
        df[col] = data.mask((data - data.mean()).abs() > 2 * data.std())

    return df


def down_check(df):

    while (df["ave"].diff() > 0).any():
        df = df.drop(df[df["ave"].diff() > 0].index)

    return df


def cyctab_rev(min_sums):

    if len(min_sums) == 0:
        return pd.DataFrame()

    ms = pd.concat(min_sums, axis=1)

    ms.columns = [f"cell_{i}" for i in range(len(ms.columns))]

    ms = zscore_check(ms)

    ms = ms.dropna(thresh=len(ms.columns) / 4)

    if ms.empty:  # 🔥 FIX
        return pd.DataFrame()

    ms["ave"] = ms.mean(axis=1)
    ms["std"] = ms.std(axis=1)

    ms = ms.reset_index().rename(columns={"index": "cycle"})

    return ms


# ------------------------------------------------
# Desktop loader (optional, bleibt erhalten)
# ------------------------------------------------


def load_project(project_path):

    varM = {}

    variant_names = os.listdir(project_path)

    for mat in variant_names:

        mat_path = os.path.join(project_path, mat)

        if not os.path.isdir(mat_path):
            continue

        datasets = []

        for d in os.listdir(mat_path):
            folder = os.path.join(mat_path, d)

            file = os.path.join(folder, "combined_test.csv")

            if os.path.exists(file):
                df = pd.read_csv(file)
                datasets.append(df)

        varM[mat] = datasets

    return varM


# ------------------------------------------------
# collect data (DataFrame-based)
# ------------------------------------------------


def collect_data(varM):

    min_sums = {}

    for mat, dfs in varM.items():

        min_sums[mat] = []

        for df in dfs:

            soh_df = compute_soh(df)

            if not soh_df.empty:  # 🔥 FIX
                min_sums[mat].append(soh_df["SoH"])

    return min_sums


# ------------------------------------------------
# batch processing
# ------------------------------------------------


def process_batch(varM):
    """
    Process complete varM dictionary and return:
    1) Full SoH results for all discharge cycles
    2) Capacity-check-only SoH results

    Parameters
    ----------
    varM : dict
        {
            "Material_A": [df_cell1, df_cell2, ...],
            "Material_B": [df_cell1, df_cell2, ...],
        }

    Returns
    -------
    full_results : dict
        Aggregated full cycling SoH statistics per material

    capcheck_results : dict
        Aggregated capacity-check-only SoH statistics per material
    """

    full_results = {}
    capcheck_results = {}

    for mat, dfs in varM.items():

        full_data = []
        cap_data = []

        for df in dfs:

            # -------------------------------
            # Full SoH Calculation
            # -------------------------------
            full_soh = compute_soh(df)

            if not full_soh.empty and "cycle" in full_soh.columns:
                full_data.append(full_soh.set_index("cycle")["SoH"])

            # -------------------------------
            # Capacity Check SoH Calculation
            # -------------------------------
            cap_soh = compute_capacitycheck_soh(df)

            if not cap_soh.empty and "cycle" in cap_soh.columns:
                cap_data.append(cap_soh.set_index("cycle")["SoH"])

        # -------------------------------
        # Aggregate Statistics
        # -------------------------------
        if len(full_data) > 0:
            full_results[mat] = cyctab_rev(full_data)
        else:
            full_results[mat] = pd.DataFrame()

        if len(cap_data) > 0:
            capcheck_results[mat] = cyctab_rev(cap_data)
        else:
            capcheck_results[mat] = pd.DataFrame()

    return full_results, capcheck_results


# ------------------------------------------------
# plot (Desktop)
# ------------------------------------------------


def plot_results(results):

    fig, ax = plt.subplots(figsize=(6, 5))

    cmap = plt.get_cmap("Set1")

    for i, (mat, df) in enumerate(results.items()):

        x = df.index
        y = df["ave"]
        e = df["std"]

        ax.plot(x, y, "--s", label=mat, color=cmap(i))
        ax.errorbar(x, y, e, capsize=4, color=cmap(i))

    ax.set_xlabel("Cycle")
    ax.set_ylabel("SoH [%]")
    ax.set_ylim(0, 100)
    ax.grid(True)

    ax.legend()
    plt.show()


# ------------------------------------------------
# main (Desktop usage)
# ------------------------------------------------

if __name__ == "__main__":

    project_path = input("Enter project path: ")

    varM = load_project(project_path)

    full_results, capcheck_results = process_batch(varM)

    plot_results(full_results)
