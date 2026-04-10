import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# ------------------------------------------------
# compute SoH (DataFrame-based → Web + Desktop)
# ------------------------------------------------


def compute_soh(df):

    sign = np.sign(df["current_A"])
    sign = pd.Series(sign).replace(0, np.nan).ffill()

    cycle_start = (sign < 0) & (sign.shift(1) >= 0)  # 🔥 erlaubt 0 → -0)

    df = df.copy()
    df["cycle"] = cycle_start.cumsum()

    cap = df.groupby("cycle")["Q_Ah"].max()

    # 🔥 FIX: leere Fälle abfangen
    if len(cap) == 0:
        return pd.DataFrame({"SoH": []})

    soh = cap / cap.iloc[0] * 100

    return pd.DataFrame({
    "cycle": cap.index,
    "SoH": soh.values
})

def compute_capacitycheck_soh(df, threshold_factor=0.6):

    full_soh = compute_soh(df)

    I_max = df["current_A"].abs().max()
    threshold = I_max * threshold_factor

    sign = np.sign(df["current_A"])
    sign = pd.Series(sign).replace(0, np.nan).ffill()

    cycle_start = (sign < 0) & (sign.shift(1) >= 0)
    df = df.copy()
    df["cycle"] = cycle_start.cumsum()

    capcheck_cycles = df.loc[
        (df["current_A"] < 0) &
        (df["current_A"].abs() < threshold),
        "cycle"
    ].unique()

    return full_soh[full_soh["cycle"].isin(capcheck_cycles)]


# ------------------------------------------------
# statistics functions
# ------------------------------------------------


def zscore_check(df):

    for col in df.columns:
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

    DoE = {}

    materials = os.listdir(project_path)

    for mat in materials:

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

        DoE[mat] = datasets

    return DoE


# ------------------------------------------------
# collect data (DataFrame-based)
# ------------------------------------------------


def collect_data(DoE):

    min_sums = {}

    for mat, dfs in DoE.items():

        min_sums[mat] = []

        for df in dfs:

            soh_df = compute_soh(df)

            if not soh_df.empty:  # 🔥 FIX
                min_sums[mat].append(soh_df["SoH"])

    return min_sums


# ------------------------------------------------
# batch processing
# ------------------------------------------------

def process_batch(DoE):
    """
    Process complete DoE dictionary and return:
    1) Full SoH results for all discharge cycles
    2) Capacity-check-only SoH results

    Parameters
    ----------
    DoE : dict
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

    for mat, dfs in DoE.items():

        full_data = []
        cap_data = []

        for df in dfs:

            # -------------------------------
            # Full SoH Calculation
            # -------------------------------
            full_soh = compute_soh(df)

            if not full_soh.empty and "cycle" in full_soh.columns:
                full_data.append(
                    full_soh.set_index("cycle")["SoH"]
                )

            # -------------------------------
            # Capacity Check SoH Calculation
            # -------------------------------
            cap_soh = compute_capacitycheck_soh(df)

            if not cap_soh.empty and "cycle" in cap_soh.columns:
                cap_data.append(
                    cap_soh.set_index("cycle")["SoH"]
                )

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

    DoE = load_project(project_path)

    full_results, capcheck_results = process_batch(DoE)

    plot_results(full_results)
