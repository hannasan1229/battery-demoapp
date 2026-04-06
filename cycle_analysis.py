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

    cycle_start = (sign < 0) & (sign.shift(1) > 0)

    df = df.copy()
    df["cycle"] = cycle_start.cumsum()

    cap = df.groupby("cycle")["Q_Ah"].max()

    soh = cap / cap.iloc[0] * 100

    return pd.DataFrame({"SoH": soh})


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

    ms = pd.concat(min_sums, axis=1)

    ms = zscore_check(ms)

    ms = ms.dropna(thresh=len(ms.columns) / 4)

    ms["ave"] = ms.mean(axis=1)
    ms["std"] = ms.std(axis=1)

    ms = down_check(ms)

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
            min_sums[mat].append(soh_df["SoH"])

    return min_sums


# ------------------------------------------------
# batch processing
# ------------------------------------------------

def process_batch(min_sums):

    results = {}

    for mat, data in min_sums.items():

        ms = cyctab_rev(data)
        results[mat] = ms

    return results


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
    ax.set_ylim(80, 100)
    ax.grid(True)

    ax.legend()
    plt.show()


# ------------------------------------------------
# main (Desktop usage)
# ------------------------------------------------

if __name__ == "__main__":

    project_path = input("Enter project path: ")

    DoE = load_project(project_path)

    min_sums = collect_data(DoE)

    results = process_batch(min_sums)

    plot_results(results)