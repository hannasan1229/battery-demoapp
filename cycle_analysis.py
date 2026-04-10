import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# ------------------------------------------------
# detect capacity check via lowest rounded C-rate
# ------------------------------------------------

def detect_capcheck_phase(df):

    df = df.copy()

    discharge = df["current_A"] < 0

    Q_ref = df.loc[discharge, "Q_Ah"].max()

    if pd.isna(Q_ref) or Q_ref == 0:
        return pd.Series(False, index=df.index)

    df["C_rate"] = ((df["current_A"].abs() / Q_ref) * 2).round() / 2

    valid_rates = sorted(
        df.loc[df["C_rate"] > 0, "C_rate"].unique()
    )

    if len(valid_rates) == 0:
        return pd.Series(False, index=df.index)

    cap_rate = valid_rates[0]

    return df["C_rate"] == cap_rate


# ------------------------------------------------
# compute SoH from capacity checks only
# ------------------------------------------------

def compute_soh(df):

    df = df.copy()

    is_cap = detect_capcheck_phase(df)

    discharge = df["current_A"] < 0

    cap_phase = is_cap & discharge

    new_block = cap_phase & ~cap_phase.shift(1).fillna(False)

    df["cap_block"] = new_block.cumsum()

    cap = df.loc[cap_phase].groupby("cap_block")["Q_Ah"].max()

    if len(cap) == 0:
        return pd.DataFrame()

    soh = cap / cap.iloc[0] * 100

    return pd.DataFrame({
        "Q_Ah": cap,
        "SoH": soh
    })


# ------------------------------------------------
# statistics functions
# ------------------------------------------------

def zscore_check(df):

    df = df.copy()

    for col in df.columns:

        data = df[col]

        df[col] = data.mask(
            (data - data.mean()).abs() > 2 * data.std()
        )

    return df


def down_check(df):

    df = df.copy()

    while (df["ave"].diff() > 0).any():

        df = df.drop(
            df[df["ave"].diff() > 0].index
        )

    return df


def cyctab_rev(min_sums):

    if len(min_sums) == 0:
        return pd.DataFrame()

    ms = pd.concat(min_sums, axis=1)

    ms.columns = [
        f"cell_{i}" for i in range(len(ms.columns))
    ]

    ms = zscore_check(ms)

    ms = ms.dropna(
        thresh=len(ms.columns) / 4
    )

    if ms.empty:
        return pd.DataFrame()

    ms["ave"] = ms.mean(axis=1)
    ms["std"] = ms.std(axis=1)

    ms = down_check(ms)

    return ms


# ------------------------------------------------
# Desktop loader
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
# collect data
# ------------------------------------------------

def collect_data(DoE):

    min_sums = {}

    for mat, dfs in DoE.items():

        min_sums[mat] = []

        for df in dfs:

            soh_df = compute_soh(df)

            if not soh_df.empty:

                min_sums[mat].append(
                    soh_df["SoH"]
                )

    return min_sums


# ------------------------------------------------
# batch processing
# ------------------------------------------------

def process_batch(min_sums):

    results = {}

    for mat, data in min_sums.items():

        if len(data) == 0:
            continue

        ms = cyctab_rev(data)

        if ms.empty:
            continue

        results[mat] = ms

    return results


# ------------------------------------------------
# 2-plot summary
# ------------------------------------------------

def plot_results(results):

    fig, ax = plt.subplots(
        1, 2,
        figsize=(14, 5)
    )

    cmap = plt.get_cmap("Set1")

    for i, (mat, df) in enumerate(results.items()):

        x = df.index
        y = df["ave"]
        e = df["std"]

        # LEFT plot
        ax[0].plot(
            x, y,
            "--s",
            label=mat,
            color=cmap(i)
        )

        ax[0].errorbar(
            x, y, e,
            capsize=4,
            color=cmap(i)
        )

        # RIGHT plot
        ax[1].scatter(
            x, y,
            s=70,
            label=mat,
            color=cmap(i)
        )

        ax[1].errorbar(
            x, y, e,
            fmt="none",
            capsize=4,
            color=cmap(i)
        )

    ax[0].set_title("SoH vs Capacity Check")
    ax[0].set_xlabel("Capacity Check Index")
    ax[0].set_ylabel("SoH [%]")
    ax[0].set_ylim(0, 105)
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_title("Capacity Check Summary")
    ax[1].set_xlabel("Capacity Check Index")
    ax[1].set_ylabel("SoH [%]")
    ax[1].set_ylim(0, 105)
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.show()


# ------------------------------------------------
# main
# ------------------------------------------------

if __name__ == "__main__":

    project_path = input("Enter project path: ")

    DoE = load_project(project_path)

    min_sums = collect_data(DoE)

    results = process_batch(min_sums)

    plot_results(results)
