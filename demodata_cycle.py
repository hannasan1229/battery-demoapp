import numpy as np
import pandas as pd
from datetime import datetime
import os

# ------------------------------------------------
# global parameters
# ------------------------------------------------

capacity_nom = 1.0
R_internal = 0.02

dt = 10

charge_rate_C = 1.0
discharge_rate_C = 1.0

rest_steps = 10

SOC_start = 0.20
SOC_min = 0.05
SOC_max = 0.95

capacity_fade_per_cycle = 0.01

# 🔥 EIN gemeinsamer Zeitstart
start_time = datetime.now()

# ------------------------------------------------
# OCV model
# ------------------------------------------------


def ocv(soc):

    soc = np.clip(soc, 0, 1)

    V = 3.0 + 0.85 * soc

    # Plateau-Strukturen
    V += 0.10 * np.tanh((soc - 0.2) * 10)
    V += 0.08 * np.tanh((soc - 0.5) * 12)
    V += 0.06 * np.tanh((soc - 0.75) * 15)

    # High voltage region
    V += 0.20 / (1 + np.exp(-(soc - 0.9) * 25))

    return V


# ------------------------------------------------
# material variation
# ------------------------------------------------


def get_material_fade(base_fade, direction=None):

    if direction is None:
        direction = -1

    variation = 1 + direction * np.random.uniform(0.1, 0.4)

    return base_fade * variation


# ------------------------------------------------
# cycle block generator
# ------------------------------------------------


def generate_cycle_block(
    soc,
    Q,
    capacity,
    block_id,
    fade,
    elapsed_time,
    n_cycles=10,
):

    rows = []

    temperature = 25

    for cycle in range(n_cycles):

        I_charge = capacity * charge_rate_C
        I_discharge = -capacity * discharge_rate_C

        # ============================================
        # CHARGE
        # ============================================

        while soc < SOC_max - 1e-6:

            Q += I_charge * dt / 3600

            soc = np.clip(Q / capacity, 0, 1)

            noise = np.random.normal(0, 0.002)

            V = ocv(soc) + I_charge * R_internal + noise

            timestamp = start_time + pd.Timedelta(seconds=elapsed_time)

            rows.append(
                {
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "test_type": "cycle",
                    "cycle_block": block_id,
                    "cycle": cycle,
                    "SOC": soc,
                    "Q_Ah": Q,
                    "current_A": I_charge,
                    "voltage_V": V,
                    "temperature_C": temperature,
                }
            )

            elapsed_time += dt

        # ============================================
        # REST
        # ============================================

        for _ in range(rest_steps):

            timestamp = start_time + pd.Timedelta(seconds=elapsed_time)

            rows.append(
                {
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "test_type": "rest",
                    "cycle_block": block_id,
                    "cycle": cycle,
                    "SOC": soc,
                    "Q_Ah": Q,
                    "current_A": 0,
                    "voltage_V": ocv(soc),
                    "temperature_C": temperature,
                }
            )

            elapsed_time += dt

        # ============================================
        # DISCHARGE
        # ============================================

        while soc > SOC_min + 1e-6:

            Q += I_discharge * dt / 3600

            soc = np.clip(Q / capacity, 0, 1)

            noise = np.random.normal(0, 0.002)

            V = ocv(soc) + I_discharge * R_internal + noise

            timestamp = start_time + pd.Timedelta(seconds=elapsed_time)

            rows.append(
                {
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "test_type": "cycle",
                    "cycle_block": block_id,
                    "cycle": cycle,
                    "SOC": soc,
                    "Q_Ah": Q,
                    "current_A": I_discharge,
                    "voltage_V": V,
                    "temperature_C": temperature,
                }
            )

            elapsed_time += dt

        # ============================================
        # REST
        # ============================================

        for _ in range(rest_steps):

            timestamp = start_time + pd.Timedelta(seconds=elapsed_time)

            rows.append(
                {
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "test_type": "rest",
                    "cycle_block": block_id,
                    "cycle": cycle,
                    "SOC": soc,
                    "Q_Ah": Q,
                    "current_A": 0,
                    "voltage_V": ocv(soc),
                    "temperature_C": temperature,
                }
            )

            elapsed_time += dt

        # ============================================
        # CAPACITY FADE
        # ============================================

        capacity *= 1 - fade

    return pd.DataFrame(rows), soc, Q, capacity, elapsed_time


# ------------------------------------------------
# capacity check
# ------------------------------------------------


def generate_capacity_check(
    soc,
    Q,
    capacity,
    elapsed_time,
):

    rows = []

    temperature = 25

    I_charge = 0.5 * capacity
    I_discharge = -0.5 * capacity

    # ============================================
    # CAPACITY CHARGE
    # ============================================

    while soc < 0.99:

        Q += I_charge * dt / 3600

        soc = np.clip(Q / capacity, 0, 1)

        timestamp = start_time + pd.Timedelta(seconds=elapsed_time)

        rows.append(
            {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "test_type": "capacity_charge",
                "SOC": soc,
                "Q_Ah": Q,
                "current_A": I_charge,
                "voltage_V": ocv(soc),
                "temperature_C": temperature,
            }
        )

        elapsed_time += dt

    # ============================================
    # CAPACITY DISCHARGE
    # ============================================

    while soc > SOC_min + 1e-6:

        Q += I_discharge * dt / 3600

        soc = np.clip(Q / capacity, 0, 1)

        timestamp = start_time + pd.Timedelta(seconds=elapsed_time)

        rows.append(
            {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "test_type": "capacity_discharge",
                "SOC": soc,
                "Q_Ah": Q,
                "current_A": I_discharge,
                "voltage_V": ocv(soc),
                "temperature_C": temperature,
            }
        )

        elapsed_time += dt

    return pd.DataFrame(rows), soc, Q, elapsed_time


# ------------------------------------------------
# combine dataset
# ------------------------------------------------


def combine_dataframe(
    n_cycle_blocks=3,
    n_cycles=10,
    output_folder=None,
    fade=capacity_fade_per_cycle,
):

    dfs = []

    # 🔥 Sekunden-Zähler
    elapsed_time = 0

    soc = SOC_start

    capacity = capacity_nom

    Q = soc * capacity

    # ============================================
    # INITIAL CAPACITY CHECK
    # ============================================

    df_cap0, soc, Q, elapsed_time = generate_capacity_check(
        soc,
        Q,
        capacity,
        elapsed_time,
    )

    dfs.append(df_cap0)

    # ============================================
    # MAIN LOOP
    # ============================================

    for block in range(n_cycle_blocks):

        # cycle block
        df_block, soc, Q, capacity, elapsed_time = generate_cycle_block(
            soc,
            Q,
            capacity,
            block,
            fade,
            elapsed_time,
            n_cycles=n_cycles,
        )

        dfs.append(df_block)

        # capacity check
        df_cap, soc, Q, elapsed_time = generate_capacity_check(
            soc,
            Q,
            capacity,
            elapsed_time,
        )

        dfs.append(df_cap)

    # ============================================
    # COMBINE
    # ============================================

    final_df = pd.concat(dfs, ignore_index=True)

    # optional export
    if output_folder is not None:

        os.makedirs(output_folder, exist_ok=True)

        combined_path = os.path.join(
            output_folder,
            "combined_test.csv"
        )

        final_df.to_csv(combined_path, index=False)

    return final_df


# ------------------------------------------------
# dataset generator
# ------------------------------------------------


def generate_dataset(
    output_folder=None,
    n_cycle_blocks=3,
    n_cycles=10,
    fade=capacity_fade_per_cycle,
):

    return combine_dataframe(
        n_cycle_blocks=n_cycle_blocks,
        n_cycles=n_cycles,
        output_folder=output_folder,
        fade=fade,
    )


# ------------------------------------------------
# user input
# ------------------------------------------------


def user_input_varM():

    materials = {}

    n_var = int(input("How many materials? (max 10): "))

    for i in range(n_var):

        name = input(f"Material name (A,B,C...): ")

        n_cells = int(
            input(f"How many cells for {name}?: ")
        )

        materials[name] = {
            "n_cells": n_cells,
            "direction": None,
        }

    return materials


# ------------------------------------------------
# main varM generator
# ------------------------------------------------


def generate_varM_datasets(
    materials,
    project_name,
    base_folder="demo_data"
):

    project_path = os.path.join(
        base_folder,
        f"Projekt_{project_name}"
    )

    os.makedirs(project_path, exist_ok=True)

    for mat, props in materials.items():

        variant_path = os.path.join(
            project_path,
            f"Variant_{mat}"
        )

        os.makedirs(variant_path, exist_ok=True)

        for i in range(1, props["n_cells"] + 1):

            fade = get_material_fade(
                capacity_fade_per_cycle,
                props["direction"]
            )

            timestamp = datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )

            dataset_path = os.path.join(
                variant_path,
                f"dataset_{timestamp}"
            )

            generate_dataset(
                output_folder=dataset_path,
                n_cycle_blocks=3,
                fade=fade,
            )

            print(f"✔ Created: {dataset_path}")


# ------------------------------------------------
# Streamlit dataframe generator
# ------------------------------------------------


def generate_varM_dataframes(
    materials,
    n_cycle_blocks=3,
    n_cycles=10
):

    varM = {}

    for mat, props in materials.items():

        varM[mat] = []

        for i in range(props["n_cells"]):

            fade = get_material_fade(
                capacity_fade_per_cycle,
                props["direction"]
            )

            df = generate_dataset(
                output_folder=None,
                n_cycle_blocks=n_cycle_blocks,
                n_cycles=n_cycles,
                fade=fade,
            )

            varM[mat].append(df)

    return varM


# ------------------------------------------------
# run
# ------------------------------------------------

if __name__ == "__main__":

    project_name = input("Project name: ")

    materials = user_input_varM()

    generate_varM_datasets(
        materials,
        project_name
    )
