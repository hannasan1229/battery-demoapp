import numpy as np
import pandas as pd
from datetime import datetime
import os

# ------------------------------------------------
# global parameters
# ------------------------------------------------

capacity_nom = 1.0
R_internal = 0.02

dt = 60

charge_rate_C = 1.0
discharge_rate_C = 1.0

rest_steps = 10

SOC_start = 0.20
SOC_min = 0.05
SOC_max = 0.95

capacity_fade_per_cycle = 0.0005


# ------------------------------------------------
# OCV model
# ------------------------------------------------


def ocv(soc):

    soc = np.clip(soc, 0, 1)

    return (
        3.0
        + 0.9 * soc
        + 0.25 * np.tanh((soc - 0.5) * 8)
        + 0.03 * np.tanh((soc - 0.9) * 30)
    )


# ------------------------------------------------
# material variation
# ------------------------------------------------


def get_material_fade(base_fade, direction=None):

    if direction is None:
        direction = np.random.choice([-1, 1])

    variation = 1 + direction * np.random.uniform(0, 0.08)

    return base_fade * variation


# ------------------------------------------------
# cycle block generator
# ------------------------------------------------


def generate_cycle_block(soc, Q, capacity, block_id, fade, n_cycles=10):

    global current_time

    rows = []
    temperature = 25

    for cycle in range(n_cycles):

        I_charge = capacity * charge_rate_C
        I_discharge = -capacity * discharge_rate_C

        # ---------------- charge ----------------
        while soc < SOC_max:

            Q += I_charge * dt / 3600
            soc = np.clip(Q / capacity, 0, 1)

            V = ocv(soc) + I_charge * R_internal

            rows.append(
                {
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
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

            current_time += pd.Timedelta(seconds=dt)

        # ---------------- rest ----------------
        for _ in range(rest_steps):

            rows.append(
                {
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
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

            current_time += pd.Timedelta(seconds=dt)

        # ---------------- discharge ----------------
        while soc > SOC_min:

            Q += I_discharge * dt / 3600
            soc = np.clip(Q / capacity, 0, 1)

            V = ocv(soc) + I_discharge * R_internal

            rows.append(
                {
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
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

            current_time += pd.Timedelta(seconds=dt)

        # ---------------- rest ----------------
        for _ in range(rest_steps):

            rows.append(
                {
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
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

            current_time += pd.Timedelta(seconds=dt)

        # capacity fade
        capacity *= 1 - fade

    return pd.DataFrame(rows), soc, Q, capacity


# ------------------------------------------------
# capacity check
# ------------------------------------------------


def generate_capacity_check(soc, Q, capacity):

    global current_time

    rows = []
    temperature = 25

    I_charge = 0.5 * capacity
    I_discharge = -0.5 * capacity

    while soc < 0.99:

        Q += I_charge * dt / 3600
        soc = np.clip(Q / capacity, 0, 1)

        rows.append(
            {
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_type": "capacity_charge",
                "SOC": soc,
                "Q_Ah": Q,
                "current_A": I_charge,
                "voltage_V": ocv(soc),
                "temperature_C": temperature,
            }
        )

        current_time += pd.Timedelta(seconds=dt)

    while soc > SOC_min:

        Q += I_discharge * dt / 3600
        soc = np.clip(Q / capacity, 0, 1)

        rows.append(
            {
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_type": "capacity_discharge",
                "SOC": soc,
                "Q_Ah": Q,
                "current_A": I_discharge,
                "voltage_V": ocv(soc),
                "temperature_C": temperature,
            }
        )

        current_time += pd.Timedelta(seconds=dt)

    return pd.DataFrame(rows), soc, Q


# ------------------------------------------------
# combine dataset
# ------------------------------------------------


def combine_dataframe(
    n_cycle_blocks=3, output_folder=None, fade=capacity_fade_per_cycle
):

    global current_time

    dfs = []

    soc = SOC_start
    capacity = capacity_nom
    Q = soc * capacity

    for block in range(n_cycle_blocks):

        # 🔹 cycle block
        df_block, soc, Q, capacity = generate_cycle_block(soc, Q, capacity, block, fade)

        # 👉 NEU: speichern
        if output_folder:
            cycle_path = os.path.join(output_folder, f"cycle_block_{block}.csv")
            df_block.to_csv(cycle_path, index=False)

        dfs.append(df_block)

        # 🔹 capacity check
        df_cap, soc, Q = generate_capacity_check(soc, Q, capacity)

        # 👉 NEU: speichern
        if output_folder:
            cap_path = os.path.join(output_folder, f"capacity_check_{block}.csv")
            df_cap.to_csv(cap_path, index=False)

        dfs.append(df_cap)

    # 🔹 combined file
    final_df = pd.concat(dfs, ignore_index=True)
    if output_folder:
        combined_path = os.path.join(output_folder, "combined_test.csv")
        final_df.to_csv(combined_path, index=False)

    return final_df


# ------------------------------------------------
# dataset generator
# ------------------------------------------------


def generate_dataset(
    output_folder=None, n_cycle_blocks=3, fade=capacity_fade_per_cycle
):

    global current_time
    current_time = datetime.now()

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    return combine_dataframe(
        n_cycle_blocks=n_cycle_blocks, output_folder=output_folder, fade=fade
    )


# ------------------------------------------------
# user input
# ------------------------------------------------


def user_input_DoE():

    materials = {}

    n_mat = int(input("How many materials? (max 10): "))

    for i in range(n_mat):

        name = input(f"Material name (A,B,C...): ")
        n_cells = int(input(f"How many cells for {name}?: "))

        materials[name] = {
            "n_cells": n_cells,
            "direction": None,  # optional später steuerbar
        }

    return materials


# ------------------------------------------------
# main DoE generator
# ------------------------------------------------


def generate_DoE_datasets(materials, project_name, base_folder="demo_data"):

    # 🔹 Projektordner
    project_path = os.path.join(base_folder, f"Projekt_{project_name}")
    os.makedirs(project_path, exist_ok=True)

    for mat, props in materials.items():

        # 🔹 Materialordner
        mat_path = os.path.join(project_path, f"Material_{mat}")
        os.makedirs(mat_path, exist_ok=True)

        for i in range(1, props["n_cells"] + 1):

            fade = get_material_fade(capacity_fade_per_cycle, props["direction"])

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            dataset_path = os.path.join(mat_path, f"dataset_{timestamp}")

            generate_dataset(output_folder=dataset_path, n_cycle_blocks=3, fade=fade)

            print(f"✔ Created: {dataset_path}")


# ------------------------------------------------
# run
# ------------------------------------------------

if __name__ == "__main__":

    project_name = input("Project name: ")

    materials = user_input_DoE()

    generate_DoE_datasets(materials, project_name)
