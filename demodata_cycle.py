import numpy as np
import pandas as pd
from datetime import datetime
import os

# ------------------------------------------------
# global parameters
# ------------------------------------------------

capacity_nom = 1.0
R_internal = 0.02

dt = 20

charge_rate_C = 1.0
discharge_rate_C = 1.0

rest_steps = 10

SOC_start = 0.20
SOC_min = 0.05
SOC_max = 0.95

capacity_fade_per_cycle = 0.01


# ------------------------------------------------
# OCV model
# ------------------------------------------------

def ocv(soc):

    soc = np.clip(soc, 0, 1)

    V = 3.0 + 0.9 * soc

    # Phase transitions
    V += 0.12 * np.exp(-((soc - 0.25) / 0.04) ** 2)
    V += 0.10 * np.exp(-((soc - 0.50) / 0.05) ** 2)
    V += 0.08 * np.exp(-((soc - 0.75) / 0.04) ** 2)

    # High voltage region
    V += 0.25 / (1 + np.exp(-(soc - 0.9) * 40))

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
    n_cycles=10
):

    global current_time

    rows = []
    temperature = 25

    for cycle in range(n_cycles):

        I_charge = capacity * charge_rate_C
        I_discharge = -capacity * discharge_rate_C

        # ---------------- charge ----------------
        while soc < SOC_max - 1e-6:

            Q += I_charge * dt / 3600
            soc = np.clip(Q / capacity, 0, 1)

            noise = np.random.normal(0, 0.002)

            V = ocv(soc) + I_charge * R_internal + noise

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

    while soc > SOC_min + 1e-6:

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
    n_cycle_blocks=3,
    n_cycles=10,
    output_folder=None,
    fade=capacity_fade_per_cycle
):

    dfs = []

    soc = SOC_start
    capacity = capacity_nom
    Q = soc * capacity

    # initial capacity check
    df_cap0, soc, Q = generate_capacity_check(
        soc,
        Q,
        capacity
    )

    dfs.append(df_cap0)

    for block in range(n_cycle_blocks):

        # cycle block
        df_block, soc, Q, capacity = generate_cycle_block(
            soc,
            Q,
            capacity,
            block,
            fade,
            n_cycles=n_cycles
        )

        dfs.append(df_block)

        # capacity check
        df_cap, soc, Q = generate_capacity_check(
            soc,
            Q,
            capacity
        )

        dfs.append(df_cap)

    final_df = pd.concat(dfs, ignore_index=True)

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
    fade=capacity_fade_per_cycle
):

    global current_time

    # 🔥 WICHTIG:
    # jedes Dataset startet neu
    current_time = datetime.now()

    return combine_dataframe(
        n_cycle_blocks=n_cycle_blocks,
        n_cycles=n_cycles,
        output_folder=output_folder,
        fade=fade,
    )


# ------------------------------------------------
# main varM generator
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
