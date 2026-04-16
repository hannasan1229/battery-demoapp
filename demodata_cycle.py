import numpy as np
import pandas as pd
from datetime import datetime

capacity_nom = 1.0
R_internal = 0.02

dt = 60

SOC_start = 0.20
SOC_min = 0.05
SOC_max = 0.95

capacity_fade_per_cycle = 0.01


def ocv(soc):
    soc = np.clip(soc, 0, 1)
    return 3.0 + 0.9 * soc


def generate_dataset(n_cycles=10):

    rows = []
    current_time = datetime.now()

    soc = SOC_start
    capacity = capacity_nom
    Q = soc * capacity

    for cycle in range(n_cycles):

        # charge
        while soc < SOC_max:
            Q += 0.01
            soc = Q / capacity

            rows.append({
                "timestamp": current_time,
                "test_type": "cycle",
                "cycle": cycle,
                "SOC": soc,
                "Q_Ah": Q,
                "current_A": 1.0,
                "voltage_V": ocv(soc)
            })

        # discharge
        while soc > SOC_min:
            Q -= 0.01
            soc = Q / capacity

            rows.append({
                "timestamp": current_time,
                "test_type": "cycle",
                "cycle": cycle,
                "SOC": soc,
                "Q_Ah": Q,
                "current_A": -1.0,
                "voltage_V": ocv(soc)
            })

    return pd.DataFrame(rows)


def generate_varM_dataframes(materials):

    varM = {}

    for mat, props in materials.items():

        varM[mat] = []

        for i in range(props["n_cells"]):
            df = generate_dataset()
            varM[mat].append(df)

    return varM
