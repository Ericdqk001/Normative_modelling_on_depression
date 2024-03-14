from pathlib import Path

import pandas as pd

DX_PATH = Path("ABCD_mVAE_LizaEric/data/all_psych_dx_r5.csv")

dx_data = pd.read_csv(Path(DX_PATH), index_col=0)

diagnoses = [
    "Has_ADHD",
    "Has_Depression",
    "Has_Bipolar",
    "Has_Anxiety",
    "Has_OCD",
    "Has_ASD",
    "Has_DBD",
]

dx_data["Total_Diagnoses"] = dx_data[diagnoses].sum(axis=1)
