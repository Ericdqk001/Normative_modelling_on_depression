import json
import pickle
from pathlib import Path

import torch
from discover.utils import (
    filter_controls,
    get_latent_deviation_pvalues,
    plot_latent_deviation,
    process,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = Path(
    "ABCD_mVAE_LizaEric/data/rsfmri_gordon_no_dup_postCombat_residSexAge_060623.csv"
)

DX_PATH = Path("ABCD_mVAE_LizaEric/data/all_psych_dx_r5.csv")

with open(Path("ABCD_mVAE_LizaEric/data/train_val_subs.pkl"), "rb") as f:
    TRAIN_VAL_SUBS = pickle.load(f)

with open(Path("ABCD_mVAE_LizaEric/data/subs_test.pkl"), "rb") as f:
    TEST_SUBS = pickle.load(f)


with open(Path("ABCD_mVAE_LizaEric/data", "phenotype_roi_mapping_with_fmri.json")) as f:
    phenotype_roi_mapping = json.loads(f.read())

check_point_path = Path("artifacts/bright-donkey-3:v0/model_weights.pt")

model_config = {
    "hidden_dim": [45, 45],
    "latent_dim": 15,
    "learning_rate": 0.001,
}

diagnoses = [
    "Has_ADHD",
    "Has_Depression",
    "Has_Bipolar",
    "Has_Anxiety",
    "Has_OCD",
    "Has_ASD",
    "Has_DBD",
]

overall_diagnosis = "psych_dx"


if __name__ == "__main__":
    output_data = process(
        data_path=DATA_PATH,
        features=phenotype_roi_mapping["rsfmri_gordon_no_dup"],
        train_subjects=TRAIN_VAL_SUBS,
        test_subjects=TEST_SUBS,
        check_point_path=check_point_path,
        model_config=model_config,
        dx_path=DX_PATH,
        device=DEVICE,
    )

    plot_latent_deviation(
        output_data.copy(), diagnoses=diagnoses, overall_diagnosis=overall_diagnosis
    )

    dx_global_pvalues_coef = {}

    for diagnosis in diagnoses:
        print("global")
        print(diagnosis)

        filtered_controls = filter_controls(output_data, diagnosis)

        pvalues, coef = get_latent_deviation_pvalues(
            filtered_controls[["latent_deviation"]].to_numpy(),
            filtered_controls,
            diagnosis,
        )

        print(pvalues)

        print(coef)

        dx_global_pvalues_coef[diagnosis] = {
            "p_value": pvalues.iloc[1][1],
            "coef": coef.iloc[1][1],
        }

    dx_individual_pvalues_coef = {}

    for diagnosis in diagnoses:
        print("individual")

        print(diagnosis)

        dim_pvalues_coef = {}

        filtered_controls = filter_controls(output_data, diagnosis)

        for i in range(model_config["latent_dim"]):
            p_value, coef = get_latent_deviation_pvalues(
                filtered_controls[[f"latent_deviation_{i}"]].to_numpy(),
                filtered_controls,
                diagnosis,
            )

            dim_pvalues_coef[f"latent_dim_{i}_p_value"] = p_value.iloc[1][1]

            dim_pvalues_coef[f"latent_dim_{i}_coef"] = coef.iloc[1][1]

        dx_individual_pvalues_coef[diagnosis] = dim_pvalues_coef

    print(dx_global_pvalues_coef)

    print(dx_individual_pvalues_coef)
