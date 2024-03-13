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

DATA_PATH = "ABCD_mVAE_LizaEric/data/rsfmri_gordon_postCombat_residSexAge_060623.csv"

DX_PATH = "ABCD_mVAE_LizaEric/data/all_psych_dx_r5.csv"

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
    # "Has_ADHD",
    "Has_Depression",
    "Has_Bipolar",
    "Has_Anxiety",
    # "Has_OCD",
    # "Has_ASD",
    # "Has_DBD",
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

    dx_global_pvalues = {}

    for diagnosis in diagnoses:
        print("global")
        print(diagnosis)

        filtered_controls = filter_controls(output_data, diagnosis)

        pvalues = get_latent_deviation_pvalues(
            output_data[["latent_deviation"]].to_numpy(), output_data, diagnosis
        )

        dx_global_pvalues[diagnosis] = pvalues.iloc[1][1]

    dx_individual_pvalues = {}

    for diagnosis in diagnoses:
        print("individual")

        print(diagnosis)

        dim_pvalues = {}

        for i in range(model_config["latent_dim"]):
            individual_deviation = get_latent_deviation_pvalues(
                output_data[[f"latent_deviation_{i}"]].to_numpy(),
                output_data,
                diagnosis,
            )

            dim_pvalues[f"latent_dim_{i}"] = individual_deviation.iloc[1][1]

            print(dim_pvalues[f"latent_dim_{i}"])

        dx_individual_pvalues[diagnosis] = dim_pvalues

    print(dx_global_pvalues)

    print(dx_individual_pvalues)

    from statsmodels.stats.multitest import multipletests

    # Flatten your dictionary of p-values into a single list
    p_values = [
        value
        for condition in dx_individual_pvalues.values()
        for value in condition.values()
    ]

    # Perform FDR correction
    _, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

    # sum the number of significant p-values

    print(sum(p_values_corrected < 0.05))
