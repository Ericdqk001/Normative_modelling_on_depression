import json
import pickle
from pathlib import Path

import pandas as pd
import torch
from discover.utils import (
    filter_controls,
    get_latent_deviation_pvalues,
    plot_latent_deviation,
    process_mVAE,
)
from multiviewae import mVAE
from sklearn.preprocessing import StandardScaler

CT_DATA_PATH = Path("ABCD_mVAE_LizaEric/data/ct_postCombat_residSexAge_060623.csv")

FC_DATA_PATH = Path(
    "ABCD_mVAE_LizaEric/data/rsfmri_gordon_no_dup_postCombat_residSexAge_060623.csv"
)

DX_PATH = Path("ABCD_mVAE_LizaEric/data/all_psych_dx_r5.csv")

dx_data = pd.read_csv(Path(DX_PATH), index_col=0)

with open(Path("ABCD_mVAE_LizaEric/data/train_val_subs.pkl"), "rb") as f:
    TRAIN_VAL_SUBS = pickle.load(f)

with open(Path("ABCD_mVAE_LizaEric/data/subs_test.pkl"), "rb") as f:
    TEST_SUBS = pickle.load(f)

with open(Path("ABCD_mVAE_LizaEric/data/phenotype_roi_mapping_with_fmri.json")) as f:
    phenotype_roi_mapping = json.loads(f.read())

check_point_path = Path(
    "results/two_views/mvae_ct and rsfmri_two_Views_param_comb15_010624/model.ckpt"
)


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

ct_data = pd.read_csv(Path(CT_DATA_PATH), index_col=0)

fc_data = pd.read_csv(Path(FC_DATA_PATH), index_col=0)

ct_features = phenotype_roi_mapping["ct"]

fc_features = phenotype_roi_mapping["rsfmri_gordon_no_dup"]

ct_train_pd = ct_data.loc[TRAIN_VAL_SUBS, ct_features]

fc_train_pd = fc_data.loc[TRAIN_VAL_SUBS, fc_features]

ct_test_pd = ct_data.loc[TEST_SUBS, ct_features]

fc_test_pd = fc_data.loc[TEST_SUBS, fc_features]

ct_scaler = StandardScaler()

fc_scaler = StandardScaler()

ct_train_values = ct_scaler.fit_transform(ct_train_pd)

fc_train_values = fc_scaler.fit_transform(fc_train_pd)

ct_test_values = ct_scaler.transform(ct_test_pd)

fc_test_values = fc_scaler.transform(fc_test_pd)


ct_train_data = torch.tensor(
    ct_train_values,
    dtype=torch.float32,
)

fc_train_data = torch.tensor(
    fc_train_values,
    dtype=torch.float32,
)

ct_test_data = torch.tensor(
    ct_test_values,
    dtype=torch.float32,
)

fc_test_data = torch.tensor(
    fc_test_values,
    dtype=torch.float32,
)

test_data = [ct_test_data, fc_test_data]

train_data = [ct_train_data, fc_train_data]


mvae = mVAE.load_from_checkpoint(check_point_path)


if __name__ == "__main__":
    output_data = process_mVAE(
        model=mvae,
        test_pd=fc_test_pd,
        dx_data=dx_data,
        train_data=train_data,
        test_data=test_data,
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

        for i in range(mvae.z_dim):
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

    from statsmodels.stats.multitest import multipletests

    # Flatten your dictionary of p-values into a single list
    p_values = [
        value
        for condition in dx_individual_pvalues_coef.values()
        for value in condition.values()
    ]

    # Perform FDR correction
    _, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

    # sum the number of significant p-values

    print(sum(p_values_corrected < 0.05))
