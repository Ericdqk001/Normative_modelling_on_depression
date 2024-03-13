import json
import pickle
from pathlib import Path

import pandas as pd
import torch
from discover.utils import process_mvae
from multiviewae import mVAE
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

CT_DATA_PATH = Path("processed_data/ct_postCombat_residSexAge_060623.csv")

FC_DATA_PATH = Path(
    "ABCD_mVAE_LizaEric/data/rsfmri_gordon_postCombat_residSexAge_060623.csv"
)

DX_PATH = "ABCD_mVAE_LizaEric/data/all_psych_dx_r5.csv"

with open(Path("ABCD_mVAE_LizaEric/data/train_val_subs.pkl"), "rb") as f:
    TRAIN_VAL_SUBS = pickle.load(f)

with open(Path("ABCD_mVAE_LizaEric/data/subs_test.pkl"), "rb") as f:
    TEST_SUBS = pickle.load(f)

with open(Path("ABCD_mVAE_LizaEric/data/phenotype_roi_mapping_with_fmri.json")) as f:
    phenotype_roi_mapping = json.loads(f.read())

check_point_path = Path(
    "results/two_views/mvae_ct and rsfmri_two_Views_param_comb7_010624/model.ckpt"
)

model_config = {
    "batch_size": 64,
    "hidden_dim": [45, 45],
    "latent_dim": 15,
    "learning_rate": 0.001,
    "epochs": round((251 + 260 + 224 + 260 + 250) / 5),
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

ct_data = pd.read_csv(Path(CT_DATA_PATH), index_col=0)

fc_data = pd.read_csv(Path(FC_DATA_PATH), index_col=0)

ct_features = phenotype_roi_mapping["ct"]

fc_features = phenotype_roi_mapping["rsfmri_gordon_no_dup"]

scaler = StandardScaler()

ct_train_values = scaler.fit_transform(
    ct_data.loc[
        TRAIN_VAL_SUBS,
        ct_features,
    ]
)

fc_train_values = scaler.fit_transform(
    fc_data.loc[
        TRAIN_VAL_SUBS,
        fc_features,
    ]
)

ct_test_values = scaler.fit_transform(
    ct_data.loc[
        TEST_SUBS,
        ct_features,
    ]
)

fc_test_values = scaler.fit_transform(
    fc_data.loc[
        TEST_SUBS,
        fc_features,
    ]
)

ct_test_data = torch.tensor(
    ct_test_values,
    dtype=torch.float32,
)

fc_test_data = torch.tensor(
    fc_test_values,
    dtype=torch.float32,
)

ct_train_data = torch.tensor(
    ct_train_values,
    dtype=torch.float32,
)

fc_train_data = torch.tensor(
    fc_train_values,
    dtype=torch.float32,
)

test_data = [ct_test_data, fc_test_data]

train_data = [ct_train_data, fc_train_data]


mvae = mVAE.load_from_checkpoint(check_point_path)

train_latent = mvae.encode(train_data)

train_latent_mu = train_latent[0].loc.detach().numpy()

mvae.eval()

mvae_latent = mvae.encode(test_data)

test_latent_mu, test_latent_std = (
    mvae_latent[0].loc.detach().numpy(),
    mvae_latent[0].scale.detach().numpy(),
)

if __name__ == "__main__":
    output_data = process_mvae(
        mvae,
        train_data,
        test_data,
        DX_PATH,
    )

# TODO Remove indexes that does not exist in the dx_data here, before using the process
# function
