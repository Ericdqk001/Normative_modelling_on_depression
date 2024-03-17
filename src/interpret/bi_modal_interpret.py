# 'latent_dim_11': 0.016972343836183935,
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from multiviewae import mVAE
from multiviewae.base.distributions import Normal

with open(Path("ABCD_mVAE_LizaEric/data/phenotype_roi_mapping_with_fmri.json")) as f:
    phenotype_roi_mapping = json.loads(f.read())

var_names_path = Path("processed_data/ct_features_var_names.csv")

var_names = pd.read_csv(var_names_path, header=None)

var_names.columns = var_names.iloc[0]

check_point_path = Path(
    "results/two_views/mvae_ct and rsfmri_two_Views_param_comb15_010624/model.ckpt"
)
mvae = mVAE.load_from_checkpoint(check_point_path)

mvae._training = False

loc_tensor = torch.zeros(1, 15)
loc_tensor[0, 11] = -5
logvar_tensor = torch.zeros(1, 15)

# Initialise an object of the class with these tensors
z = Normal(loc=loc_tensor, logvar=logvar_tensor)

z = mvae.decode([z])

reconstruction = [
    (
        [d__._sample().cpu().detach().numpy() for d__ in d_]
        if isinstance(d_, (list))
        else (
            d_.cpu().detach().numpy()
            if isinstance(d_, torch.Tensor)
            else d_._sample().cpu().detach().numpy()
        )
    )
    for d_ in z
]

ct_features = phenotype_roi_mapping["ct"]

fc_features = phenotype_roi_mapping["rsfmri_gordon_no_dup"]

reconstruction[0][0].shape
reconstruction[0][1].shape

reconstructed_ct = reconstruction[0][0]
reconstructed_fc = reconstruction[0][1]

reconstructed_ct_flat = reconstructed_ct.flatten()

reconstructed_fc_flat = reconstructed_fc.flatten()

# Find the indexes of the values that are outside the range of -1 to 1
indexes_outside_range = np.where(
    (reconstructed_ct_flat > 0.5) | (reconstructed_ct_flat < -1)
)[0]

relevant_ct_features = [ct_features[i] for i in indexes_outside_range]

ct_features_path = Path("src/interpret/files/relevant_ct_features.txt")

with open(ct_features_path, "w") as file:
    for feature in relevant_ct_features:
        file.write("%s\n" % feature)

fc_features_weight = {}

for i, feature in enumerate(fc_features):
    fc_features_weight[feature] = float(reconstructed_fc_flat[i])

fc_features_weight_path = Path("src/interpret/files/fc_features_weight.json")

with open(fc_features_weight_path, "w") as f:
    f.write(json.dumps(fc_features_weight))

# # Plotting the distribution
# plt.figure(figsize=(10, 6))
# plt.hist(reconstructed_ct_flat, bins=30, alpha=0.75, color="blue", edgecolor="black")
# plt.title("Distribution of Cortical Thickness Measurements")
# plt.xlabel("Cortical Thickness Value")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.show()

# # Plotting the distribution
# plt.figure(figsize=(10, 6))
# plt.hist(reconstructed_fc_flat, bins=30, alpha=0.75, color="blue", edgecolor="black")
# plt.title("Distribution of Cortical Thickness Measurements")
# plt.xlabel("Cortical Thickness Value")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.show()
