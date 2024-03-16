from pathlib import Path

import pandas as pd
from nilearn import datasets

# %%
### To get the roi names
destrieux_2009 = datasets.fetch_atlas_destrieux_2009(legacy_format=False)

labels = destrieux_2009.labels

### NOTE index 42 and 116 are not present in the ABCD dextrieux atlas so are removed
labels_dropped = labels.drop(index=42).reset_index(drop=True)
labels_dropped = labels_dropped.drop(index=116).reset_index(drop=True)

read_file_path = Path(
    "src/interpret/files/relevant_ct_features.txt",
)

with open(read_file_path, "r") as file:
    relevant_ct_features = [line.strip() for line in file.readlines()]

roi_indices = [int(feature_name.split("_")[1]) for feature_name in relevant_ct_features]

relevant_roi_labels = labels_dropped.iloc[roi_indices]["name"].tolist()
# %%
# Check if the roi labels matches with ABCD variable description

ct_var_names_path = Path("processed_data/ct_features_var_names.csv")

ct_var_names = pd.read_csv(ct_var_names_path)

matching_var_names = ct_var_names[ct_var_names["var_name"].isin(relevant_ct_features)]

# Extracting the 'var_label' values for these matching rows
matching_var_labels = matching_var_names["var_label"].tolist()


relevant_ct_df = pd.DataFrame(
    {
        "relevant_ct_features": relevant_ct_features,
        "relevant_roi_labels": relevant_roi_labels,
        "matching_var_labels": matching_var_labels,
    }
)

relevant_ct_df.to_csv(
    Path("src/interpret/files/relevant_ct_features_with_labels.csv"), index=False
)

# %%


# # Fetch Destrieux atlas
# destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
# parcellation_left = destrieux_atlas["map_left"]
# parcellation_right = destrieux_atlas["map_right"]
# labels = destrieux_atlas["labels"]

# # Find the label indices for the PCC
# pcc_left_label = labels.index(b"G_cingul-Post-dorsal")
# pcc_right_label = labels.index(b"G_cingul-Post-ventral")

# # Initialize the ROI maps with zeros
# roi_map_left = np.zeros_like(parcellation_left)
# roi_map_right = np.zeros_like(parcellation_right)

# # Assign a unique value to the ROIs in the parcellation maps
# roi_map_left[parcellation_left == pcc_left_label] = 1  # Highlight left PCC
# roi_map_right[parcellation_right == pcc_right_label] = 1  # Highlight right PCC
