# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import datasets, plotting
from PIL import Image

# Fetch surface atlas
fsaverage = datasets.fetch_surf_fsaverage()

# Fetch Destrieux atlas
destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
parcellation_left = destrieux_atlas["map_left"]
parcellation_right = destrieux_atlas["map_right"]
labels = destrieux_atlas["labels"]


# Load the relevant features from locate_roi.py

relevant_roi_path = Path("src/interpret/files/relevant_ct_features_with_labels.csv")

relevant_roi_df = pd.read_csv(relevant_roi_path)

whole_brain_roi_labels = relevant_roi_df["relevant_roi_labels"].tolist()

# Splitting into left and right ROI labels
left_roi_labels = [name[2:] for name in whole_brain_roi_labels if name.startswith("L ")]
right_roi_labels = [
    name[2:] for name in whole_brain_roi_labels if name.startswith("R ")
]

# Assuming `labels` is a list of byte strings as indicated by the original code
pcc_left_labels = [labels.index(label.encode("utf-8")) for label in left_roi_labels]
pcc_right_labels = [labels.index(label.encode("utf-8")) for label in right_roi_labels]

pcc_left_mask = np.isin(parcellation_left, pcc_left_labels)

# Create masks for multiple regions of interest in the right hemisphere
pcc_right_mask = np.isin(parcellation_right, pcc_right_labels)
# Load the fsaverage5 pial surface for left and right hemispheres
fsaverage_pial_left = fsaverage["pial_left"]
fsaverage_pial_right = fsaverage["pial_right"]


image_save_path = Path("src/interpret/plots")

# Plotting the left hemisphere, lateral view
plotting.plot_surf_roi(
    fsaverage_pial_left,
    roi_map=pcc_left_mask,
    hemi="left",
    view="lateral",
    bg_map=fsaverage["sulc_left"],
    bg_on_data=True,
    title="Left Hemisphere Lateral",
    colorbar=False,
    cmap="gist_rainbow",
)
# .savefig(Path(image_save_path, "left_lateral.png"))

# Plotting the left hemisphere, medial view
plotting.plot_surf_roi(
    fsaverage_pial_left,
    roi_map=pcc_left_mask,
    hemi="left",
    view="medial",
    bg_map=fsaverage["sulc_left"],
    bg_on_data=True,
    title="Left Hemisphere Medial",
    colorbar=False,
    cmap="gist_rainbow",
)
# .savefig(Path(image_save_path, "left_medial.png"))

# Plotting the right hemisphere, lateral view
plotting.plot_surf_roi(
    fsaverage_pial_right,
    roi_map=pcc_right_mask,
    hemi="right",
    view="lateral",
    bg_map=fsaverage["sulc_right"],
    bg_on_data=True,
    title="Right Hemisphere Lateral",
    colorbar=False,
    cmap="gist_rainbow",
)
# .savefig(Path(image_save_path, "right_lateral.png"))

# Plotting the right hemisphere, medial view
plotting.plot_surf_roi(
    fsaverage_pial_right,
    roi_map=pcc_right_mask,
    hemi="right",
    view="medial",
    bg_map=fsaverage["sulc_right"],
    bg_on_data=True,
    title="Right Hemisphere Medial",
    colorbar=False,
    cmap="gist_rainbow",
)
# .savefig(Path(image_save_path, "right_medial.png"))

plotting.show()


# %%
# Plot the images together

image_paths = [
    Path("src/interpret/plots/left_lateral.png"),
    Path("src/interpret/plots/left_medial.png"),
    Path("src/interpret/plots/right_lateral.png"),
    Path("src/interpret/plots/right_medial.png"),
]

sorted_image_paths = sorted(image_paths, key=lambda x: x.stem.split("_")[1])

# Create figure with subplots
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(24, 6))

# Loop over axes and image paths
for ax, image_path in zip(axes, image_paths):
    img = Image.open(image_path)
    ax.imshow(img)
    ax.axis("off")

# Adjust layout
plt.subplots_adjust(
    left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.01
)
plt.tight_layout(pad=0.1)

# Adjust layout to ensure the plots are nicely spaced
plt.tight_layout()
plt.show()
