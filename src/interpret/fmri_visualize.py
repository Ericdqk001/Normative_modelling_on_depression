import json
from pathlib import Path

import matplotlib.pyplot as plt
from nichord.chord import plot_chord

with open(Path("ABCD_mVAE_LizaEric/data", "phenotype_roi_mapping_with_fmri.json")) as f:
    phenotype_roi_mapping = json.loads(f.read())

fc_features = phenotype_roi_mapping["rsfmri_gordon_no_dup"]

fc_weights_path = Path("src/interpret/files/fc_features_weight.json")

with open(fc_weights_path) as f:
    fc_weights = json.loads(f.read())

weights = list(fc_weights.values())

# MinMax normalisation

weights = [(i - min(weights)) / (max(weights) - min(weights)) for i in weights]

# Thresholding at 0.5

# weights = [i if i > 0.5 else 0 for i in weights]

idx_to_label = {
    0: "AUD",  # auditory network
    1: "CON",  # cingulo-opercular network
    2: "CPN",  # cingulo-parietal network
    3: "DMN",  # default mode network
    4: "DAN",  # dorsal attention network
    5: "FPN",  # fronto-parietal network
    6: "NN",  # none network
    7: "RTN",  # retrosplenial-temporal network
    8: "SN",  # salience network
    9: "SHN",  # somatomotor hand network
    10: "SMN",  # somatomotor mouth network
    11: "VAN",  # ventral attention network
    12: "VIS",  # visual network
}

# Edges without duplicates

edges = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 9),
    (0, 10),
    (0, 11),
    (0, 12),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (1, 9),
    (1, 10),
    (1, 11),
    (1, 12),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 10),
    (2, 11),
    (2, 12),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    (3, 8),
    (3, 9),
    (3, 10),
    (3, 11),
    (3, 12),
    (4, 4),
    (4, 5),
    (4, 6),
    (4, 7),
    (4, 8),
    (4, 9),
    (4, 10),
    (4, 11),
    (4, 12),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 8),
    (5, 9),
    (5, 10),
    (5, 11),
    (5, 12),
    (6, 5),
    (6, 6),
    (6, 7),
    (6, 8),
    (6, 9),
    (6, 10),
    (6, 11),
    (6, 12),
    (7, 7),
    (7, 8),
    (7, 9),
    (7, 10),
    (7, 11),
    (7, 12),
    (8, 8),
    (8, 9),
    (8, 10),
    (8, 11),
    (8, 12),
    (9, 9),
    (9, 10),
    (9, 11),
    (9, 12),
    (10, 10),
    (10, 11),
    (10, 12),
    (11, 11),
    (11, 12),
    (12, 11),
    (12, 12),
]

# fp_chord = 'ex0_chord.png'
plot_chord(
    idx_to_label,
    edges,
    edge_weights=weights,
    fp_chord=None,
    linewidths=5,
    alphas=0.9,
    do_ROI_circles=True,
    label_fontsize=70,
    # July 2023 update allows changing label fontsize
    do_ROI_circles_specific=True,
    ROI_circle_radius=0.02,
)
plt.show()
