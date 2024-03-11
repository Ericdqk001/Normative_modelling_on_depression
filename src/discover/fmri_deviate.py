import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from discover.utils import (
    latent_deviation,
    separate_latent_deviation,
)
from models.VAE import VAE
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "ABCD_mVAE_LizaEric/data/rsfmri_gordon_postCombat_residSexAge_060623.csv"

DX_PATH = "ABCD_mVAE_LizaEric/data/all_psych_dx_r5.csv"

with open(Path("ABCD_mVAE_LizaEric/data/train_val_subs.pkl"), "rb") as f:
    TRAIN_VAL_SUBS = pickle.load(f)

with open(Path("ABCD_mVAE_LizaEric/data/subs_test.pkl"), "rb") as f:
    TEST_SUBS = pickle.load(f)


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


def process():
    data = pd.read_csv(Path(DATA_PATH), index_col=0)

    ### Within this block of code, the train/test data is independently scaled and placed back to the original dataframe
    data_to_scaled = data.copy()

    scaler = StandardScaler()

    columns_to_scale = data.columns[0:93]

    train_values = data_to_scaled.loc[TRAIN_VAL_SUBS, columns_to_scale]
    scaled_train_values = scaler.fit_transform(train_values)
    data_to_scaled.loc[TRAIN_VAL_SUBS, columns_to_scale] = scaled_train_values

    test_values = data_to_scaled.loc[TEST_SUBS, columns_to_scale]
    scaled_test_values = scaler.transform(test_values)

    data_to_scaled.loc[TEST_SUBS, columns_to_scale] = scaled_test_values

    ### End of scaling block

    train_data = data_to_scaled.loc[TRAIN_VAL_SUBS, columns_to_scale].copy()

    test_data = data_to_scaled.loc[TEST_SUBS, columns_to_scale].copy()

    input_dim = train_data.shape[1]

    print("load trained model")
    model = VAE(
        input_dim=input_dim,
        hidden_dim=model_config["hidden_dim"],
        latent_dim=model_config["latent_dim"],
        learning_rate=model_config["learning_rate"],
        non_linear=True,
    )
    model.to(DEVICE)

    model.load_state_dict(
        torch.load(
            check_point_path,
            map_location=DEVICE,
        ),
    )

    test_latent, test_var = model.pred_latent(test_data, DEVICE)
    train_latent, _ = model.pred_latent(train_data, DEVICE)

    dx_data = pd.read_csv(Path(DX_PATH), index_col=0)

    ### Some test samples are not in the dx_data (n = 82), so we need to remove them from the test latents.
    output_data = test_data.join(dx_data, how="inner")

    retained_indexes = output_data.index
    mask = test_data.index.isin(retained_indexes)

    test_latent_aligned = test_latent[mask]
    test_var_aligned = test_var[mask]
    ###

    ### Calculate the deviation and add it to the output data
    output_data["latent_deviation"] = latent_deviation(
        train_latent, test_latent_aligned, test_var_aligned
    )

    deviation = separate_latent_deviation(
        train_latent, test_latent_aligned, test_var_aligned
    )
    for i in range(model_config["latent_dim"]):
        output_data["latent_deviation_{0}".format(i)] = deviation[:, i]
    ###

    return output_data


def plot_latent_deviation(
    output_data,
    deviation_dim="latent_deviation",
):
    # Setup the matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the figure size as needed
    fig.suptitle("Distribution of Latent Deviation by Diagnosis")

    # To store data for boxplot
    data_to_plot = []
    labels = []

    control_data = output_data[output_data[overall_diagnosis] == "control"][
        deviation_dim
    ]

    print(len(control_data))

    if not control_data.empty:
        data_to_plot.append(control_data)
        labels.append("Control")

    for diagnosis in diagnoses:
        # Filter output_data for rows where diagnosis is True
        output_data[diagnosis] = output_data[diagnosis].replace(True, 1)
        filtered_data = output_data[output_data[diagnosis] == 1][deviation_dim]
        print(len(filtered_data))
        if not filtered_data.empty:
            data_to_plot.append(filtered_data)
            labels.append(diagnosis)

    # Create the box plot
    ax.boxplot(data_to_plot, labels=labels, notch=True, patch_artist=True)

    # Improve the visualization
    ax.set_ylabel(deviation_dim)
    ax.set_xlabel("Diagnosis")
    plt.xticks(rotation=45)  # Rotate labels to improve readability
    plt.grid(True)  # Add a grid for better readability
    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )  # Adjust layout to make room for the suptitle and x-labels

    # Show the plot
    plt.show()


if __name__ == "__main__":
    output_data = process()

    plot_latent_deviation(output_data)

    # for i in range(model_config["latent_dim"]):
    #     plot_latent_deviation(output_data, deviation_dim=f"latent_deviation_{i}")
