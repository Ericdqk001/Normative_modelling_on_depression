from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels as sm
import torch
from models.VAE import VAE
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS
from statsmodels.discrete.discrete_model import Logit


def reconstruction_deviation(x, x_pred):
    return np.sum((x - x_pred) ** 2, axis=1) / x.shape[1]


def latent_deviation(mu_train, mu_sample, var_sample):
    var = np.var(mu_train, axis=0)
    return (
        np.sum(
            np.abs(mu_sample - np.mean(mu_train, axis=0)) / np.sqrt(var + var_sample),
            axis=1,
        )
        / mu_sample.shape[1]
    )


def separate_latent_deviation(mu_train, mu_sample, var_sample):
    var = np.var(mu_train, axis=0)
    return (mu_sample - np.mean(mu_train, axis=0)) / np.sqrt(var + var_sample)


def latent_pvalues(latent, target, type):
    pval_df = pd.DataFrame({"labels": ["const", "latent"]})

    coef_df = pd.DataFrame({"labels": ["const", "latent"]})

    for i in range(latent.shape[1]):
        latent_curr = latent[:, i]
        latent_curr = sm.tools.tools.add_constant(latent_curr)
        if type == "continuous":
            model = OLS(target, latent_curr)
        else:
            model = Logit(target, latent_curr)
        model_fit = model.fit()
        print("P VALUES")
        print(model_fit.pvalues.values)
        print("coefs")
        print(model_fit.params.values)
        print("SUMMARY")
        print(model_fit.summary())

        pval_df["latent {0}".format(i)] = list(model_fit.pvalues.values)

        coef_df["latent {0}".format(i)] = list(model_fit.params.values)

    return pval_df, coef_df


def process(
    data_path: pd.DataFrame,
    features: list,
    train_subjects: list,
    test_subjects: list,
    check_point_path: Path,
    model_config: dict,
    dx_path: Path,
    device: torch.device,
) -> pd.DataFrame:
    """This function returns an output_data dataframe which contains the latent.

    Deviation and individual latent deviation for each subject.

    Args:
        data_path (pd.DataFrame): The path to the whole data
        features (list): The features to be used in the model
        train_subjects (list): The list of subjects to be used in the training set
        test_subjects (list): The list of subjects to be used in the test set
        check_point_path (Path): The path to the trained model
        model_config (dict): The model configuration
        dx_path (Path): The path to the diagnosis data
        device (torch.device): The device to be used for the model

    Returns:
        pd.DataFrame: The output data with global latent deviation and individual
        latent deviations
    """
    data = pd.read_csv(Path(data_path), index_col=0)

    ### Within this block of code, the train/test data is scaled and placed back.
    data_to_scaled = data.copy()

    scaler = StandardScaler()

    columns_to_scale = features

    train_values = data_to_scaled.loc[train_subjects, columns_to_scale]
    scaled_train_values = scaler.fit_transform(train_values)
    data_to_scaled.loc[train_subjects, columns_to_scale] = scaled_train_values

    test_values = data_to_scaled.loc[test_subjects, columns_to_scale]
    scaled_test_values = scaler.transform(test_values)

    data_to_scaled.loc[test_subjects, columns_to_scale] = scaled_test_values

    ### End of scaling block

    train_data = data_to_scaled.loc[train_subjects, columns_to_scale].copy()

    test_data = data_to_scaled.loc[test_subjects, columns_to_scale].copy()

    input_dim = train_data.shape[1]

    print("load trained model")
    model = VAE(
        input_dim=input_dim,
        hidden_dim=model_config["hidden_dim"],
        latent_dim=model_config["latent_dim"],
        learning_rate=model_config["learning_rate"],
        non_linear=True,
    )
    model.to(device)

    model.load_state_dict(torch.load(check_point_path))

    test_latent, test_var = model.pred_latent(test_data, device)
    train_latent, _ = model.pred_latent(train_data, device)

    dx_data = pd.read_csv(Path(dx_path), index_col=0)

    ### NOTE Some test samples are not in the dx_data (n = 82), so we need to remove
    # them from the test latents.
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

    individual_deviation = separate_latent_deviation(
        train_latent, test_latent_aligned, test_var_aligned
    )
    for i in range(model_config["latent_dim"]):
        output_data["latent_deviation_{0}".format(i)] = individual_deviation[:, i]
    ###

    return output_data


def process_mVAE(
    model,
    test_pd: pd.DataFrame,
    dx_data: pd.DataFrame,
    train_data: List[torch.Tensor],
    test_data: List[torch.Tensor],
):
    output_data = test_pd.join(dx_data, how="inner")

    retained_indexes = output_data.index
    mask = test_pd.index.isin(retained_indexes)

    model.eval()

    train_latent = model.encode(train_data)

    train_latent_mu = train_latent[0].loc.detach().numpy()

    test_latent = model.encode(test_data)

    test_latent_mu, test_latent_std = (
        test_latent[0].loc.detach().numpy(),
        test_latent[0].scale.detach().numpy(),
    )

    print(test_latent_mu.shape)
    print(test_latent_std.shape)

    test_latent_mu_aligned = test_latent_mu[mask]
    test_latent_std_aligned = test_latent_std[mask]

    # Square the standard deviation to get the variance as the function expects variance
    output_data["latent_deviation"] = latent_deviation(
        train_latent_mu, test_latent_mu_aligned, test_latent_std_aligned**2
    )

    individual_deviation = separate_latent_deviation(
        train_latent_mu, test_latent_mu_aligned, test_latent_std_aligned**2
    )

    for i in range(model.z_dim):
        output_data[f"latent_deviation_{i}"] = individual_deviation[:, i]

    return output_data


def plot_latent_deviation(
    output_data,
    diagnoses,
    overall_diagnosis,
    deviation_dim="latent_deviation",
):
    # Setup the matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the figure size as needed
    fig.suptitle("")

    # To store data for boxplot
    data_to_plot = []
    labels = []

    control_data = output_data[output_data[overall_diagnosis] == "control"][
        deviation_dim
    ]

    if not control_data.empty:
        data_to_plot.append(control_data)
        labels.append("Control")

    for diagnosis in diagnoses:
        # Filter output_data for rows where diagnosis is True
        output_data[diagnosis] = output_data[diagnosis].replace(True, 1)
        filtered_data = output_data[output_data[diagnosis] == 1][deviation_dim]
        if not filtered_data.empty:
            data_to_plot.append(filtered_data)
            labels.append(diagnosis)

    # Create the box plot
    ax.boxplot(data_to_plot, labels=labels, notch=True, patch_artist=True)

    ax.set_ylabel(deviation_dim, fontweight="bold")
    ax.set_xlabel("Diagnosis", fontweight="bold")

    # Set axis and tick labels to bold
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


def get_latent_deviation_pvalues(
    latent_deviations,
    output_data,
    diagnosis,
):
    DX_pval, DX_coef = latent_pvalues(
        latent_deviations, output_data[diagnosis], type="discrete"
    )

    return DX_pval, DX_coef


def filter_controls(output_data, diagnosis):
    """Filter out False values which do not represent controls.

    This function is run before computing p values against controls for each diagnosis.
    """
    output_data = output_data.copy()
    mask = (output_data[diagnosis] == False) & (output_data["psych_dx"] != "control")

    # Apply the mask to the DataFrame to filter rows
    filtered_data = output_data[~mask]

    return filtered_data


# def compute_deviations_pvalues(output_data, diagnoses):
#     dx_global_pvalues = {}

#     for diagnosis in diagnoses:
#         print("global")
#         print(diagnosis)

#         filtered_controls = filter_controls(output_data, diagnosis)

#         pvalues = get_latent_deviation_pvalues(
#             output_data[["latent_deviation"]].to_numpy(), output_data, diagnosis
#         )

#         dx_global_pvalues[diagnosis] = pvalues.iloc[1][1]

#     dx_individual_pvalues = {}

#     for diagnosis in diagnoses:
#         print("individual")

#         print(diagnosis)

#         dim_pvalues = {}

#         for i in range(mvae.z_dim):
#             individual_deviation = get_latent_deviation_pvalues(
#                 output_data[[f"latent_deviation_{i}"]].to_numpy(),
#                 output_data,
#                 diagnosis,
#             )

#             dim_pvalues[f"latent_dim_{i}"] = individual_deviation.iloc[1][1]

#             print(dim_pvalues[f"latent_dim_{i}"])

#         dx_individual_pvalues[diagnosis] = dim_pvalues

#     print(dx_global_pvalues)

#     print(dx_individual_pvalues)

#     from statsmodels.stats.multitest import multipletests

#     # Flatten your dictionary of p-values into a single list
#     p_values = [
#         value
#         for condition in dx_individual_pvalues.values()
#         for value in condition.values()
#     ]

#     # Perform FDR correction
#     _, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

#     # sum the number of significant p-values

#     print(sum(p_values_corrected < 0.05))


# def latent_deviation(cohort, holdout):
#     if cohort.ndim == 1:
#         print("hello")
#         latent_dim = 1
#         mean_holdout = np.mean(holdout)
#         sd_holdout = np.std(holdout)
#         z_scores = (cohort - mean_holdout) / sd_holdout
#     else:
#         latent_dim = cohort.shape[1]
#         mean_holdout = np.mean(holdout, axis=0)
#         sd_holdout = np.std(holdout, axis=0)
#         z_scores = (
#             np.sum(np.abs(cohort - mean_holdout) / sd_holdout, axis=1) / latent_dim
#         )
#     return z_scores


# def recon_deviation(cohort, recon):
#     feat_dim = cohort.shape[1]
#     dev = np.sum(np.sqrt((cohort - recon) ** 2), axis=1) / feat_dim
#     return dev


# def latent_deviations_mahalanobis_across_sig(cohort, train):
#     latent_dim = cohort.shape[1]
#     dists = calc_robust_mahalanobis_distance(cohort, train)
#     pvals = 1 - chi2.cdf(dists, latent_dim - 1)
#     return dists, pvals


# def calc_robust_mahalanobis_distance(values, train_values):
#     robust_cov = MinCovDet(random_state=42).fit(train_values)
#     mahal_robust_cov = robust_cov.mahalanobis(values)
#     return mahal_robust_cov
