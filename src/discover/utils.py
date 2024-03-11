import numpy as np
import pandas as pd
import statsmodels as sm
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
    for i in range(latent.shape[1]):
        latent_curr = latent[:, i]
        latent_curr = sm.tools.tools.add_constant(latent_curr)
        if type == "continuous":
            model = OLS(target, latent_curr)
        else:
            model = Logit(target, latent_curr)
        model_fit = model.fit()
        pval_df["latent {0}".format(i)] = list(model_fit.pvalues.values)
    return pval_df


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
