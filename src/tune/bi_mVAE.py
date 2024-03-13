# %%
import collections
import itertools
import json
import os
import pickle
from pathlib import Path

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from multiviewae import mVAE
from omegaconf import OmegaConf, open_dict
from sklearn.preprocessing import StandardScaler


def create_folder(dir_path):
    check_folder = os.path.isdir(dir_path)
    if not check_folder:
        os.makedirs(dir_path)


def update_dict(d, u, l):
    for k, v in u.items():
        if k in l:
            if isinstance(v, collections.abc.Mapping):
                d[k] = update_dict(d.get(k, {}), v, l=v.keys())
            else:
                d[k] = v
    return d


def updateconfig(orig, update, input_dim):
    CONFIG_KEYS = [
        "model",
        "datamodule",
        "encoder",
        "decoder",
        "trainer",
        "callbacks",
        "logger",
        "out_dir",
    ]
    OmegaConf.set_struct(orig, True)
    with open_dict(orig):
        # update default cfg with user config
        if update is not None:
            update_keys = list(set(update.keys()) & set(CONFIG_KEYS))
            orig = update_dict(orig, update, l=update_keys)

        # update encoder/decoder config
        for i, d in enumerate(input_dim):
            enc_key = f"enc{i}"
            if enc_key not in orig["encoder"].keys():
                if (
                    update is not None
                    and "encoder" in update.keys()
                    and enc_key in update["encoder"].keys()
                ):  # use user-defined
                    orig["encoder"][enc_key] = update["encoder"][enc_key].copy()
                else:  # use default
                    orig["encoder"][enc_key] = orig["encoder"].default.copy()

            dec_key = f"dec{i}"
            if dec_key not in orig["decoder"].keys():
                if (
                    update is not None
                    and "decoder" in update.keys()
                    and dec_key in update["decoder"].keys()
                ):  # use user-defined
                    orig["decoder"][dec_key] = update["decoder"][dec_key].copy()
                else:  # use default
                    orig["decoder"][dec_key] = orig["decoder"].default.copy()

    #     if update is not None:
    #         print(update['out_dir'])
    #         OmegaConf.set_struct(orig, True)
    #         orig.outdir = update['out_dir']
    return orig


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

S_TRAIN_PATH = Path("processed_data/ct_postCombat_residSexAge_060623.csv")

F_TRAIN_PATH = Path(
    "ABCD_mVAE_LizaEric/data/rsfmri_gordon_no_dup_postCombat_residSexAge_060623.csv"
)

with open("ABCD_mVAE_LizaEric/data/train_val_subs.pkl", "rb") as f:
    train_val_subs = pickle.load(f)


with open(Path("ABCD_mVAE_LizaEric/data", "phenotype_roi_mapping.json")) as f:
    phenotype_roi_mapping = json.loads(f.read())


# Load and prepare dataset
s_data = pd.read_csv(Path(S_TRAIN_PATH), index_col=0)

s_data_subset = s_data.loc[train_val_subs]

scaler = StandardScaler()

s_rescaled_data = scaler.fit_transform(
    s_data.loc[train_val_subs, phenotype_roi_mapping["ct"]]
)

s_train_data = torch.tensor(
    s_rescaled_data,
    dtype=torch.float32,
)

print(s_train_data.shape)

# %%

f_data = pd.read_csv(Path(F_TRAIN_PATH), index_col=0)

f_data_subset = f_data.loc[train_val_subs]

fmri_roi_mapping = f_data_subset.columns[0:93]

f_rescaled_data = scaler.fit_transform(
    f_data_subset.loc[train_val_subs, fmri_roi_mapping]
)

f_train_data = torch.tensor(
    f_rescaled_data,
    dtype=torch.float32,
)

train_data = [s_train_data, f_train_data]

input_dims = [train_data[0].shape[1], train_data[1].shape[1]]

# %%

curr_phenotype = "ct and rsfmri"
latent_dim = [10, 15, 20]
hidden_layer_dim = [[74, 37], [37, 37, 37], [74, 74, 74]]
beta = [1]
non_linear = [True]
batch_size = [64, 128, 256, 512]
learning_rate = [
    0.001,
    0.005,
    0.01,
]


parameters = {
    "z_dim": latent_dim,
    "hidden_layer_dim": hidden_layer_dim,
    "beta": beta,
    "non_linear": non_linear,
    "batch_size": batch_size,
}

keys, values = zip(*parameters.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
max_epochs = 5000


for i, param_dict in enumerate(permutations_dicts):
    dir = f"results/two_views/mvae_{curr_phenotype}_two_Views_param_comb{i}_010624"
    in_dict = {
        "out_dir": dir,
        "model": {"z_dim": param_dict["z_dim"], "beta": param_dict["beta"]},
        "encoder": {
            "default": {
                "hidden_layer_dim": param_dict["hidden_layer_dim"],
                "non_linear": param_dict["non_linear"],
            }
        },
        "decoder": {
            "default": {
                "hidden_layer_dim": param_dict["hidden_layer_dim"][::-1],
                "non_linear": param_dict["non_linear"],
            }
        },
        "datamodule": {
            "batch_size": param_dict["batch_size"],
        },
    }

    with initialize_config_dir(version_base=None, config_dir=os.getcwd()):
        user_cfg = compose(config_name="configs/bi_view.yaml", return_hydra_config=True)

    new_cfg = {}
    for key, value in user_cfg.items():
        if str(key) != "hydra":
            new_cfg[key] = value
    new_cfg = OmegaConf.create(new_cfg)
    new_cfg = updateconfig(new_cfg, in_dict, input_dims)
    create_folder(new_cfg["out_dir"])
    with open(os.path.join(new_cfg["out_dir"], "config.yaml"), "w") as f:
        f.write("# @package _global_\n")
        OmegaConf.save(new_cfg, f)
    # OmegaConf.save(new_cfg, join(new_cfg['out_dir'], 'config.yaml'))
    input_config = os.path.join(new_cfg["out_dir"], "config.yaml")

    mvae = mVAE(
        cfg=input_config,
        input_dim=input_dims,
    )

    mvae.fit(
        train_data[0],
        train_data[1],
        max_epochs=max_epochs,
        batch_size=param_dict["batch_size"],
    )

    # mvae.validation_step()
# %%
