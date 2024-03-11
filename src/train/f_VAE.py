import pickle
from pathlib import Path

import pandas as pd
import torch
from load.load import MyDataset
from models.VAE import VAE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import wandb

TRAIN_PATH = Path(
    "ABCD_mVAE_LizaEric/data/rsfmri_gordon_no_dup_postCombat_residSexAge_060623.csv"
)


with open("ABCD_mVAE_LizaEric/data/train_val_subs.pkl", "rb") as f:
    train_val_subs = pickle.load(f)

check_points_path = Path("checkpoints")


def build_model(config, input_dim):
    print(input_dim)
    model = VAE(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        learning_rate=config.learning_rate,
        non_linear=True,
    )

    return model


def train(
    config,
    model,
    train_loader,
):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

    model.to(DEVICE)

    model_artifact = wandb.Artifact(wandb.run.name, type="model")

    for epoch in range(config.epochs):
        wandb.log({"epoch": epoch})

        model.train()

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(DEVICE)
            fwd_rtn = model.forward(batch)
            loss = model.loss_function(batch, fwd_rtn)
            model.optimizer.zero_grad()
            loss["total"].backward()
            model.optimizer.step()

            wandb.log(
                {
                    "loss": loss["total"].item(),
                }
            )

    torch.save(
        model.state_dict(),
        Path(check_points_path, "model_weights.pt"),
    )

    model_artifact.add_file(
        Path(check_points_path, "model_weights.pt"),
    )

    wandb.run.log_artifact(model_artifact)


def main(config):
    # Assuming DEVICE setup is the same
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

    # Load and prepare your dataset
    data = pd.read_csv(Path(TRAIN_PATH), index_col=0)

    # data.index = data["subjectkey"]

    train_data_subset = data.loc[train_val_subs]

    scaler = StandardScaler()

    fmri_roi_mapping = train_data_subset.columns[0:93]

    train_dataset_numpy = scaler.fit_transform(
        data.loc[train_val_subs, fmri_roi_mapping]
    )

    print(train_dataset_numpy.shape)

    train_loader = DataLoader(
        MyDataset(train_dataset_numpy), batch_size=config.batch_size, shuffle=True
    )

    # Get input_dim based on the dataset
    input_dim = train_dataset_numpy.shape[1]

    model = build_model(config, input_dim).to(DEVICE)

    train(
        config,
        model,
        train_loader,
    )


if __name__ == "__main__":
    with wandb.init(
        project="VAE sweep k-fold fmri final model",
        config={
            "batch_size": 64,
            "hidden_dim": [45, 45],
            "latent_dim": 15,
            "learning_rate": 0.001,
            "epochs": round((269 + 247 + 235 + 252 + 269) / 5),
        },
    ):
        main(wandb.config)
