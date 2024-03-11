import pickle
from pathlib import Path

import pandas as pd
import psutil
import torch
from load.load import MyDataset
from models.VAE import VAE
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import wandb

# Config


def check_mps_memory():
    process = psutil.Process()
    mem_info = process.memory_info()
    print("Memory used:", mem_info.rss / 1024 / 1024 / 1024, "GB")
    print("Virtual memory used:", mem_info.vms / 1024 / 1024 / 1024, "GB")

    vm_info = psutil.virtual_memory()
    print("Total virtual memory:", vm_info.total / 1024 / 1024 / 1024, "GB")
    print("Available virtual memory:", vm_info.available / 1024 / 1024 / 1024, "GB")
    print("Used virtual memory:", vm_info.used / 1024 / 1024 / 1024, "GB")
    print("Memory percent used:", vm_info.percent, "%")
    print("---------")

    return {
        "memory used": mem_info.rss / 1024 / 1024 / 1024,
        "virtual memory used": mem_info.vms / 1024 / 1024 / 1024,
        "total virtual memory": vm_info.total / 1024 / 1024 / 1024,
        "available virtual memory": vm_info.available / 1024 / 1024 / 1024,
        "used virtual memory": vm_info.used / 1024 / 1024 / 1024,
        "memory percent used": vm_info.percent,
    }


TRAIN_PATH = Path(
    "ABCD_mVAE_LizaEric/data/rsfmri_gordon_no_dup_postCombat_residSexAge_060623.csv"
)

with open("ABCD_mVAE_LizaEric/data/train_val_subs.pkl", "rb") as f:
    train_val_subs = pickle.load(f)


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


def validate(model, val_loader, device):
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            fwd_rtn = model.forward(batch)
            val_loss = model.loss_function(batch, fwd_rtn)
            batch_val_loss = val_loss["total"].item()
            total_val_loss += batch_val_loss

    mean_val_loss = total_val_loss / len(val_loader)

    return mean_val_loss


def train(
    config,
    model,
    train_loader,
    val_loader,
    tolerance=10,
):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

    model.to(DEVICE)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    model_artifact = wandb.Artifact(wandb.run.name, type="model")

    for epoch in range(config.epochs):
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(DEVICE)
            fwd_rtn = model.forward(batch)
            loss = model.loss_function(batch, fwd_rtn)
            model.optimizer.zero_grad()
            loss["total"].backward()
            model.optimizer.step()

        val_loss = validate(model, val_loader, DEVICE)

        wandb.log({"val_loss": val_loss})

        if val_loss < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = val_loss

            print("New best val loss:", val_loss)
            print("at epoch:", epoch)

            best_model = model.state_dict()

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= tolerance:
            print("Early stopping at epoch:", epoch)

            check_points_path = Path(
                "content", "drive", "MyDrive", "checkpoints_MRES_fMRI", "s_VAE"
            )

            if not check_points_path.exists():
                check_points_path.mkdir(parents=True)

            torch.save(
                best_model,
                Path(check_points_path, f"{epoch - tolerance}_model_weights.pt"),
            )

            model_artifact.add_file(
                Path(check_points_path, f"{epoch - tolerance}_model_weights.pt"),
            )

            wandb.run.log_artifact(model_artifact)

            return best_val_loss


def train_k_fold(config, n_splits=5):
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

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold = 0
    total_val_loss = 0.0
    for train_index, val_index in kf.split(train_dataset_numpy):
        print(f"Training on fold {fold+1}...")
        # Split dataset into training and validation sets for the current fold

        train_data, val_data = (
            train_dataset_numpy[train_index],
            train_dataset_numpy[val_index],
        )

        # Here, you could modify build_loader to accept train_data and val_data directly,
        # or just create the DataLoader instances directly in this loop.
        train_loader = DataLoader(
            MyDataset(train_data), batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            MyDataset(val_data), batch_size=config.batch_size, shuffle=False
        )

        # Get input_dim based on the dataset
        input_dim = train_data.shape[1]

        model = build_model(config, input_dim).to(DEVICE)

        val_loss = train(config, model, train_loader, val_loader)

        total_val_loss += val_loss

        fold += 1

    return total_val_loss / n_splits


def main():
    wandb.init()
    val_loss = train_k_fold(wandb.config)
    wandb.log({"average_val_loss": val_loss})


if __name__ == "__main__":
    sweep_configuration = {
        "method": "bayes",
        "name": "sweep",
        "metric": {"goal": "minimize", "name": "average_val_loss"},
        "parameters": {
            "batch_size": {"values": [32, 64, 128]},
            "learning_rate": {
                "values": [
                    0.001,
                    0.002,
                    0.003,
                    0.004,
                    0.005,
                    0.006,
                    0.007,
                    0.008,
                    0.009,
                    0.01,
                ]
            },
            "latent_dim": {"values": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
            "epochs": {"value": 500},
            "hidden_dim": {
                "values": [
                    [40, 40],
                    [45, 45],
                    [50, 50],
                    [55, 55],
                    [60, 60],
                    [40, 40, 40],
                    [45, 45, 45],
                    [50, 50, 50],
                    [55, 55, 55],
                    [60, 60, 60],
                ]
            },
        },
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="VAE sweep 5-fold fmri Colab with scalar"
    )

    wandb.agent(sweep_id, function=main, count=50)

# TODO Check if the model is save at the right epoch
