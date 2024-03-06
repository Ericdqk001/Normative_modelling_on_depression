from pathlib import Path

import pandas as pd
import torch
from load.load import MyDataset
from models.VAE import VAE
from torch.utils.data import DataLoader

# Config
batch_size = 100
epochs = 100
learning_rate = 0.001
latent_dim = 10
hidden_dim = [40, 40]


data = pd.read_csv(
    Path(
        "processed_data",
        "ct_postCombat_residSexAge_060623.csv",
    )
)

columns_to_drop = [
    "subjectkey",
    "sex",
    "interview_age",
    "eTIV",
    "has_clinical_pp",
    "ksads_dx",
]

features = data.drop(columns_to_drop, axis="columns").copy()

### TODO Change the split when Liza give the code
train_data = features.sample(frac=0.8, random_state=42)
val_data = features.drop(train_data.index)
###

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

input_dim = train_data.shape[1]

train_dataset = MyDataset(train_data.to_numpy())

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
)

val_loader = DataLoader(
    MyDataset(val_data.to_numpy()),
    batch_size=batch_size,
    shuffle=False,
)

model = VAE(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
    learning_rate=learning_rate,
    non_linear=True,
)
model.to(DEVICE)


for epoch in range(epochs):
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(DEVICE)
        fwd_rtn = model.forward(batch)
        loss = model.loss_function(batch, fwd_rtn)
        model.optimizer.zero_grad()
        loss["total"].backward()
        model.optimizer.step()
        if batch_idx == 0:
            to_print = (
                "Train Epoch:"
                + str(epoch)
                + " "
                + "Train batch: "
                + str(batch_idx)
                + " "
                + ", ".join(
                    [k + ": " + str(round(v.item(), 3)) for k, v in loss.items()]
                )
            )
            # print(to_print)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = batch.to(DEVICE)
                fwd_rtn = model.forward(batch)
                val_loss = model.loss_function(batch, fwd_rtn)
                batch_val_loss = val_loss["total"].item()
                total_val_loss += batch_val_loss
                if batch_idx == 0:
                    to_print = (
                        "Val Epoch:"
                        + str(epoch)
                        + " "
                        + "Val batch: "
                        + str(batch_idx)
                        + " "
                        + ", ".join(
                            [
                                k + ": " + str(round(v.item(), 3))
                                for k, v in val_loss.items()
                            ]
                        )
                    )
                    # print(to_print)

        print(total_val_loss / len(val_loader))
