# TODO Gather the variables of interest.
# TODO Create a function to load the data.
# TODO Separate the data by age group.
# TODO Examine the amount of samples in different age groups
# and decide how to handle the imbalance and missing data.
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, indices=False, transform=None):
        self.data = data
        if isinstance(data, list) or isinstance(data, tuple):
            self.data = [
                torch.from_numpy(d).float() if isinstance(d, np.ndarray) else d
                for d in self.data
            ]
            self.N = len(self.data[0])
            self.shape = np.shape(self.data[0])
        else:
            if isinstance(data, np.ndarray):
                self.data = torch.from_numpy(self.data).float()
            self.N = len(self.data)
            self.shape = np.shape(self.data)

        self.transform = transform
        self.indices = indices

    def __getitem__(self, index):
        if isinstance(self.data, list):
            x = [d[index] for d in self.data]
        else:
            x = self.data[index]

        if self.transform:
            x = self.transform(x)

        if self.indices:
            return x, index
        return x

    def __len__(self):
        return self.N
