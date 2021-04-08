import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join('..')))
from config import Config as cfg
from data.data_utils import get_weighted_sampler


# DATALOADER
class OfflineDataset(Dataset):

    def __init__(self, data, training=True, transform=None):
        self.data = torch.Tensor(data)

        # Load <s,a,r,s',done,episode>
        self.current_states = self.data[:, :8]
        self.actions = self.data[:, 8:10]
        self.rewards = self.data[:, 10]
        self.next_states = self.data[:, 11:19]
        self.done = self.data[:, 19]
        self.episode = self.data[:, 20]
        self.type = self.data[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return tuple <s,a,r,s',done,type>

        return (self.current_states[idx],
                self.actions[idx],
                self.rewards[idx],
                self.next_states[idx],
                self.done[idx],
                self.type[idx])


if __name__ == "__main__":
    # Test
    ratio = (0.9, 0.1)
    rollouts_df, weighted_sampler, weights = get_weighted_sampler(ratio, normalize=True)

    # Dataset
    dataset = OfflineDataset(
        dataframe=rollouts_df,
        training=True
    )
    print("Length of dataset: ", len(dataset))

    # Dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.BS,
        num_workers=cfg.WORKERS,
        sampler=weighted_sampler
    )

    # Enumeration shows split between optimal and suboptimal
    opt = 0
    subopt = 0
    for i, data in enumerate(train_loader):
        counts = np.unique(data[-1].numpy(), return_counts=True)
        opt_count = np.sum(counts[1][counts[0] == 0])
        subopt_count  = np.sum(counts[1][counts[0] != 0])
        print("Batch ratio: ", opt_count / np.sum(counts[1]), subopt_count / np.sum(counts[1]))
        opt += opt_count
        subopt += subopt_count
        if i > 10:
            break
    print("Overall ratio: ", opt / (opt + subopt), subopt / (opt + subopt))
