import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config as cfg
from data.data_utils import get_weighted_sampler

sys.path.append(os.path.abspath(os.path.join('..')))


# DATALOADER
class OfflineDataset(Dataset):

    def __init__(self, dataframe, training=True, transform=None):
        self.data = torch.Tensor(dataframe.to_numpy())

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
    ratio = (0.4, 0.6)
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
    # NOTE: sampling with ratios (1, 0), (0, 1) or (0.5, 0.5) breaks down
    # instead, I used (0.99, 0.01), (0.01, 0.99) and (0.49, 0.51) as an approximation
    # TODO: fix this!
    opt = 0
    subopt = 0
    for i, data in enumerate(train_loader):
        counts = np.unique(data[-1].numpy(), return_counts=True)
        print("Batch ratio: ", counts[1][0] / np.sum(counts[1]), np.sum(counts[1][1:]) / np.sum(counts[1]))
        opt += counts[1][0]
        subopt += np.sum(counts[1][1:])
        if i > 10:
            break
    print("Overall ratio: ", opt / (opt + subopt), subopt / (opt + subopt))
