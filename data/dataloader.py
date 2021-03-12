import os, sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.append(os.path.abspath(os.path.join('..')))
from config import Config as cfg

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



def get_weighted_sampler(ratio, normalize=False):
    """
    Returns dataframe and WeightedRandomSampler based on desired ratio of sub-optimal to 
    optimal data rollouts
        ratio: split tuple of opt to sub-opt data, e.g. (0.1, 0.9) for
               10% optimal and 90% suboptimal training data
        normalize: whether the weights should be normalized based on count
    """

    assert np.sum(ratio) == 1.0, "Ratios should some up to 1.0"
    files = glob.glob(os.path.join(cfg.DATA_PATH, "*.csv"))
    weights = torch.Tensor()
    df = pd.DataFrame()
    rollout_mapping = {"none": 0, "laggy": 1, "noisy": 2}
    rollout_count = {"opt": 0, "subopt": 0}
    for file in files:
        rollout = pd.read_csv(file, header=0, index_col=None)
        rollout_type = file.split("_")[1]
        # Append rollout type column to dataframe
        rollout['type'] = rollout_mapping[rollout_type]

        if rollout_type == cfg.OPT_LABEL:
            rollout_count["opt"] += len(rollout)
            weights = torch.cat((weights, ratio[0]*torch.ones(len(rollout))))
        else:
            rollout_count["subopt"] += len(rollout)
            weights = torch.cat((weights, ratio[1]*torch.ones(len(rollout))))

        df = pd.concat((df, rollout))

    # Balancing based on opt vs subopt data availability
    if normalize:
        weights[weights==ratio[0]] /= (rollout_count["opt"])#*ratio[0]
        weights[weights==ratio[1]] /= (rollout_count["subopt"])#*ratio[1]

    # NOTE: sampeld with replacement by default
    sampler = WeightedRandomSampler(weights, len(weights))
    return df, sampler



if __name__ == "__main__":
    # Test
    ratio = (0.3, 0.7)
    rollouts_df, weighted_sampler = get_weighted_sampler(ratio, normalize=True)

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
        print("Batch ratio: ", counts[1][0]/np.sum(counts[1]), np.sum(counts[1][1:])/np.sum(counts[1]))
        opt += counts[1][0]
        subopt += np.sum(counts[1][1:])
        if i > 10:
            break
    print("Overall ratio: ", opt/(opt+subopt), subopt/(opt+subopt))