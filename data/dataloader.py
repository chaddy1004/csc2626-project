import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from cv2 import cvtColor, COLOR_BGR2GRAY, resize, INTER_AREA
from PIL import Image

# DATALOADER
class ExpertDemonstrations(Dataset):

    def __init__(self, dataframe, root_dir, training=True, transform=None):
        self.data = dataframe
        self.root_dir = root_dir
        self.transform = transform

        # Load <s,a,r,s'>
        self.current_states = self.data['state'].to_list()
        self.actions = self.data['action'].to_numpy()
        self.rewards = self.data['reward'].to_numpy()
        self.next_states = self.data['next'].to_list()

    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        # Return tuple <s,a,r,s'>

        current_state = Image.open(os.path.join(self.root_dir, self.current_states[idx]))
        next_state = Image.open(os.path.join(self.root_dir, self.next_states[idx]))
        action = self.actions[idx]
        reward = self.rewards[idx]

        if self.transform:
            current_state = self.transform(current_state)
            next_state = self.transform(next_state)
        
        return (current_state, action, reward, next_state)


class GrayResizeTransform:
    """Transformation to match pickled data
        Returns gray (84x84) numpy array
    """
    
    def __call__(self, img):
        gray = cvtColor(np.array(img), COLOR_BGR2GRAY)
        resized = resize(gray, (84, 105), interpolation=INTER_AREA)
        # Crop the score heading and black bar at the bottom of the frame
        cropped = resized[15:99]
        return cropped


if __name__ == "__main__":
    # Test
    root_data_dir = "./AtariHEADArchives/montezuma_revenge/"
    demo_data = pd.read_csv("./AtariHEADArchives/mr_demo_data_df.csv")

    # Transforms
    transform = transforms.Compose([
        GrayResizeTransform(),
        # transforms.ToTensor() # uncomment for training
    ])

    # Dataset
    dataset = ExpertDemonstrations(
        dataframe=demo_data,
        root_dir=root_data_dir,
        transform=transform,
        training=True
    )

    print("Length of dataset: ", len(dataset))
    plt.imshow(dataset[26146][3])
    plt.title("Reward {}".format(dataset[26146][2]))
    plt.show()