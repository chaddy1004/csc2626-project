import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from Config import Config as cfg
import gym
import pickle


def prep_demo_data(frames, actions, rewards, endings):
    """Prepare demonstration data to be stored in replay buffer"""
    # TODO: maybe include a skip-frame parameter here
    demo_buffer = deque()
    demo = []
    score = 0
    episode = 0
    N = len(frames) # total number of frames (spanning multiple trials and episodes)
    i = 0
    while i < N:
        state = frames[i]
        action = actions[i]
        next_sate = frames[i+1]
        score += rewards[i+1]
        done = endings[i+1]
        reward = rewards[i+1] if not done else -100
        demo.append([state, action, reward, next_sate, done, 1.0]) # 1.0 here stands for expert data

        # End of episode reached
        if done:
            # TODO: add n step stuff here
            demo_buffer.extend(demo)
            print("episode: {}, score: {}, demo_buffer: {}".format(episode, score, len(demo_buffer)))
            # Reset
            done = False
            episode += 1
            score = 0
            demo = []
            i += 2
            continue
        i += 1
    # Compress and save data
    print("Pickling and saving!")
    with open(cfg.DEMO_DATA_PATH, 'wb') as fid:
        pickle.dump(demo_buffer, fid, protocol=4)
    return demo_buffer


def prep_expert_panda(frames, actions, rewards, endings, df):
    """Prepare demonstration data dataframe to be used in dataloader
        Returns dataframe containing filenames of current states and next states
    """

    d = {'state':[], 'action':[], 'reward':[], 'next':[], 'end':[], 'episode': []}
    episode = 0
    N = len(frames)
    i = 0
    while i < N:
        d['state'].append(df['png_names'][i])
        d['action'].append(actions[i])
        d['next'].append(df['png_names'][i+1])
        done = endings[i+1]
        d['end'].append(done)
        d['reward'].append(rewards[i+1] if not done else -100)
        d['episode'].append(episode)

        if done:
            done = False
            episode += 1
            i += 2
            continue
        i += 1
    demo_df = pd.DataFrame(data=d)
    print("Exporting panda to csv!")
    demo_df.to_csv(cfg.DATA_PATH + 'demo_df.csv', index=False)
    return demo_df
       


if __name__ == "__main__":
    # Load demonstrations:
    # see https://github.com/felix-kerkhoff/DQfD/blob/c154e14414c02ec4497fb9e3f8834a6cdb417101/load_data.py
    # to find out how the .npy files are generated
    # NOTE: no frame skips were performed yet
    frames = np.load(cfg.DATA_PATH + cfg.GAME_ID + "_frames.npy")
    frames_df = pd.read_csv(cfg.DATA_PATH + "mr_demo_data.csv")
    actions = np.load(cfg.DATA_PATH + cfg.GAME_ID + "_actions.npy")
    rewards = np.load(cfg.DATA_PATH + cfg.GAME_ID + "_rewards.npy")
    episode_endings = np.load(cfg.DATA_PATH + cfg.GAME_ID +  "_episode_endings.npy")

    # Prepare demonstration data
    demo_buffer = prep_demo_data(frames, actions, rewards, episode_endings)

    # Prepare demonstration dataframe
    demo_df = prep_expert_panda(frames, actions, rewards, episode_endings, frames_df)

    # Env
    # game_id = 'MontezumaRevengeNoFrameskip-v4'
    # env = gym.make(game_id)
    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n