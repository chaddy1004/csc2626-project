import numpy as np
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
    while i < N -1:
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
    # with open(cfg.DEMO_DATA_PATH, 'wb') as fid:
    #     pickle.dump(demo_buffer, fid, protocol=4)
    return demo_buffer


if __name__ == "__main__":
    # Load demonstrations:
    # see https://github.com/felix-kerkhoff/DQfD/blob/c154e14414c02ec4497fb9e3f8834a6cdb417101/load_data.py
    # to find out how the .npy files are generated
    # NOTE: no frame skips were performed yet
    frames = np.load(cfg.DATA_PATH + cfg.GAME_ID + "_frames.npy")
    actions = np.load(cfg.DATA_PATH + cfg.GAME_ID + "_actions.npy")
    rewards = np.load(cfg.DATA_PATH + cfg.GAME_ID + "_rewards.npy")
    episode_endings = np.load(cfg.DATA_PATH + cfg.GAME_ID +  "_episode_endings.npy")

    # Prepare demonstration data
    demo_buffer = prep_demo_data(frames, actions, rewards, episode_endings)

    # Env
    # game_id = 'MontezumaRevengeNoFrameskip-v4'
    # env = gym.make(game_id)
    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n