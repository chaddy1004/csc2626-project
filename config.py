import os

class Config:
    # DATALOADING
    ENV_NAME = 'LunarLanderContinuous-v2'
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/LunarLanderData/')
    OPT_LABEL = 'none'
    
    # TRAINING
    BS = 64
    WORKERS = 1
