import os

class Config:
    ENV_NAME = 'MontezumaRevengeNoFrameskip-v4'
    GAME_ID = 'montezuma_revenge'
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/AtariHEADArchives/')
    DEMO_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_PATH, 'mr_demo_data.p')
