import os

class Config:
    ENV_NAME = 'MontezumaRevengeNoFrameskip-v4'
    GAME_ID = 'montezuma_revenge'
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AtariHEADArchives/')
    DEMO_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mr_demo_data.p')
