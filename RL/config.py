WINDOW_SIZE = 21
EPISODE_SIZE = 84
NB_EPISODE = 250
INITIAL_STEP = -1
N_TRAIN = 2
N_TEST = 1
MODE = {
        'include_price': False,
        'include_historic_position': False,
        'include_historic_action': False,
        'include_historic_wallet': False
        }

LSTM_LAYER = [16,8]
ESPILON = 1
EPSILON_MIN = 0.01
EPSILON_DECAY = EPSILON_MIN**(1/NB_EPISODE)
BUFFER_SIZE = 15000
GAMMA = 0.995
BATCH_SIZE = 16

ITER_SAVE_MODEL_SCORE = 25
ITER_SAVE_TARGET_MODEL = 10
ITER_TEST = 1

DATA_PATH = 'C:/Users/Ugo/Documents/AI/Forex_ML/RL/DATA/FAKE_DATA_TRAIN.csv'
MODEL_SAVE_PATH = 'C:/Users/Ugo/Documents/AI/Forex_ML/RL/MODEL/model_FAKE_DATA_RW_spread.keras'
SCORE_SAVE_PATH = 'C:/Users/Ugo/Documents/AI/Forex_ML/RL/RESULTS/Score_FAKE_DATA_RW_spread.npy'



