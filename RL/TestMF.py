from AgentMasterFinance import DQNTrader
from RL.utils.EnvMasterFinance import TradingEnv
from utils.tools import PrepareData

import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import load_model

"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

dataset = pd.read_csv('C:/Users/Ugo/Documents/AI/Forex_ML/RL/DATA/FAKE_DATA_TEST.csv',index_col=None)
dataset= dataset[:]
#dataset=dataset.drop(['real_date'], axis=1)
#plt.plot(dataset['PRICE'])
#plt.show()

prix = dataset.to_numpy()[:,0]
PD = PrepareData(dataset)
data = PD.data_for_test()


env = TradingEnvIAR(data,window_size=21,episode_size=252)
env.reset()
agent = DQNTrader(state_size=env.state_size, action_size=env.action_size)
model = load_model('C:/Users/Ugo/Documents/AI/Forex_ML/RL/MODEL/model_FAKE_DATA_RW.keras')
agent.model = model
agent.epsilon = 0

done = False
state = env.reset()
#state = agent.normalize(state)
#state = np.expand_dims(state, axis=-1)
state = np.array([state])
total_reward = 0

profit = []
duree = []

bar_progress = tqdm(range((len(dataset)-5)))

while not done:
    action = agent.act(state)
    next_state, reward, done, action, info = env.step(action)
    #next_state = agent.normalize(next_state)
    #next_state = np.expand_dims(next_state, axis=-1)
    next_state = np.array([next_state])
    #actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
    state = next_state

    total_reward += reward
    bar_progress.update(1)
    if done:
        print("total reward after {} steps is {}".format(1, total_reward))
        perte = 0
        gain = 0
        for order in env.orders:
            end_date = order.end_date
            start_date = order.start_date
            p = (prix[end_date]-prix[start_date])*order.order_type
            profit.append(p)
            duree.append(end_date-start_date)
            if p < 0:
                perte+=1
            else :
                gain+=1
        print("nombre de position ouverte: ",env.historic_order(env.orders))
        print("durée moyenne d'une action: {}, nbre de position ouverte à perte:{},nbre de position ouverte à profit:{}, profit moyenne au cour de la partie{}, profit total:{}".format(np.mean(duree),perte,gain,np.mean(profit),sum(profit)/0.001))
        
