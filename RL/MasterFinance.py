from AgentMasterFinance import DQNTrader
from EnvMasterFinance import TradingEnvIAR
from Tools import PrepareData

import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np


"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
dataset = pd.read_csv('C:/Users/Ugo/Documents/AI/Forex_ML/RL/DATA/FAKE_DATA_TRAIN.csv')

PD = PrepareData(dataset)
DATA = PD.data_for_training(size=80)
print(len(DATA))
env = TradingEnvIAR(DATA[0],window_size=20)

env.reset()
agent = DQNTrader(state_size=env.state_size, action_size=env.action_size,lstm_layer=[16,8])

episodes = 100
if len(DATA)<episodes:
    raise ValueError("Le nombre de sample n'est pas suffisant DATA:{}".format(len(DATA)))
batch_size = 16
Score = []
bar_progress = tqdm(range(episodes*(len(DATA[0])-20)))

for episode in range(1,episodes+1):
    
    data = DATA[episode-1]
    env = TradingEnvIAR(data,window_size=20)
    state = env.reset()
    #state = agent.normalize(state)
    #state = np.expand_dims(state, axis=-1)
    state = np.array([state])

    done = env.done
    total_reward = 0

    while not done:

        action = agent.act(state)

        next_state, reward, done,action, _= env.step(action)
        #print(reward)
        #next_state = agent.normalize(next_state)
        #next_state = np.expand_dims(next_state, axis=-1)
        next_state = np.array([next_state])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        # print("wallet",env.wallet)
        # print("reward",reward)
        total_reward = reward
        bar_progress.update(1)
        

        if done:
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
            print("Épisode :", episode,"Récompense totale :", total_reward)
            print("nombre de position ouverte: ",len(env.historic_order(env.orders)))
            if episode%10 == 0:
                agent.update_target_model()
                #plt.plot(Score,color='b')
                #plt.show()
            if episode%25 == 0 :
                agent.target_model.save('C:/Users/Ugo/Documents/AI/Forex_ML/RL/MODEL/model_FAKE_DATA_RW.keras')
                np.save('C:/Users/Ugo/Documents/AI/Forex_ML/RL/RESULTS/Score_FAKE_DATA_RW.npy', Score)
                print('saves done')
            break

        if len(agent.memory.buffer) > batch_size:
            agent.replay(batch_size)
    
    
    total_reward = reward
    Score.append(total_reward)

agent.target_model.save('C:/Users/Ugo/Documents/AI/Forex_ML/RL/MODEL/model_FAKE_DATA_RW.keras')
np.save('C:/Users/Ugo/Documents/AI/Forex_ML/RL/RESULTS/Score_FAKE_DATA_RW.npy', Score)
print('saves done')

Score = np.array(Score)
#Wallet = np.array(Wallet)

plt.plot(Score)
plt.show()

