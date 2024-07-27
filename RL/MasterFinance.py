from AgentMasterFinance import DQNTrader
from EnvMasterFinance import TradingEnvIAR
from Tools import PrepareData

import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np


"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
dataset = pd.read_csv('C:/Users/Ugo/Documents/AI/Forex_ML/RL/DATA/FAKE_DATA_TRAIN.csv')
dataset = dataset.copy().to_numpy()
window_size = 21
episode_size = 84
episodes = 250
batch_size = 16


env = TradingEnvIAR(data=dataset,window_size=window_size,episode_size=episode_size,n=2)
env_test = TradingEnvIAR(data=dataset,window_size=window_size,episode_size=episode_size,n=1)
env.reset()

agent = DQNTrader(state_size=env.state_size, action_size=env.action_size,lstm_layer=[16,8],epsilon_decay=0.01**(1/episodes),epsilon_min=0.01,buffer_size=15000)


Score = []
Score_test = []
bar_progress = tqdm(range(episodes*episode_size))

for episode in range(1,episodes+1):
    state = env.reset()
    #state = agent.normalize(state)
    #state = np.expand_dims(state, axis=-1)
    state = np.array([state])

    while True:

        action = agent.act(state)

        next_state, reward, done,action, _= env.step(action)
        #print(reward)
        #next_state = agent.normalize(next_state)
        #next_state = np.expand_dims(next_state, axis=-1)
        next_state = np.array([next_state])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        
        bar_progress.update(1)
        
        if done:
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
            
            if episode%10 == 0:
                agent.update_target_model()
                #plt.plot(Score,color='b')
                #plt.show()
            
            if episode%1==0:
                state_test=env_test.reset(env.current_step)
                state_test = np.array([state_test])
                done_test = False
                while not done_test :
                    action_test = agent.act(state_test)
                    next_state_test, reward_test, done_test, action_test, info = env_test.step(action_test)
                    next_state_test = np.array([next_state_test])
                    state_test = next_state_test
                print("Test terminé")
                Score_test.append(env_test.wallet)

           
            if episode%25 == 0 :
                agent.target_model.save('C:/Users/Ugo/Documents/AI/Forex_ML/RL/MODEL/model_FAKE_DATA_RW_spread.keras')
                np.save('C:/Users/Ugo/Documents/AI/Forex_ML/RL/RESULTS/Score_FAKE_DATA_RW_spread.npy', Score)
            
            print("Épisode :", episode,"Récompense totale :", env.wallet)
            print("nombre de position ouverte: ",len(env.historic_order(env.orders)))
            break

        if len(agent.memory.buffer) > batch_size:
            agent.replay(batch_size)
    
    Score.append(env.wallet)

print('saves done')

Score = np.array(Score)
X = np.array([i for i in range(1, episodes+1)])
X_test = np.array([i for i in range(1, episodes+1) if i % 5 == 0])
plt.plot(X,Score)
plt.plot(X_test,Score_test)
plt.show()

