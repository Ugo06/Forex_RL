from AgentMasterFinance import DQNTrader
from EnvMasterFinance import TradingEnv
from Tools import PrepareData

import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from config import *

# Load and prepare dataset
dataset_path = DATA_PATH
dataset = pd.read_csv(dataset_path).to_numpy()

# Initialize environments
env = TradingEnv(data=dataset, window_size=WINDOW_SIZE, episode_size=EPISODE_SIZE, n=N_TRAIN,mode=MODE)
env_test = TradingEnv(data=dataset, window_size=WINDOW_SIZE, episode_size=EPISODE_SIZE, n=N_TEST,mode=MODE)
env.reset()

# Initialize agent
agent = DQNTrader(
    state_size=env.state_size, 
    action_size=env.action_size,
    lstm_layer=LSTM_LAYER,
    epsilon_decay=EPSILON_DECAY,
    epsilon_min=EPSILON_MIN,
    buffer_size=BUFFER_SIZE,
    gamma=GAMMA
)

# Training variables
train_scores = []
test_scores = []
progress_bar = tqdm(range(EPISODE_SIZE * NB_EPISODE))

for episode in range(1, NB_EPISODE + 1):
    state = np.array([env.reset()])

    while True:
        action = agent.act(state)
        next_state, reward, done, action, _ = env.step(action)
        next_state = np.array([next_state])
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        
        progress_bar.update(1)
        
        if done:
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
            
            if episode % ITER_SAVE_TARGET_MODEL == 0:
                agent.update_target_model()
            
            if episode % ITER_TEST == 0:
                state_test = np.array([env_test.reset(env.current_step)])
                done_test = False
                while not done_test:
                    action_test = agent.act(state_test)
                    next_state_test, reward_test, done_test, action_test, info = env_test.step(action_test)
                    state_test = np.array([next_state_test])
                print("Test completed")
                test_scores.append(env_test.wallet)
            
            if episode % ITER_SAVE_MODEL_SCORE == 0:
                model_save_path = MODEL_SAVE_PATH
                score_save_path = SCORE_SAVE_PATH
                agent.target_model.save(model_save_path)
                np.save(score_save_path, train_scores)
            
            print(f"Episode: {episode}, Total Reward: {env.wallet}")
            print(f"Number of open positions: {len(env.historic_order(env.orders))}")
            break

        if len(agent.memory.buffer) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)
    
    train_scores.append(env.wallet)

print('Training completed and models saved.')

# Plot results
train_scores = np.array(train_scores)
X = np.arange(1, NB_EPISODE + 1)
X_test = np.array([i for i in range(1, NB_EPISODE + 1) if i % ITER_TEST == 0])

plt.plot(X, train_scores, label='Training Score')
plt.plot(X_test, test_scores, label='Test Score')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.legend()
plt.show()

