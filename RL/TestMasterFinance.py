import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import argparse

from utils.tools import PrepareData
from utils.AgentMasterFinance import DQNTrader
from utils.EnvMasterFinance import TradingEnv

from tensorflow.keras.models import load_model

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main(config):
    # Create the run directory
    run_folder = os.path.join(config['SAVE_DIR'], f"config_{config['RUN_ID']}")
    
    # Load dataset
    dataset = pd.read_csv(filepath_or_buffer=config['DATA_PATH'])
    dataset = dataset.drop(['OPEN','SMA_5','SMA_50'],axis=1).to_numpy()
    data = PrepareData(dataset)
    print(pd.DataFrame(data.data).head())
    data.normalize()
    dataset = data.norm_data[len(data.norm_data)-(config['SPLIT']+config["N_TRAIN"]*config['EPISODE_SIZE']+1):]

    # Initialize environments
    mode = {
        'include_price': config['MODE']['include_price'],
        'include_historic_position': config['MODE']['include_historic_position'],
        'include_historic_action': config['MODE']['include_historic_action'],
        'include_historic_wallet': config['MODE']['include_historic_wallet'],
        'include_historic_orders': config['MODE']['include_historic_orders'],
    }
    env = TradingEnv(data=dataset,nb_action=config['NB_ACTION'], 
                     window_size=config['WINDOW_SIZE'], 
                     episode_size=config['EPISODE_SIZE'],
                     initial_step='sequential',
                     n=config['N_TEST'], 
                     mode=mode, 
                     reward_function=config['REWARD_FUNCTION'],
                     wallet=config['WALLET'],
                     zeta=config['ZETA'],
                     beta=config['BETA'])

    # Initialize agent
    agent = DQNTrader(
        state_size=env.state_size,
        action_size=env.action_size,
        type=config['TYPE'],
        config_layer=config['CONFIG_LAYER'],
        epsilon_decay=config['EPSILON_DECAY'],
        epsilon_min=config['EPSILON_MIN'],
        buffer_size=config['BUFFER_SIZE'],
        gamma=config['GAMMA'],
        alpha=config['ALPHA'],
        batch_size=config['BATCH_SIZE']
    )
    
    model_save_path = os.path.join(run_folder, f"model_final.keras")
    agent.model = load_model(model_save_path)
    episodes = int(config['SPLIT']/config['EPISODE_SIZE'])
    # Training variables

    online_training_test = []
    offline_training_test = []
    
    progress_bar = tqdm(range(config['EPISODE_SIZE'] * episodes))
    state = np.array([env.reset(initial_step=env.initial_step)])
    
    for episode in range(1, episodes + 1):

        while True:
            action = agent.act(state)
            next_state, reward, done, action, _ = env.step(action)
            next_state = np.array([next_state])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            progress_bar.update(1)

            if done:

                if episode % config['ITER_SAVE_TARGET_MODEL'] == 0:
                    agent.update_target_model()



                # Save rolling mean for training scores
                online_training_test.append(env.wallet)

                duration = np.array([order.end_date-order.start_date for order in env.orders])
                duration = np.mean(duration)

                print("Épisode :", episode, "Récompense totale :", env.wallet)
                print("nombre de position ouverte: ", len(env.orders), "Durée moyenne d'une position ouverte: ", duration)
                state = np.array([env.reset(initial_step=config['INITIAL_STEP'],pas=config['EPISODE_SIZE'])])
                break
            
            if len(agent.memory.buffer) > config['BATCH_SIZE']:
                 agent.replay()
        
    progress_bar = tqdm(range(config['EPISODE_SIZE'] * episode))


    env = TradingEnv(data=dataset,nb_action=config['NB_ACTION'], 
                window_size=config['WINDOW_SIZE'], 
                episode_size=config['EPISODE_SIZE'],
                initial_step='sequential',
                n=config['N_TEST'], 
                mode=mode, 
                reward_function=config['REWARD_FUNCTION'],
                wallet=config['WALLET'],
                zeta=config['ZETA'],
                beta=config['BETA'])

    agent.model = load_model(model_save_path)
    state = np.array([env.reset(initial_step=env.initial_step)])
    
    for episode in range(1, episodes + 1):
        
        while True:
            action = agent.act(state)
            next_state, reward, done, action, _ = env.step(action)
            next_state = np.array([next_state])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            progress_bar.update(1)

            if done:

                # Save rolling mean for training scores
                offline_training_test.append(env.wallet)

                duration = np.array([order.end_date-order.start_date for order in env.orders])
                duration = np.mean(duration)

                state = np.array([env.reset(initial_step=config['INITIAL_STEP'],pas=config['EPISODE_SIZE'])])
                break

    
    print('TEST completed and models saved.')

    # Save final scores

    score_save_path = os.path.join(run_folder, "offline_training_test.npy")
    np.save(score_save_path, offline_training_test)
    score_save_path = os.path.join(run_folder, "online_training_test.npy")
    np.save(score_save_path, online_training_test)
 

    # Plot results
    X = np.arange(1, episodes + 1)

    plt.figure()
    plt.plot(X, offline_training_test, label='offline Score')
    plt.plot(X, online_training_test, label='online Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.title(config['FIGURE_TITLE'])
    figure_save_path = os.path.join(run_folder, "test_scores_plot.png")
    plt.savefig(figure_save_path)
    plt.close()

    print('All configurations tested and results saved.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)
