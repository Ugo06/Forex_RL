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

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main(config,online_training,nb_agent):
    # Create the run directory
    run_folder = os.path.join(config['SAVE_DIR'], f"config_{config['RUN_ID']}")
    
    # Load dataset
    dataset = pd.read_csv(filepath_or_buffer=config['DATA_PATH'])
    dataset = dataset.drop(['OPEN','SMA_5','SMA_50'],axis=1).to_numpy()
    data = PrepareData(dataset)
    print(pd.DataFrame(data.data).head())
    data.normalize()
    dataset = data.norm_data[len(data.norm_data)-(config['SPLIT']+config["WINDOW_SIZE"]+config["N_TRAIN"]*config['EPISODE_SIZE']+1):]
    episodes = config['SPLIT']+config["WINDOW_SIZE"]+config["N_TEST"]*config['EPISODE_SIZE']
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
                     episode_size=episodes,
                     initial_step='sequential',
                     n=0, 
                     mode=mode, 
                     reward_function=config['REWARD_FUNCTION'],
                     wallet=config['WALLET'],
                     zeta=config['ZETA'],
                     beta=config['BETA'])


    AGENT = []
    for _ in range(nb_agent):
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
        agent.epsilon = config['EPSILON_MIN']
        model_save_path = os.path.join(run_folder, "model_final.keras")
        agent.model = load_model(model_save_path)
        AGENT.append(agent)
    
    # Training variables
    
    online_training_test = []
    offline_training_test = []

    mean_online = []
    mean_offline = []
    
    if online_training:
        progress_bar = tqdm(range(episodes))
        state = np.array([env.reset(initial_step=env.initial_step)])
        
        for episode in range(1, episodes+1):
            buy = 0
            sell = 0
            for agent in AGENT:
                action = agent.act(state)
                if action == 0:
                    buy+=1
                elif action ==1 :
                    sell+=1
            if buy >= sell:
                action =0
            else:
                action=1
            
            next_state, reward, done, action, _ = env.step(action)
            next_state = np.array([next_state])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            progress_bar.update(1)

            online_training_test.append(env.wallet)

            if len(online_training_test)>=50:
                mean_online.append(np.mean(online_training_test[-50:]))
            
            if episode % config['ITER_SAVE_TARGET_MODEL'] == 0:
                    agent.update_target_model()
            
    
            if len(agent.memory.buffer)>config['BATCH_SIZE']:
                agent.replay()

        video_save_path = os.path.join(run_folder,f'online_training_validation_agent.mp4')
        env._render_agent_actions(video_save_path)
    
    progress_bar = tqdm(range(episodes))
    
    env = TradingEnv(data=dataset,nb_action=config['NB_ACTION'], 
                window_size=config['WINDOW_SIZE'], 
                episode_size=episodes,
                initial_step='sequential',
                n=0, 
                mode=mode, 
                reward_function=config['REWARD_FUNCTION'],
                wallet=config['WALLET'],
                zeta=config['ZETA'],
                beta=config['BETA'])

    agent.model = load_model(model_save_path)
    state = np.array([env.reset(initial_step=env.initial_step)])
    
    for episode in range(1, episodes+1):
        buy = 0
        sell = 0
        for agent in AGENT:
            action = agent.act(state)
            if action == 0:
                buy+=1
            elif action ==1 :
                sell+=1
        if buy >= sell:
            action =0
        else:
            action=1
        
        next_state, reward, done, action, _ = env.step(action)
        next_state = np.array([next_state])
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        progress_bar.update(1)

        offline_training_test.append(env.wallet)
        if len(offline_training_test)>=50:
            mean_offline.append(np.mean(offline_training_test[-50:]))
        if done:
            break

    video_save_path = os.path.join(run_folder,f'offline_training_validation_agent.mp4')
    env._render_agent_actions(video_save_path)


    
    print('TEST completed and models saved.')

    # Save final scores

    score_save_path = os.path.join(run_folder, "offline_training_vaidation.npy")
    np.save(score_save_path, offline_training_test)
    if online_training:
        score_save_path = os.path.join(run_folder, "online_training_validation.npy")
        np.save(score_save_path, online_training_test)
 

    # Plot results

    plt.figure()
    plt.plot(offline_training_test, label='offline Score')
    if online_training:
        plt.plot(online_training_test, label='online Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.title(config['FIGURE_TITLE'])
    figure_save_path = os.path.join(run_folder, "validation_scores_plot.png")
    plt.savefig(figure_save_path)
    plt.close()

    plt.figure()
    plt.plot(mean_offline, label='offline Score')
    if online_training:
        plt.plot(mean_online, label='online Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.title(config['FIGURE_TITLE'])
    figure_save_path = os.path.join(run_folder, "mean_validation_scores_plot.png")
    plt.savefig(figure_save_path)
    plt.close()

    print('All configurations tested and results saved.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--online_training', type=str2bool, nargs='?', const=True, default=True, help="Enable or disable online training (True/False)")
    parser.add_argument('--nb_agent', type=int, required=True, default=1)
    args = parser.parse_args()
    
    config = load_config(args.config_path)
    online_training = args.online_training
    nb_agent = args.nb_agent
    main(config, online_training,nb_agent)
