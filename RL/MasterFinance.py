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
    dataset = data.norm_data
    dataset = data.norm_data[:len(data.norm_data)-(config['SPLIT'])]

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
                     initial_step=config["INITIAL_STEP"],
                     n=config['N_TRAIN'], 
                     mode=mode, 
                     reward_function=config['REWARD_FUNCTION'],
                     wallet=config['WALLET'],
                     zeta=config['ZETA'],
                     beta=config['BETA'])
    
    env_test = TradingEnv(data=dataset,
                          nb_action=config['NB_ACTION'], 
                          window_size=config['WINDOW_SIZE'], 
                          episode_size=config['EPISODE_SIZE'],
                          initial_step=config["INITIAL_STEP"], 
                          n=config['N_TEST'], mode=mode, 
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

    # Training variables
    REWARD = []
    train_scores = []
    test_scores = []
    rolling_train_scores = []
    rolling_test_scores = []
    order_duration = []
    nb_order = []
    X_rolling_train =[]
    X_rolling_test =[]
    progress_bar = tqdm(range(config['EPISODE_SIZE'] * config['NB_EPISODE']+(config['NB_EPISODE']//config['ITER_TEST'])*config['EPISODE_SIZE']))

    for episode in range(1, config['NB_EPISODE'] + 1):
        state = np.array([env.reset(initial_step=config['INITIAL_STEP'],pas=config['PAS'])])
        total_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, action, _ = env.step(action)
            next_state = np.array([next_state])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward
            progress_bar.update(1)

            if done:
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay

                if episode % config['ITER_SAVE_TARGET_MODEL'] == 0:
                    agent.update_target_model()

                if episode % config['ITER_TEST'] == 0:
                    state_test = np.array([env_test.reset(env.current_step,config['PAS'])])
                    done_test = False
                    while not done_test:
                        action_test = agent.act(state_test)
                        next_state_test, reward_test, done_test, action_test, info = env_test.step(action_test)
                        state_test = np.array([next_state_test])
                        progress_bar.update(1)
                    test_scores.append(env_test.wallet)

                    # Save rolling mean for test scores
                    if len(test_scores) >= 100:
                        X_rolling_test.append(episode)
                        rolling_test_scores.append(np.mean(test_scores[-2:]))
                
                if episode % config['ITER_SAVE_MODEL_SCORE'] == 0 or episode==1:
                    model_save_path = os.path.join(run_folder, f"model_episode_{episode}.keras")
                    agent.target_model.save(model_save_path)
                    score_save_path = os.path.join(run_folder, f"train_scores_episode_{episode}.npy")
                    np.save(score_save_path, train_scores)
                    video_save_path = os.path.join(run_folder,f'agent_trading_episode_{episode}.mp4')
                    env._render_agent_actions(video_save_path)

                # Save rolling mean for training scores
                train_scores.append(env.wallet)
                if len(train_scores) >= 100:
                    X_rolling_train.append(episode)
                    rolling_train_scores.append(np.mean(train_scores[-100:]))

                nb_order.append(len(env.orders))
                duration = np.array([order.end_date-order.start_date for order in env.orders])
                duration = np.mean(duration)
                order_duration.append(duration)
                REWARD.append(total_reward)
                print("Épisode :", episode, "Récompense totale :", env.wallet)
                print("nombre de position ouverte: ", len(env.orders), "Durée moyenne d'une position ouverte: ", duration)
                break
            
            if len(agent.memory.buffer) > config['BATCH_SIZE']:
                 agent.replay()

    print('Training completed and models saved.')

    # Save final scores
    model_save_path = os.path.join(run_folder, f"model_final.keras")
    agent.target_model.save(model_save_path)
    score_save_path = os.path.join(run_folder, "train_scores_final.npy")
    np.save(score_save_path, train_scores)
    score_save_path = os.path.join(run_folder, "test_scores_final.npy")
    np.save(score_save_path, test_scores)
    np.save(os.path.join(run_folder, "rolling_train_scores.npy"), rolling_train_scores)
    np.save(os.path.join(run_folder, "rolling_test_scores.npy"), rolling_test_scores)
    duration_save_path = os.path.join(run_folder, "duration_opened_position.npy")
    np.save(duration_save_path, order_duration)
    number_save_path = os.path.join(run_folder, "number_opened_position.npy")
    np.save(number_save_path, nb_order)

    # Plot results
    X = np.arange(1, config['NB_EPISODE'] + 1)
    X_test = np.array([i for i in range(1, config['NB_EPISODE'] + 1) if i % config['ITER_TEST'] == 0])

    plt.figure()
    plt.plot(X, train_scores, label='Training Score')
    plt.plot(X_test, test_scores, label='Test Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.title(config['FIGURE_TITLE'])
    figure_save_path = os.path.join(run_folder, "scores_plot.png")
    plt.savefig(figure_save_path)
    plt.close()

    # Plot rolling means

    
    plt.figure()
    plt.plot(X_rolling_train, rolling_train_scores, label='Rolling Mean Training Score')
    plt.plot(X_rolling_test, label='Rolling Mean Test Score')
    plt.xlabel('Episode')
    plt.ylabel('Rolling Mean Score (Window=10)')
    plt.legend()
    plt.title('Rolling Mean of Training and Test Scores')
    rolling_figure_save_path = os.path.join(run_folder, "rolling_mean_scores_plot.png")
    plt.savefig(rolling_figure_save_path)
    plt.close()

    print('All configurations tested and results saved.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)
