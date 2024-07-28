import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import argparse

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
    dataset = pd.read_csv(config['DATA_PATH']).to_numpy()

    # Initialize environments
    mode = {
        'include_price': config['MODE']['include_price'],
        'include_historic_position': config['MODE']['include_historic_position'],
        'include_historic_action': config['MODE']['include_historic_action'],
        'include_historic_wallet': config['MODE']['include_historic_wallet']
    }
    env = TradingEnv(data=dataset, window_size=config['WINDOW_SIZE'], episode_size=config['EPISODE_SIZE'], n=config['N_TRAIN'], mode=mode)
    env_test = TradingEnv(data=dataset, window_size=config['WINDOW_SIZE'], episode_size=config['EPISODE_SIZE'], n=config['N_TEST'], mode=mode)
    env.reset()

    # Initialize agent
    agent = DQNTrader(
        state_size=env.state_size,
        action_size=env.action_size,
        lstm_layer=config['LSTM_LAYER'],
        epsilon_decay=config['EPSILON_DECAY'],
        epsilon_min=config['EPSILON_MIN'],
        buffer_size=config['BUFFER_SIZE'],
        gamma=config['GAMMA']
    )

    # Training variables
    train_scores = []
    test_scores = []
    progress_bar = tqdm(range(config['EPISODE_SIZE'] * config['NB_EPISODE']))

    for episode in range(1, config['NB_EPISODE'] + 1):
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

                if episode % config['ITER_SAVE_TARGET_MODEL'] == 0:
                    agent.update_target_model()

                if episode % config['ITER_TEST'] == 0:
                    state_test = np.array([env_test.reset(env.current_step)])
                    done_test = False
                    while not done_test:
                        action_test = agent.act(state_test)
                        next_state_test, reward_test, done_test, action_test, info = env_test.step(action_test)
                        state_test = np.array([next_state_test])
                    test_scores.append(env_test.wallet)

                if episode % config['ITER_SAVE_MODEL_SCORE'] == 0:
                    model_save_path = os.path.join(run_folder, f"model_episode_{episode}.keras")
                    agent.target_model.save(model_save_path)
                    score_save_path = os.path.join(run_folder, f"train_scores_episode_{episode}.npy")
                    np.save(score_save_path, train_scores)

                train_scores.append(env.wallet)
                break

            if len(agent.memory.buffer) > config['BATCH_SIZE']:
                agent.replay(config['BATCH_SIZE'])

    print('Training completed and models saved.')

    # Save final scores
    score_save_path = os.path.join(run_folder, "train_scores_final.npy")
    np.save(score_save_path, train_scores)

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

    print('All configurations tested and results saved.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)
