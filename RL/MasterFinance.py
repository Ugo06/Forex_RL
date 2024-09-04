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
    dataset = pd.read_csv(filepath_or_buffer=config['DATA_PATH'],index_col=0).to_numpy()
    data = PrepareData(dataset)
    print(pd.DataFrame(data.data).head())
    data.normalize()
    dataset = data.norm_data

    # Initialize environments
    mode = {
        'include_price': config['MODE']['include_price'],
        'include_historic_position': config['MODE']['include_historic_position'],
        'include_historic_action': config['MODE']['include_historic_action'],
        'include_historic_wallet': config['MODE']['include_historic_wallet'],
        'include_historic_orders': config['MODE']['include_historic_orders'],
    }
    env = TradingEnv(data=dataset,nb_action=config['NB_ACTION'], window_size=config['WINDOW_SIZE'], episode_size=config['EPISODE_SIZE'], n=config['N_TRAIN'], mode=mode, reward_function=config['REWARD_FUNCTION'],wallet=config['WALLET'],zeta=config['ZETA'],beta=config['BETA'])
    env_test = TradingEnv(data=dataset,nb_action=config['NB_ACTION'], window_size=config['WINDOW_SIZE'], episode_size=config['EPISODE_SIZE'], n=config['N_TEST'], mode=mode, reward_function=config['REWARD_FUNCTION'],wallet=config['WALLET'],zeta=config['ZETA'],beta=config['BETA'])

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
    order_duration = []
    nb_order = []
    progress_bar = tqdm(range(config['EPISODE_SIZE'] * config['NB_EPISODE']))

    for episode in range(1, config['NB_EPISODE'] + 1):
        state = np.array([env.reset()])
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
                nb_order.append(len(env.orders))
                duration = np.array([order.end_date-order.start_date for order in env.orders])
                duration = np.mean(duration)
                order_duration.append(duration)
                REWARD.append(total_reward)
                print("Épisode :", episode,"Récompense totale :", env.wallet)
                print("nombre de position ouverte: ",len(env.orders),"Durée moyenne d'une position ouverte: ",duration)
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

    plt.figure()
    plt.plot(REWARD)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Evolution of reward during training')
    figure_save_path = os.path.join(run_folder, "reward_plot.png")
    plt.savefig(figure_save_path)
    plt.close()

    fig, ax1 = plt.subplots()

    # Tracer order_duration sur l'axe y de gauche
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Mean duration of an opened position', color='tab:blue')
    ax1.plot(X, order_duration, color='tab:blue', label='Mean duration of a opened position')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Créer un second axe y pour nb_order
    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of opened position', color='tab:red')
    ax2.plot(X, nb_order, color='tab:red', label='Number of opened position')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Ajouter la légende pour les deux axes
    fig.tight_layout()  # Pour ajuster automatiquement la disposition
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # Ajouter un titre
    plt.title('Mean Duration of Opened Positions and Number of Opened Positions per Episode')

    # Sauvegarder la figure
    figure_save_path = os.path.join(run_folder, "time_order_plot.png")
    plt.savefig(figure_save_path)
    plt.close()

    print('All configurations tested and results saved.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)
