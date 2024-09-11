import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import argparse
import streamlit as st

from utils.tools import PrepareData
from utils.streamlit_utils import apply_css
from utils.AgentMasterFinance import DQNTrader
from utils.EnvMasterFinance import TradingEnv

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main(config,dataset,excluded_variable):
    st.success("The Training is launched")
    run_folder = os.path.join(config['SAVE_DIR'], f"config_{config['RUN_ID']}")
    
    dataset = dataset.drop(excluded_variable,axis=1).to_numpy()
    data = PrepareData(dataset)
    data.normalize()
    dataset = data.norm_data
    dataset = data.norm_data[:len(data.norm_data)-(config['SPLIT'])]


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
    
    
    progress_text = "Training in progress. Please wait."
    progress_bar = st.progress(0, text=progress_text)
    end_bar = config['EPISODE_SIZE'] * config['NB_EPISODE']+(config['NB_EPISODE']//config['ITER_TEST'])*config['EPISODE_SIZE']
    progress = 0

    bar = tqdm(range(config['EPISODE_SIZE'] * config['NB_EPISODE']+(config['NB_EPISODE']//config['ITER_TEST'])*config['EPISODE_SIZE']))

    for episode in range(1, config['NB_EPISODE'] + 1):

        if st.session_state.stop_requested:
            st.warning("Training has been stopped by the user.")
            st.session_state.is_running = False
            return
        
        state = np.array([env.reset(initial_step=config['INITIAL_STEP'],pas=config['PAS'])])
        total_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, action, _ = env.step(action)
            next_state = np.array([next_state])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward
            
            progress +=1
            progress_bar.progress(round(progress/end_bar,2))
            bar.update(1)
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
                        
                        progress +=1
                        progress_bar.progress(round(progress/end_bar,2))
                        bar.update(1)
                    test_scores.append(env_test.wallet)

                    # Save rolling mean for test scores
                    if len(test_scores) >= 100:
                        X_rolling_test.append(episode)
                        rolling_test_scores.append(np.mean(test_scores[-100:]))
                
                if episode % config['ITER_SAVE_MODEL_SCORE'] == 0 or episode==1:
                    model_save_path = os.path.join(run_folder, f"model_episode_{episode}.keras")
                    agent.target_model.save(model_save_path)
                    score_save_path = os.path.join(run_folder, f"train_scores_episode_{episode}.npy")
                    np.save(score_save_path, train_scores)
                    score_save_path = os.path.join(run_folder, f"test_scores_episode_{episode}.npy")
                    np.save(score_save_path, test_scores)
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

    st.success('Training completed and models saved.')
    
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
    plt.plot(X_rolling_test,rolling_test_scores, label='Rolling Mean Test Score')
    plt.xlabel('Episode')
    plt.ylabel('Rolling Mean Score (Window=100)')
    plt.legend()
    plt.title('Rolling Mean of Training and Test Scores')
    rolling_figure_save_path = os.path.join(run_folder, "rolling_mean_scores_plot.png")
    plt.savefig(rolling_figure_save_path)
    plt.close()

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Mean duration of an opened position', color='tab:blue')
    ax1.plot(X, order_duration, color='tab:blue', label='Mean duration of a opened position')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of opened position', color='tab:red')
    ax2.plot(X, nb_order, color='tab:red', label='Number of opened position')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('Mean Duration of Opened Positions and Number of Opened Positions per Episode')

    figure_save_path = os.path.join(run_folder, "time_order_plot.png")
    plt.savefig(figure_save_path)
    plt.close()

    st.success('All configurations tested and results saved.')
    st.session_state.is_running = False


apply_css()


st.markdown("<h1>Training Page</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:20px;'>This is the training page where you train the agent to trade on the forex markets.</p>", unsafe_allow_html=True)

load_setup = st.selectbox('Do you prefer to load an existing configuration or set up the Agent?', ["Load", "Set up"])


if load_setup == "Load":
    config_path = st.text_input("Config File Path", value="")
    if config_path:
        try:
            config = load_config(config_path)
            st.success('Config file loaded successfully!')
        except Exception as e:
            st.warning('Enter a valid Config File Path!!', icon="⚠️")
            config = None
    else:
        st.warning('Please provide a valid config file path.', icon="⚠️")
        config = None
else:
    config = st.session_state.get("config", None)


if config is None:
    st.warning('Load or Set up the Trading Environment and Agent!!', icon="⚠️")
else:

    st.markdown("<h2 style='font-size:30px;'>Configuration used for the training</h2>", unsafe_allow_html=True)
    st.session_state.config = config
    st.json(st.session_state.config)

    st.markdown("<h2 style='font-size:30px;'>Data used for the training</h2>", unsafe_allow_html=True)
    try:
        dataset = pd.read_csv(filepath_or_buffer=config['DATA_PATH'])
        st.dataframe(dataset)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        dataset = None

    if dataset is not None:
        keys = list(dataset.keys())
        st.session_state.options = st.multiselect(
            "What variables do you want to exclude?",
            keys
        )
    
    st.markdown("<h2 style='font-size:30px;'>Ready for the training</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Training"):
            st.session_state.is_running = True
            st.session_state.stop_requested = False 

    with col2:
        if st.button("Stop Training"):
            st.session_state.stop_requested = True

    if st.session_state.is_running:
        try:
            main(st.session_state.config,dataset, st.session_state.options)
        except Exception as e:
            st.error(e)
            st.session_state.is_running = False






    
