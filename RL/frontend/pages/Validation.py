import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from tqdm.auto import tqdm
import argparse

from utils.tools import PrepareData
from utils.AgentMasterFinance import DQNTrader
from utils.EnvMasterFinance import TradingEnv

from tensorflow.keras.models import load_model
from utils.streamlit_utils import apply_css

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise st.error('Boolean value expected.')
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main(config,dataset,excluded_variable,online_training,nb_agent,num_backup):
    # Create the run directory
    run_folder = os.path.join(config['SAVE_DIR'], f"config_{config['RUN_ID']}")
    
    # Load dataset
    dataset = dataset.drop(excluded_variable,axis=1).to_numpy()
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
        model_save_path = os.path.join(run_folder, f"model_episode_{num_backup}.keras")
        agent.model = load_model(model_save_path)
        AGENT.append(agent)
    
    # Training variables
    
    online_training_test = []
    offline_training_test = []

    mean_online = []
    mean_offline = []
    
    if online_training:
        
        progress_text = "Testing in progress with online training. Please wait."
        progress_bar = st.progress(0, text=progress_text)
        end_bar = episodes
        progress = 0
        
        
        bar = tqdm(range(episodes))
        state = np.array([env.reset(initial_step=env.initial_step)])
        
        for episode in range(1, episodes+1):
            if st.session_state.stop_requested:
                st.warning("Validation has been stopped by the user.")
                st.session_state.is_running = False
                st.session_state.is_finished= False
                return
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
            bar.update(1)
            progress +=1
            progress_bar.progress(round(progress/end_bar,2))

            online_training_test.append(env.wallet)

            if len(online_training_test)>=50:
                mean_online.append(np.mean(online_training_test[-50:]))
            
            if episode % config['ITER_SAVE_TARGET_MODEL'] == 0:
                    agent.update_target_model()
            
    
            if len(agent.memory.buffer)>config['BATCH_SIZE']:
                agent.replay()

        video_save_path = os.path.join(run_folder,f'online_training_validation_agent.mp4')
        env._render_agent_actions(video_save_path)
    
    bar = tqdm(range(episodes))
    progress_text = "Testing in progress with offline training. Please wait."
    progress_bar = st.progress(0, text=progress_text)
    end_bar = episodes
    progress = 0
    
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
        if st.session_state.stop_requested:
            st.warning("Validation has been stopped by the user.")
            st.session_state.is_running = False
            st.session_state.is_finished= False
            return
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

        bar.update(1)
        progress +=1
        progress_bar.progress(round(progress/end_bar,2))

        offline_training_test.append(env.wallet)
        if len(offline_training_test)>=50:
            mean_offline.append(np.mean(offline_training_test[-50:]))
        if done:
            break

    video_save_path = os.path.join(run_folder,f'offline_training_validation_agent.mp4')
    env._render_agent_actions(video_save_path)


    
    st.success('TEST completed and models saved.')

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

    st.success('All configurations tested and results saved.')
    st.session_state.is_running = False
    st.session_state.is_finished = True


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

if "is_finished" not in st.session_state:
    st.session_state.is_finished = False

if "is_trained" not in st.session_state:
    st.session_state.is_trained=False

if "options" not in st.session_state:
    st.session_state.options = []


apply_css()

st.markdown("<h1>Validation</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:20px;'>Here you can load your models and test them on your environment or other data.</p>", unsafe_allow_html=True)

load_setup = st.selectbox('Do you prefer to load an existing agent or use the Agent that you trained?', ["Load", "Use the trained Agent"])
if load_setup == "Load":
    config_path = st.text_input("Config File Path of the Agent", value="")
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
    if "config" not in st.session_state:
        st.warning('Set up and train the agent!!', icon="⚠️")
        config = None
    else:
        config = st.session_state.config
        if st.session_state.is_trained:
            st.success('Config file loaded successfully!')
        else:
            st.warning('Train the Agent!!', icon="⚠️")
            config = None

if config or st.session_state.is_trained:
    
    st.markdown("<h1 style='font-size:30px;'>Configuration used to train your Agent</h1>", unsafe_allow_html=True)
    st.json(config)

    st.markdown("<h1 style='font-size:30px;'>Data used for the training</h1>", unsafe_allow_html=True)
    try:
        dataset = pd.read_csv(filepath_or_buffer=config['DATA_PATH'])
        st.dataframe(dataset)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        dataset = None

    if dataset is not None:
        keys = list(dataset.keys())
        options = st.multiselect(f"What variables do you exclude when you trained the Agent ID?",keys,st.session_state.options)
    
    
    st.markdown("<h1 style='font-size:20px;'>You have the possibility to use bagging methods to validate your agent.</h1>", unsafe_allow_html=True)
    nb_agent = st.number_input("How many Agent do you want?",min_value=1,value=1)

    st.markdown("<h1 style='font-size:20px;'>You have the possibility to test the Agent with an online training.</h1>", unsafe_allow_html=True)
    online_training = str2bool(st.selectbox("Do you want an online training?",["No","Yes"]))
    
    num_backup = st.slider("Choice the numero of the model backup",min_value=0,max_value=config['NB_EPISODE'],step=config['ITER_SAVE_MODEL_SCORE'])
    if num_backup == 0:
        num_backup = 1
    st.markdown("<h1 style='font-size:30px;'>Ready for the Validation</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Validation"):
            st.session_state.is_running = True
            st.session_state.is_finished = False
            st.session_state.stop_requested = False 

    with col2:
        if st.button("Stop Validation"):
            st.session_state.stop_requested = True

    if st.session_state.is_running:
        try:
            main(config,dataset,options,online_training,nb_agent,num_backup)
        except Exception as e:
            st.error(e)
            st.session_state.is_running = False
            st.session_state.is_finished = False
        
        if st.session_state.is_finished:
            st.markdown("<h1 style='font-size:30px;'>Results</h1>",unsafe_allow_html=True)
            run_folder = os.path.join(config['SAVE_DIR'], f"config_{config['RUN_ID']}")

            path = os.path.join(run_folder, "validation_scores_plot.png")
            st.image(path)

            path = os.path.join(run_folder, "mean_validation_scores_plot.png")
            st.image(path)

            path=os.path.join(run_folder,'offline_training_validation_agent.mp4')
            st.video(path, format="video/mp4")

            if online_training:
                path=os.path.join(run_folder,'online_training_validation_agent.mp4')
                st.video(path, format="video/mp4")
                

