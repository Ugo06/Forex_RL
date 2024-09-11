import os
import ast
import json
import streamlit as st
from utils.streamlit_utils import apply_css  # Import the function

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def save_config(run_id, save_dir, nb_action, window_size, episode_size, nb_episode, initial_step, pas,
                include_price, include_historic_position, include_historic_action, include_historic_wallet,
                include_historic_orders, reward_function, zeta, beta, split, wallet, type_model, config_layer,
                epsilon, epsilon_min, epsilon_decay, buffer_size, gamma, alpha, batch_size, iter_save_model_score,
                iter_save_target_model, iter_test, figure_title, data_path):
    
    config = {
        "SAVE_DIR": save_dir,
        "RUN_ID": run_id,
        "NB_ACTION": nb_action,
        "WINDOW_SIZE": window_size,
        "EPISODE_SIZE": episode_size,
        "NB_EPISODE": nb_episode,
        "INITIAL_STEP": initial_step,
        "PAS": pas,
        "N_TRAIN": 2,
        "N_TEST": 1,
        "MODE": {
            'include_price': str2bool(include_price),
            'include_historic_position': str2bool(include_historic_position),
            'include_historic_action': str2bool(include_historic_action),
            'include_historic_wallet': str2bool(include_historic_wallet),
            'include_historic_orders': str2bool(include_historic_orders)
        },
        "WALLET": wallet,
        "REWARD_FUNCTION": reward_function,
        "ZETA": zeta,
        "BETA": beta,
        "SPLIT": split,
        "TYPE": type_model,
        "CONFIG_LAYER": ast.literal_eval(config_layer),
        "EPSILON": epsilon,
        "EPSILON_MIN": epsilon_min,
        "EPSILON_DECAY": epsilon_min ** (1 / nb_episode) if epsilon_decay == 'default' else float(epsilon_decay),
        "BUFFER_SIZE": buffer_size,
        "GAMMA": float(gamma),
        "ALPHA": float(alpha),
        "BATCH_SIZE": batch_size,
        "ITER_SAVE_MODEL_SCORE": iter_save_model_score,
        "ITER_SAVE_TARGET_MODEL": iter_save_target_model,
        "ITER_TEST": iter_test,
        "FIGURE_TITLE": figure_title,
        "DATA_PATH": data_path,
        "MODEL_SAVE_PATH": os.path.join(save_dir, f"config_{run_id}", 'model.keras'),
        "SCORE_SAVE_PATH": os.path.join(save_dir, f"config_{run_id}", 'scores.npy'),
        "FIGURE_PATH": os.path.join(save_dir, f"config_{run_id}", 'figure.png')
    }

    st.session_state.config = config
    
    run_dir = os.path.join(save_dir, f"config_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    st.success(f"Configuration saved to {config_path}")

apply_css()

st.markdown("<h1>Configuration Page</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:20px;'>This is the configuration page where you set up the environment and the agent.</p>", unsafe_allow_html=True)

config_path= st.text_input("Load an existing config.json", "")
st.markdown("<h1 style='font-size:30px;'>Configuration of save folder for the results</h1>", unsafe_allow_html=True)

print(config_path)
if config_path == "":
    run_id = st.text_input("Run ID", "001")
    save_dir = st.text_input("Save Directory", "C:/Users/Ugo/Documents/AI/Forex_ML/RL/RESULTS")
    data_path = st.text_input("Data File Path for the Training", "C:/Users/Ugo/Documents/AI/Forex_ML/RL/DATA/FAKE_DATA_TRAIN.csv")

    st.markdown("<h1 style='font-size:30px;'>Configuration of Trading Environment</h1>", unsafe_allow_html=True)

    st.markdown("<h1 style='font-size:20px;'>Initialisation Parameter</h1>", unsafe_allow_html=True)

    nb_action = st.number_input("Number of Actions", min_value=1, value=3)
    window_size = st.number_input("Window Size", min_value=1, value=21)
    episode_size = st.number_input("Episode Size", min_value=1, value=84)
    nb_episode = st.number_input("Number of Episodes", min_value=1, value=200)
    initial_step = st.text_input("Initial Step (random or sequential)", "random")
    pas = st.number_input("Step", min_value=1, value=1)

    include_price = st.selectbox("Include Price", ["True", "False"])
    include_historic_position = st.selectbox("Include Historic Position", ["True", "False"])
    include_historic_action = st.selectbox("Include Historic Action", ["True", "False"])
    include_historic_wallet = st.selectbox("Include Historic Wallet", ["True", "False"])
    include_historic_orders = st.selectbox("Include Historic Orders", ["True", "False"])

    st.markdown("<h1 style='font-size:20px;'>Initialisation of Reward Function</h1>", unsafe_allow_html=True)
    reward_function = st.text_input("Reward Function", "default")
    zeta = st.number_input("Zeta", min_value=0.0, value=1.0)
    beta = st.number_input("Beta", min_value=0.0, value=1.0)
    split = st.number_input("Number of step to test the Agent", min_value=1, value=250)
    wallet = st.number_input("Value of the Initial Wallet", min_value=0, value=0)


    st.markdown("<h1 style='font-size:30px;'>Configuration of Trading Agent</h1>", unsafe_allow_html=True)

    type_model = st.selectbox("Model Type", ["lstm"])
    config_layer = st.text_input("Layer Configuration", "[64, 8]")
    epsilon = st.number_input("Epsilon", min_value=0.0, value=1.0)
    epsilon_min = st.number_input("Minimum Epsilon", min_value=0.0, value=0.01)
    epsilon_decay = st.text_input("Epsilon Decay", "default")
    buffer_size = st.number_input("Buffer Size", min_value=500, value=15000)
    gamma = st.text_input("Gamma", "0.995")
    alpha = st.text_input("Alpha", "0.001")
    batch_size = st.number_input("Batch Size", min_value=1, value=16)
    iter_save_model_score = st.number_input("Model Save Interval", min_value=1, value=25)
    iter_save_target_model = st.number_input("Target Model Save Interval", min_value=1, value=10)
    iter_test = st.number_input("Test Interval", min_value=1, value=1)
    figure_title = st.text_input("Figure Title", "Values of portfolio function of episodes")

else:
    config = load_config(config_path)

    run_id = st.text_input("Run ID", "001")
    save_dir = st.text_input("Save Directory", "C:/Users/Ugo/Documents/AI/Forex_ML/RL/RESULTS")
    data_path = st.text_input("Data File Path for the Training", config["DATA_PATH"])

    st.markdown("<h1 style='font-size:30px;'>Configuration of Trading Environment</h1>", unsafe_allow_html=True)

    st.markdown("<h1 style='font-size:20px;'>Initialisation Parameter</h1>", unsafe_allow_html=True)

    nb_action = st.number_input("Number of Actions", min_value=1, value=config["NB_ACTION"])
    window_size = st.number_input("Window Size", min_value=1, value=config["WINDOW_SIZE"])
    episode_size = st.number_input("Episode Size", min_value=1, value=config["EPISODE_SIZE"])
    nb_episode = st.number_input("Number of Episodes", min_value=1, value=config["NB_EPISODE"])
    initial_step = st.text_input("Initial Step (random or sequential)", config["INITIAL_STEP"])
    pas = st.number_input("Step", min_value=1, value=config["PAS"])

    include_price = st.selectbox("Include Price", ["True", "False"])
    include_historic_position = st.selectbox("Include Historic Position", ["True", "False"])
    include_historic_action = st.selectbox("Include Historic Action", ["True", "False"])
    include_historic_wallet = st.selectbox("Include Historic Wallet", ["True", "False"])
    include_historic_orders = st.selectbox("Include Historic Orders", ["True", "False"])

    st.markdown("<h1 style='font-size:20px;'>Initialisation of Reward Function</h1>", unsafe_allow_html=True)
    reward_function = st.text_input("Reward Function", config["REWARD_FUNCTION"])
    zeta = st.number_input("Zeta", min_value=0.0, value=config["ZETA"])
    beta = st.number_input("Beta", min_value=0.0, value=config["BETA"])
    split = st.number_input("Number of step to test the Agent", min_value=1, value=config["SPLIT"])
    wallet = st.number_input("Value of the Initial Wallet", min_value=0, value=config["WALLET"])


    st.markdown("<h1 style='font-size:30px;'>Configuration of Trading Agent</h1>", unsafe_allow_html=True)

    type_model = st.selectbox("Model Type", ["lstm"])
    config_layer = st.text_input("Layer Configuration", str(config["CONFIG_LAYER"]))
    epsilon = st.number_input("Epsilon", min_value=0.0, value=config["EPSILON"])
    epsilon_min = st.number_input("Minimum Epsilon", min_value=0.0, value=config["EPSILON_MIN"])
    epsilon_decay = st.text_input("Epsilon Decay",config["EPSILON_DECAY"])
    buffer_size = st.number_input("Buffer Size", min_value=1, value=config["BUFFER_SIZE"])
    gamma = st.text_input("Gamma", str(config["GAMMA"]))
    alpha = st.text_input("Alpha", str(config["ALPHA"]))
    batch_size = st.number_input("Batch Size", min_value=1, value=config["BATCH_SIZE"])
    iter_save_model_score = st.number_input("Model Save Interval", min_value=1, value=config["ITER_SAVE_MODEL_SCORE"])
    iter_save_target_model = st.number_input("Target Model Save Interval", min_value=1, value=config["ITER_SAVE_TARGET_MODEL"])
    iter_test = st.number_input("Test Interval", min_value=1, value=config["ITER_TEST"])
    figure_title = st.text_input("Figure Title", config["FIGURE_TITLE"])

if st.button("Sauvegarder la configuration"):
    save_config(run_id, save_dir, nb_action, window_size, episode_size, nb_episode, initial_step, pas,
                include_price, include_historic_position, include_historic_action, include_historic_wallet,
                include_historic_orders, reward_function, zeta, beta, split, wallet, type_model, config_layer,
                epsilon, epsilon_min, epsilon_decay, buffer_size, gamma, alpha, batch_size, iter_save_model_score,
                iter_save_target_model, iter_test, figure_title, data_path)