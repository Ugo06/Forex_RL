#!/bin/bash

# Base directory for saving models, scores, and figures
SAVE_DIR="C:/Users/Ugo/Documents/AI/Forex_ML/RL/RESULTS"

# Generate a unique run ID based on the current timestamp
RUN_ID=$(date +"%Y%m%d%H%M%S")

# Run the config script to save the configuration
python -m RL.config \
  --run_id $RUN_ID \
  --save_dir $SAVE_DIR \
  --window_size 21 \
  --episode_size 84 \
  --nb_episode 200 \
  --initial_step -1 \
  --n_train 2 \
  --n_test 1 \
  --include_price False \
  --include_historic_position False \
  --include_historic_action False \
  --include_historic_wallet False \
  --lstm_layer 64 8 \
  --epsilon 1 \
  --epsilon_min 0.01 \
  --epsilon_decay $(python -c 'import config; print(config.EPSILON_MIN ** (1 / config.NB_EPISODE))') \
  --buffer_size 15000 \
  --gamma 0.995 \
  --batch_size 16 \
  --iter_save_model_score 25 \
  --iter_save_target_model 10 \
  --iter_test 1 \
  --figure_title "Values of portfolio function of episodes" \
  --data_path "C:/Users/Ugo/Documents/AI/Forex_ML/RL/DATA/FAKE_DATA_TRAIN.csv"

# Run the MasterFinance script using the saved configuration
python -m RL.MasterFinance --config_path "${SAVE_DIR}/config_${RUN_ID}/config.json"

