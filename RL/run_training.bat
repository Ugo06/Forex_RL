@echo off

REM Define variables
set SAVE_DIR=C:\Users\Ugo\Documents\AI\Forex_ML\RL\RESULTS

REM Generate a unique run ID based on the current timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"

set "RUN_ID=%YYYY%%MM%%DD%_%HH%%Min%"

REM Create the run directory
mkdir "%SAVE_DIR%\config_%RUN_ID%"

REM Set the PYTHONPATH to the project directory
set PYTHONPATH=%PYTHONPATH%;C:\Users\Ugo\Documents\AI\Forex_ML\RL

REM Run the config script to save the configuration
python -m RL.config ^
  --run_id %RUN_ID% ^
  --save_dir %SAVE_DIR% ^
  --nb_action 2 ^
  --window_size 40 ^
  --episode_size 90 ^
  --nb_episode 200 ^
  --initial_step "sequential" ^
  --pas 5 ^
  --n_train 2 ^
  --n_test 1 ^
  --include_price "True" ^
  --include_historic_position "True" ^
  --include_historic_action "False" ^
  --include_historic_wallet "False" ^
  --include_historic_orders "True" ^
  --wallet 0 ^
  --reward_function "sharpe_ratio" ^
  --zeta 1 ^
  --beta 1 ^
  --type "lstm"^
  --config_layer "[64,32]" ^
  --epsilon 1 ^
  --epsilon_decay "default" ^
  --epsilon_min 0.01 ^
  --buffer_size 5000 ^
  --gamma 0.99 ^
  --alpha 1e-3 ^
  --batch_size 64 ^
  --iter_save_model_score 25 ^
  --iter_save_target_model 10 ^
  --iter_test 4 ^
  --figure_title "Values of portfolio function of episodes" ^
  --data_path "C:/Users/Ugo/Documents/AI/Forex_ML/RL/DATA/DATASET_VI_SMA.csv"

REM Run the MasterFinance script using the saved configuration
python -m RL.MasterFinance --config_path "%SAVE_DIR%\config_%RUN_ID%\config.json"
