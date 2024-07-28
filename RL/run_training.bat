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
  --window_size 21 ^
  --episode_size 84 ^
  --nb_episode 10 ^
  --initial_step "sequential" ^
  --n_train 2 ^
  --n_test 1 ^
  --include_price False ^
  --include_historic_position False ^
  --include_historic_action False ^
  --include_historic_wallet False ^
  --reward_function "default" ^
  --lstm_layer 16 8 ^
  --epsilon 1 ^
  --epsilon_decay None ^
  --epsilon_min 0.01 ^
  --buffer_size 15000 ^
  --gamma 0.995 ^
  --batch_size 16 ^
  --iter_save_model_score 5 ^
  --iter_save_target_model 2 ^
  --iter_test 1 ^
  --figure_title "Values of portfolio function of episodes" ^
  --data_path "C:\Users\Ugo\Documents\AI\Forex_ML\RL\DATA\FAKE_DATA_TRAIN.csv"

REM Run the MasterFinance script using the saved configuration
python -m RL.MasterFinance --config_path "%SAVE_DIR%\config_%RUN_ID%\config.json"
