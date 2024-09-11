@echo off
REM Set the PYTHONPATH to the project directory
set PYTHONPATH=%PYTHONPATH%;C:\Users\Ugo\Documents\AI\Forex_ML\RL

REM Run the MasterFinance script using the saved configuration
python -m RL.TestMasterFinance ^
    --config_path "C:\Users\Ugo\Documents\AI\Forex_ML\RL\RESULTS\config_20240911_0957\config.json" ^
    --online_training True ^
    --nb_agent 1


                            
