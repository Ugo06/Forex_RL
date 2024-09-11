@echo off
REM Set the PYTHONPATH to the project directory (make sure this is correct)
set PYTHONPATH=%PYTHONPATH%;C:\Users\Ugo\Documents\AI\Forex_ML\APPLICATION

REM Navigate to the directory where the Streamlit app (app.py) is located
cd /d C:\Users\Ugo\Documents\AI\Forex_ML\APPLICATION\frontend

REM Run the Streamlit app using the correct path
streamlit run app.py

REM Pause the command window to keep it open after execution
pause