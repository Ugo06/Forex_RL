# **Reinforcement Learning-Based Trading Algorithm for Analyzing Macroeconomic Impact on EUR/USD Exchange Rate**

## **Description:**

This project is an introduction to quantitative finance, aiming to explore the influence of macroeconomic indicators on the EUR/USD exchange rate. The exchange rate between the Euro and the US Dollar is influenced by several external factors, including monetary policies, political situations, economic health, and conflicts in each region. Macroeconomic indicators from the Eurozone and the United States, which reflect the economic health of these regions, play a crucial role.

We hypothesize that the value of the EUR/USD exchange rate can be expressed as a weighted sum of different economic indicators. Specifically, we assume that the price of EUR/USD at time $`t+T`$ (where $`T`$ represents a certain time horizon) can be approximated by the price at time $`t`$ , plus a weighted sum of the values of indicators $`I`$ at time $`t`$, composed by a function $`f`$, as represented by:

$$
P_{t+T} = P_t + \sum_{i=1}^n w_i f_i(I_i(t)) + \epsilon
$$

Where:
- $`P_{t+T}`$ is the price of EUR/USD at time $`t+T`$, where $`T`$ represents a specific time horizon,
- $`P_t`$ is the price of EUR/USD at time $`t`$,
- $`I_i(t)`$ represents the value of the $`i`$-th macroeconomic indicator at time $`t`$,
- $`f_i`$ represents a transformation function, which can be normalization, logarithmic transformation, differentiation, etc.,
- $`w_i`$ is the weight associated with the indicator $`I_i(t)`$,
- $`\epsilon`$ is an error term.

If this hypothesis holds true, it will allow the prediction of the future price of EUR/USD, which is useful for options pricing or portfolio management. The objective of this study is to verify the validity of this hypothesis by analyzing the relationship between macroeconomic indicators and the EUR/USD exchange rate.

To achieve this, we will:
1. Build a dataset with the price of EUR/USD and several key macroeconomic indicators.
2. Analyze the correlation between these indicators and the EUR/USD price.
3. Develop predictive Reinforcement predictive models based on these indicators to estimate the future price of EUR/USD.
4. Test and validate these models with real historical data.

We also focus on developing a trading algorithm based on the described hypothesis. This algorithm will make use of reinforcement learning and other machine learning techniques to make real-time trading decisions, leveraging the relationship between macroeconomic indicators and the EUR/USD price. The trading strategy will be backtested with historical data and evaluated on performance metrics such as profit, risk-adjusted returns, and volatility.

## **Project Structure:**

```plaintext
|- FOREX_ML
   |- DATA
   |- RL
      |- frontend
           |- pages
               |- Config.py
               |- Learning.py
               |- Results.py
               |- Validation.py
           |- Welcome.py
      |- RESULTS
      |- utils
           |- __pycache__
           |- __init__.py
           |- AgentMasterFinance.py
           |- EnvMasterFinance.py
           |- streamlit_utils.py
           |- tools.py
      |- Benchmark.py
      |- config.py
      |- FakeMarket.py
      |- MasterFinance.py
      |- run_app.bat
      |- run_test.bat
      |- run_training.bat
      |- run_training.sh
      |- test.ipynb
      |- TestMasterFinance.py
   |- Projet_Quant.docx
   |- Projet_Quant.ipynb
   |- README.md
   |- requirements.txt
```
## **Installation:**
To run the project locally, follow the steps below:

```
git clone https://github.com/your-username/forex_ml.git
pip install -r requirements.txt
```
## **Usage:**
This project is composed of several components, each serving a specific purpose for managing, testing, training, and running the trading algorithm based on macroeconomic indicators. Below is an explanation of the key files and directories in the project:

### **Key Files and Directories:**
'DATA/': This directory stores the data of the real Forex Market to train the Agent. Use 'DATA/DATASET_3/DATASET_IV.csv' to train your agent.

'RESULTS/': This directory stores the results generated from tests, simulations, or model outputs. After running training or testing scripts, the resulting data will be stored here.

`utils/`: This folder contains utility scripts and helper functions that are used across the project. These files are shared between different parts of the project to avoid redundancy.

`Benchmark.py`: This script is used for performance benchmarking, helping to evaluate the performance of the trading algorithms or models based on certain criteria.

`config.py`: This configuration file contains the settings and parameters for the project, such as data paths, model configurations, and environment settings. It ensures that all scripts use the same settings consistently.

`FakeMarket.py`: This script simulates a market environment. It's used for testing and training the trading algorithm in a controlled environment, possibly using reinforcement learning or financial modeling techniques.

`MasterFinance.py`: This is the main script that handles all the financial logic and processes in the project. It likely orchestrates data processing, model execution, and interactions with the simulated market or real financial data.

`run_app.bat`: This batch file is used to run the application. It automates the process of starting the frontend or other necessary services.

`run_test.bat`: This batch file is designed to run tests for the project. It executes unit or integration tests to ensure the code is functioning correctly.

`run_training.bat`: This batch file triggers the training process on Windows. It likely runs the training scripts for the machine learning models or the trading algorithm.

`run_training.sh`: This shell script is used for training on Linux or Mac systems. Similar to run_training.bat, it starts the model training process.

`test.ipynb`: This Jupyter notebook is used for testing code, models, or specific functionalities. It provides an interactive environment for testing different parts of the project.

`TestMasterFinance.py`: This Python file contains unit or integration tests for the MasterFinance.py script. It ensures that the financial logic and processes work as expected.

`Projet_Quant.docx`: A Word document that describes the project in detail. It may be used for documentation or presentation purposes.

`Projet_Quant.ipynb`: A Jupyter notebook explaining the project. This notebook likely includes the steps for building and testing models, along with visualizations and explanations of the experiments.

Frontend Components:
`frontend/`: This directory contains the components for the frontend (user interface). It allows you to visualize results and interact with the system, possibly through a web-based dashboard.

`pages/`: This folder contains Python files related to individual pages of the frontend.

`Config.py`: This file handles the configuration settings for the frontend or individual pages.

`Learning.py`: Contains the logic for learning processes, such as the training of models or agents.

`Results.py`: Manages the display or processing of results in the frontend.

`Validation.py`: Handles validation logic for models or input data, ensuring the accuracy and consistency of results.

### **Running the Project:**
1. Running the Training Script:
To train the model or algorithm, use one of the following commands depending on your operating system:

On Windows:

Open a terminal and run the following command:

`./run_training.bat`

2. Running the Tests:
To run the tests and verify that all components are working correctly, use the following command:

On Windows:

Open a terminal and run:

`./run_test.bat`

This will execute the tests, including those in TestMasterFinance.py, and provide feedback on whether the code is functioning as expected.

3. Running the Application:
To launch the frontend application and interact with the project through a graphical interface, use the following command:

On Windows:

Open a terminal and run:

`./run_app.bat`

This will start the Streamlit-based user interface, allowing you to interact with the models, view results, and adjust settings through a web browser.


## **License:**

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial** — You may not use the material for commercial purposes.

For full license details, please visit [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

