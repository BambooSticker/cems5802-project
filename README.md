# CHEN5802 Final Project: Adaptive Demand-Side Management using Reinforcement Learning and Model Predictive Control under Electricity Market Uncertainty

## Overview

Industrial demand-side management (DSM) presents significant economic opportunities for energy-intensive industries participating actively in electricity markets. In this project, the scheduling of a generic industrial process, represented as a flexible battery storage system, was modeled as a sequential decision-making problem and solved using reinforcement learning (RL). The RL agent was trained on day-ahead electricity price data from the German electricity spot market in 2017. To benchmark the RL agent’s per- formance, a model predictive control (MPC) approach was also developed, using a neural network for short-term price forecasting.

## Directory Structure

├── 0_time-series-analysis.ipynb
├── 1_agent-training.ipynb
├── 2_agent-test.ipynb
├── 3_deterministic-model-Julia.ipynb
├── 4_neural-network-for-mpc.ipynb
├── 5_mpc-Julia.ipynb
├── StorageEnvEffLos_rate.py
├── tools.py
├── data/
│   └── DayAheadSpotPrices.csv (MAIN)
│   └── PredictedPrices.csv
│   └── P_mpc.csv
│   └── S_mpc.csv
│   └── oracle.json
├── figures/
│   └── cost_comparison.png
│   └── result_comparison.png
│   └── price_comparison_mpc.png
├── model/
│   └── ppo_energy_storage.zip
├── report/
│   └── 25-Sp_CEMS-5802_project-report_Li-Nassaji-Janey.pdf
├── requirements.txt
└── README.md

## File and Directory Descriptions

- `0_time-series-analysis.ipynb`: Exploratory analysis of historical day-ahead electricity prices, including decomposition and visualization of seasonal and trend components.
- `1_agent-training.ipynb`: Notebook implementing and training the reinforcement learning (PPO) agent on processed price data.
- `2_agent-test.ipynb`: Evaluation notebook testing the trained RL agent’s performance against unseen price scenarios (A pre-trained model is stored in the folder `model`).
- `3_deterministic-model-Julia.ipynb`: Julia implementation of the deterministic Model Predictive Control (MPC) baseline, with price forecasting by neural network.
- `4_neural-network-for-mpc.ipynb`: Notebook showing training and validation of a neural network for short-term price forecasting used in the MPC loop.
- `5_mpc-Julia.ipynb`: Julia notebook integrating the neural network forecast into the MPC algorithm and simulating control performance.
- `StorageEnvEffLos_rate.py`: Python module defining the custom Gymnasium-compatible RL environment modeling storage dynamics and efficiency loss.
- `tools.py`: Utility functions for data preprocessing, result visualization, and metric calculations used across notebooks.
- `data/`: Directory containing raw and processed data files:
  - `DayAheadSpotPrices.csv`: Main historical price dataset.
  - `PredictedPrices.csv`, `P_mpc.csv`, `S_mpc.csv`, `oracle.json`: Files storing model forecasts and reference trajectories.
- `figures/`: Folder with output plots comparing RL and MPC results.
- `model/`: Saved RL model checkpoint for the energy storage environment.
- `requirements.txt`: List of Python dependencies required to reproduce the analysis.

## Authors

- Zhe Li
- Alexandra Janey
- Amin Nassaji