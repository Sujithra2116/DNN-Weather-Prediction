# Weather Prediction Using Deep Neural Networks (DNN)

A beginner-friendly Python mini project that estimates next-day temperature from weather measurements using a feed-forward neural network.

> **Important limitation:** This starter project generates synthetic data so it can run offline without API keys or downloads. Its metrics and predictions are for learning only, not real weather forecasting. For a real project, replace the demo CSV with verified historical observations from a chosen location and evaluate on a later time period.

## Features
- Generates a reproducible demo dataset automatically.
- Trains a multilayer neural network with `scikit-learn`'s `MLPRegressor`.
- Uses a chronological train/test split.
- Scales the input and target variables.
- Reports MAE, RMSE, and R².
- Saves actual-vs-predicted and training-loss graphs.
- Saves test predictions to a CSV file.
- Accepts weather inputs and estimates the next-day temperature.
- Saves the model and scalers.

## Requirements
- Python 3.10 or newer
- VS Code and the Python extension
- Git (to upload the project to GitHub)

## Run in VS Code on Windows

1. Download and extract the project ZIP.
2. Open the extracted `DNN_Weather_Prediction_Mini_Project` folder in VS Code.
3. Select **Terminal → New Terminal**.
4. Create a virtual environment:

   ```powershell
   py -m venv .venv
   ```

5. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, select Command Prompt as the VS Code terminal and run:

   ```bat
   .venv\Scripts\activate.bat
   ```

6. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Run the program:

   ```powershell
   python app.py
   ```

On first run, the app creates the synthetic CSV, trains the model, prints evaluation metrics, creates graphs, and asks you for current weather measurements.

## Project structure

```text
DNN_Weather_Prediction_Mini_Project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── historical_weather.csv   (generated on first run)
├── model/
│   └── model files              (generated on first run)
└── outputs/
    ├── actual_vs_predicted.png
    ├── training_loss.png
    └── prediction_results.csv
```

## Model details
- Task: regression
- Input features: temperature, humidity, pressure, wind speed
- Hidden layers: 64, 32, and 16 neurons
- Activation: ReLU
- Optimizer: Adam
- Output: estimated next-day temperature in °C
- Metrics: MAE, RMSE, and R²

## Upload to GitHub

1. Sign in to GitHub and create a new repository named `DNN-Weather-Prediction`.
2. If you will push this existing project, do not initialize the new repository with another README.
3. Open the VS Code terminal in the project folder and run:

   ```powershell
   git init
   git add .
   git commit -m "Add DNN weather prediction mini project"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/DNN-Weather-Prediction.git
   git push -u origin main
   ```

4. Replace `YOUR-USERNAME` with your GitHub username.
5. Complete the browser sign-in prompt if Git asks you to authenticate.

If `git` is not recognized, install Git for Windows, restart VS Code, and try again.

## Troubleshooting
- **`No module named ...`**: activate `.venv`, then run `pip install -r requirements.txt`.
- **`git is not recognized`**: install Git for Windows and restart VS Code.
- **PowerShell activation blocked**: use the Command Prompt activation command above.
- **Git push authentication error**: sign in through the browser prompt or Git Credential Manager.

## Viva questions
1. What is a Deep Neural Network?
2. Is this classification or regression, and why?
3. What is the purpose of hidden layers?
4. Why are input features scaled?
5. What is the ReLU activation function?
6. What are MAE and RMSE?
7. Why is the dataset split chronologically?
8. Why is synthetic data insufficient for real forecasting?
9. What is overfitting?
10. How could verified historical weather data improve the project?
