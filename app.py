import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
MODEL_DIR = "model"
DATA_PATH = os.path.join(DATA_DIR, "historical_weather.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "weather_dnn.joblib")
SCALER_X_PATH = os.path.join(MODEL_DIR, "feature_scaler.joblib")
SCALER_Y_PATH = os.path.join(MODEL_DIR, "target_scaler.joblib")

FEATURES = ["temperature_c", "humidity_percent", "pressure_hpa", "wind_speed_kmh"]
TARGET = "next_day_temperature_c"

for folder in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    os.makedirs(folder, exist_ok=True)

def create_demo_dataset():
    """Generate a reproducible synthetic weather dataset for a safe offline demo."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        required = FEATURES + [TARGET]
        if all(col in df.columns for col in required):
            return df

    rng = np.random.default_rng(7)
    n = 1200
    day = np.arange(n)
    seasonal = 7 * np.sin(2 * np.pi * day / 365.25)
    temperature = 28 + seasonal + rng.normal(0, 3.5, n)
    humidity = np.clip(68 - 0.7 * (temperature - 28) + rng.normal(0, 10, n), 25, 100)
    pressure = 1012 + rng.normal(0, 6, n) - 0.15 * (temperature - 28)
    wind = np.clip(rng.gamma(shape=2.2, scale=5, size=n), 0, 55)
    next_temp = (
        0.72 * temperature
        + 0.10 * (temperature - (humidity - 65) * 0.04)
        + 0.025 * (pressure - 1012)
        - 0.018 * wind
        + 0.28 * np.roll(temperature, -1)
        + rng.normal(0, 1.8, n)
    )
    df = pd.DataFrame({
        "temperature_c": temperature,
        "humidity_percent": humidity,
        "pressure_hpa": pressure,
        "wind_speed_kmh": wind,
        "next_day_temperature_c": next_temp
    }).iloc[:-1].copy()
    df = df.round(2)
    df.to_csv(DATA_PATH, index=False)
    return df

def train_model(df):
    X = df[FEATURES].astype(float)
    y = df[[TARGET]].astype(float)

    # Split without shuffling to preserve chronological order.
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_train_scaled = y_scaler.fit_transform(y_train).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42
    )
    model.fit(X_train_scaled, y_train_scaled)

    pred_scaled = model.predict(X_test_scaled).reshape(-1, 1)
    predictions = y_scaler.inverse_transform(pred_scaled).ravel()
    actual = y_test[TARGET].to_numpy()

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(x_scaler, SCALER_X_PATH)
    joblib.dump(y_scaler, SCALER_Y_PATH)

    # Plot actual and predicted temperatures for the test period.
    plt.figure(figsize=(10, 5))
    plt.plot(actual[:100], label="Actual temperature")
    plt.plot(predictions[:100], label="Predicted temperature")
    plt.xlabel("Test sample (day order)")
    plt.ylabel("Temperature (°C)")
    plt.title("Actual vs Predicted Next-Day Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    # Plot training loss.
    plt.figure(figsize=(8, 4))
    plt.plot(model.loss_curve_)
    plt.xlabel("Training iteration")
    plt.ylabel("Loss")
    plt.title("DNN Training Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"), dpi=150)
    plt.close()

    results = pd.DataFrame({"actual_temperature_c": actual, "predicted_temperature_c": predictions})
    results.to_csv(os.path.join(OUTPUT_DIR, "prediction_results.csv"), index=False)

    print("\n===== MODEL EVALUATION =====")
    print(f"Mean Absolute Error (MAE): {mae:.2f} °C")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} °C")
    print(f"R-squared (R²): {r2:.3f}")
    print("\nGraphs and prediction results saved in the outputs folder.")
    return model, x_scaler, y_scaler

def predict_next_day(model, x_scaler, y_scaler):
    print("\n===== NEXT-DAY TEMPERATURE DEMO =====")
    print("Enter today's weather measurements.")
    try:
        temperature = float(input("Current temperature (°C): "))
        humidity = float(input("Humidity (%, 0-100): "))
        pressure = float(input("Atmospheric pressure (hPa, e.g. 1012): "))
        wind = float(input("Wind speed (km/h): "))
    except ValueError:
        print("Invalid input. Please enter numbers and run the program again.")
        return

    if not (-20 <= temperature <= 60 and 0 <= humidity <= 100
            and 800 <= pressure <= 1100 and 0 <= wind <= 250):
        print("One or more values are outside the expected range. Please run again with sensible values.")
        return

    sample = pd.DataFrame([{
        "temperature_c": temperature,
        "humidity_percent": humidity,
        "pressure_hpa": pressure,
        "wind_speed_kmh": wind
    }], columns=FEATURES)

    scaled = x_scaler.transform(sample)
    pred_scaled = model.predict(scaled).reshape(-1, 1)
    prediction = float(y_scaler.inverse_transform(pred_scaled)[0, 0])
    print(f"\nEstimated next-day temperature: {prediction:.1f} °C")
    print("Demo limitation: this model was trained on synthetic data, not live or verified local weather observations.")

def main():
    print("=" * 58)
    print(" WEATHER PREDICTION USING A DEEP NEURAL NETWORK (DNN)")
    print("=" * 58)
    df = create_demo_dataset()
    print(f"Dataset loaded: {len(df)} synthetic daily examples.")
    model, x_scaler, y_scaler = train_model(df)
    predict_next_day(model, x_scaler, y_scaler)

if __name__ == "__main__":
    main()
