import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
from sklearn.metrics import r2_score

def train_soc_regressor():
    csv_path = "data/processed/dataset.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please place your JAC IEV40 CSV there.")
        
    df = pd.read_csv(csv_path)
    
    # Map JAC IEV40 columns if they are named using raw abbreviations
    # Standardizing feature columns for training
    df["speed_kmh"] = df.get("SPD", df.get("speed_kmh", 50.0))
    df["elevation_gain_m"] = df.get("ALT", df.get("elevation_gain_m", 0.0)).diff().clip(lower=0).fillna(0)
    df["voltage_v"] = df.get("VOL", df.get("voltage_v", 350.0))
    df["current_a"] = df.get("CUR", df.get("current_a", 15.0))
    df["eco_mode"] = df.get("ECO", df.get("eco_mode", 0))
    df["ac_status"] = df.get("AIR", df.get("ac_status", 0))
    
    # Target variable: Battery charge depletion per km (CH drop over ODO)
    if "soc_loss_per_km" not in df.columns:
        ch = df.get("CH", df.get("soc_percentage", 100.0))
        odo = df.get("ODO", df.get("odometer", pd.Series(range(len(df))) * 0.1))
        df["soc_loss_per_km"] = (ch.shift(1) - ch).abs() / (odo.diff().abs() + 1e-5)
        df["soc_loss_per_km"] = df["soc_loss_per_km"].fillna(method='bfill').clip(0.001, 2.0)

    features = ["speed_kmh", "elevation_gain_m", "voltage_v", "current_a", "eco_mode", "ac_status"]
    X = df[features].fillna(0)
    y = df["soc_loss_per_km"].fillna(0.1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)


    r2 = r2_score(y_test, predictions)
    print(f"R² Score (Coefficient of Determination): {r2:.4f}")
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"XGBoost Regressor Trained successfully on JAC IEV40 Telemetry!")
    print(f"MAE: {mae:.5f} | RMSE: {rmse:.5f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_soc_model.pkl")
    return mae, rmse

if __name__ == "__main__":
    train_soc_regressor()