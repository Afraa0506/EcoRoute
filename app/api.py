import os
import joblib
import sqlite3
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="EcoRoute-ML API with User Profiles",
    description="EV Energy Degradation & Telemetry Prediction Service",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
XGB_MODEL_PATH = os.path.normpath(os.path.join(MODEL_DIR, "xgboost_soc_model.pkl"))
DBSCAN_MODEL_PATH = os.path.normpath(os.path.join(MODEL_DIR, "dbscan_model.pkl"))
SCALER_PATH = os.path.normpath(os.path.join(MODEL_DIR, "scaler.pkl"))
DB_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "database", "ecoroute.db"))

xgboost_model = None
dbscan_model = None
scaler = None

@app.on_event("startup")
def load_artifacts():
    global xgboost_model, dbscan_model, scaler
    try:
        if os.path.exists(XGB_MODEL_PATH):
            xgboost_model = joblib.load(XGB_MODEL_PATH)
        if os.path.exists(DBSCAN_MODEL_PATH):
            dbscan_model = joblib.load(DBSCAN_MODEL_PATH)
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        print("ML Artifacts loaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to load model artifacts: {e}")

class TripRequest(BaseModel):
    username: str = Field(..., example="driver_john")
    car_model: str = Field(..., example="JAC IEV40")
    battery_capacity_kwh: float = Field(..., example=40.0)
    ev_weight_kg: float = Field(..., example=1400.0)
    start_lat: float = Field(..., example=-21.7803)
    start_lon: float = Field(..., example=-46.5682)
    end_lat: float = Field(..., example=-22.2016)
    end_lon: float = Field(..., example=-46.9458)
    speed_kmh: float = Field(..., example=65.0)
    elevation_gain_m: float = Field(..., example=120.0)
    starting_soc: float = Field(..., example=90.0)

class TripResponse(BaseModel):
    username: str
    car_model: str
    temperature_c: float
    headwind_kmh: float
    predicted_soc_loss_per_km: float
    estimated_total_soc_loss: float
    final_estimated_soc: float
    is_anomaly: bool
    anomaly_flag: int

def fetch_open_meteo_weather(lat: float, lon: float):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        current = response.json().get("current_weather", {})
        return {
            "temperature_c": float(current.get("temperature", 22.0)),
            "headwind_kmh": float(current.get("windspeed", 10.0))
        }
    except Exception:
        return {"temperature_c": 22.0, "headwind_kmh": 10.0}

def log_prediction_sqlite(trip: TripRequest, loss_rate: float, total_loss: float, is_anomaly: bool):
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_trip_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                car_model TEXT,
                speed_kmh REAL,
                elevation_gain_m REAL,
                predicted_soc_loss_per_km REAL,
                total_loss REAL,
                is_anomaly INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            INSERT INTO user_trip_logs (username, car_model, speed_kmh, elevation_gain_m, predicted_soc_loss_per_km, total_loss, is_anomaly)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (trip.username, trip.car_model, trip.speed_kmh, trip.elevation_gain_m, loss_rate, total_loss, 1 if is_anomaly else 0))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Log error: {e}")

@app.post("/predict-trip-energy", response_model=TripResponse)
def predict_trip_energy(trip: TripRequest):
    weather = fetch_open_meteo_weather(trip.start_lat, trip.start_lon)
    
    features = pd.DataFrame([{
        "speed_kmh": trip.speed_kmh,
        "elevation_gain_m": trip.elevation_gain_m,
        "voltage_v": 350.0,
        "current_a": trip.ev_weight_kg * 0.01,
        "eco_mode": 1 if "Eco" in trip.car_model else 0,
        "ac_status": 1
    }])
    
    if xgboost_model is not None:
        soc_loss_per_km = float(xgboost_model.predict(features)[0])
    else:
        soc_loss_per_km = 0.22
        
    estimated_total_soc_loss = soc_loss_per_km * 15.0
    final_soc = max(0.0, trip.starting_soc - estimated_total_soc_loss)
    
    if scaler is not None:
        cluster_features = pd.DataFrame([{
            "speed_kmh": trip.speed_kmh,
            "voltage_v": 350.0,
            "current_a": trip.ev_weight_kg * 0.01
        }])
        scaled = scaler.transform(cluster_features)
        
        is_anomaly = trip.speed_kmh > 115.0 or trip.elevation_gain_m > 300.0
        cluster_flag = -1 if is_anomaly else 0
    else:
        cluster_flag = 0
        is_anomaly = False
        
    log_prediction_sqlite(trip, soc_loss_per_km, estimated_total_soc_loss, is_anomaly)
    
    return {
        "username": trip.username,
        "car_model": trip.car_model,
        "temperature_c": weather["temperature_c"],
        "headwind_kmh": weather["headwind_kmh"],
        "predicted_soc_loss_per_km": round(soc_loss_per_km, 4),
        "estimated_total_soc_loss": round(estimated_total_soc_loss, 2),
        "final_estimated_soc": round(final_soc, 2),
        "is_anomaly": is_anomaly,
        "anomaly_flag": cluster_flag
    }

@app.get("/history")
def get_history(username: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM user_trip_logs WHERE username = ? ORDER BY id DESC", conn, params=(username,))
        conn.close()
        return df.to_dict(orient="records")
    except Exception:
        return []
