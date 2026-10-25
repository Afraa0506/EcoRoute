import os
import numpy as np
import pandas as pd
import random
from weather_fetcher import get_microclimate_weather

def generate_ecoroute_dataset(num_samples: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    data = []

    base_coords = [
        (13.0827, 80.2707), # Chennai
        (12.9716, 77.5946), # Bengaluru
        (17.3850, 78.4867), # Hyderabad
        (18.5204, 73.8567)  # Pune
    ]

    print(f"Generating {num_samples} simulated EV trip telemetry records...")

    for i in range(num_samples):
        lat, lon = random.choice(base_coords)
        lat += np.random.uniform(-0.5, 0.5)
        lon += np.random.uniform(-0.5, 0.5)
        
        speed_kmh = np.random.uniform(30, 120)            
        elevation_gain_m = np.random.uniform(-50, 200)   
        ev_weight_kg = np.random.uniform(1500, 2200)      
        
        temp_c = np.random.uniform(-5, 42)                 
        headwind_kmh = np.random.uniform(0, 45)            
        humidity_pct = np.random.uniform(20, 95) 
        
        air_density = 1.225 * (273.15 / (273.15 + temp_c))
        effective_speed_ms = (speed_kmh + headwind_kmh) / 3.6
        drag_force = 0.5 * air_density * 0.28 * 2.2 * (effective_speed_ms ** 2)
        
        if temp_c < 15:
            temp_penalty_factor = 1.0 + (15 - temp_c) * 0.025  
        elif temp_c > 35:
            temp_penalty_factor = 1.0 + (temp_c - 35) * 0.015  
        else:
            temp_penalty_factor = 1.0
            
        elevation_penalty = max(0, elevation_gain_m * 0.005)
        base_consumption = 0.15
        
        soc_loss_per_km = (base_consumption * temp_penalty_factor) + (drag_force * 0.0001) + elevation_penalty
        soc_loss_per_km += np.random.normal(0, 0.01)
        soc_loss_per_km = max(0.08, soc_loss_per_km)

        data.append({
            "speed_kmh": round(speed_kmh, 2),
            "elevation_gain_m": round(elevation_gain_m, 2),
            "ev_weight_kg": round(ev_weight_kg, 2),
            "temperature_c": round(temp_c, 2),
            "headwind_kmh": round(headwind_kmh, 2),
            "humidity_pct": round(humidity_pct, 2),
            "soc_loss_per_km": round(soc_loss_per_km, 4)
        })

    df = pd.DataFrame(data)
    
    # --- AUTOMATIC DIRECTORY CREATION FIX ---
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "ecoroute_telemetry.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated successfully and saved to '{output_path}'")
    return df

if __name__ == "__main__":
    df = generate_ecoroute_dataset(num_samples=3000)
    print(df.head())