import requests
import json

def get_microclimate_weather(lat: float, lon: float) -> dict:
    """
    Fetches real-time micro-climate weather metrics from Open-Meteo (100% Free API).
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json().get("current", {})
        return {
            "temperature_c": data.get("temperature_2m", 20.0),
            "humidity_pct": data.get("relative_humidity_2m", 50.0),
            "wind_speed_kmh": data.get("wind_speed_10m", 10.0),
            "precipitation_mm": data.get("precipitation", 0.0)
        }
    else:
        # Fallback values if API is unreachable
        return {
            "temperature_c": 20.0,
            "humidity_pct": 50.0,
            "wind_speed_kmh": 10.0,
            "precipitation_mm": 0.0
        }

if __name__ == "__main__":
    # Test query for Chennai (Latitude: 13.0827, Longitude: 80.2707)
    sample_weather = get_microclimate_weather(13.0827, 80.2707)
    print("Fetched Open-Meteo Weather Data:")
    print(json.dumps(sample_weather, indent=2))