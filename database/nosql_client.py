from pymongo import MongoClient

def get_mongo_client():
    # Uses local MongoDB instance or Atlas Sandbox URI
    client = MongoClient("mongodb://localhost:27017/")
    db = client["ecoroute_db"]
    return db["weather_raw"]

def save_weather_payload(weather_dict):
    try:
        collection = get_mongo_client()
        collection.insert_one(weather_dict)
    except Exception as e:
        print(f"MongoDB offline/not connected, skipping log: {e}")