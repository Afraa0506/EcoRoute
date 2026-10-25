import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib
import os
from sklearn.metrics import silhouette_score

def train_anomaly_detector():
    csv_path = "data/processed/dataset.csv"
    df = pd.read_csv(csv_path)
    
    # Extract real telemetry features for clustering battery/power anomalies
    df["speed_kmh"] = df.get("SPD", df.get("speed_kmh", 50.0))
    df["voltage_v"] = df.get("VOL", df.get("voltage_v", 350.0))
    df["current_a"] = df.get("CUR", df.get("current_a", 15.0))
    
    features = df[["speed_kmh", "voltage_v", "current_a"]].copy().fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # DBSCAN anomaly detection
    dbscan = DBSCAN(eps=0.4, min_samples=5)
    clusters = dbscan.fit_predict(scaled_features)
    
    num_anomalies = (clusters == -1).sum()
    print(f"DBSCAN Clustering Complete on Telemetry. Flagged {num_anomalies} anomaly segments.")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(dbscan, "models/dbscan_model.pkl")

    

    # Calculate silhouette score (ignoring noise points labeled as -1 if needed, or including them)
    valid_mask = clusters != -1
    if len(set(clusters[valid_mask])) > 1:
        sil_score = silhouette_score(scaled_features[valid_mask], clusters[valid_mask])
        print(f"Silhouette Score (Cluster Separation): {sil_score:.4f}")
    else:
        print("Silhouette Score: Not enough distinct clusters found.")

    # Print distribution report
    total_points = len(clusters)
    anomalies = (clusters == -1).sum()
    normal_points = total_points - anomalies
    print(f"--- DBSCAN Cluster Report ---")
    print(f"Total Telemetry Points: {total_points}")
    print(f"Normal Driving Segments: {normal_points} ({(normal_points/total_points)*100:.1f}%)")
    print(f"Anomaly / High-Stress Segments (-1): {anomalies} ({(anomalies/total_points)*100:.1f}%)")

if __name__ == "__main__":
    train_anomaly_detector()