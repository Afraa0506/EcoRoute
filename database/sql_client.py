import sqlite3
import pandas as pd

DB_PATH = "database/ecoroute.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trip_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            speed_kmh REAL,
            temperature_c REAL,
            soc_loss_predicted REAL,
            is_anomaly INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def log_trip(speed, temp, soc_loss, is_anomaly):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO trip_logs (speed_kmh, temperature_c, soc_loss_predicted, is_anomaly)
        VALUES (?, ?, ?, ?)
    ''', (speed, temp, soc_loss, is_anomaly))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("SQLite database initialized successfully.")