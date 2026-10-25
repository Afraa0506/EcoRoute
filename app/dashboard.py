import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
import os  

st.set_page_config(page_title="EcoRoute-ML Portal", page_icon="⚡", layout="wide")

API_URL = os.environ.get("BACKEND_URL", "http://fastapi_app:8000")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

# ================= LOGIN SCREEN =================
if not st.session_state.authenticated:
    st.title("🔐 EcoRoute-ML User Portal")
    st.subheader("Sign in with your driver handle to load your personal EV profile")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username_input = st.text_input("Driver Username", value="driver_alex")
        password_input = st.text_input("Password", type="password", value="ev123")
        if st.button("🚀 Enter Portal", type="primary", use_container_width=True):
            if username_input.strip():
                st.session_state.authenticated = True
                st.session_state.username = username_input
                st.rerun()
            else:
                st.error("Please enter a valid username.")
    st.stop()

# ================= MAIN DASHBOARD =================
st.title(f"⚡ EcoRoute-ML Dashboard — Welcome, {st.session_state.username}!")
st.markdown("### Real-Time EV Energy Degradation & JAC Telemetry Routing")

with st.sidebar:
    st.header("🚘 User & EV Profile")
    st.success(f"Logged in as: **{st.session_state.username}**")
    
    car_model = st.selectbox(
        "Select Your EV Model",
        ["JAC IEV40 (40kWh)", "Tesla Model 3 (75kWh)", "Tata Nexon EV (30kWh)", "Hyundai Ioniq 5 (58kWh)"]
    )
    
    if "JAC" in car_model:
        default_weight, default_batt = 1400.0, 40.0
    elif "Tesla" in car_model:
        default_weight, default_batt = 1750.0, 75.0
    elif "Tata" in car_model:
        default_weight, default_batt = 1400.0, 30.0
    else:
        default_weight, default_batt = 1900.0, 58.0

    ev_weight_kg = st.number_input("Vehicle Weight (kg)", value=default_weight, step=50.0)
    battery_capacity_kwh = st.number_input("Battery Capacity (kWh)", value=default_batt, step=5.0)
    starting_soc = st.slider("Current Battery Charge (SoC %)", min_value=10, max_value=100, value=85)
    
    st.divider()
    st.subheader("🗺️ Trip Parameters")
    speed_kmh = st.slider("Average Speed (km/h)", 30, 120, 65)
    elevation_gain_m = st.number_input("Route Elevation Gain (m)", value=120.0)
    
    start_lat = st.number_input("Start Latitude", value=-21.7803, format="%.4f")
    start_lon = st.number_input("Start Longitude", value=-46.5682, format="%.4f")
    end_lat = st.number_input("End Latitude", value=-22.2016, format="%.4f")
    end_lon = st.number_input("End Longitude", value=-46.9458, format="%.4f")
    
    predict_btn = st.button("⚡ Calculate Personal Energy Drain", type="primary", use_container_width=True)
    
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.prediction_data = None
        st.rerun()

if predict_btn:
    payload = {
        "username": st.session_state.username,
        "car_model": car_model,
        "battery_capacity_kwh": battery_capacity_kwh,
        "ev_weight_kg": ev_weight_kg,
        "start_lat": start_lat,
        "start_lon": start_lon,
        "end_lat": end_lat,
        "end_lon": end_lon,
        "speed_kmh": float(speed_kmh),
        "elevation_gain_m": float(elevation_gain_m),
        "starting_soc": float(starting_soc)
    }
    
    with st.spinner("Analyzing JAC telemetry models & fetching live weather..."):
        try:
            res = requests.post(f"{API_URL}/predict-trip-energy", json=payload, timeout=10)
            if res.status_code == 200:
                st.session_state.prediction_data = res.json()
            else:
                st.error("Backend prediction failed.")
        except Exception as e:
            st.error(f"Could not connect to FastAPI: {e}")

if st.session_state.prediction_data:
    data = st.session_state.prediction_data
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌡️ Ambient Temp", f"{data['temperature_c']} °C")
    with col2:
        st.metric("💨 Headwind", f"{data['headwind_kmh']} km/h")
    with col3:
        st.metric("🔋 Est. Total Drain", f"{data['estimated_total_soc_loss']:.1f}%")
    with col4:
        st.metric("⚡ Final Battery SoC", f"{data['final_estimated_soc']:.1f}%", delta=f"-{data['estimated_total_soc_loss']:.1f}%")
        
    st.divider()
    st.subheader("🗺️ Route Telemetry & Anomaly Analysis")
    
    if data["is_anomaly"]:
        st.error("⚠️ **TELEMETRY ANOMALY / HIGH DRAIN ZONE DETECTED (DBSCAN Flag: -1)**")
        r_color = "red"
    else:
        st.success("✅ **NORMAL DRIVING TELEMETRY**")
        r_color = "green"
        
    m = folium.Map(location=[(start_lat+end_lat)/2, (start_lon+end_lon)/2], zoom_start=9)
    folium.Marker([start_lat, start_lon], popup="Origin", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker([end_lat, end_lon], popup="Destination", icon=folium.Icon(color="black")).add_to(m)
    folium.PolyLine([[start_lat, start_lon], [end_lat, end_lon]], color=r_color, weight=6).add_to(m)
    st_folium(m, width=1100, height=400)

st.divider()
st.subheader(f"📜 Trip History for {st.session_state.username}")
if st.button("🔄 Refresh My History"):
    try:
        history_res = requests.get(f"{API_URL}/history?username={st.session_state.username}", timeout=5)
        if history_res.status_code == 200:
            history_logs = history_res.json()
            if history_logs:
                st.dataframe(pd.DataFrame(history_logs), use_container_width=True)
            else:
                st.info("No recorded trips found for this user.")
    except Exception as e:
        st.warning(f"Error fetching history: {e}")
