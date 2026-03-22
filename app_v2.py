import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the model (Streamlit way)
@st.cache_resource
def load_model():
    bundle = joblib.load("ruas_model_v2.pkl")
    return bundle["model"], bundle["features"]

model, feature_order = load_model()

# 2. Sidebar Inputs
st.sidebar.title("SAYUKTA | RUAS AI")
campus = st.sidebar.selectbox("Campus Node", ["Gnanagangothri", "Peenya"])
event = st.sidebar.selectbox("Active Event", ["none", "akaira", "pravrutti", "fresher", "senior"])
weather = st.sidebar.selectbox("Climate Profile", ["normal", "summer", "monsoon"])
hostel = st.sidebar.selectbox("Residential Load", ["full", "partial", "empty"])

if st.sidebar.button("GENERATE FORECAST"):
    # --- Feature Mapping Logic ---
    features = {
        'Is_Weekend': 0,
        'Is_Vacation': 1 if hostel == 'empty' else 0,
        'Exams': 0, 'Placement': 0, 'Convocation': 0, 'Rajyotsava': 0, 
        'Religious_Fest': 0, 'Industrial_Visit': 0,
        'Akaira': 1 if event == 'akaira' else 0,
        'Pravrutti': 1 if event == 'pravrutti' else 0,
        'Freshers_Day': 1 if event == 'fresher' else 0,
        'Senior_Sendoff': 1 if event == 'senior' else 0,
        'Temp': 34 if weather == 'summer' else (22 if weather == 'monsoon' else 26),
        'Humidity': 80 if weather == 'monsoon' else (40 if weather == 'summer' else 60),
    }

    # Occupancy Logic
    h = 750 if hostel == 'full' else (300 if hostel == 'partial' else 50)
    d = 1800 if hostel == 'full' else (1200 if hostel == 'partial' else 100)
    features.update({'Hostel_Occupancy': h, 'Day_Scholar_Occupancy': d, 'Total_Occupancy': h + d})
    features.update({'Water_Price_Index': 1.0, 'Peak_Factor': 1.2 if event != 'none' else 1.0})

    # 3. Predict
    input_data = np.array([[features[f] for f in feature_order]])
    prediction = model.predict(input_data)[0]

    # 4. Poetic AI Reasoning
    reason = f"Analyzing the pulse of **{campus}**... "
    if event != 'none':
        reason += f"The {event} festivities bring a surge of life—and demand. "
    reason += "Remember, every drop saved is a tear unshed for the future. "

    # 5. Display Result
    st.balloons()
    st.metric("Predicted Water Demand", f"{round(prediction):,} Liters")
    st.info(reason)
else:
    st.title("System Initialized.")
    st.write("Configure the sidebar and click **Generate Forecast**.")
