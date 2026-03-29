import streamlit as st
import pandas as pd
import joblib

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="SAYUKTA AI", page_icon="💧", layout="wide")

# ================================
# CUSTOM CSS (MATCH YOUR HTML STYLE)
# ================================
st.markdown("""
<style>
body {
    background-color: #f8fafc;
}
.big-title {
    font-size: 40px;
    font-weight: 800;
    color: #003366;
}
.card {
    background: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.05);
}
.metric {
    font-size: 60px;
    font-weight: 900;
    color: #003366;
}
.note {
    background: #fdfaf3;
    padding: 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ================================
# LOAD MODEL
# ================================
model_data = joblib.load("ruas_model_v2.pkl")
model = model_data["model"]
features = model_data["features"]

# ================================
# SIDEBAR (LIKE YOUR HTML)
# ================================
st.sidebar.title("💧 SAYUKTA")

campus = st.sidebar.selectbox("Campus", ["Gnanagangothri", "Peenya"])
event = st.sidebar.selectbox("Event", ["none", "akaira", "fresher", "senior"])
weather = st.sidebar.selectbox("Weather", ["normal", "summer", "monsoon"])
hostel = st.sidebar.selectbox("Hostel", ["full", "partial", "empty"])

# ================================
# INPUT FUNCTION
# ================================
def prepare_input():
    input_dict = {f: 0 for f in features}

    # Events
    if event == "akaira":
        input_dict["Akaira"] = 1
    elif event == "fresher":
        input_dict["Freshers_Day"] = 1
    elif event == "senior":
        input_dict["Senior_Sendoff"] = 1

    # Weather
    if weather == "summer":
        input_dict["Temp"] = 35
        input_dict["Humidity"] = 40
    elif weather == "monsoon":
        input_dict["Temp"] = 24
        input_dict["Humidity"] = 85
    else:
        input_dict["Temp"] = 26
        input_dict["Humidity"] = 60

    # Hostel
    if hostel == "full":
        input_dict["Hostel_Occupancy"] = 750
        input_dict["Day_Scholar_Occupancy"] = 1800
    elif hostel == "partial":
        input_dict["Hostel_Occupancy"] = 300
        input_dict["Day_Scholar_Occupancy"] = 1200
    else:
        input_dict["Hostel_Occupancy"] = 50
        input_dict["Day_Scholar_Occupancy"] = 200

    # Total
    input_dict["Total_Occupancy"] = (
        input_dict["Hostel_Occupancy"] +
        input_dict["Day_Scholar_Occupancy"]
    )

    # Defaults
    input_dict["Is_Weekend"] = 0
    input_dict["Is_Vacation"] = 1 if hostel == "empty" else 0
    input_dict["Water_Price_Index"] = 1.2
    input_dict["Peak_Factor"] = 1.5

    return pd.DataFrame([input_dict])[features]

# ================================
# MAIN UI
# ================================
st.markdown('<div class="big-title">SAYUKTA AI Water Intelligence</div>', unsafe_allow_html=True)

st.markdown("Predict and optimize campus water usage using AI")

# ================================
# BUTTON
# ================================
if st.button("🚀 CONSULT AI AGENT"):

    input_df = prepare_input()
    prediction = model.predict(input_df)[0]

    # ================================
    # OUTPUT CARD
    # ================================
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 📊 Estimated Water Demand")
    st.markdown(f'<div class="metric">{int(prediction):,} L</div>', unsafe_allow_html=True)

    st.markdown("### 🤖 AI Insight")

    st.markdown(f"""
    <div class="note">
    Based on the selected campus conditions, water demand is predicted to reach 
    <b>{int(prediction):,} liters</b>.

    <br><br>
    <b>Key Drivers:</b>
    <ul>
    <li>Event-based crowd increase</li>
    <li>Weather conditions</li>
    <li>Hostel & campus occupancy</li>
    </ul>

    <b>Recommendation:</b><br>
    Adjust pump schedules and monitor peak hours to reduce water wastage.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
