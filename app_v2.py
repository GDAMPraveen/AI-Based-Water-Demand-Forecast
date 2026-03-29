import streamlit as st
import pandas as pd
import joblib

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="SAYUKTA AI", page_icon="💧", layout="wide")

# ================================
# CSS (LIGHT MODE OPTIMIZED)
# ================================
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: #f4f7fb;
    color: #2c3e50;
}

/* HEADER */
.header {
    font-size: 40px;
    font-weight: 900;
    color: white;
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(90deg, #003366, #0055aa);
    text-align: center;
}

/* MOTTO */
.motto {
    text-align: center;
    font-style: italic;
    color: #5f6c7b;
    margin-top: 10px;
    font-size: 14px;
}

/* CARD */
.card {
    background: #ffffff;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #e6ecf2;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    margin-top: 20px;
}

/* METRIC */
.metric {
    font-size: 60px;
    font-weight: 900;
    color: #003366;
    text-align: center;
}

/* KPI */
.kpi {
    background: #ffffff;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e6ecf2;
    text-align: center;
}

/* NOTE */
.note {
    background: #f9fbfd;
    padding: 18px;
    border-left: 5px solid #C5A059;
    border-radius: 10px;
    color: #2c3e50;
    line-height: 1.6;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #003366, #002244);
}

section[data-testid="stSidebar"] * {
    color: white !important;
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
# SIDEBAR INPUTS
# ================================
st.sidebar.title("💧 SAYUKTA AI")

campus = st.sidebar.selectbox("🏫 Campus", ["Gnanagangothri", "Peenya"])
day_type = st.sidebar.radio("📅 Day Type", ["Academic Day", "Event Day"])

exam = 0
selected_event = None

if day_type == "Academic Day":
    academic_type = st.sidebar.selectbox("Academic Type", ["Regular Day", "Exam Day"])
    if academic_type == "Exam Day":
        exam = 1

if day_type == "Event Day":
    selected_event = st.sidebar.selectbox(
        "🎉 Event",
        ["Akaira", "Pravrutti", "Freshers Day", "Senior Sendoff"]
    )

st.sidebar.markdown("### 👥 Occupancy")
hostel_occ = st.sidebar.slider("Hostel Students", 0, 800, 500)
day_occ = st.sidebar.slider("Day Scholars", 0, 2500, 1500)

st.sidebar.markdown("### 🌦 Weather")
weather = st.sidebar.selectbox("Weather", ["Normal", "Summer", "Monsoon"])

if weather == "Summer":
    temp, humidity = 35, 40
elif weather == "Monsoon":
    temp, humidity = 24, 85
else:
    temp, humidity = 26, 60

# ================================
# INPUT FUNCTION
# ================================
def prepare_input():
    d = {f: 0 for f in features}

    d["Exams"] = exam

    if selected_event:
        mapping = {
            "Akaira": "Akaira",
            "Pravrutti": "Pravrutti",
            "Freshers Day": "Freshers_Day",
            "Senior Sendoff": "Senior_Sendoff"
        }
        d[mapping[selected_event]] = 1

    d["Temp"] = temp
    d["Humidity"] = humidity

    d["Hostel_Occupancy"] = hostel_occ
    d["Day_Scholar_Occupancy"] = day_occ
    d["Total_Occupancy"] = hostel_occ + day_occ

    d["Is_Weekend"] = 0
    d["Is_Vacation"] = 0
    d["Water_Price_Index"] = 1.2
    d["Peak_Factor"] = 1.5

    return pd.DataFrame([d])[features]

# ================================
# HEADER + QUOTE
# ================================
st.markdown('<div class="header">💧 SAYUKTA AI Water Intelligence System</div>', unsafe_allow_html=True)

st.markdown('''
<div class="motto">
"Water is precious. Don't waste your tears on the past, nor the water of our future. 
Every drop is a life."
</div>
''', unsafe_allow_html=True)

# ================================
# SYSTEM READY CARD
# ================================
st.markdown("""
<div class="card">
<h3>System Ready.</h3>
<p>Welcome back. The system has analyzed thousands of campus records to help optimize water usage efficiently.</p>
</div>
""", unsafe_allow_html=True)

# ================================
# KPI ROW
# ================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f'<div class="kpi">🏫 Campus<br><b>{campus}</b></div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="kpi">👥 Total Users<br><b>{hostel_occ + day_occ}</b></div>', unsafe_allow_html=True)

with col3:
    st.markdown(f'<div class="kpi">🌡 Temperature<br><b>{temp}°C</b></div>', unsafe_allow_html=True)

# ================================
# PREDICTION BUTTON
# ================================
if st.button("🚀 RUN AI PREDICTION"):

    input_df = prepare_input()
    prediction = model.predict(input_df)[0]

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 📊 Predicted Water Demand")
    st.markdown(f'<div class="metric">{int(prediction):,} Liters</div>', unsafe_allow_html=True)

    st.markdown("### 🤖 AI Insight")

    reason = f"Predicted demand is **{int(prediction):,} liters**.\n\n"

    if exam:
        reason += "- Exams increase daytime usage.\n"
    if selected_event:
        reason += f"- Event {selected_event} increases consumption.\n"
    if weather == "Summer":
        reason += "- High temperature increases water usage.\n"

    reason += "\n**Recommendation:** Optimize pump schedules and monitor peak usage."

    st.markdown(f'<div class="note">{reason}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📋 Input Data")
    st.dataframe(input_df)
