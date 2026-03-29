import streamlit as st
import pandas as pd
import joblib

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="SAYUKTA AI", page_icon="💧", layout="wide")

# ================================
# ADVANCED CSS
# ================================
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #eef2f7, #f8fafc);
}

/* HEADER */
.header {
    font-size: 42px;
    font-weight: 900;
    color: white;
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(90deg, #003366, #0055aa);
    text-align: center;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.2);
}

/* CARD */
.card {
    background: rgba(255,255,255,0.9);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    backdrop-filter: blur(10px);
    margin-top: 20px;
}

/* METRIC BIG */
.metric {
    font-size: 65px;
    font-weight: 900;
    color: #003366;
    text-align: center;
}

/* KPI BOX */
.kpi {
    background: white;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
}

/* AI NOTE */
.note {
    background: #fdfaf3;
    padding: 20px;
    border-left: 5px solid #C5A059;
    border-radius: 10px;
    margin-top: 15px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #003366, #002244);
    color: white;
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
# SIDEBAR
# ================================
st.sidebar.title("💧 SAYUKTA AI")

campus = st.sidebar.selectbox("🏫 Campus", ["Gnanagangothri", "Peenya"])
day_type = st.sidebar.radio("📅 Day Type", ["Academic Day", "Event Day"])

exam = 0
selected_event = None

if day_type == "Academic Day":
    academic_type = st.sidebar.selectbox("Academic Type", ["Regular", "Exam Day"])
    if academic_type == "Exam Day":
        exam = 1

if day_type == "Event Day":
    selected_event = st.sidebar.selectbox(
        "🎉 Event",
        ["Akaira", "Pravrutti", "Freshers Day", "Senior Sendoff"]
    )

st.sidebar.markdown("### 👥 Occupancy")
hostel_occ = st.sidebar.slider("Hostel", 0, 800, 500)
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
# HEADER
# ================================
st.markdown('<div class="header">💧 SAYUKTA AI Water Intelligence System</div>', unsafe_allow_html=True)

# ================================
# KPI ROW
# ================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="kpi">🏫 Campus<br><b>{}</b></div>'.format(campus), unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi">👥 Total Users<br><b>{}</b></div>'.format(hostel_occ + day_occ), unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi">🌡 Temp<br><b>{}°C</b></div>'.format(temp), unsafe_allow_html=True)

# ================================
# BUTTON
# ================================
if st.button("🚀 RUN AI PREDICTION"):

    input_df = prepare_input()
    prediction = model.predict(input_df)[0]

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 📊 Predicted Water Demand")
    st.markdown(f'<div class="metric">{int(prediction):,} L</div>', unsafe_allow_html=True)

    st.markdown("### 🤖 AI Insight")

    reason = f"Predicted demand is **{int(prediction):,} liters**.\n\n"

    if exam:
        reason += "- Exams increase daytime usage.\n"
    if selected_event:
        reason += f"- Event {selected_event} increases consumption.\n"
    if weather == "Summer":
        reason += "- Heat increases water usage.\n"

    reason += "\n**Recommendation:** Monitor peak usage and optimize distribution."

    st.markdown(f'<div class="note">{reason}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📋 Input Data")
    st.dataframe(input_df)
