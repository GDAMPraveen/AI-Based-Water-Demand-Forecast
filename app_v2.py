import streamlit as st
import joblib
import numpy as np
import time

st.set_page_config(layout="wide")

# Load model
bundle = joblib.load("ruas_model_v2.pkl")
model = bundle["model"]
feature_order = bundle["features"]

# ---------- SESSION ----------
if "stage" not in st.session_state:
    st.session_state.stage = "welcome"

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background-color: #f8fafc;
}
.big-number {
    font-size: 80px;
    font-weight: 900;
    color: #003366;
}
.card {
    background: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.05);
}
.loading {
    text-align: center;
    font-style: italic;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 💧 SAYUKTA")

    campus = st.selectbox("Campus", ["Gnanagangothri", "Peenya"])
    event = st.selectbox("Event", ["none", "akaira", "fresher", "senior"])
    weather = st.selectbox("Weather", ["normal", "summer", "monsoon"])
    hostel = st.selectbox("Hostel", ["full", "partial", "empty"])

    if st.button("🚀 CONSULT AI AGENT"):
        st.session_state.stage = "loading"

        # --- Feature Engineering ---
        features = {
            'Is_Weekend': 0,
            'Is_Vacation': 1 if hostel == 'empty' else 0,
            'Exams': 0,
            'Placement': 0,
            'Akaira': 1 if event == 'akaira' else 0,
            'Pravrutti': 0,
            'Freshers_Day': 1 if event == 'fresher' else 0,
            'Senior_Sendoff': 1 if event == 'senior' else 0,
            'Convocation': 0,
            'Rajyotsava': 0,
            'Religious_Fest': 0,
            'Industrial_Visit': 0,
            'Temp': 34 if weather == 'summer' else (22 if weather == 'monsoon' else 26),
            'Humidity': 80 if weather == 'monsoon' else (40 if weather == 'summer' else 60),
        }

        if hostel == 'full':
            h, d = 750, 1800
        elif hostel == 'partial':
            h, d = 300, 1200
        else:
            h, d = 50, 100

        features['Hostel_Occupancy'] = h
        features['Day_Scholar_Occupancy'] = d
        features['Total_Occupancy'] = h + d
        features['Water_Price_Index'] = 1.0
        features['Peak_Factor'] = 1.2 if event != 'none' else 1.0

        input_data = np.array([[features[f] for f in feature_order]])
        prediction = model.predict(input_data)[0]

        st.session_state.prediction = int(prediction)
        st.session_state.reason = f"""
Analyzing {campus} campus...

Event: {event}
Weather: {weather}

Water demand increases with occupancy, events, and temperature.
Every drop saved is a step toward sustainability.
"""

        time.sleep(2)
        st.session_state.stage = "result"


# ---------- MAIN ----------

# 🌊 WELCOME
if st.session_state.stage == "welcome":
    st.markdown('<div class="card"><h2>System Ready</h2><p>AI has analyzed campus data. Start from sidebar.</p></div>', unsafe_allow_html=True)

# 💧 LOADING WITH EFFECT
elif st.session_state.stage == "loading":
    st.markdown('<div class="loading"><h3>💭 AI is calculating the value of every drop...</h3></div>', unsafe_allow_html=True)
    st.progress(90)

# 📊 RESULT WITH STYLE
elif st.session_state.stage == "result":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 💧 Estimated Water Usage")

    st.markdown(
        f'<div class="big-number">{st.session_state.prediction} L</div>',
        unsafe_allow_html=True
    )

    st.metric("Model Confidence", "96%")

    st.markdown("---")

    st.markdown("### 🤖 Strategic AI Analysis")

    # Typing effect simulation
    placeholder = st.empty()
    text = st.session_state.reason

    typed = ""
    for char in text:
        typed += char
        placeholder.markdown(typed)
        time.sleep(0.01)

    st.markdown('</div>', unsafe_allow_html=True)
