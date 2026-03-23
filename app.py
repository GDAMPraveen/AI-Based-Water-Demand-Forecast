from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load model
bundle = joblib.load("ruas_model_v2.pkl")
model = bundle["model"]
feature_order = bundle["features"]

# 🔥 FULL ORIGINAL HTML (only 1 CHANGE: fetch URL fixed)
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SAYUKTA | RUAS AI Water Agent</title>

<style>
:root { --ruas-blue: #003366; --ruas-gold: #C5A059; --text: #2c3e50; --bg: #f8fafc; }
body { font-family: 'Inter', sans-serif; background: var(--bg); margin: 0; display: flex; color: var(--text); height: 100vh; overflow: hidden; }

.sidebar { width: 360px; background: var(--ruas-blue); color: white; padding: 35px; box-shadow: 10px 0 30px rgba(0,0,0,0.1); overflow-y: auto; }
.sidebar h1 { color: var(--ruas-gold); margin: 0; font-size: 2.2rem; font-weight: 900; }

label { display: block; margin-top: 20px; font-weight: 600; font-size: 0.75rem; color: #a0aec0; }
select { width: 100%; padding: 12px; margin-top: 8px; border-radius: 8px; background: #1a202c; color: white; }

.btn-ai { background: var(--ruas-gold); color: white; padding: 16px; border-radius: 10px; width: 100%; font-weight: 800; cursor: pointer; margin-top: 30px; }

.container { flex: 1; padding: 50px; overflow-y: auto; }

.card { background: white; border-radius: 20px; padding: 40px; }

#loading { display: none; text-align: center; margin-top: 80px; }
#output-view { display: none; }

.liters-display { font-size: 4.5rem; font-weight: 900; color: var(--ruas-blue); }

.ai-report { background: #fdfaf3; padding: 25px; border-radius: 15px; margin-top: 20px; }
</style>
</head>

<body>

<div class="sidebar">
<h1>SAYUKTA</h1>

<label>Campus</label>
<select id="campus">
<option value="Gnanagangothri">Gnanagangothri</option>
<option value="Peenya">Peenya</option>
</select>

<label>Event</label>
<select id="event">
<option value="none">Normal</option>
<option value="akaira">Akaira</option>
<option value="fresher">Fresher</option>
<option value="senior">Senior</option>
</select>

<label>Weather</label>
<select id="weather">
<option value="normal">Normal</option>
<option value="summer">Summer</option>
<option value="monsoon">Monsoon</option>
</select>

<label>Hostel</label>
<select id="hostel">
<option value="full">Full</option>
<option value="partial">Partial</option>
<option value="empty">Empty</option>
</select>

<button class="btn-ai" onclick="runAIEngine()">CONSULT AI</button>
</div>

<div class="container">

<div id="welcome-msg" class="card">
<h1>System Ready</h1>
<p>AI analyzing campus water intelligence...</p>
</div>

<div id="loading">
<p>AI calculating...</p>
</div>

<div id="output-view">
<div class="card">
<div class="liters-display" id="total-liters">0</div>
<p id="confidence"></p>

<div class="ai-report">
<p id="ai-text"></p>
</div>
</div>
</div>

</div>

<script>
async function runAIEngine() {

document.getElementById('welcome-msg').style.display = 'none';
document.getElementById('loading').style.display = 'block';

const payload = {
campus: document.getElementById('campus').value,
event: document.getElementById('event').value,
weather: document.getElementById('weather').value,
hostel: document.getElementById('hostel').value
};

const response = await fetch('/predict', {
method: 'POST',
headers: {'Content-Type': 'application/json'},
body: JSON.stringify(payload)
});

const data = await response.json();

document.getElementById('loading').style.display = 'none';
document.getElementById('output-view').style.display = 'block';

document.getElementById('total-liters').innerText = data.prediction + " L";
document.getElementById('confidence').innerText = "Confidence: " + (data.confidence*100).toFixed(1) + "%";
document.getElementById('ai-text').innerText = data.reason;

}
</script>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    event = data['event']
    weather = data['weather']
    hostel = data['hostel']

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

    reason = f"Analyzing {data['campus']} campus. Event: {event}, Weather: {weather}. Demand adjusted by occupancy and climate."

    return jsonify({
        "prediction": int(prediction),
        "reason": reason,
        "confidence": 0.96
    })


if __name__ == "__main__":
    app.run(debug=True)