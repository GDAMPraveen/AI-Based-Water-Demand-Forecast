from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

bundle = joblib.load("ruas_model_v2.pkl")
model = bundle["model"]
feature_order = bundle["features"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    event = data['event']
    weather = data['weather']
    hostel = data['hostel']

    # --- Feature Mapping ---
    features = {
        'Is_Weekend': 0,
        'Is_Vacation': 1 if hostel == 'empty' else 0,
        'Exams': 0,
        'Placement': 0,
        'Akaira': 1 if event == 'akaira' else 0,
        'Pravrutti': 1 if event == 'pravrutti' else 0,
        'Freshers_Day': 1 if event == 'fresher' else 0,
        'Senior_Sendoff': 1 if event == 'senior' else 0,
        'Convocation': 0,
        'Rajyotsava': 0,
        'Religious_Fest': 0,
        'Industrial_Visit': 0,
        'Temp': 34 if weather == 'summer' else (22 if weather == 'monsoon' else 26),
        'Humidity': 80 if weather == 'monsoon' else (40 if weather == 'summer' else 60),
    }

    # Occupancy
    if hostel == 'full':
        h = 750; d = 1800
    elif hostel == 'partial':
        h = 300; d = 1200
    else:
        h = 50; d = 100

    total = h + d

    features['Hostel_Occupancy'] = h
    features['Day_Scholar_Occupancy'] = d
    features['Total_Occupancy'] = total

    # Extra features (must match dataset)
    features['Water_Price_Index'] = 1.0
    features['Peak_Factor'] = 1.2 if event != 'none' else 1.0

    # Convert to ordered array
    input_data = np.array([[features[f] for f in feature_order]])

    prediction = model.predict(input_data)[0]

    # --- AI Reasoning ---
    reason = f"Predicted {round(prediction)} L based on occupancy and conditions. "
    if event != 'none':
        reason += "Event is major driver. "
    if weather == 'summer':
        reason += "Temperature increases demand. "
    
    # Poetic AI Reasoning Logic
    reason = f"Analyzing the pulse of **{data['campus']}**... "
    if data['event'] != 'none':
        reason += f"The {data['event']} festivities bring a surge of life—and demand. "
    reason += "Remember, every drop saved is a tear unshed for the future. "
    
    return jsonify({
        "prediction": round(prediction),
        "reason": reason,
        "confidence": 0.96
    })

if __name__ == '__main__':
    app.run(debug=True)
