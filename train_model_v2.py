import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# 1. Load the RUAS Dataset
try:
    df = pd.read_csv('RUAS_Final_Dataset_10k_v2.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'RUAS_Final_Dataset_10k_v2.csv' not found. Please upload it to Colab.")
    exit()

# 2. Feature Selection
X = df.drop(['Date', 'Campus', 'Water_Usage_L'], axis=1)
y = df['Water_Usage_L']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define REGRESSION Models (Corrected from Classifiers)
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100),
    "KNN": KNeighborsRegressor(),
    "SVM": SVR(),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100)
}

results = []

print("\n🚀 Evaluating AI Engines...")
print("-" * 50)

for name, model in models.items():
    # Create Pipeline with Scaling
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    # Train
    pipe.fit(X_train, y_train)
    
    # Predict & Evaluate
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({"Model": name, "MAE": mae, "R2": r2, "Pipe": pipe})
    print(f"{name:18} | MAE: {mae:8.2f} L | R2: {r2:.4f}")

# 4. Find the Best Model
best_result = max(results, key=lambda x: x['R2'])
print("-" * 50)
print(f"🏆 THE WINNER IS: {best_result['Model']}")
print(f"Final Accuracy: {best_result['R2']*100:.2f}%")

# 5. Save the Winning Model for your Web App
joblib.dump({
    "model": best_result['Pipe'],
    "features": X.columns.tolist()
}, "ruas_water_model_v2.pkl")

print(f"✅ Best model saved as 'ruas_water_model_v2.pkl'")
