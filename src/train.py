import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os


# -----------------------------
# 1. Load Dataset
# -----------------------------
data_path = "data/winequality-red.csv"
df = pd.read_csv(data_path, sep=";")

# -----------------------------
# 2. Pre-processing + Feature Selection
# -----------------------------
X = df.drop("quality", axis=1)
y = df["quality"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Train Model
# (Modify model for each experiment)
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# -----------------------------
# 6. Save Outputs
# -----------------------------
os.makedirs("outputs", exist_ok=True)

# Save trained model
joblib.dump(model, "model.pkl")

# Save metrics JSON
metrics = {"MSE": mse, "R2": r2}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# -----------------------------
# 7. Print Metrics (GitHub Actions Reads This)
# -----------------------------
print(f"MSE: {mse}")
print(f"R2: {r2}")
