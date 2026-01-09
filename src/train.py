import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Create output directory
# -----------------------------
Path("outputs").mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/winequality-red.csv", sep=';')

X = df.drop("quality", axis=1)
y = df["quality"]

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model - (Modify per experiment)
# -----------------------------
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# Print for GitHub Actions
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, "outputs/model.pkl")

# -----------------------------
# Save metrics
# -----------------------------
metrics = {"MSE": mse, "R2": r2}

with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
