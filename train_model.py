import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv("car_data.csv")

# Feature Engineering
df["Car_Age"] = 2025 - df["Year"]

# Remove unnecessary columns
df.drop(["Car_Name", "Year"], axis=1, inplace=True)

# Convert categorical variables
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
score = r2_score(y_test, predictions)

print("Model Accuracy (R² Score):", round(score * 100, 2), "%")

# Save Model
pickle.dump(model, open("model.pkl", "wb"))

print("Model Saved Successfully")