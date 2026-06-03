import streamlit as st
import pickle
import numpy as np

# Load Model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Car Price Prediction")

st.title("🚗 Car Price Prediction App")

st.write("Enter Car Details")

present_price = st.number_input(
    "Present Price (Lakhs)",
    min_value=0.0
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0
)

car_age = st.number_input(
    "Car Age",
    min_value=0
)

owner = st.selectbox(
    "Number of Previous Owners",
    [0, 1, 2, 3]
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

# Encoding Inputs

fuel_diesel = 1 if fuel_type == "Diesel" else 0
seller_individual = 1 if seller_type == "Individual" else 0
transmission_manual = 1 if transmission == "Manual" else 0

if st.button("Predict Price"):

    features = np.array([
        [
            present_price,
            kms_driven,
            owner,
            car_age,
            fuel_diesel,
            seller_individual,
            transmission_manual
        ]
    ])

    prediction = model.predict(features)

    st.success(
        f"Estimated Car Price: ₹ {prediction[0]:.2f} Lakhs"
    )
    import pandas as pd
import matplotlib.pyplot as plt

st.header("📊 Data Analysis & Visualization")

df = pd.read_csv("car_data.csv")

# Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Statistics
st.subheader("Statistical Summary")
st.write(df.describe())

# Scatter Plot
st.subheader("Present Price vs Selling Price")

fig1, ax1 = plt.subplots()
ax1.scatter(df["Present_Price"], df["Selling_Price"])
ax1.set_xlabel("Present Price")
ax1.set_ylabel("Selling Price")
ax1.set_title("Present Price vs Selling Price")

st.pyplot(fig1)

# Histogram
st.subheader("Selling Price Distribution")

fig2, ax2 = plt.subplots()
ax2.hist(df["Selling_Price"], bins=10)
ax2.set_xlabel("Selling Price")
ax2.set_ylabel("Frequency")
ax2.set_title("Selling Price Distribution")

st.pyplot(fig2)

# Correlation Matrix
st.subheader("Correlation Matrix")

numeric_df = df.select_dtypes(include=["number"])
st.dataframe(numeric_df.corr())
