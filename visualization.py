import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("car_data.csv")

# Statistics
print(df.describe())

# Scatter Plot
plt.scatter(
    df["Present_Price"],
    df["Selling_Price"]
)

plt.xlabel("Present Price")
plt.ylabel("Selling Price")
plt.title("Price Relationship")
plt.show()

# Histogram

plt.hist(
    df["Selling_Price"],
    bins=20
)

plt.title("Selling Price Distribution")
plt.show()

# Correlation

corr = df.corr(numeric_only=True)

print(corr)