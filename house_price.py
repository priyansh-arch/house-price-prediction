import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("dataset.csv")

# Features and target
X = df[["Area_sqft"]]
y = df["Price_INR"]

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
area = int(input("Enter area in sq ft: "))
prediction = model.predict([[area]])

print("Predicted Price (INR):", int(prediction[0]))

# Graph
plt.scatter(df["Area_sqft"], df["Price_INR"])
plt.plot(df["Area_sqft"], model.predict(X))
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (INR)")
plt.title("House Price Prediction")
plt.show()