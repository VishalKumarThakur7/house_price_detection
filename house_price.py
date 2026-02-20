import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset

data = pd.read_csv("data.csv")

print("Dataset:")
print(data)

# Features (input)

X = data[["area", "bedrooms", "bathrooms"]]

# Target (output)

y = data["price"]

# Split data into training and testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model

model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained successfully ")

# Make prediction

new_house = pd.DataFrame([[2000, 3, 2]], columns=["area", "bedrooms", "bathrooms"])
prediction = model.predict(new_house)

print("\nPredicted price for 2000 sq ft house:", prediction[0])
