import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.DataFrame({
    "area": [
        800, 1000, 1200, 1400, 1600,
        1800, 2000, 2200, 2400, 2600,
        1100, 1700, 2100, 2300, 2500,
        1300, 1500, 1900, 2700, 2900
    ],
    "bedrooms": [
        2, 2, 3, 3, 3,
        4, 4, 4, 5, 5,
        2, 3, 4, 4, 5,
        3, 3, 4, 5, 5
    ],
    "price_lkr": [
        16, 20, 25, 28, 32,
        38, 42, 45, 52, 56,
        22, 34, 44, 47, 54,
        26, 30, 40, 58, 62
    ]
})

print("Dataset shape:", df.shape)
print(df.head())
print("Dataset dimensions:", df.ndim)

X = df[["area", "bedrooms"]]
y = df["price_lkr"]

print("\nFeatures (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPredicted prices:", y_pred)
print("Actual prices:", y_test.values)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
root_mse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", root_mse)
print("R-squared (R2):", r2)

new_house = pd.DataFrame({
    "area": [1750],
    "bedrooms": [3]
})

predicted_price = model.predict(new_house)

print("Predicted price in millions of LKR:",
      predicted_price[0])