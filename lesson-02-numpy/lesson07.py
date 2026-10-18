import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "area": [1200, 1800, 1000, 2200, 1500, 2000],
    "bedrooms": [3, 4, 2, 4, 3, 4],
    "age": [10, 5, 20, 3, 8, 6],
    "price_lkr": [
        25_000_000, 40_000_000, 20_000_000,
        48_000_000, 32_000_000, 45_000_000
    ]
})

print("Dataset shape:", df.shape)
print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe())

print("\nMedian price:", df["price_lkr"].median())

# df["price_lkr"].plot.hist(bins=5, edgecolor="black")
# plt.title("Distribution of House Prices")
# plt.xlabel("Price (LKR)")
# plt.ylabel("Number of Houses (Frequency)")
# plt.show()

plt.scatter(df["area"], df["price_lkr"])
plt.title("House Price vs Area")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (LKR)")
plt.tight_layout()
plt.show()

print(df.corr(numeric_only=True))