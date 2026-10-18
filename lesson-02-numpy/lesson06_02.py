import pandas as pd

df = pd.DataFrame({
    "house": ["A", "B", "C", "C", "D"],
    "area": [1200, None, 1000, 1000, -500],
    "bedrooms": [3, 4, 2, 2, 3],
    "price_lkr": [25_000_000, 40_000_000, 20_000_000,
                  20_000_000, 15_000_000],
})

print("Missing values before cleaning:")
print(df.isna().sum())

# Remove exact duplicate records.
df = df.drop_duplicates()

# Replace missing areas with the median of the available areas.
df["area"] = df["area"].fillna(df["area"].median())

# Keep only records with valid positive area.
df = df[df["area"] > 0].copy()

print("\nCleaned dataset:")
print(df)

print("\nMissing values after cleaning:")
print(df.isna().sum())