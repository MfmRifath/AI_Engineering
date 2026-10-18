import pandas as pd

data = {
    "house": ["A", "B", "C", "C", "D", "E"],
    "area": [1200, None, 1000, 1000, -500, 1800],
    "bedrooms": [3, 4, 2, 2, 3, 4],
    "price_lkr": [
        25_000_000, 40_000_000, 20_000_000,
        20_000_000, 15_000_000, 35_000_000
    ]
}

df = pd.DataFrame(data)

print(df)
print(df.isna().sum())

print(df.duplicated().sum())
df = df.drop_duplicates()

print(df)

print(df["area"] <= 0)

df = df.drop(df[df["area"] <= 0].index)

df["area"] = df["area"].fillna(df["area"].median())

print(df)


print(df.shape)
print(df.ndim)