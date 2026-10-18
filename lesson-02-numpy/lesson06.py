import pandas as pd

data = {    
    "house": ["A", "B", "C", "D"],
    "area": [1200, None, 1000, 2200],
    "bedrooms": [3, 4, 2, 4],
    "price_lkr": [25_000_000, 40_000_000, 20_000_000, 48_000_000],
}   

df = pd.DataFrame(data)

print(df)

print(df.isna().sum())

df_dropped = df.dropna()

print(df)
print(df_dropped)

df_filled = df.copy()
df_filled['area'] = df_filled["area"].fillna(df_filled["area"].median())

print(df_filled)

print(df.duplicated())
print(df.duplicated().sum())

df = df.drop_duplicates()

df = df.drop_duplicates(subset="house")

df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
df["price_lkr"] = pd.to_numeric(df["price_lkr"], errors="coerce")
df['area'] = pd.to_numeric(df['area'], errors='coerce')
invalid_area = df[df['area'] <= 0]

print("Invalid area values:", invalid_area)

