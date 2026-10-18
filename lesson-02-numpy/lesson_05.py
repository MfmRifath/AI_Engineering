import pandas as pd

data = {
    "house": ["A", "B", "C"],
    "area": [1200, 1800, 1000],
    "bedrooms": [3, 4, 2],
    "price_lkr": [25_000_000, 40_000_000, 20_000_000],
}

houses = pd.DataFrame(data)

print(houses)
print(houses.shape)
print(houses.columns)
print(houses.dtypes)

print(houses['area'])
print(houses[['area', 'price_lkr']])

print(houses.loc[0])
print(houses.iloc[0])
print(houses.loc[0, 'area'])