# pyright: reportMissingImports=false
import numpy as np

# house = np.array(["A","B","C","D","E"])
# house_areas = np.array([1200, 1800, 1000, 2200, 1500])
# badroom_counts = np.array([3, 4, 2, 4, 3])
# house_prices = np.array([250000, 400000, 200000, 480000, 320000])

# house_areas_in_squre_meters = house_areas * 0.092903
# multiplied_prices = house_prices * 1.1

# print("House Prices Mean:", np.mean(house_prices))
# print("House Total Prices:", np.sum(house_prices))
# print("House Maximum Price:", np.max(house_prices))
# print("House Minimum Price:", np.min(house_prices))

# print("Price of the house C:", house_prices[2])

# print("All Houses grater than 1,400 square feet:", house[house_areas > 1400])
# print("Price of the Houses with at lest 4 bedrooms:", house_prices[badroom_counts >= 4])

houses =np.array([["A", 1200, 3, 250000],
                 ["B", 1800, 4, 400000],
                 ["C", 1000, 2, 200000],
                 ["D", 2200, 4, 480000],
                 ["E", 1500, 3, 320000]])

print("Third House Price:", houses[2, 1])
print("second, third, and fourth houses prices:", houses[1:4, 3])
print("All Houses grater than 1,500 square feet:", houses[houses[:, 1].astype(float) > 1500, 0])
print("the Houses with at least 4 bedrooms:", houses[houses[:, 2].astype(float) >= 4, 0])
print("All Houses Area greater than 1,400 square feet and Price below 45 million LKR .", houses[(houses[:, 1].astype(float) > 1400) & (houses[:, 3].astype(float) < 450000), 0])