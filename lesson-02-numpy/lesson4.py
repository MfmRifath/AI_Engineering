import numpy as np  # pyright: ignore[reportMissingImports]

houses = (np.array([
    [1200, 3, 250000],
    [1800, 4, 400000],
    [1000, 2, 200000],
    [2200, 4, 480000],
    [1500, 3, 320000],
    [2000, 4, 450000],
]))
print(houses.shape)
print(houses.ndim)

numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

numbers_reshaped = numbers.reshape(3, 4)
print(numbers_reshaped.shape)
print("Average area of the houses:", np.mean(houses[:, 0]))
print("Average number of bedrooms in the houses:", np.mean(houses[:, 1]))
print("Average price of the houses:", np.mean(houses[:, 2]))

print("Column Mean:", np.mean(houses, axis=0))

print("row sum:", np.sum(houses, axis=1))


