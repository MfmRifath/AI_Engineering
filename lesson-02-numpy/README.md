# Lesson 02 — NumPy

An introduction to **NumPy**, the core library for numerical computing in Python and the foundation that pandas, scikit-learn, PyTorch, and most of the AI/ML ecosystem are built on.

The example in [main.py](main.py) uses a small housing dataset to show the NumPy basics you'll use all the time when preparing data for machine learning.

**Also in this README:**
- [Lesson 5 — pandas: Working with Tabular Data](#lesson-5--pandas-working-with-tabular-data)
- [Lesson 6 — Data Cleaning and Preprocessing with pandas](#lesson-6--data-cleaning-and-preprocessing-with-pandas)
- [Lesson 07 — Exploratory Data Analysis (EDA)](#lesson-07--exploratory-data-analysis-eda)

---

## What is NumPy?

NumPy (Numerical Python) gives you the **`ndarray`**: a fast, fixed-type, n-dimensional array. Compared with a regular Python list:

| | Python `list` | NumPy `ndarray` |
|---|---|---|
| Element types | Mixed | One type (`int64`, `float64`, ...) |
| Speed | Slow (Python loops) | Fast (C code under the hood) |
| Math on all elements | Needs a loop | One expression (`arr * 2`) |
| Memory | Higher | Compact, contiguous |

```python
[1, 2, 3] * 2           # [1, 2, 3, 1, 2, 3]  -> list repetition
np.array([1, 2, 3]) * 2 # [2, 4, 6]           -> element-wise math
```

---

## Setup

This lesson uses a local virtual environment (`.venv`, Python 3.9, NumPy 2.0.2).

```bash
# create and activate the venv (skip if .venv already exists)
python3 -m venv .venv
source .venv/bin/activate

# install NumPy
pip install numpy

# run the lesson
python main.py
```

---

## Concepts covered in `main.py`

### 1. Creating arrays

```python
import numpy as np

house          = np.array(["A", "B", "C", "D", "E"])
house_areas    = np.array([1200, 1800, 1000, 2200, 1500])        # sq ft
bedroom_counts = np.array([3, 4, 2, 4, 3])
house_prices   = np.array([250000, 400000, 200000, 480000, 320000])
```

Every array holds a single data type. NumPy infers it from the values (strings, `int64`, `float64`, ...). Check it with `arr.dtype`.

### 2. Vectorized operations (broadcasting)

Arithmetic with a scalar applies to **every element**, with no loop needed:

```python
house_areas_in_square_meters = house_areas * 0.092903   # sq ft -> m²
multiplied_prices            = house_prices * 1.1        # +10% price increase
```

This is called **broadcasting**: NumPy "stretches" the scalar to match the array's shape.

### 3. Aggregate functions

```python
np.mean(house_prices)   # 330000.0  average
np.sum(house_prices)    # 1650000   total
np.max(house_prices)    # 480000    highest
np.min(house_prices)    # 200000    lowest
```

Other useful ones: `np.median`, `np.std` (standard deviation), `np.argmax` / `np.argmin` (index of the max/min).

### 4. Indexing

Indexing picks out **single elements** by position.

```text
house_prices:   250000   400000   200000   480000   320000
index:             0        1        2        3        4
negative index:   -5       -4       -3       -2       -1
```

#### Positive and negative indexes

Arrays are **zero-indexed**: the first element is at index `0`. Negative indexes count from the end.

```python
house_prices[0]     # 250000  -> first house (A)
house_prices[2]     # 200000  -> house C
house_prices[-1]    # 320000  -> last house (E)
house_prices[-2]    # 480000  -> second to last (D)
```

An index outside the array raises an error: `house_prices[5]` gives `IndexError`.

#### Fancy indexing (a list of indexes)

Pass a **list of positions** to get several elements at once, in the order you ask for them:

```python
house_prices[[0, 2, 4]]   # [250000 200000 320000]
house[[3, 1]]             # ['D' 'B']
```

#### Indexing 2D arrays

Stack the 1D arrays into a table: one row per house, one column per feature.

```python
data = np.column_stack((house_areas, bedroom_counts, house_prices))
# [[  1200      3 250000]   <- row 0: house A
#  [  1800      4 400000]   <- row 1: house B
#  [  1000      2 200000]   <- row 2: house C
#  [  2200      4 480000]   <- row 3: house D
#  [  1500      3 320000]]  <- row 4: house E
#    area  beds   price

data.shape     # (5, 3) -> 5 rows, 3 columns
```

Use `arr[row, column]`:

```python
data[1]        # [  1800      4 400000] -> whole row for house B
data[1, 2]     # 400000                 -> house B, price column
data[-1, 0]    # 1500                   -> last house, area column
```

> `data[1, 2]` and `data[1][2]` return the same value. `data[1, 2]` is the usual NumPy style and is faster.

### 5. Slicing

Slicing picks out a **range** of elements with the syntax `start:stop:step`.

- `start` is **included** (default `0`)
- `stop` is **excluded** (default: the end of the array)
- `step` is how many positions to move each time (default `1`)

```python
house_prices[1:4]    # [400000 200000 480000] -> indexes 1, 2, 3 (4 is excluded)
house_prices[:3]     # [250000 400000 200000] -> first 3
house_prices[2:]     # [200000 480000 320000] -> from index 2 to the end
house_prices[-3:]    # [200000 480000 320000] -> last 3
house_prices[::2]    # [250000 200000 320000] -> every 2nd element
house_prices[::-1]   # [320000 480000 200000 400000 250000] -> reversed
```

#### Slicing 2D arrays

Slice rows and columns separately: `arr[row_slice, column_slice]`. A bare `:` means "all".

```python
data[:, 0]         # [1200 1800 1000 2200 1500] -> area column (every row)
data[:, 2]         # [250000 400000 200000 480000 320000] -> price column

data[1:3]          # rows 1-2 (houses B and C), all columns
# [[  1800      4 400000]
#  [  1000      2 200000]]

data[1:3, 0:2]     # rows 1-2, columns 0-1 (area and beds)
# [[1800    4]
#  [1000    2]]

data[:, [0, 2]]    # all rows, area and price columns only
```

#### Slices are views, not copies

A slice **shares memory** with the original array. Changing the slice changes the original:

```python
s = house_prices[0:2]
s[0] = 999
print(house_prices)   # [   999 400000 200000 480000 320000]  <- original changed!
```

Use `.copy()` when you need an independent array:

```python
s = house_prices[0:2].copy()
s[0] = 999            # house_prices is unchanged
```

> This differs from Python lists, where slicing always makes a copy.

### 6. Boolean filtering (masking)

Boolean filtering selects elements by **condition** instead of by position. It takes two steps.

**Step 1: a comparison gives a mask.** Comparing an array with a value checks every element and returns an array of `True`/`False` values:

```python
house_areas > 1400
# [False  True False  True  True]
#    A     B     C     D     E
```

**Step 2: index with the mask.** NumPy keeps only the elements where the mask is `True`:

```python
house[house_areas > 1400]
# ['B' 'D' 'E']  -> houses larger than 1,400 sq ft

house_prices[bedroom_counts >= 4]
# [400000 480000]  -> prices of houses with 4+ bedrooms
```

Because all the arrays line up by position, a mask built from **one** array can filter **another**. This pattern shows up all the time in data cleaning and feature selection.

Comparison operators: `>`, `>=`, `<`, `<=`, `==`, `!=`.

#### Combining conditions

| Operator | Meaning | Example | Result |
|---|---|---|---|
| `&` | and | `house[(house_areas > 1400) & (house_prices < 450000)]` | `['B' 'E']` |
| `\|` | or | `house[(bedroom_counts == 2) \| (house_prices > 450000)]` | `['C' 'D']` |
| `~` | not | `house[~(house_areas > 1400)]` | `['A' 'C']` |

> **Two common mistakes:**
> 1. Use `&`, `|`, `~` instead of Python's `and`, `or`, `not`. These operate on whole arrays, and `and`/`or` raise `ValueError: The truth value of an array ... is ambiguous`.
> 2. Always wrap each condition in **parentheses**. `&` binds more tightly than `>`, so `house_areas > 1400 & house_prices < 450000` does not do what you expect.

#### Counting and aggregating with masks

`True` counts as `1` and `False` as `0`, so you can count and summarize matches directly:

```python
np.sum(house_areas > 1400)                  # 3        -> number of houses over 1,400 sq ft
np.mean(house_prices[bedroom_counts == 3])  # 285000.0 -> average price of 3-bedroom houses
np.any(house_prices > 450000)               # True     -> is at least one house over 450k?
np.all(bedroom_counts >= 2)                 # True     -> does every house have 2+ beds?
```

#### Filtering rows of a 2D array

Build a mask from one column and use it to select whole rows:

```python
data[data[:, 2] > 300000]    # every house priced above 300,000
# [[  1800      4 400000]
#  [  2200      4 480000]
#  [  1500      3 320000]]
```

#### Updating values with a mask

Assigning to a masked array changes only the matching elements. Here every price above 400,000 is capped at 400,000:

```python
capped = house_prices.copy()
capped[capped > 400000] = 400000
# [250000 400000 200000 400000 320000]
```

#### `np.where`: label or locate matches

```python
np.where(house_prices > 300000, "Expensive", "Affordable")
# ['Affordable' 'Expensive' 'Affordable' 'Expensive' 'Expensive']

np.where(house_areas > 1400)
# (array([1, 3, 4]),)  -> the indexes of the matching elements
```

### Indexing vs slicing vs Boolean filtering

| | Selects by | Example | Returns |
|---|---|---|---|
| Indexing | One position | `arr[2]` | A single value |
| Fancy indexing | A list of positions | `arr[[0, 2]]` | A new array (copy) |
| Slicing | A range of positions | `arr[1:4]` | A **view** of the original |
| Boolean filtering | A condition | `arr[arr > 10]` | A new array (copy) |

### 7. Reshaping

#### Shape, dimensions, and size

Every array has a **shape**: the number of elements along each dimension.

```python
house_prices.shape   # (5,)   -> 1D: 5 elements
house_prices.ndim    # 1      -> number of dimensions
house_prices.size    # 5      -> total number of elements

data.shape           # (5, 3) -> 2D: 5 rows, 3 columns
data.ndim            # 2
data.size            # 15     -> 5 × 3
```

> `(5,)` (with a comma) is a 1D array of 5 elements. `(5, 1)` is a 2D array with 5 rows and 1 column. They hold the same values but have different shapes.

#### `reshape`: rearrange elements into a new shape

`reshape` keeps the same elements in the same order and arranges them into new rows and columns.

```python
a = np.arange(12)
# [ 0  1  2  3  4  5  6  7  8  9 10 11]

a.reshape(3, 4)        # 3 rows × 4 columns
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

a.reshape(2, 6)        # 2 rows × 6 columns
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]]
```

Elements fill the new shape **row by row**.

**The total size must stay the same.** 12 elements fit into `(3, 4)`, `(2, 6)`, `(4, 3)` or `(6, 2)`, but not `(5, 2)`:

```python
a.reshape(5, 2)
# ValueError: cannot reshape array of size 12 into shape (5,2)
```

#### `-1`: let NumPy work out one dimension

Put `-1` in one position and NumPy calculates it from the array's size:

```python
a.reshape(4, -1).shape   # (4, 3)  -> 12 / 4 = 3 columns
a.reshape(-1, 2).shape   # (6, 2)  -> 12 / 2 = 6 rows
```

#### Row and column vectors

Many ML libraries, such as scikit-learn, expect features as a **2D column** with one row per sample. `reshape(-1, 1)` converts a 1D array into that form:

```python
house_prices.reshape(-1, 1)    # shape (5, 1) -> column vector
# [[250000]
#  [400000]
#  [200000]
#  [480000]
#  [320000]]

house_prices.reshape(1, -1)    # shape (1, 5) -> row vector
# [[250000 400000 200000 480000 320000]]

house_prices[:, np.newaxis].shape   # (5, 1) -> another way to add a dimension
```

#### Flattening and transposing

```python
m = a.reshape(3, 4)
m.flatten()   # [ 0  1  2 ... 11] -> back to 1D (always a copy)
m.ravel()     # [ 0  1  2 ... 11] -> back to 1D (a view when possible, so faster)

data.T        # transpose: rows become columns
# [[  1200   1800   1000   2200   1500]   <- areas
#  [     3      4      2      4      3]   <- bedrooms
#  [250000 400000 200000 480000 320000]]  <- prices
data.T.shape  # (3, 5)
```

> Like slices, `reshape` usually returns a **view**. Changing the reshaped array also changes the original. Call `.copy()` if you need an independent array.

### 8. Axes

An **axis** is a direction along an array's dimensions. A 2D array has two:

```text
                axis 1 → (across columns)
              ┌──────────────────────────┐
  axis 0  ↓   │  1200      3   250000    │  house A
  (down       │  1800      4   400000    │  house B
   rows)      │  1000      2   200000    │  house C
              │  2200      4   480000    │  house D
              │  1500      3   320000    │  house E
              └──────────────────────────┘
                 area   beds    price
```

- **`axis=0`** works **down the rows** and gives **one result per column**.
- **`axis=1`** works **across the columns** and gives **one result per row**.

> **How to remember it:** the axis you name is the one that **disappears**. `data` has shape `(5, 3)`. `axis=0` removes the 5 rows and gives shape `(3,)`, while `axis=1` removes the 3 columns and gives shape `(5,)`.

#### Example: price history

Here each house's price is recorded over 3 years (one row per house, one column per year):

```python
price_history = np.array([
    [250000, 265000, 280000],   # A
    [400000, 410000, 430000],   # B
    [200000, 195000, 210000],   # C
    [480000, 500000, 525000],   # D
    [320000, 330000, 345000],   # E
])                              # year1  year2  year3
price_history.shape             # (5, 3)
```

```python
price_history.mean(axis=0)   # [330000. 340000. 358000.]
# -> average market price for each YEAR (one value per column)

price_history.mean(axis=1)   # [265000. 413333.33 201666.67 501666.67 331666.67]
# -> average price for each HOUSE across the 3 years (one value per row)

price_history.mean()         # 342666.67
# -> no axis: average of all 15 values
```

On the house table, `axis=0` gives a summary of each feature:

```python
data.mean(axis=0)   # area 1540, beds 3.2, price 330000
data.max(axis=0)    # [  2200      4 480000] -> largest value in each column
data.min(axis=0)    # [  1000      2 200000] -> smallest value in each column
```

#### `keepdims=True`: keep the shape for broadcasting

Aggregating along an axis removes that dimension. Pass `keepdims=True` to keep it as size 1, so the result lines up with the original array:

```python
price_history.mean(axis=1).shape                  # (5,)
price_history.mean(axis=1, keepdims=True).shape   # (5, 1)

# how far each year's price is from that house's average
price_history - price_history.mean(axis=1, keepdims=True)
```

The `(5, 1)` result broadcasts across the 3 columns, so each row has its own average subtracted. Without `keepdims` the shapes `(5, 3)` and `(5,)` don't line up and NumPy raises an error.

### 9. Aggregation

Aggregation **reduces many values to fewer**, for example a whole array to one number or a table to one value per row or column. Section 3 covered the basics on 1D arrays. Here's the wider toolkit:

| Function | What it returns | `house_prices` result |
|---|---|---|
| `np.sum` | Total | `1650000` |
| `np.mean` | Average | `330000.0` |
| `np.median` | Middle value | `320000.0` |
| `np.std` | Standard deviation (spread) | `100796.83` |
| `np.min` / `np.max` | Smallest / largest | `200000` / `480000` |
| `np.argmin` / `np.argmax` | **Index** of the smallest / largest | `2` / `3` |
| `np.ptp` | Range (max − min) | `280000` |
| `np.percentile(arr, 50)` | Value at a given percentile | `320000.0` |
| `np.cumsum` | Running total (not reduced) | `[250000 650000 850000 1330000 1650000]` |

#### Function or method: same result

```python
np.sum(house_prices)   # 1650000
house_prices.sum()     # 1650000  -> the same; use whichever reads better
```

#### `argmax` / `argmin`: find which item

These return a **position**, which you can use to look up a matching name in another array:

```python
house[np.argmax(house_prices)]   # 'D' -> most expensive house
house[np.argmin(house_prices)]   # 'C' -> cheapest house
```

#### Aggregation + axes

Every aggregate function accepts `axis`:

```python
price_history.max(axis=0)      # [480000 500000 525000]          -> top price each year
price_history.min(axis=1)      # [250000 400000 195000 480000 320000] -> lowest price per house
house[price_history.argmax(axis=0)]   # ['D' 'D' 'D'] -> most expensive house each year

growth = price_history[:, -1] - price_history[:, 0]   # last year − first year
# [30000 30000 10000 45000 25000]
house[np.argmax(growth)]       # 'D' -> house whose price grew the most
```

#### Aggregation + Boolean filtering

Filter first, then aggregate (see section 6):

```python
np.mean(house_prices[bedroom_counts == 3])   # 285000.0 -> average price of 3-bedroom houses
np.sum(house_areas > 1400)                   # 3        -> how many houses are over 1,400 sq ft
```

#### Summary: reshaping vs axes vs aggregation

| Concept | Question it answers | Example |
|---|---|---|
| Reshaping | "How are the values arranged?" | `arr.reshape(-1, 1)` |
| Axes | "Which direction do I calculate along?" | `axis=0` per column, `axis=1` per row |
| Aggregation | "What single value summarizes these?" | `arr.mean(axis=0)` |

---

## Expected output

```text
House Prices Mean: 330000.0
House Total Prices: 1650000
House Maximum Price: 480000
House Minimum Price: 200000
Price of the house C: 200000
All Houses grater than 1,400 square feet: ['B' 'D' 'E']
Price of the Houses with at lest 4 bedrooms: [400000 480000]
```

---

## Quick reference

| Task | Code |
|---|---|
| Create from list | `np.array([1, 2, 3])` |
| Zeros / ones | `np.zeros(5)`, `np.ones((2, 3))` |
| Range | `np.arange(0, 10, 2)` |
| Evenly spaced | `np.linspace(0, 1, 5)` |
| Shape / size / type | `arr.shape`, `arr.size`, `arr.dtype` |
| Reshape | `arr.reshape(2, 3)`, `arr.reshape(-1, 1)` |
| Transpose / flatten | `arr.T`, `arr.ravel()` |
| Aggregate per column / row | `arr.sum(axis=0)`, `arr.mean(axis=1)` |
| Index of max / min | `np.argmax(arr)`, `np.argmin(arr)` |
| Element-wise math | `arr + 1`, `arr * arr`, `np.sqrt(arr)` |
| Stats | `np.mean`, `np.median`, `np.std`, `np.sum` |
| Filter | `arr[arr > 10]` |
| Sort | `np.sort(arr)`, `np.argsort(arr)` |

---

## Why this matters for AI engineering

- **Datasets are arrays.** Features, labels, images, and embeddings are all represented as NumPy-style arrays.
- **Speed.** Vectorized code runs much faster than Python loops on large data.
- **Interoperability.** pandas DataFrames, scikit-learn models, and PyTorch/TensorFlow tensors convert to and from NumPy arrays easily.

## Practice ideas

1. Calculate the **price per square foot** for each house (`house_prices / house_areas`).
2. Find the name of the most expensive house using `np.argmax`.
3. List houses with 3 bedrooms **and** a price under 300,000.
4. Combine the arrays into a 2D array with `np.column_stack` and print its `.shape`.

## Resources

- [NumPy official docs](https://numpy.org/doc/stable/)
- [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)

---

# Lesson 5 — pandas: Working with Tabular Data

**pandas** is a Python library for working with **tables**: rows and columns of data, like a spreadsheet or a SQL table. It's built on NumPy, so everything from the earlier lessons still applies, but pandas adds:

- **Column names**, so you write `houses["price_lkr"]` instead of `houses[:, 3]`
- **Mixed types in one table**: text, integers and floats each keep their own type in separate columns
- **Built-in tools** for loading CSV and Excel files, handling missing values, grouping, sorting and summarizing

The example in [lesson_05.py](lesson_05.py) uses the same housing data with prices in LKR.

> **Why pandas after NumPy?** In [main.py](main.py), mixing house names and numbers in one NumPy array turned **everything into strings**, so every comparison needed `.astype(float)`. pandas fixes this by giving every column its own type.

## Setup

pandas is already installed in this lesson's `.venv` (pandas 2.3.3). To install it in a new environment:

```bash
source .venv/bin/activate
pip install pandas
python lesson_05.py
```

## The two building blocks

| | What it is | NumPy equivalent |
|---|---|---|
| **`Series`** | One **column**: a 1D array of values with labels (the index) | 1D `ndarray` |
| **`DataFrame`** | A **table**: several Series that share the same index | 2D `ndarray`, but each column can have a different type |

```text
              columns →
         house  area  bedrooms  price_lkr
index  0     A  1200         3   25000000    ← row
  ↓    1     B  1800         4   40000000
       2     C  1000         2   20000000
                 ↑
          one column = a Series
```

## Concepts covered in `lesson_05.py`

### 1. Creating a DataFrame from a dictionary

Each **key** becomes a column name and each **list** becomes that column's values:

```python
import pandas as pd

data = {
    "house": ["A", "B", "C"],
    "area": [1200, 1800, 1000],
    "bedrooms": [3, 4, 2],
    "price_lkr": [25_000_000, 40_000_000, 20_000_000],
}

houses = pd.DataFrame(data)
print(houses)
```

```text
  house  area  bedrooms  price_lkr
0     A  1200         3   25000000
1     B  1800         4   40000000
2     C  1000         2   20000000
```

- The numbers on the left (`0`, `1`, `2`) are the **index**: row labels pandas adds automatically.
- `25_000_000` is the same as `25000000`. Python ignores the underscores, which only make long numbers easier to read.
- All the lists must be the **same length**. Otherwise you get `ValueError: All arrays must be of the same length`.

### 2. Inspecting the structure

```python
houses.shape     # (3, 4) -> 3 rows, 4 columns (same as NumPy)
houses.columns   # Index(['house', 'area', 'bedrooms', 'price_lkr'], dtype='object')
houses.dtypes
# house        object   <- text
# area          int64
# bedrooms      int64
# price_lkr     int64
```

Each column keeps **its own type**: `house` is text (`object`) and the rest are integers. A single NumPy array can't do this.

### 3. Selecting columns

**One column** (single brackets) returns a **Series**:

```python
houses["area"]
# 0    1200
# 1    1800
# 2    1000
# Name: area, dtype: int64
```

**Several columns** (a list inside brackets, so **double brackets**) returns a **DataFrame**:

```python
houses[["area", "price_lkr"]]
#    area  price_lkr
# 0  1200   25000000
# 1  1800   40000000
# 2  1000   20000000
```

> `houses["area"]` → Series (1D). `houses[["area"]]` → DataFrame with one column (2D). The extra brackets matter.

### Expected output of `lesson_05.py`

```text
  house  area  bedrooms  price_lkr
0     A  1200         3   25000000
1     B  1800         4   40000000
2     C  1000         2   20000000
(3, 4)
Index(['house', 'area', 'bedrooms', 'price_lkr'], dtype='object')
house        object
area          int64
bedrooms      int64
price_lkr     int64
dtype: object
0    1200
1    1800
2    1000
Name: area, dtype: int64
   area  price_lkr
0  1200   25000000
1  1800   40000000
2  1000   20000000
```

## Going further

The examples below use a larger table with 5 houses and a `city` column:

```python
df = pd.DataFrame({
    "house":     ["A", "B", "C", "D", "E"],
    "city":      ["Colombo", "Kandy", "Galle", "Colombo", "Kandy"],
    "area":      [1200, 1800, 1000, 2200, 1500],
    "bedrooms":  [3, 4, 2, 4, 3],
    "price_lkr": [25_000_000, 40_000_000, 20_000_000, 48_000_000, 32_000_000],
})
```

### 4. Quick look at the data

| Method | What it shows |
|---|---|
| `df.head(n)` | First `n` rows (default 5) |
| `df.tail(n)` | Last `n` rows |
| `df.info()` | Column names, types, non-null counts, memory |
| `df.describe()` | Statistics for every numeric column |

```python
df.describe()
#               area  bedrooms     price_lkr
# count     5.000000   5.00000  5.000000e+00
# mean   1540.000000   3.20000  3.300000e+07
# std     477.493455   0.83666  1.126943e+07
# min    1000.000000   2.00000  2.000000e+07
# 25%    1200.000000   3.00000  2.500000e+07
# 50%    1500.000000   3.00000  3.200000e+07
# 75%    1800.000000   4.00000  4.000000e+07
# max    2200.000000   4.00000  4.800000e+07
```

> `3.300000e+07` is scientific notation for 3.3 × 10⁷, which is 33,000,000.

### 5. Selecting rows: `loc` and `iloc`

| | Selects by | Example |
|---|---|---|
| **`loc`** | **Labels**: index labels and column names | `df.loc[1, "price_lkr"]` |
| **`iloc`** | **Positions**: integer positions, like NumPy | `df.iloc[1, 4]` |

```python
df.loc[1, "price_lkr"]              # 40000000 -> row label 1, price column
df.loc[0:2, ["house", "price_lkr"]] # rows 0, 1, AND 2 (3 rows)
df.iloc[0:2, 0:3]                   # rows 0 and 1 only, first 3 columns
df.iloc[-1]["house"]                # 'E' -> last row
```

> **Watch out:** a `loc` slice **includes** the end label (`0:2` gives 3 rows). An `iloc` slice **excludes** the end, like NumPy and Python (`0:2` gives 2 rows).

### 6. Boolean filtering

This works the same way as in NumPy (section 6 above), but you use column names instead of `.astype(float)`:

```python
df[df["area"] > 1400]
#   house     city  area  bedrooms  price_lkr
# 1     B    Kandy  1800         4   40000000
# 3     D  Colombo  2200         4   48000000
# 4     E    Kandy  1500         3   32000000

df[(df["area"] > 1400) & (df["price_lkr"] < 45_000_000)]   # houses B and E

df[df["city"].isin(["Colombo", "Galle"])]                  # houses A, C, D

df.loc[df["bedrooms"] >= 4, "house"]                       # B, D -> filter rows, pick a column
```

The same rules as NumPy apply: use `&`, `|` and `~`, and wrap each condition in parentheses.

Compare this with the NumPy version from [main.py](main.py):

```python
# NumPy: column numbers + type conversion
houses[houses[:, 1].astype(float) > 1400, 0]

# pandas: column names, no conversion
df.loc[df["area"] > 1400, "house"]
```

### 7. Adding and changing columns

Operations on a column are **vectorized**, just like NumPy arrays:

```python
df["price_per_sqft"] = df["price_lkr"] / df["area"]
df["is_large"] = df["area"] > 1500
#   house  ...  price_per_sqft  is_large
# 0     A  ...         20833.3     False
# 1     B  ...         22222.2      True
# 2     C  ...         20000.0     False
# 3     D  ...         21818.2      True
# 4     E  ...         21333.3     False

df = df.drop(columns=["is_large"])                 # remove a column
df = df.rename(columns={"price_lkr": "price"})     # rename a column
```

> `drop` and `rename` return a **new** DataFrame. Assign the result back (`df = ...`) to keep the change.

### 8. Sorting

```python
df.sort_values("price_lkr", ascending=False)   # most expensive first: D, B, E, A, C
df.sort_values(["city", "price_lkr"])          # sort by city, then by price within each city
```

### 9. Grouping and aggregation

`groupby` **splits** the rows into groups, **applies** a calculation to each group, and **combines** the results. It's like `axis=0` aggregation from NumPy, but done separately for each category:

```python
df.groupby("city")["price_lkr"].mean()
# city
# Colombo    36500000.0
# Galle      20000000.0
# Kandy      36000000.0
```

Use `agg` to calculate several summaries at once:

```python
df.groupby("city").agg(
    houses=("house", "count"),
    avg_price=("price_lkr", "mean"),
    max_area=("area", "max"),
)
#          houses   avg_price  max_area
# city
# Colombo       2  36500000.0      2200
# Galle         1  20000000.0      1000
# Kandy         2  36000000.0      1800

df["city"].value_counts()   # Colombo 2, Kandy 2, Galle 1 -> rows per category
```

### 10. Missing values

Real data often has gaps. pandas marks them as `NaN` ("not a number"):

```python
df.isna().sum()                              # count of missing values per column
df["area"].fillna(df["area"].mean())         # replace missing values with the column average
df.dropna()                                  # remove rows that have any missing value
```

> A missing value turns an `int64` column into `float64`, because `NaN` is a float.

### 11. Loading and saving files

```python
df = pd.read_csv("houses.csv")            # load a CSV file into a DataFrame
df.to_csv("houses.csv", index=False)      # save; index=False skips the 0, 1, 2... column
df = pd.read_excel("houses.xlsx")         # Excel files need: pip install openpyxl
```

### 12. Back to NumPy

ML libraries usually expect NumPy arrays. `to_numpy()` converts a DataFrame or Series:

```python
X = df[["area", "bedrooms"]].to_numpy()   # features -> 2D array, shape (5, 2)
y = df["price_lkr"].to_numpy()            # target   -> 1D array, shape (5,)
```

This `X` (features) and `y` (target) split is the standard input format for scikit-learn models.

## NumPy vs pandas

| | NumPy `ndarray` | pandas `DataFrame` |
|---|---|---|
| Access columns by | Position: `arr[:, 1]` | Name: `df["area"]` |
| Mixed types | ❌ Converts everything to one type | ✅ Each column has its own type |
| Row labels | Positions only | An index (labels) |
| Missing values | No built-in support | `isna`, `fillna`, `dropna` |
| File I/O | Basic | CSV, Excel, JSON, SQL, ... |
| Speed for pure math | Fastest | Slightly slower (built on NumPy) |
| Best for | Numerical computation, ML model input | Loading, cleaning and exploring data |

**Typical workflow:** load and clean the data with **pandas**, then convert it with `.to_numpy()` and train with **NumPy**-based libraries.

## pandas quick reference

| Task | Code |
|---|---|
| Create from dict | `pd.DataFrame({"a": [1, 2], "b": [3, 4]})` |
| Load / save CSV | `pd.read_csv("f.csv")`, `df.to_csv("f.csv", index=False)` |
| Size / columns / types | `df.shape`, `df.columns`, `df.dtypes` |
| Preview | `df.head()`, `df.tail()`, `df.info()`, `df.describe()` |
| One column (Series) | `df["col"]` |
| Several columns | `df[["col1", "col2"]]` |
| Row by label / position | `df.loc[label]`, `df.iloc[pos]` |
| Filter rows | `df[df["col"] > 10]` |
| New column | `df["new"] = df["a"] / df["b"]` |
| Sort | `df.sort_values("col", ascending=False)` |
| Group and summarize | `df.groupby("col")["val"].mean()` |
| Missing values | `df.isna().sum()`, `df.fillna(0)`, `df.dropna()` |
| To NumPy | `df.to_numpy()` |

## Practice ideas

1. Add the houses D and E to `data` in [lesson_05.py](lesson_05.py) and print the new `shape`.
2. Add a `price_per_sqft` column and find the house with the lowest value using `idxmin()`.
3. Show only the houses with 3+ bedrooms priced under 35,000,000 LKR.
4. Sort the houses by area from largest to smallest.
5. Add a `city` column and find the average price per city with `groupby`.

## Resources

- [pandas official docs](https://pandas.pydata.org/docs/)
- [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)

---

# Lesson 6 — Data Cleaning and Preprocessing with pandas

Real-world data is messy: values are missing, rows are duplicated, numbers are stored as text, and some values are impossible (a house with `-500` sq ft). A model trained on bad data gives bad predictions (**"garbage in, garbage out"**), so cleaning usually takes more of an AI engineer's time than training.

- **Data cleaning** fixes problems in the data: missing values, duplicates, wrong types and invalid values.
- **Preprocessing** turns clean data into the form a model needs: numbers only, on similar scales, with useful features.

## Files in this lesson

| File | What it shows |
|---|---|
| [lesson06.py](lesson06.py) | Each cleaning tool on its own: `isna`, `dropna`, `fillna`, `duplicated`, `drop_duplicates`, `to_numeric` |
| [lesson06_02.py](lesson06_02.py) | A short cleaning pipeline: remove duplicates → fill missing → remove invalid |
| [lesson06_ex.py](lesson06_ex.py) | Exercise: the same steps in a better order (remove invalid **before** filling) |

The examples below use the messy dataset from [lesson06_ex.py](lesson06_ex.py):

```python
import pandas as pd

df = pd.DataFrame({
    "house":     ["A", "B", "C", "C", "D", "E"],
    "area":      [1200, None, 1000, 1000, -500, 1800],
    "bedrooms":  [3, 4, 2, 2, 3, 4],
    "price_lkr": [25_000_000, 40_000_000, 20_000_000,
                  20_000_000, 15_000_000, 35_000_000],
})
```

```text
  house    area  bedrooms  price_lkr
0     A  1200.0         3   25000000
1     B     NaN         4   40000000    <- missing area
2     C  1000.0         2   20000000
3     C  1000.0         2   20000000    <- duplicate of row 2
4     D  -500.0         3   15000000    <- impossible area
5     E  1800.0         4   35000000
```

> `None` becomes **`NaN`** in pandas, and the `area` column changes from `int64` to **`float64`** because `NaN` is a float. That's why the areas print as `1200.0`.

## 1. Missing values

### Find them

```python
df.isna()                    # True/False for every cell
df.isna().sum()              # missing values per column
# house        0
# area         1
# bedrooms     0
# price_lkr    0

df[df.isna().any(axis=1)]    # show the rows that have any missing value
#   house  area  bedrooms  price_lkr
# 1     B   NaN         4   40000000
```

`axis=1` checks **across** each row, the same as in NumPy (section 8).

### Option A: drop them

```python
df_dropped = df.dropna()                # remove rows with ANY missing value
df.dropna(subset=["area"])              # only check specific columns
```

> `dropna()` returns a **new** DataFrame and leaves `df` unchanged. In [lesson06.py](lesson06.py), `print(df)` after `dropna()` still shows house B. Assign the result (`df = df.dropna()`) to keep the change.

**When to use it:** only a few rows are missing and you can afford to lose them. Dropping too much loses data your model could learn from.

### Option B: fill them (imputation)

```python
df["area"] = df["area"].fillna(df["area"].median())   # with the median
df["area"].fillna(df["area"].mean())                  # with the mean
df["bedrooms"].fillna(df["bedrooms"].mode()[0])       # with the most common value
df["area"].fillna(0)                                  # with a fixed value
```

| Fill with | Best for | Watch out |
|---|---|---|
| **Median** | Numeric columns, especially with outliers | Usually the safest default |
| **Mean** | Numeric columns without outliers | Pulled toward extreme values |
| **Mode** | Categories (`city`) or small whole numbers (`bedrooms`) | `mode()` returns a Series, so take `[0]` |
| **A constant** | When "missing" has a meaning (`0` = no garage) | `0` for area would be wrong here |

**Why median over mean?** Before cleaning, the `-500` row drags the mean down:

```python
df["area"].mean()     # 900.0   <- pulled down by -500
df["area"].median()   # 1000.0  <- less affected
```

> Always assign the result back: `df["area"] = df["area"].fillna(...)`. Calling `fillna` on its own doesn't change `df`.

## 2. Duplicate rows

```python
df.duplicated()          # True for each row that repeats an earlier row
# [False, False, False, True, False, False]   <- row 3 repeats row 2

df.duplicated().sum()    # 1 -> how many duplicates

df = df.drop_duplicates()   # keep the first copy, remove the rest
```

| Option | Effect |
|---|---|
| `drop_duplicates()` | A row counts as a duplicate only if **every column** matches |
| `drop_duplicates(subset="house")` | Rows count as duplicates if just the **`house`** column matches |
| `keep="first"` (default) / `keep="last"` | Which copy to keep |
| `duplicated(keep=False)` | Marks **all** copies, so you can inspect them: `[F, F, True, True, F, F]` |

> Use `subset` carefully. Two different houses can share a value, such as the same area. Use it only for columns that should be unique, like an ID.

## 3. Fixing data types

Numbers often arrive as **text**, from CSV files, forms or web scraping. Math and comparisons don't work on text, just like the `.astype(float)` problem in [main.py](main.py).

```python
messy = pd.Series(["1200", "1,800", "abc", " 1000 ", None])

pd.to_numeric(messy)
# ValueError: Unable to parse string "1,800" at position 1

pd.to_numeric(messy, errors="coerce")
# [1200.0, nan, nan, 1000.0, nan]   <- anything unparseable becomes NaN
```

`errors="coerce"` turns bad values into `NaN`, which you can then handle in step 1. Clean the text first to save more values:

```python
pd.to_numeric(messy.str.replace(",", "").str.strip(), errors="coerce")
# [1200.0, 1800.0, nan, 1000.0, nan]   <- "1,800" is saved
```

| Method | On a bad value |
|---|---|
| `astype(int)` | Raises an error |
| `pd.to_numeric(..., errors="raise")` (default) | Raises an error |
| `pd.to_numeric(..., errors="coerce")` | Turns it into `NaN` |

> In [lesson06.py](lesson06.py), the columns are already numeric, so `to_numeric` changes nothing. It's useful once the data comes from a file.

## 4. Invalid values

Some values are present and numeric but **impossible**: a negative area, 0 bedrooms, or a price of 1 LKR. Check them with **rules** that make sense for the data:

```python
df[df["area"] <= 0]                    # show the invalid rows
df["area"].between(1, 10_000)          # True if area is in a sensible range
```

There are two ways to remove invalid rows, and they treat `NaN` differently:

```python
# Way 1: keep the valid rows
df[df["area"] > 0]                      # A, C, C, E    <- B is gone too!

# Way 2: drop the invalid rows
df.drop(df[df["area"] <= 0].index)      # A, B, C, C, E <- B is kept
```

> **Any comparison with `NaN` is `False`.** `NaN > 0` is `False`, so Way 1 also removes house B. `NaN <= 0` is also `False`, so Way 2 keeps B so it can be filled later. To keep missing rows with Way 1, say so explicitly: `df[(df["area"] > 0) | df["area"].isna()]`.

## 5. Order matters

[lesson06_02.py](lesson06_02.py) and [lesson06_ex.py](lesson06_ex.py) use the same steps in a different order, and they give house B a **different area**:

| | [lesson06_02.py](lesson06_02.py) | [lesson06_ex.py](lesson06_ex.py) |
|---|---|---|
| Step order | duplicates → **fill** → remove invalid | duplicates → **remove invalid** → fill |
| Areas used for the median | `1200, 1000, -500` | `1200, 1000, 1800` |
| House B's area | **1000.0** (the `-500` pulled it down) | **1200.0** ✅ |

When the median is calculated **before** removing invalid data, the bad value leaks into the filled value. The rule:

> **Remove bad data first, then calculate statistics (median, mean) from what's left.**

A sensible cleaning order:

1. **Fix types** (`to_numeric`) so comparisons work
2. **Remove duplicates**
3. **Remove invalid values** (rule checks)
4. **Fill missing values** using statistics from the clean rows
5. **Reset the index**
6. **Check the result** (`isna().sum()`, `shape`, `describe()`)

## 6. Resetting the index

Removing rows leaves **gaps** in the index. After cleaning, `lesson06_ex.py` has index `0, 1, 2, 5`:

```python
df = df.reset_index(drop=True)   # renumber the rows 0, 1, 2, 3
```

`drop=True` throws away the old index. Without it, pandas keeps the old index as a new `index` column.

> This matters with `loc`. After cleaning, `df.loc[3]` raises a `KeyError` until you reset the index, because label `3` was removed.

## 7. Preprocessing for machine learning

After cleaning, the data from `lesson06_ex.py` looks like this:

```text
  house    area  bedrooms  price_lkr
0     A  1200.0         3   25000000
1     B  1200.0         4   40000000
2     C  1000.0         2   20000000
3     E  1800.0         4   35000000
```

### Cleaning text

Text columns often differ in spacing and capitalization, so pandas treats `"Colombo"` and `" colombo"` as different values:

```python
cities = pd.Series([" Colombo", "colombo", "KANDY ", "Kandy"])
cities.str.strip().str.title()
# ['Colombo', 'Colombo', 'Kandy', 'Kandy']
```

The `.str` accessor applies a string method to every value: `strip`, `lower`, `upper`, `title`, `replace`, `contains` and more.

### Encoding categories

Models need numbers, not text. **One-hot encoding** creates a 0/1 column for each category:

```python
x = pd.DataFrame({"house": ["A", "B", "C"],
                  "city":  ["Colombo", "Kandy", "Colombo"]})

pd.get_dummies(x, columns=["city"], dtype=int)
#   house  city_Colombo  city_Kandy
# 0     A             1           0
# 1     B             0           1
# 2     C             1           0
```

> Don't number categories as `Colombo = 1, Kandy = 2, Galle = 3`. The model would assume Galle is "bigger" than Colombo. One-hot encoding avoids this.

### Feature engineering

Feature engineering means creating new, more useful columns from existing ones:

```python
df["price_per_sqft"] = df["price_lkr"] / df["area"]
# A 20833.3, B 33333.3, C 20000.0, E 19444.4
```

### Scaling

`area` (about 1,000) and `bedrooms` (about 3) have very different scales. Many models work better when features are on a **similar scale**:

```python
area = df["area"]   # [1200, 1200, 1000, 1800]

# Min-max scaling -> values between 0 and 1
(area - area.min()) / (area.max() - area.min())
# [0.25, 0.25, 0.0, 1.0]

# Standardization (z-score) -> mean 0, standard deviation 1
(area - area.mean()) / area.std()
# [-0.29, -0.29, -0.87, 1.44]
```

| Method | Result | Use when |
|---|---|---|
| **Min-max** | Range 0 to 1 | You need a fixed range (e.g., neural network inputs, images) |
| **Standardization** | Mean 0, std 1 | The default for most models; less affected by outliers |

> Later you'll use scikit-learn's `MinMaxScaler` and `StandardScaler`, which do the same thing. The formulas above show what's happening inside them.

### Final step: features and target

```python
X = df[["area", "bedrooms"]].to_numpy()   # features, shape (4, 2)
y = df["price_lkr"].to_numpy()            # target,   shape (4,)
```

## Expected output

### `lesson06_02.py`

```text
Missing values before cleaning:
house        0
area         1
bedrooms     0
price_lkr    0
dtype: int64

Cleaned dataset:
  house    area  bedrooms  price_lkr
0     A  1200.0         3   25000000
1     B  1000.0         4   40000000
2     C  1000.0         2   20000000

Missing values after cleaning:
house        0
area         0
bedrooms     0
price_lkr    0
dtype: int64
```

### `lesson06_ex.py` (final table and shape)

```text
  house    area  bedrooms  price_lkr
0     A  1200.0         3   25000000
1     B  1200.0         4   40000000
2     C  1000.0         2   20000000
5     E  1800.0         4   35000000
(4, 4)
2
```

The index `0, 1, 2, 5` shows the gaps left by the removed rows (see section 6).

## Data cleaning checklist

- [ ] Look at the data first: `head()`, `info()`, `describe()`
- [ ] Every column has the right type (`dtypes`)
- [ ] No duplicates (`duplicated().sum() == 0`)
- [ ] No impossible values (rule checks such as `area > 0`)
- [ ] Missing values dropped or filled **after** removing invalid rows
- [ ] Index reset (`reset_index(drop=True)`)
- [ ] Final check: `isna().sum()` is all zeros, and `shape` is what you expect

## Cleaning quick reference

| Task | Code |
|---|---|
| Count missing | `df.isna().sum()` |
| Rows with missing values | `df[df.isna().any(axis=1)]` |
| Drop missing | `df.dropna()`, `df.dropna(subset=["col"])` |
| Fill missing | `df["col"] = df["col"].fillna(df["col"].median())` |
| Count / remove duplicates | `df.duplicated().sum()`, `df.drop_duplicates()` |
| Text → number | `pd.to_numeric(df["col"], errors="coerce")` |
| Clean text | `df["col"].str.strip().str.lower()` |
| Find invalid rows | `df[df["col"] <= 0]` |
| Drop invalid rows | `df.drop(df[df["col"] <= 0].index)` |
| Renumber rows | `df.reset_index(drop=True)` |
| One-hot encode | `pd.get_dummies(df, columns=["col"], dtype=int)` |
| Min-max scale | `(s - s.min()) / (s.max() - s.min())` |
| Standardize | `(s - s.mean()) / s.std()` |

## Practice ideas

1. In [lesson06_02.py](lesson06_02.py), move the invalid-area step **before** `fillna` and check that house B's area changes from `1000.0` to `1200.0`.
2. Add `df = df.reset_index(drop=True)` at the end of [lesson06_ex.py](lesson06_ex.py) and check the index is `0, 1, 2, 3`.
3. Add a `price_lkr` value stored as text, such as `"30,000,000"`, and fix it with `str.replace` + `pd.to_numeric`.
4. Add a `city` column with messy spelling (`" colombo"`, `"COLOMBO"`), clean it with `.str`, then one-hot encode it.
5. Add a rule that removes houses with `bedrooms <= 0` or `price_lkr <= 0`.

## Resources

- [pandas: Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [pandas: `DataFrame.drop_duplicates`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html)
- [pandas: `to_numeric`](https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html)
- [pandas: Working with text data](https://pandas.pydata.org/docs/user_guide/text.html)

---

# Lesson 07 — Exploratory Data Analysis (EDA)

**Exploratory Data Analysis (EDA)** means getting to know a dataset **before** you build a model: what it contains, how the values are spread, how the columns relate, and what looks strange. You do it with summary statistics and charts.

EDA answers questions like:

- How big is the data, and what types are the columns?
- What's a typical value, and how spread out are the values?
- Are there outliers or suspicious values?
- Which features are related to the target (`price_lkr`)?
- Which features overlap with each other?

> **Where EDA fits:** load the data (Lesson 5) → clean it (Lesson 6) → **explore it (Lesson 7)** → build a model. EDA often uncovers new cleaning problems, so you'll move back and forth between steps 2 and 3.

## Setup

This lesson adds **matplotlib** for charts (already installed in `.venv`, version 3.9.4):

```bash
source .venv/bin/activate
pip install matplotlib
python lesson07.py
```

## The dataset

[lesson07.py](lesson07.py) uses 6 houses with a new **`age`** column (years since the house was built):

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "area":      [1200, 1800, 1000, 2200, 1500, 2000],
    "bedrooms":  [3, 4, 2, 4, 3, 4],
    "age":       [10, 5, 20, 3, 8, 6],
    "price_lkr": [25_000_000, 40_000_000, 20_000_000,
                  48_000_000, 32_000_000, 45_000_000],
})
```

- **Features** (inputs): `area`, `bedrooms`, `age`
- **Target** (what we want to predict): `price_lkr`

## Step 1: Understand the structure

```python
df.shape          # (6, 4) -> 6 houses, 4 columns
df.dtypes         # all int64 -> every column is numeric
df.isna().sum()   # 0 in every column -> no missing values
df.duplicated().sum()   # 0 -> no duplicates
df.nunique()      # area 6, bedrooms 3, age 6, price_lkr 6
```

`nunique()` counts the distinct values in each column. `bedrooms` has only **3** (2, 3 and 4), so it behaves more like a **category** than a continuous number. That affects which charts suit it (see step 4).

## Step 2: Summary statistics

```python
df.describe()
```

```text
              area  bedrooms        age     price_lkr
count     6.000000  6.000000   6.000000  6.000000e+00
mean   1616.666667  3.333333   8.666667  3.500000e+07
std     466.547604  0.816497   6.055301  1.120714e+07
min    1000.000000  2.000000   3.000000  2.000000e+07
25%    1275.000000  3.000000   5.250000  2.675000e+07
50%    1650.000000  3.500000   7.000000  3.600000e+07
75%    1950.000000  4.000000   9.500000  4.375000e+07
max    2200.000000  4.000000  20.000000  4.800000e+07
```

### How to read it

| Row | Meaning | Example (`price_lkr`) |
|---|---|---|
| `count` | Non-missing values | 6 |
| `mean` | Average | 35,000,000 |
| `std` | **Spread**: how far values typically are from the mean | 11,207,141 |
| `min` / `max` | Smallest / largest | 20,000,000 / 48,000,000 |
| `25%` | 25% of values are below this (**Q1**) | 26,750,000 |
| `50%` | The **median**: half are below, half above | 36,000,000 |
| `75%` | 75% of values are below this (**Q3**) | 43,750,000 |

> `3.500000e+07` means 3.5 × 10⁷ = 35,000,000. To print with commas instead, use `pd.set_option("display.float_format", "{:,.2f}".format)`.

### Mean vs median: is the data skewed?

Comparing the mean and median shows whether a few extreme values are pulling the average:

| Column | Mean | Median | What it tells you |
|---|---|---|---|
| `price_lkr` | 35,000,000 | 36,000,000 | Close together → roughly **symmetric** |
| `age` | 8.67 | 7.0 | Mean > median → **right-skewed**: one old house (20 years) pulls the mean up |

```python
df.skew()
# area        -0.14   <- about symmetric
# bedrooms    -0.86
# age          1.63   <- strong right skew
# price_lkr   -0.23
```

Skewness near `0` means symmetric, positive means a long tail to the right, and negative means a long tail to the left.

### Look at the max vs Q3

A large jump from `75%` to `max` points to an outlier. `age` jumps from **9.5** to **20**, more than double. Step 5 confirms it.

## Step 3: Distributions (one column at a time)

A **distribution** shows how often each range of values appears.

### Histogram

[lesson07.py](lesson07.py) has this commented out:

```python
df["price_lkr"].plot.hist(bins=5, edgecolor="black")
plt.title("Distribution of House Prices")
plt.xlabel("Price (LKR)")
plt.ylabel("Number of Houses (Frequency)")
plt.show()
```

`bins=5` splits the price range (20M to 48M) into 5 equal intervals of 5.6M each and counts the houses in each:

```text
20.0M – 25.6M   ██  2   (C, A)
25.6M – 31.2M       0
31.2M – 36.8M   █   1   (E)
36.8M – 42.4M   █   1   (B)
42.4M – 48.0M   ██  2   (F, D)
```

> With only 6 rows, a histogram looks patchy, and changing `bins` changes the picture a lot. Histograms become useful with hundreds of rows.

### Box plot

A box plot shows the **median, Q1, Q3 and outliers** in a single chart:

```python
df.boxplot(column="age")
plt.title("House Age")
plt.show()
```

```text
          Q1   median   Q3                         outlier
          ┌─────┬───────┐
  ├───────┤     │       ├──────┤                      ●
          └─────┴───────┘
  3      5.25   7      9.5   ~10                     20
```

The box covers the middle 50% of values (Q1 to Q3), the line inside it is the median, and the **dots** beyond the whiskers are outliers.

### Category counts

For a column with only a few distinct values, count each one:

```python
df["bedrooms"].value_counts().sort_index()
# 2    1
# 3    2
# 4    3
```

## Step 4: Relationships (feature vs target)

### Scatter plot

This is the chart that [lesson07.py](lesson07.py) draws:

```python
plt.scatter(df["area"], df["price_lkr"])
plt.title("House Price vs Area")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (LKR)")
plt.tight_layout()
plt.show()
```

Each dot is one house. The points form a nearly **straight line rising to the right**: bigger houses cost more, very consistently.

| Pattern | Meaning |
|---|---|
| Dots rise left → right | **Positive** relationship (area ↑, price ↑) |
| Dots fall left → right | **Negative** relationship (age ↑, price ↓) |
| Dots scattered randomly | **No** relationship |
| Dots tight around a line | **Strong** relationship |

Plot every feature against the target side by side:

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, col in zip(axes, ["area", "bedrooms", "age"]):
    ax.scatter(df[col], df["price_lkr"])
    ax.set_xlabel(col)
axes[0].set_ylabel("Price (LKR)")
plt.tight_layout()
plt.show()
```

### Group comparison (for category-like columns)

`bedrooms` has only 3 values, so a scatter plot shows just 3 vertical stacks of dots. Comparing group averages works better:

```python
df.groupby("bedrooms")["price_lkr"].agg(["count", "mean"])
# (pandas prints means as 2.000000e+07; commas added here for readability)
#           count        mean
# bedrooms
# 2             1  20,000,000
# 3             2  28,500,000
# 4             3  44,333,333

df.groupby("bedrooms")["price_lkr"].mean().plot.bar()
plt.ylabel("Average price (LKR)")
plt.show()
```

You can also turn a continuous column into groups with `pd.cut`:

```python
age_group = pd.cut(df["age"], bins=[0, 5, 10, 25], labels=["0-5", "6-10", "11+"])
df.groupby(age_group, observed=True)["price_lkr"].mean()
# 0-5     44,000,000   <- newest houses are the most expensive
# 6-10    34,000,000
# 11+     20,000,000
```

## Step 5: Outliers

An **outlier** is a value far from the rest. It might be a data error, which you'd fix as in Lesson 6, or a real but unusual case, which you'd keep and note.

### The IQR rule (the box plot method)

```python
s = df["age"]
q1, q3 = s.quantile([0.25, 0.75])   # 5.25, 9.5
iqr = q3 - q1                       # 4.25  -> "interquartile range"
lower = q1 - 1.5 * iqr              # -1.125
upper = q3 + 1.5 * iqr              # 15.875

s[(s < lower) | (s > upper)]        # [20]  -> house C (20 years) is an outlier
```

Running the same check on `area` and `price_lkr` finds **no outliers**.

### Z-score

A z-score measures how many standard deviations a value is from the mean:

```python
z = (df["age"] - df["age"].mean()) / df["age"].std()
# [0.22, -0.61, 1.87, -0.94, -0.11, -0.44]
```

A common rule flags `|z| > 3`. House C's z-score is only **1.87**, so by this rule it **isn't** an outlier.

> **Why do the two methods disagree?** The outlier inflates the mean and standard deviation that the z-score depends on, and with only 6 rows a z-score can barely reach 3. The **IQR rule** uses quartiles, which outliers affect less, so it's more reliable for small or skewed data.

## Step 6: Correlation

**Correlation** (`r`) measures how strongly two columns move together on a **straight line**, as a single number from **−1 to +1**:

```text
 -1.0 ────────── -0.5 ────────── 0 ────────── +0.5 ────────── +1.0
 perfect          moderate       no           moderate        perfect
 negative         negative       linear       positive        positive
                                 relationship
```

| `|r|` | Strength (rule of thumb) |
|---|---|
| 0.0 – 0.3 | Weak |
| 0.3 – 0.7 | Moderate |
| 0.7 – 1.0 | Strong |

### The correlation matrix

[lesson07.py](lesson07.py) prints it at the end:

```python
df.corr(numeric_only=True)
```

```text
               area  bedrooms       age  price_lkr
area       1.000000  0.927543 -0.882569   0.998343
bedrooms   0.927543  1.000000 -0.943880   0.939830
age       -0.882569 -0.943880  1.000000  -0.884139
price_lkr  0.998343  0.939830 -0.884139   1.000000
```

- The **diagonal** is always `1.0`: each column correlates perfectly with itself.
- The matrix is **symmetric**: `area`/`bedrooms` = `bedrooms`/`area`.
- `numeric_only=True` skips text columns such as `house` or `city`, which would otherwise raise an error.

### Rank features against the target

```python
df.corr(numeric_only=True)["price_lkr"].drop("price_lkr").sort_values(ascending=False)
# area        0.998   <- strongest: almost perfectly linear
# bedrooms    0.940
# age        -0.884   <- strong NEGATIVE: older houses are cheaper
```

All three features are strongly related to price, so all three are good candidates for a model.

### Correlation heatmap

matplotlib can draw the matrix as a colour grid without any extra library:

```python
corr = df.corr(numeric_only=True)

plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="correlation")
plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
```

`vmin=-1, vmax=1` keeps `0` in the middle of the colour scale, so red means positive and blue means negative.

For a scatter plot of **every pair** of columns at once:

```python
pd.plotting.scatter_matrix(df, figsize=(8, 8))
plt.show()
```

### Features that overlap (multicollinearity)

Look at correlations **between the features**, not just with the target:

```text
area     ↔ bedrooms   0.93   bigger houses have more bedrooms
bedrooms ↔ age       -0.94   newer houses have more bedrooms
```

When two features are this strongly correlated, they carry **mostly the same information**. This is called **multicollinearity**. It makes it hard to tell which feature is really driving the price, and it can make some models (like linear regression) unstable. You'll learn how to handle it when you start building models.

### Three warnings about correlation

1. **Correlation is not causation.** Older houses here are cheaper, but age may not be the *reason*. Older houses here are also smaller, which could explain the lower price.
2. **It only measures straight-line relationships.** A strong curved pattern can still have `r ≈ 0`, so always look at the scatter plot too.
3. **Small samples exaggerate it.** With 6 houses, `r = 0.998` is far stronger than you'd see in real data. One extra house could change it a lot.

> `df.corr(method="spearman")` measures whether values **rise and fall together**, even along a curve, and outliers affect it less. Here it gives similar results: area 1.00, bedrooms 0.93, age −0.94.

## Step 7: Write down the findings

EDA isn't finished until you write down what you found. For [lesson07.py](lesson07.py):

| Finding | Evidence | What to do next |
|---|---|---|
| Data is clean | 0 missing, 0 duplicates, all numeric | No cleaning needed |
| **Area** is the strongest predictor | r = 0.998, a near-straight line in the scatter plot | Keep as a key feature |
| **Bedrooms** raise the price | r = 0.94; 2 → 3 → 4 bedrooms = 20M → 28.5M → 44.3M | Keep |
| **Age** lowers the price | r = −0.88; houses 0–5 years old average 44M, 11+ years 20M | Keep |
| House C's age (20) is an **outlier** | Above the IQR upper fence of 15.875 | Check whether it's real; keep it if so |
| Features overlap | area ↔ bedrooms 0.93, bedrooms ↔ age −0.94 | Watch for multicollinearity in models |
| Very small sample | Only 6 houses | Collect more data before trusting these numbers |

## Tips for `lesson07.py`

- **`plt.show()` pauses the script.** The correlation matrix on the last line prints only **after you close the chart window**. To see all the text output first, move `print(df.corr(...))` above the plot, or call `plt.show()` once at the very end.
- **Save a chart instead of showing it:** `plt.savefig("price_vs_area.png", dpi=150)`. Call it **before** `plt.show()`, because the figure is cleared once the window closes.
- **Readable money axis:** large numbers appear as `1e7` on the axis. Show them in millions instead:

  ```python
  import matplotlib.ticker as mticker
  plt.gca().yaxis.set_major_formatter(
      mticker.FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M")
  )
  ```

- **Always label charts:** a title, axis labels and units (`Price (LKR)`, `Area (sq ft)`). A chart without labels can't be read.

## Expected output of `lesson07.py`

```text
Dataset shape: (6, 4)

Data types:
area         int64
bedrooms     int64
age          int64
price_lkr    int64
dtype: object

Statistical summary:
              area  bedrooms        age     price_lkr
count     6.000000  6.000000   6.000000  6.000000e+00
mean   1616.666667  3.333333   8.666667  3.500000e+07
std     466.547604  0.816497   6.055301  1.120714e+07
min    1000.000000  2.000000   3.000000  2.000000e+07
25%    1275.000000  3.000000   5.250000  2.675000e+07
50%    1650.000000  3.500000   7.000000  3.600000e+07
75%    1950.000000  4.000000   9.500000  4.375000e+07
max    2200.000000  4.000000  20.000000  4.800000e+07

Median price: 36000000.0
```

Next, a **scatter plot window** opens (price vs area). After you close it:

```text
               area  bedrooms       age  price_lkr
area       1.000000  0.927543 -0.882569   0.998343
bedrooms   0.927543  1.000000 -0.943880   0.939830
age       -0.882569 -0.943880  1.000000  -0.884139
price_lkr  0.998343  0.939830 -0.884139   1.000000
```

## Which chart should I use?

| Question | Chart | Code |
|---|---|---|
| How is one numeric column spread? | Histogram | `df["col"].plot.hist(bins=10)` |
| Median, spread and outliers? | Box plot | `df.boxplot(column="col")` |
| How often does each category appear? | Bar chart | `df["col"].value_counts().plot.bar()` |
| How do two numeric columns relate? | Scatter plot | `plt.scatter(df["x"], df["y"])` |
| How does a numeric value differ by group? | Bar chart of group means | `df.groupby("cat")["y"].mean().plot.bar()` |
| How do all columns relate? | Heatmap / scatter matrix | `plt.imshow(df.corr())`, `pd.plotting.scatter_matrix(df)` |

## EDA checklist

- [ ] **Structure:** `shape`, `dtypes`, `head()`, `nunique()`
- [ ] **Quality:** `isna().sum()`, `duplicated().sum()` (clean as in Lesson 6 if needed)
- [ ] **Summary:** `describe()`; compare mean vs median; check `max` vs `75%`
- [ ] **Distributions:** a histogram or box plot for each numeric column
- [ ] **Outliers:** the IQR rule; decide whether each one is an error or real
- [ ] **Relationships:** a scatter plot of each feature vs the target; group means for categories
- [ ] **Correlation:** rank features against the target; look for features that overlap
- [ ] **Findings:** write down what you learned and what to do next

## EDA quick reference

| Task | Code |
|---|---|
| Summary statistics | `df.describe()` |
| Distinct values per column | `df.nunique()` |
| Skewness | `df.skew()` |
| Quartiles | `df["col"].quantile([0.25, 0.5, 0.75])` |
| Count categories | `df["col"].value_counts()` |
| Group summary | `df.groupby("cat")["y"].agg(["count", "mean"])` |
| Numbers → groups | `pd.cut(df["col"], bins=[...], labels=[...])` |
| Correlation matrix | `df.corr(numeric_only=True)` |
| Correlation with target | `df.corr(numeric_only=True)["y"].sort_values()` |
| Save a chart | `plt.savefig("chart.png", dpi=150)` |

## Practice ideas

1. Uncomment the histogram in [lesson07.py](lesson07.py), try `bins=3` and `bins=10`, and see how the picture changes.
2. Draw scatter plots of `age` vs `price_lkr` and `bedrooms` vs `price_lkr`. Which looks most like a straight line?
3. Use the IQR rule to check every column for outliers in a single loop.
4. Add a `price_per_sqft` column. Is it correlated with `area`? (Hint: the values are all between 20,000 and 22,500.)
5. Add a 7th house with `area=1300, price_lkr=60_000_000` and rerun `corr()`. How much does one unusual house change `r`?
6. Write a findings table like the one in step 7 for the dataset from Lesson 6.

## Resources

- [pandas: `DataFrame.describe`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)
- [pandas: `DataFrame.corr`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html)
- [pandas: Chart visualization](https://pandas.pydata.org/docs/user_guide/visualization.html)
- [matplotlib: Pyplot tutorial](https://matplotlib.org/stable/tutorials/pyplot.html)
