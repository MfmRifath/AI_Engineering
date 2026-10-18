# Lesson 08 — Introduction to Machine Learning

In Lessons 5–7 you loaded, cleaned and explored a dataset. This lesson covers the next step: how a **machine-learning model learns patterns from data** and uses them to make **predictions**.

It follows the progression of Aurélien Géron's *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed., O'Reilly, 2019):

- **Chapter 1, "The Machine Learning Landscape"**: what ML is, the types of ML systems, and the main challenges
- **Chapter 2, "End-to-End Machine Learning Project"**: the workflow from raw data to a working model

## Learning objectives

By the end of this lesson, you should be able to:

- Explain what machine learning is and when to use it.
- Distinguish supervised, unsupervised, and reinforcement learning.
- Understand features, targets, models, and predictions.
- Explain training versus inference.
- Identify overfitting and understand why models need evaluation.
- Build a foundation for your first model using scikit-learn.

---

## 1. What is machine learning?

> **Machine learning** is the science (and art) of programming computers so they can **learn from data**. — Géron, Chapter 1

A more precise definition by Tom Mitchell (1997), which Géron also quotes:

> A computer program is said to learn from **experience E** with respect to some **task T** and some **performance measure P**, if its performance on T, as measured by P, improves with experience E.

Applied to the house dataset from Lesson 7:

| Part | Meaning | House example |
|---|---|---|
| **Task T** | What the program must do | Predict a house's price |
| **Experience E** | The data it learns from | Past houses with known prices (the *training data*) |
| **Performance P** | How we measure success | Average error of the predictions, in LKR |

If the predictions become more accurate as the model sees more houses, the program is **learning**.

### Traditional programming vs machine learning

```text
Traditional programming:

   Rules (written by you)  ──┐
                             ├──►  Program  ──►  Answers
   Data                    ──┘

Machine learning:

   Data                    ──┐
                             ├──►  Learning algorithm  ──►  Rules (a model)
   Answers (known results) ──┘
```

Géron's example is a **spam filter**:

- **Traditional approach:** you study spam, write rules such as *"if the subject contains 'FREE' or '4U', mark it as spam"*, and keep adding rules. Spammers change their wording, the rule list grows long and hard to maintain, and it keeps breaking.
- **ML approach:** you give the algorithm thousands of emails that users have labelled *spam* or *not spam*. It works out which words and patterns predict spam. When spammers change their wording, you **retrain** on new examples instead of rewriting rules.

For house prices, a hand-written rule might be *"price = 15,000 LKR × area"*. But how much is a bedroom worth? How much does each year of age take off? An ML model **estimates those numbers from the data** instead of you guessing them.

---

## 2. When should you use machine learning?

Géron lists four situations where ML works well:

| Use ML when... | Example |
|---|---|
| A problem needs **lots of hand-tuned rules** that keep changing | Spam filtering |
| There's **no known formula** for the problem | Speech recognition, recognising objects in images |
| The environment **keeps changing** and the system must adapt | Fraud detection, product recommendations |
| You want to find **insights in large amounts of data** (*data mining*) | Finding which factors drive customer churn |

### When **not** to use ML

- A **simple, exact rule** already works. Calculating tax or sorting a list doesn't need ML.
- You **don't have enough good data**. A model can only learn what the data shows.
- Every decision must be **fully explainable and guaranteed correct**. ML gives *probably right* answers, not certain ones.

---

## 3. Key vocabulary

Here's the Lesson 7 dataset again:

```text
         area  bedrooms  age     price_lkr
house A  1200         3   10    25,000,000   ◄── one instance (row)
house B  1800         4    5    40,000,000
house C  1000         2   20    20,000,000
  ...
         └───────── features ─────────┘  └ target ┘
                       X                    y
```

| Term | Also called | Meaning | Example |
|---|---|---|---|
| **Instance** | sample, example, row, observation | One item in the dataset | House A |
| **Feature** | attribute, input, predictor, column | A measurable property used to make the prediction | `area`, `bedrooms`, `age` |
| **Target** | label, output, `y` | The value you want to predict | `price_lkr` |
| **Model** | | The learned relationship between features and target | `price ≈ a × area + b × bedrooms + c × age + d` |
| **Parameters** | weights, coefficients | The numbers the model learns from data | `a`, `b`, `c`, `d` |
| **Prediction** | `ŷ` ("y-hat") | The model's output for an instance | 31,200,000 LKR |
| **Training set** | | Data the model learns from | 80% of the houses |
| **Test set** | | Data held back to check the model | the other 20% |

By convention:

- **`X`** (capital) is the **feature matrix**, a 2D table with one row per instance and one column per feature.
- **`y`** (lowercase) is the **target vector**, a 1D array with one value per instance.

```python
X = df[["area", "bedrooms", "age"]]   # 2D: shape (n_houses, 3)
y = df["price_lkr"]                   # 1D: shape (n_houses,)
```

> **Feature vs target is a choice you make, not a property of the data.** The same table could be used to predict `age` from `area` and `price_lkr`. What you choose as the target defines the task.

---

## 4. Types of machine learning systems

Géron sorts ML systems along three questions. The most important one is **how much supervision (labelled answers) the system gets during training**.

### 4.1 Supervised learning

The training data **includes the answers** (labels). The model learns to map features to the target.

```text
 features (X)                     target (y)
 area=1200, beds=3, age=10   ──►  25,000,000      the model learns from
 area=1800, beds=4, age=5    ──►  40,000,000      these pairs...
 area=1000, beds=2, age=20   ──►  20,000,000

 area=1600, beds=3, age=5    ──►  ?               ...then predicts this one
```

There are two main kinds of supervised task:

| Task | Target type | Question | Examples |
|---|---|---|---|
| **Regression** | A **number** | "How much?" / "How many?" | House price, temperature, sales next month |
| **Classification** | A **category** | "Which class?" | Spam / not spam, digit 0–9, disease yes/no |

Common algorithms: linear regression, logistic regression (a *classification* algorithm, despite the name), k-nearest neighbours, decision trees, random forests, support vector machines, neural networks.

> Predicting `price_lkr` is **regression**. Predicting "expensive / affordable" from the same data would be **classification**.

### 4.2 Unsupervised learning

The training data has **no labels**. The algorithm finds structure on its own.

| Task | What it does | Example |
|---|---|---|
| **Clustering** | Groups similar instances together | Segmenting customers into groups with similar buying habits |
| **Dimensionality reduction** | Squeezes many features into fewer while keeping most of the information | Compressing 100 correlated features into 10 |
| **Anomaly detection** | Learns what "normal" looks like and flags anything unusual | Spotting fraudulent card transactions |
| **Association rule learning** | Finds items that occur together | "People who buy bread also buy butter" |

In scikit-learn, the difference shows up in the code: supervised models are trained with `fit(X, y)`, unsupervised ones with **`fit(X)` and no `y`**:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)                    # no y: nobody tells it the "right" groups
groups = kmeans.predict(X)       # e.g. [0, 2, 1, 0, ...] -> a group number per house
```

### 4.3 Semi-supervised learning

Some of the data is labelled, but most isn't. Labelling is often slow and expensive, so this is common in practice.

> Géron's example is Google Photos. It **clusters** your photos to work out that the same person appears in photos 1, 5 and 11 (unsupervised). You then **name** that person once (a label), and it labels every photo of them.

### 4.4 Reinforcement learning

This one works very differently. An **agent** observes an **environment**, takes **actions**, and receives **rewards** (or penalties). Over time it learns a **policy**, which is a strategy for choosing the action that earns the most reward.

```text
        ┌──────────── observation, reward ◄────────────┐
        ▼                                              │
     ┌───────┐          action                  ┌─────────────┐
     │ Agent │ ───────────────────────────────► │ Environment │
     └───────┘                                  └─────────────┘
```

Examples: robots learning to walk, DeepMind's **AlphaGo** (which beat the Go world champion in 2017), and game-playing agents.

There's no dataset of correct answers. The agent learns from the consequences of its own actions.

### Summary

| Type | Labels in training data? | Goal | Example |
|---|---|---|---|
| **Supervised** | Yes, all | Predict the target for new instances | House price, spam filter |
| **Unsupervised** | No | Find structure or groups | Customer segments |
| **Semi-supervised** | A few | Use a few labels to label the rest | Photo tagging |
| **Reinforcement** | No (rewards instead) | Learn the best actions | Game AI, robotics |

### Two other ways to classify ML systems

Géron also asks two more questions:

**Can it learn incrementally?**

| Batch (offline) learning | Online learning |
|---|---|
| Trained on **all the data at once**, then deployed | Learns **continuously** from small batches of new data |
| To update it, retrain from scratch on old + new data | Updates on the fly; adapts to change quickly |
| Simple; fine when data changes slowly | Suits stock prices, huge datasets, or limited compute |

**How does it generalise to new instances?**

| Instance-based learning | Model-based learning |
|---|---|
| Memorises the training examples | Builds a **model** (a formula) from the examples |
| Predicts by comparing a new instance with **similar stored ones** | Predicts by **plugging the new instance into the formula** |
| *"This 1,600 sq ft house is similar to these 3 houses, so average their prices"* (k-nearest neighbours) | *"price ≈ 15,000 × area + ..."* (linear regression) |

---

## 5. Training vs inference

Every ML model has two phases:

```text
 ┌──────────────────── TRAINING ────────────────────┐   ┌──────────── INFERENCE ────────────┐
 │                                                  │   │                                   │
 │  training data (X_train, y_train)                │   │  new data (X_new, no answer)      │
 │            │                                     │   │            │                      │
 │            ▼                                     │   │            ▼                      │
 │   learning algorithm adjusts the parameters      │   │   trained model                   │
 │   to make the predictions match y_train          │──►│            │                      │
 │            │                                     │   │            ▼                      │
 │            ▼                                     │   │   prediction ŷ                    │
 │      trained model                               │   │                                   │
 └──────────────────────────────────────────────────┘   └───────────────────────────────────┘
       model.fit(X_train, y_train)                            model.predict(X_new)
```

| | Training | Inference |
|---|---|---|
| **What happens** | The model **learns** its parameters | The model **uses** its parameters |
| **Needs the target?** | Yes (`y_train`) | No, predicting the target is the point |
| **scikit-learn** | `model.fit(X, y)` | `model.predict(X_new)` |
| **How often** | Once, or whenever you retrain | Every time you need a prediction |
| **Cost** | Can be slow (minutes to weeks for big models) | Usually fast (milliseconds) |
| **Parameters** | Change | **Fixed** |

> **Parameters vs hyperparameters:** **parameters** are learned during training (for example the coefficients of a linear model). **Hyperparameters** are settings **you** choose *before* training (for example a decision tree's `max_depth`). The model doesn't learn them.

---

## 6. Main challenges of machine learning

Géron sums it up: the two things that can go wrong are **"bad data"** and **"bad algorithm"**.

### Bad data

| Problem | What it means | Fix |
|---|---|---|
| **Not enough data** | Most algorithms need thousands of examples; complex ones need millions | Collect more data |
| **Non-representative data** | The training data doesn't match the cases you'll predict on (**sampling bias**) | Sample carefully. A model trained only on Colombo houses will misprice rural houses |
| **Poor-quality data** | Errors, outliers, missing values, noise | Clean it (Lesson 6) |
| **Irrelevant features** | Garbage in, garbage out | **Feature engineering**: pick useful features, combine them, create new ones |

### Bad algorithm: overfitting and underfitting

**Overfitting** means the model learns the training data **too well**, including its noise and random quirks, so it **fails on new data**.

> Géron's analogy: you're abroad and a taxi driver rips you off. You conclude that *all* taxi drivers in that country are thieves. That's over-generalising from too little data, and machines do it too.

**Underfitting** is the opposite: the model is **too simple** to capture the real pattern.

```text
 price                     price                     price
   │        •   •            │        •  ╱•            │        •╲  •
   │     •                   │     • ╱                 │     •╱  ╲╱
   │ ─────────────────       │    ╱•                   │    ╱•
   │  •    •                 │  •╱   •                 │  •╱   •
   │ •                       │ ╱•                      │ ╱•
   └──────────── area        └──────────── area        └──────────── area
    UNDERFITTING              GOOD FIT                  OVERFITTING
    too simple: a flat        captures the trend,       zig-zags through every
    line misses the trend     ignores the noise         point, including noise
```

| | Underfitting | Good fit | Overfitting |
|---|---|---|---|
| **Training error** | High | Low | **Very low (often ~0)** |
| **Test error** | High | Low | **High** |
| **Cause** | Model too simple | | Model too complex for the amount of data |
| **Fix** | More powerful model, better features, less regularisation | | More data, simpler model, **regularisation** (constraints such as `max_depth`), less noise |

> **The warning sign of overfitting** is a **big gap between training error and test error**. A model that scores perfectly on its training data should make you suspicious, not happy.

---

## 7. Why models need evaluation: testing and validating

You can't judge a model on the data it trained on. That's like grading students using the exact questions they memorised. You need to see how it handles **new, unseen data**. This is called **generalisation**.

### Train/test split

Hold back part of the data (typically **20%**) and **never touch it during training**:

```text
 all 100 houses
 ┌────────────────────────────────────────────────┬────────────┐
 │              training set (80%)                │ test (20%) │
 │          the model learns from this            │ final exam │
 └────────────────────────────────────────────────┴────────────┘
```

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- `test_size=0.2`: 20% goes to the test set.
- `random_state=42`: fixes the random shuffle so you get **the same split every run** (42 is just a convention).

The error on the test set estimates the **generalisation error**, which is how wrong the model will be on new cases in real life.

### Validation set

If you try many models or hyperparameters and keep the one that scores best **on the test set**, you've indirectly tuned the model to the test set, and its score becomes over-optimistic. The fix is to split three ways:

| Set | Used for |
|---|---|
| **Training set** | Fitting the model's parameters |
| **Validation set** | Comparing models and choosing hyperparameters |
| **Test set** | **One** final check at the very end |

**Cross-validation** is a more data-efficient version: split the training data into *k* parts, train on *k − 1* of them, validate on the remaining one, and rotate. You'll use it in later lessons.

### Always compare against a baseline

A **baseline** is the simplest possible "model", for example *always predict the average price*. If your ML model can't beat that, it hasn't learned anything useful.

### Regression metrics

| Metric | Formula (in words) | Read it as | Better |
|---|---|---|---|
| **MAE**, mean absolute error | Average of \|actual − predicted\| | "On average, predictions are off by X LKR" | Lower |
| **RMSE**, root mean squared error | √(average of (actual − predicted)²) | Like MAE, but **punishes big errors more** | Lower |
| **R²**, coefficient of determination | 1 − (model's error / baseline's error) | Fraction of the variation in price the model explains | Higher (max 1.0) |

```text
 R² = 1.0   perfect predictions
 R² = 0.0   no better than always predicting the mean (the baseline)
 R² < 0     worse than the baseline!
```

MAE and RMSE are in the **same units as the target** (LKR), which makes them easy to explain to anyone.

---

## 8. The end-to-end machine learning workflow

Chapter 2 of Géron walks through a complete project. The main steps:

```text
 1. Frame the problem       What's the goal? Regression or classification? How will success be measured?
         │
 2. Get the data            Load it (Lesson 5)
         │
 3. Explore the data        EDA: distributions, correlations, outliers (Lesson 7)
         │
 4. Prepare the data        Clean, handle missing values, encode categories, scale (Lesson 6)
         │
 5. Split train / test      ◄── do this EARLY, before looking too closely at the data
         │
 6. Select & train models   Try a few: start simple
         │
 7. Evaluate & fine-tune    Validation set / cross-validation, tune hyperparameters
         │
 8. Final test              Score once on the test set
         │
 9. Deploy, monitor, retrain
```

You've already done steps 2–4. This lesson adds steps **5–8**.

> **Data snooping bias:** Géron recommends creating the test set **before** you explore the data in depth. If you study the whole dataset, including the test part, you may unconsciously pick a model that suits the test data, and your test score will be over-optimistic.

---

## 9. Your first model with scikit-learn

### Setup

This folder has its own virtual environment (`.venv`, Python 3.9) with **scikit-learn 1.6.1**, NumPy, pandas and matplotlib already installed:

```bash
cd ML
source .venv/bin/activate

# only needed if you recreate the venv
pip install numpy pandas matplotlib scikit-learn
```

### The scikit-learn pattern

Every scikit-learn model (an **estimator**) follows the same steps, so once you learn them you can use any algorithm:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()        # 1. choose a model (and set its hyperparameters)
model.fit(X_train, y_train)       # 2. TRAIN: learn parameters from the data
y_pred = model.predict(X_test)    # 3. INFERENCE: predict for new data
```

To try a different algorithm, change **only line 1**, for example `DecisionTreeRegressor()`. Steps 2 and 3 stay exactly the same.

### Step 1: A bigger dataset

With only 6 houses (Lesson 7) you can't hold any back for testing. So we generate **100 houses** from a **hidden rule** plus random noise, as if real-world prices were set this way:

```text
price = 5,000,000
      + 15,000    × area        (each sq ft adds 15,000 LKR)
      + 1,500,000 × bedrooms    (each bedroom adds 1.5M)
      − 250,000   × age         (each year of age takes off 250k)
      + noise                   (random ±, around 2M: things the features don't capture)
```

The model **never sees this rule**, only the 100 rows it produced. The test is whether it can **work the rule out from the data**.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 100

area = rng.integers(800, 2600, n)
bedrooms = np.clip(np.round(area / 500 + rng.normal(0, 0.5, n)), 1, 5).astype(int)
age = rng.integers(0, 30, n)
noise = rng.normal(0, 2_000_000, n)
price_lkr = 5_000_000 + 15_000 * area + 1_500_000 * bedrooms - 250_000 * age + noise

df = pd.DataFrame({
    "area": area,
    "bedrooms": bedrooms,
    "age": age,
    "price_lkr": price_lkr.round(-3).astype(int),
})

print(df.shape)   # (100, 4)
print(df.head())
```

- `rng.integers(800, 2600, n)`: 100 random areas from 800 up to (but not including) 2,600 sq ft
- `bedrooms` grows with `area` (roughly one bedroom per 500 sq ft), clipped to 1–5, which mimics the area ↔ bedrooms correlation you found in Lesson 7
- `.round(-3)` rounds prices to the nearest 1,000 LKR

### Step 2: Features and target

```python
X = df[["area", "bedrooms", "age"]]
y = df["price_lkr"]
```

### Step 3: Train/test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape, X_test.shape)   # (80, 3) (20, 3)
```

### Step 4: Training

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

for name, coef in zip(X.columns, model.coef_):
    print(f"{name:<9} {coef:>12,.0f}")
print(f"intercept {model.intercept_:>12,.0f}")
```

**Linear regression** finds the line (or, with several features, the flat plane) that best fits the training data:

```text
price ≈ coef_area × area + coef_bedrooms × bedrooms + coef_age × age + intercept
```

- `model.coef_`: one learned weight per feature
- `model.intercept_`: the base value when every feature is 0

Learned attributes in scikit-learn end with an **underscore** (`coef_`, `intercept_`). They exist only **after** `fit()`.

**What to look for:** the learned values should come out **close to the hidden rule**: about **15,000** for `area`, about **1,500,000** for `bedrooms`, about **−250,000** for `age`, and about **5,000,000** for the intercept. They won't match exactly, because the noise and the small sample blur the picture. This is learning from data: the model recovered the pattern without being told it.

> **Interpreting a coefficient:** *"holding the other features fixed, one extra sq ft adds about 15,000 LKR."* Because `area` and `bedrooms` are correlated (multicollinearity, Lesson 7), the split of credit between those two coefficients can wobble from sample to sample, even when the predictions stay good.

### Step 5: Inference

```python
new_house = pd.DataFrame({"area": [1600], "bedrooms": [3], "age": [5]})
price = model.predict(new_house)[0]
print(f"Predicted price: {price:,.0f} LKR")
```

Using the hidden rule, the "true" expected price is 5M + 24M + 4.5M − 1.25M = **32.25M LKR**, so the prediction should land near that.

- `predict` always expects **2D input** (a table), even for one house, which is why `new_house` is a DataFrame with one row.
- Use the **same column names and order** as in training.
- `predict` returns an **array**; `[0]` takes its first (and only) value.

### Step 6: Evaluate and detect overfitting

Compare three models: a **baseline**, **linear regression**, and an unconstrained **decision tree**.

```python
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)   # always predicts the average
tree = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)

for name, m in [("Baseline", baseline), ("Linear", model), ("Tree", tree)]:
    train_mae = mean_absolute_error(y_train, m.predict(X_train))
    test_mae = mean_absolute_error(y_test, m.predict(X_test))
    test_r2 = r2_score(y_test, m.predict(X_test))
    print(f"{name:<9} train MAE {train_mae:>12,.0f}   test MAE {test_mae:>12,.0f}   test R² {test_r2:.3f}")
```

Run it and look at the **pattern**, not the exact numbers:

| Model | Train MAE | Test MAE | What it tells you |
|---|---|---|---|
| **Baseline** (mean) | High | High | The bar to beat; test R² ≈ 0 |
| **Linear regression** | Low (≈ noise level) | **Similar to train** | **Good fit**: it learned the real pattern, so it generalises |
| **Decision tree** (no limits) | **≈ 0** | Clearly higher than linear | **Overfitting**: it memorised every training house, noise included |

Why the tree overfits: with no `max_depth`, it keeps splitting until **every training house sits in its own leaf**, so it "predicts" the training prices perfectly. Those splits fit the noise, which doesn't carry over to new houses.

**Try this:** limit the tree with `DecisionTreeRegressor(max_depth=3, random_state=42)` (a hyperparameter acting as **regularisation**). The training error goes **up**, and the gap between train and test error **shrinks**.

> **The linear model's errors won't reach 0, and they shouldn't.** The noise (about ±2M) is random, so no model can predict it from the features. That's the **irreducible error**. A test MAE of roughly 1.5M LKR is about as good as this data allows.

### Step 7: Visualise predicted vs actual

```python
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

plt.scatter(y_test / 1e6, y_pred / 1e6)
lo, hi = y.min() / 1e6, y.max() / 1e6
plt.plot([lo, hi], [lo, hi], "r--", label="perfect prediction")
plt.title("Predicted vs Actual (test set)")
plt.xlabel("Actual price (M LKR)")
plt.ylabel("Predicted price (M LKR)")
plt.legend()
plt.tight_layout()
plt.show()
```

Each dot is a test house. Dots **on the red line** were predicted perfectly; the vertical distance from the line is the error. For a good model, the dots hug the line with no pattern in how they miss it.

---

## 10. Exercises

1. **Mitchell's definition.** Write down the **T**, **E** and **P** for: (a) a spam filter, (b) a model that predicts tomorrow's rainfall, (c) a chess-playing program.
2. **Pick the type.** Is each one supervised (regression or classification), unsupervised, or reinforcement?
   - Predicting a student's exam score from hours studied
   - Grouping news articles by topic, with no topics given
   - Detecting whether an X-ray shows pneumonia
   - A robot vacuum learning the fastest route around a room
   - Flagging unusual logins on a server
3. **More data, less overfitting.** Change `n = 100` to `n = 30` and then `n = 1000`. How do the learned coefficients and the decision tree's train/test gap change?
4. **More noise.** Change the noise standard deviation from `2_000_000` to `8_000_000`. What happens to the test R² of the linear model? Why can't any model fix this?
5. **Tune the tree.** Try `max_depth` = 1, 2, 3, 5 and 10. Make a table of train MAE and test MAE. Which depth underfits? Which overfits? Which is best?
6. **Instance-based learning.** Replace `LinearRegression()` with `KNeighborsRegressor(n_neighbors=5)` from `sklearn.neighbors`. How does it compare? (Hint: it compares houses by distance, and `area` is in the thousands while `bedrooms` is 1–5. Which feature will dominate? Feature **scaling** is coming in a later lesson.)
7. **Data snooping.** Explain in your own words why you shouldn't choose your final model based on its test-set score.

---

## 11. Key takeaways

- **ML = learning rules from data** instead of writing them by hand. Use it when the rules are too complex, unknown, or keep changing.
- **Supervised** learning has labels (regression → numbers, classification → categories). **Unsupervised** finds structure without labels. **Reinforcement** learns from rewards.
- **Features (`X`)** go in, the **target (`y`)** comes out; the **model's parameters** connect them.
- **Training** (`fit`) learns the parameters; **inference** (`predict`) uses them on new data.
- A model is only as good as its **data**: quantity, representativeness, quality and relevant features.
- **Overfitting** means great training scores and poor test scores. Watch the **gap**.
- **Always evaluate on data the model hasn't seen**, and always compare against a **baseline**.

## Further reading

- Géron, *Hands-On Machine Learning* (2nd ed.):
  - **Chapter 1**: The Machine Learning Landscape (everything in sections 1–7 above)
  - **Chapter 2**: End-to-End Machine Learning Project (section 8, and the next lessons)
- [scikit-learn: Getting Started](https://scikit-learn.org/stable/getting_started.html)
- [scikit-learn: Choosing the right estimator](https://scikit-learn.org/stable/machine_learning_map.html)

**Next lesson:** an end-to-end project on a real dataset (Géron Chapter 2), covering stratified sampling, feature scaling, pipelines, and cross-validation.
