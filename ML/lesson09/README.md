# Lesson 09 — Linear Regression and Model Evaluation

In Lesson 8 you trained your first scikit-learn model ([lesson_08.py](../lesson08/lesson_08.py)): a `LinearRegression` that predicts `price_lkr` (in millions) from `area` and `bedrooms`, and printed its MAE, MSE and R².

This lesson opens the black box:

- **How** does linear regression turn features into a prediction?
- **What** do the learned numbers mean?
- **How** do you measure the errors, and how do you tell whether the model is actually **useful**?

It follows Géron's *Hands-On Machine Learning* (2nd ed.):

- **Chapter 2**: performance measures (RMSE, MAE) and comparing models
- **Chapter 4, "Training Models"**: the linear regression equations, the normal equation, gradient descent, polynomial regression, and learning curves

## Learning objectives

By the end of this lesson, you'll be able to:

- Explain the mathematics behind linear regression.
- Interpret model weights and the bias term.
- Calculate and compare MAE and mean squared error.
- Understand $R^2$ and its limitations.
- Recognize underfitting and overfitting.
- Compare a model against a simple baseline.

---

## The worked example

To do the maths **by hand**, this lesson uses 5 houses with a single feature. Every number below is calculated from these values, so you can check each step with a calculator.

| House | Area (sq ft) $x$ | Price (M LKR) $y$ |
|---|---|---|
| A | 1000 | 20 |
| B | 1500 | 28 |
| C | 2000 | 37 |
| D | 2500 | 44 |
| E | 3000 | 51 |

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "area":      [1000, 1500, 2000, 2500, 3000],
    "price_lkr": [20, 28, 37, 44, 51],          # millions of LKR
})
X = df[["area"]]       # 2D, even with one feature
y = df["price_lkr"]
```

---

## 1. How linear regression makes a prediction

### One feature: a straight line

$$\hat{y} = w \cdot x + b$$

| Symbol | Name | Meaning |
|---|---|---|
| $\hat{y}$ ("y-hat") | prediction | The model's estimated price |
| $x$ | feature | The house's area |
| $w$ | **weight** (slope, coefficient) | How much the prediction changes when $x$ goes up by 1 |
| $b$ | **bias** (intercept) | The prediction when $x = 0$; it shifts the whole line up or down |

`w` and `b` are the model's **parameters**. Training means finding the best values for them.

```text
 price (M)
   55 │                                  ● E
   50 │                               ╱
   45 │                        ● D ╱
   40 │                         ╱
   35 │                ● C  ╱          ŷ = 0.0156 · area + 4.8
   30 │              ╱
   25 │      ● B ╱
   20 │  ● A ╱
      └───┬──────┬──────┬──────┬──────┬───── area (sq ft)
        1000   1500   2000   2500   3000
```

### Many features: a weighted sum

Your Lesson 8 model has two features, so it learns one weight **per feature** plus one bias:

$$\hat{y} = w_1 x_1 + w_2 x_2 + b \quad\Longrightarrow\quad \widehat{\text{price}} = w_{\text{area}} \cdot \text{area} + w_{\text{bedrooms}} \cdot \text{bedrooms} + b$$

In general, with $n$ features:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

Géron writes this in vector form, with $\theta$ ("theta") for all the parameters and $\theta_0$ for the bias:

$$\hat{y} = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n = \boldsymbol{\theta}^\top \mathbf{x}$$

It's the same equation in different notation. With two features the model is a flat **plane** instead of a line; with more, it's a **hyperplane**. It's still "linear" because each feature is only multiplied by a weight and added.

In scikit-learn:

| Maths | scikit-learn | Your Lesson 8 model |
|---|---|---|
| $w_1, w_2, \dots$ | `model.coef_` | `[w_area, w_bedrooms]` |
| $b$ | `model.intercept_` | one number |
| $\hat{y}$ | `model.predict(X)` | predicted prices |

---

## 2. How linear regression learns the weights

### Step 1: Define "error" for one house

The **residual** is the gap between the true value and the prediction:

$$e_i = y_i - \hat{y}_i$$

A positive residual means the model **under**-predicted; a negative one means it **over**-predicted.

### Step 2: Combine the errors into a cost function

Linear regression picks the $w$ and $b$ that make the **mean squared error** on the training set as small as possible:

$$\text{MSE}(w, b) = \frac{1}{m}\sum_{i=1}^{m}\left(y_i - \hat{y}_i\right)^2$$

where $m$ is the number of training instances. This is called **ordinary least squares (OLS)**: "least squares" because it minimises the squared errors.

Why squares and not just the errors?

- Positive and negative errors would **cancel out** if you simply added them.
- Squaring makes **big errors cost much more** than small ones.
- The squared version is smooth, so the best $w$ and $b$ can be found **exactly** with calculus.

### Step 3: Solve for the best weights

For **one feature**, the solution has a simple form:

$$w = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} \qquad b = \bar{y} - w\,\bar{x}$$

where $\bar{x}$ and $\bar{y}$ are the means.

**Worked example.** The means are $\bar{x} = 2000$ and $\bar{y} = 180 / 5 = 36$.

| House | $x - \bar{x}$ | $y - \bar{y}$ | $(x-\bar{x})(y-\bar{y})$ | $(x-\bar{x})^2$ |
|---|---|---|---|---|
| A | −1000 | −16 | 16,000 | 1,000,000 |
| B | −500 | −8 | 4,000 | 250,000 |
| C | 0 | 1 | 0 | 0 |
| D | 500 | 8 | 4,000 | 250,000 |
| E | 1000 | 15 | 15,000 | 1,000,000 |
| **Sum** | | | **39,000** | **2,500,000** |

$$w = \frac{39{,}000}{2{,}500{,}000} = 0.0156 \qquad b = 36 - 0.0156 \times 2000 = 36 - 31.2 = 4.8$$

So the learned model is:

$$\widehat{\text{price}} = 0.0156 \times \text{area} + 4.8$$

scikit-learn finds exactly the same numbers:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X, y)
print(model.coef_)        # [0.0156]
print(model.intercept_)   # 4.8 (it may print as 4.799999... because of floating-point rounding)
```

### The normal equation (any number of features)

For several features, the same idea is written with matrices. Géron calls it the **normal equation**:

$$\hat{\boldsymbol{\theta}} = \left(\mathbf{X}^\top \mathbf{X}\right)^{-1} \mathbf{X}^\top \mathbf{y}$$

You don't need to do this by hand, but you can check it with NumPy. Add a column of 1s so the bias is learned as $\theta_0$:

```python
X_b = np.c_[np.ones(len(X)), X]                       # [[1, 1000], [1, 1500], ...]
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(theta)                                          # ≈ [4.8, 0.0156]  -> [bias, weight]
```

> scikit-learn's `LinearRegression` uses a more numerically stable method (based on the SVD), but the answer is the same.

### Gradient descent (the other way to learn)

The normal equation gets slow with very many features. **Gradient descent** finds the minimum **step by step** instead:

```text
 MSE
  │╲                          ╱
  │ ╲                       ╱
  │  ●╲  start: random w   ╱
  │    ╲●                 ╱
  │      ╲ ●            ╱       each step moves w "downhill",
  │         ╲ ● ●  ● ╱          in the direction that lowers the MSE
  │              ★              ★ = minimum: the best w
  └──────────────────────────────── w
```

1. Start with random weights.
2. Measure the slope (**gradient**) of the MSE with respect to each weight.
3. Move each weight a small step **downhill**. The step size is the **learning rate**, a hyperparameter.
4. Repeat until the MSE stops going down.

For linear regression, both methods reach the same answer. Gradient descent matters because it's **how neural networks are trained**, so you'll see it again.

> scikit-learn's `SGDRegressor` trains a linear model with (stochastic) gradient descent. It needs **feature scaling** to work well, because otherwise the "valley" is stretched and the steps zig-zag.

---

## 3. Interpreting the weights and the bias

From the worked example, $\widehat{\text{price}} = 0.0156 \times \text{area} + 4.8$:

| Parameter | Value | Units | Plain English |
|---|---|---|---|
| $w$ (weight) | 0.0156 | M LKR per sq ft | Each extra sq ft adds **0.0156 M = 15,600 LKR** to the predicted price |
| $b$ (bias) | 4.8 | M LKR | The predicted price of a house with **0 sq ft**, which doesn't exist |

**Always state the units.** In your Lesson 8 data, prices are in **millions**, so every weight is in *millions of LKR per unit of the feature*.

### Rules for reading weights

**1. Sign = direction.** A positive weight means the prediction goes up as the feature goes up. A negative weight (for example, `age` in Lesson 8's notes) means it goes down.

**2. With several features: "holding the others fixed".** For your Lesson 8 model:

> $w_{\text{area}}$ is the change in predicted price for **+1 sq ft, when bedrooms stay the same**.
> $w_{\text{bedrooms}}$ is the change for **+1 bedroom, when area stays the same**.

The second one is subtle. One more bedroom **in the same floor area** means smaller rooms, which isn't the same as "a bigger house".

**3. The bias is often not meaningful on its own.** It's where the line crosses $x = 0$. A 0 sq ft house doesn't exist, so $b = 4.8$ isn't "the price of land". It just positions the line correctly within the range of your data.

**4. A bigger weight doesn't mean a more important feature.** Weights depend on the feature's **units**:

| Area measured in | Weight |
|---|---|
| sq ft | 0.0156 |
| thousands of sq ft | 15.6 |
| sq metres (1 m² ≈ 10.76 sq ft) | ≈ 0.168 |

It's the same model and the same predictions, just different numbers. To compare importance across features, put them on the same scale first (**standardisation**, in a later lesson).

**5. Correlated features make weights unstable.** In your Lesson 8 data, `area` and `bedrooms` rise together (multicollinearity, Lesson 7). The model can shift credit between them almost freely, so the individual weights can change a lot with a slightly different training set. A weight can even come out with a surprising sign. The **predictions** are usually still fine; it's the **interpretation** of each weight that becomes unreliable.

**6. Only trust predictions inside the data's range.** The model was fitted on 1,000–3,000 sq ft. Predicting for 10,000 sq ft (**extrapolation**) assumes the straight line keeps going, and nothing in the data supports that.

---

## 4. Measuring errors: MAE, MSE and RMSE

### Predictions and residuals for the worked example

Using $\hat{y} = 0.0156 \times \text{area} + 4.8$:

| House | $y$ | $\hat{y}$ | Residual $y - \hat{y}$ | $\lvert y - \hat{y} \rvert$ | $(y - \hat{y})^2$ |
|---|---|---|---|---|---|
| A | 20 | 20.4 | −0.4 | 0.4 | 0.16 |
| B | 28 | 28.2 | −0.2 | 0.2 | 0.04 |
| C | 37 | 36.0 | 1.0 | 1.0 | 1.00 |
| D | 44 | 43.8 | 0.2 | 0.2 | 0.04 |
| E | 51 | 51.6 | −0.6 | 0.6 | 0.36 |
| **Sum** | | | **0.0** | **2.4** | **1.60** |

> The residuals sum to **0**. That's always true for least-squares linear regression with a bias term on its training data: the line sits exactly "in the middle" of the points. It's also why you can't measure error by simply averaging the residuals.

### Mean absolute error (MAE)

$$\text{MAE} = \frac{1}{m}\sum_{i=1}^{m} \lvert y_i - \hat{y}_i \rvert = \frac{2.4}{5} = 0.48 \text{ M LKR}$$

**"On average, predictions are off by 0.48 M (480,000 LKR)."**

### Mean squared error (MSE)

$$\text{MSE} = \frac{1}{m}\sum_{i=1}^{m} (y_i - \hat{y}_i)^2 = \frac{1.6}{5} = 0.32 \text{ (M LKR)}^2$$

The unit is **millions of LKR squared**, which has no everyday meaning. MSE is useful for **training** (it's the cost function) and for **comparing** models, but it's hard to explain to anyone.

### Root mean squared error (RMSE)

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{0.32} \approx 0.566 \text{ M LKR}$$

Taking the square root brings the units **back to M LKR**. Géron uses RMSE as the main metric for regression.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

y_pred = model.predict(X)
mean_absolute_error(y, y_pred)        # 0.48
mean_squared_error(y, y_pred)         # 0.32
root_mean_squared_error(y, y_pred)    # 0.566 (scikit-learn ≥ 1.4)
np.sqrt(mean_squared_error(y, y_pred))  # same thing, works in any version
```

### MAE vs MSE: how they treat big errors

Two models each predict 4 houses. Their absolute errors (in M LKR):

| | House 1 | House 2 | House 3 | House 4 | MAE | MSE | RMSE |
|---|---|---|---|---|---|---|---|
| **Model A** | 2 | 2 | 2 | 2 | **2** | (4+4+4+4)/4 = **4** | **2** |
| **Model B** | 0 | 0 | 0 | 8 | **2** | (0+0+0+64)/4 = **16** | **4** |

Both have the **same MAE**, but Model B's MSE is **4× higher** because of one big miss. Squaring makes an 8M error count as much as sixteen 2M errors.

| | MAE | MSE / RMSE |
|---|---|---|
| **Units** | Same as the target | MSE: squared · RMSE: same as the target |
| **Big errors** | Count in proportion to their size | **Count much more** (squared) |
| **Outliers in the data** | More robust | Strongly affected |
| **Use when** | All errors cost about the same; data has outliers | Large errors are especially bad (badly mispricing one house is worse than being slightly off on many) |

> **Always RMSE ≥ MAE.** The bigger the gap between them, the more your errors are dominated by a few large misses.

---

## 5. $R^2$ (the coefficient of determination)

### The idea

$R^2$ compares your model's errors with the errors of the **simplest possible model**, one that always predicts the **mean** of $y$:

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

| Term | Name | Meaning |
|---|---|---|
| $\text{SS}_{\text{res}}$ | residual sum of squares | Squared error of **your model** |
| $\text{SS}_{\text{tot}}$ | total sum of squares | Squared error of **always predicting the mean** |

**Worked example.** From the tables above, $\text{SS}_{\text{res}} = 1.6$. The deviations from the mean are −16, −8, 1, 8, 15:

$$\text{SS}_{\text{tot}} = 256 + 64 + 1 + 64 + 225 = 610$$

$$R^2 = 1 - \frac{1.6}{610} \approx 1 - 0.0026 = 0.997$$

**"The model explains about 99.7% of the variation in price"**, compared with just guessing the average.

```python
from sklearn.metrics import r2_score
r2_score(y, y_pred)    # ≈ 0.997
model.score(X, y)      # the same: .score() returns R² for regressors
```

### Reading $R^2$

```text
  R² < 0          R² = 0             0 < R² < 1                 R² = 1
 ───────────────────┼──────────────────────────────────────────────┼──
 worse than         same as             explains part of           perfect
 guessing the       guessing the        the variation              predictions
 mean               mean
```

> **Can $R^2$ be negative?** On the **training** set, linear regression always gets $R^2 \ge 0$. On the **test** set it can be negative, which means the model does worse on new data than just predicting the average. That's a strong sign of a broken or overfitted model.

### The limitations of $R^2$

**1. It has no units, so it doesn't tell you if the errors are acceptable.** $R^2 = 0.95$ sounds great, but if prices range from 20M to 500M, the typical error could still be tens of millions. Always report **MAE or RMSE alongside $R^2$**.

**2. It depends on how spread out the target is.** The same model with the same errors gets a **higher** $R^2$ on a test set with very varied prices, and a **lower** one on a set of similar prices, because $\text{SS}_{\text{tot}}$ changes. That makes $R^2$ hard to compare across datasets.

**3. Training $R^2$ never goes down when you add features**, even useless random ones. The model can always use an extra feature to fit noise a little better. **Adjusted $R^2$** adds a penalty per feature:

$$R^2_{\text{adj}} = 1 - (1 - R^2)\,\frac{m - 1}{m - p - 1}$$

where $m$ is the number of instances and $p$ the number of features. For example, $R^2 = 0.99$ with $m = 16$ and $p = 2$ gives $1 - 0.01 \times 15/13 \approx 0.9885$. The better fix is to evaluate on a **test set**, where useless features don't help.

**4. A high $R^2$ doesn't prove the model is right.** A straight line through clearly **curved** data can still score a high $R^2$. Check the **residuals** (section 7) to see whether a line is the right shape.

**5. It's unreliable on tiny test sets.** In your Lesson 8 script, `test_size=0.2` of 20 houses is **4 houses**. One unusual house can swing $R^2$ dramatically. Use **cross-validation** (section 9) for a steadier estimate.

**6. It says nothing about cause and effect.** A high $R^2$ means the features **predict** the target, not that they **cause** it.

---

## 6. Comparing against a baseline

A model is only useful if it beats a **trivial strategy that needs no learning**. That trivial strategy is the **baseline**.

| Baseline | Predicts | scikit-learn |
|---|---|---|
| Mean | Always $\bar{y}$ | `DummyRegressor(strategy="mean")` |
| Median | Always the median of $y$ | `DummyRegressor(strategy="median")` |
| Simple rule | A domain rule of thumb, e.g. "price = 15,000 × area" | Write it yourself |

**Worked example.** Mean baseline: always predict 36. Median baseline: always predict 37.

| House | $y$ | Error vs mean (36) | Error vs median (37) |
|---|---|---|---|
| A | 20 | −16 | −17 |
| B | 28 | −8 | −9 |
| C | 37 | 1 | 0 |
| D | 44 | 8 | 7 |
| E | 51 | 15 | 14 |

| Model | MAE | MSE | RMSE | $R^2$ |
|---|---|---|---|---|
| Mean baseline | 48/5 = **9.6** | 610/5 = **122** | **11.05** | **0** |
| Median baseline | 47/5 = **9.4** | 615/5 = **123** | **11.09** | ≈ −0.008 |
| **Linear regression** | **0.48** | **0.32** | **0.566** | **0.997** |

What this shows:

- Linear regression cuts the typical error from about **9.6M to 0.48M**, a **20× improvement**, so the model is clearly learning something real.
- $R^2$ **is** a baseline comparison: $1 - \text{MSE}_{\text{model}} / \text{MSE}_{\text{mean baseline}} = 1 - 0.32/122 \approx 0.997$.
- The **median** baseline has the lower **MAE** and the **mean** baseline has the lower **MSE**. That's not a coincidence: the median is the single number that minimises absolute error, and the mean is the one that minimises squared error.

```python
from sklearn.dummy import DummyRegressor

baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

print("Baseline MAE:", mean_absolute_error(y_test, baseline_pred))
print("Model MAE:   ", mean_absolute_error(y_test, y_pred))
```

> Fit the baseline on the **training set only**, just like the real model. Its "mean" must be the training mean, not the test mean.

**If your model doesn't clearly beat the baseline**, the features may not contain useful information, the model may be broken, or there may be a bug (for example, misaligned `X` and `y`).

---

## 7. Residual analysis: checking whether a line is the right shape

Metrics give one number. A **residual plot** shows *how* the model is wrong:

```python
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted price (M LKR)")
plt.ylabel("Residual (actual − predicted)")
plt.title("Residual plot")
plt.show()
```

```text
 GOOD: random scatter          BAD: curved pattern           BAD: funnel shape
 residual                      residual                      residual
   │  •    •   •                 │ •                 •         │              •
   │ •  •    •   •               │   •             •           │          •
  0┼─────────────────           0┼─────•─────────•───         0┼──•──•──•───────
   │   •   •  •   •              │        •  •  •              │          •
   │ •    •    •                 │                             │              •
   └───────────── ŷ              └───────────── ŷ              └───────────── ŷ
 no pattern left to learn      the data curves: a straight     errors grow with price:
                               line is the wrong shape         less reliable for
                               (UNDERFITTING)                  expensive houses
```

The worked example's residuals (−0.4, −0.2, 1.0, 0.2, −0.6) are small and show no clear pattern, so a straight line suits that data.

---

## 8. Underfitting and overfitting

| | Underfitting | Overfitting |
|---|---|---|
| **What happens** | The model is **too simple** to capture the pattern | The model is **too flexible** and fits the **noise** too |
| **Training error** | High | Very low (often ≈ 0) |
| **Test error** | High (close to training error) | **Much higher than training error** |
| **Géron calls it** | High **bias** | High **variance** |
| **Fixes** | More features, a more flexible model, less regularisation | More data, a simpler model, more regularisation, fewer features |

### Demonstration: polynomial regression

Linear regression can fit **curves** if you add powers of the feature ($x^2$, $x^3$, …). This is **polynomial regression** (Géron Chapter 4), and the **degree** controls how flexible the model is:

```text
degree 0:  ŷ = b                                     (a flat line, i.e. the mean baseline)
degree 1:  ŷ = w₁x + b                               (a straight line)
degree 4:  ŷ = w₁x + w₂x² + w₃x³ + w₄x⁴ + b         (a wiggly curve)
```

On the 5-house worked example:

| Degree | Training error | Prediction for **4,000 sq ft** | Prediction for **500 sq ft** | Verdict |
|---|---|---|---|---|
| 0 (mean) | MSE 122 | 36 | 36 | **Underfits**: ignores area completely |
| 1 (line) | MSE 0.32 | 0.0156 × 4000 + 4.8 = **67.2** | 0.0156 × 500 + 4.8 = **12.6** | **Good fit**: sensible trend |
| 4 | **MSE 0** | **≈ 98** | **≈ 21** | **Overfits** |

The degree-4 curve has 5 parameters for 5 points, so it passes **exactly** through every house (training error 0). But look at what it learned:

- A 500 sq ft house would cost **about 21M**, *more* than the 1,000 sq ft house A (20M). That makes no sense.
- At 4,000 sq ft it predicts **about 98M**, shooting far above the trend, because the curve bends wildly outside the data.

It memorised the noise in 5 prices instead of learning "bigger houses cost more".

```python
from numpy.polynomial import Polynomial

x = df["area"].to_numpy()
for degree in [0, 1, 4]:
    p = Polynomial.fit(x, y, degree)
    train_mse = np.mean((y - p(x)) ** 2)
    print(f"degree {degree}: train MSE {train_mse:7.2f}   "
          f"4000 sq ft -> {p(4000):5.1f}   500 sq ft -> {p(500):5.1f}")
```

> **A training error of 0 should worry you, not please you.** With enough parameters, a model can memorise any dataset, so training error alone tells you nothing about generalisation.

### How to spot each one in practice

Compare **training** and **test** error side by side. Add this to your Lesson 8 script:

```python
train_mae = mean_absolute_error(y_train, model.predict(X_train))
test_mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"Train MAE: {train_mae:.2f}M   Test MAE: {test_mae:.2f}M")
```

```text
 error
   │ ╲                                    ╱  test error
   │  ╲                                ╱
   │   ╲          sweet spot        ╱
   │    ╲            ↓           ╱
   │     ╲_______________________
   │      ╲
   │        ╲___
   │             ╲____                       training error
   │                   ╲_________________
   └──────────────────────────────────────── model complexity (e.g. degree)
     UNDERFITTING   │   GOOD   │   OVERFITTING
     both high      │          │   big gap between them
```

| Train error | Test error | Diagnosis |
|---|---|---|
| High | High | Underfitting |
| Low | Low (similar) | Good fit |
| Very low | Much higher | Overfitting |
| Higher than test | Lower | Probably luck from a tiny test set; use cross-validation |

### The bias–variance trade-off

Géron splits a model's error on new data into three parts:

- **Bias**: error from **wrong assumptions** (for example, assuming a line when the data curves). Causes underfitting.
- **Variance**: error from being **too sensitive** to the particular training sample. Causes overfitting.
- **Irreducible error**: **noise** in the data itself (things the features don't capture: the neighbourhood, the condition, the seller's mood). No model can remove it.

Making a model more flexible **lowers bias but raises variance**. The goal is the balance point.

---

## 9. More reliable evaluation: cross-validation

With 20 houses, a single 4-house test set is too small to trust. **k-fold cross-validation** splits the data into $k$ parts (folds), trains $k$ times, and tests on each fold once:

```text
 fold:     1        2        3        4        5
 run 1: [ TEST ][ train ][ train ][ train ][ train ]
 run 2: [ train ][ TEST ][ train ][ train ][ train ]
 run 3: [ train ][ train ][ TEST ][ train ][ train ]
 run 4: [ train ][ train ][ train ][ TEST ][ train ]
 run 5: [ train ][ train ][ train ][ train ][ TEST ]
                                       → 5 scores → report the mean ± std
```

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    LinearRegression(), X, y, cv=5, scoring="neg_mean_absolute_error"
)
mae_scores = -scores        # scikit-learn wants "higher is better", so error metrics come out negative
print(f"MAE: {mae_scores.mean():.2f} ± {mae_scores.std():.2f} M LKR")
```

Every house gets used for testing exactly once, and the **± std** shows how much the score depends on which houses happen to be in the test set.

---

## 10. Putting it together: an evaluation checklist

Whenever you train a regression model:

1. **Split** the data before training (or use cross-validation for small datasets).
2. **Fit a baseline** (`DummyRegressor`) on the training set.
3. **Fit the model** on the training set.
4. **Report MAE and RMSE** (in the target's units) and **$R^2$**, on the **test** set.
5. **Compare with the baseline.** Is the improvement large enough to matter?
6. **Compare train vs test error** to spot underfitting or overfitting.
7. **Plot the residuals** to check that the model's shape fits the data.
8. **Read the weights**, with units, and only "holding the others fixed".

---

## 11. Exercises

### By hand

1. A model predicts `[12, 18, 35]` for three houses whose actual prices are `[10, 20, 30]` (M LKR). Calculate the residuals, MAE, MSE, RMSE and $R^2$.
2. Using $\widehat{\text{price}} = 0.0156 \times \text{area} + 4.8$, predict the price of a 1,750 sq ft house. Is this interpolation or extrapolation?
3. Two models have the same MAE but different RMSE. What does that tell you about their errors?
4. If you measured area in **square metres** instead of sq ft, what would happen to $w$, to $b$, and to the predictions?

<details>
<summary>Answers</summary>

1. Residuals $y - \hat{y}$: −2, 2, −5. MAE = (2 + 2 + 5)/3 = **3**. MSE = (4 + 4 + 25)/3 = **11**. RMSE = √11 ≈ **3.32**. The mean is 20, so $\text{SS}_{\text{tot}}$ = 100 + 0 + 100 = 200 and $\text{SS}_{\text{res}}$ = 33, giving $R^2$ = 1 − 33/200 = **0.835**.
2. 0.0156 × 1750 + 4.8 = 27.3 + 4.8 = **32.1 M LKR**. It's **interpolation**, because 1,750 lies inside the training range of 1,000–3,000.
3. The one with the higher RMSE has its error concentrated in **a few large misses**; the other has more even errors.
4. $w$ gets about **10.76× larger** (≈ 0.168 M per m²), because each unit is now bigger. $b$ and **all the predictions stay the same**.

</details>

### On your Lesson 8 script ([lesson_08.py](../lesson08/lesson_08.py))

5. Print `model.coef_` and `model.intercept_`. Write the model as an equation with units, and explain each weight in one sentence. Does the sign of the `bedrooms` weight surprise you? (Hint: `area` and `bedrooms` are strongly correlated.)
6. Add `root_mean_squared_error` and compare it with MAE. Is the gap small or large?
7. Add a `DummyRegressor(strategy="mean")` baseline. How many times smaller is your model's MAE?
8. Print **training** MAE next to **test** MAE. Underfitting, overfitting, or a good fit?
9. Your test set has only 4 houses. Change `random_state` to 0, 1, 2 and 3 and record the test $R^2$ each time. How much does it move? Then run 5-fold `cross_val_score` and compare.
10. Train a model with **only** `area`, then one with **only** `bedrooms`. Compare their cross-validated MAE with the two-feature model. Does adding `bedrooms` actually help?
11. Draw the residual plot for your test predictions. Do you see a pattern?

---

## 12. Key takeaways

- Linear regression predicts with a **weighted sum plus a bias**: $\hat{y} = w_1 x_1 + \dots + w_n x_n + b$.
- It learns the weights by **minimising MSE** (least squares), either exactly with the **normal equation** or step by step with **gradient descent**.
- A **weight** is the change in prediction per unit of its feature, **holding the other features fixed**. Its size depends on units, and correlated features make it unstable. The **bias** often has no real-world meaning.
- **MAE** is the average error in the target's units and is robust to outliers. **MSE** punishes big errors and is used for training. **RMSE** is MSE back in the target's units.
- **$R^2$** compares your model with always predicting the mean. It's unitless, rises with extra features on training data, can be negative on test data, and is unstable on small test sets. **Always pair it with MAE or RMSE.**
- **Underfitting** means high error everywhere. **Overfitting** means low training error and high test error. Watch the **gap**.
- A model is only useful if it **clearly beats a baseline**.

## Further reading

- Géron, *Hands-On Machine Learning* (2nd ed.):
  - **Chapter 2**: "Select a Performance Measure" (RMSE vs MAE), and "Better Evaluation Using Cross-Validation"
  - **Chapter 4**: Linear Regression, the Normal Equation, Gradient Descent, Polynomial Regression, Learning Curves
- [scikit-learn: LinearRegression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
- [scikit-learn: Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)

**Next up:** feature scaling, regularised linear models (Ridge, Lasso), and learning curves, the tools for controlling overfitting.
