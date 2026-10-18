# Lesson 10 — Classification with Machine Learning

In Lessons 8 and 9 every model predicted a **number**: a house price. Many real software problems instead ask you to predict a **category**:

| Problem | Question | Answer is a... |
|---|---|---|
| **Spam detection** | Is this email spam? | category: `spam` / `not spam` |
| **Fraud detection** | Is this card transaction fraudulent? | category: `fraud` / `legit` |
| **Customer churn** | Will this customer cancel their subscription? | category: `churn` / `stay` |
| **Ticket routing** | Which team should handle this support ticket? | category: `billing` / `technical` / `account` |

Predicting categories is called **classification**. This lesson follows **Chapter 3, "Classification"** of Géron's *Hands-On Machine Learning* (2nd ed.), which builds a digit classifier and uses it to explain classification metrics.

## Learning objectives

By the end of this lesson, you'll be able to:

- Distinguish regression from classification.
- Understand binary and multiclass classification.
- Train a classification model using scikit-learn.
- Interpret accuracy, precision, recall, and F1-score.
- Understand why accuracy alone can be misleading.
- Build a small classification program in VS Code.

---

## 1. Regression vs classification

Both are **supervised learning**: you train on examples that include the right answer. The difference is the **type of target**.

| | Regression (Lessons 8–9) | Classification (this lesson) |
|---|---|---|
| **Target** | A **number** on a continuous scale | A **category** (class, label) from a fixed set |
| **Question** | "How much?" / "How many?" | "Which one?" / "Yes or no?" |
| **Example output** | `32.1` (M LKR) | `"churn"` or `1` |
| **Being "close" means anything?** | Yes: predicting 31 when the answer is 32 is nearly right | No: the prediction is either the right class or the wrong one |
| **Typical metrics** | MAE, RMSE, $R^2$ | Accuracy, precision, recall, F1 |
| **scikit-learn examples** | `LinearRegression`, `DecisionTreeRegressor` | `LogisticRegression`, `DecisionTreeClassifier`, `KNeighborsClassifier` |

> **The same data can support either task.** "Predict the house price" is regression. "Predict whether a house sells for more than 40M" is classification. You decide which by choosing the target.

### Classes are usually stored as numbers

Models need numbers, so classes are usually **encoded** as integers:

```text
binary:      0 = stay        1 = churn
multiclass:  0 = billing     1 = technical     2 = account
```

These numbers are **labels, not quantities**. Class 2 isn't "bigger" than class 1, and averaging them is meaningless. In the binary case, the class you care about detecting (spam, fraud, churn) is by convention the **positive class (`1`)**, even when it's bad news.

---

## 2. Binary and multiclass classification

### Binary classification: two classes

Each instance belongs to one of **two** classes: spam / not spam, fraud / legit, churn / stay. Géron starts Chapter 3 with a **"5-detector"**: a classifier that answers just *"Is this handwritten digit a 5, or not?"*

### Multiclass classification: three or more classes

Each instance belongs to exactly **one** of several classes: ticket routing (billing / technical / account), or recognising a handwritten digit (0–9).

Some algorithms handle many classes natively (decision trees, k-nearest neighbours, logistic regression in scikit-learn). Others are binary by nature, so they're combined using one of two strategies:

| Strategy | How it works | Classifiers trained for 10 digits |
|---|---|---|
| **One-versus-the-rest (OvR)** | One binary classifier **per class** ("0 or not?", "1 or not?", …). Pick the class whose classifier is most confident | 10 |
| **One-versus-one (OvO)** | One binary classifier **per pair** of classes ("0 or 1?", "0 or 2?", …). Pick the class that wins the most duels | 10 × 9 / 2 = 45 |

scikit-learn chooses a strategy **automatically** when you give a binary algorithm a multiclass target, so you rarely do this by hand.

### Related variants (for awareness)

| Type | Each instance has... | Example |
|---|---|---|
| **Binary** | 1 of 2 classes | Spam or not |
| **Multiclass** | 1 of 3+ classes | Which team handles the ticket |
| **Multilabel** | **Several** yes/no labels at once | A photo tagged `beach` **and** `sunset` **and** `people` |
| **Multioutput** | Several targets, each multiclass | Géron's example: denoising an image, predicting every pixel's intensity |

---

## 3. How a classifier makes a prediction: logistic regression

Despite its name, **logistic regression is a classification algorithm**. It's the natural next step from Lesson 9's linear regression.

### Step 1: The same weighted sum as linear regression

$$z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

$z$ can be any number from $-\infty$ to $+\infty$, so it isn't a probability yet.

### Step 2: Squash it into a probability with the sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

```text
 σ(z)
 1.0 ┤                            ________________
     │                       ___/
     │                     /
 0.5 ┤- - - - - - - - - - ●           ← z = 0 gives exactly 0.5
     │                  /
     │             ___/
 0.0 ┤____________/
     └──────────┬─────────┬─────────┬──────────── z
               -5         0         5
```

| $z$ | −3 | −1.4 | 0 | 0.2 | 1 | 3 |
|---|---|---|---|---|---|---|
| $\sigma(z)$ | 0.047 | 0.198 | **0.5** | 0.550 | 0.731 | 0.953 |

Large positive $z$ gives a probability near 1, large negative $z$ gives one near 0, and $z = 0$ gives exactly 0.5.

### Step 3: Turn the probability into a class with a threshold

$$\hat{y} = \begin{cases} 1 & \text{if } \sigma(z) \ge 0.5 \\ 0 & \text{otherwise} \end{cases}$$

**Worked example: churn from support calls.** Suppose a trained model has learned $z = 0.8 \times \text{support\_calls} - 3$:

| Support calls | $z$ | $P(\text{churn}) = \sigma(z)$ | Prediction (threshold 0.5) |
|---|---|---|---|
| 0 | −3.0 | 0.047 | stay |
| 2 | −1.4 | 0.198 | stay |
| 4 | 0.2 | 0.550 | **churn** |
| 5 | 1.0 | 0.731 | **churn** |

The **decision boundary** is where $z = 0$: $0.8 \times \text{calls} - 3 = 0$, so $\text{calls} = 3.75$. Customers with 4 or more support calls are predicted to churn.

### Interpreting the weights

A positive weight **raises** the probability of the positive class as the feature increases; a negative weight **lowers** it. The weight of 0.8 means more support calls mean higher churn risk. Unlike linear regression, a weight doesn't add a fixed amount to the *probability*; it adds a fixed amount to $z$, and the sigmoid bends that into a probability.

> **How it's trained:** linear regression minimises MSE. Logistic regression minimises **log loss** (cross-entropy), which strongly penalises being **confidently wrong**, for example predicting 0.99 for a customer who stays. There's no closed-form solution, so it's solved iteratively, with gradient descent-style methods (Lesson 9).

### Other classifiers you'll meet

| Classifier | Idea | scikit-learn |
|---|---|---|
| **Logistic regression** | Weighted sum → sigmoid → threshold | `LogisticRegression` |
| **k-nearest neighbours** | Vote among the *k* most similar training instances | `KNeighborsClassifier` |
| **Decision tree** | A series of yes/no questions ("support_calls > 3?") | `DecisionTreeClassifier` |
| **Random forest** | Many trees vote | `RandomForestClassifier` |
| **SGD classifier** | A linear model trained with stochastic gradient descent (Géron's first classifier in Ch. 3) | `SGDClassifier` |

They all share the same API: `fit`, `predict`, and (for most) `predict_proba`.

---

## 4. Training a classifier in scikit-learn

The workflow is the same as Lesson 8. Only the model and the metrics change.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)               # TRAIN

model.predict(X_test)                     # INFERENCE: classes, e.g. [0, 1, 0, 0, ...]
model.predict_proba(X_test)               # probability of EACH class, e.g. [[0.80, 0.20], [0.27, 0.73], ...]
```

| Method | Returns | Shape |
|---|---|---|
| `predict(X)` | The predicted class for each row | `(n,)` |
| `predict_proba(X)` | One probability per class per row; each row sums to 1 | `(n, n_classes)` |
| `model.classes_` | The order of the columns in `predict_proba` | e.g. `[0, 1]` |

`predict_proba(X)[:, 1]` gives the probability of the **positive class** (column 1).

### Stratified splitting

Classification data is often **imbalanced**. If only 10% of customers churn, a random split might put 14% of the churners in the test set or only 6%. Use `stratify=y` to keep the **same class proportions** in both sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Géron uses stratified sampling for the same reason in Chapter 2.

---

## 5. The confusion matrix

Every metric in this lesson comes from counting the **four possible outcomes** of a binary prediction:

|  | **Predicted: negative (0)** | **Predicted: positive (1)** |
|---|---|---|
| **Actual: negative (0)** | **TN**: true negative ✔ | **FP**: false positive ✘ |
| **Actual: positive (1)** | **FN**: false negative ✘ | **TP**: true positive ✔ |

The names mean: **True/False** = was the prediction correct? **Positive/Negative** = what did the model predict?

| Outcome | Spam filter | Fraud detector | Churn model |
|---|---|---|---|
| **TP** | Spam caught | Fraud blocked | Churner identified |
| **TN** | Real email delivered | Legit payment allowed | Loyal customer left alone |
| **FP** (a "false alarm") | **Real email sent to spam** | Legit payment **blocked**; angry customer | Retention discount given to someone who'd have stayed |
| **FN** (a "miss") | Spam reaches the inbox | **Fraud goes through**; money lost | Churner leaves without anyone noticing |

### Worked example: a spam filter

A spam filter is tested on **100 emails: 20 spam, 80 not spam**.

|  | Predicted: not spam | Predicted: spam | Total |
|---|---|---|---|
| **Actual: not spam** | TN = **70** | FP = **10** | 80 |
| **Actual: spam** | FN = **5** | TP = **15** | 20 |
| **Total** | 75 | 25 | 100 |

### In scikit-learn

```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
# [[TN, FP],      rows    = ACTUAL class (0, then 1)
#  [FN, TP]]      columns = PREDICTED class (0, then 1)
```

> **Watch the order.** scikit-learn puts **actual** on the rows, **predicted** on the columns, and sorts classes ascending, so **TP is bottom-right**. Some books and websites draw it the other way round. Always check the axis labels.

To unpack it into four variables:

```python
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
```

---

## 6. Accuracy, precision, recall and F1

Using the spam example (TP = 15, TN = 70, FP = 10, FN = 5):

### Accuracy: how often is the model right?

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{15 + 70}{100} = 0.85$$

**"85% of all emails were classified correctly."**

### Precision: when it says "spam", how often is it right?

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{15}{15 + 10} = \frac{15}{25} = 0.60$$

**"Of the 25 emails flagged as spam, only 60% really were."** The other 10 were real emails lost to the spam folder.

### Recall: of all the real spam, how much did it catch?

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{15}{15 + 5} = \frac{15}{20} = 0.75$$

**"The filter caught 75% of the spam."** Recall is also called **sensitivity** or the **true positive rate**.

```text
                 PRECISION looks at this column                 RECALL looks at this row
                 (everything predicted positive)                (everything actually positive)

                 pred 0    pred 1                               pred 0    pred 1
     actual 0  [   TN   |  [FP]  ]                  actual 0  [   TN   |   FP   ]
     actual 1  [   FN   |  [TP]  ]                  actual 1  [  [FN]  |  [TP]  ]
                           ▲                                   ◄───────────────►
                    TP / (TP + FP)                               TP / (TP + FN)
```

### F1-score: one number that balances both

The F1-score is the **harmonic mean** of precision and recall:

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.60 \times 0.75}{0.60 + 0.75} = \frac{0.90}{1.35} \approx 0.667$$

An equivalent shortcut that uses the counts directly:

$$F_1 = \frac{2\,TP}{2\,TP + FP + FN} = \frac{30}{30 + 10 + 5} = \frac{30}{45} \approx 0.667$$

**Why a harmonic mean instead of the usual average?** It's dragged down by the **weaker** of the two. A model with precision 1.0 and recall 0.0 gets a regular average of 0.5, but an F1 of **0**. To score a high F1, a model needs **both** precision and recall to be high.

### Summary

| Metric | Question it answers | Formula | Spam example |
|---|---|---|---|
| **Accuracy** | How often is it right overall? | (TP + TN) / all | 0.85 |
| **Precision** | When it predicts positive, how often is it right? | TP / (TP + FP) | 0.60 |
| **Recall** | Of the actual positives, how many did it find? | TP / (TP + FN) | 0.75 |
| **F1** | Balance of precision and recall | 2PR / (P + R) | 0.667 |

All four range from 0 to 1, and higher is better.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)    # for the positive class (1) by default
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)
```

### Which metric matters? It depends on the cost of each mistake

| Situation | Worse mistake | Prioritise | Why |
|---|---|---|---|
| **Spam filter** | FP: a real email (a job offer, an invoice) disappears into spam | **Precision** | Users tolerate some spam; they don't tolerate losing real email |
| **Fraud detection** | FN: fraud goes through | **Recall** | A missed fraud costs real money; a false alarm costs a confirmation SMS |
| **Churn prediction** | FN: a churner leaves unnoticed | Usually **recall** | A retention offer is cheap compared with losing a customer |
| **Medical screening** | FN: a disease is missed | **Recall** | A false alarm leads to a follow-up test; a miss can be fatal |
| **Video "safe for kids" filter** | FP: an unsafe video is marked safe | **Precision** of the "safe" class | Géron's example: better to reject many good videos than let one bad one through |

> **Rule of thumb:** if false alarms are expensive, prioritise **precision**. If misses are expensive, prioritise **recall**. If both matter about equally, use **F1**.

---

## 7. Why accuracy alone can be misleading

### The accuracy paradox

A bank has **1,000 transactions, 10 of which are fraud** (1%). Consider a "model" that predicts **"legit" for every transaction**, without even looking at the data:

|  | Predicted: legit | Predicted: fraud |
|---|---|---|
| **Actual: legit** | TN = 990 | FP = 0 |
| **Actual: fraud** | FN = 10 | TP = 0 |

| Metric | Value |
|---|---|
| **Accuracy** | (0 + 990) / 1000 = **99%** 🎉 |
| **Recall** | 0 / (0 + 10) = **0%**: it catches **no** fraud at all |
| **Precision** | 0 / 0, undefined (it never predicts fraud); scikit-learn reports 0 with a warning |
| **F1** | **0** |

The model scores **99% accuracy** and is **completely useless**.

Géron shows the same thing in Chapter 3 with a `Never5Classifier` that always answers "not a 5". It gets **over 90% accuracy** on MNIST simply because only about 10% of digits are 5s.

### Why it happens

When one class dominates (**class imbalance**), accuracy is mostly a measure of how well the model handles the **majority** class, which is usually the class you care about *least*. Imbalance is the normal case for the most valuable problems:

| Problem | Typical positive rate |
|---|---|
| Fraud | Well under 1% |
| Churn | Often 5–25% per period |
| Spam | Varies a lot by inbox |
| Rare disease | Can be under 0.1% |

### What to do instead

1. **Check the class balance first:** `y.value_counts(normalize=True)`.
2. **Compare with a baseline:** `DummyClassifier(strategy="most_frequent")` always predicts the majority class. Its accuracy equals the majority share, and your model must **clearly** beat it.
3. **Report precision, recall and F1** for the positive class, and look at the **confusion matrix**.
4. **Split with `stratify=y`** so the test set has the same imbalance.

```python
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("Baseline accuracy:", accuracy_score(y_test, baseline.predict(X_test)))
print("Baseline recall:  ", recall_score(y_test, baseline.predict(X_test)))   # 0.0
```

---

## 8. The precision/recall trade-off

`predict()` uses a **threshold of 0.5** on the predicted probability. Changing the threshold trades precision against recall.

**Worked example.** A spam filter scores 10 emails (5 are actually spam):

| Email | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| $P(\text{spam})$ | 0.95 | 0.85 | 0.80 | 0.70 | 0.60 | 0.45 | 0.40 | 0.30 | 0.20 | 0.10 |
| Actually spam? | ✔ | ✔ | ✘ | ✔ | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |

An email is flagged as spam when its score is **≥ the threshold**:

| Threshold | Flagged as spam | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| **0.90** (strict) | 1 | 1 | 0 | 4 | 1/1 = **1.00** | 1/5 = **0.20** | 0.33 |
| **0.50** (default) | 1–5 | 4 | 1 | 1 | 4/5 = **0.80** | 4/5 = **0.80** | 0.80 |
| **0.35** (lenient) | 1–7 | 5 | 2 | 0 | 5/7 ≈ **0.71** | 5/5 = **1.00** | 0.83 |

```text
  threshold:     LOW ◄──────────────────────────────► HIGH
                 flags more emails                   flags fewer emails
  recall:        ↑ catches more spam                 ↓ misses more spam
  precision:     ↓ more false alarms                 ↑ fewer false alarms
```

- **Raise the threshold** → the model only flags what it's very sure about → **higher precision, lower recall**.
- **Lower the threshold** → it flags more → **higher recall, lower precision**.

You can't usually maximise both. You **choose** the threshold based on the business cost of each mistake (section 6).

```python
proba = model.predict_proba(X_test)[:, 1]     # P(positive) for each row
threshold = 0.3
y_pred_custom = (proba >= threshold).astype(int)
```

> Géron's warning: *"If someone says, 'Let's reach 99% precision,' you should ask, 'At what recall?'"* Any precision is easy to hit if you're allowed to miss almost everything.

**Beyond this lesson:** `precision_recall_curve` plots precision and recall at **every** threshold, and the **ROC curve** / **ROC AUC** summarise performance across all thresholds. Géron covers both in Chapter 3.

---

## 9. Multiclass metrics

With more than two classes, precision, recall and F1 are calculated **per class**, treating each class in turn as "positive" and all the others as "negative".

**Worked example: routing 40 support tickets.**

| Actual ↓ / Predicted → | billing | technical | account | Total (support) |
|---|---|---|---|---|
| **billing** | **8** | 1 | 1 | 10 |
| **technical** | 2 | **15** | 3 | 20 |
| **account** | 0 | 2 | **8** | 10 |
| **Total predicted** | 10 | 18 | 12 | 40 |

The **diagonal** holds the correct predictions: accuracy = (8 + 15 + 8) / 40 = 31/40 = **0.775**.

| Class | Precision (÷ column total) | Recall (÷ row total) | F1 |
|---|---|---|---|
| billing | 8/10 = 0.80 | 8/10 = 0.80 | 0.80 |
| technical | 15/18 ≈ 0.83 | 15/20 = 0.75 | 30/38 ≈ 0.79 |
| account | 8/12 ≈ 0.67 | 8/10 = 0.80 | 16/22 ≈ 0.73 |

**Reading it:** the `account` class has the lowest precision. Of the 12 tickets sent to the account team, 4 didn't belong there (3 were technical, 1 billing). Looking **off the diagonal** tells you which classes the model confuses, which Géron calls **error analysis**.

### Averaging per-class scores

| Average | How | Value (F1) | Use when |
|---|---|---|---|
| **Macro** | Simple mean of the class scores; every class counts equally | (0.80 + 0.79 + 0.73) / 3 ≈ **0.77** | Small classes matter as much as big ones |
| **Weighted** | Mean weighted by each class's **support** (number of actual instances) | (10 × 0.80 + 20 × 0.79 + 10 × 0.73) / 40 ≈ **0.78** | You want the score to reflect the class mix |

```python
f1_score(y_test, y_pred, average="macro")
f1_score(y_test, y_pred, average="weighted")
```

### `classification_report`: everything at once

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

For the ticket example above, it prints this (classes in alphabetical order):

```text
              precision    recall  f1-score   support

     account       0.67      0.80      0.73        10
     billing       0.80      0.80      0.80        10
   technical       0.83      0.75      0.79        20

    accuracy                           0.78        40
   macro avg       0.77      0.78      0.77        40
weighted avg       0.78      0.78      0.78        40
```

- One row per class, with its precision, recall, F1 and **support** (how many actual instances it has).
- `accuracy`: overall accuracy.
- `macro avg` / `weighted avg`: the two averages above.

This works for binary problems too, where it shows a row for **each** class. In a binary report, focus on the row for the **positive class**.

---

## 10. Build a small classification program in VS Code

You'll build a **customer churn predictor** for a subscription service.

### 10.1 Setup

1. **Open the folder:** in VS Code, *File → Open Folder…* → `AI_Engineering`.
2. **Create a virtual environment** (once). Open the terminal with **Ctrl + `** and run:

   ```bash
   cd ML
   python3 -m venv .venv
   source .venv/bin/activate
   pip install numpy pandas matplotlib scikit-learn
   ```

3. **Select the interpreter:** **Cmd + Shift + P** → *Python: Select Interpreter* → choose `ML/.venv/bin/python`. This makes VS Code's ▶ Run button and error highlighting use that environment.
4. **Create** `ML/lesson10/lesson_10.py`.
5. **Run** it with the ▶ button (top right), or in the terminal: `python lesson10/lesson_10.py`.

> If you see `ModuleNotFoundError: No module named 'sklearn'`, VS Code is using a different Python. Check the interpreter shown in the bottom-right status bar.

### 10.2 The program

Build it up **one step at a time**, running after each step so you can see what changes.

**Step 1: Create a churn dataset.** You'll generate 1,000 customers from a hidden rule, as in Lesson 8, so the data has a realistic pattern and imbalance:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

tenure_months = rng.integers(1, 73, n)                  # how long they've been a customer
monthly_charges = rng.uniform(1500, 10000, n).round(-1) # LKR per month
support_calls = rng.poisson(1.5, n)                     # calls in the last 3 months
monthly_contract = rng.integers(0, 2, n)                # 1 = month-to-month, 0 = yearly

# hidden rule: short tenure, high charges, many calls and no contract -> more likely to churn
z = (-2.5 - 0.05 * tenure_months + 0.0002 * monthly_charges
     + 0.6 * support_calls + 1.2 * monthly_contract)
churn_probability = 1 / (1 + np.exp(-z))                # the sigmoid from section 3
churn = (rng.random(n) < churn_probability).astype(int)

df = pd.DataFrame({
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges,
    "support_calls": support_calls,
    "monthly_contract": monthly_contract,
    "churn": churn,
})

print(df.head())
print("\nClass balance:")
print(df["churn"].value_counts(normalize=True))
```

The last line shows that only a **minority** of customers churn, so the data is **imbalanced**. Note the exact share; you'll need it in step 4.

**Step 2: Features, target and a stratified split.**

```python
from sklearn.model_selection import train_test_split

X = df[["tenure_months", "monthly_charges", "support_calls", "monthly_contract"]]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train churn rate:", y_train.mean().round(3), " Test churn rate:", y_test.mean().round(3))
```

The two churn rates should be almost identical. That's what `stratify=y` does. (The mean of a 0/1 column is the share of 1s.)

**Step 3: Train a logistic regression.**

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

> **Why `StandardScaler`?** `monthly_charges` is in the thousands while `monthly_contract` is 0 or 1. Logistic regression's solver works much better when features are on similar scales; without scaling it may print a `ConvergenceWarning`. `make_pipeline` chains the scaler and the model so they're always applied together. Scaling and pipelines get a full lesson later; for now, treat this as "the standard way to use logistic regression".

**Step 4: Evaluate, and compare with a baseline.**

```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
base_pred = baseline.predict(X_test)

print(f"\n{'Model':<20} {'Accuracy':>9} {'Precision':>10} {'Recall':>7} {'F1':>6}")
for name, pred in [("Baseline (majority)", base_pred), ("Logistic regression", y_pred)]:
    print(f"{name:<20} {accuracy_score(y_test, pred):>9.3f} "
          f"{precision_score(y_test, pred, zero_division=0):>10.3f} "
          f"{recall_score(y_test, pred):>7.3f} {f1_score(y_test, pred):>6.3f}")

print("\nConfusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["stay", "churn"]))
```

**What to look for:**

- The **baseline's accuracy equals the "stay" share** from step 1, yet its **recall is 0**. That's the accuracy paradox from section 7.
- Logistic regression's **accuracy may be only somewhat higher** than the baseline's. That doesn't look impressive, but its **recall and F1 are far above 0**: it actually finds churners. This is why you can't judge on accuracy alone.
- In the confusion matrix, find **FN** (bottom-left): churners the model missed.

`zero_division=0` tells scikit-learn to report precision as 0 (instead of printing a warning) when the baseline never predicts the positive class.

**Step 5: Predict for new customers, with probabilities.**

```python
new_customers = pd.DataFrame({
    "tenure_months":    [3,    60],
    "monthly_charges":  [9000, 2500],
    "support_calls":    [5,    0],
    "monthly_contract": [1,    0],
})

proba = model.predict_proba(new_customers)[:, 1]
for i, p in enumerate(proba):
    label = "CHURN RISK" if p >= 0.5 else "likely to stay"
    print(f"Customer {i + 1}: P(churn) = {p:.2f} -> {label}")
```

Customer 1 (new, expensive plan, many complaints, no contract) should get a **high** churn probability, and customer 2 (long-term, cheap plan, no complaints, yearly contract) a **low** one. The probability is more useful than the bare label: a retention team can call the **highest-risk** customers first.

**Step 6: Tune the threshold for recall.** Say the business would rather offer a few unnecessary discounts than lose churners:

```python
proba_test = model.predict_proba(X_test)[:, 1]

print(f"\n{'Threshold':>9} {'Precision':>10} {'Recall':>7} {'F1':>6}")
for threshold in [0.7, 0.5, 0.3, 0.2]:
    pred = (proba_test >= threshold).astype(int)
    print(f"{threshold:>9} {precision_score(y_test, pred, zero_division=0):>10.3f} "
          f"{recall_score(y_test, pred):>7.3f} {f1_score(y_test, pred):>6.3f}")
```

As the threshold **drops**, recall should **rise** and precision should **fall**, the trade-off from section 8. Pick the threshold that best fits the business goal.

### 10.3 Extend the program (exercises)

1. **Another classifier:** replace `LogisticRegression()` with `DecisionTreeClassifier(max_depth=4, random_state=42)` from `sklearn.tree` (no scaler needed). Compare precision, recall and F1. Then remove `max_depth`. Does the tree overfit? (Compare **training** and **test** F1, as in Lesson 9.)
2. **Class weights:** try `LogisticRegression(class_weight="balanced")`. This makes mistakes on the rare class cost more during training. What happens to recall and precision?
3. **Read the weights:** `model[-1].coef_` gives the logistic regression's weights (on the scaled features). Which features push toward churn (positive) and which toward staying (negative)? Do they match the hidden rule in step 1?
4. **Plot the confusion matrix:**

   ```python
   import matplotlib.pyplot as plt
   from sklearn.metrics import ConfusionMatrixDisplay

   ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["stay", "churn"])
   plt.title("Churn model: confusion matrix")
   plt.show()
   ```

5. **Multiclass, Géron-style:** `from sklearn.datasets import load_digits` loads 1,797 handwritten digits (8 × 8 pixels = 64 features, 10 classes). It's a small built-in cousin of Géron's MNIST. Train a `LogisticRegression(max_iter=1000)`, print the `classification_report`, and use the confusion matrix to find **which digits get confused with each other**.

---

## 11. Practice by hand

1. A fraud model is tested on 500 transactions: TP = 8, FP = 12, FN = 2, TN = 478. Calculate accuracy, precision, recall and F1. Is this a good fraud model?
2. Model A has precision 0.95 and recall 0.40. Model B has precision 0.70 and recall 0.75. Which would you choose for (a) a spam filter, and (b) cancer screening? Calculate each model's F1.
3. Explain why a model with 97% accuracy might be worse than one with 90% accuracy.
4. You lower a classifier's threshold from 0.5 to 0.2. What happens to the number of positive predictions, to precision, and to recall?
5. Classify each task as regression, binary or multiclass classification: (a) predict the number of support tickets next week; (b) predict whether a pull request will be merged; (c) predict the programming language of a code snippet; (d) predict a server's CPU usage in 5 minutes.

<details>
<summary>Answers</summary>

1. Accuracy = (8 + 478)/500 = **0.972**. Precision = 8/20 = **0.40**. Recall = 8/10 = **0.80**. F1 = 16/(16 + 12 + 2) = 16/30 ≈ **0.533**. It catches 80% of fraud, which is decent, but 60% of its alerts are false alarms. Whether that's acceptable depends on the cost of an alert. Note that "always legit" would score 490/500 = **0.98** accuracy, *higher* than this model, while catching nothing.
2. F1(A) = 2 × 0.95 × 0.40 / 1.35 ≈ **0.563**. F1(B) = 2 × 0.70 × 0.75 / 1.45 ≈ **0.724**. (a) For spam, false alarms are costly, so **A** (high precision) is defensible. (b) For cancer screening, misses are costly, so choose **B** (higher recall).
3. On imbalanced data, the 97% model might just predict the majority class (recall ≈ 0), while the 90% model actually finds the positive cases. Check recall and the confusion matrix.
4. **More** positive predictions, recall **goes up (or stays the same)**, and precision usually **goes down**.
5. (a) regression (b) binary classification (c) multiclass classification (d) regression.

</details>

---

## 12. Key takeaways

- **Regression** predicts numbers; **classification** predicts categories. **Binary** means 2 classes; **multiclass** means 3 or more (handled internally with OvR or OvO when needed).
- **Logistic regression** = linear regression's weighted sum → **sigmoid** → probability → **threshold** → class.
- The scikit-learn API is the same as before: `fit`, `predict`, plus `predict_proba` for probabilities. Use `stratify=y` when splitting.
- The **confusion matrix** (TN, FP, FN, TP) is the source of every classification metric. In scikit-learn, rows are actual and columns predicted.
- **Precision** asks "when it says yes, is it right?". **Recall** asks "did it find all the yeses?". **F1** balances them.
- **Accuracy misleads on imbalanced data.** Always check the class balance, compare with a `DummyClassifier`, and report precision, recall and F1.
- The **threshold** trades precision for recall. Choose it based on what each kind of mistake **costs**.

## Further reading

- Géron, *Hands-On Machine Learning* (2nd ed.), **Chapter 3, "Classification"**: MNIST, training a binary classifier, performance measures (confusion matrix, precision/recall, the trade-off, ROC curve), multiclass classification, and error analysis
- Géron, **Chapter 4**: "Logistic Regression" (the sigmoid, the cost function, decision boundaries)
- [scikit-learn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [scikit-learn: LogisticRegression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

**Next up:** feature scaling and pipelines, cross-validation for classifiers, and ROC curves.
