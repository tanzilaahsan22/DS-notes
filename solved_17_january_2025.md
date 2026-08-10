# Solved Exam Notebook — 17th January 2025

Course: **1MS041 Introduction to Data Science**

This notebook contains the questions and solved code answers for all 3 problems from the uploaded exam PDF.

Keep the required data files in a folder named `data/` next to this notebook:

- `data/SVD.csv`
- `data/websites.csv`
- `data/fraud.csv`
- `Utils.py` if your course environment provides it

Run the cells from top to bottom.



```python
# Insert your anonymous exam ID as a string in the variable below
examID = "XXX"

# Common imports used throughout the notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```

---

# Problem 1 — SVD and Anomaly Detection

**Maximum points: 14**

This problem is about Singular Value Decomposition (SVD), explained variance, low-rank approximation, and anomaly detection using reconstruction error.

We use a data matrix \(X\) with shape:

\[
n_{samples} \times n_{dimensions}
\]

The SVD is:

\[
X = U D V^T
\]

where:

- \(U\) contains the left singular vectors
- \(D\) contains the singular values
- \(V\) contains the right singular vectors


## Problem 1, Part 1

Load `data/SVD.csv`, compute the SVD, and store the required variables.



```python
# Part 1: Load data and compute SVD

# Load the CSV as a numpy array.
# header=None is used because exam data files often do not contain column names.
problem1_data = pd.read_csv("SVD.csv", header=None).values.astype(float)

# Check dtype and shape
print("problem1_data dtype:", problem1_data.dtype)
print("problem1_data shape:", problem1_data.shape)

# Compute reduced SVD.
# np.linalg.svd returns U, singular values s, and Vt where Vt = V.T
problem1_U, problem1_D, problem1_Vt = np.linalg.svd(problem1_data, full_matrices=False)

# The exam asks for V, not V.T
problem1_V = problem1_Vt.T

# First singular vectors as 1D arrays
problem1_first_right_singular_vector = problem1_V[:, 0].flatten()
problem1_first_left_singular_vector = problem1_U[:, 0].flatten()

# Shape checks
print("U shape:", problem1_U.shape)
print("D shape:", problem1_D.shape)
print("V shape:", problem1_V.shape)
print("First right singular vector shape:", problem1_first_right_singular_vector.shape)
print("First left singular vector shape:", problem1_first_left_singular_vector.shape)

# Optional reconstruction check: X should be close to U @ diag(D) @ V.T
reconstructed_X = problem1_U @ np.diag(problem1_D) @ problem1_V.T
print("SVD reconstruction close:", np.allclose(problem1_data, reconstructed_X))

```

## Problem 1, Part 2

Calculate the explained variance for using 1, 2, ..., \(n_{dimensions}\) singular vectors.

For SVD, variance explained by singular values is based on squared singular values:

\[
\text{explained variance}(k)
=
\frac{\sum_{i=1}^{k} d_i^2}{\sum_{i=1}^{d} d_i^2}
\]

Choose the smallest number of components needed to explain at least **95%** of the variance.



```python
# Part 2: Explained variance

singular_values_squared = problem1_D ** 2

problem1_explained_variance = np.cumsum(singular_values_squared) / np.sum(singular_values_squared)

# Smallest number of components explaining at least 95%
problem1_num_components = int(np.searchsorted(problem1_explained_variance, 0.95) + 1)

print("Explained variance:")
print(problem1_explained_variance)
print("Number of components needed for at least 95% variance:", problem1_num_components)
print("Last explained variance value:", problem1_explained_variance[-1])

```

## Problem 1, Part 3

Construct the best rank-\(k\) approximation using the number of components chosen in Part 2.

The approximation is:

\[
X_k = U_k D_k V_k^T
\]



```python
# Part 3: Best low-rank approximation

k = problem1_num_components

problem1_approximation = (
    problem1_U[:, :k]
    @ np.diag(problem1_D[:k])
    @ problem1_V[:, :k].T
)

print("Approximation shape:", problem1_approximation.shape)
print("Original data shape:", problem1_data.shape)

```

### Free text answer for Problem 1, Part 3

Each row of `problem1_approximation` is the reconstructed version of the corresponding row in `problem1_data`, using only the first `problem1_num_components` singular vectors.

So, for sample \(i\), row \(i\) of the approximating matrix is the best low-dimensional reconstruction of row \(i\) of the original data, after keeping only the main SVD directions that explain at least 95% of the variance.

This removes smaller-variance directions/noise and keeps the dominant structure of the original data. If a row is reconstructed poorly, it may be unusual compared with the main pattern of the dataset, which is why the reconstruction error can be used for anomaly detection.


## Problem 1, Part 4

Compute row-wise Euclidean reconstruction error:

\[
e_i = \lVert X_i - \hat{X}_i \rVert_2
\]

Then plot the empirical distribution function and choose a threshold so that exactly 10 samples are above it.



```python
# Part 4: Reconstruction error and outlier detection

# Row-wise Euclidean distance between original data and approximation
problem1_reconstruction_error = np.linalg.norm(
    problem1_data - problem1_approximation,
    axis=1
)

print("Reconstruction error shape:", problem1_reconstruction_error.shape)

# Empirical distribution function plot
sorted_errors = np.sort(problem1_reconstruction_error)
ecdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

plt.figure(figsize=(7, 5))
plt.step(sorted_errors, ecdf, where="post")
plt.xlabel("Reconstruction error")
plt.ylabel("Empirical distribution function")
plt.title("ECDF of reconstruction error")
plt.grid(True)
plt.show()

# Choose threshold so that exactly 10 samples are above it.
# sorted_errors[-10] is the 10th largest error.
# sorted_errors[-11] is the 11th largest error.
# Using their midpoint makes exactly the top 10 strictly larger, assuming no ties.
if len(sorted_errors) <= 10:
    raise ValueError("Need more than 10 samples to select 10 outliers.")

problem1_threshold = (sorted_errors[-10] + sorted_errors[-11]) / 2

problem1_outliers = problem1_data[problem1_reconstruction_error > problem1_threshold]

print("Threshold:", problem1_threshold)
print("Number of outliers:", problem1_outliers.shape[0])
print("Outliers shape:", problem1_outliers.shape)

```

---

# Problem 2 — Markov Chain Website Preloading

**Maximum points: 14**

The dataset contains website behavior. Each row contains:

- user
- source page
- destination page

We estimate a homogeneous Markov chain transition matrix from source to destination pages.


## Problem 2, Part 1

Load `data/websites.csv` and estimate the transition matrix.

The maximum likelihood estimate is:

\[
\hat{P}_{ij}
=
\frac{\#\{\text{transitions from page } i \text{ to page } j\}}
{\#\{\text{transitions from page } i\}}
\]



```python
# Part 1: Estimate transition matrix

websites_df = pd.read_csv("data/websites.csv", header=None)

print(websites_df.head())
print("Shape:", websites_df.shape)

# The problem says each row is: user, source, destination.
# Therefore source is column 1 and destination is column 2.
source = websites_df.iloc[:, 1].astype(int).values
destination = websites_df.iloc[:, 2].astype(int).values

problem2_n_states = int(max(source.max(), destination.max()) + 1)

counts = np.zeros((problem2_n_states, problem2_n_states), dtype=float)

for s, d in zip(source, destination):
    counts[s, d] += 1

row_sums = counts.sum(axis=1, keepdims=True)

# Normalize rows to get transition probabilities.
# If a row has no outgoing transitions, we use a self-loop to avoid division by zero.
problem2_transition_matrix = np.zeros_like(counts, dtype=float)

for i in range(problem2_n_states):
    if row_sums[i, 0] > 0:
        problem2_transition_matrix[i, :] = counts[i, :] / row_sums[i, 0]
    else:
        problem2_transition_matrix[i, i] = 1.0

print("Number of states:", problem2_n_states)
print("Transition matrix shape:", problem2_transition_matrix.shape)
print("Rows sum to 1:", np.allclose(problem2_transition_matrix.sum(axis=1), 1))

```

## Problem 2, Part 2

Starting from page 1, simulate 10,000 next-page load times.

Rules:

- If the next page is not preloaded: load time \(\sim Exp(1)\), mean \(1\)
- If the next page is preloaded: load time \(\sim Exp(10)\), mean \(1/10\)

In NumPy, `np.random.exponential(scale=mean)` uses the mean as the `scale`.



```python
# Part 2: Simulate load times from page 1

np.random.seed(1)

n_simulations = 10000
start_page = 1

transition_probs_from_page_1 = problem2_transition_matrix[start_page]

# Simulate actual next pages according to transition probabilities
next_pages = np.random.choice(
    np.arange(problem2_n_states),
    size=n_simulations,
    p=transition_probs_from_page_1
)

# Most likely next page
top_1_page = np.argsort(transition_probs_from_page_1)[-1:]

# Two most likely next pages
top_2_pages = np.argsort(transition_probs_from_page_1)[-2:]

# Case 1: preload only the most likely page
is_preloaded_top = np.isin(next_pages, top_1_page)

problem2_page_load_times_top = np.where(
    is_preloaded_top,
    np.random.exponential(scale=1/10, size=n_simulations),
    np.random.exponential(scale=1, size=n_simulations)
)

# Case 2: preload the two most likely pages
is_preloaded_two = np.isin(next_pages, top_2_pages)

problem2_page_load_times_two = np.where(
    is_preloaded_two,
    np.random.exponential(scale=1/10, size=n_simulations),
    np.random.exponential(scale=1, size=n_simulations)
)

print("Top 1 page:", top_1_page)
print("Top 2 pages:", top_2_pages)
print("problem2_page_load_times_top shape:", problem2_page_load_times_top.shape)
print("problem2_page_load_times_two shape:", problem2_page_load_times_two.shape)
print("Average load time, top 1 preload:", problem2_page_load_times_top.mean())
print("Average load time, top 2 preload:", problem2_page_load_times_two.mean())

```

## Problem 2, Part 3

Compare average empirical load time with theoretical load time for no preloading.

Without preloading, every page load is \(Exp(1)\), so the expected load time is:

\[
E[T] = 1
\]



```python
# Part 3: Compare with no preloading

# Expected load time without preloading
problem2_avg = 1.0

# Is no-preloading average larger than average load time after preloading top page?
problem2_comparison = bool(problem2_avg > problem2_page_load_times_top.mean())

print("Expected average load time without preloading:", problem2_avg)
print("Average load time with top 1 preloading:", problem2_page_load_times_top.mean())
print("Average load time with top 2 preloading:", problem2_page_load_times_two.mean())
print("Does preloading top 1 improve load time?", problem2_comparison)

```

### Free text answer for Problem 2, Part 3

The theoretical expected load time without preloading is 1 second because the load time follows an exponential distribution with mean 1.

I compared this value with the empirical average of `problem2_page_load_times_top`. If the simulated average after preloading the most likely next page is smaller than 1, then preloading improves the load time.

The variable `problem2_comparison` is `True` when the no-preloading expected load time is larger than the simulated average load time after preloading, meaning that preloading improves the average load time.


## Problem 2, Part 4

Calculate the stationary distribution \(\pi\), satisfying:

\[
\pi P = \pi
\]

Then calculate the expected load time with respect to the stationary distribution when preloading the most likely next page for each current page.



```python
# Part 4: Stationary distribution and stationary expected load time

# Stationary distribution is the left eigenvector of P with eigenvalue 1.
# Equivalently, it is the right eigenvector of P.T with eigenvalue 1.
eigenvalues, eigenvectors = np.linalg.eig(problem2_transition_matrix.T)

index = np.argmin(np.abs(eigenvalues - 1))

stationary = np.real(eigenvectors[:, index])

# Make non-negative and normalize
stationary = np.abs(stationary)
problem2_stationary_distribution = stationary / stationary.sum()

print("Stationary distribution shape:", problem2_stationary_distribution.shape)
print("Stationary distribution sums to:", problem2_stationary_distribution.sum())

# Expected load time for each state when preloading the most likely next page:
# If probability of most likely page is p, expected load time is:
# p * 0.1 + (1 - p) * 1
most_likely_probs = problem2_transition_matrix.max(axis=1)

expected_load_time_per_state = most_likely_probs * (1/10) + (1 - most_likely_probs) * 1

problem2_avg_stationary = float(
    np.sum(problem2_stationary_distribution * expected_load_time_per_state)
)

print("Expected load time per state:")
print(expected_load_time_per_state)
print("Stationary average load time:", problem2_avg_stationary)

```

---

# Problem 3 — Fraud Detection Threshold and Cost

**Maximum points: 12**

We are given classifier probabilities for fraud. The goal is to choose thresholds using cost and 0-1 loss.

Costs:

| Outcome | Meaning | Cost |
|---|---|---:|
| TP | Detecting fraud and blocking transaction | 100 |
| TN | Allowing legitimate transaction | 0 |
| FP | Legitimate transaction incorrectly blocked | 120 |
| FN | Fraud missed | 600 |


## Problem 3 — Data initialization

Run this cell first. It loads the fraud data, splits train/test/validation, trains logistic regression, and creates predicted probabilities.



```python
# RUN THIS CELL TO GET THE DATA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROBLEM3_DF = pd.read_csv("data/fraud.csv")

Y = PROBLEM3_DF["Class"].values
X = PROBLEM3_DF[[f"V{i}" for i in range(1, 5)] + ["Amount"]].values

# Use the course Utils.py if available.
# If it is not available, this fallback uses sklearn train_test_split.
try:
    from Utils import train_test_validation
    PROBLEM3_X_train, PROBLEM3_X_test, PROBLEM3_X_val, PROBLEM3_y_train, PROBLEM3_y_test, PROBLEM3_y_val = train_test_validation(
        X, Y, shuffle=True, random_state=1
    )
except Exception as e:
    print("Could not import/use Utils.train_test_validation. Using sklearn fallback.")
    print("Original error:", e)
    from sklearn.model_selection import train_test_split

    X_temp, PROBLEM3_X_test, y_temp, PROBLEM3_y_test = train_test_split(
        X, Y, test_size=0.2, random_state=1, shuffle=True, stratify=Y
    )
    PROBLEM3_X_train, PROBLEM3_X_val, PROBLEM3_y_train, PROBLEM3_y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=1, shuffle=True, stratify=y_temp
    )

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(PROBLEM3_X_train, PROBLEM3_y_train)

PROBLEM3_y_pred_proba_val = lr.predict_proba(PROBLEM3_X_val)[:, 1]
PROBLEM3_y_true_val = PROBLEM3_y_val

PROBLEM3_y_pred_proba_test = lr.predict_proba(PROBLEM3_X_test)[:, 1]
PROBLEM3_y_true_test = PROBLEM3_y_test

print("Validation size:", len(PROBLEM3_y_true_val))
print("Test size:", len(PROBLEM3_y_true_test))

```

## Problem 3, Part 1

Implement the cost function and plot cost as a function of threshold from 0 to 1 with 0.01 increments.



```python
# Part 1: Cost function and cost-threshold plot

def cost(y_true, y_predict_proba, threshold):
    # Average cost per sample for a binary classifier.
    # Positive class 1 = fraud
    # Negative class 0 = legitimate
    #
    # Costs:
    # TP = 100
    # TN = 0
    # FP = 120
    # FN = 600

    y_true = np.asarray(y_true)
    y_predict_proba = np.asarray(y_predict_proba)

    y_pred = (y_predict_proba >= threshold).astype(int)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    total_cost = TP * 100 + TN * 0 + FP * 120 + FN * 600

    return float(total_cost / len(y_true))

thresholds = np.arange(0, 1.01, 0.01)

validation_costs = np.array([
    cost(PROBLEM3_y_true_val, PROBLEM3_y_pred_proba_val, t)
    for t in thresholds
])

plt.figure(figsize=(7, 5))
plt.plot(thresholds, validation_costs)
plt.xlabel("Threshold")
plt.ylabel("Average cost")
plt.title("Validation cost as a function of threshold")
plt.grid(True)
plt.show()

print("Minimum validation cost in grid:", validation_costs.min())
print("Threshold at minimum validation cost:", thresholds[np.argmin(validation_costs)])

```

## Problem 3, Part 2

Find the threshold minimizing cost on validation data. Then calculate:

- validation cost
- predicted labels
- precision and recall for class 1
- precision and recall for class 0



```python
# Part 2: Best threshold by cost, precision, and recall

problem3_threshold = float(thresholds[np.argmin(validation_costs)])

problem3_cost_val = cost(
    PROBLEM3_y_true_val,
    PROBLEM3_y_pred_proba_val,
    problem3_threshold
)

problem3_y_pred_val = (PROBLEM3_y_pred_proba_val >= problem3_threshold).astype(int)

def precision_recall_for_class(y_true, y_pred, cls):
    # Precision and recall for a chosen class.
    # Treat cls as the positive class.
    y_true_cls = (y_true == cls)
    y_pred_cls = (y_pred == cls)

    TP_cls = np.sum(y_true_cls & y_pred_cls)
    FP_cls = np.sum(~y_true_cls & y_pred_cls)
    FN_cls = np.sum(y_true_cls & ~y_pred_cls)

    precision = TP_cls / (TP_cls + FP_cls) if (TP_cls + FP_cls) > 0 else 0.0
    recall = TP_cls / (TP_cls + FN_cls) if (TP_cls + FN_cls) > 0 else 0.0

    return float(precision), float(recall)

problem3_precision_1, problem3_recall_1 = precision_recall_for_class(
    PROBLEM3_y_true_val,
    problem3_y_pred_val,
    1
)

problem3_precision_0, problem3_recall_0 = precision_recall_for_class(
    PROBLEM3_y_true_val,
    problem3_y_pred_val,
    0
)

print("Best cost threshold:", problem3_threshold)
print("Validation cost at best threshold:", problem3_cost_val)
print("Precision class 1:", problem3_precision_1)
print("Recall class 1:", problem3_recall_1)
print("Precision class 0:", problem3_precision_0)
print("Recall class 0:", problem3_recall_0)

```

## Problem 3, Part 3

Find the threshold minimizing 0-1 loss on validation data.

0-1 loss is the classification error rate:

\[
\frac{\#\{y_i \neq \hat{y}_i\}}{n}
\]

Then calculate the cost difference:

\[
\text{cost at 0-1 threshold} - \text{cost at cost-optimal threshold}
\]



```python
# Part 3: Best threshold by 0-1 loss

zero_one_losses = []

for t in thresholds:
    y_pred_t = (PROBLEM3_y_pred_proba_val >= t).astype(int)
    zero_one_loss_t = np.mean(y_pred_t != PROBLEM3_y_true_val)
    zero_one_losses.append(zero_one_loss_t)

zero_one_losses = np.array(zero_one_losses)

problem3_threshold_01 = float(thresholds[np.argmin(zero_one_losses)])

cost_at_threshold_01 = cost(
    PROBLEM3_y_true_val,
    PROBLEM3_y_pred_proba_val,
    problem3_threshold_01
)

problem3_cost_difference = float(cost_at_threshold_01 - problem3_cost_val)

print("Best 0-1 loss threshold:", problem3_threshold_01)
print("Minimum 0-1 loss:", zero_one_losses.min())
print("Cost at 0-1 threshold:", cost_at_threshold_01)
print("Cost at cost-optimal threshold:", problem3_cost_val)
print("Cost difference:", problem3_cost_difference)

```

## Problem 3, Part 4

Use Hoeffding's inequality to provide a 95% confidence interval around the optimal cost on the test data.

For bounded random variables \(X_i \in [a,b]\), Hoeffding gives:

\[
P\left(|\bar{X} - E[X]| \geq \epsilon\right)
\leq
2\exp\left(
\frac{-2n\epsilon^2}{(b-a)^2}
\right)
\]

Solving for \(\epsilon\):

\[
\epsilon =
(b-a)
\sqrt{
\frac{\log(2/\delta)}{2n}
}
\]

Here:

- confidence = 95%, so \(\delta = 0.05\)
- minimum possible cost is 0
- maximum possible cost is 600



```python
# Part 4: Hoeffding confidence interval for test cost

test_threshold = problem3_threshold

# Predicted labels on test data using the cost-optimal validation threshold
test_pred = (PROBLEM3_y_pred_proba_test >= test_threshold).astype(int)

# Individual costs on the test set
test_individual_costs = np.zeros(len(PROBLEM3_y_true_test), dtype=float)

test_individual_costs[(PROBLEM3_y_true_test == 1) & (test_pred == 1)] = 100  # TP
test_individual_costs[(PROBLEM3_y_true_test == 0) & (test_pred == 0)] = 0    # TN
test_individual_costs[(PROBLEM3_y_true_test == 0) & (test_pred == 1)] = 120  # FP
test_individual_costs[(PROBLEM3_y_true_test == 1) & (test_pred == 0)] = 600  # FN

test_average_cost = float(np.mean(test_individual_costs))

# Hoeffding interval
delta = 0.05
n = len(test_individual_costs)
a = 0
b = 600

epsilon = (b - a) * np.sqrt(np.log(2 / delta) / (2 * n))

problem3_lower_bound = float(test_average_cost - epsilon)
problem3_upper_bound = float(test_average_cost + epsilon)

# Cost cannot be below 0, so clipping the lower bound is reasonable.
problem3_lower_bound = max(0.0, problem3_lower_bound)

print("Test average cost:", test_average_cost)
print("Hoeffding epsilon:", epsilon)
print("95% lower bound:", problem3_lower_bound)
print("95% upper bound:", problem3_upper_bound)

```

### Free text answer for Problem 3, Part 4

I used the cost-optimal threshold found on the validation data and applied it to the independent test data.

For each test observation, I calculated its individual cost according to the problem statement:

- TP costs 100
- TN costs 0
- FP costs 120
- FN costs 600

The sample mean of these individual costs is the empirical test cost.

To construct a 95% confidence interval, I used Hoeffding's inequality. The assumption is that the test observations are independent and that the individual costs are bounded. The cost is always between 0 and 600, so I used \(a=0\) and \(b=600\). With \(\delta=0.05\), Hoeffding gives an error radius \(\epsilon\), and the interval is:

\[
[\bar{C} - \epsilon, \bar{C} + \epsilon]
\]

where \(\bar{C}\) is the empirical average test cost.

