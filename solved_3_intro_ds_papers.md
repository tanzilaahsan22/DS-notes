# Solved Jupyter Notebook — 3 Intro to Data Science Papers

Papers included:
1. **17 January 2025**
2. **16 June 2025**
3. **21 August 2025**

The code assumes the exam data files are available in the same structure as the official exam notebook, for example `data/SVD.csv`, `data/websites.csv`, `data/fraud.csv`, and `data/salaries.csv`.

Run each paper section separately if you copy it into the official exam notebook, because some variable names repeat across papers.


```python
# Common imports and helper functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

np.random.seed(42)

def empirical_cdf_values(x):
    xs = np.sort(np.asarray(x))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys

def plot_edf(x, confidence=None, title="Empirical distribution function", xlabel="x"):
    xs, ys = empirical_cdf_values(x)
    plt.figure()
    plt.step(xs, ys, where="post", label="EDF")
    if confidence is not None:
        n = len(xs)
        alpha = 1 - confidence
        eps = np.sqrt(np.log(2 / alpha) / (2 * n))
        plt.step(xs, np.maximum(ys - eps, 0), where="post", linestyle="--", label=f"{int(confidence*100)}% lower")
        plt.step(xs, np.minimum(ys + eps, 1), where="post", linestyle="--", label=f"{int(confidence*100)}% upper")
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("F_n(x)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def stationary_distribution(P):
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))
    pi = np.real(eigvecs[:, idx])
    pi = np.abs(pi)
    pi = pi / pi.sum()
    return pi

def simulate_preload_load_times(P, start_state, n_users, preload_k, rate_not_preloaded, rate_preloaded, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    probs = P[start_state]
    next_pages = rng.choice(np.arange(P.shape[0]), size=n_users, p=probs)
    preloaded_pages = np.argsort(probs)[-preload_k:]
    is_preloaded = np.isin(next_pages, preloaded_pages)

    load_times = np.empty(n_users)
    load_times[is_preloaded] = rng.exponential(scale=1 / rate_preloaded, size=is_preloaded.sum())
    load_times[~is_preloaded] = rng.exponential(scale=1 / rate_not_preloaded, size=(~is_preloaded).sum())
    return load_times

def expected_load_time_with_policy(P, state_distribution, preload_k, rate_not_preloaded, rate_preloaded):
    means = []
    for i in range(P.shape[0]):
        probs = P[i]
        preloaded_pages = np.argsort(probs)[-preload_k:]
        p_preloaded = probs[preloaded_pages].sum()
        means.append(p_preloaded * (1 / rate_preloaded) + (1 - p_preloaded) * (1 / rate_not_preloaded))
    return float(np.dot(state_distribution, np.array(means)))
```

# Paper 1 — 17 January 2025

## Problem 1: SVD and anomaly detection


```python
# Part 1
problem1_data = pd.read_csv("data/SVD.csv", header=None).values.astype(float)

problem1_U, problem1_D, problem1_Vt = np.linalg.svd(problem1_data, full_matrices=False)
problem1_V = problem1_Vt.T

problem1_first_right_singular_vector = problem1_V[:, 0].flatten()
problem1_first_left_singular_vector = problem1_U[:, 0].flatten()

print(problem1_data.shape, problem1_U.shape, problem1_D.shape, problem1_V.shape)

# Part 2 — at least 95% explained variance
singular_value_energy = problem1_D ** 2
problem1_explained_variance = np.cumsum(singular_value_energy) / np.sum(singular_value_energy)
problem1_num_components = int(np.searchsorted(problem1_explained_variance, 0.95) + 1)

# Part 3
k = problem1_num_components
problem1_approximation = problem1_U[:, :k] @ np.diag(problem1_D[:k]) @ problem1_Vt[:k, :]

# Part 4
problem1_reconstruction_error = np.linalg.norm(problem1_data - problem1_approximation, axis=1)
plot_edf(problem1_reconstruction_error, title="EDF of reconstruction error", xlabel="Row-wise Euclidean reconstruction error")

sorted_errors = np.sort(problem1_reconstruction_error)
problem1_threshold = float(sorted_errors[-11])  # threshold so top 10 are above, assuming no tie
top10_idx = np.argsort(problem1_reconstruction_error)[-10:]
problem1_outliers = problem1_data[top10_idx]

problem1_explained_variance, problem1_num_components, problem1_threshold, problem1_outliers.shape
```

**Free text answer for SVD approximation**

Each row of `problem1_approximation` is the reconstructed version of the corresponding original row in `problem1_data`, but only using the first `problem1_num_components` singular directions. Geometrically, every original data point is projected onto the lower-dimensional subspace spanned by the first singular vectors, then mapped back to the original feature space. So each row is the best rank-`k` approximation of that sample, keeping the main structure/variance and removing smaller directions that are treated as noise.

## Problem 2: Website Markov chain


```python
# Part 1
websites = pd.read_csv("data/websites.csv")
cols_lower = {c.lower(): c for c in websites.columns}
source_col = cols_lower.get("source", websites.columns[-2])
destination_col = cols_lower.get("destination", websites.columns[-1])

sources = websites[source_col].astype(int).values
destinations = websites[destination_col].astype(int).values
problem2_n_states = int(max(sources.max(), destinations.max()) + 1)

counts = np.zeros((problem2_n_states, problem2_n_states), dtype=float)
for s, d in zip(sources, destinations):
    counts[s, d] += 1

row_sums = counts.sum(axis=1, keepdims=True)
problem2_transition_matrix = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)

for i in range(problem2_n_states):
    if row_sums[i, 0] == 0:
        problem2_transition_matrix[i, i] = 1.0

# Part 2 — January: start page 1, Exp(1) if not preloaded, Exp(10) if preloaded
rng = np.random.default_rng(42)
problem2_page_load_times_top = simulate_preload_load_times(
    problem2_transition_matrix, start_state=1, n_users=10000, preload_k=1,
    rate_not_preloaded=1, rate_preloaded=10, rng=rng
)
rng = np.random.default_rng(43)
problem2_page_load_times_two = simulate_preload_load_times(
    problem2_transition_matrix, start_state=1, n_users=10000, preload_k=2,
    rate_not_preloaded=1, rate_preloaded=10, rng=rng
)

# Part 3
problem2_avg = float(1 / 1)
problem2_comparison = bool(problem2_avg > np.mean(problem2_page_load_times_top))

# Part 4
problem2_stationary_distribution = stationary_distribution(problem2_transition_matrix)
problem2_avg_stationary = expected_load_time_with_policy(
    problem2_transition_matrix, problem2_stationary_distribution,
    preload_k=1, rate_not_preloaded=1, rate_preloaded=10
)

problem2_avg, np.mean(problem2_page_load_times_top), np.mean(problem2_page_load_times_two), problem2_comparison, problem2_avg_stationary
```

**Free text answer for January Problem 2, Part 3**

Without preloading, the theoretical expected load time is `1` second because the load time is exponentially distributed with rate 1. I compare this with the empirical average when preloading the most likely next page. If the empirical average is smaller than 1, then preloading improves the average load time. Preloading two pages should usually improve it further because a larger probability mass is loaded using the faster exponential distribution.

## Problem 3: Fraud threshold and cost


```python
# RUN THIS CELL TO GET THE DATA
PROBLEM3_DF = pd.read_csv("data/fraud.csv")
Y = PROBLEM3_DF["Class"].values
X = PROBLEM3_DF[["V%d" % i for i in range(1, 5)] + ["Amount"]].values

try:
    from Utils import train_test_validation
    PROBLEM3_X_train, PROBLEM3_X_test, PROBLEM3_X_val, PROBLEM3_y_train, PROBLEM3_y_test, PROBLEM3_y_val = train_test_validation(
        X, Y, shuffle=True, random_state=1
    )
except Exception:
    X_temp, PROBLEM3_X_test, y_temp, PROBLEM3_y_test = train_test_split(
        X, Y, test_size=0.2, random_state=1, shuffle=True, stratify=Y
    )
    PROBLEM3_X_train, PROBLEM3_X_val, PROBLEM3_y_train, PROBLEM3_y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=1, shuffle=True, stratify=y_temp
    )

lr = LogisticRegression(max_iter=1000)
lr.fit(PROBLEM3_X_train, PROBLEM3_y_train)

PROBLEM3_y_pred_proba_val = lr.predict_proba(PROBLEM3_X_val)[:, 1]
PROBLEM3_y_true_val = PROBLEM3_y_val
PROBLEM3_y_pred_proba_test = lr.predict_proba(PROBLEM3_X_test)[:, 1]
PROBLEM3_y_true_test = PROBLEM3_y_test

# Part 1
def cost(y_true, y_predict_proba, threshold):
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_predict_proba) >= threshold).astype(int)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    total_cost = 100 * TP + 0 * TN + 120 * FP + 600 * FN
    return float(total_cost / len(y_true))

thresholds = np.arange(0, 1.01, 0.01)
costs = np.array([cost(PROBLEM3_y_true_val, PROBLEM3_y_pred_proba_val, t) for t in thresholds])

plt.figure()
plt.plot(thresholds, costs)
plt.xlabel("Threshold")
plt.ylabel("Average cost")
plt.title("Validation cost vs threshold")
plt.grid(True)
plt.show()

# Part 2
best_idx = int(np.argmin(costs))
problem3_threshold = float(thresholds[best_idx])
problem3_cost_val = float(cost(PROBLEM3_y_true_val, PROBLEM3_y_pred_proba_val, problem3_threshold))
problem3_y_pred_val = (PROBLEM3_y_pred_proba_val >= problem3_threshold).astype(int)

problem3_precision_1 = float(precision_score(PROBLEM3_y_true_val, problem3_y_pred_val, pos_label=1, zero_division=0))
problem3_recall_1 = float(recall_score(PROBLEM3_y_true_val, problem3_y_pred_val, pos_label=1, zero_division=0))
problem3_precision_0 = float(precision_score(PROBLEM3_y_true_val, problem3_y_pred_val, pos_label=0, zero_division=0))
problem3_recall_0 = float(recall_score(PROBLEM3_y_true_val, problem3_y_pred_val, pos_label=0, zero_division=0))

# Part 3
loss01 = np.array([
    np.mean((PROBLEM3_y_pred_proba_val >= t).astype(int) != PROBLEM3_y_true_val)
    for t in thresholds
])
problem3_threshold_01 = float(thresholds[int(np.argmin(loss01))])
problem3_cost_difference = float(
    cost(PROBLEM3_y_true_val, PROBLEM3_y_pred_proba_val, problem3_threshold_01)
    - cost(PROBLEM3_y_true_val, PROBLEM3_y_pred_proba_val, problem3_threshold)
)

# Part 4
y_pred_test = (PROBLEM3_y_pred_proba_test >= problem3_threshold).astype(int)
test_costs = np.where(
    (PROBLEM3_y_true_test == 1) & (y_pred_test == 1), 100,
    np.where(
        (PROBLEM3_y_true_test == 0) & (y_pred_test == 0), 0,
        np.where((PROBLEM3_y_true_test == 0) & (y_pred_test == 1), 120, 600)
    )
)
mean_test_cost = float(np.mean(test_costs))
n = len(test_costs)
epsilon = 600 * np.sqrt(np.log(2 / 0.05) / (2 * n))
problem3_lower_bound = float(max(0, mean_test_cost - epsilon))
problem3_upper_bound = float(mean_test_cost + epsilon)

{
    "threshold_cost": problem3_threshold,
    "cost_val": problem3_cost_val,
    "precision_1": problem3_precision_1,
    "recall_1": problem3_recall_1,
    "precision_0": problem3_precision_0,
    "recall_0": problem3_recall_0,
    "threshold_01": problem3_threshold_01,
    "cost_difference": problem3_cost_difference,
    "test_CI": (problem3_lower_bound, problem3_upper_bound)
}
```

**Free text answer for fraud confidence interval**

I fixed the threshold using validation data, then evaluated the cost on the test data only once. Each test example gives a bounded cost in `[0, 600]`. Hoeffding's inequality gives a distribution-free 95% confidence interval around the empirical mean test cost. I assume the test samples are independent and representative of future transactions, and that the threshold was chosen before looking at the test costs.

# Paper 2 — 16 June 2025

## Problem 1: Rejection sampling and Monte Carlo integration


```python
# Part 1
def problem1_rejection(n_samples=1):
    # Target density f(x)=2x(exp(x^2)-1)/(e-2), 0<x<1.
    # Proposal g(x)=2x, sampled with sqrt(U). Acceptance prob=(exp(x^2)-1)/(e-2).
    samples = []
    rng = np.random.default_rng()
    while len(samples) < n_samples:
        batch = max(1000, 2 * (n_samples - len(samples)))
        x = np.sqrt(rng.random(batch))
        u = rng.random(batch)
        accept_prob = (np.exp(x**2) - 1) / (np.e - 2)
        samples.extend(x[u <= accept_prob].tolist())
    return np.array(samples[:n_samples])

# Part 2
problem1_samples = problem1_rejection(100000)
x_grid = np.linspace(0, 1, 400)
true_density = 2 * x_grid * (np.exp(x_grid**2) - 1) / (np.e - 2)

plt.figure()
plt.hist(problem1_samples, bins=60, density=True, alpha=0.5, label="Samples")
plt.plot(x_grid, true_density, label="True density")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Rejection sampling check")
plt.legend()
plt.grid(True)
plt.show()

# Part 3: integral is E[sin(X)]
problem1_integral = float(np.mean(np.sin(problem1_samples)))

# Part 4: Hoeffding 95% CI for sin(X), bounded in [0, sin(1)]
n = len(problem1_samples)
epsilon = np.sin(1) * np.sqrt(np.log(2 / 0.05) / (2 * n))
problem1_interval = [float(problem1_integral - epsilon), float(problem1_integral + epsilon)]

# Part 5
def problem1_rejection_2(n_samples=1):
    # Target CDF F(x)=20*x*exp(20 - 1/x), 0<x<1/20.
    # Transform Z=1/X-20. Proposal Z~Exp(1).
    # Acceptance ratio <= 1.05.
    samples = []
    rng = np.random.default_rng()
    M = 1.05
    while len(samples) < n_samples:
        batch = max(1000, 2 * (n_samples - len(samples)))
        z = rng.exponential(scale=1.0, size=batch)
        ratio = 20 * (1 / (20 + z) + 1 / (20 + z)**2)
        u = rng.random(batch)
        z_acc = z[u <= ratio / M]
        samples.extend((1 / (20 + z_acc)).tolist())
    return np.array(samples[:n_samples])

problem1_integral, problem1_interval, problem1_rejection_2(10)
```

## Problem 2: Website Markov chain


```python
# Part 1
websites = pd.read_csv("data/websites.csv")
cols_lower = {c.lower(): c for c in websites.columns}
source_col = cols_lower.get("source", websites.columns[-2])
destination_col = cols_lower.get("destination", websites.columns[-1])

sources = websites[source_col].astype(int).values
destinations = websites[destination_col].astype(int).values
problem2_n_states = int(max(sources.max(), destinations.max()) + 1)

counts = np.zeros((problem2_n_states, problem2_n_states), dtype=float)
for s, d in zip(sources, destinations):
    counts[s, d] += 1

row_sums = counts.sum(axis=1, keepdims=True)
problem2_transition_matrix = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)

for i in range(problem2_n_states):
    if row_sums[i, 0] == 0:
        problem2_transition_matrix[i, i] = 1.0

# Part 2 — January: start page 8, Exp(3) if not preloaded, Exp(20) if preloaded
rng = np.random.default_rng(42)
problem2_page_load_times_top = simulate_preload_load_times(
    problem2_transition_matrix, start_state=8, n_users=10000, preload_k=1,
    rate_not_preloaded=3, rate_preloaded=20, rng=rng
)
rng = np.random.default_rng(43)
problem2_page_load_times_two = simulate_preload_load_times(
    problem2_transition_matrix, start_state=8, n_users=10000, preload_k=2,
    rate_not_preloaded=3, rate_preloaded=20, rng=rng
)

# Part 3
problem2_avg = float(1 / 3)
problem2_comparison = bool(problem2_avg > np.mean(problem2_page_load_times_top))

# Part 4
problem2_stationary_distribution = stationary_distribution(problem2_transition_matrix)
problem2_avg_stationary = expected_load_time_with_policy(
    problem2_transition_matrix, problem2_stationary_distribution,
    preload_k=1, rate_not_preloaded=3, rate_preloaded=20
)

problem2_avg, np.mean(problem2_page_load_times_top), np.mean(problem2_page_load_times_two), problem2_comparison, problem2_avg_stationary
```

**Free text answer for June Problem 2, Part 3**

Without preloading, the theoretical expected load time is `1/3` seconds because the load time is exponentially distributed with rate 3. I compare this with the empirical mean from the simulation where the most likely next page is preloaded. If the simulated average is lower than `1/3`, the preloading strategy improves the expected load time.

## Problem 3: Fraud threshold and cost


```python
# RUN THIS CELL TO GET THE DATA
PROBLEM3_DF = pd.read_csv("data/fraud.csv")
Y = PROBLEM3_DF["Class"].values
X = PROBLEM3_DF[["V%d" % i for i in range(1, 5)] + ["Amount"]].values

try:
    from Utils import train_test_validation
    PROBLEM3_X_train, PROBLEM3_X_test, PROBLEM3_X_val, PROBLEM3_y_train, PROBLEM3_y_test, PROBLEM3_y_val = train_test_validation(
        X, Y, shuffle=True, random_state=1
    )
except Exception:
    X_temp, PROBLEM3_X_test, y_temp, PROBLEM3_y_test = train_test_split(
        X, Y, test_size=0.2, random_state=1, shuffle=True, stratify=Y
    )
    PROBLEM3_X_train, PROBLEM3_X_val, PROBLEM3_y_train, PROBLEM3_y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=1, shuffle=True, stratify=y_temp
    )

lr = LogisticRegression(max_iter=1000)
lr.fit(PROBLEM3_X_train, PROBLEM3_y_train)

PROBLEM3_y_pred_proba_val = lr.predict_proba(PROBLEM3_X_val)[:, 1]
PROBLEM3_y_true_val = PROBLEM3_y_val
PROBLEM3_y_pred_proba_test = lr.predict_proba(PROBLEM3_X_test)[:, 1]
PROBLEM3_y_true_test = PROBLEM3_y_test

# Part 1
def cost(y_true, y_predict_proba, threshold):
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_predict_proba) >= threshold).astype(int)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    total_cost = 100 * TP + 0 * TN + 120 * FP + 600 * FN
    return float(total_cost / len(y_true))

thresholds = np.arange(0, 1.01, 0.01)
costs = np.array([cost(PROBLEM3_y_true_val, PROBLEM3_y_pred_proba_val, t) for t in thresholds])

plt.figure()
plt.plot(thresholds, costs)
plt.xlabel("Threshold")
plt.ylabel("Average cost")
plt.title("Validation cost vs threshold")
plt.grid(True)
plt.show()

# Part 2
best_idx = int(np.argmin(costs))
problem3_threshold = float(thresholds[best_idx])
problem3_cost_val = float(cost(PROBLEM3_y_true_val, PROBLEM3_y_pred_proba_val, problem3_threshold))
problem3_y_pred_val = (PROBLEM3_y_pred_proba_val >= problem3_threshold).astype(int)

problem3_precision_1 = float(precision_score(PROBLEM3_y_true_val, problem3_y_pred_val, pos_label=1, zero_division=0))
problem3_recall_1 = float(recall_score(PROBLEM3_y_true_val, problem3_y_pred_val, pos_label=1, zero_division=0))
problem3_precision_0 = float(precision_score(PROBLEM3_y_true_val, problem3_y_pred_val, pos_label=0, zero_division=0))
problem3_recall_0 = float(recall_score(PROBLEM3_y_true_val, problem3_y_pred_val, pos_label=0, zero_division=0))

# Part 3
loss01 = np.array([
    np.mean((PROBLEM3_y_pred_proba_val >= t).astype(int) != PROBLEM3_y_true_val)
    for t in thresholds
])
problem3_threshold_01 = float(thresholds[int(np.argmin(loss01))])
problem3_cost_difference = float(
    cost(PROBLEM3_y_true_val, PROBLEM3_y_pred_proba_val, problem3_threshold_01)
    - cost(PROBLEM3_y_true_val, PROBLEM3_y_pred_proba_val, problem3_threshold)
)

# Part 4
y_pred_test = (PROBLEM3_y_pred_proba_test >= problem3_threshold).astype(int)
test_costs = np.where(
    (PROBLEM3_y_true_test == 1) & (y_pred_test == 1), 100,
    np.where(
        (PROBLEM3_y_true_test == 0) & (y_pred_test == 0), 0,
        np.where((PROBLEM3_y_true_test == 0) & (y_pred_test == 1), 120, 600)
    )
)
mean_test_cost = float(np.mean(test_costs))
n = len(test_costs)
epsilon = 600 * np.sqrt(np.log(2 / 0.05) / (2 * n))
problem3_lower_bound = float(max(0, mean_test_cost - epsilon))
problem3_upper_bound = float(mean_test_cost + epsilon)

{
    "threshold_cost": problem3_threshold,
    "cost_val": problem3_cost_val,
    "precision_1": problem3_precision_1,
    "recall_1": problem3_recall_1,
    "precision_0": problem3_precision_0,
    "recall_0": problem3_recall_0,
    "threshold_01": problem3_threshold_01,
    "cost_difference": problem3_cost_difference,
    "test_CI": (problem3_lower_bound, problem3_upper_bound)
}
```

**Free text answer for fraud confidence interval**

I fixed the threshold using validation data, then evaluated the cost on the test data only once. Each test example gives a bounded cost in `[0, 600]`. Hoeffding's inequality gives a distribution-free 95% confidence interval around the empirical mean test cost. I assume the test samples are independent and representative of future transactions, and that the threshold was chosen before looking at the test costs.

# Paper 3 — 21 August 2025

## Problem 1: SVD and anomaly detection


```python
# Part 1
problem1_data = pd.read_csv("data/SVD.csv", header=None).values.astype(float)

problem1_U, problem1_D, problem1_Vt = np.linalg.svd(problem1_data, full_matrices=False)
problem1_V = problem1_Vt.T

problem1_first_right_singular_vector = problem1_V[:, 0].flatten()
problem1_first_left_singular_vector = problem1_U[:, 0].flatten()

print(problem1_data.shape, problem1_U.shape, problem1_D.shape, problem1_V.shape)

# Part 2 — at least 90% explained variance
singular_value_energy = problem1_D ** 2
problem1_explained_variance = np.cumsum(singular_value_energy) / np.sum(singular_value_energy)
problem1_num_components = int(np.searchsorted(problem1_explained_variance, 0.90) + 1)

# Part 3
k = problem1_num_components
problem1_approximation = problem1_U[:, :k] @ np.diag(problem1_D[:k]) @ problem1_Vt[:k, :]

# Part 4
problem1_reconstruction_error = np.linalg.norm(problem1_data - problem1_approximation, axis=1)
plot_edf(problem1_reconstruction_error, title="EDF of reconstruction error", xlabel="Row-wise Euclidean reconstruction error")

sorted_errors = np.sort(problem1_reconstruction_error)
problem1_threshold = float(sorted_errors[-11])  # threshold so top 10 are above, assuming no tie
top10_idx = np.argsort(problem1_reconstruction_error)[-10:]
problem1_outliers = problem1_data[top10_idx]

problem1_explained_variance, problem1_num_components, problem1_threshold, problem1_outliers.shape
```

**Free text answer for SVD approximation**

Each row of `problem1_approximation` is the reconstructed version of the corresponding original row in `problem1_data`, but only using the first `problem1_num_components` singular directions. Geometrically, every original data point is projected onto the lower-dimensional subspace spanned by the first singular vectors, then mapped back to the original feature space. So each row is the best rank-`k` approximation of that sample, keeping the main structure/variance and removing smaller directions that are treated as noise.

## Problem 2: Salary linear regression


```python
# Part 1
problem2_df = pd.read_csv("data/salaries.csv")

problem2_features = ["work_year", "experience_level", "employment_type", "remote_ratio"]
problem2_target = "salary_in_usd"

# Part 2
X = problem2_df[problem2_features]
y = problem2_df[problem2_target]
problem2_X_train, problem2_X_test, problem2_y_train, problem2_y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

# Part 3
categorical_features = ["employment_type"]
numeric_features = ["work_year", "experience_level", "remote_ratio"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

problem2_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LinearRegression())
])

problem2_model.fit(problem2_X_train, problem2_y_train)

# Part 4
problem2_y_pred = problem2_model.predict(problem2_X_test)
problem2_absolute_relative_error = np.abs((problem2_y_test.values - problem2_y_pred) / problem2_y_test.values)
problem2_mare = float(np.mean(problem2_absolute_relative_error))
problem2_residual = problem2_y_test.values - problem2_y_pred

plot_edf(
    problem2_residual,
    confidence=0.99,
    title="EDF of residuals with 99% DKW confidence bands",
    xlabel="Residual = true - predicted"
)

# Part 5
plt.figure()
plt.scatter(problem2_y_pred, problem2_y_test)
min_val = min(problem2_y_pred.min(), problem2_y_test.min())
max_val = max(problem2_y_pred.max(), problem2_y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Perfect prediction")
plt.xlabel("Predicted salary in USD")
plt.ylabel("True salary in USD")
plt.title("Predicted vs true salary on test set")
plt.legend()
plt.grid(True)
plt.show()

problem2_mare
```

**Part 6 discussion**

A smaller MARE means better performance, because it is the average prediction error as a fraction of the true salary. For example, a MARE of 0.20 means the model is wrong by about 20% of the true salary on average. Whether the value is good or bad depends on the exam data result, but salary prediction using only year, experience, employment type, and remote ratio is likely limited because many important predictors are missing.

In the predicted-versus-true scatter plot, a good model should have points close to the diagonal line `true = predicted`. If the points are widely scattered, the model has large prediction error. If the model underpredicts high salaries or overpredicts low salaries, that suggests a simple linear model is not flexible enough or that important features are missing.

## Problem 3: Markov chains A and B


```python
# PART 1 — transition matrices
# State order for Chain A: A, B, C, D
problem3_A = np.array([
    [0.0, 0.2, 0.0, 0.8],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0],
])

# State order for Chain B: A, B, C, D, E, F
problem3_B = np.array([
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
])

# PART 2
problem3_A_irreducible = False
problem3_B_irreducible = True

# PART 3
problem3_A_is_aperiodic = False
problem3_B_is_aperiodic = False
problem3_A_periods = np.array([2, 2, 2, 2])
problem3_B_periods = np.array([2, 2, 2, 2, 2, 2])

# PART 4
problem3_A_PB5 = float(np.linalg.matrix_power(problem3_A, 5)[0, 1])
problem3_B_PB5 = float(np.linalg.matrix_power(problem3_B, 5)[0, 1])

# PART 5
problem3_A_PT1 = 0.8
problem3_A_PT2 = 0.0
problem3_A_PT3 = 0.0
problem3_A_PT4 = 0.0
problem3_A_PT5 = 0.0
problem3_A_PT_inf = 0.2

problem3_B_PT1 = 0.0
problem3_B_PT2 = 0.0
problem3_B_PT3 = 0.5
problem3_B_PT4 = 0.0
problem3_B_PT5 = 0.25
problem3_B_PT_inf = 0.0

{
    "A_PB5": problem3_A_PB5,
    "B_PB5": problem3_B_PB5,
    "A_hitting": [problem3_A_PT1, problem3_A_PT2, problem3_A_PT3, problem3_A_PT4, problem3_A_PT5, problem3_A_PT_inf],
    "B_hitting": [problem3_B_PT1, problem3_B_PT2, problem3_B_PT3, problem3_B_PT4, problem3_B_PT5, problem3_B_PT_inf],
}
```
