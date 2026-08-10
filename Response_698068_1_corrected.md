# Exam 16th of June 2026, 8.00-13.00 for the course 1MS041 (Introduction to Data Science / Introduktion till dataanalys)

## Instructions:
1. Complete the problems by following instructions.
2. When done, submit this file with your solutions saved, following the instruction sheet.

This exam has 3 problems for a total of 40 points, to pass you need
20 points. The bonus will be added to the score of the exam and rounded afterwards.

## Some general hints and information:
* Try to answer all questions even if you are uncertain.
* Comment your code, so that if you get the wrong answer I can understand how you thought
this can give you some points even though the code does not run.
* Follow the instruction sheet rigorously.
* This exam is partially autograded, but your code and your free text answers are manually graded anonymously.
* If there are any questions, please ask the exam guards, they will escalate it to me if necessary.

## Tips for free text answers
* Be VERY clear with your reasoning, there should be zero ambiguity in what you are referring to.
* If you want to include math, you can write LaTeX in the Markdown cells, for instance `$f(x)=x^2$` will be rendered as $f(x)=x^2$ and `$$f(x) = x^2$$` will become an equation line, as follows
$$f(x) = x^2$$
Another example is `$$f_{Y \mid X}(y,x) = P(Y = y \mid X = x) = \exp(\alpha \cdot x + \beta)$$` which renders as
$$f_{Y \mid X}(y,x) = P(Y = y \mid X = x) = \exp(\alpha \cdot x + \beta)$$

## Finally some rules:
* You may not communicate with others during the exam, for example:
    * You cannot ask for help in Stack-Overflow or other such help forums during the Exam.
    * You may not communicate with AI's, for instance ChatGPT.
    * Your on-line and off-line activity is being monitored according to the examination rules.

## Good luck!



```python
# Insert your anonymous exam ID as a string in the variable below
examID="0010-MKZ"

```

---
## Exam vB, PROBLEM 1
Maximum Points = 14


This problem is about **PCA/SVD** for handwritten digit data. Unless stated otherwise, every vector or matrix you create should be a NumPy array.

The file `data/digits.csv` contains one row per image. The first 64 columns are pixel intensities for an 8 by 8 handwritten digit image, and the last column `target` is the digit label.

1. **[4p] Data and SVD.** Load the data. Store the feature matrix in `problem1_X` and the labels in `problem1_y`. Center the feature matrix column-wise and store it in `problem1_X_centered`. Compute the compact SVD
   $$X_c = U D V^T$$
   and store the matrices in `problem1_U`, `problem1_D`, and `problem1_V`, where `problem1_D` is a square diagonal matrix and `problem1_V` contains the right singular vectors as columns. If `problem1_X` has shape `(n_samples, 64)`, the compact shapes should be `problem1_U.shape == (n_samples, 64)`, `problem1_D.shape == (64, 64)`, and `problem1_V.shape == (64, 64)`. If you use `np.linalg.svd`, remember that compact SVD means `full_matrices=False`.

2. **[3p] Explained variance.** Compute the cumulative explained variance from the singular values on the diagonal of `problem1_D`, ordered from largest to smallest, and store it in `problem1_explained_variance`. If the singular values are $d_1 \geq d_2 \geq \cdots \geq d_{64}$, then the cumulative explained variance after the first $k$ components is
   $$
   \frac{\sum_{j=1}^k d_j^2}{\sum_{j=1}^{64} d_j^2}.
   $$
   Thus `problem1_explained_variance[k-1]` should contain this value. Store in `problem1_num_components` the smallest number of components needed to explain at least 90% of the variance.

3. **[3p] Two-dimensional PCA coordinates and interpretation.** Store the projection onto the first two principal directions in `problem1_scores_2d`. Plot these coordinates and color the points by the digit labels. In the markdown cell below, briefly explain what the plot suggests and why PCA can or cannot separate all digits perfectly.

4. **[4p] Nearest-centroid classification in PCA space.** Use the centered data and PCA directions already computed above; do not recompute PCA after the train/test split. Store the first `problem1_num_components` PCA coordinates in `problem1_scores_k`. Use a deterministic 80/20 split where the first 80% of the rows are training rows and the remaining 20% are test rows. For each digit label `0, 1, ..., 9`, compute the centroid of the training points in PCA space and store these ten centroids in `problem1_centroids`, with row `i` corresponding to digit `i`. Classify each test row by the nearest centroid in Euclidean distance and store the predicted labels in `problem1_test_predictions`, in increasing row-index order. Store the test accuracy in `problem1_test_accuracy`.



```python
#pip install pandas
```


```python
# Part 1: 4 points

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data. First 64 columns are pixel features, last column / target column is label.
problem1_df = pd.read_csv("data/digits.csv")

problem1_X = problem1_df.iloc[:, :64].to_numpy(dtype=float)

if "target" in problem1_df.columns:
    problem1_y = problem1_df["target"].to_numpy()
else:
    problem1_y = problem1_df.iloc[:, -1].to_numpy()

# Center column-wise
problem1_X_centered = problem1_X - np.mean(problem1_X, axis=0)

# Compact SVD: X_c = U D V^T
problem1_U, problem1_singular_values, problem1_Vt = np.linalg.svd(
    problem1_X_centered,
    full_matrices=False
)

problem1_D = np.diag(problem1_singular_values)
problem1_V = problem1_Vt.T

```


```python
# Part 2: 3 points

singular_values = np.diag(problem1_D)

problem1_explained_variance = np.cumsum(singular_values ** 2) / np.sum(singular_values ** 2)

# searchsorted returns a zero-based index, so add 1 to get number of components
problem1_num_components = int(np.searchsorted(problem1_explained_variance, 0.90) + 1)

```


```python
# Part 3: 3 points

problem1_scores_2d = problem1_X_centered @ problem1_V[:, :2]

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    problem1_scores_2d[:, 0],
    problem1_scores_2d[:, 1],
    c=problem1_y,
    cmap="viridis",
    s=15
)
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.title("Digits projected onto first two principal components")
plt.colorbar(scatter, label="Digit label")
plt.show()

```

## Free text answer for Part 3

The plot shows each handwritten digit image represented using only the first two principal components. Some digit classes form visible clusters, which means PCA captures useful structure in the data. However, the clusters overlap because two dimensions cannot preserve all information from the original 64 pixel features. PCA is unsupervised, so it chooses directions with maximum variance, not directions that best separate digit labels. Therefore PCA can help visualize and reduce dimension, but it cannot perfectly separate all digits in only two dimensions.



```python
# Part 4: 4 points

n_samples = problem1_X.shape[0]
problem1_split_index = int(0.8 * n_samples)

k = int(problem1_num_components)

# Project all rows onto the first k PCA directions
problem1_scores_k = problem1_X_centered @ problem1_V[:, :k]

# Deterministic 80/20 split
problem1_scores_train = problem1_scores_k[:problem1_split_index]
problem1_scores_test = problem1_scores_k[problem1_split_index:]

problem1_y_train = problem1_y[:problem1_split_index]
problem1_y_test = problem1_y[problem1_split_index:]

# Compute centroid for each digit 0,...,9 using training rows only
problem1_centroids = np.zeros((10, k))

for digit in range(10):
    problem1_centroids[digit] = problem1_scores_train[problem1_y_train == digit].mean(axis=0)

# Nearest-centroid classification
distances = np.linalg.norm(
    problem1_scores_test[:, None, :] - problem1_centroids[None, :, :],
    axis=2
)

problem1_test_predictions = np.argmin(distances, axis=1)

problem1_test_accuracy = float(np.mean(problem1_test_predictions == problem1_y_test))

```

---
#### Local Test for Exam vB, PROBLEM 1
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.


```python

# Optional local format checks for Problem 1. These checks do not prove correctness,
# but they are meant to catch the most common SVD shape issue.
import numpy as np

try:
    assert isinstance(problem1_X, np.ndarray)
    assert problem1_X.shape[1] == 64
    assert isinstance(problem1_y, np.ndarray)
    assert problem1_y.shape[0] == problem1_X.shape[0]

    n_samples, n_dimensions = problem1_X.shape
    expected_U_shape = (n_samples, n_dimensions)
    expected_D_shape = (n_dimensions, n_dimensions)
    expected_V_shape = (n_dimensions, n_dimensions)

    print("Expected compact SVD shapes:")
    print("  problem1_U:", expected_U_shape)
    print("  problem1_D:", expected_D_shape)
    print("  problem1_V:", expected_V_shape)
    print("Your shapes:")
    print("  problem1_U:", getattr(problem1_U, 'shape', None))
    print("  problem1_D:", getattr(problem1_D, 'shape', None))
    print("  problem1_V:", getattr(problem1_V, 'shape', None))

    if getattr(problem1_U, 'shape', None) == (n_samples, n_samples):
        print("Warning: problem1_U has the full SVD shape. NumPy uses full_matrices=True by default.")
        print("Use np.linalg.svd(problem1_X_centered, full_matrices=False) for the compact SVD.")

    assert problem1_U.shape == expected_U_shape
    assert problem1_D.shape == expected_D_shape
    assert problem1_V.shape == expected_V_shape
    assert np.allclose(problem1_X_centered, problem1_U @ problem1_D @ problem1_V.T, atol=5e-3)
    assert problem1_scores_2d.shape == (problem1_X.shape[0], 2)

    k = int(problem1_num_components)
    split_index = int(0.8 * n_samples)
    n_test = n_samples - split_index
    assert problem1_scores_k.shape == (n_samples, k)
    assert problem1_centroids.shape == (10, k)
    assert problem1_test_predictions.shape == (n_test,)
    assert 0 <= float(problem1_test_accuracy) <= 1
    print("Problem 1 format checks passed.")
except Exception as error:
    print("Problem 1 format check failed:", error)

```


```python

```

---
## Exam vB, PROBLEM 2
Maximum Points = 12


This problem is about **linear regression** and evaluating prediction error on held-out data. Unless stated otherwise, every vector or matrix you create should be a NumPy array.

The file `data/auto.csv` contains car measurements. We will predict fuel efficiency `mpg` from the features `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, and `model-year`, in that order.

1. **[2p] Load and clean data.** Load the file, remove rows where `horsepower` is missing, store the feature matrix in `problem2_X` and the target vector in `problem2_y`. Missing horsepower values are encoded as `?` or as an empty value.

2. **[2p] Train/test split and standardization.** Let `problem2_split_index = int(0.8 * n)`, where `n` is the number of cleaned rows. Use rows before this index as the training set and the remaining rows as the test set. Store the four arrays `problem2_X_train`, `problem2_X_test`, `problem2_y_train`, and `problem2_y_test`. Standardize features using the training mean and training standard deviation only, computed with NumPy's default `np.std(..., axis=0)`. Store the standardized train and test matrices in `problem2_X_train_standardized` and `problem2_X_test_standardized`.

3. **[3p] Fit linear regression.** Fit linear regression with an intercept using least squares on the standardized training features. Store the coefficient vector, including the intercept as the first entry, in `problem2_beta`. If you use `sklearn.linear_model.LinearRegression`, the intercept is stored in `model.intercept_` and the feature coefficients are stored in `model.coef_`, so the required order is `[model.intercept_, model.coef_[0], ..., model.coef_[5]]`. Store test predictions in `problem2_y_pred_test` and residuals `y_test - y_pred` in `problem2_residuals_test`.

4. **[3p] Test metrics and baseline.** Compute test MSE, MAE, and R^2 in `problem2_mse_test`, `problem2_mae_test`, and `problem2_r2_test`. Also compute `problem2_baseline_mse_test` for the predictor that always predicts the training-set mean of `mpg`, and set `problem2_model_beats_baseline` to `True` exactly when the linear model has smaller test MSE.

5. **[2p] Hoeffding interval.** Clip the absolute residuals at 50 and compute their empirical mean on the test set. Construct a two-sided Hoeffding confidence interval with confidence level 95% for the expected clipped absolute error. You may use `epsilon_bounded` from `Utils.py`, but you should decide its arguments from the sample size, the bound, and the confidence level. Since the clipped absolute error is between 0 and 50, intersect your final interval with `[0, 50]`. Store the interval endpoints in `problem2_lower_bound` and `problem2_upper_bound`.



```python
# Part 1: 2 points

import numpy as np
import pandas as pd

problem2_df = pd.read_csv("data/auto.csv")

# Missing horsepower is encoded as ? or empty; convert invalid values to NaN
problem2_df["horsepower"] = pd.to_numeric(problem2_df["horsepower"], errors="coerce")

# Remove rows where horsepower is missing
problem2_df = problem2_df.dropna(subset=["horsepower"])

# Features in the exact required order
problem2_features = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model-year"
]

problem2_X = problem2_df[problem2_features].to_numpy(dtype=float)

# Target is mpg
problem2_y = problem2_df["mpg"].to_numpy(dtype=float)

```


```python
# Part 2: 2 points

n = problem2_X.shape[0]
problem2_split_index = int(0.8 * n)

problem2_X_train = problem2_X[:problem2_split_index]
problem2_X_test = problem2_X[problem2_split_index:]

problem2_y_train = problem2_y[:problem2_split_index]
problem2_y_test = problem2_y[problem2_split_index:]

# Standardize using training mean and training std only
problem2_train_mean = np.mean(problem2_X_train, axis=0)
problem2_train_std = np.std(problem2_X_train, axis=0)

problem2_X_train_standardized = (problem2_X_train - problem2_train_mean) / problem2_train_std
problem2_X_test_standardized = (problem2_X_test - problem2_train_mean) / problem2_train_std

```


```python
# Part 3: 3 points

import numpy as np
from sklearn.linear_model import LinearRegression

problem2_model = LinearRegression()

# Fit on standardized training data
problem2_model.fit(problem2_X_train_standardized, problem2_y_train)

# Intercept first, then the six feature coefficients
problem2_beta = np.concatenate(([problem2_model.intercept_], problem2_model.coef_))

# Predict on standardized test data
problem2_y_pred_test = problem2_model.predict(problem2_X_test_standardized)

# Residuals = y_test - y_pred
problem2_residuals_test = problem2_y_test - problem2_y_pred_test

```


```python
# Part 4: 3 points

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

problem2_mse_test = float(mean_squared_error(problem2_y_test, problem2_y_pred_test))
problem2_mae_test = float(mean_absolute_error(problem2_y_test, problem2_y_pred_test))
problem2_r2_test = float(r2_score(problem2_y_test, problem2_y_pred_test))

# Baseline: always predict training-set mean of mpg
problem2_baseline_prediction = np.full(
    shape=problem2_y_test.shape,
    fill_value=np.mean(problem2_y_train),
    dtype=float
)

problem2_baseline_mse_test = float(mean_squared_error(problem2_y_test, problem2_baseline_prediction))

problem2_model_beats_baseline = bool(problem2_mse_test < problem2_baseline_mse_test)

```


```python
# Part 5: 2 points

# Clipped absolute residuals are bounded between 0 and 50
problem2_clipped_absolute_residuals = np.clip(np.abs(problem2_residuals_test), 0, 50)

problem2_empirical_mean_clipped_error = float(np.mean(problem2_clipped_absolute_residuals))

problem2_n_test = len(problem2_clipped_absolute_residuals)
problem2_alpha = 0.05
problem2_bound_range = 50

# Hoeffding two-sided epsilon:
# P(|mean - expected_mean| >= eps) <= alpha
problem2_epsilon = problem2_bound_range * np.sqrt(np.log(2 / problem2_alpha) / (2 * problem2_n_test))

problem2_lower_bound = float(max(0, problem2_empirical_mean_clipped_error - problem2_epsilon))
problem2_upper_bound = float(min(50, problem2_empirical_mean_clipped_error + problem2_epsilon))

```

---
#### Local Test for Exam vB, PROBLEM 2
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.


```python

# Optional local format checks for Problem 2. These checks do not prove correctness.
import numpy as np

try:
    assert isinstance(problem2_X, np.ndarray)
    assert isinstance(problem2_y, np.ndarray)
    assert problem2_X.shape[0] == problem2_y.shape[0]
    assert problem2_X.shape[1] == 6
    assert problem2_split_index == int(0.8 * problem2_X.shape[0])
    assert problem2_X_train.shape[0] == problem2_split_index
    assert problem2_X_test.shape[0] == problem2_X.shape[0] - problem2_split_index
    assert problem2_y_train.shape[0] == problem2_X_train.shape[0]
    assert problem2_y_test.shape[0] == problem2_X_test.shape[0]
    assert problem2_X_train_standardized.shape == problem2_X_train.shape
    assert problem2_X_test_standardized.shape == problem2_X_test.shape
    assert problem2_beta.shape == (7,)
    print("Problem 2 format checks passed.")
except Exception as error:
    print("Problem 2 format check failed:", error)

```


```python

```

---
## Exam vB, PROBLEM 3
Maximum Points = 14


This problem is about modelling warehouse package movement as a finite homogeneous Markov chain.

The file `data/warehouse_transitions.csv` contains observed transitions between five zones:

`Dock`, `Sort`, `Storage`, `Packing`, `Dispatch`.

Use this exact state order whenever you create vectors or matrices.

1. **[3p] Estimate transition matrix.** Load the transition data and estimate the transition matrix by maximum likelihood. Store it in `problem3_transition_matrix` as a 5 by 5 row-stochastic NumPy array, where entry `(i, j)` is the estimated probability of moving from state `i` to state `j`.

2. **[2p] Four-step probability.** Starting from `Dock`, compute the probability of being in `Dispatch` after exactly 4 steps and store it in `problem3_prob_dispatch_after_4_from_dock`.

3. **[2p] Simulation.** Starting from `Dock`, simulate 20000 chains for 8 steps using `np.random.default_rng(20260616)` and the transition probabilities in `problem3_transition_matrix`. Store the empirical distribution after 8 steps in `problem3_simulated_distribution_after_8` as a length-5 probability vector in the state order above.

4. **[2p] Chain structure.** Decide whether the estimated chain is irreducible and aperiodic. Store Boolean answers in `problem3_is_irreducible` and `problem3_is_aperiodic`.

5. **[2p] Stationary distribution.** Compute a stationary distribution for the estimated transition matrix and store it in `problem3_stationary_distribution` as a length-5 probability vector in the state order above. In the markdown cell below, briefly explain what the stationary distribution means in this warehouse context.

6. **[3p] Hitting time.** Compute the expected number of steps to hit `Dispatch` for the first time when starting from `Dock`. Store it in `problem3_expected_steps_to_dispatch_from_dock`. An exact computation gives full credit; a sufficiently accurate simulation estimate can receive partial credit.



```python
# Part 1: 3 points

import numpy as np
import pandas as pd

problem3_states = ["Dock", "Sort", "Storage", "Packing", "Dispatch"]
problem3_state_index = {state: i for i, state in enumerate(problem3_states)}

# Load transitions. Use header=None and then filter valid rows.
problem3_transitions = pd.read_csv("data/warehouse_transitions.csv", header=None)

# Use first two columns as from-state and to-state
problem3_transitions = problem3_transitions.iloc[:, :2]

# If the file accidentally includes a header row, this filter removes it.
problem3_transitions = problem3_transitions[
    problem3_transitions.iloc[:, 0].isin(problem3_states)
    & problem3_transitions.iloc[:, 1].isin(problem3_states)
]

problem3_counts = np.zeros((5, 5), dtype=float)

for from_state, to_state in problem3_transitions.to_numpy():
    i = problem3_state_index[from_state]
    j = problem3_state_index[to_state]
    problem3_counts[i, j] += 1

# Maximum likelihood estimate: row counts divided by row totals
problem3_transition_matrix = problem3_counts / problem3_counts.sum(axis=1, keepdims=True)

```


```python
# Scratch cell intentionally left empty.
# The official Part 2 answer is in the next cell.

```


```python
# Part 2: 2 points

P = problem3_transition_matrix

problem3_prob_dispatch_after_4_from_dock = float(
    np.linalg.matrix_power(P, 4)[
        problem3_state_index["Dock"],
        problem3_state_index["Dispatch"]
    ]
)

```


```python
# Scratch cell intentionally left empty.
# The official Part 3 answer is in the next cell.

```


```python
# Part 3: 2 points

P = problem3_transition_matrix

rng = np.random.default_rng(20260616)

problem3_n_chains = 20000
problem3_n_steps = 8
problem3_n_states = len(problem3_states)

problem3_current_states = np.full(
    problem3_n_chains,
    problem3_state_index["Dock"],
    dtype=int
)

for _ in range(problem3_n_steps):
    probabilities = P[problem3_current_states]
    cumulative_probabilities = np.cumsum(probabilities, axis=1)
    random_values = rng.random(problem3_n_chains)
    problem3_current_states = (random_values[:, None] <= cumulative_probabilities).argmax(axis=1)

problem3_counts_after_8 = np.bincount(problem3_current_states, minlength=problem3_n_states)

problem3_simulated_distribution_after_8 = problem3_counts_after_8 / problem3_n_chains

```


```python
# Scratch cell intentionally left empty.
# The official Part 4 answer is in the next cell.

```


```python
# Part 4: 2 points

import math

P = problem3_transition_matrix
n_states = P.shape[0]

# Irreducibility: every state can reach every other state
reachability = P > 0

# Floyd-Warshall style transitive closure
for k in range(n_states):
    reachability = reachability | (reachability[:, [k]] & reachability[[k], :])

problem3_is_irreducible = bool(reachability.all())

# Period of a state = gcd of return times t where P^t[i,i] > 0
def period_of_state(P, state, max_steps=100):
    power = np.eye(P.shape[0])
    period = 0

    for step in range(1, max_steps + 1):
        power = power @ P
        if power[state, state] > 1e-12:
            period = step if period == 0 else math.gcd(period, step)

    return period

problem3_periods = [period_of_state(P, i) for i in range(n_states)]

problem3_is_aperiodic = bool(all(period == 1 for period in problem3_periods))

```


```python
# Scratch cell intentionally left empty.

```


```python
# Scratch cell intentionally left empty.

```


```python
# Part 5: 2 points

P = problem3_transition_matrix
n_states = P.shape[0]

# Solve pi P = pi with sum(pi)=1
A = P.T - np.eye(n_states)
A[-1, :] = np.ones(n_states)

b = np.zeros(n_states)
b[-1] = 1

problem3_stationary_distribution = np.linalg.solve(A, b)

# Remove tiny numerical errors and normalize
problem3_stationary_distribution = np.maximum(problem3_stationary_distribution, 0)
problem3_stationary_distribution = problem3_stationary_distribution / problem3_stationary_distribution.sum()

```

## Free text answer for Part 5

The stationary distribution describes the long-run fraction of time that packages spend in each warehouse zone if the package movement process continues for a long time. For example, if the stationary probability of `Dispatch` is large, then in the long run many package states will be observed in the Dispatch zone. It does not describe one single package at one exact time; it describes the long-run average behavior of the Markov chain.



```python
# Scratch cell intentionally left empty.
# The official Part 6 answer is in the next cell.

```


```python
# Part 6: 3 points

P = problem3_transition_matrix

target = problem3_state_index["Dispatch"]
start = problem3_state_index["Dock"]

non_target_states = [i for i in range(P.shape[0]) if i != target]

# Q contains transitions among non-target states only
Q = P[np.ix_(non_target_states, non_target_states)]

# Expected hitting times solve: h = 1 + Qh, so (I - Q)h = 1
h = np.linalg.solve(
    np.eye(len(non_target_states)) - Q,
    np.ones(len(non_target_states))
)

problem3_expected_steps_to_dispatch_from_dock = float(h[non_target_states.index(start)])

```

---
#### Local Test for Exam vB, PROBLEM 3
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.


```python

# Optional local format checks for Problem 3. These checks do not prove correctness.
import numpy as np

try:
    assert problem3_transition_matrix.shape == (5, 5)
    assert np.allclose(np.sum(problem3_transition_matrix, axis=1), 1, atol=2e-4)
    assert problem3_simulated_distribution_after_8.shape == (5,)
    assert np.all(problem3_simulated_distribution_after_8 >= -1e-12)
    assert abs(np.sum(problem3_simulated_distribution_after_8) - 1) < 2e-4
    assert problem3_stationary_distribution.shape == (5,)
    assert np.all(problem3_stationary_distribution >= -1e-12)
    assert abs(np.sum(problem3_stationary_distribution) - 1) < 2e-4
    print("Problem 3 format checks passed.")
except Exception as error:
    print("Problem 3 format check failed:", error)

```


```python

```

The number of points you have scored in total for this entire set of Problems is 4.250000000000003 out of 40.
