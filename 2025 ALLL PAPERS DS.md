```python
#2025 QUESTION 1 SVD ONE January 2025
PROBLEM 1: SVD + anomaly detection
Part 1 — Question

Load data/SVD.csv, do SVD, store:

problem1_data, problem1_U, problem1_D, problem1_V, first right singular vector, first left singular vector.

Part 1 — Code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
try:
    problem1_data = pd.read_csv("data/SVD.csv", header=None).to_numpy(dtype=float)
except ValueError:
    problem1_data = pd.read_csv("data/SVD.csv").to_numpy(dtype=float)

# SVD
# np.linalg.svd gives X = U @ diag(D) @ Vt
problem1_U, problem1_D, Vt = np.linalg.svd(problem1_data, full_matrices=False)

# They want V, not V transpose
problem1_V = Vt.T

# First singular vectors
problem1_first_right_singular_vector = problem1_V[:, 0].flatten()
problem1_first_left_singular_vector = problem1_U[:, 0].flatten()

print(problem1_data.shape)
print(problem1_U.shape)
print(problem1_D.shape)
print(problem1_V.shape)

```


```python
Part 2 — Question

Find explained variance using 1, 2, 3, ... singular values. Then find smallest number of singular values needed to explain at least 95%.

Part 2 — Code
# Singular values squared give the variance/energy
singular_value_energy = problem1_D ** 2

# Cumulative explained variance
problem1_explained_variance = np.cumsum(singular_value_energy) / np.sum(singular_value_energy)

# Smallest number of components that gives at least 95%
problem1_num_components = int(np.argmax(problem1_explained_variance >= 0.95) + 1)

print(problem1_explained_variance)
print(problem1_num_components)

Baby meaning:

We keep adding singular values until we reach 95% of the total information.
```


```python
Part 3 — Question

Use the number from Part 2 and reconstruct the best approximation of the original matrix.

Part 3 — Code
k = problem1_num_components

problem1_approximation = (
    problem1_U[:, :k]
    @ np.diag(problem1_D[:k])
    @ problem1_V[:, :k].T
)

print(problem1_approximation.shape)

#Free text answer

The approximating matrix is a lower-rank reconstruction of the original data.
It keeps the most important patterns/directions from the data and removes weaker patterns/noise.
```


```python
Part 4 — Question

Calculate row-wise Euclidean reconstruction error. Plot ECDF. Choose a threshold so exactly 10 samples are above it. Store those 10 rows as outliers.

Part 4 — Code
# Row-wise Euclidean distance between original and approximation
problem1_error = np.linalg.norm(problem1_data - problem1_approximation, axis=1)

# Plot empirical distribution function
sorted_errors = np.sort(problem1_error)
ecdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

plt.figure(figsize=(7, 4))
plt.step(sorted_errors, ecdf, where="post")
plt.xlabel("Reconstruction error")
plt.ylabel("Empirical distribution function")
plt.title("ECDF of reconstruction errors")
plt.grid(True)
plt.show()

# Choose threshold so exactly 10 are above it
# Threshold between 10th largest and 11th largest error
problem1_threshold = (sorted_errors[-10] + sorted_errors[-11]) / 2

# Store outliers
problem1_outliers = problem1_data[problem1_error > problem1_threshold]

print(problem1_threshold)
print(problem1_outliers.shape)

Baby meaning:

Big reconstruction error = the row does not fit the normal pattern.
So those are anomalies/outliers.
```


```python
#2025 QUESTION 1 SVD ONE AUGUST2025
#Part 1 code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
problem1_data = pd.read_csv("data/SVD.csv", header=None).to_numpy()

# SVD: numpy gives X = U @ diag(D) @ Vt
problem1_U, problem1_D, Vt = np.linalg.svd(problem1_data, full_matrices=False)

# The question wants V, where X = U D V^T
problem1_V = Vt.T

# First right singular vector = first column of V
problem1_first_right_singular_vector = problem1_V[:, 0].flatten()

# First left singular vector = first column of U
problem1_first_left_singular_vector = problem1_U[:, 0].flatten()
```


```python
#Part 2 code — 90% variance
# Explained variance from singular values
problem1_explained_variance = np.cumsum(problem1_D ** 2) / np.sum(problem1_D ** 2)

# Smallest number of components to explain at least 90%
problem1_num_components = int(np.argmax(problem1_explained_variance >= 0.90) + 1)
```


```python
#Part 3 code
k = problem1_num_components

problem1_approximation = (
    problem1_U[:, :k]
    @ np.diag(problem1_D[:k])
    @ problem1_V[:, :k].T
)
#Part 3 free text answer

Each row in the approximation represents the original row after projecting it onto the most important singular-vector directions.
So it is a compressed version of the original data that keeps the main structure/pattern and removes weaker information or noise.
```


```python
#Part 4 code
# Row-wise Euclidean reconstruction error
problem1_reconstruction_error = np.linalg.norm(
    problem1_data - problem1_approximation,
    axis=1
)

# Plot empirical distribution function
sorted_errors = np.sort(problem1_reconstruction_error)
ecdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

plt.figure(figsize=(7, 4))
plt.step(sorted_errors, ecdf, where="post")
plt.xlabel("Reconstruction error")
plt.ylabel("Empirical distribution")
plt.title("ECDF of reconstruction error")
plt.grid(True)
plt.show()

# Choose threshold so exactly 10 samples are above it
problem1_threshold = (sorted_errors[-10] + sorted_errors[-11]) / 2

# Store the 10 outliers
problem1_outliers = problem1_data[
    problem1_reconstruction_error > problem1_threshold
]

#Baby summary: yes same SVD, just change 95% → 90% and use the exact variable name problem1_reconstruction_error.
```


```python
#2025 QUESTION 2 MARKOV CHAIN ONE January 2025
#PART 1
import numpy as np
import pandas as pd

# Load the data
problem2_data = pd.read_csv("data/websites.csv", header=None).to_numpy()

# Columns:
# column 0 = user
# column 1 = source page
# column 2 = destination page
sources = problem2_data[:, 1].astype(int)
destinations = problem2_data[:, 2].astype(int)

# Number of states/pages
# Since pages are 0, 1, 2, ..., max page
problem2_n_states = int(max(sources.max(), destinations.max()) + 1)

# Count matrix
transition_counts = np.zeros((problem2_n_states, problem2_n_states))

for source, destination in zip(sources, destinations):
    transition_counts[source, destination] += 1

# Convert counts to probabilities
row_sums = transition_counts.sum(axis=1, keepdims=True)

problem2_transition_matrix = transition_counts / row_sums

print(problem2_transition_matrix)
print(problem2_transition_matrix.shape)
print(problem2_n_states)

```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Cell In[1], line 7
          4 import pandas as pd
          6 # Load the data
    ----> 7 problem2_data = pd.read_csv("data/websites.csv", header=None).to_numpy()
          9 # Columns:
         10 # column 0 = user
         11 # column 1 = source page
         12 # column 2 = destination page
         13 sources = problem2_data[:, 1].astype(int)
    

    File /lib/python3.13/site-packages/pandas/io/parsers/readers.py:1026, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
       1013 kwds_defaults = _refine_defaults_read(
       1014     dialect,
       1015     delimiter,
       (...)   1022     dtype_backend=dtype_backend,
       1023 )
       1024 kwds.update(kwds_defaults)
    -> 1026 return _read(filepath_or_buffer, kwds)
    

    File /lib/python3.13/site-packages/pandas/io/parsers/readers.py:620, in _read(filepath_or_buffer, kwds)
        617 _validate_names(kwds.get("names", None))
        619 # Create the parser.
    --> 620 parser = TextFileReader(filepath_or_buffer, **kwds)
        622 if chunksize or iterator:
        623     return parser
    

    File /lib/python3.13/site-packages/pandas/io/parsers/readers.py:1620, in TextFileReader.__init__(self, f, engine, **kwds)
       1617     self.options["has_index_names"] = kwds["has_index_names"]
       1619 self.handles: IOHandles | None = None
    -> 1620 self._engine = self._make_engine(f, self.engine)
    

    File /lib/python3.13/site-packages/pandas/io/parsers/readers.py:1880, in TextFileReader._make_engine(self, f, engine)
       1878     if "b" not in mode:
       1879         mode += "b"
    -> 1880 self.handles = get_handle(
       1881     f,
       1882     mode,
       1883     encoding=self.options.get("encoding", None),
       1884     compression=self.options.get("compression", None),
       1885     memory_map=self.options.get("memory_map", False),
       1886     is_text=is_text,
       1887     errors=self.options.get("encoding_errors", "strict"),
       1888     storage_options=self.options.get("storage_options", None),
       1889 )
       1890 assert self.handles is not None
       1891 f = self.handles.handle
    

    File /lib/python3.13/site-packages/pandas/io/common.py:873, in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        868 elif isinstance(handle, str):
        869     # Check whether the filename is to be opened in binary mode.
        870     # Binary mode does not support 'encoding' and 'newline'.
        871     if ioargs.encoding and "b" not in ioargs.mode:
        872         # Encoding
    --> 873         handle = open(
        874             handle,
        875             ioargs.mode,
        876             encoding=ioargs.encoding,
        877             errors=errors,
        878             newline="",
        879         )
        880     else:
        881         # Binary mode
        882         handle = open(handle, ioargs.mode)
    

    FileNotFoundError: [Errno 44] No such file or directory: 'data/websites.csv'



```python
#Part 2 — Code
# Random generator
rng = np.random.default_rng(1)

n_simulations = 10000
start_page = 1

P = problem2_transition_matrix

# -----------------------------
# Case 1: preload ONLY the most likely page
# -----------------------------

# Find the most likely next page from page 1
top_page = np.argmax(P[start_page])

# Simulate actual next pages for 10,000 users starting from page 1
next_pages_top = rng.choice(
    np.arange(problem2_n_states),
    size=n_simulations,
    p=P[start_page]
)

# Start with normal load times: Exp(1), mean 1
problem2_page_load_times_top = rng.exponential(
    scale=1.0,
    size=n_simulations
)

# If actual next page equals preloaded page, load faster: Exp(10), mean 0.1
correct_top = next_pages_top == top_page

problem2_page_load_times_top[correct_top] = rng.exponential(
    scale=0.1,
    size=correct_top.sum()
)


# -----------------------------
# Case 2: preload TWO most likely pages
# -----------------------------

# Find two most likely next pages from page 1
top_two_pages = np.argsort(P[start_page])[-2:]

# Simulate actual next pages again
next_pages_two = rng.choice(
    np.arange(problem2_n_states),
    size=n_simulations,
    p=P[start_page]
)

# Start with normal load times
problem2_page_load_times_two = rng.exponential(
    scale=1.0,
    size=n_simulations
)

# If actual next page is one of the two preloaded pages, load faster
correct_two = np.isin(next_pages_two, top_two_pages)

problem2_page_load_times_two[correct_two] = rng.exponential(
    scale=0.1,
    size=correct_two.sum()
)

print(problem2_page_load_times_top.shape)
print(problem2_page_load_times_two.shape)

print("Average top preload:", problem2_page_load_times_top.mean())
print("Average two preload:", problem2_page_load_times_two.mean())
```


```python
#Part 3 — Code
# Without preloading, load time is Exp(1)
# Expected value of Exp(1) is 1
problem2_avg = 1.0

# Compare no preload average with empirical preload average
problem2_comparison = problem2_avg > np.mean(problem2_page_load_times_top)

print(problem2_avg)
print(problem2_comparison)

#Part 3 free text answer

The expected load time without preloading is 1 second because the page load time follows Exp(1).
I compared this value with the simulated average load time after preloading the most likely next page.
If problem2_comparison is True, then preloading improves the load time because the average load time becomes smaller.
```


```python
## Part 4: Stationary distribution + average load time

# We want a stationary distribution pi such that:
# pi @ P = pi
# This is same as:
# P.T @ pi = pi

P = problem2_transition_matrix

# Find eigenvalues and eigenvectors of P transpose
eigenvalues, eigenvectors = np.linalg.eig(P.T)

# Find the eigenvalue closest to 1
idx = np.argmin(np.abs(eigenvalues - 1))

# Take the eigenvector for eigenvalue 1
stationary = np.real(eigenvectors[:, idx])

# Normalize so all probabilities add up to 1
stationary = stationary/stationary.sum()

# Remove tiny numerical negative errors
stationary = np.maximum(stationary, 0)
stationary = stationary/stationary.sum()

# Store stationary distribution
problem2_stationary_distribution = stationary


# Now calculate expected load time with preloading
# From each page, we preload the most likely next page
# Probability of correct preload from each page = max probability in that row
max_probs = np.max(P, axis=1)

# If preload correct: mean load time = 0.1
# If preload wrong: mean load time = 1
expected_load_each_page = (max_probs * 0.1) + ((1 - max_probs) * 1.0)

# Average load time weighted by stationary distribution
problem2_avg_stationary = np.sum(
    problem2_stationary_distribution * expected_load_each_page
)

print("Stationary distribution:")
print(problem2_stationary_distribution)

print("Shape:")
print(problem2_stationary_distribution.shape)

print("Sum:")
print(problem2_stationary_distribution.sum())

print("Average load time:")
print(problem2_avg_stationary)
```


```python
#2025 QUESTION 2 MARKOV CHAIN ONE JUNE 2025
#PART 1
import numpy as np
import pandas as pd

# Load the data
problem2_data = pd.read_csv("data/websites.csv", header=None).to_numpy()

# Columns:
# column 0 = user
# column 1 = source page
# column 2 = destination page
sources = problem2_data[:, 1].astype(int)
destinations = problem2_data[:, 2].astype(int)

# Number of states/pages
# Since pages are 0, 1, 2, ..., max page
problem2_n_states = int(max(sources.max(), destinations.max()) + 1)

# Count matrix
transition_counts = np.zeros((problem2_n_states, problem2_n_states))

for source, destination in zip(sources, destinations):
    transition_counts[source, destination] += 1

# Convert counts to probabilities
row_sums = transition_counts.sum(axis=1, keepdims=True)

problem2_transition_matrix = transition_counts / row_sums

print(problem2_transition_matrix)
print(problem2_transition_matrix.shape)
print(problem2_n_states)

```


```python
#Part 2 corrected code
rng = np.random.default_rng(1)

n_simulations = 10000
start_page = 8

P = problem2_transition_matrix

# Most likely page from page 8
top_page = np.argmax(P[start_page])

next_pages_top = rng.choice(
    np.arange(problem2_n_states),
    size=n_simulations,
    p=P[start_page]
)

# Not preloaded: Exp(3), mean = 1/3
problem2_page_load_times_top = rng.exponential(
    scale=1/3,
    size=n_simulations
)

# Preloaded correctly: Exp(20), mean = 1/20
correct_top = next_pages_top == top_page

problem2_page_load_times_top[correct_top] = rng.exponential(
    scale=1/20,
    size=correct_top.sum()
)


# Two most likely pages
top_two_pages = np.argsort(P[start_page])[-2:]

next_pages_two = rng.choice(
    np.arange(problem2_n_states),
    size=n_simulations,
    p=P[start_page]
)

problem2_page_load_times_two = rng.exponential(
    scale=1/3,
    size=n_simulations
)

correct_two = np.isin(next_pages_two, top_two_pages)

problem2_page_load_times_two[correct_two] = rng.exponential(
    scale=1/20,
    size=correct_two.sum()
)
```


```python
#Part 3 corrected code
# No preloading means Exp(3), mean = 1/3
problem2_avg = float(1/3)

# Must be Python bool, not numpy bool
problem2_comparison = bool(problem2_avg > np.mean(problem2_page_load_times_top))
```


```python
#Part 4 corrected code
P = problem2_transition_matrix

# Stationary distribution using eigenvector of P.T with eigenvalue 1
eigenvalues, eigenvectors = np.linalg.eig(P.T)

idx = np.argmin(np.abs(eigenvalues - 1))

stationary = np.real(eigenvectors[:, idx])

stationary = stationary / stationary.sum()

stationary = np.maximum(stationary, 0)
stationary = stationary / stationary.sum()

problem2_stationary_distribution = stationary


# For each page, preload the most likely next page
max_probs = np.max(P, axis=1)

# If correct preload: mean = 1/20
# If wrong preload: mean = 1/3
expected_load_each_page = (max_probs * (1/20)) + ((1 - max_probs) * (1/3))

problem2_avg_stationary = float(
    np.sum(problem2_stationary_distribution * expected_load_each_page)
)
```


```python
#2025 QUESTION 2 LINEAR REGRESSION ONE AUGUST 2025

#PART 1import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load salary data
problem2_df = pd.read_csv("data/salaries.csv")

# Features = columns used to predict salary
problem2_features = [
    "work_year",
    "experience_level",
    "employment_type",
    "remote_ratio"
]

# Target = column we want to predict
problem2_target = "salary_in_usd"

```


```python
#PART 2

from sklearn.model_selection import train_test_split

# X = feature columns
problem2_X = problem2_df[problem2_features]

# Convert categorical/text columns into numeric dummy variables
problem2_X = pd.get_dummies(problem2_X, drop_first=True)

# y = target column
problem2_y = problem2_df[problem2_target]

# Split into train and test
problem2_X_train, problem2_X_test, problem2_y_train, problem2_y_test = train_test_split(
    problem2_X,
    problem2_y,
    train_size=0.8,
    random_state=42
)
```


```python
#PART 3
from sklearn.linear_model import LinearRegression

# Create linear regression model
problem2_model = LinearRegression()

# Train the model
problem2_model.fit(problem2_X_train, problem2_y_train)
```


```python
#Part 4 EDF plot code

This plots residuals:

residual = true - predicted
# Residuals
problem2_residuals = problem2_y_true - problem2_predictions

# Try using course helper function first
try:
    try:
        from utils import plotEDF
    except ImportError:
        from Utils import plotEDF

    plotEDF.makeEDF_combo(problem2_residuals, alpha=0.01)

except Exception:
    # Manual EDF with 99% DKW confidence bands
    residuals_sorted = np.sort(problem2_residuals)
    n = len(residuals_sorted)

    edf = np.arange(1, n + 1) / n

    alpha = 0.01
    epsilon = np.sqrt(np.log(2 / alpha) / (2 * n))

    lower_band = np.maximum(edf - epsilon, 0)
    upper_band = np.minimum(edf + epsilon, 1)

    plt.figure(figsize=(7, 4))
    plt.step(residuals_sorted, edf, where="post", label="EDF")
    plt.step(residuals_sorted, lower_band, where="post", linestyle="--", label="Lower 99% band")
    plt.step(residuals_sorted, upper_band, where="post", linestyle="--", label="Upper 99% band")
    plt.xlabel("Residual: true - predicted")
    plt.ylabel("Empirical distribution")
    plt.title("EDF of residuals with 99% DKW confidence bands")
    plt.legend()
    plt.grid(True)
    plt.show()
```


```python
# PART 5
plt.figure(figsize=(7, 5))

plt.scatter(problem2_predictions, problem2_y_true, alpha=0.7)

# Add perfect prediction line y = x
min_value = min(problem2_predictions.min(), problem2_y_true.min())
max_value = max(problem2_predictions.max(), problem2_y_true.max())

plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

plt.xlabel("Predicted salary")
plt.ylabel("True salary")
plt.title("Predicted salary vs true salary")
plt.grid(True)
plt.show()
```


```python
#Part 6 — Discussion answer
Discussion on the value of MARE

Write this:

The MARE tells us the average relative prediction error. For example, if the MARE is 0.25, then the model is wrong by about 25% on average. A smaller MARE means a better model. If the MARE is large, then the linear regression model is not predicting salaries very accurately.

Discussion on predicted vs true scatterplot

Write this:

In the scatter plot, a good model would have points close to the diagonal line where predicted salary equals true salary. If the points are spread far away from the diagonal line, then the model has large prediction errors. If the points form a rough upward trend, then the model captures some relationship, but it may still not be very accurate.
```


```python
#2025 QUESTION 3 CONFUSION MATRIX ONE January 2025
#Part 1 code
import numpy as np
import matplotlib.pyplot as plt

def cost(y_true, y_predict_proba, threshold):
    # Convert probabilities into 0/1 predictions
    y_pred = (y_predict_proba >= threshold).astype(int)

    # True Positive: real fraud, predicted fraud
    TP = np.sum((y_true == 1) & (y_pred == 1))

    # True Negative: real not fraud, predicted not fraud
    TN = np.sum((y_true == 0) & (y_pred == 0))

    # False Positive: real not fraud, predicted fraud
    FP = np.sum((y_true == 0) & (y_pred == 1))

    # False Negative: real fraud, predicted not fraud
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Total cost
    total_cost = (100 * TP) + (0 * TN) + (120 * FP) + (600 * FN)

    # Average cost per sample
    average_cost = total_cost / len(y_true)

    return float(average_cost)
#plot

# Thresholds from 0 to 1
problem3_threshold_grid = np.arange(0, 1.01, 0.01)

# Cost for each threshold
problem3_costs = np.array([
    cost(PROBLEM3_y_true_val, PROBLEM3_y_pred_proba_val, threshold)
    for threshold in problem3_threshold_grid
])

# Plot threshold vs cost
plt.figure(figsize=(7, 4))
plt.plot(problem3_threshold_grid, problem3_costs)
plt.xlabel("Threshold")
plt.ylabel("Average cost")
plt.title("Cost as a function of threshold")
plt.grid(True)
plt.show()

```


```python
#Part 2 — Find threshold that minimizes cost

from sklearn.metrics import precision_score, recall_score

# Find index of minimum cost
best_index = np.argmin(problem3_costs)

# Best threshold
problem3_threshold = float(problem3_threshold_grid[best_index])

# Cost at that threshold
problem3_cost_val = float(
    cost(PROBLEM3_y_true_val, PROBLEM3_y_pred_proba_val, problem3_threshold)
)

# Predicted labels using best cost threshold
problem3_y_pred_val = (PROBLEM3_y_pred_proba_val >= problem3_threshold).astype(int)

# Precision and recall for class 1 = fraud
problem3_precision_1 = float(
    precision_score(PROBLEM3_y_true_val, problem3_y_pred_val, pos_label=1, zero_division=0)
)

problem3_recall_1 = float(
    recall_score(PROBLEM3_y_true_val, problem3_y_pred_val, pos_label=1, zero_division=0)
)

# Precision and recall for class 0 = not fraud
problem3_precision_0 = float(
    precision_score(PROBLEM3_y_true_val, problem3_y_pred_val, pos_label=0, zero_division=0)
)

problem3_recall_0 = float(
    recall_score(PROBLEM3_y_true_val, problem3_y_pred_val, pos_label=0, zero_division=0)
)

print("Best cost threshold:", problem3_threshold)
print("Validation cost:", problem3_cost_val)
print("Precision class 1:", problem3_precision_1)
print("Recall class 1:", problem3_recall_1)
print("Precision class 0:", problem3_precision_0)
print("Recall class 0:", problem3_recall_0)
```


```python
#Part 3 — Find best threshold for 0-1 loss

# Calculate 0-1 loss for each threshold
problem3_zero_one_losses = np.array([
    np.mean((PROBLEM3_y_pred_proba_val >= threshold).astype(int) != PROBLEM3_y_true_val)
    for threshold in problem3_threshold_grid
])

# Best threshold for 0-1 loss
best_01_index = np.argmin(problem3_zero_one_losses)

problem3_threshold_01 = float(problem3_threshold_grid[best_01_index])

# Cost using the 0-1 optimal threshold
cost_at_01_threshold = cost(
    PROBLEM3_y_true_val,
    PROBLEM3_y_pred_proba_val,
    problem3_threshold_01
)

# Difference in cost:
# cost using 0-1 threshold minus cost using cost-optimal threshold
problem3_cost_difference = float(cost_at_01_threshold - problem3_cost_val)

print("0-1 loss threshold:", problem3_threshold_01)
print("Cost difference:", problem3_cost_difference)

##Baby meaning:Part 2 finds threshold that minimizes money cost.Part 3 finds threshold that minimizes number of wrong predictions.They can be different because fraud mistakes are not equally expensive.
```


```python
#Part 4 — Hoeffding confidence interval

# Use the cost-optimal threshold from Part 2 on the test data
problem3_y_pred_test = (PROBLEM3_y_pred_proba_test >= problem3_threshold).astype(int)

# Calculate cost per test sample
problem3_test_costs = np.zeros(len(PROBLEM3_y_true_test))

# TP cost = 100
problem3_test_costs[
    (PROBLEM3_y_true_test == 1) & (problem3_y_pred_test == 1)
] = 100

# TN cost = 0
problem3_test_costs[
    (PROBLEM3_y_true_test == 0) & (problem3_y_pred_test == 0)
] = 0

# FP cost = 120
problem3_test_costs[
    (PROBLEM3_y_true_test == 0) & (problem3_y_pred_test == 1)
] = 120

# FN cost = 600
problem3_test_costs[
    (PROBLEM3_y_true_test == 1) & (problem3_y_pred_test == 0)
] = 600

# Average test cost
problem3_test_cost_mean = np.mean(problem3_test_costs)

# Hoeffding 95% confidence interval
alpha = 0.05
n = len(problem3_test_costs)

# Costs are bounded between 0 and 600
a = 0
b = 600

epsilon = (b - a) * np.sqrt(np.log(2 / alpha) / (2 * n))

problem3_lower_bound = float(problem3_test_cost_mean - epsilon)
problem3_upper_bound = float(problem3_test_cost_mean + epsilon)

print("Test cost mean:", problem3_test_cost_mean)
print("Lower bound:", problem3_lower_bound)
print("Upper bound:", problem3_upper_bound)




#Part 4 free text explanation
#Write this:
#I used the threshold selected on the validation data and applied it to the test data. For each test sample, I computed the cost according to whether the prediction was TP, TN, FP, or FN. I then used Hoeffding’s inequality with 95% confidence. The assumptions are that the test samples are independent, the threshold was fixed before evaluating the test set, and the cost for each sample is bounded between 0 and 600.
```
