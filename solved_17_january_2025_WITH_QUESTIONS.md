# Solved Exam Notebook — 17th January 2025

Course: **1MS041 Introduction to Data Science / Introduktion till dataanalys**  
Exam: **17th of January 2025, 8.00–13.00**

This notebook now includes the **original-style questions from the PDF** followed by solved code answers and free-text explanations.

Keep these files beside this notebook:

- `data/SVD.csv`
- `data/websites.csv`
- `data/fraud.csv`
- `Utils.py` if provided by the course

Run cells from top to bottom.



```python
# Insert your anonymous exam ID as a string in the variable below
examID = "XXX"

# Common imports used throughout the notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```

---

# Exam vB, Problem 1 — SVD and Anomaly Detection

**Maximum points: 14**

**Question statement:**

This problem is about SVD and anomaly detection. In all the problems where you are asked to produce a matrix or vector, they should be **numpy arrays**.


## Problem 1, Part 1 — Question [4p]

Load the file `data/SVD.csv` as instructed in the code cell. Compute the Singular Value Decomposition, i.e. construct the three matrices `U`, `D`, `V` such that if `X` is the data matrix of shape

`n_samples x n_dimensions`

then

\[
X = UDV^T.
\]

Put the resulting matrices in their variables, check that the shapes align with the instructions in the code cell. Finally, extract the first right and left singular vectors and store those as 1-dimensional arrays in the instructed variables.

**Required variables:**

- `problem1_data`
- `problem1_U`
- `problem1_D`
- `problem1_V`
- `problem1_first_right_singular_vector`
- `problem1_first_left_singular_vector`



```python
# Part 1: Load data and compute SVD

# Load the CSV as a numpy array.
# header=None is used because exam data files often do not contain column names.
problem1_data = pd.read_csv("data/SVD.csv", header=None).values.astype(float)

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

## Problem 1, Part 2 — Question [3p]

The first goal is to calculate the explained variance. Calculate the explained variance of using 1, 2, ..., number of singular vectors and select how many singular vectors are needed in order to explain at least **95%** of the variance.

**Required variables:**

- `problem1_explained_variance`
- `problem1_num_components`

For SVD, the explained variance is calculated from squared singular values:

\[
\text{explained variance}(k)=\frac{\sum_{i=1}^{k} d_i^2}{\sum_{i=1}^{d} d_i^2}.
\]



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

## Problem 1, Part 3 — Question [3p]

With the number of components chosen in Part 2, construct the best approximating matrix with the rank as the number of components. Explain what each row represents in the approximating matrix in terms of the original data.

**Required variable:**

- `problem1_approximation`

The best rank-\(k\) approximation is:

\[
X_k = U_kD_kV_k^T.
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


## Problem 1, Part 4 — Question [4p]

Create a vector which corresponds to the row-wise Euclidean distance between the original matrix `problem1_data` and the approximating matrix `problem1_approximation` and plot the empirical distribution function of that distance.

Based on the empirical distribution function, choose a threshold such that **10 samples are above it** and the rest are below. Store the 10 samples in the instructed variable.

**Required variables:**

- `problem1_reconstruction_error`
- `problem1_threshold`
- `problem1_outliers`

The reconstruction error is:

\[
e_i = \lVert X_i - \hat{X}_i \rVert_2.
\]



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

# Exam vB, Problem 2 — Markov Chain Website Preloading

**Maximum points: 14**

**Question statement:**

In this problem we have data consisting of user behavior on a website. The pages of the website are just numbers in the dataset `0, 1, 2, ...` and each row consists of a user, a source and a destination page. This signifies that the user was on the source page and clicked a link leading them to the destination page.

The goal is to improve the user experience by decreasing load time of the next page visited, so we need a good estimate for the next site likely to be visited. We model this using a homogeneous Markov chain. Each row in the data-file corresponds to a single realization of a transition.


## Problem 2, Part 1 — Question [3p]

Load the data in the file `data/websites.csv` and construct a matrix of size

`n_pages x n_pages`

which is the maximum likelihood estimate of the true transition matrix for the Markov chain.

The ordering of the states is exactly the one in the data-file. That is, page `0` has index `0` in the matrix.

**Required variables:**

- `problem2_transition_matrix`
- `problem2_n_states`

The maximum likelihood estimate is:

\[
\hat{P}_{ij}=\frac{\#\{\text{transitions from page }i\text{ to page }j\}}{\#\{\text{transitions from page }i\}}.
\]



```python
# Part 1: Estimate transition matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
websites_df = pd.read_csv("websites.csv", header=None)

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

          0       1            2
    0  user  source  destination
    1     0       0            3
    2     0       3            6
    3     0       6            9
    4     0       9            2
    Shape: (1000, 3)
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[2], line 12
          8 print("Shape:", websites_df.shape)
         10 # The problem says each row is: user, source, destination.
         11 # Therefore source is column 1 and destination is column 2.
    ---> 12 source = websites_df.iloc[:, 1].astype(int).values
         13 destination = websites_df.iloc[:, 2].astype(int).values
         15 problem2_n_states = int(max(source.max(), destination.max()) + 1)
    

    File c:\Users\hp\miniconda3\New folder\Lib\site-packages\pandas\core\generic.py:6541, in NDFrame.astype(self, dtype, copy, errors)
       6537     results = [ser.astype(dtype, errors=errors) for _, ser in self.items()]
       6539 else:
       6540     # else, only a single dtype is given
    -> 6541     new_data = self._mgr.astype(dtype=dtype, errors=errors)
       6542     res = self._constructor_from_mgr(new_data, axes=new_data.axes)
       6543     return res.__finalize__(self, method="astype")
    

    File c:\Users\hp\miniconda3\New folder\Lib\site-packages\pandas\core\internals\managers.py:614, in BaseBlockManager.astype(self, dtype, errors)
        613 def astype(self, dtype, errors: str = "raise") -> Self:
    --> 614     return self.apply("astype", dtype=dtype, errors=errors)
    

    File c:\Users\hp\miniconda3\New folder\Lib\site-packages\pandas\core\internals\managers.py:445, in BaseBlockManager.apply(self, f, align_keys, **kwargs)
        443         applied = b.apply(f, **kwargs)
        444     else:
    --> 445         applied = getattr(b, f)(**kwargs)
        446     result_blocks = extend_blocks(applied, result_blocks)
        448 out = type(self).from_blocks(result_blocks, [ax.view() for ax in self.axes])
    

    File c:\Users\hp\miniconda3\New folder\Lib\site-packages\pandas\core\internals\blocks.py:607, in Block.astype(self, dtype, errors, squeeze)
        604         raise ValueError("Can not squeeze with more than one column.")
        605     values = values[0, :]  # type: ignore[call-overload]
    --> 607 new_values = astype_array_safe(values, dtype, errors=errors)
        609 new_values = maybe_coerce_values(new_values)
        611 refs = None
    

    File c:\Users\hp\miniconda3\New folder\Lib\site-packages\pandas\core\dtypes\astype.py:240, in astype_array_safe(values, dtype, copy, errors)
        237     dtype = dtype.numpy_dtype
        239 try:
    --> 240     new_values = astype_array(values, dtype, copy=copy)
        241 except (ValueError, TypeError):
        242     # e.g. _astype_nansafe can fail on object-dtype of strings
        243     #  trying to convert to float
        244     if errors == "ignore":
    

    File c:\Users\hp\miniconda3\New folder\Lib\site-packages\pandas\core\dtypes\astype.py:182, in astype_array(values, dtype, copy)
        178     return values
        180 if not isinstance(values, np.ndarray):
        181     # i.e. ExtensionArray
    --> 182     values = values.astype(dtype, copy=copy)
        184 else:
        185     values = _astype_nansafe(values, dtype, copy=copy)
    

    File c:\Users\hp\miniconda3\New folder\Lib\site-packages\pandas\core\arrays\string_.py:946, in StringArray.astype(self, dtype, copy)
        943     values[mask] = np.nan
        944     return values
    --> 946 return super().astype(dtype, copy)
    

    File c:\Users\hp\miniconda3\New folder\Lib\site-packages\pandas\core\arrays\numpy_.py:280, in NumpyExtensionArray.astype(self, dtype, copy)
        277         return self.copy()
        278     return self
    --> 280 result = astype_array(self._ndarray, dtype=dtype, copy=copy)
        281 return result
    

    File c:\Users\hp\miniconda3\New folder\Lib\site-packages\pandas\core\dtypes\astype.py:185, in astype_array(values, dtype, copy)
        182     values = values.astype(dtype, copy=copy)
        184 else:
    --> 185     values = _astype_nansafe(values, dtype, copy=copy)
        187 # in pandas we don't store numpy str dtypes, so convert to object
        188 if isinstance(dtype, np.dtype) and issubclass(values.dtype.type, str):
    

    File c:\Users\hp\miniconda3\New folder\Lib\site-packages\pandas\core\dtypes\astype.py:134, in _astype_nansafe(arr, dtype, copy, skipna)
        130     raise ValueError(msg)
        132 if copy or object in (arr.dtype, dtype):
        133     # Explicit copy, or required since NumPy can't view from / to object.
    --> 134     return arr.astype(dtype, copy=True)
        136 return arr.astype(dtype, copy=copy)
    

    ValueError: invalid literal for int() with base 10: 'source'


## Problem 2, Part 2 — Question [4p]

A page loads in `Exp(1)` seconds if not preloaded. This means it is exponentially distributed with mean `1`.

A page loads in `Exp(10)` seconds if preloaded. This means it is exponentially distributed with mean `1/10`.

We only preload the most likely next site.

Given that we start in page `1`, simulate **10000** load times from page `1` for one single step, and store the result. Repeat the experiment, but this time preload the **two most likely pages**, and store the result.

**Required variables:**

- `problem2_page_load_times_top`
- `problem2_page_load_times_two`

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


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[3], line 8
          5 n_simulations = 10000
          6 start_page = 1
    ----> 8 transition_probs_from_page_1 = problem2_transition_matrix[start_page]
         10 # Simulate actual next pages according to transition probabilities
         11 next_pages = np.random.choice(
         12     np.arange(problem2_n_states),
         13     size=n_simulations,
         14     p=transition_probs_from_page_1
         15 )
    

    NameError: name 'problem2_transition_matrix' is not defined


## Problem 2, Part 3 — Question [3p]

Compare the average empirical load time from Part 2 with the theoretical one of no pre-loading.

Does the load time improve? Explain how you came to this conclusion in the free text field.

**Required variables:**

- `problem2_avg`
- `problem2_comparison`

Without preloading, every page load follows `Exp(1)`, so:

\[
E[T]=1.
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


## Problem 2, Part 4 — Question [4p]

Calculate the stationary distribution of the Markov chain and calculate the expected load time with respect to it.

**Required variables:**

- `problem2_stationary_distribution`
- `problem2_avg_stationary`

The stationary distribution \(\pi\) satisfies:

\[
\pi P = \pi.
\]

For each current page, if the probability of the most likely next page is \(p\), then the expected load time after preloading that most likely page is:

\[
p\cdot 0.1 + (1-p)\cdot 1.
\]



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

# Exam vB, Problem 3 — Fraud Detection Threshold and Cost

**Maximum points: 12**

**Question statement:**

In this problem we are interested in fraud detection in an e-commerce system. We are given the outputs of a classifier that predicts the probabilities of fraud. The goal is to explore the threshold choice as in individual assignment 4.

The costs associated with the predictions are:

| Outcome | Meaning | Cost |
|---|---|---:|
| True Positive (TP) | Detecting fraud and blocking the transaction, manual review etc. | 100 |
| True Negative (TN) | Allowing a legitimate transaction | 0 |
| False Positive (FP) | Incorrectly classifying a legitimate transaction as fraudulent | 120 |
| False Negative (FN) | Missing a fraudulent transaction | 600 |

The first code cell initializes the variables.


## Problem 3 — Data initialization cell

Run this cell first. It loads `data/fraud.csv`, splits the data into training/testing/validation sets, trains a logistic regression model, and creates the prediction probability arrays needed for the problem.

**Created variables:**

- `PROBLEM3_y_pred_proba_val`
- `PROBLEM3_y_true_val`
- `PROBLEM3_y_pred_proba_test`
- `PROBLEM3_y_true_test`



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

## Problem 3, Part 1 — Question [3p]

Complete the function `cost` to compute the **average cost** of a prediction model under a certain prediction threshold.

Then plot the cost as a function of the threshold using the validation data provided in the initialization cell. The plot should use thresholds between `0` and `1` with `0.01` increments.

**Required function / work:**

- `cost(y_true, y_predict_proba, threshold)`
- plot threshold on x-axis and average cost on y-axis



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

## Problem 3, Part 2 — Question [2.5p]

Find the threshold that minimizes the cost and calculate the cost at that threshold on the validation data.

Also calculate the precision and recall at the optimal threshold on class `1` and class `0`.

**Required variables:**

- `problem3_threshold`
- `problem3_cost_val`
- `problem3_y_pred_val`
- `problem3_precision_1`
- `problem3_recall_1`
- `problem3_precision_0`
- `problem3_recall_0`



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

## Problem 3, Part 3 — Question [2.5p]

Repeat Part 2, but this time find the best threshold to minimize the `0-1` loss.

Then calculate the difference in cost between the threshold found in Part 2 and the threshold found in Part 3.

The cost difference should be:

\[
\text{cost at 0-1 threshold} - \text{cost at cost-optimal threshold}.
\]

**Required variables:**

- `problem3_threshold_01`
- `problem3_cost_difference`



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

## Problem 3, Part 4 — Question [4p]

Provide a confidence interval around the optimal cost with **95% confidence**, applied to the test data, and explain all assumptions made.

Use Hoeffding's inequality with the threshold `problem3_threshold`.

**Required variables:**

- `problem3_lower_bound`
- `problem3_upper_bound`

For bounded random variables \(X_i\in[a,b]\), Hoeffding gives:

\[
P(|\bar{X}-E[X]|\ge \epsilon)\le 2\exp\left(\frac{-2n\epsilon^2}{(b-a)^2}\right).
\]

Solving for \(\epsilon\):

\[
\epsilon=(b-a)\sqrt{\frac{\log(2/\delta)}{2n}}.
\]

Here, for 95% confidence, \(\delta=0.05\), minimum cost is \(0\), and maximum cost is \(600\).



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
[\bar{C}-\epsilon,\bar{C}+\epsilon]
\]

where \(\bar{C}\) is the empirical average test cost.

