# Solved Notebook — Exam 21th of August 2025

Course: **1MS041 Introduction to Data Science / Introduktion till dataanalys**

This notebook includes the **whole questions**, followed by **solved code answers with step-by-step comments**.

> Important: To run the data-based parts, keep the course data files beside this notebook:
> - `data/SVD.csv`
> - `data/salaries.csv`
> - `Utils.py` if your course folder uses it



```
examID = "XXX"
```

# Problem 1 — SVD and anomaly detection

**Maximum Points = 14**

This problem is about SVD and anomaly detection. In all the problems where you are asked to produce a matrix or vector, they should be numpy arrays.

## Whole question

1. **[4p]** Load the file `data/SVD.csv` as instructed in the code cell. Compute the Singular Value Decomposition, i.e. construct the three matrices $U, D, V$ such that if $X$ is the data matrix of shape `n_samples x n_dimensions` then

$$X = UDV^T.$$

Put the resulting matrices in their variables, check that the shapes align with the instructions in the code cell. Finally, extract the first right and left singular vectors and store those as 1-d arrays in the instructed variables.

**Hint:** make sure that the first right and left singular vectors are correct by using the matrix, also be careful about the shape!!

2. **[3p]** Calculate the explained variance of using 1, 2, ... number of singular vectors and select the smallest number of singular vectors that is needed in order to explain at least **90%** of the variance.

3. **[3p]** With the number of components chosen in part 2, construct the best approximating matrix with the rank as the number of components. Explain geometrically what each row represents in the approximating matrix in terms of the original data.

4. **[4p]** Create a vector which corresponds to the row-wise Euclidean distance between the original matrix `problem1_data` and the approximating matrix `problem1_approximation` and plot the empirical distribution function of that distance. Based on the empirical distribution function choose a threshold such that 10 samples are above it and the rest below. Store the 10 samples in the instructed variable.


## Problem 1 — Answer Part 1


```
# Part 1: Load data and compute SVD

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from data/SVD.csv.
# header=None makes the code robust if the csv has no column names.
problem1_data = pd.read_csv("data/SVD.csv", header=None).values.astype(float)

# Check the dtype to confirm numbers were parsed correctly.
print("Data dtype:", problem1_data.dtype)
print("Data shape:", problem1_data.shape)

# Compute thin / reduced SVD.
# full_matrices=False gives:
# U shape = (n_samples, n_dimensions)
# D shape = (n_dimensions,)
# Vt shape = (n_dimensions, n_dimensions)
problem1_U, problem1_D, problem1_Vt = np.linalg.svd(problem1_data, full_matrices=False)

# The exam asks for V, where X = U D V^T.
# numpy returns V^T as Vt, so V is Vt.T.
problem1_V = problem1_Vt.T

# First right singular vector = first column of V.
problem1_first_right_singular_vector = problem1_V[:, 0].flatten()

# First left singular vector = first column of U.
problem1_first_left_singular_vector = problem1_U[:, 0].flatten()

print("U shape:", problem1_U.shape)
print("D shape:", problem1_D.shape)
print("V shape:", problem1_V.shape)
print("First right singular vector shape:", problem1_first_right_singular_vector.shape)
print("First left singular vector shape:", problem1_first_left_singular_vector.shape)

# Sanity check reconstruction using all components:
full_reconstruction = problem1_U @ np.diag(problem1_D) @ problem1_V.T
print("Full SVD reconstruction close to original:", np.allclose(problem1_data, full_reconstruction))
```

## Problem 1 — Answer Part 2


```
# Part 2: Explained variance

# For SVD/PCA-style explained variance, the contribution of each singular value
# is proportional to singular_value^2.
singular_value_energy = problem1_D ** 2

# Cumulative explained variance from using 1, 2, ..., n_dimensions components.
problem1_explained_variance = np.cumsum(singular_value_energy) / np.sum(singular_value_energy)

# Smallest number of components explaining at least 90%.
problem1_num_components = int(np.searchsorted(problem1_explained_variance, 0.90) + 1)

print("Explained variance array:")
print(problem1_explained_variance)

print("Number of components needed for at least 90% variance:", problem1_num_components)
print("Explained variance at that number:", problem1_explained_variance[problem1_num_components - 1])
```

## Problem 1 — Answer Part 3


```
# Part 3: Best rank-k approximation

k = problem1_num_components

# Keep only the first k singular vectors/values.
U_k = problem1_U[:, :k]
D_k = np.diag(problem1_D[:k])
V_k = problem1_V[:, :k]

# Best rank-k approximation according to the Eckart-Young theorem.
problem1_approximation = U_k @ D_k @ V_k.T

print("Approximation shape:", problem1_approximation.shape)
print("Original shape:", problem1_data.shape)
```

### Free-text answer for Problem 1 Part 3

Each row of `problem1_approximation` is the low-rank reconstruction of the corresponding original row in `problem1_data`.

Geometrically, the original row is a point in the original feature space. The approximation is the projection/reconstruction of that point using only the first `problem1_num_components` singular directions. These directions capture at least 90% of the total variance/energy in the data, so each approximated row keeps the main structure of the original sample but removes smaller variation/noise from the discarded singular directions.

If a row is reconstructed badly, it means that the sample is not well explained by the main low-dimensional pattern, which is why large reconstruction error can indicate an anomaly.


## Problem 1 — Answer Part 4


```
# Part 4: Reconstruction error, EDF plot, threshold, and outliers

# Row-wise Euclidean distance between original data and approximation.
problem1_reconstruction_error = np.linalg.norm(problem1_data - problem1_approximation, axis=1)

# Empirical distribution function.
sorted_errors = np.sort(problem1_reconstruction_error)
n = len(sorted_errors)
edf_y = np.arange(1, n + 1) / n

plt.figure(figsize=(7, 5))
plt.step(sorted_errors, edf_y, where="post")
plt.xlabel("Row-wise reconstruction error")
plt.ylabel("Empirical distribution function")
plt.title("EDF of reconstruction error")
plt.grid(True)
plt.show()

# Choose the 10 largest reconstruction errors as outliers.
outlier_indices = np.argsort(problem1_reconstruction_error)[-10:]

# Choose threshold just below the smallest of the 10 largest errors
# so exactly these 10 are above it, assuming no exact ties.
problem1_threshold = float(np.min(problem1_reconstruction_error[outlier_indices]) - 1e-12)

# Store the 10 outlier samples.
problem1_outliers = problem1_data[problem1_reconstruction_error > problem1_threshold]

print("Threshold:", problem1_threshold)
print("Outliers shape:", problem1_outliers.shape)
print("Outlier indices:", outlier_indices)
```

# Problem 2 — Linear regression on salaries dataset

**Maximum Points = 13**

You are given the data-science job salaries dataset found in `data/salaries.csv`, which contains the salaries of jobs, their experience level and how much of the working hours are remote. Your task is to train a linear regression model to predict the salary of a job based on its attributes.

## Dataset description from the exam

- `work_year`: The year the salary was paid.
- `experience_level`: The experience level in the job during the year with possible values:
  - 0 Entry-level / Junior
  - 1 Mid-level / Intermediate
  - 2 Senior-level / Expert
  - 3 Executive-level
- `employment_type`: The type of employment for the role: Part-time, Full-time, Contract, Freelance.
- `salary_in_usd`: The total gross salary amount paid in US Dollars.
- `remote_ratio`: The overall amount of work done remotely:
  - 0 No remote work, less than 20%
  - 50 Partially remote
  - 100 Fully remote, more than 80%

## Whole question

1. Load the data into a pandas dataframe `problem2_df`. Based on the column names, figure out what are the features and the target and fill in the answer in the correct cell below. **[2p]**
2. Split the data into train and test. **[2p]**
3. Train the model. **[1p]**
4. On the test set, evaluate the model by computing the mean absolute relative error and plot the empirical distribution function of the residual with confidence bands using the DKW inequality and 99% confidence. **[3p]**

$$
\text{Absolute relative error} =
\left|
\frac{\text{true} - \text{predicted}}{\text{true}}
\right|
$$

5. Provide a scatter plot where the x-axis corresponds to the predicted value and the y-axis is the true value, over the test set. **[2p]**
6. Reason about the performance, for instance, is the value of the mean absolute relative error good/bad and what do you think about the scatter plot in point 5? **[3p]**


## Problem 2 — Answer Part 1


```
# Part 1: Load salaries data and choose features/target

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

problem2_df = pd.read_csv("data/salaries.csv")

print(problem2_df.head())
print(problem2_df.columns)

# Features are the input columns used to predict salary.
problem2_features = ["work_year", "experience_level", "employment_type", "remote_ratio"]

# Target is the value we want to predict.
problem2_target = "salary_in_usd"

print("Features:", problem2_features)
print("Target:", problem2_target)
```

## Problem 2 — Answer Part 2


```
# Part 2: Split the data into train and test

from sklearn.model_selection import train_test_split

# Select X and y.
X_raw = problem2_df[problem2_features].copy()
y = problem2_df[problem2_target].copy()

# LinearRegression needs numeric input.
# employment_type is categorical, so we convert it to dummy/one-hot variables.
X = pd.get_dummies(X_raw, columns=["employment_type"], drop_first=True)

problem2_X_train, problem2_X_test, problem2_y_train, problem2_y_test = train_test_split(
    X,
    y,
    train_size=0.8,
    random_state=42
)

print("X train shape:", problem2_X_train.shape)
print("X test shape:", problem2_X_test.shape)
print("y train shape:", problem2_y_train.shape)
print("y test shape:", problem2_y_test.shape)
```

## Problem 2 — Answer Part 3


```
# Part 3: Train the linear regression model

from sklearn.linear_model import LinearRegression

problem2_model = LinearRegression()

# Train the model on the training data.
problem2_model.fit(problem2_X_train, problem2_y_train)

print("Model trained.")
print("Intercept:", problem2_model.intercept_)
print("Coefficients:", problem2_model.coef_)
```

## Problem 2 — Answer Part 4


```
# Part 4: Mean absolute relative error and EDF of residuals with 99% DKW bands

# Predict salary on the test set.
problem2_y_pred = problem2_model.predict(problem2_X_test)

# Mean Absolute Relative Error:
# mean(abs((true - predicted) / true))
problem2_mare = float(np.mean(np.abs((problem2_y_test.values - problem2_y_pred) / problem2_y_test.values)))

print("Mean Absolute Relative Error (MARE):", problem2_mare)

# Residual = true - predicted.
problem2_residuals = problem2_y_test.values - problem2_y_pred

# EDF values.
residuals_sorted = np.sort(problem2_residuals)
n = len(residuals_sorted)
edf = np.arange(1, n + 1) / n

# DKW inequality:
# P(sup_x |F_n(x) - F(x)| > epsilon) <= 2 exp(-2 n epsilon^2)
# For 99% confidence, alpha = 0.01.
alpha = 0.01
epsilon = np.sqrt(np.log(2 / alpha) / (2 * n))

lower_band = np.maximum(edf - epsilon, 0)
upper_band = np.minimum(edf + epsilon, 1)

plt.figure(figsize=(7, 5))
plt.step(residuals_sorted, edf, where="post", label="Empirical CDF")
plt.step(residuals_sorted, lower_band, where="post", linestyle="--", label="99% lower band")
plt.step(residuals_sorted, upper_band, where="post", linestyle="--", label="99% upper band")
plt.xlabel("Residual = true salary - predicted salary")
plt.ylabel("Empirical distribution function")
plt.title("EDF of residuals with 99% DKW confidence bands")
plt.legend()
plt.grid(True)
plt.show()
```

## Problem 2 — Answer Part 5


```
# Part 5: Scatter plot of predicted value vs true value

plt.figure(figsize=(7, 5))
plt.scatter(problem2_y_pred, problem2_y_test.values, alpha=0.7)
plt.xlabel("Predicted salary in USD")
plt.ylabel("True salary in USD")
plt.title("Predicted vs True Salaries on Test Set")

# Add reference line y = x.
min_val = min(np.min(problem2_y_pred), np.min(problem2_y_test.values))
max_val = max(np.max(problem2_y_pred), np.max(problem2_y_test.values))
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.grid(True)
plt.show()
```

## Problem 2 — Answer Part 6

### Discussion on the value of the MARE

`problem2_mare` is the average size of the prediction error relative to the true salary. For example, a MARE of `0.25` means that, on average, predictions are off by about 25% of the true salary.

A small MARE is good because it means the model predicts salaries close to their real values. A large MARE is bad because it means the model makes large relative errors. Salary prediction is usually difficult because salaries can vary a lot due to factors not included in the features, such as country, company, job title, currency, and employee negotiation. Therefore, if the MARE is moderately large, the reason may be that the model is too simple or the feature list is incomplete.

### Discussion on the predicted vs. true scatterplot

In the scatter plot, good predictions should lie close to the diagonal line $y=x$. Points far above the line mean the model underpredicted the salary. Points far below the line mean the model overpredicted the salary.

If the plot has points widely scattered away from the diagonal, the linear regression model is not capturing the salary pattern very well. If the points are close to the diagonal, the model is performing well.

### Discussion

A basic linear regression model is easy to interpret, but it may be too simple for salary data. The relationship between features and salary may not be fully linear, and the dataset may need more important predictors or better feature engineering. The MARE and scatter plot together tell us whether the model is accurate enough for practical use.


# Problem 3 — Markov chains

**Maximum Points = 13**

The exam gives two Markov chain diagrams: **Markov Chain A** and **Markov Chain B**.

## Whole question

Answer each question for all chains:

1. **[2p]** What is the transition matrix?
2. **[1p]** Is the Markov chain irreducible?
3. **[4p]** Is the Markov chain aperiodic? What is the period for each state? Hint: Recall the definition of period; let

$$
T := \{t \in \mathbb{N}: P^t(x,x) > 0\}
$$

and the greatest common divisor of $T$ is the period.

4. **[2p]** Being in state A at time 0, what is the probability of being in state B at time 5, after 5 steps?
5. **[4p]** Define $T$ as the first time being in state D starting in state A. That is, if $X_0, X_1, \ldots$ is the Markov chain then define for $X_0 = A$

$$
T(\omega) = \inf_{t\in\mathbb{N}} \{t : X_t(\omega) = D\}
$$

where the infimum over the empty set is $\infty$. Calculate:

$$
P(T=1), P(T=2), P(T=3), P(T=4), P(T=5), P(T=\infty).
$$

## State ordering used in this solution

- Markov Chain A uses states `[A, B, C, D]`, corresponding to indices `[0, 1, 2, 3]`.
- Markov Chain B uses states `[A, B, C, D, E, F]`, corresponding to indices `[0, 1, 2, 3, 4, 5]`.


## Problem 3 — Answer Part 1


```
# PART 1: Transition matrices

import numpy as np

# State order for Chain A: A, B, C, D
# From the diagram:
# A -> B with 0.2, A -> D with 0.8
# B -> C with 1
# C -> B with 1
# D -> A with 0.5, D -> C with 0.5
problem3_A = np.array([
    [0.0, 0.2, 0.0, 0.8],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0]
])

# State order for Chain B: A, B, C, D, E, F
# From the diagram:
# A -> B with 1
# B -> C with 1
# C -> B with 0.5, C -> D with 0.5
# D -> C with 0.5, D -> E with 0.5
# E -> F with 1
# F -> A with 0.5, F -> E with 0.5
problem3_B = np.array([
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.5, 0.0, 0.0, 0.0, 0.5, 0.0]
])

print("Chain A transition matrix:")
print(problem3_A)
print("Row sums A:", problem3_A.sum(axis=1))

print("\nChain B transition matrix:")
print(problem3_B)
print("Row sums B:", problem3_B.sum(axis=1))
```

## Problem 3 — Answer Part 2


```
# PART 2: Irreducibility

# Chain A is reducible because {B, C} is a closed communicating class.
# Once the chain enters B or C, it cannot return to A or D.
problem3_A_irreducible = False

# Chain B is irreducible because every state can reach every other state through directed paths.
problem3_B_irreducible = True

print("Chain A irreducible:", problem3_A_irreducible)
print("Chain B irreducible:", problem3_B_irreducible)
```

## Problem 3 — Answer Part 3


```
# PART 3: Aperiodicity and periods

# Chain A:
# All possible return times are even for the states in the relevant classes.
# So each state's period is 2, and the chain is not aperiodic.
problem3_A_periods = np.array([2, 2, 2, 2])
problem3_A_is_aperiodic = False

# Chain B:
# The graph is bipartite:
# {A, C, E} and {B, D, F}, with every step alternating between the two sets.
# Therefore all returns happen at even times, so period = 2 for every state.
problem3_B_periods = np.array([2, 2, 2, 2, 2, 2])
problem3_B_is_aperiodic = False

print("Chain A is aperiodic:", problem3_A_is_aperiodic)
print("Chain A periods:", problem3_A_periods)

print("Chain B is aperiodic:", problem3_B_is_aperiodic)
print("Chain B periods:", problem3_B_periods)
```

## Problem 3 — Answer Part 4


```
# PART 4: Probability of being in state B at time 5, starting from A

# This is the (A, B) entry of P^5.
problem3_A_PB5 = float(np.linalg.matrix_power(problem3_A, 5)[0, 1])
problem3_B_PB5 = float(np.linalg.matrix_power(problem3_B, 5)[0, 1])

print("Chain A P(X_5 = B | X_0 = A):", problem3_A_PB5)
print("Chain B P(X_5 = B | X_0 = A):", problem3_B_PB5)
```

## Problem 3 — Answer Part 5


```
# PART 5: First hitting time of D starting from A

# Chain A:
# Starting at A, either we go directly to D at time 1 with probability 0.8,
# or we go to B at time 1 with probability 0.2.
# From B we go to C, and from C we go back to B forever, so D is never reached.
problem3_A_PT1 = 0.8
problem3_A_PT2 = 0.0
problem3_A_PT3 = 0.0
problem3_A_PT4 = 0.0
problem3_A_PT5 = 0.0
problem3_A_PT_inf = 0.2

# Chain B:
# Starting at A:
# A -> B at time 1 with probability 1.
# B -> C at time 2 with probability 1.
# From C we hit D with probability 0.5 at the next step.
# If we go C -> B instead, we come back to C two steps later and try again.
problem3_B_PT1 = 0.0
problem3_B_PT2 = 0.0
problem3_B_PT3 = 0.5
problem3_B_PT4 = 0.0
problem3_B_PT5 = 0.25

# Chain B is finite and irreducible, so D is eventually hit with probability 1.
problem3_B_PT_inf = 0.0

print("Chain A first hit probabilities:")
print("T=1:", problem3_A_PT1)
print("T=2:", problem3_A_PT2)
print("T=3:", problem3_A_PT3)
print("T=4:", problem3_A_PT4)
print("T=5:", problem3_A_PT5)
print("T=inf:", problem3_A_PT_inf)

print("\nChain B first hit probabilities:")
print("T=1:", problem3_B_PT1)
print("T=2:", problem3_B_PT2)
print("T=3:", problem3_B_PT3)
print("T=4:", problem3_B_PT4)
print("T=5:", problem3_B_PT5)
print("T=inf:", problem3_B_PT_inf)
```

# Final checklist

This notebook includes:

- The full question text before each problem.
- Solved Python code for every code cell.
- Step-by-step comments explaining the logic.
- Free-text answers for interpretation/discussion parts.
- Plots requested by the exam.

