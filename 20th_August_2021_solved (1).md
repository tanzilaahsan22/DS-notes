# Re-exam 20th of August 2021 — 1MS041 Introduction to Data Science

This notebook contains the **full exam questions**, solved Python code, comments, and outputs where possible.

Required external course files for some problems:
- `regression.csv`
- `a_sequence.txt`
- `data.csv`

Put those files in the same folder as this notebook before running the relevant cells.


```python
# Enter your anonymous exam id by replacing XXXX in this cell below
# do NOT delete this cell
MyAnonymousExamID = 'XXX'

total_points = 0.0

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import optimize
```

---
# Problem 1 [10p]

In many areas of data science and machine learning we need to produce random samples in different ways.
This can be done to compute difficult integrals or validate algorithms.

1. **[2p]** Implement a Linear Congruential Generator that produces pseudo random numbers from the uniform distribution `[0,1]`.
2. **[2p]** Use that to construct samples from the uniform distribution over the unit ball in `100` dimensions.
3. **[2p]** Estimate `E[X]` using `1000` samples.
4. **[4p]** Use the bootstrap method to produce a bootstrap confidence interval for `E[X]` from your samples above. Implement your own bootstrap sampler using your implemented sampler from Problem 1.1.

## Problem 1.1 — Linear Congruential Generator

A Linear Congruential Generator has the form

\[
u_{n+1} = (a u_n + c) \mod m
\]

Then we divide by `m` to convert integers to numbers in `[0,1)`.


```python
def uniform_pseudo_random(n_samples, seed=1):
    """
    Linear Congruential Generator producing pseudo-random Uniform[0,1) samples.

    Parameters
    ----------
    n_samples : int
        Number of pseudo-random numbers to generate.
    seed : int
        Starting value.

    Returns
    -------
    list
        List of floats in [0,1).
    """
    # Good classic parameters:
    # m = 2^31, a = 1103515245, c = 12345
    m = 2**31
    a = 1103515245
    c = 12345

    x = seed
    out = []

    for _ in range(n_samples):
        x = (a * x + c) % m
        out.append(x / m)

    return out


# Small output test
print(uniform_pseudo_random(10, seed=1))
```


```python
total_points += 0
```

## Problem 1.2 — Sample uniformly from the 100-dimensional unit ball

We want samples from

\[
B^{100} = \{x \in \mathbb{R}^{100}: \|x\|_2 \le 1\}.
\]

Method:

1. Generate a random direction by sampling a normal vector and normalizing it.
2. Generate a radius \(R = U^{1/d}\), where \(U \sim \text{Uniform}(0,1)\).
3. Return \(R \cdot \Theta\), where \(\Theta\) is the random unit direction.


```python
def sampler_problem_1(n_samples, d=100, seed=1):
    """
    Samples uniformly from the d-dimensional unit ball.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, d).
    """
    rng = np.random.default_rng(seed)

    # Step 1: sample random directions from a standard normal distribution.
    Z = rng.normal(size=(n_samples, d))

    # Normalize each vector to length 1.
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Theta = Z / norms

    # Step 2: generate radii. For a uniform d-ball, R = U^(1/d).
    U = np.array(uniform_pseudo_random(n_samples, seed=seed + 123))
    R = U ** (1 / d)

    # Step 3: multiply radius and direction.
    samples = Theta * R[:, None]

    return samples


samples_test = sampler_problem_1(5)
print("Shape:", samples_test.shape)
print("Norms should be <= 1:", np.linalg.norm(samples_test, axis=1))
```


```python
total_points += 0
```

## Problem 1.3 — Estimate \(E[X]\) using 1000 samples

Since \(X\) is a 100-dimensional random vector, \(E[X]\) is also a 100-dimensional vector.
For a symmetric unit ball, the true expectation is the zero vector, so the sample mean should be close to zero.


```python
# Generate 1000 samples from the 100-dimensional unit ball.
samples = sampler_problem_1(1000, d=100, seed=42)

# Estimate E[X] coordinate-wise.
V_X = samples.mean(axis=0)

print("samples shape:", samples.shape)
print("Estimated E[X] shape:", V_X.shape)
print("First 10 coordinates of estimated E[X]:")
print(V_X[:10])

# A scalar summary: norm of the estimated mean vector.
print("Norm of estimated mean vector:", np.linalg.norm(V_X))
```


```python
total_points += 0
```

## Problem 1.4 — Bootstrap 95% confidence interval

We bootstrap by resampling the rows of `samples` with replacement.

Because \(X\) is vector-valued, the interval can be computed coordinate-wise.
Below I compute a coordinate-wise 95% bootstrap interval for \(E[X]\).


```python
def bootstrap_indices(n, B=1000, seed=123):
    """
    Bootstrap index sampler using the LCG from Problem 1.1.
    """
    u = np.array(uniform_pseudo_random(B * n, seed=seed))
    idx = np.floor(u * n).astype(int)
    idx = np.clip(idx, 0, n - 1)
    return idx.reshape(B, n)


def bootstrap_mean_ci(samples, B=1000, alpha=0.05, seed=123):
    """
    Coordinate-wise percentile bootstrap confidence interval for E[X].
    """
    n = samples.shape[0]
    idx = bootstrap_indices(n, B=B, seed=seed)

    boot_means = []
    for b in range(B):
        boot_sample = samples[idx[b]]
        boot_means.append(boot_sample.mean(axis=0))

    boot_means = np.array(boot_means)

    lower = np.quantile(boot_means, alpha / 2, axis=0)
    upper = np.quantile(boot_means, 1 - alpha / 2, axis=0)

    return lower, upper, boot_means


ci_lower, ci_upper, boot_means = bootstrap_mean_ci(samples, B=1000, alpha=0.05, seed=7)

print("95% bootstrap CI for first coordinate E[X_1]:")
print("[%.6f, %.6f]" % (ci_lower[0], ci_upper[0]))

print("\nFirst 5 coordinate-wise confidence intervals:")
for j in range(5):
    print("Coordinate %d: [%.6f, %.6f]" % (j + 1, ci_lower[j], ci_upper[j]))
```


```python
total_points += 0
```

---
# Problem 2 [10p]

Sometimes it is important to regress on time-to-event data. Here \(Y_i\) corresponds to the time to an event and takes values in \([0,\infty)\).

A reasonable distribution for time-to-event data is the Exponential distribution:

\[
Y_i \mid X_i \sim \text{Exponential}(\lambda(X_i)),
\]

where

\[
\lambda(X_i)=G(\beta_0+\beta_1 X_i), \quad G(x)=e^x.
\]

Recall that if \(X \sim \text{Exponential}(\lambda)\), then its density is

\[
f(x;\lambda)=\lambda e^{-\lambda x}, \quad \lambda>0,\; x\ge 0.
\]

Tasks:

1. **[4p]** Implement the ExponentialRegression class by writing down the conditional likelihood as the loss and writing optimization code.
2. **[2p]** Load `regression.csv`, whose header contains columns `X` and `Y`.
3. **[2p]** Fit ExponentialRegression.
4. **[2p]** Explain convergence and final loss.

## Problem 2.1 — Implement Exponential Regression

For one observation:

\[
\lambda_i = \exp(\beta_0 + \beta_1 x_i)
\]

\[
f(y_i|x_i)=\lambda_i \exp(-\lambda_i y_i)
\]

The negative log-likelihood is

\[
L(\beta)= -\sum_i [\log(\lambda_i)-\lambda_i y_i].
\]


```python
class ExponentialRegression(object):
    def __init__(self, lam=0, max_iter=10000):
        self.coeffs = None
        self.result = None
        self.lam = lam
        self.max_iter = max_iter

    def _prepare_X(self, X):
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        intercept = np.ones((X.shape[0], 1))
        return np.hstack([intercept, X])

    def fit(self, X, Y):
        X_design = self._prepare_X(X)
        Y = np.asarray(Y).reshape(-1)

        def f(beta):
            eta = X_design @ beta
            rate = np.exp(eta)

            # Negative log-likelihood:
            nll = -np.sum(np.log(rate) - rate * Y)

            # Optional L2 penalty:
            penalty = self.lam * np.sum(beta**2)

            return nll + penalty

        beta0 = np.zeros(X_design.shape[1])

        self.result = optimize.minimize(
            f,
            beta0,
            method="BFGS",
            options={"maxiter": self.max_iter}
        )

        self.coeffs = self.result.x
        return self

    def predict_rate(self, X):
        if self.coeffs is None:
            raise ValueError("Model is not fitted yet.")

        X_design = self._prepare_X(X)
        return np.exp(X_design @ self.coeffs)

    def predict(self, X):
        # For Exponential(rate), E[Y|X] = 1/rate.
        rate = self.predict_rate(X)
        return 1 / rate

    def loss(self, X, Y):
        if self.coeffs is None:
            raise ValueError("Model is not fitted yet.")

        X_design = self._prepare_X(X)
        Y = np.asarray(Y).reshape(-1)

        eta = X_design @ self.coeffs
        rate = np.exp(eta)

        nll = -np.sum(np.log(rate) - rate * Y)
        penalty = self.lam * np.sum(self.coeffs**2)

        return nll + penalty


# The exam template accidentally calls it PoissonRegression, but the model is exponential regression.
PoissonRegression = ExponentialRegression
```


```python
total_points += 0
```

## Problem 2.2 — Load `regression.csv`

The file was not included in the uploaded PDF. Put `regression.csv` in the same folder as this notebook.


```python
try:
    regression_df = pd.read_csv("regression.csv")
    X_regression = regression_df["X"].values
    Y_regression = regression_df["Y"].values

    print(regression_df.head())
    print("X shape:", X_regression.shape)
    print("Y shape:", Y_regression.shape)

except FileNotFoundError:
    print("regression.csv not found. Put regression.csv in the same folder as this notebook.")
```


```python
total_points += 0
```

## Problem 2.3 — Fit the model


```python
try:
    model_problem2 = ExponentialRegression(lam=0, max_iter=10000)
    model_problem2.fit(X_regression, Y_regression)

    beta = model_problem2.coeffs

    print("Estimated beta:", beta)
    print("Optimizer success:", model_problem2.result.success)
    print("Optimizer message:", model_problem2.result.message)
    print("Final loss:", model_problem2.loss(X_regression, Y_regression))

except NameError:
    print("Run Problem 2.2 first after placing regression.csv in the notebook folder.")
```


```python
total_points += 0
```

## Problem 2.4 — Did it converge?

The optimizer converges if `model_problem2.result.success` is `True`.

If it converges, report `Estimated beta` and `Final loss`.

If it does not converge, common reasons include:
- too few iterations,
- bad scaling of `X`,
- extremely large or small `Y`,
- overflow in `exp(beta0 + beta1 X)`,
- a flat or numerically unstable likelihood surface.


```python
try:
    print("Converged:", model_problem2.result.success)
    print("Message:", model_problem2.result.message)
    print("Final loss:", model_problem2.loss(X_regression, Y_regression))
    beta = model_problem2.coeffs
except NameError:
    beta = None
    print("Model not fitted yet because regression.csv was not available.")
```


```python
total_points += 0
```

---
# Problem 3 [10p]

1. **[2p]** Take the file `a_sequence.txt` and load it as a list of integers.
2. **[2p]** Define a Markov chain from this list of integers.
   - What are the states?
3. **[4p]** Estimate the transition matrix using the maximum likelihood estimator.
4. **[2p]** Find the steady state distribution.

## Problem 3.1 — Read `a_sequence.txt`

The file was not included in the uploaded PDF. Put `a_sequence.txt` in the same folder as this notebook.


```python
try:
    with open("a_sequence.txt", "r") as f:
        text = f.read()

    cleaned = text.replace(",", " ").replace("\n", " ").replace("\t", " ")
    numbers = [int(x) for x in cleaned.split()]

    print("First 20 numbers:", numbers[:20])
    print("Number of observations:", len(numbers))

except FileNotFoundError:
    numbers = []
    print("a_sequence.txt not found. Put a_sequence.txt in the same folder as this notebook.")
```


```python
total_points += 0
```

## Problem 3.2 — Define the Markov chain

A Markov chain is a stochastic process where the next state depends only on the current state.

Here:
- the **states** are the unique integers appearing in `numbers`,
- each adjacent pair `(numbers[t], numbers[t+1])` is one observed transition,
- the transition probability \(P_{ij}\) means:

\[
P_{ij} = P(X_{t+1}=j \mid X_t=i)
\]


```python
states = sorted(set(numbers))
n_states = len(states)

state_to_index = {state: i for i, state in enumerate(states)}
index_to_state = {i: state for state, i in state_to_index.items()}

print("States:", states[:20], "..." if len(states) > 20 else "")
print("Number of states:", n_states)
```


```python
total_points += 0
```

## Problem 3.3 — Estimate transition matrix by maximum likelihood

For each state \(i\), count how many times the chain goes from \(i\) to \(j\).
Then normalize each row:

\[
\hat{P}_{ij} = \frac{\#(i \to j)}{\#(i \to \text{anything})}
\]


```python
if len(numbers) >= 2:
    counts = np.zeros((n_states, n_states))

    for current_state, next_state in zip(numbers[:-1], numbers[1:]):
        i = state_to_index[current_state]
        j = state_to_index[next_state]
        counts[i, j] += 1

    row_sums = counts.sum(axis=1, keepdims=True)

    P = np.zeros_like(counts, dtype=float)
    for i in range(n_states):
        if row_sums[i, 0] > 0:
            P[i, :] = counts[i, :] / row_sums[i, 0]
        else:
            P[i, i] = 1.0

    print("Transition matrix shape:", P.shape)
    print("First 5 row sums:", P.sum(axis=1)[:5])

else:
    P = None
    print("Need at least 2 numbers to estimate a transition matrix.")
```


```python
total_points += 0
```

## Problem 3.4 — Steady state distribution

The steady state distribution \(\pi\) satisfies:

\[
\pi P = \pi
\]

Interpretation: if the Markov chain runs for a long time, \(\pi_i\) is the long-run fraction of time spent in state \(i\).


```python
if P is not None:
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1))

    stationary = np.real(eigvecs[:, idx])
    stationary = np.abs(stationary)
    fixed_point = stationary / stationary.sum()

    print("Steady state distribution:")
    print(fixed_point)
    print("Sum:", fixed_point.sum())

else:
    fixed_point = None
    print("Transition matrix unavailable because a_sequence.txt was not loaded.")
```


```python
total_points += 0
```

---
# Problem 4 [10p]

In the exam assignment, there is a csv file called `data.csv`. The first line is the header.

1. **[2p]** Load the file as a numpy array of shape `400 x 4096`.
2. **[2p]** Implement the power method to compute the first singular value and singular vector.
3. **[2p]** Compute the second singular value and component using the power method.
4. **[2p]** Perform full singular value decomposition using numpy.
5. **[2p]** Compare the results from the power method and numpy.

## Problem 4.1 — Load `data.csv`

The file was not included in the uploaded PDF. Put `data.csv` in the same folder as this notebook.


```python
try:
    data_df = pd.read_csv("data.csv")
    X_data = data_df.values.astype(float)

    print("Loaded data shape:", X_data.shape)

    # If the csv has an index column by mistake, this tries to fix it.
    if X_data.shape[1] == 4097:
        X_data = X_data[:, 1:]
        print("Dropped first column. New shape:", X_data.shape)

except FileNotFoundError:
    X_data = None
    print("data.csv not found. Put data.csv in the same folder as this notebook.")
```


```python
total_points += 0
```

## Problem 4.2 — First singular value/vector using power method

The right singular vector \(v_1\) is the top eigenvector of \(X^T X\).

Then:

\[
\sigma_1 = \|Xv_1\|_2
\]

and

\[
u_1 = \frac{Xv_1}{\sigma_1}.
\]


```python
def power_method_svd_first_component(X, n_iter=200, tol=1e-10, seed=1):
    """
    Compute first singular value and vectors using the power method on X.T @ X.
    """
    rng = np.random.default_rng(seed)

    n_features = X.shape[1]
    v = rng.normal(size=n_features)
    v = v / np.linalg.norm(v)

    A = X.T @ X

    for _ in range(n_iter):
        v_new = A @ v
        v_new = v_new / np.linalg.norm(v_new)

        if np.linalg.norm(v_new - v) < tol:
            v = v_new
            break

        v = v_new

    Xv = X @ v
    sigma = np.linalg.norm(Xv)
    u = Xv / sigma

    return sigma, u, v


if X_data is not None:
    sigma1_power, u1_power, v1_power = power_method_svd_first_component(X_data)
    print("First singular value from power method:", sigma1_power)
    print("u1 shape:", u1_power.shape)
    print("v1 shape:", v1_power.shape)
else:
    sigma1_power = u1_power = v1_power = None
    print("Cannot run power method because data.csv was not loaded.")
```


```python
total_points += 0
```

## Problem 4.3 — Second singular value/vector using deflation

After finding the first component, remove it:

\[
X_2 = X - \sigma_1 u_1 v_1^T
\]

Then run the power method again on \(X_2\).


```python
if X_data is not None:
    X_deflated = X_data - sigma1_power * np.outer(u1_power, v1_power)

    sigma2_power, u2_power, v2_power = power_method_svd_first_component(
        X_deflated,
        n_iter=200,
        tol=1e-10,
        seed=2
    )

    print("Second singular value from power method:", sigma2_power)
    print("u2 shape:", u2_power.shape)
    print("v2 shape:", v2_power.shape)

else:
    sigma2_power = u2_power = v2_power = None
    print("Cannot compute second component because data.csv was not loaded.")
```


```python
total_points += 0
```

## Problem 4.4 — Full SVD using numpy


```python
if X_data is not None:
    U_np, S_np, Vt_np = np.linalg.svd(X_data, full_matrices=False)

    print("First 10 numpy singular values:")
    print(S_np[:10])

    print("U shape:", U_np.shape)
    print("S shape:", S_np.shape)
    print("Vt shape:", Vt_np.shape)

else:
    U_np = S_np = Vt_np = None
    print("Cannot run numpy SVD because data.csv was not loaded.")
```


```python
total_points += 0
```

## Problem 4.5 — Compare power method and numpy

The signs of singular vectors can flip, so compare vectors using absolute dot product.


```python
if X_data is not None:
    print("Power method sigma1:", sigma1_power)
    print("Numpy sigma1:", S_np[0])
    print("Absolute error sigma1:", abs(sigma1_power - S_np[0]))

    print("\nPower method sigma2:", sigma2_power)
    print("Numpy sigma2:", S_np[1])
    print("Absolute error sigma2:", abs(sigma2_power - S_np[1]))

    v1_similarity = abs(np.dot(v1_power, Vt_np[0]))
    v2_similarity = abs(np.dot(v2_power, Vt_np[1]))

    print("\nSimilarity between power v1 and numpy v1:", v1_similarity)
    print("Similarity between power v2 and numpy v2:", v2_similarity)

else:
    print("Cannot compare because data.csv was not loaded.")
```


```python
total_points += 0
```

---
# Total score cell


```python
print("Your total score is: %.2f" % total_points)
```
