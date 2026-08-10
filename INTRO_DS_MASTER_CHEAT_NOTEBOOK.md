# 1MS041 Intro to Data Science — MASTER OPEN-BOOK CHEAT NOTEBOOK

Use this as your exam template bank. Copy the matching block, change file names / columns / matrices, and run.

Sections:
1. Imports
2. Markov chains
3. Linear regression
4. Logistic regression / classification
5. EDF + DKW confidence bands
6. Hoeffding intervals
7. Sampling templates
8. Poisson regression
9. SVD / dimensionality reduction
10. Common free-text answers


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, precision_score, recall_score, accuracy_score
)

np.random.seed(42)
```

# 2. Markov Chains

Use for transition matrix, probability after n steps, irreducible, aperiodic, stationary distribution, reversibility, expected hitting time.

State order example: A=0, B=1, C=2.


```python
# Transition matrix template
P = np.array([
    [0.3, 0.7, 0.0],
    [0.2, 0.5, 0.3],
    [0.0, 0.5, 0.5]
])

print("Row sums:", P.sum(axis=1))

# Probability from state i to state j after n steps
n = 10
i = 0
j = 2

Pn = np.linalg.matrix_power(P, n)
prob_after_n_steps = float(Pn[i, j])
prob_after_n_steps
```


```python
# Stationary distribution: pi P = pi, sum(pi)=1
eigvals, eigvecs = np.linalg.eig(P.T)
idx = np.argmin(np.abs(eigvals - 1))

pi = np.real(eigvecs[:, idx])
pi = pi / pi.sum()
pi[np.abs(pi) < 1e-12] = 0
pi = pi / pi.sum()

stationary_distribution = pi

print("pi:", stationary_distribution)
print("pi P:", stationary_distribution @ P)
```


```python
# Irreducible check
def communication_matrix(P):
    P = np.asarray(P)
    n = P.shape[0]
    reach = P > 1e-12
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i, j] = reach[i, j] or (reach[i, k] and reach[k, j])
    return reach

def is_irreducible(P):
    return bool(communication_matrix(P).all())

irreducible = is_irreducible(P)
print("Reachability:")
print(communication_matrix(P))
print("Irreducible:", irreducible)
```


```python
# Periods and aperiodic check
import math

def state_periods(P, max_power=100):
    P = np.asarray(P)
    n = P.shape[0]
    periods = np.zeros(n, dtype=int)
    power = np.eye(n)

    for t in range(1, max_power + 1):
        power = power @ P
        for i in range(n):
            if power[i, i] > 1e-12:
                periods[i] = t if periods[i] == 0 else math.gcd(periods[i], t)
    return periods

periods = state_periods(P)
aperiodic = bool(np.all(periods == 1))

print("Periods:", periods)
print("Aperiodic:", aperiodic)
```


```python
# Reversibility check: pi_i P_ij = pi_j P_ji
def is_reversible(P, pi):
    return bool(np.allclose(pi[:, None] * P, pi[None, :] * P.T, atol=1e-8))

reversible = is_reversible(P, stationary_distribution)
reversible
```


```python
# Expected hitting time to target_state from all states
def expected_hitting_times(P, target_state):
    P = np.asarray(P)
    n = P.shape[0]
    non_target = [i for i in range(n) if i != target_state]

    A = np.eye(len(non_target))
    b = np.ones(len(non_target))

    for row_idx, i in enumerate(non_target):
        for col_idx, j in enumerate(non_target):
            A[row_idx, col_idx] -= P[i, j]

    h_non_target = np.linalg.solve(A, b)

    h = np.zeros(n)
    for idx, state in enumerate(non_target):
        h[state] = h_non_target[idx]

    return h

target_state = 1
start_state = 0
h = expected_hitting_times(P, target_state)
expected_time = float(h[start_state])

print("Expected hitting times:", h)
print("Expected time from start:", expected_time)
```


```python
# First hitting probability after exactly 2 steps
# Example: start=1, target=0
start = 1
target = 0

prob_first_hit_exact_2 = 0.0
for middle in range(P.shape[0]):
    if middle != target:
        prob_first_hit_exact_2 += P[start, middle] * P[middle, target]

prob_first_hit_exact_2 = float(prob_first_hit_exact_2)
prob_first_hit_exact_2
```

# 3. Linear Regression

Use for salary, abalone, continuous target prediction, MAE/RMSE/R², predicted-vs-true plots.


```python
# Linear regression full template
df = pd.read_csv("data/salaries.csv")  # CHANGE FILE

features = ["work_year", "experience_level", "employment_type", "remote_ratio"]  # CHANGE
target = "salary_in_usd"  # CHANGE

# One-hot encode categorical columns if needed
df_model = pd.get_dummies(df, drop_first=True)

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = float(mean_absolute_error(y_test, pred))
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
r2 = float(r2_score(y_test, pred))

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

coef_table = pd.DataFrame({"feature": X.columns, "coefficient": model.coef_})
coef_table
```


```python
# Predicted vs true scatterplot
plt.figure(figsize=(7, 5))
plt.scatter(pred, y_test, alpha=0.6)

min_val = min(pred.min(), y_test.min())
max_val = max(pred.max(), y_test.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Perfect prediction")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Predicted vs True")
plt.legend()
plt.show()

# Residual histogram
residuals = y_test.values - pred

plt.figure(figsize=(7, 5))
plt.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
plt.xlabel("Residual = true - predicted")
plt.ylabel("Count")
plt.title("Residual histogram")
plt.show()
```


```python
# Predict one new observation example
new_person = pd.DataFrame([{
    "work_year": 2023,
    "experience_level": 1,
    "employment_type": 1,
    "remote_ratio": 0
}])

predicted_value = float(model.predict(new_person)[0])
predicted_value
```

# 4. Logistic Regression / Classification

Use for diabetes, spam, CHD, precision, recall, threshold, cost-sensitive classifier.


```python
# Logistic regression full template
df = pd.read_csv("data/diabetes.csv")  # CHANGE FILE

target = "diabetes"  # CHANGE
features = [col for col in df.columns if col != target]

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42, stratify=y
)

log_model = LogisticRegression(max_iter=2000, C=1.0)
log_model.fit(X_train, y_train)

pred = log_model.predict(X_test)
pred_proba = log_model.predict_proba(X_test)[:, 1]

print("Confusion matrix:")
print(confusion_matrix(y_test, pred))
print("Accuracy:", accuracy_score(y_test, pred))

precision0 = precision_score(y_test, pred, pos_label=0, zero_division=0)
recall0 = recall_score(y_test, pred, pos_label=0, zero_division=0)
precision1 = precision_score(y_test, pred, pos_label=1, zero_division=0)
recall1 = recall_score(y_test, pred, pos_label=1, zero_division=0)

print("Precision 0:", precision0)
print("Recall 0:", recall0)
print("Precision 1:", precision1)
print("Recall 1:", recall1)
```


```python
# Threshold classifier
threshold = 0.5
pred_threshold = (pred_proba >= threshold).astype(int)

confusion_matrix(y_test, pred_threshold)
```


```python
# Cost-sensitive threshold template
# Example costs: FN=300, FP=10, TP=0, TN=0

def cost_at_threshold(model, threshold, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    predictions = (pred_proba >= threshold).astype(int)

    false_negative = (y == 1) & (predictions == 0)
    false_positive = (y == 0) & (predictions == 1)

    cost = 300 * false_negative + 10 * false_positive
    return float(np.mean(cost))

thresholds = np.linspace(0, 1, 101)
costs = np.array([cost_at_threshold(log_model, t, X_test, y_test) for t in thresholds])

best_idx = np.argmin(costs)
optimal_threshold = float(thresholds[best_idx])
optimal_cost = float(costs[best_idx])

print("Optimal threshold:", optimal_threshold)
print("Optimal cost:", optimal_cost)

plt.plot(thresholds, costs)
plt.xlabel("Threshold")
plt.ylabel("Average cost")
plt.title("Cost by threshold")
plt.show()
```

# 5. EDF + DKW Confidence Bands

Use for residual EDF with confidence bands.

Formula:

\[
\epsilon=\sqrt{\frac{\log(2/\delta)}{2n}}
\]


```python
def plot_edf_dkw(values, confidence=0.95, title="EDF with DKW confidence bands"):
    values = np.asarray(values)
    values_sorted = np.sort(values)
    n = len(values_sorted)

    edf = np.arange(1, n + 1) / n
    delta = 1 - confidence
    epsilon = np.sqrt(np.log(2 / delta) / (2 * n))

    lower = np.maximum(edf - epsilon, 0)
    upper = np.minimum(edf + epsilon, 1)

    plt.figure(figsize=(8, 5))
    plt.step(values_sorted, edf, where="post", label="EDF")
    plt.step(values_sorted, lower, where="post", linestyle="--", label="Lower band")
    plt.step(values_sorted, upper, where="post", linestyle="--", label="Upper band")
    plt.xlabel("Value")
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend()
    plt.show()

    return epsilon

# Example:
# eps = plot_edf_dkw(residuals, confidence=0.95, title="Residual EDF")
```

# 6. Hoeffding Confidence Interval

Use for mean, expectation, 0-1 loss, precision/recall, accuracy, cost.

\[
\epsilon=(b-a)\sqrt{\frac{\log(2/\delta)}{2n}}
\]


```python
def hoeffding_interval(mean_estimate, n, a=0, b=1, confidence=0.95):
    delta = 1 - confidence
    epsilon = (b - a) * np.sqrt(np.log(2 / delta) / (2 * n))
    return (
        float(max(a, mean_estimate - epsilon)),
        float(min(b, mean_estimate + epsilon))
    )

# Example for 0-1 loss in [0,1]
loss_mean = 0.2
interval_loss = hoeffding_interval(loss_mean, n=100, a=0, b=1, confidence=0.95)
interval_loss
```


```python
# Hoeffding interval for precision/recall rate
def hoeffding_rate_interval(successes, total, confidence=0.95):
    if total == 0:
        return (0.0, 1.0)
    rate = successes / total
    return hoeffding_interval(rate, total, a=0, b=1, confidence=confidence)

# Example:
# problem_precision_interval = hoeffding_rate_interval(correct_predicted_positive, predicted_positive_count)
```

# 7. Sampling Templates

Use for inversion sampling, accept-reject, Monte Carlo integration.


```python
# Inversion sampling example
# F(x)=e^x-1, 0<x<ln(2)
# U=e^X-1 -> X=log(1+U)

n = 1000
U = np.random.uniform(0, 1, size=n)
samples_inversion = np.log(1 + U)

plt.hist(samples_inversion, bins=30, density=True, edgecolor="black", alpha=0.7)
x_grid = np.linspace(0, np.log(2), 500)
plt.plot(x_grid, np.exp(x_grid), linewidth=2, label="True density e^x")
plt.legend()
plt.show()
```


```python
# General accept-reject sampler
def accept_reject_sampler(n_samples, sample_proposal, proposal_density, target_density, M):
    samples = []
    total_proposed = 0

    while len(samples) < n_samples:
        x = sample_proposal()
        total_proposed += 1

        u = np.random.uniform(0, 1)
        accept_prob = target_density(x) / (M * proposal_density(x))

        if u <= accept_prob:
            samples.append(x)

    return np.array(samples), len(samples) / total_proposed
```


```python
# Accept-reject example for target f(x)=e^x on [0, ln(2)]
def sample_proposal():
    return np.random.uniform(0, np.log(2))

def proposal_density(x):
    return 1 / np.log(2) if 0 <= x <= np.log(2) else 0

def target_density(x):
    return np.exp(x) if 0 <= x <= np.log(2) else 0

M = 2 * np.log(2)

samples_ar, acceptance_rate = accept_reject_sampler(
    1000, sample_proposal, proposal_density, target_density, M
)

print("Acceptance rate:", acceptance_rate)
```


```python
# Rejection sampling for density proportional to x^a(1-x)^b on [0,1]
def beta_shape_rejection(n_samples=10000, a=0.2, b=1.3):
    def h(x):
        return (x ** a) * ((1 - x) ** b)

    x_star = a / (a + b)
    M = h(x_star)

    samples = []
    batch_size = max(1000, n_samples)

    while len(samples) < n_samples:
        x = np.random.uniform(0, 1, size=batch_size)
        u = np.random.uniform(0, 1, size=batch_size)
        accepted = x[u <= h(x) / M]
        samples.extend(accepted.tolist())

    return np.array(samples[:n_samples])

samples_beta = beta_shape_rejection(10000)
plt.hist(samples_beta, bins=50, density=True, edgecolor="black", alpha=0.7)
plt.show()
```


```python
# Monte Carlo integration
# Integral_0^1 f(x) dx = E[f(U)] for U~Uniform(0,1)

n = 100000
U = np.random.uniform(0, 1, size=n)
integral_estimate = float(np.mean(np.sin(U)))
integral_estimate
```

# 8. Poisson Regression

Use for count targets like visits.

Model:

\[
Y|X\sim Poisson(\lambda), \quad \lambda(x)=\exp(\alpha\cdot x+\beta)
\]

Loss to minimize, ignoring constant \(\log(y!)\):

\[
\lambda-y\log(\lambda)
\]

Since \(\log(\lambda)=\eta\), loss is:

\[
\lambda-y\eta
\]


```python
class PoissonRegression(object):
    def __init__(self):
        self.coeffs = None
        self.result = None

    def fit(self, X, Y):
        from scipy import optimize

        X = np.asarray(X)
        Y = np.asarray(Y)

        def loss(coeffs):
            eta = np.dot(X, coeffs[:-1]) + coeffs[-1]
            eta = np.clip(eta, -20, 20)
            lam = np.exp(eta)
            return float(np.mean(lam - Y * eta))

        initial_arguments = np.zeros(shape=X.shape[1] + 1)
        self.result = optimize.minimize(loss, initial_arguments, method="cg")
        self.coeffs = self.result.x

    def predict(self, X):
        if self.coeffs is None:
            raise ValueError("Model not fitted yet.")
        eta = np.dot(X, self.coeffs[:-1]) + self.coeffs[-1]
        eta = np.clip(eta, -20, 20)
        return np.exp(eta)
```


```python
# Poisson regression usage template
df = pd.read_csv("data/visits_clean.csv")  # CHANGE IF NEEDED

target = "ofp"

# Avoid leakage: do not use other healthcare usage outcome variables
exclude_cols = ["ofp", "ofnp", "opp", "opnp", "emr", "hosp"]
features = [col for col in df.columns if col not in exclude_cols]

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

poisson_model = PoissonRegression()
poisson_model.fit(X_train, y_train)

print(poisson_model.result)

pred = poisson_model.predict(X_test)

mae_poisson = float(mean_absolute_error(y_test, pred))
naive_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)
mae_naive = float(mean_absolute_error(y_test, naive_pred))

print("Poisson MAE:", mae_poisson)
print("Naive MAE:", mae_naive)
```

# 9. SVD / Dimensionality Reduction / Anomaly Detection

Use for SVD, explained variance, low-rank approximation, reconstruction error, outliers.


```python
# SVD template
X = pd.read_csv("data/SVD.csv").values.astype(float)  # CHANGE FILE

U, D, Vt = np.linalg.svd(X, full_matrices=False)

problem_U = U
problem_D = D
problem_V = Vt.T

first_left_singular_vector = U[:, 0].flatten()
first_right_singular_vector = Vt[0, :].flatten()

print("X:", X.shape)
print("U:", U.shape)
print("D:", D.shape)
print("V:", problem_V.shape)
```


```python
# Explained variance
explained_variance = np.cumsum(D**2) / np.sum(D**2)
num_components_90 = int(np.argmax(explained_variance >= 0.90) + 1)

print("Explained variance:", explained_variance)
print("Components for 90%:", num_components_90)

plt.plot(np.arange(1, len(D)+1), explained_variance, marker="o")
plt.axhline(0.90, linestyle="--", label="90%")
plt.xlabel("Number of components")
plt.ylabel("Explained variance")
plt.legend()
plt.show()
```


```python
# Low-rank approximation and reconstruction error
k = num_components_90

X_approx = U[:, :k] @ np.diag(D[:k]) @ Vt[:k, :]
reconstruction_error = np.linalg.norm(X - X_approx, axis=1)

# Top 10 outliers by reconstruction error
top10_idx = np.argsort(reconstruction_error)[-10:]
outliers_top10 = X[top10_idx]

print("Approximation shape:", X_approx.shape)
print("Outliers shape:", outliers_top10.shape)

plot_edf_dkw(reconstruction_error, confidence=0.95, title="EDF of reconstruction error")
```

# 10. Feature Selection / One-Hot Encoding / Leakage

Use all columns except target, but avoid leakage columns.


```python
# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

target = "target_column"  # CHANGE
features = [col for col in df_encoded.columns if col != target]

X = df_encoded[features]
y = df_encoded[target]
```


```python
# Avoid leakage example
target = "ofp"
leakage_columns = ["ofp", "ofnp", "opp", "opnp", "emr", "hosp"]

features_no_leakage = [col for col in df.columns if col not in leakage_columns]
features_no_leakage
```

# 11. Common Free-Text Answers

## Linear regression performance

I used MAE because it is easy to interpret: it gives the average absolute error in the target units. A lower MAE is better. The predicted-vs-true scatter plot should ideally lie close to the diagonal line. If points are widely scattered, the model is weak or missing important predictors.

## Precision and recall

Precision for class 1 means: among all observations predicted as class 1, the fraction that were actually class 1. Recall for class 1 means: among all actual class 1 observations, the fraction correctly found by the model.

## DKW confidence band

The DKW band gives a uniform confidence band around the empirical distribution function. With 95% confidence, the true CDF lies within the band for every value. It can be used to estimate uncertainty in residual quantiles and understand the distribution of model errors.

## Hoeffding interval

Hoeffding's inequality gives a confidence interval for the expectation of a bounded random variable. The interval gets tighter as sample size increases.

## Irreducible Markov chain

The chain is irreducible if every state can reach every other state with positive probability in some number of steps. If there is an absorbing state or a closed class that cannot be left, the chain is not irreducible.

## Stationary distribution

The stationary distribution satisfies \(\pi P=\pi\) and \(\sum_i\pi_i=1\). It represents the long-run proportion of time spent in each state.

## Reversibility

A chain is reversible if detailed balance holds: \(\pi_iP_{ij}=\pi_jP_{ji}\) for all states.

## SVD approximation

The low-rank SVD approximation keeps only the largest singular values. It projects the data into a lower-dimensional subspace capturing most of the variance. Large reconstruction error can indicate anomalies.

## Poisson regression

Because the target is a count, Poisson regression is reasonable. The exponential link keeps \(\lambda\) positive. I used MAE because it gives the average absolute error in number of visits, and compared it to a naive mean predictor.

# 12. Final Exam Checklist

Before submitting:

- variable names match the question
- arrays have the requested shapes
- intervals are tuples/lists of length 2
- used `random_state=42` where asked
- wrote free-text answers
- checked Markov rows sum to 1
- checked model `.fit()` ran
- avoided leakage features


```python
def check(name, value):
    print(name, type(value), getattr(value, "shape", ""))

# Example:
# check("X_train", X_train)
# check("y_train", y_train)
```
