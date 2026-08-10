# Solved Exam Notebook — 15th of August 2023

Course: **1MS041 Introduction to Data Science / Introduktion till dataanalys**

This notebook contains the **whole questions**, followed by the **solved code answers with all steps, comments, outputs, plots, and explanations**.

> Insert your anonymous exam ID below.


```python
examID = "XXX"
```

## Problem 1 — Full Question

Maximum Points = 14

A courier company operates a fleet of delivery trucks that make deliveries to different parts of the city. The trucks are equipped with GPS tracking devices that record the location of each truck at regular intervals. The locations are divided into three regions: downtown, the suburbs, and the countryside, however there is always the possibility the truck breaks down and it goes to the workshop.

The following table shows the probabilities of a truck transitioning between these regions at each time step:

| Current region | To downtown | To suburbs | To countryside | To Workshop |
|---|---:|---:|---:|---:|
| Downtown | 0.3 | 0.7 | 0 | 0 |
| Suburbs | 0.2 | 0.5 | 0.3 | 0 |
| Countryside | 0 | 0 | 0.5 | 0.5 |
| Workshop | 0 | 0 | 0 | 1 |

1. **[2p]** If a truck is currently in the downtown, what is the probability that it will be in the countryside region after 10 time steps?

2. **[2p]** If a truck is currently in the downtown, what is the probability that it will be in the countryside region the first time after three time steps or more?

3. **[3p]** Is this Markov chain irreducible? Explain your answer.

4. **[3p]** What is the stationary distribution? Furthermore is it reversible? Explain your answer.

5. **[4p] Advanced question:** What is the expected number of steps it takes starting from the Downtown region to first reach the Workshop? Hint: to get within 1 decimal point, it is enough to compute the probabilities for hitting times below 50. Motivate your answer in detail. You could also solve this question by simulation, but this gives you a maximum of 2p.

### Problem 1 — Solution idea

Use state order:

\[
(\text{Downtown},\text{Suburbs},\text{Countryside},\text{Workshop})
\]

The transition matrix is

\[
P=
\begin{pmatrix}
0.3&0.7&0&0\\
0.2&0.5&0.3&0\\
0&0&0.5&0.5\\
0&0&0&1
\end{pmatrix}.
\]

Part 1 is entry \((P^{10})_{\text{Downtown},\text{Countryside}}\).

For the expected hitting time to Workshop, solve linear equations:

\[
E_W=0,
\]

\[
E_D=1+0.3E_D+0.7E_S,
\]

\[
E_S=1+0.2E_D+0.5E_S+0.3E_C,
\]

\[
E_C=1+0.5E_C.
\]


```python
# Problem 1 imports
import numpy as np

# State order: Downtown, Suburbs, Countryside, Workshop
P = np.array([
    [0.3, 0.7, 0.0, 0.0],
    [0.2, 0.5, 0.3, 0.0],
    [0.0, 0.0, 0.5, 0.5],
    [0.0, 0.0, 0.0, 1.0]
])

P
```


```python
# Part 1
import numpy as np

# State order: Downtown, Suburbs, Countryside, Workshop
P = np.array([
    [0.3, 0.7, 0.0, 0.0],
    [0.2, 0.5, 0.3, 0.0],
    [0.0, 0.0, 0.5, 0.5],
    [0.0, 0.0, 0.0, 1.0]
])
# Probability of being in Countryside after 10 steps when starting from Downtown.
P10 = np.linalg.matrix_power(P, 10)
problem1_p1 = float(P10[0, 2])

problem1_p1
```




    0.08487353489999998




```python
# Part 2
# First hitting time of Countryside starting from Downtown.
# It is impossible to hit Countryside in 1 step.
# The first possible time is 2: Downtown -> Suburbs -> Countryside, probability 0.7*0.3.
#
# "first time after three time steps or more" means P(T_C >= 3 and eventually hit C).
# From Downtown/Suburbs, hitting Countryside eventually has probability 1.
# Therefore P(T_C >= 3) = 1 - P(T_C = 2).

p_first_hit_c_at_2 = P[0, 1] * P[1, 2]
problem1_p2 = float(1 - p_first_hit_c_at_2)

problem1_p2
```




    0.79



### Problem 1 Part 3 — Free-text answer

This Markov chain is **not irreducible**.

Reason: once the truck reaches the **Workshop**, it stays there forever because the Workshop row is \([0,0,0,1]\). From Workshop it is impossible to go back to Downtown, Suburbs, or Countryside. Therefore not every state can communicate with every other state, so the chain is reducible.


```python
# Part 3
problem1_irreducible = False
problem1_irreducible
```


```python
# Part 4
# The unique stationary distribution places all mass on Workshop, because Workshop is absorbing.
problem1_stationary = np.array([0.0, 0.0, 0.0, 1.0])

# Check stationarity: pi P = pi
stationary_check = problem1_stationary @ P

# Reversibility check using detailed balance pi_i P_ij = pi_j P_ji.
# With all mass on Workshop and no exit from Workshop, detailed balance holds.
problem1_reversible = bool(np.allclose(problem1_stationary[:, None] * P, problem1_stationary[None, :] * P.T))

problem1_stationary, stationary_check, problem1_reversible
```

### Problem 1 Part 4 — Free-text answer

The stationary distribution is

\[
\pi=(0,0,0,1).
\]

This means that in the long run the truck is in the Workshop with probability 1, because Workshop is absorbing.

The chain is reversible with respect to this stationary distribution because the detailed balance equations

\[
\pi_iP_{ij}=\pi_jP_{ji}
\]

hold. Since all non-Workshop states have stationary probability 0, both sides are 0 for transitions involving those states; and Workshop stays in Workshop with probability 1.


```python
# Part 5
# Solve expected hitting time equations exactly using linear algebra.

# Unknowns: E_D, E_S, E_C. E_W = 0.
# Equations:
# E_D = 1 + 0.3E_D + 0.7E_S
# E_S = 1 + 0.2E_D + 0.5E_S + 0.3E_C
# E_C = 1 + 0.5E_C
#
# Rearranged:
# 0.7E_D - 0.7E_S + 0E_C = 1
# -0.2E_D + 0.5E_S - 0.3E_C = 1
# 0E_D + 0E_S + 0.5E_C = 1

A = np.array([
    [0.7, -0.7, 0.0],
    [-0.2, 0.5, -0.3],
    [0.0, 0.0, 0.5]
])

b = np.array([1.0, 1.0, 1.0])

E_D, E_S, E_C = np.linalg.solve(A, b)

problem1_ET = float(E_D)
problem1_ET
```




    7.714285714285714



### Problem 1 Part 5 — Free-text answer

Let \(E_D,E_S,E_C,E_W\) be the expected number of steps to first reach Workshop starting from Downtown, Suburbs, Countryside, and Workshop.

Since Workshop is already reached,

\[
E_W=0.
\]

From the Markov property,

\[
E_D=1+0.3E_D+0.7E_S,
\]

\[
E_S=1+0.2E_D+0.5E_S+0.3E_C,
\]

\[
E_C=1+0.5E_C+0.5E_W.
\]

Solving this system gives the expected time from Downtown:

\[
E_D \approx 7.714.
\]

So the expected number of steps is stored in `problem1_ET`.

## Problem 2 — Full Question

Maximum Points = 13

You are given a **Data Science Salaries** dataset found in `data/salaries.csv`, which contains employment information of data scientists up to 2023 and the salary obtained. Your task is to train a linear regression model to predict the salary of a data scientist based on the employment information.

To evaluate your model, you will split the dataset into a training set and a testing set. You will use the training set to train your model, and the testing set to evaluate its performance.

Experience level:

- 0 = Entry Level
- 1 = Mid Level
- 2 = Senior Level
- 3 = Executive Level

Employment type:

- 0 = Part Time
- 1 = Full Time
- 2 = Contractor
- 3 = Freelancer

1. **[1p]** Load the data into a pandas dataframe `problem2_df`. Based on the column names, figure out what are the features and the target and fill in the answer in the correct cell below.

2. **[1p]** Split the data into train and test.

3. **[1p]** Train the model.

4. **[4p]** Come up with a reasonable metric and compute it. Provide plots that show the performance of the model. Reason about the performance.

5. **[3p]** Predict the 2023 salary of a data scientist that works full time (1) at mid employment level (1) with 0 remote ratio. Then, looking at the output of `problem2_model.coef_`, which are the coefficients of the linear model, would a higher remote ratio result in a higher predicted salary or vice versa?

6. **[3p] Advanced question:** On the test set, plot the empirical distribution function of the residual with confidence bands using the DKW inequality and 95% confidence. What does the confidence band tell us? What can the confidence band be used for?

### Problem 2 — Solution idea

We use a linear regression model to predict `salary_in_usd` from numeric employment features:

- `work_year`
- `experience_level`
- `employment_type`
- `remote_ratio`

The main metric used here is **Mean Absolute Error (MAE)**, because it gives the average prediction error in dollars. We also compute RMSE and \(R^2\).

For the DKW confidence band around the empirical distribution function,

\[
\epsilon=\sqrt{\frac{\log(2/\delta)}{2n}},
\]

where \(\delta=0.05\) for 95% confidence.


```python
# Problem 2 imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```


```python
# Part 1
# Load the salaries dataset.
problem2_df = pd.read_csv("data/salaries.csv")

problem2_df.head()
```


```python
# Part 1
# Fill in the features and target.
# These are the columns described in the question.

problem2_features = ["work_year", "experience_level", "employment_type", "remote_ratio"]
problem2_target = "salary_in_usd"

# If employment_type is accidentally stored as text, convert it.
# This keeps the solution robust.
if problem2_df["employment_type"].dtype == "object":
    emp_map = {
        "PT": 0, "Part-time": 0, "Part Time": 0,
        "FT": 1, "Full-time": 1, "Full Time": 1,
        "CT": 2, "Contract": 2, "Contractor": 2,
        "FL": 3, "Freelance": 3, "Freelancer": 3
    }
    problem2_df["employment_type"] = problem2_df["employment_type"].map(emp_map).fillna(problem2_df["employment_type"]).astype(float)

problem2_X = problem2_df[problem2_features]
problem2_y = problem2_df[problem2_target]

problem2_features, problem2_target
```


```python
# Part 2
# Split the data into train and test using train_test_split.
# Keep train size 0.8 and use random_state=42.

problem2_X_train, problem2_X_test, problem2_y_train, problem2_y_test = train_test_split(
    problem2_X,
    problem2_y,
    train_size=0.8,
    random_state=42
)

problem2_X_train.shape, problem2_X_test.shape, problem2_y_train.shape, problem2_y_test.shape
```


```python
# Part 3
# Initialize and train linear regression model.
problem2_model = LinearRegression()
problem2_model.fit(problem2_X_train, problem2_y_train)

problem2_model
```

### Problem 2 Part 4 — Free-text answer

A reasonable metric is **Mean Absolute Error (MAE)** because it tells us the average absolute salary prediction error in USD. A lower MAE is better.

I also compute RMSE, which punishes large mistakes more strongly, and \(R^2\), which measures how much variation in salaries is explained by the model.

The predicted-vs-true scatter plot should ideally lie close to the diagonal line \(y=x\). If points are widely scattered, the model is weak or missing important variables.


```python
# Part 4
# Diagnose the model using MAE, RMSE, R^2 and plots.

problem2_y_pred = problem2_model.predict(problem2_X_test)

problem2_mae = float(mean_absolute_error(problem2_y_test, problem2_y_pred))
problem2_rmse = float(np.sqrt(mean_squared_error(problem2_y_test, problem2_y_pred)))
problem2_r2 = float(r2_score(problem2_y_test, problem2_y_pred))

print("MAE:", problem2_mae)
print("RMSE:", problem2_rmse)
print("R^2:", problem2_r2)

# Plot 1: predicted vs true
plt.figure(figsize=(7, 5))
plt.scatter(problem2_y_pred, problem2_y_test, alpha=0.6)
min_val = min(problem2_y_pred.min(), problem2_y_test.min())
max_val = max(problem2_y_pred.max(), problem2_y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Perfect prediction")
plt.xlabel("Predicted salary")
plt.ylabel("True salary")
plt.title("Predicted vs true salary")
plt.legend()
plt.show()

# Plot 2: residual histogram
problem2_residuals = problem2_y_test.values - problem2_y_pred

plt.figure(figsize=(7, 5))
plt.hist(problem2_residuals, bins=30, edgecolor="black", alpha=0.7)
plt.xlabel("Residual = true - predicted")
plt.ylabel("Count")
plt.title("Residual histogram")
plt.show()
```

### Problem 2 Part 5 — Free-text answer

To predict the 2023 salary for a full-time, mid-level, non-remote data scientist, use:

\[
\text{work_year}=2023,\quad
\text{experience_level}=1,\quad
\text{employment_type}=1,\quad
\text{remote_ratio}=0.
\]

The coefficient of `remote_ratio` in `problem2_model.coef_` tells us whether salary increases or decreases when remote ratio increases, holding other features fixed.

- If the coefficient is positive, higher remote ratio increases predicted salary.
- If the coefficient is negative, higher remote ratio decreases predicted salary.


```python
# Part 5
# Predict 2023 salary of data scientist:
# full time employment_type=1, mid experience_level=1, remote_ratio=0

new_person = pd.DataFrame([{
    "work_year": 2023,
    "experience_level": 1,
    "employment_type": 1,
    "remote_ratio": 0
}])

problem2_predicted_salary = float(problem2_model.predict(new_person)[0])

# Coefficients
coef_table = pd.DataFrame({
    "feature": problem2_features,
    "coefficient": problem2_model.coef_
})

remote_coef = float(coef_table.loc[coef_table["feature"] == "remote_ratio", "coefficient"].iloc[0])

print("Predicted salary:", problem2_predicted_salary)
print(coef_table)

if remote_coef > 0:
    print("A higher remote ratio increases the predicted salary, according to this model.")
elif remote_coef < 0:
    print("A higher remote ratio decreases the predicted salary, according to this model.")
else:
    print("Remote ratio has no effect in this fitted linear model.")
```

### Problem 2 Part 6 — Free-text answer

The empirical distribution function (EDF) of the residuals shows the distribution of prediction errors.

The DKW confidence band gives a uniform confidence interval around the whole EDF. With 95% confidence, the true CDF of the residuals lies inside the band for all residual values.

This can be used to understand the uncertainty in the error distribution, check whether residuals are centered near 0, and estimate quantiles of the prediction error.


```python
# Part 6
# EDF of residuals with DKW confidence bands, 95% confidence.

residuals_sorted = np.sort(problem2_residuals)
n = len(residuals_sorted)
edf = np.arange(1, n + 1) / n

delta = 0.05
epsilon = np.sqrt(np.log(2 / delta) / (2 * n))

lower_band = np.maximum(edf - epsilon, 0)
upper_band = np.minimum(edf + epsilon, 1)

plt.figure(figsize=(8, 5))
plt.step(residuals_sorted, edf, where="post", label="Empirical CDF of residuals")
plt.step(residuals_sorted, lower_band, where="post", linestyle="--", label="Lower 95% DKW band")
plt.step(residuals_sorted, upper_band, where="post", linestyle="--", label="Upper 95% DKW band")
plt.xlabel("Residual = true - predicted")
plt.ylabel("CDF")
plt.title("EDF of residuals with 95% DKW confidence bands")
plt.legend()
plt.show()

epsilon
```

## Problem 3 — Full Question

Maximum Points = 13

### Random variable generation

1. **[4p]** Using inversion sampling, construct 1000 samples from the below distribution:

\[
F[x]=
\begin{cases}
0, & x\le 0\\
e^x-1, & 0<x<\ln(2)\\
1, & x\ge \ln(2)
\end{cases}
\]

2. **[2p]** Use the above 1000 samples to estimate the mean and variance.

3. **[4p]** Using the Accept-Reject sampler (Algorithm 1 in TFDS notes), construct 1000 samples from the same distribution. What proposal distribution did you choose and why? What proportion of samples were accepted?

4. **[3p]** Explain if it is possible to sample from the density

\[
f(x)=Ce^{-(x^2-2)^2}
\]

using the Accept-Reject sampler with sampling density given by the Gaussian. Here \(C\) is a constant to make sure that \(f\) is a density, between roughly 1.34 and 1.35.

### Problem 3 — Solution idea

For Part 1, solve

\[
u=e^x-1
\]

so

\[
x=\log(1+u).
\]

For accept-reject sampling, the density is

\[
f(x)=e^x,\quad 0<x<\ln(2).
\]

Use a Uniform proposal on \([0,\ln(2)]\). Since the proposal density is \(g(x)=1/\ln(2)\), the ratio is

\[
\frac{f(x)}{g(x)}=e^x\ln(2).
\]

The maximum occurs at \(x=\ln(2)\), so

\[
M=2\ln(2).
\]

The acceptance probability becomes

\[
\frac{f(x)}{Mg(x)}=\frac{e^x}{2}.
\]


```python
# Problem 3 imports
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
```


```python
# Part 1
# Inversion sampling:
# U = e^X - 1 => X = log(1+U)

U = np.random.uniform(0, 1, size=1000)
problem3_samples = np.log(1 + U)

plt.figure(figsize=(7, 5))
plt.hist(problem3_samples, bins=30, density=True, edgecolor="black", alpha=0.7)
x_grid = np.linspace(0, np.log(2), 500)
plt.plot(x_grid, np.exp(x_grid), linewidth=2, label="True density e^x")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Problem 3 inversion samples")
plt.legend()
plt.show()

problem3_samples[:10], problem3_samples.shape
```


```python
# Part 2
problem3_mean = float(np.mean(problem3_samples))
problem3_variance = float(np.var(problem3_samples, ddof=1))

problem3_mean, problem3_variance
```


```python
# Part 3
# Accept-Reject sampler using Uniform(0, ln(2)) as proposal.

def problem3_accept_reject_sampler(n_samples=1000):
    accepted = []
    total_proposed = 0

    while len(accepted) < n_samples:
        # proposal: X ~ Uniform(0, ln(2))
        x = np.random.uniform(0, np.log(2))
        total_proposed += 1

        # acceptance probability = e^x / 2
        u = np.random.uniform(0, 1)
        if u <= np.exp(x) / 2:
            accepted.append(x)

    return np.array(accepted), len(accepted) / total_proposed

problem3_samples_accept_reject, problem3_acceptance_rate = problem3_accept_reject_sampler(1000)

problem3_samples_accept_reject[:10], problem3_acceptance_rate
```

### Problem 3 Part 3 — Free-text answer

I chose the proposal distribution

\[
X\sim \mathrm{Uniform}(0,\ln 2)
\]

because the target density only has support on \([0,\ln 2]\). This means the proposal never wastes samples outside the support.

The target density is

\[
f(x)=e^x,\quad 0<x<\ln 2.
\]

The proposal density is

\[
g(x)=\frac{1}{\ln 2}.
\]

So

\[
\frac{f(x)}{g(x)}=e^x\ln 2.
\]

This is maximized at \(x=\ln 2\), giving

\[
M=2\ln 2.
\]

Therefore the acceptance probability is

\[
\frac{f(x)}{Mg(x)}=\frac{e^x}{2}.
\]

The observed acceptance rate is stored in `problem3_acceptance_rate`.

### Problem 3 Part 4 — Free-text answer

Yes, it is possible to use a Gaussian proposal distribution for

\[
f(x)=Ce^{-(x^2-2)^2}.
\]

For accept-reject sampling, we need a constant \(M<\infty\) such that

\[
f(x)\le Mg(x)
\]

for all \(x\), where \(g(x)\) is the Gaussian proposal density.

This works because the target density has tails like

\[
e^{-x^4}
\]

for large \(|x|\), while a Gaussian has tails like

\[
e^{-x^2/2}.
\]

The target therefore decays faster than the Gaussian in the tails, so the ratio \(f(x)/g(x)\) is bounded. Since the ratio is bounded, a finite \(M\) exists, and accept-reject sampling is possible. In practice, \(M\) can be found numerically by maximizing \(f(x)/g(x)\).


```python
# Optional local checks
assert isinstance(problem3_samples, np.ndarray)
assert problem3_samples.shape == (1000,)
assert isinstance(problem3_mean, float)
assert isinstance(problem3_variance, float)
assert isinstance(problem3_samples_accept_reject, np.ndarray)
assert problem3_samples_accept_reject.shape == (1000,)
assert isinstance(problem3_acceptance_rate, float)

print("Problem 3 variables have the expected formats.")
```
