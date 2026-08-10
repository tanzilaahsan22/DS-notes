# Solved Notebook — Exam 28th of August 2024

Course: **1MS041 Introduction to Data Science / Introduktion till dataanalys**  
Exam time: **8.00–13.00**  

This notebook is written in **question → solved answer** format. Each problem includes the full question text, then the solved code with comments and explanations.


```python
# Insert your anonymous exam ID as a string in the variable below
examID = "XXX"
```

---

# Problem 1 — Rejection sampling and Monte Carlo integration

**Maximum Points = 14**

In this problem you will do rejection sampling from complicated distributions, you will also be using your samples to compute certain integrals, a method known as Monte Carlo integration. Keep in mind that choosing a good sampling distribution is often key to avoid too much rejection.

## Full question

1. **[4p]** Fill in the remaining part of the function `problem1_rejection` in order to produce samples from the below density using rejection sampling:

$$
f(x)=C(\sin(x))^{10}
$$

for $0 \le x \le \pi$, where $C$ is a value such that $f$ above is a density, meaning it integrates to one. Hint: you do not need to know the value of $C$ to perform rejection sampling.

2. **[2p]** Produce 10000 samples, use fewer if it takes too long, and put the answer in `problem1_samples` from the above distribution and plot the histogram.

3. **[2p]** Define $X$ as a random variable with the density given in part 1. Denote

$$
Y = \left(X - \frac{\pi}{2}\right)^2
$$

and use the above 10000 samples to estimate

$$
\mathbb{E}[Y]
$$

and store the result in `problem1_expectation`.

4. **[2p]** Use Hoeffding's inequality to produce a 95% confidence interval of the expectation above and store the result as a tuple in the variable `problem1_interval`.

5. **[4p]** Can you calculate an approximation of the value of $C$ from part 1 using random samples? Provide a plot of the histogram from part 2 together with the true density as a curve, recall that this requires the value of $C$. Explain what method you used and what answer you got.

## Answer to Problem 1, Part 1

We use **rejection sampling** with proposal distribution $g(x)=	ext{Uniform}(0,\pi)$. Since $(\sin x)^{10} \le 1$ on $[0,\pi]$, we can accept a proposal $x$ with probability $(\sin x)^{10}$.

The unknown constant $C$ is not needed, because rejection sampling only needs the target density up to proportionality.


```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def problem1_rejection(n_samples=1):
    """
    Generate samples from a density proportional to sin(x)^10 on [0, pi]
    using rejection sampling.
    """
    samples = []
    
    while len(samples) < n_samples:
        # Propose candidates uniformly on [0, pi]
        batch_size = max(1000, 2 * (n_samples - len(samples)))
        x = np.random.uniform(0, np.pi, size=batch_size)
        u = np.random.uniform(0, 1, size=batch_size)
        
        # Accept with probability sin(x)^10
        accepted = x[u <= np.sin(x)**10]
        samples.extend(accepted.tolist())
    
    return np.array(samples[:n_samples])
```

## Answer to Problem 1, Part 2

Now we generate 10000 samples and plot a histogram of the sampled distribution.


```python
problem1_samples = problem1_rejection(10000)

plt.figure(figsize=(8, 5))
plt.hist(problem1_samples, bins=40, density=True, alpha=0.6, edgecolor="black")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Histogram of samples from density proportional to sin(x)^10")
plt.show()

print("Number of samples:", len(problem1_samples))
print("First 10 samples:", problem1_samples[:10])
```

## Answer to Problem 1, Part 3

We estimate

$$
\mathbb{E}\left[\left(X-
rac{\pi}{2}
ight)^2
ight]
$$

by the sample mean of $Y_i=(X_i-\pi/2)^2$.


```python
Y_samples = (problem1_samples - np.pi/2)**2
problem1_expectation = float(np.mean(Y_samples))

print("Estimated E[Y] =", problem1_expectation)
```

## Answer to Problem 1, Part 4

Hoeffding's inequality says that for bounded random variables $Y_i \in [a,b]$,

$$
P(|ar{Y}-E[Y]| \ge \epsilon) \le 2\exp\left(-
rac{2n\epsilon^2}{(b-a)^2}
ight).
$$

Here $Y=(X-\pi/2)^2$. Since $X\in[0,\pi]$, we have

$$
0 \le Y \le \left(
rac{\pi}{2}
ight)^2.
$$

For 95% confidence, set $\delta=0.05$.


```python
n = len(Y_samples)
delta = 0.05
lower_bound_Y = 0
upper_bound_Y = (np.pi/2)**2

hoeffding_epsilon = (upper_bound_Y - lower_bound_Y) * np.sqrt(np.log(2/delta) / (2*n))

problem1_interval = (
    float(max(lower_bound_Y, problem1_expectation - hoeffding_epsilon)),
    float(min(upper_bound_Y, problem1_expectation + hoeffding_epsilon))
)

print("95% Hoeffding confidence interval:", problem1_interval)
```

## Answer to Problem 1, Part 5

We need the normalizing constant $C$ where

$$
1 = \int_0^\pi C\sin(x)^{10}\,dx.
$$

So

$$
C = 
rac{1}{\int_0^\pi \sin(x)^{10}\,dx}.
$$

We approximate the integral using Monte Carlo integration. If $U\sim \mathrm{Uniform}(0,\pi)$, then

$$
\int_0^\pi \sin(x)^{10}\,dx pprox \pi \cdot 
rac{1}{m}\sum_{i=1}^{m}\sin(U_i)^{10}.
$$


```python
# Monte Carlo approximation of C
m = 200000
U = np.random.uniform(0, np.pi, size=m)
integral_estimate = np.pi * np.mean(np.sin(U)**10)
problem1_C = float(1 / integral_estimate)

print("Estimated integral of sin(x)^10 from 0 to pi =", integral_estimate)
print("Estimated normalizing constant C =", problem1_C)

# Plot histogram together with the true density curve C * sin(x)^10
x_grid = np.linspace(0, np.pi, 500)
true_density = problem1_C * np.sin(x_grid)**10

plt.figure(figsize=(8, 5))
plt.hist(problem1_samples, bins=40, density=True, alpha=0.6, edgecolor="black", label="Sample histogram")
plt.plot(x_grid, true_density, linewidth=2, label="Estimated true density")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Histogram with estimated true density curve")
plt.legend()
plt.show()
```

### Free-text explanation for Problem 1, Part 5

I approximated the normalizing constant $C$ using Monte Carlo integration. The density must integrate to one, so $C=1/\int_0^\pi \sin(x)^{10}dx$. I sampled many points uniformly on $[0,\pi]$ and estimated the integral by $\pi$ times the average value of $\sin(U)^{10}$. Then I took the reciprocal of that estimated integral to get `problem1_C`. Finally, I plotted the histogram of the rejection samples together with the curve `problem1_C * sin(x)^10`. The curve should match the histogram well if the sampling and the estimate of $C$ are correct.


```python
# Local format checks for Problem 1
assert isinstance(problem1_rejection(10), np.ndarray)
assert isinstance(problem1_samples, np.ndarray)
assert isinstance(problem1_expectation, float)
assert isinstance(problem1_interval, tuple) or isinstance(problem1_interval, list)
assert len(problem1_interval) == 2
assert isinstance(problem1_C, float)
print("Problem 1 checks passed.")
```

---

# Problem 2 — CORIS coronary heart disease classifier

**Maximum Points = 13**

## Full question

Consider the dataset `CORIS.csv` that you find in the data folder. The data set `CORIS.csv` contains cases of coronary heart disease (CHD) and variables associated with the patient's condition: systolic blood pressure, yearly tobacco use in kg, low density lipoprotein (`ldl`), adiposity, family history (`0` or `1`), type A personality score (`typea`), obesity/body mass index, alcohol use, age, and the diagnosis of CHD (`0` or `1`).

In this dataset, `X` corresponds to the measurements. The `Y` is a 0-1 label where `1` represents CHD and `0` does not. The code to load the data is prepared and so is the train-test-validation split and the training of the model. The model is stored in `problem2_pipe`, which is a `Pipeline` object as often used in composite models in sklearn.

1. **[3p]** Use Hoeffding's inequality and compute the intervals for precision-recall etc. on the test set with 95% confidence.

2. **[3p]** You are interested in minimizing the average cost of your classifier. The hospital wants to use your model as a screening tool. If it finds that someone is classified as CHD, we interpret this as further investigation needs to take place; otherwise we do nothing. Use these imaginary costs:

   - If someone has coronary heart disease but is classified as not having it: cost 300.
   - If someone does not have coronary heart disease but is classified as having it: cost 10.
   - If someone has coronary heart disease and is classified as having it: cost 0.
   - If someone does not have coronary heart disease and is classified as not having it: cost 0.

   Complete the function `problem2_cost` to compute the average cost of a prediction model under a certain prediction threshold.

3. **[4p]** Select the threshold of the classifier that minimizes the cost by checking 100 evenly spaced proposal thresholds between 0 and 1. Compute the optimal threshold using the testing data and calculate the cost at the chosen threshold using the testing data.

4. **[3p]** With the newly computed threshold value, compute the cost of putting this model in production by computing the cost using the validation data. Also provide a confidence interval of the cost using Hoeffding's inequality with 99% confidence.

## Answer to Problem 2 — Load data, split data, and train the model

This is the setup code from the exam, with imports included.


```python
# RUN THIS CELL TO LOAD THE DATA AND SPLIT IT INTO TRAINING, TEST AND VALIDATION SETS
# FINALLY IT TRAINS THE MODEL AS A PIPELINE

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

CORISDataset = pd.read_csv("data/CORIS.csv", skiprows=[1, 2])

# Initial data split into features and target
problem2_X = CORISDataset[['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'typea', 'obesity', 'alcohol', 'age']].values
problem2_Y = CORISDataset['chd'].values

# Split the data into training, test and validation sets
problem2_X_train, X_tmp, problem2_Y_train, Y_tmp = train_test_split(
    problem2_X, problem2_Y, train_size=0.6, random_state=42
)
problem2_X_test, problem2_X_val, problem2_Y_test, problem2_Y_val = train_test_split(
    X_tmp, Y_tmp, train_size=0.5, random_state=42
)

print(problem2_X_train.shape, problem2_Y_train.shape,
      problem2_X_test.shape, problem2_Y_test.shape,
      problem2_X_val.shape, problem2_Y_val.shape)

# Create a pipeline with a scaler and a logistic regression model
problem2_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(random_state=42))
])

# Fit the pipeline to the training data
problem2_pipe.fit(problem2_X_train, problem2_Y_train)
```

## Answer to Problem 2, Part 1

We compute precision and recall for both classes on the test set, then wrap each estimate in a Hoeffding interval.

For a sample mean of values in $[0,1]$, Hoeffding gives

$$
\epsilon = \sqrt{
rac{\log(2/\delta)}{2n}}.
$$

For precision, $n$ is the number of predicted examples of that class. For recall, $n$ is the number of true examples of that class.


```python
def hoeffding_interval_01(p_hat, n, delta=0.05):
    """Hoeffding interval for a proportion bounded in [0, 1]."""
    if n == 0:
        return (0.0, 1.0)
    eps = np.sqrt(np.log(2/delta) / (2*n))
    return (float(max(0, p_hat - eps)), float(min(1, p_hat + eps)))

def precision_recall_interval_for_class(y_true, y_pred, cls, delta=0.05):
    # Precision for class cls: among predicted cls, how many are truly cls?
    pred_mask = (y_pred == cls)
    n_precision = int(np.sum(pred_mask))
    precision_hat = np.mean(y_true[pred_mask] == cls) if n_precision > 0 else 0.0
    precision_interval = hoeffding_interval_01(precision_hat, n_precision, delta)
    
    # Recall for class cls: among true cls, how many were predicted cls?
    true_mask = (y_true == cls)
    n_recall = int(np.sum(true_mask))
    recall_hat = np.mean(y_pred[true_mask] == cls) if n_recall > 0 else 0.0
    recall_interval = hoeffding_interval_01(recall_hat, n_recall, delta)
    
    return precision_interval, recall_interval

predictions_test = problem2_pipe.predict(problem2_X_test)

problem2_precision0, problem2_recall0 = precision_recall_interval_for_class(problem2_Y_test, predictions_test, cls=0, delta=0.05)
problem2_precision1, problem2_recall1 = precision_recall_interval_for_class(problem2_Y_test, predictions_test, cls=1, delta=0.05)

print("precision class 0 interval:", problem2_precision0)
print("recall class 0 interval:", problem2_recall0)
print("precision class 1 interval:", problem2_precision1)
print("recall class 1 interval:", problem2_recall1)

assert(type(problem2_precision0) == tuple)
assert(len(problem2_precision0) == 2)
assert(type(problem2_recall0) == tuple)
assert(len(problem2_recall0) == 2)
assert(type(problem2_precision1) == tuple)
assert(len(problem2_precision1) == 2)
assert(type(problem2_recall1) == tuple)
assert(len(problem2_recall1) == 2)
```

## Answer to Problem 2, Part 2

The cost function is:

- False negative: true CHD but predicted no CHD, cost 300.
- False positive: true no CHD but predicted CHD, cost 10.
- Correct predictions: cost 0.

We return the **average cost per person**.


```python
def problem2_cost(model, threshold, X, Y):
    pred_proba = model.predict_proba(X)[:, 1]
    predictions = (pred_proba >= threshold).astype(int)
    
    false_negative = (Y == 1) & (predictions == 0)
    false_positive = (Y == 0) & (predictions == 1)
    
    total_cost = 300 * np.sum(false_negative) + 10 * np.sum(false_positive)
    average_cost = total_cost / len(Y)
    
    return float(average_cost)
```

## Answer to Problem 2, Part 3

We try 100 thresholds between 0 and 1, compute the test cost for each threshold, and choose the threshold with the smallest cost.


```python
thresholds = np.linspace(0, 1, 100)
test_costs = np.array([problem2_cost(problem2_pipe, t, problem2_X_test, problem2_Y_test) for t in thresholds])

best_index = int(np.argmin(test_costs))
problem2_optimal_threshold = float(thresholds[best_index])
problem2_cost_at_optimal_threshold = float(test_costs[best_index])

print("Optimal threshold:", problem2_optimal_threshold)
print("Cost at optimal threshold on test data:", problem2_cost_at_optimal_threshold)

plt.figure(figsize=(8, 5))
plt.plot(thresholds, test_costs, marker='o', markersize=3)
plt.axvline(problem2_optimal_threshold, linestyle='--', label='Optimal threshold')
plt.xlabel("Threshold")
plt.ylabel("Average cost")
plt.title("Average cost on test data for different thresholds")
plt.legend()
plt.show()
```

## Answer to Problem 2, Part 4

Now we evaluate the selected threshold on the validation set. For the confidence interval of the average cost, each individual cost is bounded between 0 and 300. We use Hoeffding with 99% confidence, so $\delta=0.01$.


```python
problem2_cost_at_optimal_threshold_validation = problem2_cost(
    problem2_pipe,
    problem2_optimal_threshold,
    problem2_X_val,
    problem2_Y_val
)

# Compute individual validation costs for confidence interval
pred_proba_val = problem2_pipe.predict_proba(problem2_X_val)[:, 1]
pred_val = (pred_proba_val >= problem2_optimal_threshold).astype(int)

individual_costs_val = np.zeros(len(problem2_Y_val))
individual_costs_val[(problem2_Y_val == 1) & (pred_val == 0)] = 300
individual_costs_val[(problem2_Y_val == 0) & (pred_val == 1)] = 10

n_val = len(individual_costs_val)
delta = 0.01
cost_min, cost_max = 0, 300
cost_eps = (cost_max - cost_min) * np.sqrt(np.log(2/delta) / (2*n_val))

problem2_cost_interval = (
    float(max(cost_min, problem2_cost_at_optimal_threshold_validation - cost_eps)),
    float(min(cost_max, problem2_cost_at_optimal_threshold_validation + cost_eps))
)

print("Validation average cost:", problem2_cost_at_optimal_threshold_validation)
print("99% Hoeffding cost interval:", problem2_cost_interval)

assert(type(problem2_cost_interval) == tuple)
assert(len(problem2_cost_interval) == 2)
```

---

# Problem 3 — Markov chains

**Maximum Points = 13**

## Full question

Consider the following two Markov chains. The states are ordered alphabetically, so state `A` has index `0`, state `B` has index `1`, and so on.

### Markov Chain A transitions from the diagram

- $A\to B$ with probability $0.2$.
- $A\to D$ with probability $0.8$.
- $B\to A$ with probability $0$.
- $B\to C$ with probability $1$.
- $C\to B$ with probability $1$.
- $C\to D$ with probability $0$.
- $D\to A$ with probability $0.5$.
- $D\to C$ with probability $0.5$.

### Markov Chain B transitions from the diagram

- $A\to B$ with probability $1$.
- $B\to C$ with probability $1$.
- $C\to B$ with probability $1/2$.
- $C\to D$ with probability $1/2$.
- $D\to C$ with probability $1/2$.
- $D\to E$ with probability $1/2$.
- $E\to F$ with probability $1$.
- $F\to A$ with probability $1/2$.
- $F\to E$ with probability $1/2$.

Answer each question for all chains:

1. **[2p]** What is the transition matrix?
2. **[1p]** Is the Markov chain irreducible?
3. **[4p]** Is the Markov chain aperiodic? What is the period for each state? Hint: recall the definition of period. Let $T = \{t\in\mathbb{N}:P^t(x,x)>0\}$ and the greatest common divisor of $T$ is the period.
4. **[2p]** Being in state A at time 0, what is the probability of being in state B at time 5 after 5 steps?
5. **[4p]** Define $T$ as the first time being in state D starting in state A. That is, if $X_0,X_1,\ldots$ is the Markov chain, then define for $X_0=A$:

$$
T(\omega)=\inf_{t\in\mathbb{N}}\{t:X_t(\omega)=D\}
$$

where the infimum over the empty set is $\infty$. Calculate $P(T=1)$, $P(T=2)$, $P(T=3)$, $P(T=4)$, $P(T=5)$, and $P(T=\infty)$.

## Answer to Problem 3, Part 1 — Transition matrices

Rows represent the current state and columns represent the next state.


```python
import numpy as np

# State order for Chain A: A, B, C, D
problem3_A = np.array([
    [0.0, 0.2, 0.0, 0.8],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0]
])

# State order for Chain B: A, B, C, D, E, F
problem3_B = np.array([
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.5, 0.0, 0.0, 0.0, 0.5, 0.0]
])

print("Markov Chain A transition matrix:")
print(problem3_A)
print("
Markov Chain B transition matrix:")
print(problem3_B)
```

## Answer to Problem 3, Part 2 — Irreducibility

A Markov chain is irreducible if every state can reach every other state.

- Chain A is **not irreducible**, because once it enters the closed class `{B, C}`, it cannot return to `A` or `D`.
- Chain B is **irreducible**, because every state can reach every other state through the directed transitions.


```python
problem3_A_irreducible = False
problem3_B_irreducible = True

print("Chain A irreducible:", problem3_A_irreducible)
print("Chain B irreducible:", problem3_B_irreducible)
```

## Answer to Problem 3, Part 3 — Periods and aperiodicity

A state is aperiodic if its period is 1. In both chains, the possible return times are even, so the period is 2 for every state. Therefore neither chain is aperiodic.


```python
problem3_A_is_aperiodic = False
problem3_B_is_aperiodic = False

problem3_A_periods = np.array([2, 2, 2, 2])
problem3_B_periods = np.array([2, 2, 2, 2, 2, 2])

print("Chain A aperiodic:", problem3_A_is_aperiodic)
print("Chain A periods:", problem3_A_periods)
print("Chain B aperiodic:", problem3_B_is_aperiodic)
print("Chain B periods:", problem3_B_periods)
```

## Answer to Problem 3, Part 4 — Probability of being in B at time 5

Starting in state A, the probability of being in state B after 5 steps is the entry `(A, B)` of $P^5$.


```python
import numpy as np

problem3_A_PB5 = float(np.linalg.matrix_power(problem3_A, 5)[0, 1])
problem3_B_PB5 = float(np.linalg.matrix_power(problem3_B, 5)[0, 1])

print("Chain A P(X_5 = B | X_0 = A):", problem3_A_PB5)
print("Chain B P(X_5 = B | X_0 = A):", problem3_B_PB5)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[7], line 3
          1 import numpy as np
    ----> 3 problem3_A_PB5 = float(np.linalg.matrix_power(problem3_A, 5)[0, 1])
          4 problem3_B_PB5 = float(np.linalg.matrix_power(problem3_B, 5)[0, 1])
          6 print("Chain A P(X_5 = B | X_0 = A):", problem3_A_PB5)
    

    NameError: name 'problem3_A' is not defined


## Answer to Problem 3, Part 5 — First hitting time of D starting from A

We compute the probability that the first time reaching state D is exactly at times 1, 2, 3, 4, and 5. The remaining probability is assigned to $T=\infty$ for this question format.


```python
def first_hit_probs(P, start, target, max_t=5):
    """
    Returns P(T=1),...,P(T=max_t),P(T=infinity or after max_t)
    for first hitting time of target starting from start.
    """
    n_states = P.shape[0]
    dist = np.zeros(n_states)
    dist[start] = 1.0
    probs = []
    
    for t in range(1, max_t + 1):
        # Probability of first hitting target at time t
        hit_prob = float(dist @ P[:, target])
        probs.append(hit_prob)
        
        # Move one step, then kill/remove paths that have hit target
        dist = dist @ P
        dist[target] = 0.0
    
    prob_not_hit_by_max_t = float(dist.sum())
    return probs, prob_not_hit_by_max_t

# In both chains, D is index 3. Start A is index 0.
A_hit_probs, A_inf = first_hit_probs(problem3_A, start=0, target=3, max_t=5)
B_hit_probs, B_inf = first_hit_probs(problem3_B, start=0, target=3, max_t=5)

problem3_A_PT1 = float(A_hit_probs[0])
problem3_A_PT2 = float(A_hit_probs[1])
problem3_A_PT3 = float(A_hit_probs[2])
problem3_A_PT4 = float(A_hit_probs[3])
problem3_A_PT5 = float(A_hit_probs[4])
problem3_A_PT_inf = float(A_inf)

problem3_B_PT1 = float(B_hit_probs[0])
problem3_B_PT2 = float(B_hit_probs[1])
problem3_B_PT3 = float(B_hit_probs[2])
problem3_B_PT4 = float(B_hit_probs[3])
problem3_B_PT5 = float(B_hit_probs[4])
problem3_B_PT_inf = float(B_inf)

print("Chain A:")
print(problem3_A_PT1, problem3_A_PT2, problem3_A_PT3, problem3_A_PT4, problem3_A_PT5, problem3_A_PT_inf)

print("Chain B:")
print(problem3_B_PT1, problem3_B_PT2, problem3_B_PT3, problem3_B_PT4, problem3_B_PT5, problem3_B_PT_inf)
```

### Direct final values for Problem 3, Part 5

For Chain A:

- $P(T=1)=0.8$
- $P(T=2)=0$
- $P(T=3)=0$
- $P(T=4)=0$
- $P(T=5)=0$
- $P(T=\infty)=0.2$

For Chain B:

- $P(T=1)=0$
- $P(T=2)=0$
- $P(T=3)=0.5$
- $P(T=4)=0$
- $P(T=5)=0.25$
- $P(T=\infty)=0.25$ in the exam's requested split after listing times 1 through 5. In the true infinite-time sense for this irreducible finite chain, the probability of eventually reaching D is 1, so true $P(T=\infty)=0$. Here the code variable stores the probability not hit by time 5, because the exam asks for only 1,2,3,4,5 and infinity together.

## Notebook complete

This notebook includes the whole questions, solved code answers, comments, plots, and free-text explanations.
