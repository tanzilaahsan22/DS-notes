# Solved Exam Notebook — 16th of June 2025

Course: **1MS041 Introduction to Data Science / Introduktion till dataanalys**  
Exam time: **16th of June 2025, 13.00–18.00**  

This notebook includes:

1. the **whole questions** from the paper,
2. solved Python code answers,
3. step-by-step comments,
4. free-text answers for the written parts,
5. local format checks.

> Important: Problem 2 and Problem 3 need the course data files in the same folder structure as the original exam:
>
> - `data/websites.csv`
> - `data/fraud.csv`
> - `Utils.py` if your course provided it



```python
# Insert your anonymous exam ID as a string in the variable below
examID = "XXX"

# Common imports used throughout the notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For reproducibility of simulations
rng = np.random.default_rng(1)

```

---

# Exam vB, PROBLEM 1

**Maximum Points = 14**

In this problem you will do rejection sampling from complicated distributions, you will also be using your samples to compute certain integrals, a method known as Monte Carlo integration: **Keep in mind that choosing a good sampling distribution is often key to avoid too much rejection**.

## Full question

1. **[4p]** Fill in the remaining part of the function `problem1_rejection` in order to produce samples from the below distribution using rejection sampling: **Hint: F is the distribution function**

$$
F[x] =
\begin{cases}
0, & x \le 0 \\
\dfrac{e^{x^2}-x^2-1}{e-2}, & 0 < x < 1 \\
1, & x \ge 1
\end{cases}
$$

2. **[2p]** Produce **100000 samples**. Use fewer if it takes too long, more than 10 sec, and you cannot find a solution. Put the answer in `problem1_samples` from the above distribution and plot the histogram together with the true density.

3. **[2p]** Use the above 100000 samples, `problem1_samples`, to approximately compute the integral

$$
\int_0^1 \sin(x)\frac{2(e^{x^2}-1)x}{e-2}\,dx
$$

and store the result in `problem1_integral`.

4. **[2p]** Use Hoeffding's inequality to produce a **95% confidence interval** of the integral above and store the result as a tuple in the variable `problem1_interval`.

5. **[4p]** Fill in the remaining part of the function `problem1_rejection_2` in order to produce samples from the below distribution using rejection sampling:

$$
F[x] =
\begin{cases}
0, & x \le 0 \\
20xe^{20-1/x}, & 0 < x < \frac{1}{20} \\
1, & x \ge \frac{1}{20}
\end{cases}
$$

**Hint:** this is tricky because if you choose the wrong sampling distribution you reject at least 9 times out of 10. You will get points based on how long your code takes to create a certain number of samples. If you choose the correct sampling distribution you can easily create 100000 samples within 2 seconds.


## Problem 1 — step-by-step idea

For part 1, first differentiate the CDF:

$$
f(x)=F'(x)=\frac{2x(e^{x^2}-1)}{e-2}, \quad 0<x<1.
$$

A good proposal distribution is

$$
g(x)=2x, \quad 0<x<1,
$$

which is a Beta(2,1) distribution and can be sampled as `sqrt(U)`. Then

$$
\frac{f(x)}{g(x)}=\frac{e^{x^2}-1}{e-2}.
$$

The maximum is at `x=1`, so

$$
M=\frac{e-1}{e-2}.
$$

Therefore the acceptance probability is

$$
\frac{f(x)}{Mg(x)} = \frac{e^{x^2}-1}{e-1}.
$$



```python
# Part 1
# Rejection sampling from:
# F(x) = (exp(x^2) - x^2 - 1) / (e - 2), 0 < x < 1
# Density: f(x) = 2*x*(exp(x^2)-1)/(e-2)

def problem1_rejection(n_samples=1):
    """
    Return samples from the distribution with CDF
    F(x) = (exp(x^2) - x^2 - 1)/(e - 2), for 0 < x < 1.
    
    Proposal:
        g(x) = 2x on (0,1), sampled by X = sqrt(U).
    Acceptance probability:
        (exp(x^2)-1)/(e-1).
    """
    samples = []
    n_samples = int(n_samples)
    
    while len(samples) < n_samples:
        # Generate in batches to make it fast
        batch_size = max(1000, 2 * (n_samples - len(samples)))
        u = rng.random(batch_size)
        x = np.sqrt(u)  # proposal g(x)=2x
        accept_prob = (np.exp(x**2) - 1) / (np.e - 1)
        accept = rng.random(batch_size) <= accept_prob
        samples.extend(x[accept].tolist())
    
    return np.array(samples[:n_samples])

# Quick check
problem1_rejection(5)

```


```python
# Part 2
# Produce 100000 samples and plot histogram with the true density.

problem1_samples = problem1_rejection(100000)

# True density from differentiating the CDF
x_grid = np.linspace(0, 1, 500)
true_density = 2 * x_grid * (np.exp(x_grid**2) - 1) / (np.e - 2)

plt.figure(figsize=(8, 5))
plt.hist(problem1_samples, bins=60, density=True, alpha=0.6, label="Rejection samples")
plt.plot(x_grid, true_density, linewidth=2, label="True density")
plt.xlabel("x")
plt.ylabel("density")
plt.title("Problem 1: samples and true density")
plt.legend()
plt.show()

print("Number of samples:", len(problem1_samples))
print("Sample mean:", float(np.mean(problem1_samples)))

```


```python
# Part 3
# The integral is E[sin(X)] where X has the density from part 1.
# So Monte Carlo estimate = average of sin(samples).

problem1_integral = float(np.mean(np.sin(problem1_samples)))

print("Monte Carlo estimate of the integral:", problem1_integral)

```


```python
# Part 4
# Hoeffding confidence interval for E[sin(X)].
# Since X is in [0,1], sin(X) is in [0, sin(1)].
# Hoeffding: P(|mean - E| >= eps) <= 2 exp(-2n eps^2 / (b-a)^2)
# For 95% confidence, alpha = 0.05.

alpha = 0.05
n = len(problem1_samples)
a = 0.0
b = np.sin(1.0)
epsilon = (b - a) * np.sqrt(np.log(2 / alpha) / (2 * n))

problem1_interval = (
    float(problem1_integral - epsilon),
    float(problem1_integral + epsilon)
)

print("95% Hoeffding CI for the integral:", problem1_interval)

```

## Problem 1 Part 5 — step-by-step idea

The second CDF is

$$
F(x)=20xe^{20-1/x}, \quad 0<x<\frac{1}{20}.
$$

Differentiate it:

$$
f(x)=e^{20-1/x}\left(20+\frac{20}{x}\right), \quad 0<x<\frac{1}{20}.
$$

A very efficient trick is to transform

$$
T = \frac{1}{X}-20, \quad T\ge 0, \quad X=\frac{1}{T+20}.
$$

Under this transformation, the density of `T` is close to an exponential density:

$$
h(t)= e^{-t}\frac{20(t+21)}{(t+20)^2}, \quad t\ge 0.
$$

Use proposal

$$
g(t)=e^{-t}, \quad t\ge 0,
$$

which is `Exp(1)`. The multiplier is

$$
M = \max_{t\ge 0}\frac{h(t)}{g(t)} = \frac{20(21)}{20^2}=1.05.
$$

So the acceptance probability is

$$
\frac{h(t)}{Mg(t)} = \frac{20(t+21)}{1.05(t+20)^2}.
$$

After accepting `T`, convert back using

$$
X=\frac{1}{T+20}.
$$



```python
# Part 5
# Efficient rejection sampler for the second distribution.

def problem1_rejection_2(n_samples=1):
    """
    Return samples from the distribution with CDF
    F(x) = 20*x*exp(20 - 1/x), for 0 < x < 1/20.
    
    Efficient method:
        T = 1/X - 20.
        Proposal T ~ Exp(1).
        Accept with probability 20*(T+21)/(1.05*(T+20)^2).
        Then X = 1/(T+20).
    """
    samples = []
    n_samples = int(n_samples)
    M = 1.05
    
    while len(samples) < n_samples:
        batch_size = max(1000, 2 * (n_samples - len(samples)))
        t = rng.exponential(scale=1.0, size=batch_size)
        accept_prob = 20 * (t + 21) / (M * (t + 20)**2)
        accept = rng.random(batch_size) <= accept_prob
        x = 1 / (t[accept] + 20)
        samples.extend(x.tolist())
    
    return np.array(samples[:n_samples])

# Quick speed/format check
problem1_samples_2 = problem1_rejection_2(100000)
print("Number of samples from second distribution:", len(problem1_samples_2))
print("Min and max:", float(problem1_samples_2.min()), float(problem1_samples_2.max()))

```

## Local Test for Exam vB, PROBLEM 1

Evaluate the cell below to make sure the answers have the correct formats.



```python
# This cell is just to check that you got the correct formats of your answer
import numpy as np
try:
    assert(isinstance(problem1_rejection(10), np.ndarray))
except:
    print("Try again. You should return a numpy array from problem1_rejection")
else:
    print("Good, your problem1_rejection returns a numpy array")

try:
    assert(isinstance(problem1_samples, np.ndarray))
except:
    print("Try again. your problem1_samples is not a numpy array")
else:
    print("Good, your problem1_samples is a numpy array")

try:
    assert(isinstance(problem1_integral, float))
except:
    print("Try again. your problem1_integral is not a float")
else:
    print("Good, your problem1_integral is a float")

try:
    assert(isinstance(problem1_interval, list) or isinstance(problem1_interval, tuple)), "problem1_interval not a tuple or list"
    assert(len(problem1_interval) == 2), "problem1_interval does not have length 2, it should have a lower bound and an upper bound"
except Exception as e:
    print(e)
else:
    print("Good, your problem1_interval is a tuple or list of length 2")

try:
    assert(isinstance(problem1_rejection_2(10), np.ndarray))
except:
    print("Try again. You should return a numpy array from problem1_rejection_2")
else:
    print("Good, your problem1_rejection_2 returns a numpy array")

```

---

# Exam vB, PROBLEM 2

**Maximum Points = 14**

In this problem we have data consisting of user behavior on a website. The pages of the website are just numbers in the dataset `0, 1, 2, ...` and each row consists of a user, a source and a destination page. This signifies that the user was on the source page and clicked a link leading them to the destination page. The goal is to improve the user experience by decreasing load time of the next page visited. As such, we need a good estimate for the next site likely to be visited. We will model this using a homogeneous Markov chain. Each row in the data-file corresponds to a single realization of a transition.

## Full question

1. **[3p]** Load the data in the file `data/websites.csv` and construct a matrix of size `n_pages x n_pages` which is the maximum likelihood estimate of the true transition matrix for the Markov chain. Here the ordering of the states are exactly the ones in the data-file, that is page `0` has index `0` in the matrix.

2. **[4p]** A page loads in `Exp(3)`, exponentially distributed with mean `1/3` seconds, if not preloaded and loads with `Exp(20)`, exponentially distributed with mean `1/20` seconds, if preloaded and we only preload the most likely next site. Given that we start in **page 8**, simulate **10000 load times** from page 8, that is only a single step, and store the result in the variable indicated in the cell. Repeat the experiment but this time preload the two most likely pages and store the result in the indicated variable.

3. **[3p]** Compare the average, empirical, load time from part 2 with the theoretical one of no pre-loading. Does the load time improve, how did you come to this conclusion? Explain in the free text field.

4. **[4p]** Calculate the stationary distribution of the Markov chain and calculate the expected load time with respect to the stationary distribution.


## Problem 2 — step-by-step idea

The maximum likelihood estimate of a Markov transition probability is:

$$
\hat P_{ij}=\frac{\text{number of transitions from page }i\text{ to page }j}{\text{number of transitions leaving page }i}.
$$

So each row of the transition matrix must sum to 1, unless a page has no outgoing transitions. For safety, if a page has no outgoing transitions, this solution leaves a self-loop on that page.



```python
# Part 1: 3 points
# Load data/websites.csv and estimate the Markov transition matrix.
import numpy as np
import pandas as pd
websites_df = pd.read_csv("websites.csv")
print("Columns in websites.csv:", list(websites_df.columns))
print(websites_df.head())

# Robustly find source and destination columns.
# The exam says each row has user, source, destination.
# If the file has named columns, this tries to identify them.
lower_cols = {c.lower(): c for c in websites_df.columns}

source_col = None
dest_col = None

for possible in ["source", "src", "from", "source_page"]:
    if possible in lower_cols:
        source_col = lower_cols[possible]
        break

for possible in ["destination", "dest", "to", "target", "destination_page"]:
    if possible in lower_cols:
        dest_col = lower_cols[possible]
        break

# If names are not obvious, use the last two columns as source and destination.
if source_col is None or dest_col is None:
    source_col = websites_df.columns[-2]
    dest_col = websites_df.columns[-1]

sources = websites_df[source_col].astype(int).to_numpy()
destinations = websites_df[dest_col].astype(int).to_numpy()

problem2_n_states = int(max(sources.max(), destinations.max()) + 1)
counts = np.zeros((problem2_n_states, problem2_n_states), dtype=float)

for s, d in zip(sources, destinations):
    counts[s, d] += 1

problem2_transition_matrix = np.zeros_like(counts)
row_sums = counts.sum(axis=1)

for i in range(problem2_n_states):
    if row_sums[i] > 0:
        problem2_transition_matrix[i] = counts[i] / row_sums[i]
    else:
        # Safety: if a page has no outgoing transitions, make it stay in itself.
        problem2_transition_matrix[i, i] = 1.0

print("Number of states:", problem2_n_states)
print("Transition matrix shape:", problem2_transition_matrix.shape)
print("First few row sums:", problem2_transition_matrix.sum(axis=1)[:10])

```

    Columns in websites.csv: ['user', 'source', 'destination']
       user  source  destination
    0     0       0            3
    1     0       3            6
    2     0       6            9
    3     0       9            2
    4     0       2            9
    Number of states: 10
    Transition matrix shape: (10, 10)
    First few row sums: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    


```python
# Part 2: 4 points
# Simulate the website load times for 10000 users currently on page 8.
# Not preloaded: Exp(3), mean = 1/3.
# Preloaded: Exp(20), mean = 1/20.

start_page = 8
n_sim = 10000
p_start = problem2_transition_matrix[start_page]

# Most likely next pages from page 8
sorted_pages = np.argsort(p_start)[::-1]
top1_pages = set(sorted_pages[:1])
top2_pages = set(sorted_pages[:2])

# Simulate next pages according to the transition probabilities from page 8
next_pages_top = rng.choice(problem2_n_states, size=n_sim, p=p_start)
next_pages_two = rng.choice(problem2_n_states, size=n_sim, p=p_start)

# If actual next page was preloaded, load time has rate 20.
# Otherwise, load time has rate 3.
problem2_page_load_times_top = np.where(
    np.isin(next_pages_top, list(top1_pages)),
    rng.exponential(scale=1/20, size=n_sim),
    rng.exponential(scale=1/3, size=n_sim)
)

problem2_page_load_times_two = np.where(
    np.isin(next_pages_two, list(top2_pages)),
    rng.exponential(scale=1/20, size=n_sim),
    rng.exponential(scale=1/3, size=n_sim)
)

print("Top 1 preloaded page:", sorted_pages[:1])
print("Top 2 preloaded pages:", sorted_pages[:2])
print("Empirical average load time, preload top 1:", float(np.mean(problem2_page_load_times_top)))
print("Empirical average load time, preload top 2:", float(np.mean(problem2_page_load_times_two)))

```

    Top 1 preloaded page: [5]
    Top 2 preloaded pages: [5 2]
    Empirical average load time, preload top 1: 0.28516190341715464
    Empirical average load time, preload top 2: 0.2425555999175122
    


```python
# Part 3: 3 points
# Without preloading, the load time is Exp(3), so the theoretical mean is 1/3.

problem2_avg = float(1/3)
problem2_comparison = bool(problem2_avg > np.mean(problem2_page_load_times_top))

print("Theoretical average without preloading:", problem2_avg)
print("Empirical average with top-1 preloading:", float(np.mean(problem2_page_load_times_top)))
print("Is no-preload average larger than top-1 preload average?", problem2_comparison)

```

    Theoretical average without preloading: 0.3333333333333333
    Empirical average with top-1 preloading: 0.28516190341715464
    Is no-preload average larger than top-1 preload average? True
    

## Free text answer for Problem 2 Part 3

Without preloading, the load time has distribution `Exp(3)`, so the theoretical expected load time is

$$
E[T]=\frac{1}{3}.
$$

I compared this value with the simulated average load time when preloading the most likely next page from page 8. If

$$
\frac{1}{3} > \text{mean(problem2_page_load_times_top)},
$$

then the average load time is smaller after preloading, so the loading time improves. The variable `problem2_comparison` stores this comparison as `True` or `False`.



```python
# Part 4: 4 points
# Calculate stationary distribution pi satisfying pi P = pi.
# This is the eigenvector of P.T with eigenvalue 1.

eigenvalues, eigenvectors = np.linalg.eig(problem2_transition_matrix.T)
idx = np.argmin(np.abs(eigenvalues - 1))
stationary = np.real(eigenvectors[:, idx])

# Make signs positive and normalize
stationary = np.abs(stationary)
problem2_stationary_distribution = stationary / stationary.sum()

# Expected load time with respect to stationary distribution.
# For each current page i, preload its most likely next page.
# Conditional expected load time from page i:
#   P(hit preload)*(1/20) + P(miss preload)*(1/3)
row_top_probs = np.max(problem2_transition_matrix, axis=1)
expected_load_by_page = row_top_probs * (1/20) + (1 - row_top_probs) * (1/3)
problem2_avg_stationary = float(np.dot(problem2_stationary_distribution, expected_load_by_page))

print("Stationary distribution sums to:", float(problem2_stationary_distribution.sum()))
print("Expected load time under stationary distribution:", problem2_avg_stationary)

```

    Stationary distribution sums to: 1.0
    Expected load time under stationary distribution: 0.2709421039043229
    

## Local Test for Exam vB, PROBLEM 2

Evaluate the cell below to make sure the answers have the correct formats.



```python
# This cell is just to check that you got the correct formats of your answer
import numpy as np
try:
    assert isinstance(problem2_transition_matrix, np.ndarray)
except:
    print("Try again. your problem2_transition_matrix is not a numpy array")
else:
    print("Good, your problem2_transition_matrix is a numpy array")

try:
    assert isinstance(problem2_n_states, int)
except:
    print("Try again. your problem2_n_states is not an integer")
else:
    print("Good, your problem2_n_states is an integer")

try:
    assert problem2_transition_matrix.shape == (problem2_n_states, problem2_n_states)
except:
    print("Try again. your problem2_transition_matrix does not have the correct shape")
else:
    print("Good, your problem2_transition_matrix has the correct shape")

try:
    assert isinstance(problem2_page_load_times_top, np.ndarray), "problem2_page_load_times_top is not a numpy array"
    assert problem2_page_load_times_top.shape == (10000,), "problem2_page_load_times_top does not have the correct shape"
except Exception as e:
    print(e)
else:
    print("Good, your problem2_page_load_times_top is a numpy array of shape (10000,)")

try:
    assert isinstance(problem2_page_load_times_two, np.ndarray), "problem2_page_load_times_two is not a numpy array"
    assert problem2_page_load_times_two.shape == (10000,), "problem2_page_load_times_two does not have the correct shape"
except Exception as e:
    print(e)
else:
    print("Good, your problem2_page_load_times_two is a numpy array of shape (10000,)")

try:
    assert isinstance(problem2_avg, float), "problem2_avg is not a float"
except Exception as e:
    print(e)
else:
    print("Good, your problem2_avg is a float")

try:
    assert isinstance(problem2_comparison, bool), "problem2_comparison is not a boolean"
except Exception as e:
    print(e)
else:
    print("Good, your problem2_comparison is a boolean")

try:
    assert isinstance(problem2_stationary_distribution, np.ndarray), "problem2_stationary_distribution is not a numpy array"
    assert problem2_stationary_distribution.shape == (problem2_n_states,), "problem2_stationary_distribution does not have the correct shape"
except Exception as e:
    print(e)
else:
    print("Good, your problem2_stationary_distribution is a numpy array of shape (problem2_n_states,)")

try:
    assert isinstance(problem2_avg_stationary, float), "problem2_avg_stationary is not a float"
except Exception as e:
    print(e)
else:
    print("Good, your problem2_avg_stationary is a float")

```

---

# Exam vB, PROBLEM 3

**Maximum Points = 12**

In this problem we are interested in fraud detection in an e-commerce system. In this problem we are given the outputs of a classifier that predicts the probabilities of fraud. Your goal is to explore the threshold choice as in individual assignment 4.

The costs associated with the predictions are:

- **True Positive (TP):** Detecting fraud and blocking the transaction costs the company `100`, manual review etc.
- **True Negative (TN):** Allowing a legitimate transaction has no cost.
- **False Positive (FP):** Incorrectly classifying a legitimate transaction as fraudulent costs `120`, customer dissatisfaction plus operational expenses for reversing the decision.
- **False Negative (FN):** Missing a fraudulent transaction costs the company `600`, for example fraud loss plus potential reputational damage or penalties.

The code cells contain more detailed instructions. **THE FIRST CODE CELL INITIALIZES YOUR VARIABLES.**

## Full question

1. **[3p]** Complete filling the function `cost` to compute the average cost of a prediction model under a certain prediction threshold. Plot the cost as a function of the threshold, using the validation data provided in the first code cell of this problem, between `0` and `1` with `0.01` increments.

2. **[2.5p]** Find the threshold that minimizes the cost and calculate the cost at that threshold on the validation data. Also calculate the precision and recall at the optimal threshold on the validation data on class `1` and `0`.

3. **[2.5p]** Repeat step 2, but this time find the best threshold to minimize the `0-1` loss. Calculate the difference in cost between the threshold found in part 2 with the one just found in part 3.

4. **[4p]** Provide a confidence interval around the optimal cost, with **95% confidence**, applied to the test data and explain all the assumption you made.



```python
# RUN THIS CELL TO GET THE DATA
# We start by loading the data

import pandas as pd

PROBLEM3_DF = pd.read_csv('data/fraud.csv')
Y = PROBLEM3_DF['Class'].values
X = PROBLEM3_DF[['V%d' % i for i in range(1,5)] + ['Amount']].values

# We will split the data into training, testing and validation sets.
# The original exam uses Utils.train_test_validation.
# The fallback below is only to make the notebook easier to run if Utils.py is missing.
try:
    from Utils import train_test_validation
    PROBLEM3_X_train, PROBLEM3_X_test, PROBLEM3_X_val, PROBLEM3_y_train, PROBLEM3_y_test, PROBLEM3_y_val = train_test_validation(
        X, Y, shuffle=True, random_state=1
    )
except Exception as e:
    print("Could not import Utils.train_test_validation. Using sklearn fallback split.")
    print("Original error:", e)
    from sklearn.model_selection import train_test_split
    PROBLEM3_X_temp, PROBLEM3_X_test, PROBLEM3_y_temp, PROBLEM3_y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=True, random_state=1, stratify=Y
    )
    PROBLEM3_X_train, PROBLEM3_X_val, PROBLEM3_y_train, PROBLEM3_y_val = train_test_split(
        PROBLEM3_X_temp, PROBLEM3_y_temp, test_size=0.25, shuffle=True, random_state=1, stratify=PROBLEM3_y_temp
    )

# From this we will train a logistic regression model
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(PROBLEM3_X_train, PROBLEM3_y_train)

# THE FOLLOWING CODE WILL PRODUCE THE ARRAYS YOU NEED FOR THE PROBLEM
PROBLEM3_y_pred_proba_val = lr.predict_proba(PROBLEM3_X_val)[:, 1]
PROBLEM3_y_true_val = PROBLEM3_y_val

PROBLEM3_y_pred_proba_test = lr.predict_proba(PROBLEM3_X_test)[:, 1]
PROBLEM3_y_true_test = PROBLEM3_y_test

print("Validation size:", len(PROBLEM3_y_true_val))
print("Test size:", len(PROBLEM3_y_true_test))

```

## Problem 3 — step-by-step idea

A threshold turns predicted fraud probabilities into class predictions:

$$
\hat y_i =
\begin{cases}
1, & \hat p_i \ge \text{threshold} \\
0, & \hat p_i < \text{threshold}
\end{cases}
$$

Then we compute the average cost using:

- TP cost = 100
- TN cost = 0
- FP cost = 120
- FN cost = 600



```python
# Part 1: 3 points
# Implement the cost function and plot cost as a function of threshold.

def cost(y_true, y_predict_proba, threshold):
    """
    Average cost per sample.
    
    y_true: numpy array of true labels 0/1
    y_predict_proba: numpy array of predicted probabilities for class 1
    threshold: classify as 1 if probability >= threshold
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_predict_proba) >= threshold).astype(int)
    
    TP = (y_true == 1) & (y_pred == 1)
    TN = (y_true == 0) & (y_pred == 0)
    FP = (y_true == 0) & (y_pred == 1)
    FN = (y_true == 1) & (y_pred == 0)
    
    costs = np.zeros_like(y_true, dtype=float)
    costs[TP] = 100
    costs[TN] = 0
    costs[FP] = 120
    costs[FN] = 600
    
    return float(np.mean(costs))

# Plot validation cost between 0 and 1 with 0.01 increments
thresholds = np.arange(0, 1.01, 0.01)
validation_costs = np.array([cost(PROBLEM3_y_true_val, PROBLEM3_y_pred_proba_val, t) for t in thresholds])

plt.figure(figsize=(8, 5))
plt.plot(thresholds, validation_costs, marker="o", markersize=3)
plt.xlabel("threshold")
plt.ylabel("average cost")
plt.title("Validation cost as a function of threshold")
plt.grid(True)
plt.show()

print("Minimum validation cost on grid:", float(validation_costs.min()))
print("Best threshold on grid:", float(thresholds[np.argmin(validation_costs)]))

```


```python
# Part 2: 2.5 points
# Find threshold minimizing validation cost.

best_idx = int(np.argmin(validation_costs))
problem3_threshold = float(thresholds[best_idx])
problem3_cost_val = float(validation_costs[best_idx])

problem3_y_pred_val = (PROBLEM3_y_pred_proba_val >= problem3_threshold).astype(int)

# Precision/recall for class 1
TP = np.sum((PROBLEM3_y_true_val == 1) & (problem3_y_pred_val == 1))
FP = np.sum((PROBLEM3_y_true_val == 0) & (problem3_y_pred_val == 1))
FN = np.sum((PROBLEM3_y_true_val == 1) & (problem3_y_pred_val == 0))

problem3_precision_1 = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
problem3_recall_1 = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0

# Precision/recall for class 0
# Treat class 0 as the positive class.
T0 = np.sum((PROBLEM3_y_true_val == 0) & (problem3_y_pred_val == 0))  # correctly predicted 0
F0P = np.sum((PROBLEM3_y_true_val == 1) & (problem3_y_pred_val == 0)) # predicted 0 but actually 1
F0N = np.sum((PROBLEM3_y_true_val == 0) & (problem3_y_pred_val == 1)) # actually 0 but predicted 1

problem3_precision_0 = float(T0 / (T0 + F0P)) if (T0 + F0P) > 0 else 0.0
problem3_recall_0 = float(T0 / (T0 + F0N)) if (T0 + F0N) > 0 else 0.0

print("Optimal cost threshold:", problem3_threshold)
print("Validation cost at optimal threshold:", problem3_cost_val)
print("Predicted labels shape:", problem3_y_pred_val.shape)
print("Class 1 precision:", problem3_precision_1)
print("Class 1 recall:", problem3_recall_1)
print("Class 0 precision:", problem3_precision_0)
print("Class 0 recall:", problem3_recall_0)

```


```python
# Part 3: 2.5 points
# Find the threshold minimizing 0-1 loss on validation data.

def zero_one_loss(y_true, y_predict_proba, threshold):
    y_pred = (y_predict_proba >= threshold).astype(int)
    return float(np.mean(y_pred != y_true))

zero_one_losses = np.array([zero_one_loss(PROBLEM3_y_true_val, PROBLEM3_y_pred_proba_val, t) for t in thresholds])
problem3_threshold_01 = float(thresholds[np.argmin(zero_one_losses)])

cost_at_01_threshold = cost(PROBLEM3_y_true_val, PROBLEM3_y_pred_proba_val, problem3_threshold_01)
problem3_cost_difference = float(cost_at_01_threshold - problem3_cost_val)

print("Best threshold for 0-1 loss:", problem3_threshold_01)
print("Minimum 0-1 validation loss:", float(zero_one_losses.min()))
print("Cost at 0-1 threshold:", float(cost_at_01_threshold))
print("Cost difference = cost(0-1 threshold) - cost(cost-optimal threshold):", problem3_cost_difference)

```


```python
# Part 4: 4 points
# Use threshold problem3_threshold and Hoeffding's inequality for test cost.
# Cost values are bounded between 0 and 600.

# Individual test costs at the chosen threshold

def individual_costs(y_true, y_predict_proba, threshold):
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_predict_proba) >= threshold).astype(int)
    
    costs = np.zeros_like(y_true, dtype=float)
    costs[(y_true == 1) & (y_pred == 1)] = 100  # TP
    costs[(y_true == 0) & (y_pred == 0)] = 0    # TN
    costs[(y_true == 0) & (y_pred == 1)] = 120  # FP
    costs[(y_true == 1) & (y_pred == 0)] = 600  # FN
    return costs

test_costs = individual_costs(PROBLEM3_y_true_test, PROBLEM3_y_pred_proba_test, problem3_threshold)
test_mean_cost = float(np.mean(test_costs))

alpha = 0.05
n_test = len(test_costs)
a = 0.0
b = 600.0
epsilon = (b - a) * np.sqrt(np.log(2 / alpha) / (2 * n_test))

problem3_lower_bound = float(max(0.0, test_mean_cost - epsilon))
problem3_upper_bound = float(test_mean_cost + epsilon)

print("Test mean cost:", test_mean_cost)
print("95% Hoeffding CI lower bound:", problem3_lower_bound)
print("95% Hoeffding CI upper bound:", problem3_upper_bound)

```

## Free text answer for Problem 3 Part 4

I used the threshold selected on the validation data, `problem3_threshold`, and applied it to the independent test data. For each test observation, I computed the individual cost using the problem's cost table:

- TP = 100
- TN = 0
- FP = 120
- FN = 600

The empirical test cost is the mean of these individual costs. To form a 95% confidence interval, I used Hoeffding's inequality because every individual cost is bounded between 0 and 600.

Hoeffding's inequality gives

$$
P(|\bar X - E[X]| \ge \epsilon) \le 2\exp\left(-\frac{2n\epsilon^2}{(b-a)^2}\right).
$$

Here:

- $a=0$
- $b=600$
- $n$ is the number of test samples
- confidence is 95%, so $\alpha=0.05$

Solving for $\epsilon$ gives

$$
\epsilon = (b-a)\sqrt{\frac{\log(2/\alpha)}{2n}}.
$$

Therefore the confidence interval is

$$
[\bar X-\epsilon,\; \bar X+\epsilon].
$$

The assumptions are that the test examples are independent, the test set was not used to choose the threshold, and the individual costs are bounded in the interval `[0, 600]`.


## Local Test for Exam vB, PROBLEM 3

Evaluate the cell below to make sure the answers have the correct formats.



```python
# This cell is just to check that you got the correct formats of your answer
import numpy as np
try:
    assert callable(cost), "cost is not a function"
except:
    print("Try again. your cost is not a function")
else:
    print("Good, your cost is a function")

try:
    assert isinstance(PROBLEM3_y_pred_proba_val, np.ndarray), "PROBLEM3_y_pred_proba_val is not a numpy array"
    assert PROBLEM3_y_pred_proba_val.shape == (len(PROBLEM3_y_true_val),), "PROBLEM3_y_pred_proba_val does not have the correct shape"
except Exception as e:
    print(e)
else:
    print("Good, your PROBLEM3_y_pred_proba_val is a numpy array of shape (len(PROBLEM3_y_true_val),)")

try:
    assert isinstance(PROBLEM3_y_true_val, np.ndarray), "PROBLEM3_y_true_val is not a numpy array"
    assert PROBLEM3_y_true_val.shape == (len(PROBLEM3_y_pred_proba_val),), "PROBLEM3_y_true_val does not have the correct shape"
except Exception as e:
    print(e)
else:
    print("Good, your PROBLEM3_y_true_val is a numpy array of shape (len(PROBLEM3_y_pred_proba_val),)")

try:
    assert isinstance(PROBLEM3_y_pred_proba_test, np.ndarray), "PROBLEM3_y_pred_proba_test is not a numpy array"
    assert PROBLEM3_y_pred_proba_test.shape == (len(PROBLEM3_y_true_test),), "PROBLEM3_y_pred_proba_test does not have the correct shape"
except Exception as e:
    print(e)
else:
    print("Good, your PROBLEM3_y_pred_proba_test is a numpy array of shape (len(PROBLEM3_y_true_test),)")

try:
    assert isinstance(PROBLEM3_y_true_test, np.ndarray), "PROBLEM3_y_true_test is not a numpy array"
    assert PROBLEM3_y_true_test.shape == (len(PROBLEM3_y_pred_proba_test),), "PROBLEM3_y_true_test does not have the correct shape"
except Exception as e:
    print(e)
else:
    print("Good, your PROBLEM3_y_true_test is a numpy array of shape (len(PROBLEM3_y_pred_proba_test),)")

try:
    assert isinstance(cost(np.array([1,1,0,0]), np.array([0.9,0.8,0.1,0.2]), 0.5), float), "cost does not return a float"
except Exception as e:
    print(e)
else:
    print("Good, your cost function returns a float")

try:
    assert cost(np.array([1,1,0,0]), np.array([0.9,0.8,0.1,0.2]), 0.5) == 50.0, "cost does not return the correct value for the test case"
except Exception as e:
    print(e)
else:
    print("Good, your cost function returns the correct value for the test case")

try:
    assert isinstance(problem3_threshold, float), "problem3_threshold is not a float"
    assert 0 <= problem3_threshold <= 1, "problem3_threshold is not between 0 and 1"
except Exception as e:
    print(e)
else:
    print("Good, your problem3_threshold is a float between 0 and 1")

try:
    assert isinstance(problem3_cost_val, float), "problem3_cost_val is not a float"
    assert problem3_cost_val >= 0, "problem3_cost_val is not non-negative"
except Exception as e:
    print(e)
else:
    print("Good, your problem3_cost_val is a float that is non-negative")

try:
    assert isinstance(problem3_y_pred_val, np.ndarray), "problem3_y_pred_val is not a numpy array"
    assert problem3_y_pred_val.shape == (len(PROBLEM3_y_true_val),), "problem3_y_pred_val does not have the correct shape"
except Exception as e:
    print(e)
else:
    print("Good, your problem3_y_pred_val is a numpy array of shape (len(PROBLEM3_y_true_val),)")

try:
    assert np.all(np.isin(problem3_y_pred_val, [0, 1])), "problem3_y_pred_val does not contain only 0s and 1s"
except Exception as e:
    print(e)
else:
    print("Good, your problem3_y_pred_val contains only 0s and 1s")

try:
    assert isinstance(problem3_precision_1, float), "problem3_precision_1 is not a float"
    assert 0 <= problem3_precision_1 <= 1, "problem3_precision_1 is not between 0 and 1"
except Exception as e:
    print(e)
else:
    print("Good, your problem3_precision_1 is a float between 0 and 1")

try:
    assert isinstance(problem3_recall_1, float), "problem3_recall_1 is not a float"
    assert 0 <= problem3_recall_1 <= 1, "problem3_recall_1 is not between 0 and 1"
except Exception as e:
    print(e)
else:
    print("Good, your problem3_recall_1 is a float between 0 and 1")

try:
    assert isinstance(problem3_precision_0, float), "problem3_precision_0 is not a float"
    assert 0 <= problem3_precision_0 <= 1, "problem3_precision_0 is not between 0 and 1"
except Exception as e:
    print(e)
else:
    print("Good, your problem3_precision_0 is a float between 0 and 1")

try:
    assert isinstance(problem3_recall_0, float), "problem3_recall_0 is not a float"
    assert 0 <= problem3_recall_0 <= 1, "problem3_recall_0 is not between 0 and 1"
except Exception as e:
    print(e)
else:
    print("Good, your problem3_recall_0 is a float between 0 and 1")

try:
    assert isinstance(problem3_threshold_01, float), "problem3_threshold_01 is not a float"
    assert 0 <= problem3_threshold_01 <= 1, "problem3_threshold_01 is not between 0 and 1"
except Exception as e:
    print(e)
else:
    print("Good, your problem3_threshold_01 is a float between 0 and 1")

try:
    assert isinstance(problem3_cost_difference, float), "problem3_cost_difference is not a float"
    assert problem3_cost_difference >= 0, "problem3_cost_difference is not non-negative"
except Exception as e:
    print(e)
else:
    print("Good, your problem3_cost_difference is a float that is non-negative")

try:
    assert isinstance(problem3_lower_bound, float), "problem3_lower_bound is not a float"
    assert problem3_lower_bound >= 0, "problem3_lower_bound is not non-negative"
except Exception as e:
    print(e)
else:
    print("Good, your problem3_lower_bound is a float that is non-negative")

try:
    assert isinstance(problem3_upper_bound, float), "problem3_upper_bound is not a float"
    assert problem3_upper_bound >= 0, "problem3_upper_bound is not non-negative"
except Exception as e:
    print(e)
else:
    print("Good, your problem3_upper_bound is a float that is non-negative")

```
