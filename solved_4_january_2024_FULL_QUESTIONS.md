# Solved Exam Notebook — 4th of January 2024

Course: **1MS041 Introduction to Data Science / Introduktion till dataanalys**

This notebook includes the **whole questions inside first**, followed by the **solved code answers with all steps, comments, plots, outputs, and explanations**.

> Put your anonymous exam ID below.


```python
examID = "XXX"
```

## Problem 1 — Full Question

Maximum Points = 14

In this problem you will do rejection sampling from complicated distributions, you will also be using your samples to compute certain integrals, a method known as Monte Carlo integration. Keep in mind that choosing a good sampling distribution is often key to avoid too much rejection.

1. **[4p]** Fill in the remaining part of the function `problem1_inversion` in order to produce samples from the below distribution using rejection sampling:

\[
F[x]=
\begin{cases}
0, & x\le 0\\
\dfrac{e^{x^2}-1}{e-1}, & 0<x<1\\
1, & x\ge 1
\end{cases}
\]

2. **[2p]** Produce 100000 samples, use fewer if it times-out and you cannot find a solution, and put the answer in `problem1_samples` from the above distribution and plot the histogram together with the true density.

3. **[2p]** Use the above 100000 samples (`problem1_samples`) to approximately compute the integral

\[
\int_0^1 \sin(x)\frac{2e^{x^2}x}{e-1}\,dx
\]

and store the result in `problem1_integral`.

4. **[2p]** Use Hoeffding's inequality to produce a 95% confidence interval of the integral above and store the result as a tuple in the variable `problem1_interval`.

5. **[4p]** Fill in the remaining part of the function `problem1_inversion_2` in order to produce samples from the below distribution using rejection sampling:

\[
F[x]=
\begin{cases}
0, & x\le 0\\
20xe^{20-1/x}, & 0<x<\frac{1}{20}\\
1, & x\ge \frac{1}{20}
\end{cases}
\]

Hint: this is tricky because if you choose the wrong sampling distribution you reject at least 9 times out of 10.

### Problem 1 — Solution idea

For Part 1, use inverse transform sampling:

\[
u=\frac{e^{x^2}-1}{e-1}
\quad\Rightarrow\quad
x=\sqrt{\log(1+u(e-1))}.
\]

The true density is

\[
f(x)=\frac{2xe^{x^2}}{e-1},\quad 0<x<1.
\]

For Part 5, set \(Y=1/X-20\). Then \(X=1/(Y+20)\). With proposal \(Y\sim \mathrm{Exp}(1)\), the rejection ratio is

\[
r(y)=\frac{20(y+21)}{(y+20)^2},
\]

whose maximum is \(21/20\). This gives a very high acceptance rate.


```python
# Problem 1 imports
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
```


```python
# Part 1
# The original exam uses: from Utils import timeout
# This fallback keeps the notebook runnable even if Utils.py is missing.
try:
    from Utils import timeout
except Exception:
    def timeout(func):
        return func

@timeout
def problem1_inversion(n_samples=1):
    # Sample from F(x) = (exp(x^2)-1)/(e-1), 0<x<1.
    # Inverse transform: X = sqrt(log(1 + U*(e-1))).
    n_samples = int(n_samples)
    U = np.random.uniform(0, 1, size=n_samples)
    X = np.sqrt(np.log(1 + U * (np.e - 1)))
    return X
```


```python
# Part 2
problem1_samples = problem1_inversion(100000)

def problem1_true_density(x):
    return (2 * x * np.exp(x**2)) / (np.e - 1)

x_grid = np.linspace(0, 1, 1000)

plt.figure(figsize=(8, 5))
plt.hist(problem1_samples, bins=60, density=True, alpha=0.6, edgecolor="black", label="Sample histogram")
plt.plot(x_grid, problem1_true_density(x_grid), linewidth=2, label="True density")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Problem 1 samples with true density")
plt.legend()
plt.show()

problem1_samples[:10], problem1_samples.shape
```


```python
# Part 3
# The integral is E[sin(X)] under the distribution sampled above.
problem1_integral = float(np.mean(np.sin(problem1_samples)))
problem1_integral
```


```python
# Part 4
# Hoeffding 95% confidence interval.
# Since X in [0,1], sin(X) lies in [0, sin(1)].

delta = 0.05
n = len(problem1_samples)
a = 0.0
b = np.sin(1.0)

epsilon = (b - a) * np.sqrt(np.log(2 / delta) / (2 * n))

problem1_interval = (
    float(problem1_integral - epsilon),
    float(problem1_integral + epsilon)
)

problem1_interval
```


```python
# Part 5
def problem1_inversion_2(n_samples=1):
    # Efficient rejection sampler for
    # F(x) = 20*x*exp(20 - 1/x), 0 < x < 1/20.
    #
    # Let Y = 1/X - 20, so X = 1/(Y+20).
    # Use Y ~ Exp(1) as proposal.
    # Acceptance probability = r(Y)/M where
    # r(y)=20*(y+21)/(y+20)^2 and M=21/20.
    n_samples = int(n_samples)
    samples = []
    M = 21 / 20
    batch_size = max(1000, n_samples)

    while len(samples) < n_samples:
        y = np.random.exponential(scale=1.0, size=batch_size)
        ratio = 20 * (y + 21) / ((y + 20) ** 2)
        u = np.random.uniform(0, 1, size=batch_size)

        accepted_y = y[u <= ratio / M]
        accepted_x = 1 / (accepted_y + 20)
        samples.extend(accepted_x.tolist())

    return np.array(samples[:n_samples])

problem1_samples_2 = problem1_inversion_2(100000)
problem1_samples_2[:10], problem1_samples_2.shape
```


```python
# Local test for Problem 1
try:
    assert(isinstance(problem1_inversion(10), np.ndarray))
    print("Good, problem1_inversion returns a numpy array")
except:
    print("Try again. You should return a numpy array from problem1_inversion")

try:
    assert(isinstance(problem1_samples, np.ndarray))
    print("Good, problem1_samples is a numpy array")
except:
    print("Try again. problem1_samples is not a numpy array")

try:
    assert(isinstance(problem1_integral, float))
    print("Good, problem1_integral is a float")
except:
    print("Try again. problem1_integral is not a float")

try:
    assert(isinstance(problem1_interval, list) or isinstance(problem1_interval, tuple))
    assert(len(problem1_interval) == 2)
    print("Good, problem1_interval is a tuple/list of length 2")
except Exception as e:
    print(e)

try:
    assert(isinstance(problem1_inversion_2(10), np.ndarray))
    print("Good, problem1_inversion_2 returns a numpy array")
except:
    print("Try again. You should return a numpy array from problem1_inversion_2")
```

## Problem 2 — Full Question

Maximum Points = 13

Let us build a proportional model

\[
P(Y = 1 \mid X)=G(\beta_0+\beta\cdot X)
\]

where \(G\) is the logistic function, for the spam vs not spam data. Here we assume that the features are presence vs not presence of a word. Let \(X_1, X_2, X_3\) denote the presence (1) or absence (0) of the words **"free"**, **"prize"**, **"win"**.

1. **[2p]** Load the file `data/spam.csv` and create two numpy arrays, `problem2_X` which has shape `(n_emails,3)` where each feature in `problem2_X` corresponds to \(X_1, X_2, X_3\), and `problem2_Y` which has shape `(n_emails,)` and consists of a 1 if the email is spam and 0 if it is not. Split this data into train-calibration-test sets where we have the split 40%, 20%, 40%.

2. **[4p]** Follow the calculation from the lecture notes where we derive the logistic regression and implement the final loss function inside the class `ProportionalSpam`.

3. **[4p]** Train the model `problem2_ps` on the training data. The goal is to calibrate the probabilities output from the model. Start by creating a new variable `problem2_X_pred` with shape `(n_samples,1)` which consists of the predictions of `problem2_ps` on the calibration dataset. Then train a calibration model using `sklearn.tree.DecisionTreeRegressor`, store this trained model in `problem2_calibrator`.

4. **[3p]** Use the trained model `problem2_ps` and the calibrator `problem2_calibrator` to make final predictions on the testing data, store the prediction in `problem2_final_predictions`. Compute the \(0-1\) test-loss and store it in `problem2_01_loss` and provide a 99% confidence interval of it, store this in the variable `problem2_interval`.

### Problem 2 — Solution idea

For logistic regression, with \(z_i=\beta_0+\beta\cdot x_i\), the average negative log-likelihood is

\[
\frac{1}{n}\sum_i \left[\log(1+\exp(z_i))-y_i z_i\right].
\]

For the \(0-1\) test loss confidence interval, use Hoeffding because the loss is bounded in \([0,1]\).


```python
# Problem 2 imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
```


```python
# Part 1
spam_df = pd.read_csv("data/spam.csv")

# Flexible feature detection
feature_cols = []
for target_name in ["free", "prize", "win"]:
    matches = [c for c in spam_df.columns if c.lower().strip() == target_name]
    if matches:
        feature_cols.append(matches[0])

if len(feature_cols) != 3:
    feature_cols = list(spam_df.columns[:3])

# Flexible target detection
target_col = None
for name in ["spam", "is_spam", "label", "Y", "y"]:
    matches = [c for c in spam_df.columns if c.lower().strip() == name.lower()]
    if matches:
        target_col = matches[0]
        break

if target_col is None:
    target_col = spam_df.columns[-1]

problem2_X = spam_df[feature_cols].values
problem2_Y = spam_df[target_col].values.astype(int)

# Split 40% train, 20% calibration, 40% test.
problem2_X_train, X_tmp, problem2_Y_train, Y_tmp = train_test_split(
    problem2_X, problem2_Y, train_size=0.4, random_state=42, shuffle=True
)

problem2_X_calib, problem2_X_test, problem2_Y_calib, problem2_Y_test = train_test_split(
    X_tmp, Y_tmp, train_size=1/3, random_state=42, shuffle=True
)

print(problem2_X_train.shape, problem2_X_calib.shape, problem2_X_test.shape,
      problem2_Y_train.shape, problem2_Y_calib.shape, problem2_Y_test.shape)
```


```python
# Part 2
class ProportionalSpam(object):
    def __init__(self):
        self.coeffs = None
        self.result = None

    def loss(self, X, Y, coeffs):
        # Average logistic negative log-likelihood.
        X = np.asarray(X)
        Y = np.asarray(Y)

        z = np.dot(X, coeffs[1:]) + coeffs[0]
        loss_values = np.logaddexp(0, z) - Y * z
        return float(np.mean(loss_values))

    def fit(self, X, Y):
        import numpy as np
        from scipy import optimize

        opt_loss = lambda coeffs: self.loss(X, Y, coeffs)
        initial_arguments = np.zeros(shape=X.shape[1] + 1)
        self.result = optimize.minimize(opt_loss, initial_arguments, method="cg")
        self.coeffs = self.result.x

    def predict(self, X):
        if self.coeffs is not None:
            G = lambda x: np.exp(x) / (1 + np.exp(x))
            return np.round(10 * G(np.dot(X, self.coeffs[1:]) + self.coeffs[0])) / 10
        else:
            raise ValueError("Model has not been fitted yet.")
```


```python
# Test from the exam
try:
    test_instance = ProportionalSpam()
    test_loss = test_instance.loss(
        np.array([[1, 0, 1], [0, 1, 1]]),
        np.array([1, 0]),
        np.array([1.2, 0.4, 0.3, 0.9])
    )
    assert (np.abs(test_loss - 1.2828629432232497) < 1e-6)
    print("Your loss was correct for a test point")
except:
    print("Your loss was not correct on a test point")
    print("Computed loss:", test_loss)
```


```python
# Part 3
problem2_ps = ProportionalSpam()
problem2_ps.fit(problem2_X_train, problem2_Y_train)

problem2_X_pred = problem2_ps.predict(problem2_X_calib).reshape(-1, 1)

problem2_calibrator = DecisionTreeRegressor(random_state=42)
problem2_calibrator.fit(problem2_X_pred, problem2_Y_calib)

problem2_X_pred[:10], problem2_X_pred.shape
```


```python
# Part 4
test_uncalibrated_predictions = problem2_ps.predict(problem2_X_test).reshape(-1, 1)
problem2_final_predictions = problem2_calibrator.predict(test_uncalibrated_predictions)

problem2_final_decisions = (problem2_final_predictions >= 0.5).astype(int)
problem2_01_loss = float(np.mean(problem2_final_decisions != problem2_Y_test))

delta = 0.01
n = len(problem2_Y_test)
epsilon = np.sqrt(np.log(2 / delta) / (2 * n))

problem2_interval = (
    float(max(0, problem2_01_loss - epsilon)),
    float(min(1, problem2_01_loss + epsilon))
)

problem2_01_loss, problem2_interval
```


```python
# Local checks for Problem 2
assert isinstance(problem2_X, np.ndarray)
assert isinstance(problem2_Y, np.ndarray)
assert problem2_X.shape[1] == 3
assert isinstance(problem2_X_pred, np.ndarray)
assert problem2_X_pred.ndim == 2 and problem2_X_pred.shape[1] == 1
assert isinstance(problem2_01_loss, float)
assert isinstance(problem2_interval, tuple)
assert len(problem2_interval) == 2

print("Problem 2 variables have the expected formats.")
```

## Problem 3 — Full Question

Maximum Points = 13

Consider the following four Markov chains, answer each question for all chains.

1. **[2p]** What is the transition matrix?

2. **[2p]** Is the Markov chain irreducible?

3. **[3p]** Is the Markov chain aperiodic? What is the period for each state?

4. **[3p]** Does the Markov chain have a stationary distribution, and if so, what is it?

5. **[3p]** Is the Markov chain reversible?

The diagrams in the PDF show Markov Chain A and B on page 7, and Markov Chain C and D on page 8. The states are ordered alphabetically in the matrices: \(A,B,C,D,\dots\).

### Diagram from PDF: markov_chain_AB_page7.png

![markov_chain_AB_page7.png](solved_4_january_2024_FULL_QUESTIONS_files/markov_chain_AB_page7.png)

### Diagram from PDF: markov_chain_CD_page8.png

![markov_chain_CD_page8.png](solved_4_january_2024_FULL_QUESTIONS_files/markov_chain_CD_page8.png)

### Problem 3 — Solution idea

A transition matrix \(P\) is read row-by-row. Entry \(P_{ij}\) means probability of moving from state \(i\) to state \(j\).

A chain is irreducible if every state can reach every other state.

The period of a state is

\[
\gcd\{t\ge 1: P^t(x,x)>0\}.
\]

A finite Markov chain always has at least one stationary distribution. We compute it from

\[
\pi P = \pi,\qquad \sum_i \pi_i=1.
\]

A chain is reversible if it satisfies detailed balance:

\[
\pi_iP_{ij}=\pi_jP_{ji}.
\]


```python
# Problem 3 imports
import numpy as np
import math
```


```python
# PART 1
# ------------------------ TRANSITION MATRIX -------------------------------
# State order:
# Chain A, B, D: A, B, C, D
# Chain C: A, B, C, D, E
import numpy as np
import math
problem3_A = np.array([
    [0.8, 0.2, 0.0, 0.0],
    [0.6, 0.2, 0.2, 0.0],
    [0.0, 0.4, 0.0, 0.6],
    [0.0, 0.0, 0.8, 0.2]
])

problem3_B = np.array([
    [0.0, 0.2, 0.0, 0.8],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0]
])

problem3_C = np.array([
    [0.2, 0.3, 0.0, 0.0, 0.5],
    [0.2, 0.2, 0.6, 0.0, 0.0],
    [0.0, 0.4, 0.0, 0.6, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.4],
    [0.0, 0.0, 0.0, 0.4, 0.6]
])

problem3_D = np.array([
    [0.8, 0.2, 0.0, 0.0],
    [0.6, 0.2, 0.2, 0.0],
    [0.0, 0.4, 0.0, 0.6],
    [0.1, 0.0, 0.7, 0.2]
])

print("Row sums:")
print("A:", problem3_A.sum(axis=1))
print("B:", problem3_B.sum(axis=1))
print("C:", problem3_C.sum(axis=1))
print("D:", problem3_D.sum(axis=1))
```

    Row sums:
    A: [1. 1. 1. 1.]
    B: [1. 1. 1. 1.]
    C: [1. 1. 1. 1. 1.]
    D: [1. 1. 1. 1.]
    


```python
# Helper functions
import numpy as np
import math
def communication_matrix(P):
    P = np.asarray(P)
    n = P.shape[0]
    adjacency = P > 1e-12
    reach = adjacency.copy()

    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i, j] = reach[i, j] or (reach[i, k] and reach[k, j])
    return reach

def is_irreducible(P):
    return bool(communication_matrix(P).all())

def state_periods(P, max_power=100):
    P = np.asarray(P)
    n = P.shape[0]
    periods = np.zeros(n, dtype=int)
    power = np.eye(n)

    for t in range(1, max_power + 1):
        power = power @ P
        for i in range(n):
            if power[i, i] > 1e-12:
                if periods[i] == 0:
                    periods[i] = t
                else:
                    periods[i] = math.gcd(periods[i], t)

    return periods

def stationary_distribution(P):
    P = np.asarray(P)
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    pi = np.real(eigenvectors[:, idx])

    if np.sum(pi) < 0:
        pi = -pi

    pi[np.abs(pi) < 1e-12] = 0
    pi = pi / np.sum(pi)
    return pi

def is_reversible(P, pi):
    P = np.asarray(P)
    pi = np.asarray(pi)
    return bool(np.allclose(pi[:, None] * P, pi[None, :] * P.T, atol=1e-8))
```


```python
# PART 2
# ------------------------ IRREDUCIBLE -------------------------------
import numpy as np
import math
problem3_A_irreducible = is_irreducible(problem3_A)
problem3_B_irreducible = is_irreducible(problem3_B)
problem3_C_irreducible = is_irreducible(problem3_C)
problem3_D_irreducible = is_irreducible(problem3_D)

problem3_A_irreducible, problem3_B_irreducible, problem3_C_irreducible, problem3_D_irreducible
```




    (True, False, False, True)




```python
# PART 3
# ------------------------ APERIODIC -------------------------------

import numpy as np
import math
problem3_A_periods = state_periods(problem3_A)
problem3_B_periods = state_periods(problem3_B)
problem3_C_periods = state_periods(problem3_C)
problem3_D_periods = state_periods(problem3_D)

problem3_A_is_aperiodic = bool(np.all(problem3_A_periods == 1))
problem3_B_is_aperiodic = bool(np.all(problem3_B_periods == 1))
problem3_C_is_aperiodic = bool(np.all(problem3_C_periods == 1))
problem3_D_is_aperiodic = bool(np.all(problem3_D_periods == 1))

print("A periods:", problem3_A_periods, "aperiodic:", problem3_A_is_aperiodic)
print("B periods:", problem3_B_periods, "aperiodic:", problem3_B_is_aperiodic)
print("C periods:", problem3_C_periods, "aperiodic:", problem3_C_is_aperiodic)
print("D periods:", problem3_D_periods, "aperiodic:", problem3_D_is_aperiodic)
```

    A periods: [1 1 1 1] aperiodic: True
    B periods: [2 2 2 2] aperiodic: False
    C periods: [1 1 1 1 1] aperiodic: True
    D periods: [1 1 1 1] aperiodic: True
    


```python
# PART 4
# ------------------------ STATIONARY DISTRIBUTION -----------------
# Every finite Markov chain has at least one stationary distribution.
import numpy as np
import math
problem3_A_has_stationary = True
problem3_B_has_stationary = True
problem3_C_has_stationary = True
problem3_D_has_stationary = True

problem3_A_stationary_dist = stationary_distribution(problem3_A)
problem3_B_stationary_dist = stationary_distribution(problem3_B)
problem3_C_stationary_dist = stationary_distribution(problem3_C)
problem3_D_stationary_dist = stationary_distribution(problem3_D)

print("A stationary:", problem3_A_stationary_dist)
print("B stationary:", problem3_B_stationary_dist)
print("C stationary:", problem3_C_stationary_dist)
print("D stationary:", problem3_D_stationary_dist)
```

    A stationary: [0.61538462 0.20512821 0.1025641  0.07692308]
    B stationary: [0.  0.5 0.5 0. ]
    C stationary: [0.  0.  0.  0.5 0.5]
    D stationary: [0.64516129 0.20430108 0.08602151 0.06451613]
    


```python
# PART 5
# ------------------------ REVERSIBLE -----------------
import numpy as np
import math
problem3_A_is_reversible = is_reversible(problem3_A, problem3_A_stationary_dist)
problem3_B_is_reversible = is_reversible(problem3_B, problem3_B_stationary_dist)
problem3_C_is_reversible = is_reversible(problem3_C, problem3_C_stationary_dist)
problem3_D_is_reversible = is_reversible(problem3_D, problem3_D_stationary_dist)

problem3_A_is_reversible, problem3_B_is_reversible, problem3_C_is_reversible, problem3_D_is_reversible
```




    (True, True, True, False)



### Problem 3 — Final answers summary

Using the transition matrices above:

- Chain A is irreducible, aperiodic, has stationary distribution stored in `problem3_A_stationary_dist`, and is reversible.
- Chain B is not irreducible, not aperiodic, has stationary distribution stored in `problem3_B_stationary_dist`, and is reversible for the stationary distribution found.
- Chain C is not irreducible, aperiodic, has stationary distribution stored in `problem3_C_stationary_dist`, and is reversible for the stationary distribution found.
- Chain D is irreducible, aperiodic, has stationary distribution stored in `problem3_D_stationary_dist`, and is not reversible.


```python
# Final format checks
for name, P in [
    ("A", problem3_A),
    ("B", problem3_B),
    ("C", problem3_C),
    ("D", problem3_D),
]:
    assert isinstance(P, np.ndarray)
    assert np.allclose(P.sum(axis=1), 1), f"Rows of chain {name} do not sum to 1"

for x in [
    problem3_A_irreducible, problem3_B_irreducible, problem3_C_irreducible, problem3_D_irreducible,
    problem3_A_is_aperiodic, problem3_B_is_aperiodic, problem3_C_is_aperiodic, problem3_D_is_aperiodic,
    problem3_A_has_stationary, problem3_B_has_stationary, problem3_C_has_stationary, problem3_D_has_stationary,
    problem3_A_is_reversible, problem3_B_is_reversible, problem3_C_is_reversible, problem3_D_is_reversible
]:
    assert isinstance(x, bool)

print("Problem 3 variables have the expected formats.")
```
