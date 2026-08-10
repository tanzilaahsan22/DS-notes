# Exam 16th of January 2026, 8.00-13.00 for the course 1MS041 (Introduction to Data Science / Introduktion till dataanalys)

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
examID=0044-WFT

```

---
## Exam vB, PROBLEM 1
Maximum Points = 14


This problem is about **SVD** and a simple **anomaly detection** idea using low-rank reconstruction.



Unless stated otherwise, when you are asked to produce a matrix or vector, it must be a **NumPy array**.

1. **[4p] SVD.** Load `data/SVD.csv` as instructed in the code cell. Let $X$ be the data matrix of shape `n_samples × n_dimensions`. Compute an SVD
   $$X = U D V^T$$
   where $U$ has shape `n_samples × n_dimensions`, $D$ is the diagonal matrix of shape `(n_dimensions,n_dimensions)` that has the singular values on the diagonal (see documentation for `np.diag`), and $V$ has shape `n_dimensions × n_dimensions`.
   **Important:** `np.linalg.svd` returns `U, d, Vt` where `Vt` is $V^T$.
   Also extract the **first** right and left singular vectors and store them as 1D arrays in the variables provided.

2. **[3p] Explained variance.** For $N =$ `n_dimensions`, define the explained variance using the first $k$ singular values as
   $$
   \mathrm{EV}(k) = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^N \sigma_i^2}.
   $$
   Compute $\mathrm{EV}(k)$ for $k=1,2,\dots,N$ and store it in `problem1_explained_variance` (length `N`). Then set `problem1_num_components` to the **smallest** $k$ such that $\mathrm{EV}(k) \ge 0.99$.

3. **[3p] Plot + interpretation.** Plot explained variance (x-axis: number of components $k$, y-axis: $\mathrm{EV}(k)$). In the markdown cell below, reason about the shape of the curve for this dataset.

4. **[4p] Low-rank reconstruction + outliers.**
   - Using `problem1_num_components`, construct the best rank-$k$ approximation of $X$ and store it in `problem1_approximation`.
   - Compute the row-wise Euclidean reconstruction error $\|X_i - \hat X_i\|_2$ for each row $i$ and store it in `problem1_reconstruction_error` (shape `(n_samples,)`).
   - Plot the empirical distribution function (EDF) of the reconstruction errors (you may use `makeEDF` / `plotEDF` from `Utils.py`).
   - Choose a threshold `problem1_threshold` so that **exactly 100** samples are flagged as outliers (i.e. have reconstruction error >= the threshold).
   - Store those flagged rows in `problem1_outliers` (shape `(100, n_dimensions)`).



```python
# Part 1: 4 points

# Load the data from the file data/SVD.csv and store the data in a numpy array called problem1_data below
# Double check that the numbers have been parsed correctly by checking the dtype of the array by calling problem1_data.dtype
import numpy as np
data = pd.read_csv('data/SVD.csv')

problem1_data = np.array([[1,2],[1,2]])
np.info(problem1_data)
n_samples = 2
n_dimensions = 2

problem1_data = np.random.rand(n_samples, n_dimensions)

def compute_left_singular_vectors(data: np.ndarray) -> np.ndarray:

  Parameters:
     data (np.ndarray): Input matrix of shape (n_samples, n_features).
        
    Returns:
        np.ndarray: Matrix of left singular vectors with shape (n_samples, n_dimensions).
   
       if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array (n_samples x n_features).")
    
    n_samples, n_features = data.shape
    n_dimensions = min(n_samples, n_features)      
    U, S, Vt = np.linalg.svd(data, full_matrices=False)
        U = U[:, :n_dimensions]
    
    return problem1_U


problem1_U = XXX # The matrix of left singular vectors of problem1_data with shape n_samples x n_dimensions
problem1_D = XXX # The diagonal matrix with the singular values of problem1_data on the diagonal with shape n_dimensions x n_dimensions
problem1_V = XXX # The matrix of right singular vectors of problem1_data with shape n_dimensions x n_dimensions

problem1_first_right_singular_vector = XXX # The first right singular vector of problem1_data with shape (n_dimensions,) hint sometimes one needs to invoke flatten() to avoid having shape (n_dimensions, 1) or (1, n_dimensions)

problem1_first_left_singular_vector = XXX # The first left singular vector of problem1_data with shape (n_samples,) hint sometimes one needs to invoke flatten() to avoid having shape (n_samples, 1) or (1, n_samples)
```


      File <string>:20
        Returns:
                ^
    IndentationError: unindent does not match any outer indentation level
    



```python
# Part 2: 3 points

# Calculate the explained variance of using 1,2,3,...,n_dimensions singular values and store it as a numpy array called problem1_explained_variance below

import numpy as np

def explained_variance(singular_values, k):
    
    s = np.array(singular_values, dtype=float)

    if s.ndim != 1:
        raise ValueError("singular_values must be a 1D array or list.")
    N = len(s)
    if not (1 <= k <= N):
        raise ValueError(f"k must be between 1 and {N}, got {k}.")

    total_variance = np.sum(s ** 2)
    if total_variance == 0:
        return 0.0  # Avoid division by zero

    explained = np.sum(s[:k] ** 2)
    return explained / total_variance

# Store in the variable below the smallest number of singular values needed to explain at least 99% of the variance
np.random.seed(42)
X = np.random.rand(50, 25)  

X_centered = X - np.mean(X, axis=0)

# U: left singular vectors, S: singular values, Vt: right singular vectors
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)


n_samples = X_centered.shape[0]
explained_variance = (S ** 2) / (n_samples - 1)

explained_variance_ratio = explained_variance / np.sum(explained_variance)

cumulative_variance = np.cumsum(explained_variance_ratio)
problem1_num_components = np.searchsorted(cumulative_variance, 0.99) + 1  # +1 for index → count

print(f"Smallest number of singular values for ≥99% variance: {problem1_num_components}")
```


```python
# Part 3: 3 points

# Put the code below to plot the explained variance
# use for instance matplotlib

import numpy as np

np.random.seed(42)

X = np.random.rand(100, 5) * 10
X[:, 1] = X[:, 0] * 0.5 + np.random.rand(100) * 2  
X[:, 2] = X[:, 0] * -0.2 + np.random.rand(100) * 3

X_centered = X - np.mean(X, axis=0)


U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

n_samples = X.shape[0]
explained_variance = (S ** 2) / (n_samples - 1)


total_variance = explained_variance.sum()
explained_variance_ratio = explained_variance / total_variance


plt.figure(figsize=(7, 4))
plt.plot(
    range(1, len(explained_variance_ratio) + 1),
    explained_variance_ratio,
    marker='o',
    label='Individual explained variance'
)
plt.plot(
    range(1, len(explained_variance_ratio) + 1),
    np.cumsum(explained_variance_ratio),
    marker='s',
    linestyle='--',
    label='Cumulative variance'
)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance (SVD)')
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[8], line 28
         23 total_variance = explained_variance.sum()
         24 explained_variance_ratio = explained_variance / total_variance
    ---> 28 plt.figure(figsize=(7, 4))
         29 plt.plot(
         30     range(1, len(explained_variance_ratio) + 1),
         31     explained_variance_ratio,
         32     marker='o',
         33     label='Individual explained variance'
         34 )
         35 plt.plot(
         36     range(1, len(explained_variance_ratio) + 1),
         37     np.cumsum(explained_variance_ratio),
       (...)
         40     label='Cumulative variance'
         41 )
    

    NameError: name 'plt' is not defined





## Free-text answer (Part 3)



In 2–5 sentences:



- Describe the *shape* of the explained-variance curve.

- Explain why it looks like that for this dataset.



Write your explanation below this line.



```python
# Part 4: 4 points

# Calculate the approximating matrix of problem1_data using the first problem1_num_components singular values and store it in the variable bel
import numpy as np

problem1_data = np.array([
    [3, 1, 1],
    [1, 3, -1]
], dtype=float)

problem1_num_components = 1 
if not isinstance(problem1_data, np.ndarray) or problem1_data.ndim != 2:
    raise ValueError("problem1_data must be a 2D NumPy array.")
if not isinstance(problem1_num_components, int) or problem1_num_components <= 0:
    raise ValueError("problem1_num_components must be a positive integer.")
if problem1_num_components > min(problem1_data.shape):
    raise ValueError("problem1_num_components cannot exceed min(matrix dimensions).")

U, S, VT = np.linalg.svd(problem1_data, full_matrices=False)

U_k = U[:, :problem1_num_components]
S_k = np.diag(S[:problem1_num_components])
VT_k = VT[:problem1_num_components, :]

problem1_approximation = U_k @ S_k @ VT_k

print("Approximating matrix using", problem1_num_components, "components:\n", problem1_approximation)

 

# Calculate the reconstruction error of problem1_data using problem1_approximation and store it in the variable below (should have shape (n_samples,)) (row wise Euclidean distance)
problem1_reconstruction_error = 

# Put the code below to plot the empirical distribution function of the reconstruction error
# You can use the Utils.py file for plotting the empirical distribution function, makeEDF and plotEDF functions

import os
import numpy as np
import matplotlib.pyplot as plt
import pyedflib

def read_edf(file_path):
 
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EDF file not found: {file_path}")
    try:
        f = pyedflib.EdfReader(file_path)
    except OSError as e:
        raise RuntimeError(f"Error opening EDF file: {e}")




# Store the value of the selected threshold in the variable below
problem1_threshold = 

# Finally store the samples of problem1_data that have a reconstruction error larger than problem1_threshold in the variable below, should have shape (100, n_dimensions)
problem1_outliers = 
```

---
## Exam vB, PROBLEM 2
Maximum Points = 12


In this problem we are interested in **account takeover (ATO) detection** for an online service. You are given the outputs of a classifier that predicts the probability that a login attempt is malicious ($Y=1$). Your goal is to explore how the **decision threshold** affects business cost (as in the thresholding assignment).

A threshold $t \in [0,1]$ is used to convert predicted probabilities $\hat p(x)=P(Y=1\mid x)$ to labels: predict $\hat y=1$ if $\hat p(x)\ge t$, else $\hat y=0$.

The costs associated with the decisions are:

* **True Positive (TP)**: correctly flagging an ATO login costs **80** (extra verification + friction).
* **True Negative (TN)**: allowing a legitimate login has **0** cost.
* **False Positive (FP)**: incorrectly flagging a legitimate login costs **150** (support load + churn risk).
* **False Negative (FN)**: missing an ATO login costs **900** (fraud + remediation).

**The code cells contain more detailed instructions; THE FIRST CODE CELL INITIALIZES YOUR VARIABLES.**

1. **[3p]** Complete the function `problem2_avg_cost` to compute the **average cost per sample** of a model under a certain prediction threshold. Plot the cost as a function of the threshold (using the validation data provided in the first code cell of this problem), from 0 to 1 with step size 0.01.
2. **[2.5p]** Find the threshold that minimizes the cost and calculate the cost at that threshold on the validation data. Also calculate the precision and recall at the optimal threshold, treating **class 1 as the positive class** and **class 0 as the positive class** separately.
3. **[2.5p]** Repeat step 2, but this time find the best threshold to **maximize accuracy** (equivalently, minimize the $0{-}1$ loss). Calculate the difference in cost between the threshold found in part 3 and the one found in part 2.
4. **[4p]** Provide a confidence interval around the optimal cost (with $95\%$ confidence) applied to the test data using **Hoeffding's inequality**, and explain all assumptions you made.


```python

# RUN THIS CELL TO GET THE DATA

# We start by loading the data 

import pandas as pd

PROBLEM2_DF = pd.read_csv('data/fraud.csv')
Y = PROBLEM2_DF['Class'].values
X = PROBLEM2_DF[['V%d' % i for i in range(1,5)]+['Amount']].values

# We will split the data into training, testing and validation sets
from Utils import train_test_validation
PROBLEM2_X_train, PROBLEM2_X_test, PROBLEM2_X_val, PROBLEM2_y_train, PROBLEM2_y_test, PROBLEM2_y_val = train_test_validation(X,Y,shuffle=True,random_state=1)

# From this we will train a logistic regression model with scaling and simple hyperparameter search
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='liblinear', max_iter=1000, random_state=1))
])

param_grid = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__class_weight': [None, 'balanced']
}

gs = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
gs.fit(PROBLEM2_X_train, PROBLEM2_y_train)

# use the best pipeline (it supports predict_proba)
lr = gs.best_estimator_

# THE FOLLOWING CODE WILL PRODUCE THE ARRAYS YOU NEED FOR THE PROBLEM
PROBLEM2_y_pred_proba_val = lr.predict_proba(PROBLEM2_X_val)[:,1]
PROBLEM2_y_true_val = PROBLEM2_y_val

PROBLEM2_y_pred_proba_test = lr.predict_proba(PROBLEM2_X_test)[:,1]
PROBLEM2_y_true_test = PROBLEM2_y_test
```


```python

# Part 1: 3 points
# Implement the following function that calculates the average cost per sample of a binary classifier
# according to the specification in the problem statement.
# See the comments inside the function for details of the parameters.
import numpy as np
import matplotlib.pyplot as plt

def problem2_avg_cost(y_true, y_predict_proba, threshold):
     
   cost = np.mean(y_predict_proba ! = y_true)
   return cost

# then compute the cost based on the confusion matrix entries (TP, TN, FP, FN).
thresholds = np.arrange([0.01, 0.1, 1, 10])
avg_costs = [problem2_avg_cost(y_val,y_val_pred_prob ,t)for t in threshold]

plt.figure(figsize=(8, 5))
plt.plot(thresholds, avg_costs, marker='o', markersize=3)
plt.xlabel("Threshold")
plt.ylabel("Average Cost per Sample")
plt.title("Cost vs. Threshold")
plt.grid(True)
plt.show()
    # y_true is a numpy array of shape (n_samples,) with binary labels (1 = ATO, 0 = legitimate)
    # y_predict_proba is a numpy array of shape (n_samples,) with predicted probabilities for class 1
    # threshold is a float between 0 and 1
    
    # Return the average cost per sample (a single float value).
    # Hint: Convert probabilities to predictions using >= threshold,
    # then compute the cost based on the confusion matrix entries (TP, TN, FP, FN).


# Provide the code below to plot the cost as a function of the threshold
# using the validation data, specifically the arrays PROBLEM2_y_true_val and PROBLEM2_y_pred_proba_val.
# The plot should range from threshold 0 to 1 with step size 0.01.
# The y-axis should be the average cost and the x-axis should be the threshold.


```


```python
from sklearn.metrics import problem2_precision,

pred_test = model.predict(X_test)

precision_0 = precision_score(Y_test, pred_test, pos_label=0)
precision_1 = precision_score(Y_test, pred_test, pos_label=1)

recall_0 = recall_score(Y_test, pred_test, pos_label=0)
recall_1 = recall_score(Y_test, pred_test, pos_label=1)

```


```python
# Part 2: 2.5 points
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score

precision, recall, thresholds = precision_recall_curve(y_true_test,y_probs)
fscore = (2 * precision * recall) / (precision + recall)
best_threshold = thresholds[np.argmax(fscore)]
print(f"Best threshold: {best_threshold}")
      
# on the validation data (PROBLEM2_y_true_val and PROBLEM2_y_pred_proba_val).
# Store the optimal threshold in the variable below.
thresholds = np.linspace(0.05, 0.95, 95)

best_cost = float("inf")
best_t = None

for t in thresholds:
    c = cost(model, t, X_valid, Y_valid)
    if c < best_cost:
        best_cost = c
        best_t = t

optimal_threshold = best_t
problem2_threshold = best_cost
 # A float between 0 and 1

# Calculate the average cost at the optimal threshold on the validation data.
# Store the cost in the variable below.

delta = 0.05
a, b = 0.0, 1.0
n_valid = len(Y_valid)

cost_at_optimal_threshold_valid = cost(model, optimal_threshold, X_valid, Y_valid)

epsilon = (b - a) * np.sqrt(np.log(2/delta) / (2*n_valid))

cost_interval_valid = (
    cost_at_optimal_threshold_valid - epsilon,
    cost_at_optimal_threshold_valid + epsilon
)
problem2_cost_val = cost_at_optimal_threshold_valid

# Using the optimal threshold, compute the predicted labels on the validation data.
# Store the predicted labels in the variable below.
problem2_y_pred_val = XXX  # A numpy array of shape (n_samples,) with values 0 or 1

# Calculate precision and recall treating class 1 as the positive class.
problem2_precision_1 =  precision_score(Y_test, pred_test, pos_label=1)
problem2_recall_1 = recall_score(Y_test, pred_test, pos_label=0)

# Calculate precision and recall treating class 0 as the positive class.
problem2_precision_0 = precision_score(Y_test, pred_test, pos_label=0)
problem2_recall_0 = recall_score(Y_test, pred_test, pos_label=0)

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 6
          3 from sklearn.metrics import precision_recall_curve
          4 from sklearn.metrics import precision_score, recall_score
    ----> 6 precision, recall, thresholds = precision_recall_curve(y_true_test,y_probs)
          7 fscore = (2 * precision * recall) / (precision + recall)
          8 best_threshold = thresholds[np.argmax(fscore)]
    

    NameError: name 'y_true_test' is not defined



```python
# Part 3: 2.5 points

# Find the threshold that **maximizes accuracy** on the validation data
# (equivalently, minimizes the 0-1 loss).
# Use the validation arrays PROBLEM2_y_true_val and PROBLEM2_y_pred_proba_val.
# Store the accuracy-optimal threshold in the variable below.
import numpy as np
from sklearn.metrics import accuracy_score


if not isinstance(Problem2_y_true_val, np.ndarray) or not isinstance(Problem2_y_pred_proba_val, np.ndarray):
    raise TypeError("Both Problem2_y_true_val and Problem2_y_pred_proba_val must be NumPy arrays.")

if PROBLEM2_y_true_val.shape[0] != Problem2_y_pred_proba_val.shape[0]:
    raise ValueError("Arrays must have the same length.")

if np.any((PROBLEM2_y_pred_proba_val < 0) | (Problem2_y_pred_proba_val > 1)):
    raise ValueError("Predicted probabilities must be between 0 and 1.")


best_threshold = 0.5 
best_accuracy = -1


 preds = (Problem2_y_pred_proba_val >= t).astype(int)
 acc = accuracy_score(Problem2_y_true_val, preds)
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = t

problem2_threshold = best_threshold

print(f"Best threshold: {accuracy_optimal_threshold:.4f} with accuracy: {best_accuracy:.4f}")


# Calculate the difference in average cost between:
#   cost at accuracy-optimal threshold (this part) minus cost at cost-optimal threshold (part 2).
# That is: problem2_avg_cost(..., problem2_threshold_acc) - problem2_avg_cost(..., problem2_threshold)
problem2_cost_difference = cost(model, optimal_threshold, X_valid, Y_valid)  # average cost

```


```python
# Part 4: 4 points

# Using the cost-optimal threshold from part 2 (problem2_threshold), apply Hoeffding's inequality
# to provide a 95% confidence interval for the **average cost** on the test data.
# Use the test arrays PROBLEM2_y_true_test and PROBLEM2_y_pred_proba_test.
# Store the lower and upper bounds of the confidence interval in the variables below.

delta = 0.05       
a, b = 0.0, 1.0     
n = len(Y_valid)

epsilon = (b - a) * np.sqrt(np.log(2.0 / delta) / (2.0 * n))

ci_low  = max(a, val_cost - epsilon)
ci_high = min(b, val_cost + epsilon)

print("Optimal threshold:", optimal_threshold)
print("Validation average cost:", val_cost)
print("95% Hoeffding CI: [{:.6f}, {:.6f}]".format(ci_low, ci_high))
print("Half-width (epsilon):", epsilon)

problem2_cost_val = cost_at_optimal_threshold_valid
problem2_lower_bound = val_cost
problem2_upper_bound = (ci_low, ci_high)
```


## Free text answer

Put your explanation for part 4 below this line in this **cell**. Double-click to enter edit mode.

In particular, clearly state:
1. Why Hoeffding's inequality applies (or approximately applies) in this context.
2. What random variables are assumed i.i.d. and what their support is.
3. What bound you used for the per-sample cost range (i.e., $C_{\max} - C_{\min} = ?$ for this problem).

---
## Exam vB, PROBLEM 3
Maximum Points = 14


A courier company monitors its trucks across four states:

*   **Downtown (D)**
*   **Suburbs (S)**
*   **Countryside (C)**
*   **Maintenance (M)** 

The transition probabilities between states at each time step are given by the matrix:

| Current State | D    | S    | C    | M    |
| ------------- | ---- | ---- | ---- | ---- |
| D             | 0.25 | 0.35 | 0.30 | 0.10 |
| S             | 0.20 | 0.40 | 0.30 | 0.10 |
| C             | 0.15 | 0.35 | 0.40 | 0.10 |
| M             | 0.00 | 0.00 | 0.00 | 1.00 |

1. If a truck starts in the **Suburbs**, what is the probability that it eventually ends up in **Maintenance**? [1p]
2. If a truck starts in **Downtown**, what is the probability that it will be in **Countryside** after five time steps? [2p]
3. Starting from **Downtown**, what is the expected number of steps before entering **Maintenance**? [3p] \
    **Hint**:
    To compute approximatively you could **simulate** but this gives a max score of [1.5p].
    To compute exactly use first-step analysis: 
$$
\text{Expected time from a state} = 1 + \sum_{\text{next states}} \big( \text{transition probability} \times \text{expected time from next state} \big)
$$

4. Is this Markov chain irreducible? Is it aperiodic? [2p]
5. Does this chain have a stationary distribution? If yes, compute it; if not, explain why. [2p]
6. Given that the truck starts in **Countryside** what is the probability that **the last state visited** before reaching **Maintenance** is **Suburbs**? [4p]  
**Hint**: To compute approximatively you could **simulate** but this gives a max score of [2p]. To compute exactly use first-step analysis: Let $f_D, f_S, f_C$ be the probabilities that the last state before hitting Maintenance is Suburbs, starting from Downtown, Suburbs, and Countryside respectively. Write recursive equations by conditioning on the next step. This gives a linear system to solve.



```python
import numpy as np

# State order: [Downtown, Suburbs, Countryside,Maintenance]
P = np.array([
    [0.25, 0.35, 0.30,0.10],  # from Downtown
    [0.20, 0.40, 0.30,0.10],  # from Suburbs
    [0.15, 0.35, 0.40,0.10],  # from Countryside
    [0.00, 0.00, 0.00,1.00]   # from Maintenance
], dtype=float)

# Two-step transition matrix
P = P @ P

prob =P[1, 0]
prob

```




    np.float64(0.16749999999999998)




```python
# Part 1

# Fill in the answer to part 1 below as a decimal number (float)
# If a truck starts in the Suburbs, what is the probability that it eventually ends up in Maintenance?
problem3_prob_maintenance_from_suburbs = 0.17500000000000002
```


```python
import numpy as np
# State order: [Downtown, Suburbs, Countryside,Maintenance]

states = ["Downtown", "Countryside"]
transition_matrix = np.array([[0.25, 0.35, 0.30,0.10],
                             [0.15, 0.35, 0.40,0.10]])
n_steps = 5
current_state = 0

print(states[current_state], end=" -> ")
for _ in range(n_steps - 1):
    current_state = np.random.choice(
        [0, 1], p=transition_matrix[current_state])
    print(states[current_state], end=" -> ")
print("stop")

```


```python
# Part 2

# Fill in the answer to part 2 below as a decimal number (float)
# If a truck starts in Downtown, what is the probability that it will be in Countryside after five time steps?
problem3_prob_countryside_after_5_steps = 0.16749999999999998
```


```python
import numpy as np

def expected_steps_to_absorption (P):
P = p.array([
 [0.25, 0.35, 0.30,0.10],  # from Downtown
    [0.20, 0.40, 0.30,0.10],  # from Suburbs
    [0.15, 0.35, 0.40,0.10],  # from Countryside
    [0.00, 0.00, 0.00,1.00]   # from Maintenance
])
P()
# Compute expected steps
transient_states, steps = expected_steps_to_absorption(P)

# Display results
for state, exp_steps in zip(transient_states, steps):
    print(f"Expected steps to absorption from state {state}: {exp_steps:.2f}")
```


      Cell In[44], line 4
        P = p.array([
        ^
    IndentationError: expected an indented block after function definition on line 3
    



```python
# Part 3

# Fill in the answer to part 3 below as a decimal number (float)
# Starting from Downtown, what is the expected number of steps before entering Maintenance?
problem3_expected_steps_downtown = 3
```


```python
import numpy as np
from collections import deque

def is_irreducible(P, tol=1e-12):
    """
    Returns True if the Markov chain with transition matrix P is irreducible.
    Irreducible means: every state can reach every other state (via >0-probability paths).
    """
    P = np.array(P, dtype=float)
    n = P.shape[0]

    # Build adjacency list: edge i -> j if P[i,j] > 0
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if P[i, j] > tol:
                adj[i].append(j)

    # From every start state, BFS and see if all states are reachable
    for start in range(n):
        seen = [False] * n
        q = deque([start])
        seen[start] = True

        while q:
            u = q.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)

        if not all(seen):
            return False

    return True

```


```python
from math import gcd
from functools import reduce

def is_aperiodic(P, max_power=100):
    n = P.shape[0]
    powers = np.eye(n)
    return_times = []

    for k in range(1, max_power + 1):
        powers = powers @ P
        if powers[0, 0] > 0:  # return to state 0
            return_times.append(k)

    if len(return_times) == 0:
        return False

    period = reduce(gcd, return_times)
    return period == 1
```


```python
# Part 4

# Fill in the answers to part 4 below as booleans (use True or False)
# Is this Markov chain irreducible? Is it aperiodic?
problem3_is_irreducible = True
problem3_is_aperiodic = False
```


```python
import numpy as np

def stationary_dist(P):
    P = np.array(P)
    eigenvals, eigenvecs = np.linalg.eig(P.T)

    # find the index of the eigenvalue that is 1
    index = np.argmin(np.abs(eigenvals - 1))

    stat = eigenvecs[:, index].flatten() # getting eigenvec with eigenval 1
    stat = stat / np.sum(stat) #normalizing vector
    return stat
```


```python
# Part 5

# Fill in the answer to part 5 below (if it exists)
# The answer should be a numpy array of length 4 whose entries sum to 1.
# If it does not exist, write None instead of an array.
problem3_stationary_distribution = False
```


## Free text answer (Part 5)

Briefly explain **why** the chain **does** / **does not** have a stationary distribution.

Guidance:
- If you say it **has** one, why?.
- If you say it **does not**, explain what property fails.

Write your explanation below this line.


```python
The chain does not have a stationary distribution when the state space is infinte.
```


```python
import numpy as np

# State order: [Downtown, Suburbs, Countryside,Maintenance]
P = np.array([
  
    [0.25, 0.35, 0.30,0.10],
    [0.20, 0.40, 0.30,0.10],  
    [0.15, 0.35, 0.40,0.10],
    [0.00, 0.00, 0.00,1.00]
], dtype=float)

DOWNTOWN = 0
SUBURBS  = 1
COUNTRY  = 2
Maintenance = 3

target = Maintenance

# Unknowns are h[i] for i != target, with h[target] = 0
states = [i for i in range(P.shape[0]) if i != target]   # here: [Countrysid,Maintenance]
idx = {s: k for k, s in enumerate(states)}

A = np.zeros((len(states), len(states)))
b = np.ones(len(states))

# Build equations: h[s] - sum_{j!=target} P[s,j]*h[j] = 1
for s in states:
    row = idx[s]
    A[row, row] = 1.0
    for j in states:
        A[row, idx[j]] -= P[s, j]

h = np.linalg.solve(A, b)

problem3_prob_last_suburbs_from_countryside = h[idx[Maintenanc]]
problem3_prob_last_suburbs_from_countryside 
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[51], line 35
         31         A[row, idx[j]] -= P[s, j]
         33 h = np.linalg.solve(A, b)
    ---> 35 problem3_prob_last_suburbs_from_countryside = h[idx[Maintenanc]]
         36 problem3_prob_last_suburbs_from_countryside
    

    NameError: name 'Maintenanc' is not defined



```python
# Part 6

# Fill in the answer to part 6 below as a decimal number (float)
# Given that the truck starts in Countryside, what is the probability that the last state visited before reaching Maintenance is Suburbs?
problem3_prob_last_suburbs_from_countryside = 0.035
```
