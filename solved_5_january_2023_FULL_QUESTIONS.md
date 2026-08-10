# Solved Exam Notebook — 5th of January 2023

Course: **1MS041 Introduction to Data Science / Introduktion till dataanalys**

This notebook contains the **whole questions**, followed by the **solved code answers with all steps, comments, outputs, plots, and free-text explanations**.

> Insert your anonymous exam ID below.


```python
examID = "XXX"
```

## Problem 1 — Full Question

Maximum Points = 14

A courier company operates a fleet of delivery trucks that make deliveries to different parts of the city. The trucks are equipped with GPS tracking devices that record the location of each truck at regular intervals. The locations are divided into three regions: **downtown**, **the suburbs**, and **the countryside**.

The following table shows the probabilities of a truck transitioning between these regions at each time step:

| Current region | Probability of transitioning to downtown | Probability of transitioning to the suburbs | Probability of transitioning to the countryside |
|---|---:|---:|---:|
| Downtown | 0.3 | 0.4 | 0.3 |
| Suburbs | 0.2 | 0.5 | 0.3 |
| Countryside | 0.4 | 0.3 | 0.3 |

1. **[2p]** If a truck is currently in the suburbs, what is the probability that it will be in the downtown region after two time steps?

2. **[2p]** If a truck is currently in the suburbs, what is the probability that it will be in the downtown region the first time after two time steps?

3. **[3p]** Is this Markov chain irreducible? Explain your answer.

4. **[3p]** What is the stationary distribution?

5. **[4p] Advanced question:** What is the expected number of steps until the first time one enters the suburbs region having started in the downtown region? Hint: to get within 1 decimal point, it is enough to compute the probabilities for hitting times below 30. Motivate your answer in detail. You could also solve this question by simulation, but this gives you a maximum of 2p.

### Problem 1 — Solution idea

Use state order:

\[
(\text{Downtown}, \text{Suburbs}, \text{Countryside})
\]

The transition matrix is

\[
P=
\begin{pmatrix}
0.3&0.4&0.3\\
0.2&0.5&0.3\\
0.4&0.3&0.3
\end{pmatrix}.
\]

Part 1 is entry \((P^2)_{\text{Suburbs},\text{Downtown}}\).

For Part 2, “first time after two time steps” means we are starting in Suburbs and ask for first visit to Downtown at time 2. So at time 1 we must **not** be in Downtown, and at time 2 we must be in Downtown.

For Part 5, solve expected hitting time equations for reaching Suburbs starting from Downtown.


```python
# Problem 1 imports
import numpy as np

# State order: Downtown, Suburbs, Countryside
P = np.array([
    [0.3, 0.4, 0.3],
    [0.2, 0.5, 0.3],
    [0.4, 0.3, 0.3]
])

P
```


```python
# Part 1
# Probability of being in Downtown after 2 steps when starting from Suburbs.
# State index: Downtown=0, Suburbs=1, Countryside=2.
import numpy as np

# State order: Downtown, Suburbs, Countryside
P = np.array([
    [0.3, 0.4, 0.3],
    [0.2, 0.5, 0.3],
    [0.4, 0.3, 0.3]
])

P2 = np.linalg.matrix_power(P, 2)
problem1_p1 = float(P2[1, 0])

problem1_p1
```




    0.28




```python
# Part 2
# First time hitting Downtown after exactly 2 steps, starting in Suburbs.
# Paths must avoid Downtown at step 1 and hit Downtown at step 2.
#
# Possible paths:
# Suburbs -> Suburbs -> Downtown: 0.5 * 0.2
# Suburbs -> Countryside -> Downtown: 0.3 * 0.4

problem1_p2 = float(P[1, 1] * P[1, 0] + P[1, 2] * P[2, 0])

problem1_p2
```

### Problem 1 Part 3 — Free-text answer

Yes, this Markov chain is **irreducible**.

Reason: every state can reach every other state with positive probability. From each of Downtown, Suburbs, and Countryside, there is a positive probability of moving to the other regions either directly or through another state. Therefore all states communicate, so the chain is irreducible.


```python
# Part 3
problem1_irreducible = True
problem1_irreducible
```


```python
# Stationary distribution: pi P = pi, sum(pi)=1

import numpy as np
import math
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


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[2], line 5
          3 import numpy as np
          4 import math
    ----> 5 eigvals, eigvecs = np.linalg.eig(P.T)
          6 idx = np.argmin(np.abs(eigvals - 1))
          8 pi = np.real(eigvecs[:, idx])
    

    NameError: name 'P' is not defined



```python
# Part 4
# Stationary distribution pi solves pi P = pi and sum(pi)=1.
# Use the eigenvector of P.T with eigenvalue 1.

eigenvalues, eigenvectors = np.linalg.eig(P.T)
idx = np.argmin(np.abs(eigenvalues - 1))
pi = np.real(eigenvectors[:, idx])
pi = pi / np.sum(pi)

problem1_stationary = pi
problem1_stationary
```




    array([0.28888889, 0.41111111, 0.3       ])



### Problem 1 Part 4 — Free-text answer

The stationary distribution \(\pi\) satisfies

\[
\pi P=\pi,\qquad \sum_i \pi_i=1.
\]

The code above finds it from the eigenvector of \(P^T\) with eigenvalue 1. Since the chain is finite and irreducible, this stationary distribution is unique. Its entries represent the long-run proportion of time spent in Downtown, Suburbs, and Countryside.


```python
# Part 5
# Expected number of steps to first enter Suburbs starting from Downtown.
#
# Let h_D = expected time to hit Suburbs from Downtown.
# Let h_C = expected time to hit Suburbs from Countryside.
# h_S = 0 because if we are in Suburbs, the hitting time is already 0.
#
# Equations:
# h_D = 1 + 0.3 h_D + 0.3 h_C   (transition to Suburbs contributes 0 after the step)
# h_C = 1 + 0.4 h_D + 0.3 h_C

A = np.array([
    [0.7, -0.3],
    [-0.4, 0.7]
])
b = np.array([1.0, 1.0])

h_D, h_C = np.linalg.solve(A, b)

problem1_ET = float(h_D)
problem1_ET
```




    2.702702702702703



### Problem 1 Part 5 — Free-text answer

Let \(h_D\) be the expected number of steps to first reach Suburbs starting from Downtown, and \(h_C\) the same quantity starting from Countryside. Since the target state is Suburbs,

\[
h_S=0.
\]

Using first-step analysis:

\[
h_D = 1 + 0.3h_D + 0.4h_S + 0.3h_C
\]

and because \(h_S=0\),

\[
h_D = 1 + 0.3h_D + 0.3h_C.
\]

Similarly,

\[
h_C = 1 + 0.4h_D + 0.3h_S + 0.3h_C
\]

so

\[
h_C = 1 + 0.4h_D + 0.3h_C.
\]

Solving the linear system gives the expected number of steps from Downtown, stored in `problem1_ET`.

## Problem 2 — Full Question

Maximum Points = 13

You are given the **Abalone** dataset found in `data/abalone.csv`, which contains physical measurements of abalone (a type of sea shells) and the age of the abalone measured in rings. Your task is to train a linear regression model to predict the age (`Rings`) of an abalone based on its physical measurements.

To evaluate your model, you will split the dataset into a training set and a testing set. You will use the training set to train your model, and the testing set to evaluate its performance.

1. **[2p]** Load the data into a pandas dataframe `problem2_df`. Based on the column names, figure out what are the features and the target and fill in the answer in the correct cell below.

2. **[2p]** Split the data into train and test.

3. **[1p]** Train the model.

4. **[3p]** On the test set, evaluate the model by computing the mean absolute error and plot the empirical distribution function of the residual with confidence bands, i.e. using the DKW inequality and 95% confidence. Hint: you can use the function `plotEDF`, `makeEDF` combo from `Utils.py`.

5. **[2p]** Provide a scatter plot where the x-axis corresponds to the predicted value and the y-axis is the true value, over the test set.

6. **[3p]** Reason about the performance, for instance, is the value of the mean absolute error good/bad and what do you think about the scatter plot in point 5?

### Problem 2 — Solution idea

The target is `Rings`, because the task is to predict abalone age measured by rings.

Features should be physical measurements, for example length, diameter, height, and weights. If a categorical column such as `Sex` exists, we one-hot encode it before fitting linear regression.

A good metric is **Mean Absolute Error (MAE)** because it tells us the average absolute error in predicted rings.


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
problem2_df = pd.read_csv("abalone.csv")
problem2_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Length</th>
      <th>Diameter</th>
      <th>Height</th>
      <th>Whole weight</th>
      <th>Shucked weight</th>
      <th>Viscera weight</th>
      <th>Shell weight</th>
      <th>Rings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.455</td>
      <td>0.365</td>
      <td>0.095</td>
      <td>0.5140</td>
      <td>0.2245</td>
      <td>0.1010</td>
      <td>0.150</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.350</td>
      <td>0.265</td>
      <td>0.090</td>
      <td>0.2255</td>
      <td>0.0995</td>
      <td>0.0485</td>
      <td>0.070</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.530</td>
      <td>0.420</td>
      <td>0.135</td>
      <td>0.6770</td>
      <td>0.2565</td>
      <td>0.1415</td>
      <td>0.210</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.440</td>
      <td>0.365</td>
      <td>0.125</td>
      <td>0.5160</td>
      <td>0.2155</td>
      <td>0.1140</td>
      <td>0.155</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.330</td>
      <td>0.255</td>
      <td>0.080</td>
      <td>0.2050</td>
      <td>0.0895</td>
      <td>0.0395</td>
      <td>0.055</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Part 1
# Target is Rings.
problem2_target = "Rings"

# Use all columns except target as features.
# If there is a categorical column like Sex, one-hot encode it.
problem2_df_model = pd.get_dummies(problem2_df, drop_first=True)

problem2_features = [col for col in problem2_df_model.columns if col != problem2_target]

problem2_X = problem2_df_model[problem2_features]
problem2_y = problem2_df_model[problem2_target]

problem2_features, problem2_target
```




    (['Length',
      'Diameter',
      'Height',
      'Whole weight',
      'Shucked weight',
      'Viscera weight',
      'Shell weight'],
     'Rings')




```python
# Part 2

from sklearn.model_selection import train_test_split

X = problem2_df[problem2_features]
y = problem2_df[problem2_target]

# If there is a categorical column like Sex, convert it using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

problem2_X_train, problem2_X_test, problem2_y_train, problem2_y_test = train_test_split(
    X,
    y,
    train_size=0.8,
    random_state=42
)
```


```python
# Part 2
problem2_X_train, problem2_X_test, problem2_y_train, problem2_y_test = train_test_split(
    problem2_X,
    problem2_y,
    train_size=0.8,
    random_state=42
)

problem2_X_train.shape, problem2_X_test.shape, problem2_y_train.shape, problem2_y_test.shape
```




    ((3341, 7), (836, 7), (3341,), (836,))




```python
# Part 3
problem2_model = LinearRegression()
problem2_model.fit(problem2_X_train, problem2_y_train)

problem2_model
```




<style>.sk-global {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;
}

.sk-global.light {
  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: black;
  --sklearn-color-background: white;
  --sklearn-color-border-box: black;
  --sklearn-color-icon: #696969;
}

.sk-global.dark {
  --sklearn-color-text-on-default-background: white;
  --sklearn-color-background: #111;
  --sklearn-color-border-box: white;
  --sklearn-color-icon: #878787;
}

.sk-global {
  color: var(--sklearn-color-text);
}

.sk-global pre {
  padding: 0;
}

.sk-global input.sk-hidden--visually {
  border: 0;
  clip-path: inset(100%);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

.sk-global div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

.sk-global div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

.sk-global div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

.sk-global div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

.sk-global div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

.sk-global div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

.sk-global div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

.sk-global div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

.sk-global div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

.sk-global div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

.sk-global div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
.sk-global label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: center;
  justify-content: center;
  gap: 0.5em;
}

.sk-global label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

.sk-global label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

.sk-global label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

.sk-global div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

.sk-global div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

.sk-global input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

.sk-global input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

.sk-global div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
.sk-global div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

.sk-global div.sk-label label.sk-toggleable__label,
.sk-global div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
.sk-global div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
.sk-global div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

.sk-global div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  line-height: 1.2em;
}

.sk-global div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
.sk-global div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

.sk-global div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
.sk-global div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

.sk-global div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-3) 1pt solid;
  color: var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3) 1pt solid;
  color: var(--sklearn-color-fitted-level-3);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  border: var(--sklearn-color-fitted-level-0) 1pt solid;
  color: var(--sklearn-color-unfitted-level-0);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  border: var(--sklearn-color-fitted-level-0) 1pt solid;
  color: var(--sklearn-color-fitted-level-0);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

.sk-global a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

.sk-global a.estimator_doc_link.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
.sk-global a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

.sk-global a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.sk-top-container.sk-global {
  /* pydata-sphinx-theme hides overflow, so scrolling is disabled.
   We need to set it to !important and add tabindex="0" in the HTML
   to allow keyboard-only users to navigate the display. */
  overflow-x: scroll !important;
  max-width: 100%;
}

.estimator-table {
    font-family: monospace;
}

.estimator-table summary {
    padding: .5rem;
    cursor: pointer;
}

.estimator-table summary::marker {
    font-size: 0.7rem;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
    margin-top: 0;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover td {
    background-color: #e0e0e0;
}

.estimator-table table :is(td, th) {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

/*
    `table td`is set in notebook with right text-align.
    We need to overwrite it.
*/
.estimator-table table td.param {
    text-align: left;
    position: relative;
    padding: 0;
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left !important;
}

.user-set td.value {
    color:rgb(255, 94, 0);
    background-color: transparent;
}

.default td, .estimator-table th {
    color: black;
    text-align: left !important;
}

.user-set td i,
.default td i {
    color: black;
}

td.fitted-att-type {
    white-space: preserve nowrap;
}

/*
    Styles for parameter documentation links
    We need styling for visited so jupyter doesn't overwrite it
*/
a.param-doc-link,
a.param-doc-link:link,
a.param-doc-link:visited {
    text-decoration: underline dashed;
    text-underline-offset: .3em;
    color: inherit;
    display: block;
    padding: .5em;
}

@supports(anchor-name: --doc-link) {
    a.param-doc-link,
    a.param-doc-link:link,
    a.param-doc-link:visited {
    anchor-name: --doc-link;
    }
}

/* "hack" to make the entire area of the cell containing the link clickable */
a.param-doc-link::before {
    position: absolute;
    content: "";
    inset: 0;
}

.param-doc-description {
    display: none;
    position: absolute;
    z-index: 9999;
    left: 0;
    padding: .5ex;
    margin-left: 1.5em;
    color: var(--sklearn-color-text);
    box-shadow: .3em .3em .4em #999;
    width: max-content;
    text-align: left;
    max-height: 10em;
    overflow-y: auto;

    /* unfitted */
    background: var(--sklearn-color-unfitted-level-0);
    border: thin solid var(--sklearn-color-unfitted-level-3);
}

@supports(position-area: center right) {
    .param-doc-description {
    position-area: center right;
    position: fixed;
    margin-left: 0;
    }
}

/* Fitted state for parameter tooltips */
.fitted .param-doc-description {
    /* fitted */
    background: var(--sklearn-color-fitted-level-0);
    border: thin solid var(--sklearn-color-fitted-level-3);
}

.param-doc-link:hover .param-doc-description {
    display: block;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}

.features {
  font-family: monospace;
  cursor: pointer;
  background-color: var(--sklearn-color-unfitted-level-0);
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: .20em;
  margin-bottom: 0.5em;
  font-size: inherit; /* Needed for jupyter */
}

.features.fitted {
  background-color: var(--sklearn-color-fitted-level-0);
}

.features summary {
  cursor: pointer;
  display: flex;
  margin-bottom: 0;
  text-align: center;
  align-items: center;
  justify-content: center;
  gap: 0.5em;
  padding: .25em;
}

.features details[open] > summary {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
  border-radius: .20em 0 0 0;
}

.features.fitted details[open] > summary {
  background-color: var(--sklearn-color-fitted-level-2);
  border-radius: .20em 0 0 0;
}

.features details > summary .arrow::before {
  content: "▸";
  color: grey;
}

.features details[open] > summary .arrow::before {
  content: "▾";
}

.features details:hover > summary {
  margin: 0;
  background-color: var(--sklearn-color-unfitted-level-2);
}

.features.fitted details:hover > summary {
  margin: 0;
  background-color: var(--sklearn-color-fitted-level-2);
}

.features .features-container {
  max-width: 15em;
  max-height: 10em;
  overflow: auto;
  scrollbar-width: thin;
  padding: .25em 0.1rem;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 0 0 .5em .5em;
}

.features.fitted .features-container {
  background-color: var(--sklearn-color-fitted-level-0);
}

.features .image-container {
  block-size: 1em;
  inline-size: 1em;
  padding: 0;
  margin: 0%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.features .copy-paste-icon {
  background-size: 1em 1em;
  width: 1em;
  height: 1em;
  filter: grayscale(100%) opacity(60%);
}

.features .features-container table {
  width: 100%;
  margin: 0.01em;
}

.features .features-container table tr:nth-child(odd) {
  background-color: #fff;
}

.features .features-container table tr:nth-child(even) {
  background-color: #f6f6f6;
}

.features .features-container table tr:hover {
  background-color: #e0e0e0;
}

.features .features-container table {
  table-layout: inherit;
}

.features .features-container table td {
  text-align: left;
  padding: 0 0.5em;
  border: 1px solid rgba(106, 105, 104, 0.232);
  white-space: nowrap;
  color: var(--sklearn-color-text);
}

.total_features {
  display: flex;
  justify-content: center;
  margin-top: 0.5em;
}
</style><body><div id="sk-container-id-1" tabindex="0" class="sk-top-container sk-global"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually sk-global" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>LinearRegression</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fit_intercept',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-fit_intercept;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LinearRegression.html#:~:text=fit_intercept,-bool%2C%20default%3DTrue">
            fit_intercept
            <span class="param-doc-description"
            style="position-anchor: --doc-link-fit_intercept;">
            fit_intercept: bool, default=True<br><br>Whether to calculate the intercept for this model. If set<br>to False, no intercept will be used in calculations<br>(i.e. data is expected to be centered).</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy_X',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-copy_X;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LinearRegression.html#:~:text=copy_X,-bool%2C%20default%3DTrue">
            copy_X
            <span class="param-doc-description"
            style="position-anchor: --doc-link-copy_X;">
            copy_X: bool, default=True<br><br>If True, X will be copied; else, it may be overwritten.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('tol',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-tol;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LinearRegression.html#:~:text=tol,-float%2C%20default%3D1e-6">
            tol
            <span class="param-doc-description"
            style="position-anchor: --doc-link-tol;">
            tol: float, default=1e-6<br><br>The precision of the solution (`coef_`) is determined by `tol` which<br>specifies the convergence criterion of the underlying solver. `tol` is<br>set as `atol` and `btol` of :func:`scipy.sparse.linalg.lsqr` when<br>fitting on sparse training data. `tol` is set as `cond` of<br>:func:`scipy.linalg.lstsq` when fitting on dense training data.<br><br>.. versionadded:: 1.7<br>.. versionchanged:: 1.9<br>    Now supported on dense data, interpreted as the `cond` parameter.</span>
        </a>
    </td>
            <td class="value">1e-06</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_jobs',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_jobs;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LinearRegression.html#:~:text=n_jobs,-int%2C%20default%3DNone">
            n_jobs
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_jobs;">
            n_jobs: int, default=None<br><br>The number of jobs to use for the computation. This will only provide<br>speedup in case of sufficiently large problems, that is if firstly<br>`n_targets &gt; 1` and secondly `X` is sparse or if `positive` is set<br>to `True`. ``None`` means 1 unless in a<br>:obj:`joblib.parallel_backend` context. ``-1`` means using all<br>processors. See :term:`Glossary &lt;n_jobs&gt;` for more details.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('positive',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-positive;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LinearRegression.html#:~:text=positive,-bool%2C%20default%3DFalse">
            positive
            <span class="param-doc-description"
            style="position-anchor: --doc-link-positive;">
            positive: bool, default=False<br><br>When set to ``True``, forces the coefficients to be positive. This<br>option is only supported for dense arrays.<br><br>For a comparison between a linear regression model with positive constraints<br>on the regression coefficients and a linear regression without such constraints,<br>see :ref:`sphx_glr_auto_examples_linear_model_plot_nnls.py`.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>

        <div class="estimator-table">
            <details>
                <summary>Fitted attributes</summary>
                <table class="parameters-table">
                    <tbody>
                        <tr>
                        <th>Name</th>
                        <th>Type</th>
                        <th>Value</th>
                        </tr>

       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-coef_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LinearRegression.html#:~:text=coef_,-array%20of%20shape%20%28n_features%2C%20%29%20or%20%28n_targets%2C%20n_features%29">
            coef_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-coef_;">
            coef_: array of shape (n_features, ) or (n_targets, n_features)<br><br>Estimated coefficients for the linear regression problem.<br>If multiple targets are passed during the fit (y 2D), this<br>is a 2D array of shape (n_targets, n_features), while if only<br>one target is passed, this is a 1D array of length n_features.</span>
        </a>
    </td>
           <td class="fitted-att-type">ndarray[float64](7,)</td>
           <td>[ -1.52, 13.48, 11.4 ,...,-20.58, -8.85,  8.64]</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-feature_names_in_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LinearRegression.html#:~:text=feature_names_in_,-ndarray%20of%20shape%20%28n_features_in_%2C%29">
            feature_names_in_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-feature_names_in_;">
            feature_names_in_: ndarray of shape (`n_features_in_`,)<br><br>Names of features seen during :term:`fit`. Defined only when `X`<br>has feature names that are all strings.<br><br>.. versionadded:: 1.0</span>
        </a>
    </td>
           <td class="fitted-att-type">ndarray[object](7,)</td>
           <td>[&#x27;Length&#x27;,&#x27;Diameter&#x27;,&#x27;Height&#x27;,...,&#x27;Shucked weight&#x27;,&#x27;Viscera weight&#x27;,
 &#x27;Shell weight&#x27;]</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-intercept_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LinearRegression.html#:~:text=intercept_,-float%20or%20array%20of%20shape%20%28n_targets%2C%29">
            intercept_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-intercept_;">
            intercept_: float or array of shape (n_targets,)<br><br>Independent term in the linear model. Set to 0.0 if<br>`fit_intercept = False`.</span>
        </a>
    </td>
           <td class="fitted-att-type">float64</td>
           <td>2.987</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-n_features_in_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LinearRegression.html#:~:text=n_features_in_,-int">
            n_features_in_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-n_features_in_;">
            n_features_in_: int<br><br>Number of features seen during :term:`fit`.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
           <td class="fitted-att-type">int</td>
           <td>7</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-rank_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LinearRegression.html#:~:text=rank_,-int">
            rank_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-rank_;">
            rank_: int<br><br>Rank of matrix `X`. Only available when `X` is dense.</span>
        </a>
    </td>
           <td class="fitted-att-type">int</td>
           <td>7</td>


       </tr>


       <tr class="default">
           <td class="param">
        <a class="param-doc-link"
            style="anchor-name: --doc-link-singular_;"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LinearRegression.html#:~:text=singular_,-array%20of%20shape%20%28min%28X%2C%20y%29%2C%29">
            singular_
            <span class="param-doc-description"
            style="position-anchor: --doc-link-singular_;">
            singular_: array of shape (min(X, y),)<br><br>Singular values of `X`. Only available when `X` is dense.</span>
        </a>
    </td>
           <td class="fitted-att-type">ndarray[float64](7,)</td>
           <td>[33.69, 3.6 , 3.1 ,..., 1.35, 1.14, 0.69]</td>


       </tr>

                    </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div><script>/*  Authors: The scikit-learn developers
 SPDX-License-Identifier: BSD-3-Clause
*/

function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.copy-paste-icon').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';

    const parent = element.parentElement;
    if (!parent || !parent.nextElementSibling) {
        console.warn('Expected copy-paste icon is missing from the DOM structure');
        return;
    }

    const paramName = element.parentElement.nextElementSibling
        .textContent.trim().split(' ')[0];
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});

/**
 * Copy the list of feature names formatted as a Python list.
 *
 * @param {HTMLElement} element - The copy button inside a `.features` block; its siblings
 *   contain a `details` element and a table containing feature named.
 * @returns {boolean} Always returns `false` so callers can prevent the default click behavior.
 */
function copyFeatureNamesToClipboard(element) {
    var detailsElem = element.closest('.features').querySelector('details');
    var wasOpen = detailsElem.open;
    detailsElem.open = true;
    var content = element.closest('.features').querySelector('tbody')
                  .innerText.trim();
    if (!wasOpen) detailsElem.open = false;
    const rows = content.split('\n').map(row => `    "${row}"`);
    const formattedText = `[\n${rows.join(',\n')},\n]`;
    const originalHTML = element.innerHTML.replace('✔', '');
    const originalStyle = element.style;
    const copyMark = document.createElement('span');
    copyMark.innerHTML = '✔';
    copyMark.style.color = 'blue';
    copyMark.style.fontSize = '1em';

    navigator.clipboard.writeText(formattedText)
        .then(() => {
            element.style.display = 'none';
            element.parentElement.appendChild(copyMark);

            setTimeout(() => {
                copyMark.remove();
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 1000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'orange';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 1000);
        });
    return false;
}
/**
 * Adapted from Skrub
 * https://github.com/skrub-data/skrub/blob/403466d1d5d4dc76a7ef569b3f8228db59a31dc3/skrub/_reporting/_data/templates/report.js#L789
 * @returns "light" or "dark"
 */
function detectTheme(element) {
    const body = document.querySelector('body');

    // Check VSCode theme
    const themeKindAttr = body.getAttribute('data-vscode-theme-kind');
    const themeNameAttr = body.getAttribute('data-vscode-theme-name');

    if (themeKindAttr && themeNameAttr) {
        const themeKind = themeKindAttr.toLowerCase();
        const themeName = themeNameAttr.toLowerCase();

        if (themeKind.includes("dark") || themeName.includes("dark")) {
            return "dark";
        }
        if (themeKind.includes("light") || themeName.includes("light")) {
            return "light";
        }
    }

    // Check Jupyter theme
    if (body.getAttribute('data-jp-theme-light') === 'false') {
        return 'dark';
    } else if (body.getAttribute('data-jp-theme-light') === 'true') {
        return 'light';
    }

    // Guess based on a parent element's color
    const color = window.getComputedStyle(element.parentNode, null).getPropertyValue('color');
    const match = color.match(/^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*$/i);
    if (match) {
        const [r, g, b] = [
            parseFloat(match[1]),
            parseFloat(match[2]),
            parseFloat(match[3])
        ];

        // https://en.wikipedia.org/wiki/HSL_and_HSV#Lightness
        const luma = 0.299 * r + 0.587 * g + 0.114 * b;

        if (luma > 180) {
            // If the text is very bright we have a dark theme
            return 'dark';
        }
        if (luma < 75) {
            // If the text is very dark we have a light theme
            return 'light';
        }
        // Otherwise fall back to the next heuristic.
    }

    // Fallback to system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}


function forceTheme(elementId) {
    const estimatorElement = document.querySelector(`#${elementId}`);
    if (estimatorElement === null) {
        console.error(`Element with id ${elementId} not found.`);
    } else {
        const theme = detectTheme(estimatorElement);
        estimatorElement.classList.add(theme);
    }
}

forceTheme('sk-container-id-1');</script></body>




```python
# Part 4
# Evaluate using MAE.
problem2_y_pred = problem2_model.predict(problem2_X_test)

problem2_mae = float(mean_absolute_error(problem2_y_test, problem2_y_pred))
problem2_rmse = float(np.sqrt(mean_squared_error(problem2_y_test, problem2_y_pred)))
problem2_r2 = float(r2_score(problem2_y_test, problem2_y_pred))

print("MAE:", problem2_mae)
print("RMSE:", problem2_rmse)
print("R^2:", problem2_r2)
```

    MAE: 1.629248267393658
    RMSE: 2.248453055836253
    R^2: 0.532984475772452
    


```python
# Part 4
# Plot empirical distribution function of residuals with 95% DKW confidence bands.

problem2_residuals = problem2_y_test.values - problem2_y_pred
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


    
![png](solved_5_january_2023_FULL_QUESTIONS_files/solved_5_january_2023_FULL_QUESTIONS_23_0.png)
    





    np.float64(0.0469709230918113)




```python
# Part 5
# Scatter plot: predicted values on x-axis, true values on y-axis.

plt.figure(figsize=(7, 5))
plt.scatter(problem2_y_pred, problem2_y_test, alpha=0.6)
min_val = min(problem2_y_pred.min(), problem2_y_test.min())
max_val = max(problem2_y_pred.max(), problem2_y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Perfect prediction")
plt.xlabel("Predicted Rings")
plt.ylabel("True Rings")
plt.title("Predicted vs true Rings")
plt.legend()
plt.show()
```


    
![png](solved_5_january_2023_FULL_QUESTIONS_files/solved_5_january_2023_FULL_QUESTIONS_24_0.png)
    


### Problem 2 Part 6 — Free-text answer

The MAE tells us the average absolute error in number of rings. For example, if the MAE is around 1.5–2 rings, then the model is typically wrong by around 1.5–2 rings.

The predicted-vs-true scatter plot should ideally lie close to the diagonal line. If the points are widely scattered, then the model is not very accurate. A simple linear regression can capture some trend from physical measurements, but abalone age is not perfectly linear in those measurements, so some prediction error is expected.

The DKW confidence band gives uncertainty around the whole residual CDF. It can be used to estimate quantiles of prediction error and understand how large errors may be.

## Problem 3 — Full Question

Maximum Points = 13

A healthcare organization is interested in understanding the relationship between the number of visits to the doctor's office and certain patient characteristics. They collected data on the number of visits for a sample of patients and included the following variables:

- `ofp`: number of physician office visits
- `ofnp`: number of nonphysician office visits
- `opp`: number of physician outpatient visits
- `opnp`: number of nonphysician outpatient visits
- `emr`: number of emergency room visits
- `hosp`: number of hospitalizations
- `exclhlth`: the person is of excellent health (self-perceived)
- `poorhealth`: the person is of poor health (self-perceived)
- `numchron`: number of chronic conditions
- `adldiff`: the person has a condition that limits activities of daily living
- `noreast`: the person is from the north east region
- `midwest`: the person is from the midwest region
- `west`: the person is from the west region
- `age`: age in years divided by 10
- `male`: is the person male?
- `married`: is the person married?
- `school`: number of years of education
- `faminc`: family income in 10000$
- `employed`: is the person employed?
- `privins`: is the person covered by private health insurance?
- `medicaid`: is the person covered by medicaid?

Decide which patient features are reasonable to use to predict the target **number of physician office visits**. Hint: should we really use the `ofnp` etc variables?

Since the target variable is counts, a reasonable loss function is to consider the target variable as Poisson distributed where

\[
\lambda = \exp(\alpha\cdot x+\beta)
\]

where \(\alpha\) is a vector and \(\beta\) is the intercept. The conditional density is

\[
f_{Y|X}(y,x)=\frac{\lambda^y e^{-\lambda}}{y!},\qquad
\lambda(x)=\exp(\alpha\cdot x+\beta).
\]

When taking the log, the \(y!\) term does not depend on \(\lambda,\alpha,\beta\), so it can be discarded.

Instructions:

1. **[3p]** Load `data/visits_clean.csv` into `problem3_df`. Decide what should be features and target, give motivations.

2. **[3p]** Create `problem3_X` and `problem3_y` as numpy arrays. Do the train-test split with 80% training and 20% testing.

3. **[2p]** Implement `loss` inside the class `PoissonRegression`.

4. **[2p]** Use the `PoissonRegression` class to train a Poisson regression model on the training data.

5. **[3p]** Come up with a reasonable metric to evaluate your model on the test data, compute it, justify it, interpret your result and compare it to a naive model.

### Problem 3 Part 1 — Free-text answer

The target should be `ofp`, the number of physician office visits.

Reasonable features are patient characteristics available before the visit count outcome, such as:

- health indicators: `exclhlth`, `poorhealth`, `numchron`, `adldiff`
- demographic/location variables: `noreast`, `midwest`, `west`, `age`, `male`, `married`, `school`, `faminc`, `employed`
- insurance variables: `privins`, `medicaid`

I would **not** use `ofnp`, `opp`, `opnp`, `emr`, or `hosp` as features if the goal is to predict physician office visits from patient characteristics, because these are also healthcare-use outcome variables. They may leak information about the target and make the model less meaningful.

Additional useful features could include prior-year visits, diagnoses, medication use, distance to clinic, and more detailed socioeconomic information.


```python
# Problem 3 imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
```


```python
# Part 1
problem3_df = pd.read_csv("data/visits_clean.csv")
problem3_df.head()
```


```python
# Part 1
problem3_target = "ofp"

# Avoid leakage from other utilization/count outcome variables.
exclude_cols = ["ofp", "ofnp", "opp", "opnp", "emr", "hosp"]
problem3_features = [col for col in problem3_df.columns if col not in exclude_cols]

problem3_features, problem3_target
```


```python
# Part 2
problem3_X = problem3_df[problem3_features].values
problem3_y = problem3_df[problem3_target].values

problem3_X_train, problem3_X_test, problem3_y_train, problem3_y_test = train_test_split(
    problem3_X,
    problem3_y,
    train_size=0.8,
    random_state=42
)

problem3_X_train.shape, problem3_X_test.shape, problem3_y_train.shape, problem3_y_test.shape
```


```python
# Part 3
class PoissonRegression(object):
    def __init__(self):
        self.coeffs = None
        self.result = None

    def fit(self, X, Y):
        import numpy as np
        from scipy import optimize

        # Define the objective/cost/loss function we want to minimise.
        def loss(coeffs):
            # lambda = exp(alpha dot x + beta)
            # coeffs[:-1] are alpha, coeffs[-1] is beta.
            eta = np.dot(X, coeffs[:-1]) + coeffs[-1]

            # Clip eta to avoid numerical overflow in exp during optimization.
            eta = np.clip(eta, -20, 20)
            lam = np.exp(eta)

            # Negative log-likelihood ignoring log(y!) constant:
            # nll = lambda - y*log(lambda)
            # Since log(lambda)=eta:
            # nll = lambda - y*eta
            return float(np.mean(lam - Y * eta))

        initial_arguments = np.zeros(shape=X.shape[1] + 1)
        self.result = optimize.minimize(loss, initial_arguments, method="cg")
        self.coeffs = self.result.x

    def predict(self, X):
        if self.coeffs is not None:
            eta = np.dot(X, self.coeffs[:-1]) + self.coeffs[-1]
            eta = np.clip(eta, -20, 20)
            return np.exp(eta)
        else:
            raise ValueError("Model has not been fitted yet.")
```


```python
# Part 4
problem3_model = PoissonRegression()
problem3_model.fit(problem3_X_train, problem3_y_train)

# This should ideally show success=True.
print(problem3_model.result)
```


```python
# Part 5
# Use MAE as metric because visits are counts and MAE is easy to interpret:
# average absolute error in number of visits.

problem3_pred = problem3_model.predict(problem3_X_test)

problem3_metric = float(mean_absolute_error(problem3_y_test, problem3_pred))
problem3_rmse = float(np.sqrt(mean_squared_error(problem3_y_test, problem3_pred)))

# Naive baseline: always predict training mean count.
naive_pred = np.full_like(problem3_y_test, fill_value=np.mean(problem3_y_train), dtype=float)
problem3_naive_mae = float(mean_absolute_error(problem3_y_test, naive_pred))

print("Poisson regression MAE:", problem3_metric)
print("Poisson regression RMSE:", problem3_rmse)
print("Naive mean predictor MAE:", problem3_naive_mae)

plt.figure(figsize=(7, 5))
plt.scatter(problem3_pred, problem3_y_test, alpha=0.5)
min_val = min(problem3_pred.min(), problem3_y_test.min())
max_val = max(problem3_pred.max(), problem3_y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Perfect prediction")
plt.xlabel("Predicted physician office visits")
plt.ylabel("True physician office visits")
plt.title("Poisson regression: predicted vs true")
plt.legend()
plt.show()
```

### Problem 3 Part 5 — Free-text answer

A reasonable metric is **Mean Absolute Error (MAE)** because it tells us the average absolute mistake in the number of physician office visits.

I compare the Poisson regression MAE with a naive baseline that always predicts the average number of visits from the training set.

- If the Poisson regression MAE is lower than the naive MAE, then the model improves over simply guessing the average.
- If it is similar or worse, then the chosen features/model are not adding much predictive value.

Because visit counts can be noisy and overdispersed, a simple Poisson model may not perfectly fit the data, but it is a reasonable first model for count-valued targets.


```python
# Final checks
assert isinstance(problem1_p1, float)
assert isinstance(problem1_p2, float)
assert isinstance(problem1_irreducible, bool)
assert isinstance(problem1_stationary, np.ndarray)
assert np.isclose(problem1_stationary.sum(), 1)
assert isinstance(problem1_ET, float)

assert isinstance(problem2_df, pd.DataFrame)
assert isinstance(problem2_features, list)
assert isinstance(problem2_target, str)
assert isinstance(problem2_mae, float)

assert isinstance(problem3_df, pd.DataFrame)
assert isinstance(problem3_features, list)
assert isinstance(problem3_target, str)
assert isinstance(problem3_metric, float)

print("All main variables have the expected formats.")
```
