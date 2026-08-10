# Solved Exam Notebook — 14th of June 2023

Course: **1MS041 Introduction to Data Science / Introduktion till dataanalys**

This notebook contains the **whole questions**, then the **solved code answers with all steps, comments, plots, outputs, and free-text explanations**.

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
| Downtown | 0.3 | 0.7 | 0 |
| Suburbs | 0.2 | 0.5 | 0.3 |
| Countryside | 0 | 0.5 | 0.5 |

1. **[2p]** If a truck is currently in the downtown, what is the probability that it will be in the countryside region after 10 time steps?

2. **[2p]** If a truck is currently in the downtown, what is the probability that it will be in the countryside region the first time after three time steps or more?

3. **[3p]** Is this Markov chain irreducible? Explain your answer.

4. **[3p]** What is the stationary distribution?

5. **[4p] Advanced question:** What is the expected number of steps it takes starting from the Downtown region to first reach the Countryside region and then returning to Downtown? Hint: to get within 1 decimal point, it is enough to compute the probabilities for hitting times below 120. Motivate your answer in detail. You could also solve this question by simulation, but this gives you a maximum of 2p.

### Problem 1 — Solution idea

Use state order:

\[
(\text{Downtown},\text{Suburbs},\text{Countryside})
\]

The transition matrix is

\[
P=
\begin{pmatrix}
0.3&0.7&0\\
0.2&0.5&0.3\\
0&0.5&0.5
\end{pmatrix}.
\]

Part 1 is entry \((P^{10})_{\text{Downtown},\text{Countryside}}\).

For Part 2, the earliest possible first hit of Countryside from Downtown is at time 2:

\[
\text{Downtown}\to\text{Suburbs}\to\text{Countryside}.
\]

So the probability of first reaching Countryside at time 3 or later is

\[
1-P(T_C=2).
\]

For Part 5, use the Markov property:

\[
E[\text{Downtown}\to\text{Countryside}\to\text{Downtown}]
=
E_{\text{Downtown}}[T_C]+E_{\text{Countryside}}[T_D].
\]


```python
# Problem 1 imports
import numpy as np

# State order: Downtown, Suburbs, Countryside
P = np.array([
    [0.3, 0.7, 0.0],
    [0.2, 0.5, 0.3],
    [0.0, 0.5, 0.5]
])

P
```


```python
# Part 1
# Probability of being in Countryside after 10 steps when starting from Downtown.
P10 = np.linalg.matrix_power(P, 10)
problem1_p1 = float(P10[0, 2])

problem1_p1
```


```python
# Part 2
# First hitting time of Countryside starting from Downtown.
# It is impossible to reach Countryside in 1 step.
# The only way to first reach Countryside in 2 steps is:
# Downtown -> Suburbs -> Countryside, with probability 0.7 * 0.3.
#
# Since the chain is finite and irreducible, Countryside will eventually be hit with probability 1.
# Therefore:
# P(first hit of Countryside at time >= 3) = 1 - P(first hit at time 2)

p_first_hit_c_at_2 = P[0, 1] * P[1, 2]
problem1_p2 = float(1 - p_first_hit_c_at_2)

problem1_p2
```

### Problem 1 Part 3 — Free-text answer

Yes, the Markov chain is **irreducible**.

Reason: every state can eventually reach every other state.

- From Downtown, we can go to Suburbs, and from Suburbs to Countryside.
- From Countryside, we can go to Suburbs, and from Suburbs to Downtown.
- Suburbs connects to both Downtown and Countryside.

Therefore all states communicate with each other, so the chain is irreducible.


```python
# Part 3
problem1_irreducible = True
problem1_irreducible
```


```python
# Part 4
# Stationary distribution pi solves pi P = pi and sum(pi)=1.
# We compute it using the eigenvector of P.T with eigenvalue 1.

eigenvalues, eigenvectors = np.linalg.eig(P.T)
idx = np.argmin(np.abs(eigenvalues - 1))
pi = np.real(eigenvectors[:, idx])
pi = pi / np.sum(pi)

problem1_stationary = pi
problem1_stationary
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 5
          1 # Part 4
          2 # Stationary distribution pi solves pi P = pi and sum(pi)=1.
          3 # We compute it using the eigenvector of P.T with eigenvalue 1.
    ----> 5 eigenvalues, eigenvectors = np.linalg.eig(P.T)
          6 idx = np.argmin(np.abs(eigenvalues - 1))
          7 pi = np.real(eigenvectors[:, idx])
    

    NameError: name 'np' is not defined


### Problem 1 Part 4 — Free-text answer

The stationary distribution \(\pi\) satisfies

\[
\pi P=\pi,\qquad \sum_i \pi_i=1.
\]

The code above solves this using the eigenvector of \(P^T\) corresponding to eigenvalue 1. The resulting vector is stored in `problem1_stationary`.

The entries represent the long-run proportions of time the truck spends in Downtown, Suburbs, and Countryside.


```python
# Part 5
# Expected time Downtown -> first Countryside -> then return to Downtown.
#
# First solve expected time to hit Countryside.
# h_C = 0
# h_D = 1 + 0.3 h_D + 0.7 h_S
# h_S = 1 + 0.2 h_D + 0.5 h_S + 0.3 h_C

A1 = np.array([
    [0.7, -0.7],
    [-0.2, 0.5]
])
b1 = np.array([1.0, 1.0])

h_D, h_S = np.linalg.solve(A1, b1)
h_C = 0.0

# Then solve expected time to hit Downtown starting from Countryside.
# k_D = 0
# k_S = 1 + 0.5 k_S + 0.3 k_C
# k_C = 1 + 0.5 k_S + 0.5 k_C

A2 = np.array([
    [0.5, -0.3],
    [-0.5, 0.5]
])
b2 = np.array([1.0, 1.0])

k_S, k_C = np.linalg.solve(A2, b2)
k_D = 0.0

problem1_ET = float(h_D + k_C)

h_D, k_C, problem1_ET
```

### Problem 1 Part 5 — Free-text answer

We split the task into two parts:

1. Expected time from Downtown to first reach Countryside.
2. Expected time from Countryside to return to Downtown.

By the Markov property, after the chain first reaches Countryside, the future only depends on the current state Countryside, not on the previous path. Therefore:

\[
E[\text{Downtown}\to\text{Countryside}\to\text{Downtown}]
=
E_D[T_C]+E_C[T_D].
\]

Solving the linear systems gives the final value stored in `problem1_ET`.

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

6. **[3p] Advanced question:** On the test set, plot the empirical distribution function of the residual with confidence bands, i.e. using the DKW inequality and 95% confidence. What does the confidence band tell us? What can the confidence band be used for?

### Problem 2 — Solution idea

Use linear regression to predict `salary_in_usd` from:

- `work_year`
- `experience_level`
- `employment_type`
- `remote_ratio`

A good metric is **Mean Absolute Error (MAE)** because it tells the average prediction error in salary dollars. We also compute RMSE and \(R^2\).

For DKW confidence bands around the empirical CDF:

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
problem2_df = pd.read_csv("salaries.csv")
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
      <th>work_year</th>
      <th>experience_level</th>
      <th>employment_type</th>
      <th>salary_in_usd</th>
      <th>remote_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023</td>
      <td>2</td>
      <td>1</td>
      <td>85847</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023</td>
      <td>1</td>
      <td>2</td>
      <td>30000</td>
      <td>100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023</td>
      <td>1</td>
      <td>2</td>
      <td>25500</td>
      <td>100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023</td>
      <td>2</td>
      <td>1</td>
      <td>175000</td>
      <td>100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023</td>
      <td>2</td>
      <td>1</td>
      <td>120000</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Part 1
problem2_features = ["work_year", "experience_level", "employment_type", "remote_ratio"]
problem2_target = "salary_in_usd"

# Robust handling if employment_type is text rather than encoded numbers.
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




    (['work_year', 'experience_level', 'employment_type', 'remote_ratio'],
     'salary_in_usd')




```python
# Part 2
problem2_X_train, problem2_X_test, problem2_y_train, problem2_y_test = train_test_split(
    problem2_X,
    problem2_y,
    train_size=0.8,
    random_state=42
)

problem2_X_train.shape, problem2_X_test.shape
```




    ((3004, 4), (751, 4))




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
           <td class="fitted-att-type">ndarray[float64](4,)</td>
           <td>[13291.19,38923.79,-3479.04,  -13.03]</td>


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
           <td class="fitted-att-type">ndarray[object](4,)</td>
           <td>[&#x27;work_year&#x27;,&#x27;experience_level&#x27;,&#x27;employment_type&#x27;,&#x27;remote_ratio&#x27;]</td>


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
           <td>-2.68e+07</td>


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
           <td>4</td>


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
           <td>4</td>


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
           <td class="fitted-att-type">ndarray[float64](4,)</td>
           <td>[2665.05,  40.46,  33.12,   7.2 ]</td>


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



### Problem 2 Part 4 — Free-text answer

I use **MAE** as the main metric because it is easy to interpret: it is the average absolute salary error in USD.

The predicted-vs-true scatterplot should ideally lie close to the diagonal \(y=x\). If the points are very spread out, then the model is not predicting salaries very accurately. This simple linear model may perform weakly because salary depends on many missing variables such as country, company size, exact job title, industry, education, and years of experience.


```python
#Part 4

#Compute metrics and plot performance.

# Part 4

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

problem2_y_pred = problem2_model.predict(problem2_X_test)

problem2_mae = mean_absolute_error(problem2_y_test, problem2_y_pred)
problem2_rmse = np.sqrt(mean_squared_error(problem2_y_test, problem2_y_pred))
problem2_r2 = r2_score(problem2_y_test, problem2_y_pred)

problem2_mae, problem2_rmse, problem2_r2

#Plot predicted vs true salary:

# Part 4 plot 1: predicted vs true

plt.figure(figsize=(6, 6))

plt.scatter(problem2_y_pred, problem2_y_test, alpha=0.6)

min_val = min(problem2_y_pred.min(), problem2_y_test.min())
max_val = max(problem2_y_pred.max(), problem2_y_test.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Predicted salary")
plt.ylabel("True salary")
plt.title("Predicted Salary vs True Salary")
plt.grid(True)
plt.show()

#Plot residuals:

# Part 4 plot 2: residuals

residuals = problem2_y_test - problem2_y_pred

plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=30)

plt.xlabel("Residuals: true salary - predicted salary")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.grid(True)
plt.show()
```


    
![png](solved_14_june_2023_FULL_QUESTIONS_files/solved_14_june_2023_FULL_QUESTIONS_21_0.png)
    



    
![png](solved_14_june_2023_FULL_QUESTIONS_files/solved_14_june_2023_FULL_QUESTIONS_21_1.png)
    



```python
# Part 4
problem2_y_pred = problem2_model.predict(problem2_X_test)

problem2_mae = float(mean_absolute_error(problem2_y_test, problem2_y_pred))
problem2_rmse = float(np.sqrt(mean_squared_error(problem2_y_test, problem2_y_pred)))
problem2_r2 = float(r2_score(problem2_y_test, problem2_y_pred))

print("MAE:", problem2_mae)
print("RMSE:", problem2_rmse)
print("R^2:", problem2_r2)

# Plot predicted vs true
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

# Residual histogram
problem2_residuals = problem2_y_test.values - problem2_y_pred

plt.figure(figsize=(7, 5))
plt.hist(problem2_residuals, bins=30, edgecolor="black", alpha=0.7)
plt.xlabel("Residual = true - predicted")
plt.ylabel("Count")
plt.title("Residual histogram")
plt.show()
```

    MAE: 44488.11860767363
    RMSE: 56223.89785116799
    R^2: 0.19926672357275976
    


    
![png](solved_14_june_2023_FULL_QUESTIONS_files/solved_14_june_2023_FULL_QUESTIONS_22_1.png)
    



    
![png](solved_14_june_2023_FULL_QUESTIONS_files/solved_14_june_2023_FULL_QUESTIONS_22_2.png)
    


### Problem 2 Part 5 — Free-text answer

To predict the 2023 salary for a full-time, mid-level, non-remote data scientist, use:

\[
\text{work_year}=2023,\quad
\text{experience_level}=1,\quad
\text{employment_type}=1,\quad
\text{remote_ratio}=0.
\]

The sign of the `remote_ratio` coefficient tells us the direction:

- positive coefficient: higher remote ratio increases predicted salary;
- negative coefficient: higher remote ratio decreases predicted salary.


```python
# Part 5
new_person = pd.DataFrame([{
    "work_year": 2023,
    "experience_level": 1,
    "employment_type": 1,
    "remote_ratio": 0
}])

problem2_predicted_salary = float(problem2_model.predict(new_person)[0])

coef_table = pd.DataFrame({
    "feature": problem2_features,
    "coefficient": problem2_model.coef_
})

remote_coef = float(coef_table.loc[coef_table["feature"] == "remote_ratio", "coefficient"].iloc[0])

print("Predicted salary:", problem2_predicted_salary)
print(coef_table)

if remote_coef > 0:
    print("Higher remote ratio gives higher predicted salary, according to this fitted model.")
elif remote_coef < 0:
    print("Higher remote ratio gives lower predicted salary, according to this fitted model.")
else:
    print("Remote ratio has no effect in this fitted model.")
```

    Predicted salary: 121940.38437091932
                feature   coefficient
    0         work_year  13291.191050
    1  experience_level  38923.792313
    2   employment_type  -3479.038925
    3      remote_ratio    -13.029475
    Higher remote ratio gives lower predicted salary, according to this fitted model.
    

### Problem 2 Part 6 — Free-text answer

The empirical distribution function (EDF) of the residuals shows the distribution of prediction errors.

The DKW confidence band gives a uniform 95% confidence band around the true CDF of residuals. It means that, with probability at least 95%, the true residual CDF lies inside the band for every residual value.

This can be used to estimate residual quantiles and understand uncertainty in prediction errors.


```python
# Part 6
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


    
![png](solved_14_june_2023_FULL_QUESTIONS_files/solved_14_june_2023_FULL_QUESTIONS_26_0.png)
    





    np.float64(0.04955782815694993)



## Problem 3 — Full Question

Maximum Points = 13

For this problem we have the Diabetes dataset. The categorical features are already encoded using One-Hot encoding, namely:

`['smoking_No Info', 'smoking_current', 'smoking_ever', 'smoking_former', 'smoking_never', 'smoking_not current', 'sex_Female', 'sex_Male', 'sex_Other']`.

Treating this as a classification problem, we will train a logistic regression model to predict whether the patient has diabetes or not. Then the task is to evaluate the model and use it to make some conclusions.

Instructions:

1. **[3p]** Load the file `data/diabetes.csv` into the pandas dataframe `problem3_df`. Decide what should be features and target, give motivations for your choices.

2. **[2p]** Create `problem3_X` and `problem3_y` as numpy arrays, with `problem3_X` being the features and `problem3_y` being the target. Do the standard train-test split with 80% training data and 20% testing data. Store these in the variables defined in the cells.

3. **[2p]** Now train a Logistic regression model on the training data using `sklearn.linear_model.LogisticRegression`. Hint: If you use many of the One-Hot encoded features you will probably see a warning about max iterations reached; adjust the hyperparameter `C`, the penalization, when you create your LogisticRegression.

4. **[3p]** Evaluation: Calculate the precision and recall for class 0 and 1 with 95% confidence bounds. Explain their meaning.

5. **[3p] Advanced question:** Come up with a way to define the one-hot encoded feature that is most important for the prediction. Motivate your choice.

### Problem 3 Part 1 — Free-text answer

Reasonable features are all patient measurements and encoded categorical variables **except the target column** `diabetes`.

The target should be `diabetes`, because the task is to predict whether the patient has diabetes.

We should use enough features to capture useful medical information, but not so many irrelevant variables that we overfit. Since the dataset is usually large, using the listed clinical variables and one-hot encoded smoking/sex variables is reasonable.

Useful additional features that were not necessarily collected could include family history, diet, physical activity, medication use, pregnancy history, and repeated blood glucose measurements.


```python
# Problem 3 imports
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix
```


```python
# Part 1
problem3_df = pd.read_csv("data/diabetes.csv")
problem3_df.head()
```


```python
# Part 1
# The target is diabetes.
problem3_target = "diabetes"

# Use all columns except the target as features.
problem3_features = [col for col in problem3_df.columns if col != problem3_target]

problem3_features[:10], problem3_target, len(problem3_features)
```


```python
# Part 2
problem3_X = problem3_df[problem3_features].values
problem3_y = problem3_df[problem3_target].values

problem3_X_train, problem3_X_test, problem3_y_train, problem3_y_test = train_test_split(
    problem3_X,
    problem3_y,
    train_size=0.8,
    random_state=42,
    stratify=problem3_y
)

problem3_X_train.shape, problem3_X_test.shape, problem3_y_train.shape, problem3_y_test.shape
```


```python
# Part 3
# Use larger max_iter and a moderate C to avoid convergence warning.
problem3_model = LogisticRegression(max_iter=2000, C=0.5)
problem3_model.fit(problem3_X_train, problem3_y_train)

problem3_model
```

### Problem 3 Part 4 — Free-text answer

Precision for class 1 means: among all patients predicted as diabetes, what fraction actually had diabetes?

Recall for class 1 means: among all patients who actually had diabetes, what fraction did the model find?

Precision for class 0 means: among all patients predicted as non-diabetes, what fraction were actually non-diabetes?

Recall for class 0 means: among all actually non-diabetes patients, what fraction did the model correctly classify as non-diabetes?

The confidence bounds below use Hoeffding's inequality for a bounded average.


```python
# Part 4
problem3_y_pred = problem3_model.predict(problem3_X_test)

def hoeffding_interval_for_rate(rate, n, delta=0.05):
    if n == 0:
        return (0.0, 1.0)
    eps = np.sqrt(np.log(2 / delta) / (2 * n))
    return (float(max(0, rate - eps)), float(min(1, rate + eps)))

# Class 0 precision: true 0 among predicted 0
pred0 = (problem3_y_pred == 0)
true0 = (problem3_y_test == 0)

p0_denom = int(np.sum(pred0))
r0_denom = int(np.sum(true0))

precision0 = float(np.sum(pred0 & true0) / p0_denom) if p0_denom > 0 else 0.0
recall0 = float(np.sum(pred0 & true0) / r0_denom) if r0_denom > 0 else 0.0

# Class 1 precision and recall
pred1 = (problem3_y_pred == 1)
true1 = (problem3_y_test == 1)

p1_denom = int(np.sum(pred1))
r1_denom = int(np.sum(true1))

precision1 = float(np.sum(pred1 & true1) / p1_denom) if p1_denom > 0 else 0.0
recall1 = float(np.sum(pred1 & true1) / r1_denom) if r1_denom > 0 else 0.0

problem3_precision_0 = hoeffding_interval_for_rate(precision0, p0_denom, delta=0.05)
problem3_recall_0 = hoeffding_interval_for_rate(recall0, r0_denom, delta=0.05)
problem3_precision_1 = hoeffding_interval_for_rate(precision1, p1_denom, delta=0.05)
problem3_recall_1 = hoeffding_interval_for_rate(recall1, r1_denom, delta=0.05)

print("Confusion matrix:")
print(confusion_matrix(problem3_y_test, problem3_y_pred))

print("Precision 0 interval:", problem3_precision_0)
print("Recall 0 interval:", problem3_recall_0)
print("Precision 1 interval:", problem3_precision_1)
print("Recall 1 interval:", problem3_recall_1)
```

### Problem 3 Part 5 — Free-text answer

To define the most important one-hot encoded feature, I compare the absolute value of the logistic regression coefficients among only the one-hot encoded columns.

For one-hot encoded features, all variables are binary indicators, so their coefficients are more directly comparable than coefficients of variables measured on different numerical scales.

The feature with the largest absolute coefficient changes the log-odds the most when it changes from 0 to 1, so I define it as the most important one-hot encoded feature.


```python
# Part 5
one_hot_features = [
    'smoking_No Info', 'smoking_current', 'smoking_ever', 'smoking_former',
    'smoking_never', 'smoking_not current',
    'sex_Female', 'sex_Male', 'sex_Other'
]

# Keep only one-hot features that actually exist in the dataframe
one_hot_features = [f for f in one_hot_features if f in problem3_features]

coef = problem3_model.coef_[0]
coef_table = pd.DataFrame({
    "feature": problem3_features,
    "coefficient": coef,
    "absolute_coefficient": np.abs(coef)
})

one_hot_coef_table = coef_table[coef_table["feature"].isin(one_hot_features)].sort_values(
    "absolute_coefficient",
    ascending=False
)

problem3_most_important_one_hot = one_hot_coef_table.iloc[0]["feature"] if len(one_hot_coef_table) > 0 else None

print(one_hot_coef_table)
print("Most important one-hot feature:", problem3_most_important_one_hot)
```


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
assert isinstance(problem2_predicted_salary, float)

assert isinstance(problem3_df, pd.DataFrame)
assert isinstance(problem3_features, list)
assert isinstance(problem3_target, str)
assert isinstance(problem3_precision_0, tuple)
assert isinstance(problem3_recall_0, tuple)
assert isinstance(problem3_precision_1, tuple)
assert isinstance(problem3_recall_1, tuple)

print("All main variables have the expected formats.")
```
