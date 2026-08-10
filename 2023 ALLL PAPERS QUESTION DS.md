```python
## Jan 2023 question 1 markov chain

# 0 = Downtown
# 1 = Suburbs
# 2 = Countryside

import numpy as np

P = np.array([
    [0.3, 0.4, 0.3],
    [0.2, 0.5, 0.3],
    [0.4, 0.3, 0.3]
])

P2 = np.linalg.matrix_power(P, 2)

problem1_p1 = P2[1, 0]
problem1_p1
```




    np.float64(0.28)




```python
problem1_p2 = P[1, 1] * P[1, 0] + P[1, 2] * P[2, 0]
problem1_p2

##0.5 * 0.2 + 0.3 * 0.4 = 0.10 + 0.12 = 0.22


P = np.array([
    [0.3, 0.4, 0.3],
    [0.2, 0.5, 0.3],
    [0.4, 0.3, 0.3]
])

def first_hit_exactly(P, start, target, n):
    dist = np.zeros(P.shape[0])
    dist[start] = 1.0
    
    for step in range(1, n + 1):
        hit_prob = dist @ P[:, target]
        
        if step == n:
            return hit_prob
        
        dist = dist @ P
        dist[target] = 0.0

problem1_p2 = first_hit_exactly(P, start=1, target=0, n=2)
problem1_p2
```




    np.float64(0.22)




```python
# Part 3

# Fill in the answer to part 3 below as a boolean
problem1_irreducible = True
```


```python
# Part 4
# Stationary distribution pi solves pi P = pi and sum(pi)=1.
# Use the eigenvector of P.T with eigenvalue 1.
A = P.T - np.eye(3)
A[-1] = np.ones(3)

b = np.array([0, 0, 1])

problem1_stationary = np.linalg.solve(A, b)
problem1_stationary
```




    array([0.28888889, 0.41111111, 0.3       ])




```python
# Part 5
# Non-target states are Downtown and Countryside
non_target = [0, 2]

Q = P[np.ix_(non_target, non_target)]

I = np.eye(len(Q))
ones = np.ones(len(Q))

expected_steps = np.linalg.solve(I - Q, ones)

problem1_ET = expected_steps[0]
problem1_ET
```




    np.float64(2.702702702702703)




```python
## June 2023 question 1 markov chain

# 0 = Downtown
# 1 = Suburbs
# 2 = Countryside

import numpy as np

P = np.array([
    [0.3, 0.7, 0.0],
    [0.2, 0.5, 0.3],
    [0.0, 0.5, 0.5]
])
# Part 1
P10 = np.linalg.matrix_power(P, 10)

problem1_p1 = P10[0, 2]
problem1_p1
```




    np.float64(0.3181084179)




```python
problem1_p2 = 1 - (P[0, 1] * P[1, 2])
problem1_p2
```




    np.float64(0.79)




```python
##part 2

P = np.array([
    [0.3, 0.7, 0.0],
    [0.2, 0.5, 0.3],
    [0.0, 0.5, 0.5]
])

def first_hit_after_n_or_more(P, start, target, n):
    dist = np.zeros(P.shape[0])
    dist[start] = 1.0
    
    for step in range(1, n):
        dist = dist @ P
        dist[target] = 0.0
    
    return dist.sum()

problem1_p2 = first_hit_after_n_or_more(P, start=0, target=2, n=3)
problem1_p2
```




    np.float64(0.7899999999999999)




```python
#A Markov chain is irreducible if every state can reach every other state eventually.

##Even though:

##Downtown → Countryside = 0
##Countryside → Downtown = 0
## they can still reach each other through Suburbs:

##Downtown → Suburbs → Countryside
## 0.7 × 0.3 > 0
## and
##Countryside → Suburbs → Downtown
##0.5 × 0.2 > 0
##So every region can reach every other region.

# Part 3

problem1_irreducible = True
```


```python
## PART 4
# Part 4

A = P.T - np.eye(3)
A[-1] = np.ones(3)

b = np.array([0, 0, 1])

problem1_stationary = np.linalg.solve(A, b)
problem1_stationary

problem1_stationary = np.array([0.15151515, 0.53030303, 0.31818182])
```




    array([0.15151515, 0.53030303, 0.31818182])




```python
# Part 5

# We need:
# Downtown -> first reach Countryside
# then Countryside -> return to Downtown

# ----------------------------
# 1) Expected steps from Downtown to Countryside
# ----------------------------

non_target_C = [0, 1]   # Downtown and Suburbs, because Countryside is the target

Q_C = P[np.ix_(non_target_C, non_target_C)]

h_C = np.linalg.solve(
    np.eye(2) - Q_C,
    np.ones(2)
)

ET_D_to_C = h_C[0]


# ----------------------------
# 2) Expected steps from Countryside back to Downtown
# ----------------------------

non_target_D = [1, 2]   # Suburbs and Countryside, because Downtown is the target

Q_D = P[np.ix_(non_target_D, non_target_D)]

h_D = np.linalg.solve(
    np.eye(2) - Q_D,
    np.ones(2)
)

ET_C_to_D = h_D[1]


# ----------------------------
# Total expected time
# ----------------------------

problem1_ET = ET_D_to_C + ET_C_to_D

problem1_ET
```




    np.float64(15.714285714285715)




```python
# New question matrix

P = np.array([
    [0.3, 0.7, 0.0],
    [0.2, 0.5, 0.3],
    [0.0, 0.5, 0.5]
])

# First: Downtown to Countryside

non_target_C = [0, 1]   # Downtown and Suburbs
Q_C = P[np.ix_(non_target_C, non_target_C)]

h_C = np.linalg.solve(np.eye(2) - Q_C, np.ones(2))
ET_D_to_C = h_C[0]


# Second: Countryside to Downtown

non_target_D = [1, 2]   # Suburbs and Countryside
Q_D = P[np.ix_(non_target_D, non_target_D)]

h_D = np.linalg.solve(np.eye(2) - Q_D, np.ones(2))
ET_C_to_D = h_D[1]


# Total

problem1_ET = ET_D_to_C + ET_C_to_D
problem1_ET

## Output:15.714285714285717
#Main difference

#Old Part 5:

#Downtown → Suburbs

#New Part 5:

#Downtown → Countryside → Downtown

#So the new one is longer because it has two hitting times added together.
```


```python
## August 2023 question 1 Markov chain

# Part 1

import numpy as np

P = np.array([
    [0.3, 0.7, 0.0, 0.0],
    [0.2, 0.5, 0.3, 0.0],
    [0.0, 0.0, 0.5, 0.5],
    [0.0, 0.0, 0.0, 1.0]
])

P10 = np.linalg.matrix_power(P, 10)

problem1_p1 = P10[0, 2]

problem1_p1


```




    np.float64(0.08487353489999999)




```python
# Part 2

problem1_p2 = (
    P[0, 0] * P[0, 0] +   # Downtown -> Downtown -> Downtown
    P[0, 0] * P[0, 1] +   # Downtown -> Downtown -> Suburbs
    P[0, 1] * P[1, 0] +   # Downtown -> Suburbs -> Downtown
    P[0, 1] * P[1, 1]     # Downtown -> Suburbs -> Suburbs
)

problem1_p2
```




    np.float64(0.7899999999999999)




```python
# Part 3

problem1_irreducible = False
#Explanation:
#The chain is not irreducible because the Workshop is absorbing:
#Workshop → Workshop = 1

#Once the truck enters Workshop, it cannot go back to Downtown, Suburbs, or Countryside.

#So every state cannot reach every other state.
```


```python
# Part 4 for 4-state Workshop question

P = np.array([
    [0.3, 0.7, 0.0, 0.0],
    [0.2, 0.5, 0.3, 0.0],
    [0.0, 0.0, 0.5, 0.5],
    [0.0, 0.0, 0.0, 1.0]
])

A = P.T - np.eye(4)
A[-1] = np.ones(4)

b = np.array([0, 0, 0, 1])

problem1_stationary = np.linalg.solve(A, b)

problem1_stationary
```




    array([0., 0., 0., 1.])




```python
# Part 5

P = np.array([
    [0.3, 0.7, 0.0, 0.0],
    [0.2, 0.5, 0.3, 0.0],
    [0.0, 0.0, 0.5, 0.5],
    [0.0, 0.0, 0.0, 1.0]
])

# Target is Workshop = 3
# Non-target states are Downtown, Suburbs, Countryside
non_target = [0, 1, 2]

Q = P[np.ix_(non_target, non_target)]

h = np.linalg.solve(
    np.eye(3) - Q,
    np.ones(3)
)

# h[0] is expected time starting from Downtown
problem1_ET = h[0]

problem1_ET
```




    np.float64(7.714285714285714)




```python
## JANUARY 2023 QUESTION 2
# Part 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

problem2_df = pd.read_csv("data/abalone.csv")

problem2_df.head()

# Features = all columns except Rings
problem2_features = [col for col in problem2_df.columns if col != "Rings"]

# Target = Rings
problem2_target = "Rings"

problem2_features, problem2_target
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Cell In[35], line 8
          5 import numpy as np
          6 import matplotlib.pyplot as plt
    ----> 8 problem2_df = pd.read_csv("data/abalone.csv")
         10 problem2_df.head()
         12 # Features = all columns except Rings
    

    File /lib/python3.13/site-packages/pandas/io/parsers/readers.py:1026, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
       1013 kwds_defaults = _refine_defaults_read(
       1014     dialect,
       1015     delimiter,
       (...)   1022     dtype_backend=dtype_backend,
       1023 )
       1024 kwds.update(kwds_defaults)
    -> 1026 return _read(filepath_or_buffer, kwds)
    

    File /lib/python3.13/site-packages/pandas/io/parsers/readers.py:620, in _read(filepath_or_buffer, kwds)
        617 _validate_names(kwds.get("names", None))
        619 # Create the parser.
    --> 620 parser = TextFileReader(filepath_or_buffer, **kwds)
        622 if chunksize or iterator:
        623     return parser
    

    File /lib/python3.13/site-packages/pandas/io/parsers/readers.py:1620, in TextFileReader.__init__(self, f, engine, **kwds)
       1617     self.options["has_index_names"] = kwds["has_index_names"]
       1619 self.handles: IOHandles | None = None
    -> 1620 self._engine = self._make_engine(f, self.engine)
    

    File /lib/python3.13/site-packages/pandas/io/parsers/readers.py:1880, in TextFileReader._make_engine(self, f, engine)
       1878     if "b" not in mode:
       1879         mode += "b"
    -> 1880 self.handles = get_handle(
       1881     f,
       1882     mode,
       1883     encoding=self.options.get("encoding", None),
       1884     compression=self.options.get("compression", None),
       1885     memory_map=self.options.get("memory_map", False),
       1886     is_text=is_text,
       1887     errors=self.options.get("encoding_errors", "strict"),
       1888     storage_options=self.options.get("storage_options", None),
       1889 )
       1890 assert self.handles is not None
       1891 f = self.handles.handle
    

    File /lib/python3.13/site-packages/pandas/io/common.py:873, in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        868 elif isinstance(handle, str):
        869     # Check whether the filename is to be opened in binary mode.
        870     # Binary mode does not support 'encoding' and 'newline'.
        871     if ioargs.encoding and "b" not in ioargs.mode:
        872         # Encoding
    --> 873         handle = open(
        874             handle,
        875             ioargs.mode,
        876             encoding=ioargs.encoding,
        877             errors=errors,
        878             newline="",
        879         )
        880     else:
        881         # Binary mode
        882         handle = open(handle, ioargs.mode)
    

    FileNotFoundError: [Errno 44] No such file or directory: 'data/abalone.csv'



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
# Part 3

from sklearn.linear_model import LinearRegression

problem2_model = LinearRegression()

problem2_model.fit(problem2_X_train, problem2_y_train)
```


```python
# Part 4

from sklearn.metrics import mean_absolute_error

problem2_y_pred = problem2_model.predict(problem2_X_test)

problem2_mae = mean_absolute_error(problem2_y_test, problem2_y_pred)

problem2_mae
```


```python
# Part 4 plot

residuals = problem2_y_test - problem2_y_pred

residuals_sorted = np.sort(residuals)
n = len(residuals_sorted)

edf = np.arange(1, n + 1) / n

alpha = 0.05
epsilon = np.sqrt(np.log(2 / alpha) / (2 * n))

lower_band = np.maximum(edf - epsilon, 0)
upper_band = np.minimum(edf + epsilon, 1)

plt.figure(figsize=(8, 5))
plt.step(residuals_sorted, edf, where="post", label="EDF of residuals")
plt.step(residuals_sorted, lower_band, where="post", linestyle="--", label="Lower 95% band")
plt.step(residuals_sorted, upper_band, where="post", linestyle="--", label="Upper 95% band")

plt.xlabel("Residuals: true Rings - predicted Rings")
plt.ylabel("Empirical distribution")
plt.title("EDF of residuals with 95% DKW confidence bands")
plt.legend()
plt.grid(True)
plt.show()
```


```python
# Part 5

plt.figure(figsize=(6, 6))

plt.scatter(problem2_y_pred, problem2_y_test, alpha=0.6)

min_val = min(problem2_y_pred.min(), problem2_y_test.min())
max_val = max(problem2_y_pred.max(), problem2_y_test.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Predicted Rings")
plt.ylabel("True Rings")
plt.title("Predicted Rings vs True Rings")
plt.grid(True)
plt.show()
```


```python
#PART 6
The mean absolute error tells us how many rings the model is wrong by on average. 
If the MAE is around 1.5 to 2 rings, the model is reasonable but not perfect. 
The scatter plot shows that many predictions are near the diagonal line, but there is still noticeable spread. 
This means the linear regression model captures some relationship between the physical measurements and age, but it does not predict the age perfectly, especially for very young or very old abalones.
```


```python
## JUNE 2023 QUESTION 2
# Part 1

# Part 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

problem2_df = pd.read_csv("data/salaries.csv")

problem2_df.head()

# Part 1

problem2_features = ["work_year", "experience_level", "employment_type", "remote_ratio"]

problem2_target = "salary_in_usd"
```


```python
# Part 2

from sklearn.model_selection import train_test_split

X = problem2_df[problem2_features]
y = problem2_df[problem2_target]

problem2_X_train, problem2_X_test, problem2_y_train, problem2_y_test = train_test_split(
    X,
    y,
    train_size=0.8,
    random_state=42
)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[37], line 5
          1 # Part 2
          3 from sklearn.model_selection import train_test_split
    ----> 5 X = problem2_df[problem2_features]
          6 y = problem2_df[problem2_target]
          8 problem2_X_train, problem2_X_test, problem2_y_train, problem2_y_test = train_test_split(
          9     X,
         10     y,
         11     train_size=0.8,
         12     random_state=42
         13 )
    

    NameError: name 'problem2_df' is not defined



```python
# Part 3

from sklearn.linear_model import LinearRegression

problem2_model = LinearRegression()

problem2_model.fit(problem2_X_train, problem2_y_train)
```


```python
Part 4

Compute metrics and plot performance.

# Part 4

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

problem2_y_pred = problem2_model.predict(problem2_X_test)

problem2_mae = mean_absolute_error(problem2_y_test, problem2_y_pred)
problem2_rmse = np.sqrt(mean_squared_error(problem2_y_test, problem2_y_pred))
problem2_r2 = r2_score(problem2_y_test, problem2_y_pred)

problem2_mae, problem2_rmse, problem2_r2

Plot predicted vs true salary:

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

Plot residuals:

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


```python
#Part 5

#Predict salary for:

#work_year = 2023
#experience_level = 1
#employment_type = 1
#remote_ratio = 0




# Part 5

new_data_scientist = pd.DataFrame(
    [[2023, 1, 1, 0]],
    columns=problem2_features
)

problem2_predicted_salary = problem2_model.predict(new_data_scientist)[0]

problem2_predicted_salary

#Now check the coefficient of remote_ratio:

coef_table = pd.DataFrame({
    "feature": problem2_features,
    "coefficient": problem2_model.coef_
})

coef_table

#Get remote ratio coefficient:

remote_coef = coef_table.loc[
    coef_table["feature"] == "remote_ratio",
    "coefficient"
].iloc[0]

remote_coef

#Interpretation:

if remote_coef > 0:
    print("Higher remote_ratio increases predicted salary.")
elif remote_coef < 0:
    print("Higher remote_ratio decreases predicted salary.")
else:
    print("remote_ratio has no effect on predicted salary.")
```


```python
# Part 6

import numpy as np
import matplotlib.pyplot as plt

# Predict on the test set
problem2_y_pred = problem2_model.predict(problem2_X_test)

# Residual = true value - predicted value
problem2_residuals = np.asarray(problem2_y_test) - problem2_y_pred

# Sort residuals for EDF
residuals_sorted = np.sort(problem2_residuals)

n = len(residuals_sorted)

# Empirical distribution function values
edf = np.arange(1, n + 1) / n

# DKW confidence band, 95% confidence
alpha = 0.05
epsilon = np.sqrt(np.log(2 / alpha) / (2 * n))

lower_band = np.maximum(edf - epsilon, 0)
upper_band = np.minimum(edf + epsilon, 1)

# Plot EDF with confidence bands
plt.figure(figsize=(8, 5))

plt.step(residuals_sorted, edf, where="post", label="EDF of residuals")
plt.step(residuals_sorted, lower_band, where="post", linestyle="--", label="Lower 95% band")
plt.step(residuals_sorted, upper_band, where="post", linestyle="--", label="Upper 95% band")

plt.xlabel("Residuals: true value - predicted value")
plt.ylabel("Empirical distribution function")
plt.title("EDF of Residuals with 95% DKW Confidence Bands")
plt.legend()
plt.grid(True)
plt.show()

problem2_DKW_epsilon = epsilon
problem2_DKW_epsilon
```


```python
#part 6
The confidence band gives a range around the empirical distribution function where the true distribution function is likely to lie with 95% confidence. It shows the uncertainty in the estimated residual distribution. A narrow band means the EDF is estimated more accurately, while a wide band means more uncertainty. The confidence band can be used to understand how reliable the residual distribution estimate is and to compare residual distributions between models.
```


```python
##PROBLEM 3 JANUARY 2023 QUESTION 3
#PART 1


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Part 1
problem3_df = pd.read_csv("data/visits_clean.csv")
problem3_df.columns = problem3_df.columns.str.strip()

problem3_features = [
    "exclhlth",
    "poorhlth",
    "numchron",
    "adldiff",
    "noreast",
    "midwest",
    "west",
    "age",
    "male",
    "married",
    "school",
    "faminc",
    "employed",
    "privins",
    "medicaid"
]

problem3_target = "ofp"


# Part 2
from sklearn.model_selection import train_test_split

problem3_X = problem3_df[problem3_features].values
problem3_y = problem3_df[problem3_target].values

problem3_X_train, problem3_X_test, problem3_y_train, problem3_y_test = train_test_split(
    problem3_X,
    problem3_y,
    train_size=0.8,
    random_state=42
)


# Part 3
class PoissonRegression(object):
    
    def __init__(self):
        self.coeffs = None
        self.result = None
        
    def fit(self, X, Y):
        import numpy as np
        from scipy import optimize
        
        def loss(coeffs):
            lam = np.exp(np.dot(X, coeffs[:-1]) + coeffs[-1])
            return np.mean(lam - Y * np.log(lam))
        
        initial_arguments = np.zeros(shape=X.shape[1] + 1)
        
        self.result = optimize.minimize(loss, initial_arguments, method="cg")
        self.coeffs = self.result.x
        
    def predict(self, X):
        if self.coeffs is not None:
            return np.exp(np.dot(X, self.coeffs[:-1]) + self.coeffs[-1])


# Part 4
problem3_model = PoissonRegression()
problem3_model.fit(problem3_X_train, problem3_y_train)

print(problem3_model.result.success)


# Part 5
from sklearn.metrics import mean_absolute_error

problem3_y_pred = problem3_model.predict(problem3_X_test)

problem3_metric = mean_absolute_error(problem3_y_test, problem3_y_pred)

naive_prediction = np.full_like(
    problem3_y_test,
    fill_value=np.mean(problem3_y_train),
    dtype=float
)

problem3_naive_metric = mean_absolute_error(problem3_y_test, naive_prediction)

print("Poisson regression MAE:", problem3_metric)
print("Naive baseline MAE:", problem3_naive_metric)
```


```python
#14th of June 2023 QUESTION 3

# Part 1

import pandas as pd
import numpy as np

problem3_df = pd.read_csv("data/diabetes.csv")

# Clean column names just in case
problem3_df.columns = problem3_df.columns.str.strip()

# Show columns
print(problem3_df.columns.tolist())

problem3_target = "diabetes"

# Use every column except diabetes as features
problem3_features = [col for col in problem3_df.columns if col != problem3_target]

problem3_features, problem3_target
#The target is diabetes because we want to predict whether the patient has diabetes or not. 
#Reasonable features are age, BMI, blood glucose level, HbA1c level, hypertension, heart disease, smoking status, and sex, because these are medically related to diabetes risk. 
#Since the dataset has many observations, using all available medically meaningful features is reasonable. 
#Other useful features would be family history of diabetes, diet, exercise level, cholesterol, medication use, pregnancy history, and ethnicity.
```


```python
# Part 2

from sklearn.model_selection import train_test_split

problem3_X = problem3_df[problem3_features].values
problem3_y = problem3_df[problem3_target].values

problem3_X_train, problem3_X_test, problem3_y_train, problem3_y_test = train_test_split(
    problem3_X,
    problem3_y,
    train_size=0.8,
    random_state=42
)
```


```python
# Part 3

from sklearn.linear_model import LogisticRegression

problem3_model = LogisticRegression(max_iter=1000)

problem3_model.fit(problem3_X_train, problem3_y_train)
```


```python
Part 4

We need:

Precision and recall for class 0 and class 1
with 95% confidence bounds
Baby explanation
Precision

Precision asks:

When the model predicts a class, how often is it correct?

For class 1:

Of all people predicted as diabetic, how many were truly diabetic?

For class 0:

Of all people predicted as non-diabetic, how many were truly non-diabetic?
Recall

Recall asks:

Of all real examples of a class, how many did the model find?

For class 1:

Of all truly diabetic patients, how many did the model detect?

For class 0:

Of all truly non-diabetic patients, how many did the model detect?
```


```python
# Part 4

problem3_y_pred = problem3_model.predict(problem3_X_test)

def precision_recall_ci(y_true, y_pred, class_label):
    import numpy as np
    
    # True positives for this class
    tp = np.sum((y_true == class_label) & (y_pred == class_label))
    
    # Predicted positives for this class
    predicted_as_class = np.sum(y_pred == class_label)
    
    # Actually this class
    actual_class = np.sum(y_true == class_label)
    
    # Precision
    precision = tp / predicted_as_class
    
    # Recall
    recall = tp / actual_class
    
    # 95% confidence interval using normal approximation
    z = 1.96
    
    precision_margin = z * np.sqrt((precision * (1 - precision)) / predicted_as_class)
    recall_margin = z * np.sqrt((recall * (1 - recall)) / actual_class)
    
    precision_ci = (
        max(0, precision - precision_margin),
        min(1, precision + precision_margin)
    )
    
    recall_ci = (
        max(0, recall - recall_margin),
        min(1, recall + recall_margin)
    )
    
    return precision_ci, recall_ci


problem3_precision_0, problem3_recall_0 = precision_recall_ci(
    problem3_y_test,
    problem3_y_pred,
    0
)

problem3_precision_1, problem3_recall_1 = precision_recall_ci(
    problem3_y_test,
    problem3_y_pred,
    1
)

problem3_precision_0, problem3_recall_0, problem3_precision_1, problem3_recall_1

#PART 4 THEORY - Precision tells us how reliable the model's positive predictions are for a class. 
Recall tells us how many true examples of a class the model successfully finds. 
For class 1, precision tells us how often predicted diabetic patients are actually diabetic, and recall tells us how many truly diabetic patients are detected. 
For class 0, precision and recall measure the same ideas for non-diabetic patients. 
The confidence bounds show uncertainty due to using a finite test set. Narrower bounds mean the estimate is more reliable.
```


```python
# Part 5

# Put feature names and coefficients in a table
coef_table = pd.DataFrame({
    "feature": problem3_features,
    "coefficient": problem3_model.coef_[0]
})

# One-hot encoded columns
one_hot_features = [
    "smoking_No Info",
    "smoking_current",
    "smoking_ever",
    "smoking_former",
    "smoking_never",
    "smoking_not current",
    "sex_Female",
    "sex_Male",
    "sex_Other"
]

# Keep only one-hot features that actually exist in the dataframe
one_hot_features = [col for col in one_hot_features if col in problem3_features]

one_hot_coef_table = coef_table[
    coef_table["feature"].isin(one_hot_features)
].copy()

# Importance = absolute value of coefficient
one_hot_coef_table["absolute_coefficient"] = one_hot_coef_table["coefficient"].abs()

# Sort by importance
one_hot_coef_table = one_hot_coef_table.sort_values(
    by="absolute_coefficient",
    ascending=False
)

one_hot_coef_table

problem3_most_important_onehot_feature = one_hot_coef_table.iloc[0]["feature"]
problem3_most_important_onehot_coefficient = one_hot_coef_table.iloc[0]["coefficient"]

problem3_most_important_onehot_feature, problem3_most_important_onehot_coefficient
```


```python
#Written answer for Part 5
I defined the most important one-hot encoded feature as the one with the largest absolute logistic regression coefficient. 
This is reasonable because logistic regression coefficients measure how strongly each feature changes the log-odds of diabetes. 
A large positive coefficient means the feature increases the predicted probability of diabetes, while a large negative coefficient means the feature decreases the predicted probability. 
Therefore, the one-hot feature with the largest absolute coefficient has the strongest effect o
```


```python
# 15th of August 2023.pdf QUUESTION 3

# Part 1

import numpy as np

np.random.seed(42)

U = np.random.uniform(0, 1, 1000)

problem3_samples = np.log(U + 1)

problem3_samples

```




    array([0.31811922, 0.66819562, 0.54927331, 0.46916483, 0.1449819 ,
           0.14496103, 0.05645936, 0.6238915 , 0.47070027, 0.53536559,
           0.0203755 , 0.67798778, 0.60564985, 0.19255164, 0.16705983,
           0.16839546, 0.26562222, 0.42183468, 0.35903367, 0.25559459,
           0.47738438, 0.13058418, 0.25630336, 0.31215162, 0.37574101,
           0.57951699, 0.18204967, 0.41490999, 0.46525146, 0.04540388,
           0.47470808, 0.15745162, 0.06302324, 0.66725769, 0.67581384,
           0.59244101, 0.26590703, 0.09319168, 0.52131028, 0.36474901,
           0.11514688, 0.40224453, 0.03381045, 0.64674737, 0.23014298,
           0.5083359 , 0.27133245, 0.41875508, 0.43613028, 0.16961994,
           0.67782267, 0.57387525, 0.66242966, 0.63912773, 0.46869025,
           0.65330087, 0.08479371, 0.17896833, 0.04423436, 0.28166173,
           0.3283517 , 0.24007857, 0.60362584, 0.30509459, 0.2475899 ,
           0.43353159, 0.13183866, 0.58900647, 0.07190257, 0.68656906,
           0.57224697, 0.18125072, 0.00550693, 0.59633967, 0.53465387,
           0.54754735, 0.571697  , 0.07143157, 0.30635592, 0.10963353,
           0.62224361, 0.48445996, 0.28585392, 0.06162022, 0.27077672,
           0.28155081, 0.54789374, 0.49320578, 0.635101  , 0.38676802,
           0.11296634, 0.53838911, 0.56575976, 0.4455042 , 0.57152583,
           0.40132026, 0.42050663, 0.3559534 , 0.02510143, 0.10245859,
           0.0309454 , 0.49250507, 0.2733468 , 0.41116264, 0.64582833,
           0.22257717, 0.34386124, 0.56278285, 0.20603659, 0.07416074,
           0.25444953, 0.14947229, 0.65736333, 0.59228784, 0.49066603,
           0.62671919, 0.58982463, 0.17106684, 0.63792988, 0.43135521,
           0.59191157, 0.63979456, 0.27611807, 0.10440679, 0.20533403,
           0.35564987, 0.59774512, 0.6209692 , 0.00692808, 0.41260443,
           0.34883197, 0.20057708, 0.11320847, 0.29088831, 0.6641867 ,
           0.28005526, 0.41791437, 0.53240253, 0.31014997, 0.67893774,
           0.67419231, 0.22456837, 0.40362909, 0.26303966, 0.25063458,
           0.0362229 , 0.47596354, 0.40724953, 0.05019751, 0.24580207,
           0.64619492, 0.214758  , 0.13531282, 0.39840878, 0.68594655,
           0.21676748, 0.51410158, 0.56623362, 0.21320436, 0.54708986,
           0.31319128, 0.48999364, 0.49074314, 0.42903493, 0.0864435 ,
           0.60720932, 0.27822252, 0.1710234 , 0.03996576, 0.46429546,
           0.51734296, 0.01645175, 0.41349482, 0.20416114, 0.49784542,
           0.16072879, 0.52528325, 0.32695231, 0.66100098, 0.12885128,
           0.29346508, 0.10748443, 0.6547668 , 0.62985554, 0.22947676,
           0.50680799, 0.59730907, 0.44160468, 0.42503933, 0.21660405,
           0.08902023, 0.64038742, 0.64207389, 0.49048094, 0.29194532,
           0.29951892, 0.54578091, 0.64033181, 0.63503407, 0.57654344,
           0.49593428, 0.08078701, 0.14982308, 0.64109264, 0.47401374,
           0.00915502, 0.09664705, 0.50892488, 0.00504882, 0.14911636,
           0.43743769, 0.52584932, 0.50196322, 0.20234418, 0.53776696,
           0.21289044, 0.28171407, 0.55760886, 0.50055278, 0.61476577,
           0.50537855, 0.44999772, 0.08954337, 0.31314205, 0.23523208,
           0.21832367, 0.67956058, 0.33152985, 0.63765908, 0.48927831,
           0.58489989, 0.40722163, 0.45546336, 0.40046442, 0.1783495 ,
           0.54374892, 0.2474633 , 0.02402504, 0.49802745, 0.16306286,
           0.66292433, 0.669842  , 0.64964681, 0.31492657, 0.01533838,
           0.65664841, 0.35640381, 0.67633404, 0.6747897 , 0.61681105,
           0.25808504, 0.3257707 , 0.61579987, 0.2752972 , 0.1565701 ,
           0.44263324, 0.66070393, 0.52829011, 0.45111458, 0.09274006,
           0.47933943, 0.6881617 , 0.13110196, 0.41761082, 0.6298735 ,
           0.55432675, 0.52887126, 0.53208841, 0.30711048, 0.25742272,
           0.59297383, 0.59338949, 0.6243716 , 0.64879843, 0.41299826,
           0.40647546, 0.58683909, 0.50075343, 0.53178457, 0.58544652,
           0.63657966, 0.29117234, 0.31887761, 0.0898242 , 0.45633574,
           0.03531142, 0.38226336, 0.43349824, 0.25195742, 0.46425794,
           0.03004436, 0.03666764, 0.60026436, 0.30762487, 0.11961293,
           0.42018508, 0.5709759 , 0.19541959, 0.4842088 , 0.08190018,
           0.05039052, 0.42615272, 0.43219475, 0.49312788, 0.54585951,
           0.68099974, 0.41627339, 0.27986898, 0.58510875, 0.239672  ,
           0.36392857, 0.07553074, 0.02503474, 0.67429479, 0.60757846,
           0.52825733, 0.34284684, 0.15981545, 0.14534376, 0.22333785,
           0.43775588, 0.53917744, 0.5069365 , 0.24680843, 0.67032128,
           0.55267571, 0.44106006, 0.47730239, 0.35037519, 0.22132669,
           0.30451904, 0.56408926, 0.01429089, 0.10981595, 0.04497589,
           0.03992124, 0.61813296, 0.53277762, 0.38809772, 0.09333929,
           0.39986001, 0.38762137, 0.15973665, 0.36036428, 0.33540362,
           0.47986119, 0.49170008, 0.04430776, 0.31817196, 0.48603685,
           0.40755376, 0.61868752, 0.50603032, 0.15094649, 0.06819005,
           0.49617032, 0.02616598, 0.46107361, 0.66280665, 0.45455629,
           0.32798628, 0.49669925, 0.37723907, 0.43542305, 0.66344274,
           0.32649595, 0.67355172, 0.64466606, 0.178808  , 0.06706156,
           0.0960172 , 0.0180578 , 0.09024552, 0.52058194, 0.06876892,
           0.2768554 , 0.61241169, 0.02300527, 0.59579258, 0.24830807,
           0.11168879, 0.52870709, 0.48793124, 0.6299262 , 0.55104836,
           0.58971865, 0.24844833, 0.1633422 , 0.55996701, 0.59157655,
           0.68838845, 0.34544449, 0.31628271, 0.57459614, 0.29326909,
           0.65791232, 0.61972276, 0.35697072, 0.56011342, 0.56220835,
           0.09814604, 0.64319662, 0.40896057, 0.60237828, 0.27766931,
           0.63949491, 0.32872925, 0.01077935, 0.6446825 , 0.08735744,
           0.27711163, 0.66786115, 0.66814068, 0.45326296, 0.4897065 ,
           0.37049093, 0.2571281 , 0.28417434, 0.51433055, 0.56097174,
           0.58309738, 0.58200227, 0.0872836 , 0.40173838, 0.0559632 ,
           0.43795094, 0.3657054 , 0.63536137, 0.30078215, 0.11070652,
           0.13364911, 0.56617175, 0.48132558, 0.09633027, 0.08075643,
           0.53119817, 0.07023757, 0.59985799, 0.53429342, 0.07820913,
           0.0814304 , 0.68644456, 0.31792326, 0.31527935, 0.59487237,
           0.66641739, 0.6861231 , 0.56154432, 0.31936937, 0.0801972 ,
           0.57500922, 0.44366238, 0.35362571, 0.64519272, 0.10543825,
           0.40053638, 0.01128968, 0.38435086, 0.05477534, 0.1122727 ,
           0.11111753, 0.50029657, 0.55735316, 0.45955471, 0.6740523 ,
           0.3183596 , 0.25131272, 0.62518902, 0.20179393, 0.67458728,
           0.0120812 , 0.67797203, 0.04225448, 0.63718147, 0.42376406,
           0.68962338, 0.07120056, 0.44073848, 0.67767944, 0.42074632,
           0.48821101, 0.52812435, 0.37469043, 0.48708078, 0.4601517 ,
           0.64246318, 0.04444395, 0.24761229, 0.66804037, 0.63671639,
           0.37545718, 0.482508  , 0.24481203, 0.1723732 , 0.38096639,
           0.30258465, 0.45973617, 0.07486128, 0.68026192, 0.68622868,
           0.52954632, 0.42924437, 0.26966647, 0.59542135, 0.52160601,
           0.15067345, 0.64758856, 0.60022962, 0.66772676, 0.54564407,
           0.47835317, 0.34941881, 0.65893273, 0.62383134, 0.04422612,
           0.02602536, 0.31951743, 0.59363251, 0.68676492, 0.14012439,
           0.46632858, 0.32272884, 0.67799009, 0.6109165 , 0.60885685,
           0.384373  , 0.34700196, 0.24169604, 0.05484371, 0.62311218,
           0.59492833, 0.69300601, 0.69146418, 0.44175313, 0.5704073 ,
           0.66514152, 0.61499502, 0.22101978, 0.37193875, 0.12147348,
           0.66990467, 0.47385535, 0.20591015, 0.51384148, 0.48127007,
           0.30613284, 0.10755993, 0.51376522, 0.41891275, 0.57228852,
           0.4188179 , 0.61636413, 0.43948439, 0.4452869 , 0.62949019,
           0.33895691, 0.12576463, 0.02837624, 0.56254706, 0.48261721,
           0.53302524, 0.19306708, 0.12784027, 0.01443991, 0.30053973,
           0.46368225, 0.33091687, 0.36288805, 0.64404028, 0.29881151,
           0.41474821, 0.57866352, 0.33399974, 0.48371341, 0.62184649,
           0.66758351, 0.1372139 , 0.65575037, 0.40019544, 0.22971741,
           0.37784431, 0.6831133 , 0.40053169, 0.28423986, 0.49066425,
           0.21522881, 0.07312344, 0.12122574, 0.12048679, 0.14141509,
           0.12999894, 0.49522948, 0.16710646, 0.29689001, 0.64016215,
           0.38795377, 0.51136012, 0.15898458, 0.175875  , 0.04005557,
           0.15609313, 0.24575817, 0.16297774, 0.08498665, 0.11389627,
           0.3789697 , 0.18758578, 0.31061939, 0.4077407 , 0.52496213,
           0.03855909, 0.58745906, 0.48729108, 0.07858845, 0.6278503 ,
           0.65277946, 0.05928533, 0.24441776, 0.5912259 , 0.55862083,
           0.16933849, 0.19008247, 0.31515528, 0.3950935 , 0.48134827,
           0.31401746, 0.38017104, 0.55816956, 0.03602639, 0.22509121,
           0.53845028, 0.63932798, 0.41321992, 0.42664815, 0.10180903,
           0.36977739, 0.42697691, 0.21710174, 0.23842084, 0.32011356,
           0.01987243, 0.27920562, 0.19181634, 0.28329548, 0.11311628,
           0.63685577, 0.46599087, 0.51825932, 0.58175252, 0.40442603,
           0.08334827, 0.42990178, 0.46174532, 0.55700637, 0.35883429,
           0.12007401, 0.24980566, 0.30974853, 0.49829782, 0.45157123,
           0.30461052, 0.68638197, 0.47360639, 0.21287242, 0.0969293 ,
           0.14224506, 0.21990449, 0.14900722, 0.17106428, 0.25083278,
           0.15988301, 0.64015003, 0.07717745, 0.42167396, 0.3438711 ,
           0.68429745, 0.10619518, 0.33493935, 0.67776469, 0.62353293,
           0.59722645, 0.22944591, 0.15776208, 0.51201085, 0.65719663,
           0.4426086 , 0.45210228, 0.24684374, 0.57069303, 0.17146597,
           0.28041516, 0.35447804, 0.41052587, 0.21705283, 0.10870805,
           0.47661922, 0.25358007, 0.45820822, 0.14354843, 0.39281213,
           0.42695874, 0.05052536, 0.29013228, 0.12611681, 0.06144778,
           0.68811465, 0.27941336, 0.59325748, 0.2268492 , 0.51968787,
           0.56544327, 0.46727412, 0.38633406, 0.34489447, 0.29926592,
           0.65727601, 0.60465438, 0.67550594, 0.11715815, 0.54862271,
           0.66183217, 0.16655886, 0.06437876, 0.55452896, 0.45392068,
           0.61075898, 0.13082857, 0.58515393, 0.18367674, 0.15156672,
           0.15209067, 0.59585113, 0.50994357, 0.42072503, 0.30662439,
           0.62978159, 0.33106127, 0.59696631, 0.36404218, 0.31986686,
           0.38027022, 0.26342361, 0.55824879, 0.40727706, 0.20881149,
           0.64162995, 0.32489926, 0.43408681, 0.64525447, 0.48503878,
           0.11055524, 0.66260144, 0.48717292, 0.28886059, 0.13038953,
           0.5844618 , 0.48247106, 0.42752733, 0.63863428, 0.58143163,
           0.1412173 , 0.27134083, 0.22193413, 0.55615053, 0.03298248,
           0.45100535, 0.56670981, 0.62954989, 0.29422195, 0.59952709,
           0.10492899, 0.61326612, 0.11999274, 0.33453271, 0.58628296,
           0.13969014, 0.20640536, 0.54363307, 0.54234553, 0.49539577,
           0.52706216, 0.43354997, 0.22458176, 0.29691135, 0.16686752,
           0.64629169, 0.45956925, 0.33708021, 0.37980933, 0.66643524,
           0.14267197, 0.46136003, 0.40938321, 0.47713702, 0.01794815,
           0.62707357, 0.65861696, 0.44797092, 0.5286562 , 0.6536261 ,
           0.53487723, 0.14196737, 0.45507294, 0.47419175, 0.35356157,
           0.55183948, 0.65978015, 0.65522126, 0.37214227, 0.10727293,
           0.68553891, 0.60916653, 0.11748315, 0.65276357, 0.62588301,
           0.41794561, 0.46453586, 0.33575963, 0.05331481, 0.28907903,
           0.58937066, 0.00462133, 0.28780644, 0.3351633 , 0.43008982,
           0.65224998, 0.29739425, 0.29784515, 0.55244802, 0.373092  ,
           0.2026182 , 0.37324457, 0.13177975, 0.16244787, 0.40437636,
           0.34989986, 0.64963715, 0.30924337, 0.45779715, 0.48996818,
           0.01300947, 0.50894628, 0.16384862, 0.6734904 , 0.13859842,
           0.34686386, 0.08190221, 0.69158308, 0.40692738, 0.4671151 ,
           0.06492264, 0.5595932 , 0.19054233, 0.6408293 , 0.18659544,
           0.17453106, 0.03589757, 0.3866675 , 0.44778431, 0.06363997,
           0.57409763, 0.37382915, 0.42159451, 0.36517266, 0.33701713,
           0.44445524, 0.14430833, 0.16714711, 0.62153604, 0.66583532,
           0.31722339, 0.23960309, 0.49713202, 0.34269155, 0.02506947,
           0.14509777, 0.53997982, 0.50616916, 0.0267354 , 0.20046608,
           0.20788761, 0.51395636, 0.0195188 , 0.0990383 , 0.58774004,
           0.16428034, 0.50243821, 0.21364481, 0.09480223, 0.21766633,
           0.54364141, 0.61826008, 0.6044361 , 0.33445845, 0.51167634,
           0.18646653, 0.25707935, 0.63992351, 0.01291812, 0.08204857,
           0.18887194, 0.02618633, 0.16673017, 0.45934804, 0.35165957,
           0.63798943, 0.59743088, 0.29402493, 0.23065402, 0.32186058,
           0.4639195 , 0.23749104, 0.48498393, 0.34317235, 0.43957482,
           0.36194958, 0.25809807, 0.66703588, 0.56736046, 0.13112753,
           0.62511883, 0.3970506 , 0.63898252, 0.58770625, 0.35432163,
           0.02222059, 0.23797491, 0.43284303, 0.49071162, 0.22943387,
           0.13046326, 0.60700646, 0.6853177 , 0.42244689, 0.15843801,
           0.24083204, 0.01822361, 0.6493514 , 0.1113187 , 0.45521765,
           0.2422049 , 0.44094679, 0.50163576, 0.60417487, 0.18765835,
           0.01093581, 0.12829262, 0.6418637 , 0.62801653, 0.46838551,
           0.47032661, 0.50984715, 0.16158408, 0.6494105 , 0.34979067,
           0.32435521, 0.41799805, 0.04589643, 0.15382208, 0.55275437,
           0.07954905, 0.47197176, 0.2194159 , 0.32879687, 0.2536291 ,
           0.3042978 , 0.54176943, 0.26014774, 0.44878296, 0.38936987,
           0.50902671, 0.66105248, 0.54960707, 0.194695  , 0.03070682,
           0.23290697, 0.46692259, 0.05014716, 0.40303967, 0.46802846,
           0.28836476, 0.57149478, 0.10129067, 0.07244882, 0.5470739 ,
           0.40245479, 0.52378275, 0.36104452, 0.22026103, 0.59834315,
           0.5874621 , 0.52750365, 0.24070456, 0.46387908, 0.30820054,
           0.08762809, 0.65092503, 0.12823369, 0.66795109, 0.36880512])




```python
# Part 2

problem3_mean = np.mean(problem3_samples)

problem3_variance = np.var(problem3_samples, ddof=1)

problem3_mean, problem3_variance

#Your answers should be close to:

#Mean ≈ 0.386
#Variance ≈ 0.039

#Because the theoretical values are:

#E[X] = 2ln(2) - 1 ≈ 0.386
#Var(X) = 1 - 2(ln(2))² ≈ 0.039
```




    (np.float64(0.37917193716622055), np.float64(0.040264423657637695))




```python
# Part 3

problem3_samples_accept_reject = []

total_attempts = 0

while len(problem3_samples_accept_reject) < 1000:
    
    # sample from proposal Uniform(0, ln(2))
    x_candidate = np.random.uniform(0, np.log(2))
    
    # sample uniform value for accepting/rejecting
    u = np.random.uniform(0, 1)
    
    # accept with probability e^x / 2
    acceptance_probability = np.exp(x_candidate) / 2
    
    if u <= acceptance_probability:
        problem3_samples_accept_reject.append(x_candidate)
    
    total_attempts += 1

problem3_samples_accept_reject = np.array(problem3_samples_accept_reject)

problem3_acceptance_rate = len(problem3_samples_accept_reject) / total_attempts

problem3_samples_accept_reject, problem3_acceptance_rate
```




    (array([1.28324368e-01, 6.05079945e-01, 5.59065586e-01, 1.73056676e-01,
            6.54372070e-01, 6.34568132e-01, 4.83414892e-01, 6.54515655e-01,
            5.97522433e-01, 1.59429977e-01, 4.53015634e-02, 6.14112978e-01,
            1.61475203e-01, 6.03106581e-01, 6.06296998e-01, 6.50912128e-01,
            6.91715215e-01, 5.31774399e-01, 3.32624433e-01, 5.32526551e-01,
            1.65507705e-01, 2.45805349e-01, 2.05385138e-01, 2.91767757e-02,
            6.84636990e-01, 5.45052830e-01, 6.54838338e-01, 4.25184375e-01,
            6.87025739e-01, 6.53451871e-01, 4.21251046e-01, 1.59888130e-01,
            1.52829394e-01, 5.40366780e-01, 6.89617552e-01, 4.82444729e-01,
            5.65492430e-01, 1.55138549e-01, 4.10994644e-01, 1.84099921e-01,
            5.97581327e-01, 4.54179131e-01, 6.02946274e-02, 2.58327995e-01,
            5.01436612e-01, 5.61769561e-02, 4.73598886e-01, 5.90011673e-01,
            3.33117231e-01, 5.71625286e-01, 4.69964985e-01, 5.52733599e-01,
            5.89578327e-01, 4.83450377e-01, 4.28788733e-01, 6.04315883e-01,
            5.72412552e-01, 5.76791459e-01, 4.42634898e-03, 4.37939765e-01,
            4.39459266e-01, 5.40547637e-01, 5.27504345e-01, 6.67495192e-01,
            4.76806049e-01, 2.39747866e-01, 6.74771134e-01, 5.19619054e-01,
            5.25587996e-01, 1.53348774e-02, 4.73624264e-01, 2.95406368e-01,
            4.80823879e-01, 4.53672603e-01, 6.59088397e-01, 2.90932806e-01,
            2.75591436e-01, 6.82041332e-01, 6.19742342e-01, 1.47712925e-01,
            4.51701023e-01, 5.99127484e-01, 6.71100545e-01, 6.02083700e-01,
            5.34362303e-01, 5.27501634e-01, 9.09720169e-02, 6.38283089e-01,
            5.52117577e-01, 8.13118404e-02, 4.75197646e-01, 1.38993149e-01,
            2.15126517e-01, 8.05474447e-03, 2.72055802e-01, 4.15902551e-01,
            5.40551514e-01, 3.33062036e-01, 9.87703838e-02, 4.28472904e-01,
            3.87918907e-01, 2.26285735e-01, 6.09044161e-02, 2.30146412e-02,
            2.75126250e-01, 3.93389339e-01, 5.54924413e-01, 1.16090080e-01,
            4.41139833e-01, 4.94967344e-01, 3.13523144e-01, 5.65815919e-02,
            1.71278910e-01, 6.04274321e-01, 6.76418251e-01, 4.56581437e-01,
            3.84948680e-01, 6.75675132e-01, 5.00441236e-02, 1.78612168e-01,
            6.01840639e-01, 5.14804931e-01, 2.39783865e-01, 6.84586510e-01,
            6.00980437e-01, 6.24318958e-01, 1.91882407e-01, 6.32402081e-01,
            4.31807531e-01, 5.08155225e-01, 1.24546840e-01, 6.73319771e-01,
            5.92214619e-01, 3.08662082e-01, 2.49001597e-01, 1.13346111e-01,
            6.71945418e-01, 4.55215167e-01, 5.36130716e-01, 6.72228723e-01,
            1.63617713e-01, 1.17667213e-01, 2.98668472e-01, 4.27772307e-01,
            1.15784627e-01, 4.60114237e-01, 5.85134367e-01, 1.42678254e-01,
            1.86976183e-01, 2.72708643e-02, 2.19142713e-01, 2.88161100e-02,
            6.83879888e-01, 3.15201649e-01, 3.38842629e-01, 9.67988735e-02,
            6.72291800e-01, 4.66212262e-01, 6.01750356e-01, 6.54756683e-01,
            3.44615286e-01, 6.02279040e-01, 4.77943523e-01, 2.73006002e-01,
            6.92740806e-01, 6.77325531e-01, 6.03560299e-01, 3.93025723e-01,
            6.08940584e-01, 5.59956137e-01, 5.53002281e-01, 5.66879481e-01,
            3.77411083e-01, 2.24985753e-01, 2.69327952e-01, 1.64653045e-01,
            1.57531301e-01, 4.18278691e-01, 4.29397987e-01, 3.59800076e-01,
            1.77737256e-02, 2.63531521e-01, 4.02144842e-01, 5.63518773e-01,
            6.62318556e-01, 1.35702959e-01, 2.43726274e-01, 3.36780640e-01,
            1.97458849e-01, 5.56615137e-01, 2.15782054e-01, 4.96397821e-01,
            2.86650394e-01, 1.25763479e-01, 4.91473417e-01, 3.93230866e-01,
            6.67450049e-01, 5.58671463e-01, 6.59462544e-01, 1.58092851e-01,
            4.23499750e-01, 2.44973033e-01, 5.41019060e-01, 5.70192796e-01,
            4.62710331e-01, 4.32426332e-01, 4.06606002e-01, 1.50232581e-01,
            4.20805519e-01, 1.60773476e-01, 4.12059596e-01, 6.84680748e-01,
            4.81837488e-01, 8.89961397e-02, 5.02073409e-01, 1.90033693e-01,
            1.32992295e-01, 1.57106246e-01, 4.81209409e-02, 1.61996871e-01,
            6.10024338e-01, 3.69418187e-01, 6.89084895e-01, 3.86626003e-01,
            3.22455959e-01, 1.39276620e-01, 6.70110085e-02, 5.24132479e-01,
            6.42670691e-01, 2.76784701e-01, 6.87937141e-01, 3.61100553e-01,
            5.42060564e-01, 2.50644658e-02, 1.82375736e-01, 3.83866462e-01,
            2.75166619e-01, 4.16300269e-01, 6.37273954e-01, 6.87711531e-01,
            6.01691047e-01, 5.47607369e-01, 5.60889729e-02, 6.39606545e-01,
            1.21188902e-01, 2.76332154e-01, 2.54755449e-01, 1.78914505e-02,
            6.67580524e-01, 6.69456908e-01, 2.16134474e-01, 3.04646288e-01,
            4.44186953e-01, 4.29465644e-01, 1.05375598e-01, 5.41182692e-01,
            6.81834274e-01, 9.85997488e-02, 2.10214313e-01, 4.79769682e-01,
            5.64201197e-01, 3.65317305e-01, 3.11766961e-01, 2.54911357e-01,
            5.73605559e-01, 5.33241518e-01, 2.88455971e-01, 1.33030726e-02,
            5.26992747e-01, 3.71048708e-01, 8.40148077e-03, 6.76424142e-01,
            6.65127846e-01, 6.79856643e-02, 1.04005809e-01, 5.11096979e-01,
            2.60545905e-01, 5.45439864e-01, 4.76978631e-01, 3.19871803e-01,
            4.41305795e-01, 4.61996740e-02, 3.55045180e-01, 2.88854536e-02,
            4.96517537e-01, 4.93913999e-02, 6.62996247e-01, 2.44855218e-01,
            4.58427174e-01, 1.20683395e-01, 4.57686968e-01, 1.83716185e-01,
            4.32983792e-01, 1.42571546e-01, 4.46320418e-01, 5.53626490e-01,
            6.79263671e-01, 4.03704623e-01, 5.62675940e-01, 8.87892054e-02,
            6.43298559e-01, 2.57966523e-01, 3.04572330e-01, 6.53690358e-01,
            8.42183364e-02, 6.14769494e-01, 5.97056184e-01, 6.36951336e-01,
            5.23355184e-01, 5.83628923e-01, 5.38192358e-01, 1.22984251e-01,
            6.82521054e-01, 2.99257532e-02, 5.66845596e-01, 5.07939325e-01,
            4.09197837e-01, 2.06244870e-01, 4.41043558e-01, 1.10953202e-01,
            6.46818597e-03, 6.87450201e-02, 5.54566936e-01, 3.84755568e-01,
            4.26968578e-01, 3.84160546e-01, 5.27478600e-01, 5.16903269e-01,
            6.64085932e-01, 2.26611728e-01, 6.90606337e-01, 6.14264107e-01,
            2.71384934e-01, 4.82165757e-01, 4.29466602e-01, 5.50495641e-01,
            4.07710743e-01, 4.45226127e-01, 4.02014127e-01, 3.88619964e-01,
            3.45364521e-01, 4.05844115e-02, 5.44049127e-01, 5.46626239e-01,
            3.05122694e-01, 2.27485882e-01, 6.14131380e-02, 4.14658176e-01,
            6.92001763e-01, 4.45392256e-01, 8.20243971e-02, 5.82106587e-01,
            3.96391652e-01, 1.27869195e-01, 2.31865557e-01, 1.89278262e-01,
            1.79730817e-01, 6.04930513e-01, 3.17590949e-01, 9.25212665e-02,
            5.04569078e-01, 5.30613108e-01, 4.22975842e-01, 5.20813522e-01,
            6.63074727e-01, 3.95473191e-02, 1.81400557e-01, 6.28167807e-01,
            4.53083678e-02, 2.32991722e-02, 6.28294565e-01, 3.69045895e-01,
            6.67774136e-01, 6.49355848e-01, 2.90285408e-01, 1.37310816e-01,
            4.21798963e-01, 7.90851716e-02, 1.56177795e-01, 5.89602122e-01,
            3.62786935e-01, 5.96203403e-01, 3.76751983e-01, 3.17669798e-01,
            5.89158032e-01, 3.20438865e-01, 4.51220423e-01, 4.31644278e-02,
            5.58956394e-01, 1.39574738e-01, 1.14165745e-01, 5.24540173e-01,
            6.09679683e-01, 1.65520921e-01, 6.82742803e-01, 1.88310214e-02,
            3.73399632e-01, 5.69838488e-01, 1.41240147e-01, 4.00357213e-01,
            5.53071950e-01, 6.34292101e-01, 5.49846869e-01, 9.75337989e-02,
            4.53521038e-01, 3.05208936e-01, 5.57015525e-01, 6.70361087e-01,
            6.82133293e-01, 4.39920604e-01, 6.11305739e-01, 1.12451628e-01,
            3.87993074e-01, 5.67731565e-01, 4.92302793e-01, 3.18338377e-01,
            2.32348513e-01, 6.97565845e-02, 9.84074844e-02, 4.51341113e-01,
            4.39553835e-01, 4.68690571e-01, 4.75725761e-01, 4.81148513e-01,
            2.11625235e-01, 2.30051075e-02, 6.03872931e-01, 5.46510053e-01,
            4.64916211e-01, 7.69320377e-02, 3.78907673e-01, 6.77066394e-01,
            2.92888252e-01, 5.22606568e-01, 5.03689949e-01, 5.74892574e-01,
            6.04946863e-01, 4.87757951e-01, 6.69125958e-02, 8.59782277e-03,
            2.08821166e-01, 2.06029548e-01, 5.15142270e-01, 6.25839120e-01,
            4.62886933e-01, 6.18496961e-01, 5.47372521e-02, 5.50759694e-01,
            5.93123335e-01, 4.15180264e-01, 6.09261747e-01, 3.58666480e-01,
            2.19690634e-01, 5.36412501e-01, 6.18650002e-01, 4.20802430e-01,
            4.10133220e-01, 1.64576339e-01, 7.22431282e-02, 3.38025054e-01,
            6.58857919e-01, 5.15420026e-01, 4.39527422e-01, 1.76330807e-01,
            3.27507437e-01, 9.70573084e-02, 6.73415146e-01, 3.34125400e-01,
            4.23360484e-01, 1.43476322e-01, 3.81632916e-03, 1.51846927e-01,
            7.48777469e-02, 5.56310001e-01, 3.55354189e-01, 6.45842464e-01,
            7.89067615e-02, 3.42100421e-01, 6.88168193e-01, 1.90429382e-01,
            2.92389013e-01, 6.29108007e-01, 4.21367785e-01, 5.69211274e-01,
            4.41119815e-01, 1.78559718e-01, 4.18519116e-01, 7.93567835e-02,
            3.16239050e-01, 8.38881379e-02, 5.20676946e-01, 5.55765507e-02,
            6.52754522e-01, 4.70418167e-01, 4.11494304e-01, 5.01819882e-01,
            6.25532966e-01, 3.70118048e-01, 8.11440422e-03, 2.04506635e-01,
            4.00084474e-01, 4.79999450e-01, 4.66464917e-01, 5.70059750e-01,
            6.35363129e-01, 2.21021636e-01, 2.57277119e-01, 4.89074851e-01,
            2.59620278e-01, 2.98420197e-01, 6.16968250e-01, 1.07063338e-01,
            4.47649966e-01, 2.36088818e-01, 2.83933114e-01, 4.69344262e-01,
            2.52717115e-01, 6.84854344e-01, 4.77075346e-01, 3.36585131e-01,
            2.66311800e-01, 2.14569192e-02, 1.10920834e-01, 5.24167334e-01,
            6.82065950e-01, 6.66208352e-01, 2.67216490e-01, 3.75275157e-01,
            3.83803555e-01, 5.28110093e-01, 3.05290680e-01, 1.79757253e-01,
            3.99582591e-02, 3.86574300e-01, 4.02349481e-01, 3.07787673e-01,
            3.38878154e-01, 6.88883209e-01, 4.32112082e-01, 4.30209062e-01,
            2.73888613e-01, 3.29623395e-01, 6.30385899e-01, 6.54360116e-01,
            4.08077595e-01, 1.98090066e-01, 5.28038269e-01, 3.54389145e-01,
            6.79796711e-01, 5.73592408e-01, 1.38958345e-01, 5.91921330e-01,
            5.87074073e-01, 3.10751089e-01, 1.32502670e-02, 4.76692595e-01,
            4.80026902e-01, 2.07525208e-01, 5.57593382e-01, 2.04214604e-01,
            5.13258943e-01, 5.22236440e-01, 4.56465502e-01, 5.86348061e-01,
            6.14161077e-02, 5.38224174e-02, 2.82184715e-01, 2.41783970e-01,
            3.62222496e-01, 5.32767308e-02, 5.50716210e-01, 3.48140182e-02,
            2.79250480e-01, 1.61076538e-01, 2.22843839e-01, 6.78578000e-01,
            3.48711222e-01, 4.06090526e-01, 5.70137674e-01, 3.82964488e-01,
            1.88126647e-01, 1.97043786e-01, 4.36376143e-01, 5.18921333e-01,
            3.24752683e-02, 4.90038759e-01, 5.81785110e-01, 2.19221763e-01,
            2.49695125e-01, 5.84123379e-01, 3.42285810e-01, 4.38562455e-01,
            3.02438806e-03, 4.94245860e-01, 6.69613104e-01, 6.59025422e-01,
            2.06594048e-01, 5.41919007e-01, 3.33999986e-02, 6.63203094e-01,
            2.99944981e-01, 1.44786471e-01, 2.56335016e-01, 5.32037338e-01,
            5.69891670e-01, 4.21261670e-01, 5.11956746e-01, 4.46963901e-01,
            4.94357340e-01, 6.17050019e-01, 2.54929820e-01, 7.72943756e-02,
            1.01846809e-02, 2.33830115e-01, 8.62056051e-02, 3.41627697e-01,
            3.68017955e-01, 6.87866864e-03, 6.67846782e-01, 4.69636512e-01,
            5.36489942e-01, 8.70247774e-02, 5.33832993e-01, 8.33213019e-02,
            1.21663712e-02, 5.35901110e-01, 2.41281479e-01, 9.38261848e-04,
            5.87137824e-01, 6.22834750e-01, 5.40915510e-01, 2.75876247e-01,
            4.55362063e-02, 1.70932614e-01, 5.18088983e-01, 6.53446704e-01,
            6.87136140e-01, 6.67340168e-01, 2.53470580e-01, 1.35741150e-01,
            4.00845905e-01, 5.62228154e-01, 1.80152404e-01, 6.35952725e-01,
            4.90555059e-01, 1.02102690e-01, 4.37321660e-01, 3.67132144e-01,
            3.46991954e-01, 5.28449990e-02, 5.59345855e-01, 3.78858405e-01,
            3.04291215e-02, 3.08825931e-01, 6.80110835e-01, 5.91798611e-01,
            6.65855880e-01, 3.94774892e-01, 3.02902466e-01, 6.80100651e-01,
            6.54124098e-01, 5.24927435e-01, 4.62052243e-01, 3.45808123e-01,
            6.26498936e-03, 6.41988807e-01, 4.39707402e-01, 5.05208254e-01,
            2.97153597e-01, 3.33327649e-01, 8.21397594e-02, 6.88471914e-01,
            5.27490497e-01, 4.00030893e-01, 5.45887428e-01, 3.90986323e-01,
            4.55141977e-01, 7.05649358e-02, 1.53083484e-01, 2.49663423e-02,
            6.00799427e-01, 6.58808094e-01, 3.02637895e-01, 1.95811352e-01,
            6.18674219e-01, 5.40908148e-01, 3.44868346e-01, 4.09740303e-01,
            1.43534013e-01, 3.51370472e-01, 5.62058202e-01, 5.34501606e-01,
            3.73671869e-01, 2.94745718e-01, 8.06459788e-06, 3.62176459e-01,
            6.74486385e-01, 2.10854485e-01, 1.59712660e-01, 2.92944236e-01,
            5.66148384e-01, 5.22805055e-01, 4.09398949e-01, 4.91663732e-01,
            5.41092674e-01, 6.60029846e-01, 5.26102372e-01, 3.56176773e-01,
            5.73583673e-01, 4.95133494e-01, 6.38384756e-01, 1.35330473e-01,
            5.93706362e-01, 5.03957344e-01, 6.56473226e-01, 2.63754778e-01,
            5.44956231e-01, 2.66414611e-01, 5.81548632e-01, 3.03363716e-01,
            3.58140287e-02, 3.83107351e-01, 1.09013481e-01, 4.66282184e-01,
            6.28051691e-01, 4.67195007e-01, 3.80439836e-01, 2.12641335e-01,
            4.30647109e-01, 5.07868281e-01, 1.98072577e-01, 4.75113495e-01,
            3.92221880e-02, 6.54641872e-01, 4.65433879e-01, 1.37930840e-01,
            5.20511804e-01, 1.92592868e-01, 4.67891999e-02, 7.85360048e-02,
            5.11462897e-01, 2.93013163e-01, 2.75603899e-01, 1.42325948e-01,
            1.86133416e-01, 4.18409424e-01, 2.06066273e-01, 4.15226067e-01,
            4.49509288e-01, 4.89798502e-01, 2.16890878e-01, 6.56225815e-01,
            4.85992933e-01, 2.82909194e-01, 4.04053553e-01, 6.77209616e-01,
            5.76694450e-03, 6.33916828e-01, 1.79325245e-01, 5.01455815e-01,
            4.08578371e-01, 4.42011810e-01, 4.94943632e-01, 5.12043869e-01,
            1.72161305e-01, 1.87708643e-01, 4.33399256e-02, 5.07892142e-01,
            4.66399100e-01, 6.59523103e-01, 5.58046084e-01, 6.46505659e-01,
            5.14926213e-01, 3.61440921e-01, 4.05109114e-01, 1.86355055e-01,
            6.20606617e-01, 5.45775098e-01, 4.36854083e-01, 4.88988186e-01,
            3.06749176e-01, 6.48980273e-01, 5.71668450e-01, 3.07701518e-01,
            2.08669240e-01, 3.89382896e-02, 6.42624118e-01, 5.29869935e-01,
            4.54131813e-01, 1.10551612e-01, 4.88612934e-02, 5.64308924e-01,
            7.65455348e-02, 2.14741762e-01, 3.56974721e-01, 2.47374507e-01,
            5.74310977e-01, 6.60441043e-01, 2.45668654e-01, 6.52335238e-01,
            7.10946941e-02, 3.83567680e-01, 5.95250039e-01, 3.04437590e-01,
            1.59145318e-01, 1.14566478e-01, 4.98192340e-01, 3.56924600e-01,
            5.79154535e-02, 1.67420581e-01, 7.30199256e-02, 3.59899047e-01,
            1.70065265e-01, 2.71043331e-01, 4.27186944e-01, 2.88193341e-01,
            3.74926778e-01, 4.86690587e-01, 3.46651744e-01, 6.03403312e-01,
            3.69103692e-01, 4.19796195e-01, 2.33733787e-01, 4.79708897e-01,
            1.41621325e-01, 2.90616970e-01, 5.49498051e-01, 3.28728531e-01,
            4.35592027e-01, 1.75555039e-01, 5.01452935e-01, 5.79552629e-01,
            5.84549155e-01, 3.12813242e-01, 5.92622831e-01, 3.23221293e-01,
            6.25062125e-01, 2.69914936e-01, 6.71012088e-01, 4.49379799e-01,
            3.87510367e-01, 1.52569545e-01, 2.58220388e-01, 2.56169314e-01,
            5.00963635e-01, 4.91278916e-01, 3.35475866e-01, 5.57633193e-01,
            3.80135197e-01, 5.31307745e-01, 5.42013353e-01, 5.56598622e-01,
            4.30079655e-01, 1.17752540e-01, 6.17184846e-01, 6.29536016e-01,
            4.13916517e-01, 6.16296514e-01, 4.38340263e-01, 1.87803167e-01,
            2.17528167e-01, 1.48966088e-01, 6.29265706e-01, 1.30256482e-01,
            4.82538669e-01, 5.69649804e-01, 5.51912516e-01, 4.79906635e-01,
            6.50912669e-01, 2.24934439e-01, 5.64873175e-01, 5.27381955e-01,
            2.55318833e-01, 2.15909058e-02, 1.68709725e-01, 6.15596619e-01,
            5.72134540e-01, 1.72599703e-01, 4.66182471e-02, 6.92379598e-01,
            5.18601569e-01, 1.67392608e-01, 2.85016241e-01, 1.55444418e-02,
            5.43245959e-01, 1.60822748e-01, 5.29809395e-01, 5.28387915e-01,
            5.80247110e-01, 3.90515899e-01, 5.99179468e-01, 5.24525382e-01,
            4.90330236e-01, 6.43366804e-01, 2.36990680e-01, 6.80059112e-01,
            5.57577337e-01, 4.11145947e-01, 3.15218622e-01, 4.69629128e-01,
            6.52731146e-01, 3.46822739e-01, 4.83217718e-01, 1.90815249e-01,
            4.40800245e-01, 3.15501183e-01, 5.80418630e-01, 4.22610051e-01,
            5.80964758e-01, 6.66059091e-01, 3.38599370e-01, 4.04237062e-01,
            2.46015028e-01, 2.20031015e-02, 6.10643955e-01, 4.56815468e-01,
            5.98183820e-01, 6.91232639e-01, 6.22006620e-01, 6.74697418e-01,
            5.87364555e-01, 2.32397728e-01, 6.76056343e-01, 5.89892415e-02,
            4.92376331e-02, 1.58984242e-01, 1.86153109e-01, 1.16095976e-01,
            6.74485439e-01, 6.70801131e-01, 2.26141983e-01, 1.61160873e-01,
            1.77766648e-01, 4.67177185e-01, 1.01838517e-01, 5.99335894e-01,
            5.73488043e-02, 2.09621232e-01, 5.57155094e-01, 4.02509933e-01,
            1.89673084e-01, 5.70268296e-02, 3.94946440e-02, 3.98065814e-01,
            2.17561916e-01, 6.16567803e-01, 1.35279121e-01, 5.14066681e-01,
            2.81975601e-01, 8.73379504e-02, 6.59491298e-01, 3.93518590e-01,
            3.39381815e-01, 1.59303747e-01, 1.28419624e-01, 1.51598081e-01,
            5.72339218e-01, 3.95967584e-01, 2.54655848e-01, 5.51694131e-01,
            4.66324896e-01, 1.73057331e-01, 5.49384860e-01, 5.20044599e-01,
            3.17685707e-01, 6.50009348e-02, 6.38208093e-01, 2.02025310e-01,
            3.24440745e-01, 5.27190862e-01, 3.37746606e-01, 5.33341332e-01,
            5.73616065e-01, 2.11158183e-02, 2.35955310e-01, 3.53403869e-01,
            3.48258006e-02, 3.82034974e-01, 5.81675525e-01, 1.73090372e-02,
            1.64607592e-01, 5.02253260e-01, 4.21654946e-01, 1.20258298e-01,
            5.55519880e-01, 5.20393866e-01, 5.35644331e-01, 4.69533138e-01,
            6.44034041e-01, 4.58622141e-02, 5.73134680e-01, 5.37358229e-01,
            2.39193825e-01, 6.66455694e-01, 6.46848475e-01, 4.23834278e-01,
            6.56522298e-01, 5.12282191e-01, 4.89532536e-01, 6.86291155e-01]),
     0.7363770250368189)




```python
#PART 3
The acceptance rate should be around:

1 / (2ln2) ≈ 0.72

So about 72% of proposed samples should be accepted.

Written answer for Part 3
I chose the Uniform(0, ln(2)) distribution as the proposal distribution because the target density is only positive on this interval. 
It is easy to sample from and covers the full support of the target density. 
Since f(x) = e^x is maximized at x = ln(2), the maximum value is 2. 
For the uniform proposal, g(x) = 1/ln(2), so we can choose M = 2ln(2). 
The acceptance probability becomes e^x / 2. 
The expected acceptance rate is approximately 1/(2ln(2)) = 0.72, so around 72% of samples ar
```


```python
#PART 4- Yes, it is possible to use Accept-Reject sampling with a Gaussian proposal distribution. 
The target density has the form C exp(-(x - 2)^2), which is Gaussian-shaped and centered around x = 2. 
A reasonable proposal distribution is a Gaussian centered at 2, because it puts most of its probability mass in the same region as the target density. 
Accept-Reject sampling is possible if there exists a finite constant M such that f(x) ≤ M g(x) for all x. 
With a suitable Gaussian proposal, this ratio is bounded, so Accept-Reject can be used.
```
