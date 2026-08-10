# Exam 16th of June 2026, 8.00-13.00 for the course 1MS041 (Introduction to Data Science / Introduktion till dataanalys)

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
examID="0010-MKZ"

```

---
## Exam vB, PROBLEM 1
Maximum Points = 14


This problem is about **PCA/SVD** for handwritten digit data. Unless stated otherwise, every vector or matrix you create should be a NumPy array.

The file `data/digits.csv` contains one row per image. The first 64 columns are pixel intensities for an 8 by 8 handwritten digit image, and the last column `target` is the digit label.

1. **[4p] Data and SVD.** Load the data. Store the feature matrix in `problem1_X` and the labels in `problem1_y`. Center the feature matrix column-wise and store it in `problem1_X_centered`. Compute the compact SVD
   $$X_c = U D V^T$$
   and store the matrices in `problem1_U`, `problem1_D`, and `problem1_V`, where `problem1_D` is a square diagonal matrix and `problem1_V` contains the right singular vectors as columns. If `problem1_X` has shape `(n_samples, 64)`, the compact shapes should be `problem1_U.shape == (n_samples, 64)`, `problem1_D.shape == (64, 64)`, and `problem1_V.shape == (64, 64)`. If you use `np.linalg.svd`, remember that compact SVD means `full_matrices=False`.

2. **[3p] Explained variance.** Compute the cumulative explained variance from the singular values on the diagonal of `problem1_D`, ordered from largest to smallest, and store it in `problem1_explained_variance`. If the singular values are $d_1 \geq d_2 \geq \cdots \geq d_{64}$, then the cumulative explained variance after the first $k$ components is
   $$
   \frac{\sum_{j=1}^k d_j^2}{\sum_{j=1}^{64} d_j^2}.
   $$
   Thus `problem1_explained_variance[k-1]` should contain this value. Store in `problem1_num_components` the smallest number of components needed to explain at least 90% of the variance.

3. **[3p] Two-dimensional PCA coordinates and interpretation.** Store the projection onto the first two principal directions in `problem1_scores_2d`. Plot these coordinates and color the points by the digit labels. In the markdown cell below, briefly explain what the plot suggests and why PCA can or cannot separate all digits perfectly.

4. **[4p] Nearest-centroid classification in PCA space.** Use the centered data and PCA directions already computed above; do not recompute PCA after the train/test split. Store the first `problem1_num_components` PCA coordinates in `problem1_scores_k`. Use a deterministic 80/20 split where the first 80% of the rows are training rows and the remaining 20% are test rows. For each digit label `0, 1, ..., 9`, compute the centroid of the training points in PCA space and store these ten centroids in `problem1_centroids`, with row `i` corresponding to digit `i`. Classify each test row by the nearest centroid in Euclidean distance and store the predicted labels in `problem1_test_predictions`, in increasing row-index order. Store the test accuracy in `problem1_test_accuracy`.



```python
#pip install pandas
```


```python
# Part 1: 4 points

import numpy as np
import pandas as pd

# Expected: problem1_X is a NumPy array with shape (n_samples, 64).
# Expected: problem1_y is a one-dimensional array with length n_samples.
# Load data/digits.csv. The first 64 columns are features and the last column is the digit label.
#problem1_data = pd.read_csv('data/SVD.csv').to_numpy()
data = pd.read_csv('data/digits.csv').to_numpy() # Teacher edit: your loading the old exam data
problem1_X = np.array(problem1_X)  # ensure it's a NumPy array
problem1_X = problem1_X.reshape(-1, 64) 
problem1_y = df["target"].values

# Expected: a NumPy array with the same shape as problem1_X.
# Center the feature matrix column-wise by subtracting the mean of each feature column.
df = pd.DataFrame(data, columns=[f"Feature_{i+1}" for i in range(cols)])
problem1_X_centered = df - df.mean()

# Expected compact SVD X_c = U D V^T.
# problem1_U should have shape (n_samples, 64).
# problem1_D should have shape (64, 64), with singular values on the diagonal.
# problem1_V should have shape (64, 64), so that its columns are principal directions.
# If you use np.linalg.svd, use full_matrices=False and set V = Vt.T.
U, d, Vt = np.linalg.svd(problem1_data, full_matrices=False)

problem1_U = U
problem1_D = np.diag(d)
problem1_V = Vt.T

problem1_first_right_singular_vector = problem1_V[:, 0].flatten()
problem1_first_left_singular_vector = problem1_U[:, 0].flatten()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[3], line 11
          6 # Expected: problem1_X is a NumPy array with shape (n_samples, 64).
          7 # Expected: problem1_y is a one-dimensional array with length n_samples.
          8 # Load data/digits.csv. The first 64 columns are features and the last column is the digit label.
          9 #problem1_data = pd.read_csv('data/SVD.csv').to_numpy()
         10 data = pd.read_csv('data/digits.csv').to_numpy() # Teacher edit: your loading the old exam data
    ---> 11 problem1_X = np.array(problem1_X)  # ensure it's a NumPy array
         12 problem1_X = problem1_X.reshape(-1, 64) 
         13 problem1_y = df["target"].values
    

    NameError: name 'problem1_X' is not defined



```python
# Part 2: 3 points

# Expected: problem1_explained_variance is a one-dimensional NumPy array of length 64.
# Entry k should be the cumulative fraction of variance explained by the first k+1 singular values.
# The values should be increasing and the last value should be 1.
singular_values = np.diag(problem1_D)
problem1_explained_variance = np.cumsum(singular_values**2) / np.sum(singular_values**2)
# Expected: a Python int.
# Store the smallest number of principal components needed to explain at least 90% of the variance.
problem1_num_components = int(np.searchsorted(problem1_explained_variance, 0.90))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[4], line 6
          1 # Part 2: 3 points
          2 
          3 # Expected: problem1_explained_variance is a one-dimensional NumPy array of length 64.
          4 # Entry k should be the cumulative fraction of variance explained by the first k+1 singular values.
          5 # The values should be increasing and the last value should be 1.
    ----> 6 singular_values = np.diag(problem1_D)
          7 problem1_explained_variance = np.cumsum(singular_values**2) / np.sum(singular_values**2)
          8 # Expected: a Python int.
          9 # Store the smallest number of principal components needed to explain at least 90% of the variance.
    

    NameError: name 'problem1_D' is not defined



```python
# Part 3: 3 points

# Expected: a NumPy array with shape (n_samples, 2).
# Store the centered data projected onto the first two principal directions from problem1_V.
problem1_scores_2d = X_centered @ problem1_V[:, :2]

# Put your plotting code below. The plot should use problem1_scores_2d and color by problem1_y.
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    problem1_scores_2d[:, 0],
    problem1_scores_2d[:, 1],
    c=problem1_y,
    cmap="viridis",
    edgecolor="k"
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Projection onto First Two Principal Components")
plt.colorbar(scatter, label="Target")
plt.grid(True)
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[5], line 5
          1 # Part 3: 3 points
          2 
          3 # Expected: a NumPy array with shape (n_samples, 2).
          4 # Store the centered data projected onto the first two principal directions from problem1_V.
    ----> 5 problem1_scores_2d = X_centered @ problem1_V[:, :2]
          7 # Put your plotting code below. The plot should use problem1_scores_2d and color by problem1_y.
          8 plt.figure(figsize=(8, 6))
    

    NameError: name 'X_centered' is not defined



## Free text answer for Part 3

Write your interpretation below this line.



```python
# Part 4: 4 points

# Use the PCA directions from Part 1; do not recompute PCA after the train/test split.
# Use split_index = int(0.8 * n_samples): rows before split_index are training rows, the rest are test rows.

# Expected: a NumPy array with shape (n_samples, problem1_num_components).
# Store the projection onto the first problem1_num_components principal directions.
problem1_scores_k = XXX

# Expected: a NumPy array with shape (10, problem1_num_components).
# Row i should be the training-set centroid in PCA space for digit i.
problem1_centroids = XXX

# Expected: a one-dimensional NumPy array of length n_test.
# Store nearest-centroid predictions for the test rows, in increasing row-index order.
problem1_test_predictions = XXX

# Expected: a single float between 0 and 1.
# Store the fraction of test rows whose predicted label is correct.
problem1_test_accuracy = XXX
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[6], line 8
          1 # Part 4: 4 points
          2 
          3 # Use the PCA directions from Part 1; do not recompute PCA after the train/test split.
       (...)
          6 # Expected: a NumPy array with shape (n_samples, problem1_num_components).
          7 # Store the projection onto the first problem1_num_components principal directions.
    ----> 8 problem1_scores_k = XXX
         10 # Expected: a NumPy array with shape (10, problem1_num_components).
         11 # Row i should be the training-set centroid in PCA space for digit i.
         12 problem1_centroids = XXX
    

    NameError: name 'XXX' is not defined


---
#### Local Test for Exam vB, PROBLEM 1
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.


```python

# Optional local format checks for Problem 1. These checks do not prove correctness,
# but they are meant to catch the most common SVD shape issue.
import numpy as np

try:
    assert isinstance(problem1_X, np.ndarray)
    assert problem1_X.shape[1] == 64
    assert isinstance(problem1_y, np.ndarray)
    assert problem1_y.shape[0] == problem1_X.shape[0]

    n_samples, n_dimensions = problem1_X.shape
    expected_U_shape = (n_samples, n_dimensions)
    expected_D_shape = (n_dimensions, n_dimensions)
    expected_V_shape = (n_dimensions, n_dimensions)

    print("Expected compact SVD shapes:")
    print("  problem1_U:", expected_U_shape)
    print("  problem1_D:", expected_D_shape)
    print("  problem1_V:", expected_V_shape)
    print("Your shapes:")
    print("  problem1_U:", getattr(problem1_U, 'shape', None))
    print("  problem1_D:", getattr(problem1_D, 'shape', None))
    print("  problem1_V:", getattr(problem1_V, 'shape', None))

    if getattr(problem1_U, 'shape', None) == (n_samples, n_samples):
        print("Warning: problem1_U has the full SVD shape. NumPy uses full_matrices=True by default.")
        print("Use np.linalg.svd(problem1_X_centered, full_matrices=False) for the compact SVD.")

    assert problem1_U.shape == expected_U_shape
    assert problem1_D.shape == expected_D_shape
    assert problem1_V.shape == expected_V_shape
    assert np.allclose(problem1_X_centered, problem1_U @ problem1_D @ problem1_V.T, atol=5e-3)
    assert problem1_scores_2d.shape == (problem1_X.shape[0], 2)

    k = int(problem1_num_components)
    split_index = int(0.8 * n_samples)
    n_test = n_samples - split_index
    assert problem1_scores_k.shape == (n_samples, k)
    assert problem1_centroids.shape == (10, k)
    assert problem1_test_predictions.shape == (n_test,)
    assert 0 <= float(problem1_test_accuracy) <= 1
    print("Problem 1 format checks passed.")
except Exception as error:
    print("Problem 1 format check failed:", error)

```

    Problem 1 format check failed: name 'problem1_X' is not defined
    


```python

```

    Beginning tests for problem 1
    
    ---------------------------------
    Beginning test for part 1
    ---------------------------------
    
    -----Beginning test------
    name 'problem1_X' is not defined
    problem1_X does not contain the feature matrix from data/digits.csv
    You got 1.00 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    name 'problem1_y' is not defined
    problem1_y does not contain the digit labels
    You got 0.50 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    name 'problem1_X_centered' is not defined
    problem1_X_centered is not the column-centered version of your problem1_X
    You got 0.75 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    
    The SVD matrices do not have compact SVD shapes for your centered data. Expected U=(n_samples,r), D=(r,r), V=(n_dimensions,r), where r=min(n_samples,n_dimensions).
    You got 0.75 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    
    Your SVD factors do not reconstruct your problem1_X_centered matrix. If you used NumPy's default full SVD, the first r columns of U should be paired with the singular values.
    You got 0.75 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    name 'problem1_U' is not defined
    The supplied singular-vector matrices are not orthonormal in the expected sense
    You got 0.25 points deduction 
    -----Ending test---------
    
    ---------------------------------
    Beginning test for part 2
    ---------------------------------
    
    -----Beginning test------
    name 'problem1_explained_variance' is not defined
    problem1_explained_variance has the wrong shape for your SVD
    You got 0.50 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    name 'problem1_explained_variance' is not defined
    The explained variance should be increasing and end at 1
    You got 0.50 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    name 'problem1_explained_variance' is not defined
    The explained variance values are not consistent with the singular values of your problem1_X_centered matrix
    You got 1.20 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    name 'problem1_num_components' is not defined
    problem1_num_components is not the smallest k explaining at least 90% of the variance for your problem1_X_centered matrix
    You got 0.80 points deduction 
    -----Ending test---------
    
    ---------------------------------
    Beginning test for part 3
    ---------------------------------
    
    -----Beginning test------
    name 'problem1_scores_2d' is not defined
    problem1_scores_2d should have shape (n_samples, 2), using your problem1_X
    You got 0.75 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    name 'problem1_scores_2d' is not defined
    problem1_scores_2d is not the projection of your centered data onto your first two principal directions
    You got 1.25 points deduction 
    -----Ending test---------
    
    Manual points: -1
    Part 3 free-text interpretation requires manual review by the instructor.
    ---------------------------------
    Beginning test for part 4
    ---------------------------------
    
    -----Beginning test------
    name 'problem1_scores_k' is not defined
    problem1_scores_k should be the projection of your centered data onto the first problem1_num_components principal directions computed above
    You got 0.75 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    
    problem1_centroids should contain the ten first-80%-training-set class centroids in your PCA space, with row i corresponding to digit i
    You got 1.25 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    
    problem1_test_predictions should contain nearest-centroid predictions for your last-20%-test rows in increasing row-index order
    You got 1.25 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    
    problem1_test_accuracy should be the fraction of correctly classified test rows, using your prediction array and your labels
    You got 0.75 points deduction 
    -----Ending test---------
    
    
    All tests complete, you got = 0.00 points
    The number of points you have scored for this problem is 0.0 out of 14
    The number of points you have accumulated thus far is   0.0 out of 14
    

---
## Exam vB, PROBLEM 2
Maximum Points = 12


This problem is about **linear regression** and evaluating prediction error on held-out data. Unless stated otherwise, every vector or matrix you create should be a NumPy array.

The file `data/auto.csv` contains car measurements. We will predict fuel efficiency `mpg` from the features `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, and `model-year`, in that order.

1. **[2p] Load and clean data.** Load the file, remove rows where `horsepower` is missing, store the feature matrix in `problem2_X` and the target vector in `problem2_y`. Missing horsepower values are encoded as `?` or as an empty value.

2. **[2p] Train/test split and standardization.** Let `problem2_split_index = int(0.8 * n)`, where `n` is the number of cleaned rows. Use rows before this index as the training set and the remaining rows as the test set. Store the four arrays `problem2_X_train`, `problem2_X_test`, `problem2_y_train`, and `problem2_y_test`. Standardize features using the training mean and training standard deviation only, computed with NumPy's default `np.std(..., axis=0)`. Store the standardized train and test matrices in `problem2_X_train_standardized` and `problem2_X_test_standardized`.

3. **[3p] Fit linear regression.** Fit linear regression with an intercept using least squares on the standardized training features. Store the coefficient vector, including the intercept as the first entry, in `problem2_beta`. If you use `sklearn.linear_model.LinearRegression`, the intercept is stored in `model.intercept_` and the feature coefficients are stored in `model.coef_`, so the required order is `[model.intercept_, model.coef_[0], ..., model.coef_[5]]`. Store test predictions in `problem2_y_pred_test` and residuals `y_test - y_pred` in `problem2_residuals_test`.

4. **[3p] Test metrics and baseline.** Compute test MSE, MAE, and R^2 in `problem2_mse_test`, `problem2_mae_test`, and `problem2_r2_test`. Also compute `problem2_baseline_mse_test` for the predictor that always predicts the training-set mean of `mpg`, and set `problem2_model_beats_baseline` to `True` exactly when the linear model has smaller test MSE.

5. **[2p] Hoeffding interval.** Clip the absolute residuals at 50 and compute their empirical mean on the test set. Construct a two-sided Hoeffding confidence interval with confidence level 95% for the expected clipped absolute error. You may use `epsilon_bounded` from `Utils.py`, but you should decide its arguments from the sample size, the bound, and the confidence level. Since the clipped absolute error is between 0 and 50, intersect your final interval with `[0, 50]`. Store the interval endpoints in `problem2_lower_bound` and `problem2_upper_bound`.



```python
import pandas as pd
import numpy as np

problem2_df = pd.read_csv("auto.csv")

print(problem2_df.head())
print(problem2_df.columns)
```

        mpg  cylinders  displacement horsepower  weight  acceleration  model-year
    0  18.0          8         307.0      130.0    3504          12.0          70
    1  15.0          8         350.0      165.0    3693          11.5          70
    2  18.0          8         318.0      150.0    3436          11.0          70
    3  16.0          8         304.0      150.0    3433          12.0          70
    4  17.0          8         302.0      140.0    3449          10.5          70
    Index(['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
           'acceleration', 'model-year'],
          dtype='str')
    


```python
# Part 1: 2 points

import numpy as np
import pandas as pd


# Expected: problem2_X is a NumPy array with shape (n_clean, 6).
# Use the columns cylinders, displacement, horsepower, weight, acceleration, model-year, in that order.
# Remove rows with missing horsepower before creating problem2_X and problem2_y.
problem2_df = pd.read_csv("auto.csv")

problem2_df.head()

print(problem2_df.isnull().sum())

problem2_df_dropped = problem2_df.dropna()
print(problem2_df_dropped.shape)

problem2_df['horsepower'] = problem2_df['horsepower'].replace('?', np.nan)

# Drop rows where horsepower is missing (NaN or empty)
problem2_df= problem2_df.dropna(subset=['horsepower'])

problem2_X = problem2_df.drop(columns=['horsepower'])

# Expected: a one-dimensional NumPy array with length n_clean.
# Store the target mpg values corresponding to the rows in problem2_X.
problem2_y = problem2_df['horsepower']

```

    mpg             0
    cylinders       0
    displacement    0
    horsepower      0
    weight          0
    acceleration    0
    model-year      0
    dtype: int64
    (398, 7)
    


```python
# Part 2: 2 points

# Expected: a Python int equal to int(0.8 * n_clean).
n = len(problem2_y) # Teacher edit: you need to compute n_clean. But the rest of the code still did not run
problem2_split_index = int(0.8 * n)

# Expected: NumPy arrays from a contiguous split of the cleaned data.
# Training rows are before problem2_split_index; test rows are from problem2_split_index onward.
X = problem2_X.values # Teacher edit: undefined variable
y = problem2_y.values # Teacher edit: undefined variable. Ok, bailing out fixing the rest since the rest of the code did not run, too far away from being able to run.
problem2_X_train = X[:problem2_split_index]
problem2_X_test  = X[problem2_split_index:]
problem2_y_train = y[:problem2_split_index]
problem2_y_test  = y[problem2_split_index:] 


# Expected: standardized feature matrices with the same shapes as problem2_X_train and problem2_X_test.
# Use the training mean and training standard deviation only, and apply them to both train and test features.

# Compute mean and std from training data only
train_mean = np.mean(problem2_X_train, axis=0)
train_std = np.std(problem2_X_train, axis=0)  # default: population std

# Standardize using training statistics
problem2_X_train_standardized = (problem2_X_train - train_mean) / train_std
problem2_X_test_standardized = (problem2_X_test - train_mean) / train_std

```


```python
# Part 3: 3 points
import numpy as np
from sklearn.linear_model import LinearRegression

problem2_model = LinearRegression()
problem2_model.fit(X_train, y_train)

# Expected: a NumPy array with shape (7,).
# Store the fitted least-squares coefficients with the intercept first, followed by the six feature coefficients.
# If you use sklearn.linear_model.LinearRegression, combine model.intercept_ and model.coef_ in this order.
problem2_beta = np.concatenate(([model.intercept_], model.coef_))

# Expected: a one-dimensional NumPy array with length n_test.
# Store the linear-regression predictions for the test rows.
problem2_y_pred_test = model.predict(X_test)

# Expected: a one-dimensional NumPy array with length n_test.
# Store residuals as y_test - y_pred_test.
problem2_residuals_test = y_test - problem2_y_pred_test
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[11], line 6
          3 from sklearn.linear_model import LinearRegression
          5 problem2_model = LinearRegression()
    ----> 6 problem2_model.fit(X_train, y_train)
          8 # Expected: a NumPy array with shape (7,).
          9 # Store the fitted least-squares coefficients with the intercept first, followed by the six feature coefficients.
         10 # If you use sklearn.linear_model.LinearRegression, combine model.intercept_ and model.coef_ in this order.
         11 problem2_beta = np.concatenate(([model.intercept_], model.coef_))
    

    NameError: name 'X_train' is not defined



```python
# Part 4: 3 points

# Expected: scalar floats computed on the test set.
problem2_model = LinearRegression()
problem2_model.fit(problem2_X_train,problem2_X_test, problem2_y_train, problem2_y_test,)
y_pred = model.predict(problem2_X_test)
problem2_mse_test = mean_squared_error(problem2_y_test, y_pred)
problem2_mae_test = mean_absolute_error(problem2_y_test, y_pred)
problem2_r2_test = r2_test(problem2_y_test, y_pred)

# Expected: a scalar float.
# Baseline means the predictor that always predicts the training-set mean of mpg.
baseline_model = DummyRegressor(strategy="mean")
baseline_model.fit([[0], [0], [0], [0]], y_train)  
y_pred_baseline = baseline_model.predict([[0], [0], [0], [0]])

problem2_baseline_mse_test = mean_squared_error(problem2_y_test, y_pred_baseline)

# Expected: a Python bool.
# Store True exactly when the model test MSE is smaller than the baseline test MSE.
problem2_model_beats_baseline = XXX
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[12], line 5
          1 # Part 4: 3 points
          2 
          3 # Expected: scalar floats computed on the test set.
          4 problem2_model = LinearRegression()
    ----> 5 problem2_model.fit(problem2_X_train,problem2_X_test, problem2_y_train, problem2_y_test,)
          6 y_pred = model.predict(problem2_X_test)
          7 problem2_mse_test = mean_squared_error(problem2_y_test, y_pred)
    

    File /opt/homebrew/lib/python3.11/site-packages/sklearn/base.py:1365, in _fit_context.<locals>.decorator.<locals>.wrapper(estimator, *args, **kwargs)
       1358     estimator._validate_params()
       1360 with config_context(
       1361     skip_parameter_validation=(
       1362         prefer_skip_nested_validation or global_skip_validation
       1363     )
       1364 ):
    -> 1365     return fit_method(estimator, *args, **kwargs)
    

    TypeError: LinearRegression.fit() takes from 3 to 4 positional arguments but 5 were given



```python
# Part 5: 2 points

# Expected: two scalar floats.
# Clip the absolute residuals at 50, use confidence level 95%, and construct a two-sided Hoeffding interval.
# Store the final interval after intersecting it with [0, 50].
problem2_lower_bound = XXX
problem2_upper_bound = XXX
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[13], line 6
          1 # Part 5: 2 points
          2 
          3 # Expected: two scalar floats.
          4 # Clip the absolute residuals at 50, use confidence level 95%, and construct a two-sided Hoeffding interval.
          5 # Store the final interval after intersecting it with [0, 50].
    ----> 6 problem2_lower_bound = XXX
          7 problem2_upper_bound = XXX
    

    NameError: name 'XXX' is not defined


---
#### Local Test for Exam vB, PROBLEM 2
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.


```python

# Optional local format checks for Problem 2. These checks do not prove correctness.
import numpy as np

try:
    assert isinstance(problem2_X, np.ndarray)
    assert isinstance(problem2_y, np.ndarray)
    assert problem2_X.shape[0] == problem2_y.shape[0]
    assert problem2_X.shape[1] == 6
    assert problem2_split_index == int(0.8 * problem2_X.shape[0])
    assert problem2_X_train.shape[0] == problem2_split_index
    assert problem2_X_test.shape[0] == problem2_X.shape[0] - problem2_split_index
    assert problem2_y_train.shape[0] == problem2_X_train.shape[0]
    assert problem2_y_test.shape[0] == problem2_X_test.shape[0]
    assert problem2_X_train_standardized.shape == problem2_X_train.shape
    assert problem2_X_test_standardized.shape == problem2_X_test.shape
    assert problem2_beta.shape == (7,)
    print("Problem 2 format checks passed.")
except Exception as error:
    print("Problem 2 format check failed:", error)

```

    Problem 2 format check failed: 
    


```python

```

    Beginning tests for problem 2
    
    ---------------------------------
    Beginning test for part 1
    ---------------------------------
    
    -----Beginning test------
    
    problem2_X does not contain the cleaned feature matrix values with missing horsepower rows removed
    You got 1.00 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    
    problem2_X should be stored as a NumPy array with shape (n_clean, 6)
    You got 0.20 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    
    problem2_y does not contain the cleaned target values
    You got 0.60 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    
    problem2_y should be stored as a one-dimensional NumPy array
    You got 0.20 points deduction 
    -----Ending test---------
    
    ---------------------------------
    Beginning test for part 2
    ---------------------------------
    
    -----Beginning test------
    problem2_split_index is correct for your cleaned data
    -----Ending test---------
    
    -----Beginning test------
    The train/test array values are correct for your cleaned data
    -----Ending test---------
    
    -----Beginning test------
    The train/test variables are NumPy arrays with the expected shapes
    -----Ending test---------
    
    -----Beginning test------
    The standardized feature matrix values are correct for your training split
    -----Ending test---------
    
    -----Beginning test------
    The standardized feature matrices are NumPy arrays with the expected shapes
    -----Ending test---------
    
    ---------------------------------
    Beginning test for part 3
    ---------------------------------
    
    -----Beginning test------
    name 'problem2_beta' is not defined
    problem2_beta is not the least-squares coefficient vector with intercept first for your standardized training data
    You got 1.20 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    name 'problem2_y_pred_test' is not defined
    problem2_y_pred_test has incorrect numeric values for your fitted model and test data
    You got 0.70 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    name 'problem2_y_pred_test' is not defined
    problem2_y_pred_test should be stored as a one-dimensional NumPy array
    You got 0.20 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    name 'problem2_residuals_test' is not defined
    problem2_residuals_test should have values y_test - y_pred_test for your split
    You got 0.70 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    name 'problem2_residuals_test' is not defined
    problem2_residuals_test should be stored as a one-dimensional NumPy array
    You got 0.20 points deduction 
    -----Ending test---------
    
    ---------------------------------
    Beginning test for part 4
    ---------------------------------
    
    -----Beginning test------
    
    problem2_mse_test is incorrect for your test predictions
    You got 0.80 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    
    problem2_mae_test is incorrect for your test predictions
    You got 0.70 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    
    problem2_r2_test is incorrect for your test predictions
    You got 0.70 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    
    The baseline MSE or model-beats-baseline indicator is incorrect for your split
    You got 0.80 points deduction 
    -----Ending test---------
    
    ---------------------------------
    Beginning test for part 5
    ---------------------------------
    
    -----Beginning test------
    name 'problem2_lower_bound' is not defined
    The Hoeffding interval endpoints are not valid numbers
    You got 2.00 points deduction 
    -----Ending test---------
    
    
    All tests complete, you got = 2.00 points
    The number of points you have scored for this problem is 2.0000000000000027 out of 12
    The number of points you have accumulated thus far is   2.0000000000000027 out of 26
    

---
## Exam vB, PROBLEM 3
Maximum Points = 14


This problem is about modelling warehouse package movement as a finite homogeneous Markov chain.

The file `data/warehouse_transitions.csv` contains observed transitions between five zones:

`Dock`, `Sort`, `Storage`, `Packing`, `Dispatch`.

Use this exact state order whenever you create vectors or matrices.

1. **[3p] Estimate transition matrix.** Load the transition data and estimate the transition matrix by maximum likelihood. Store it in `problem3_transition_matrix` as a 5 by 5 row-stochastic NumPy array, where entry `(i, j)` is the estimated probability of moving from state `i` to state `j`.

2. **[2p] Four-step probability.** Starting from `Dock`, compute the probability of being in `Dispatch` after exactly 4 steps and store it in `problem3_prob_dispatch_after_4_from_dock`.

3. **[2p] Simulation.** Starting from `Dock`, simulate 20000 chains for 8 steps using `np.random.default_rng(20260616)` and the transition probabilities in `problem3_transition_matrix`. Store the empirical distribution after 8 steps in `problem3_simulated_distribution_after_8` as a length-5 probability vector in the state order above.

4. **[2p] Chain structure.** Decide whether the estimated chain is irreducible and aperiodic. Store Boolean answers in `problem3_is_irreducible` and `problem3_is_aperiodic`.

5. **[2p] Stationary distribution.** Compute a stationary distribution for the estimated transition matrix and store it in `problem3_stationary_distribution` as a length-5 probability vector in the state order above. In the markdown cell below, briefly explain what the stationary distribution means in this warehouse context.

6. **[3p] Hitting time.** Compute the expected number of steps to hit `Dispatch` for the first time when starting from `Dock`. Store it in `problem3_expected_steps_to_dispatch_from_dock`. An exact computation gives full credit; a sufficiently accurate simulation estimate can receive partial credit.



```python
# Part 1: 3 points

import numpy as np
import pandas as pd

# Keep this exact state order throughout the problem.
problem3_states = ["Dock", "Sort", "Storage", "Packing", "Dispatch"]

# Expected: a NumPy array with shape (5, 5).
# Row i should contain transition probabilities out of problem3_states[i].
# Column j should correspond to transitions into problem3_states[j].
# Estimate the entries from data/warehouse_transitions.csv and make each row sum to 1.

problem3_transition_matrix = pd.read_csv("data/warehouse_transitions.csv", header=None).to_numpy()

# Create a mapping from state name to index
state_to_idx = {state: i for i, state in enumerate(problem3_states)}

# Initialize count matrix
counts = np.zeros((len(problem3_states), len(problem3_states)), dtype=int)

# Count transitions
for from_state, to_state in data:
    i = state_to_idx[from_state]
    j = state_to_idx[to_state]
    counts[i, j] += 1

# Convert counts to probabilities (row-stochastic matrix)
problem3_transition_matrix = counts / counts.sum(axis=1, keepdims=True)

print("Transition Matrix (MLE):")
print(problem3_transition_matrix)

```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[16], line 23
         20 counts = np.zeros((len(problem3_states), len(problem3_states)), dtype=int)
         22 # Count transitions
    ---> 23 for from_state, to_state in data:
         24     i = state_to_idx[from_state]
         25     j = state_to_idx[to_state]
    

    ValueError: too many values to unpack (expected 2)



```python
import numpy as np

P = np.array([
    [0.0, 1, 0.0, 0.0, 0.0],
    [0.0, 0.3333, 0.33333, 0.3333,0.00],
    [0.0, 0.0, 0.0, 1.0,0.00],
    [0.0, 0.00, 0.00, 0.00, 1],
    [0.50, 0.00, 0.0,0.0,0.50]
])

# Compute 4-step transition matrix
P4 = np.linalg.matrix_power(P, 4)


problem3_prob_dispatch_after_4_from_dock = float(P4[0, 4])

print(P4)
print(problem3_prob_dispatch_after_4_from_dock)
```

    [[0.16665    0.03702593 0.03702926 0.14812482 0.61106889]
     [0.30553444 0.17899074 0.01234185 0.04937    0.45365926]
     [0.25       0.5        0.         0.         0.25      ]
     [0.125      0.41665    0.166665   0.16665    0.125     ]
     [0.0625     0.26386944 0.13888194 0.30553444 0.22915   ]]
    0.61106889
    


```python
# Part 2: 2 points

# Expected: a single float.
# It should be the probability that a chain starting in Dock is in Dispatch after exactly 4 steps.
# Use the transition matrix from Part 1 and the required state order.
problem3_prob_dispatch_after_4_from_dock = 0.125
```


```python
import numpy as np

problem3_states = ["Dock", "Sort", "Storage", "Packing", "Dispatch"]  # Required state order
problem3_state_index = {state: i for i, state in enumerate(problem3_states)}

# Example transition probability matrix (rows sum to 1)
P = np.array([
    [0.0, 1, 0.0, 0.0, 0.0],
    [0.0, 0.3333, 0.33333, 0.3333,0.00],
    [0.0, 0.0, 0.0, 1.0,0.00],
    [0.0, 0.00, 0.00, 0.00, 1],
    [0.50, 0.00, 0.0,0.0,0.50] 
], dtype=float)

problem3_transition_matrix = P # Teacher edit: lets try to use this as answer and see if the rest of the code runs


# --- Parameters ---
n_chains = 20000
n_steps = 8
start_state = "Dock"

# --- Random generator ---
rng = np.random.default_rng(20260616)

# --- Simulation ---
current_states = np.full(n_chains, state_index[start_state], dtype=int)

for _ in range(n_steps):
   
    probs = P[current_states]
    cumulative_probs = np.cumsum(probs, axis=1)
    random_values = rng.random(n_chains).reshape(-1, 1)
    next_states = (random_values < cumulative_probs).argmax(axis=1)
    current_states = next_states

# --- Empirical distribution ---
counts = np.bincount(current_states, minlength=len(states))
empirical_distribution = counts / n_chains

# --- Output ---
print("Empirical distribution after 8 steps:")
print(empirical_distribution)  # Shape (5,), sums to 1

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[19], line 27
         24 rng = np.random.default_rng(20260616)
         26 # --- Simulation ---
    ---> 27 current_states = np.full(n_chains, state_index[start_state], dtype=int)
         29 for _ in range(n_steps):
         31     probs = P[current_states]
    

    NameError: name 'state_index' is not defined



```python
# Part 3: 2 points

# Expected: a NumPy array with shape (5,).
# Simulate 20000 chains for 8 steps from Dock using np.random.default_rng(20260616).
# Store the empirical distribution after 8 steps in the required state order.
# The entries should be nonnegative probabilities that sum to 1.
problem3_simulated_distribution_after_8 = [0.1047,  0.25265, 0.1152,  0.23925, 0.2882 ]
```


```python
import numpy as np

P = np.array([
    [0.0, 1, 0.0, 0.0, 0.0],
    [0.0, 0.3333, 0.33333, 0.3333,0.00],
    [0.0, 0.0, 0.0, 1.0,0.00],
    [0.0, 0.00, 0.00, 0.00, 1],
    [0.50, 0.00, 0.0,0.0,0.50]
])

states = ["Dock", "Sort", "Storage", "Packing", "Dispatch"]
n = P.shape[0]

# -------------------------------
# Check irreducibility
# -------------------------------

# Convert P into a graph: True if transition probability > 0
B = P > 0

# Reachability matrix
# reachable[i, j] means state i can reach state j
reachable = np.eye(n, dtype=bool)

power = B.copy()

for k in range(1, n + 1):
    reachable = reachable | power
    power = power @ B

print("Reachability matrix:")
print(reachable.astype(int))

irreducible = reachable.all()

print("Irreducible?", irreducible)

if not irreducible:
    print("The chain is NOT irreducible because state cannot go back to Dock, Sort, Storage, Packing, Dispatch")


# -------------------------------
# Check aperiodicity
# -------------------------------

# If every state has a self-loop, then every state is aperiodic
self_loops = np.diag(P) > 0

print("Self loops:")
for state, loop in zip(states, self_loops):
    print(state, "has self-loop?", loop)

aperiodic = self_loops.all()

print("Aperiodic?", aperiodic)

if aperiodic:
    print("The chain is aperiodic because every state can return to itself in 1 step.")
else:
    print("The chain may be periodic.")
```

    Reachability matrix:
    [[1 1 1 1 1]
     [1 1 1 1 1]
     [1 1 1 1 1]
     [1 1 1 1 1]
     [1 1 1 1 1]]
    Irreducible? True
    Self loops:
    Dock has self-loop? False
    Sort has self-loop? True
    Storage has self-loop? False
    Packing has self-loop? False
    Dispatch has self-loop? True
    Aperiodic? False
    The chain may be periodic.
    


```python
# Part 4: 2 points

# Expected: two Python booleans, True or False.
# problem3_is_irreducible should say whether every state can reach every other state.
# problem3_is_aperiodic should say whether the chain is aperiodic.
problem3_is_irreducible = True
problem3_is_aperiodic = False
```


```python
A = P.T - np.eye(5)
A[-1] = np.ones(5)

b = np.array([0, 0, 0, 0, 1])

problem1_stationary = np.linalg.solve(A, b)
problem1_stationary
```




    array([0.1666725 , 0.24999625, 0.08333125, 0.166655  , 0.333345  ])




```python
# this for stationaryy distribution code for my understanding part 5
P = np.array([   
    [0.0, 1, 0.0, 0.0, 0.0],
    [0.0, 0.3333, 0.33333, 0.3333,0.00],
    [0.0, 0.0, 0.0, 1.0,0.00],
    [0.0, 0.00, 0.00, 0.00, 1],
    [0.50, 0.00, 0.0,0.0,0.50]
])

pi = np.array([0.50, 0.00, 0.0,0.0,0.50])

print(pi @ P)
```

    [0.25 0.5  0.   0.   0.25]
    


```python
# Part 5: 2 points

# Expected: a NumPy array with shape (5,).
# It should be a stationary probability distribution in the required state order.
# The entries should be nonnegative, sum to 1, and satisfy pi @ P = pi.
problem3_stationary_distribution = [0.1666725 , 0.24999625, 0.08333125, 0.166655  , 0.333345  ]
```


## Free text answer for Part 5

Write your interpretation below this line.



```python
# Part 5

P = np.array([
    [0.0, 1, 0.0, 0.0, 0.0],
    [0.0, 0.3333, 0.33333, 0.3333,0.00],
    [0.0, 0.0, 0.0, 1.0,0.00],
    [0.0, 0.00, 0.00, 0.00, 1],
    [0.50, 0.00, 0.0,0.0,0.50]
])

# Target is Dispatch = 4
# Non-target states are Dock, Sort, Storage, Packing, Dispatch
non_target = [0, 1, 2, 3]

Q = P[np.ix_(non_target, non_target)]

h = np.linalg.solve(
    np.eye(4) - Q,
    np.ones(4)
)

# h[0] is expected time starting from Downtown
problem3_expected_steps_to_dispatch_from_dock = h[0]

problem3_expected_steps_to_dispatch_from_dock
```




    np.float64(3.9997900104994746)




```python
# Part 6: 3 points

# Expected: a single float.
# It should be the expected number of steps until Dispatch is hit for the first time,
# starting from Dock. An exact computation gives full credit; a sufficiently accurate
# simulation estimate can receive partial credit.
problem3_expected_steps_to_dispatch_from_dock =3.9997
```

---
#### Local Test for Exam vB, PROBLEM 3
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.


```python

# Optional local format checks for Problem 3. These checks do not prove correctness.
import numpy as np

try:
    assert problem3_transition_matrix.shape == (5, 5)
    assert np.allclose(np.sum(problem3_transition_matrix, axis=1), 1, atol=2e-4)
    assert problem3_simulated_distribution_after_8.shape == (5,)
    assert np.all(problem3_simulated_distribution_after_8 >= -1e-12)
    assert abs(np.sum(problem3_simulated_distribution_after_8) - 1) < 2e-4
    assert problem3_stationary_distribution.shape == (5,)
    assert np.all(problem3_stationary_distribution >= -1e-12)
    assert abs(np.sum(problem3_stationary_distribution) - 1) < 2e-4
    print("Problem 3 format checks passed.")
except Exception as error:
    print("Problem 3 format check failed:", error)

```

    Problem 3 format check failed: 'list' object has no attribute 'shape'
    


```python

```

    Beginning tests for problem 3
    
    ---------------------------------
    Beginning test for part 1
    ---------------------------------
    
    -----Beginning test------
    problem3_states is correct
    -----Ending test---------
    
    -----Beginning test------
    problem3_transition_matrix has the right shape and row sums
    -----Ending test---------
    
    -----Beginning test------
    
    problem3_transition_matrix is not the maximum-likelihood estimate from the transition counts
    You got 1.75 points deduction 
    -----Ending test---------
    
    ---------------------------------
    Beginning test for part 2
    ---------------------------------
    
    -----Beginning test------
    
    problem3_prob_dispatch_after_4_from_dock is incorrect
    You got 2.00 points deduction 
    -----Ending test---------
    
    ---------------------------------
    Beginning test for part 3
    ---------------------------------
    
    -----Beginning test------
    
    problem3_simulated_distribution_after_8 should be a probability vector of length 5 in the required state order
    You got 0.50 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    
    The simulated distribution after 8 steps is not close to the reference 8-step distribution
    You got 1.50 points deduction 
    -----Ending test---------
    
    ---------------------------------
    Beginning test for part 4
    ---------------------------------
    
    -----Beginning test------
    problem3_is_irreducible is correct
    -----Ending test---------
    
    -----Beginning test------
    
    problem3_is_aperiodic is incorrect
    You got 1.00 points deduction 
    -----Ending test---------
    
    ---------------------------------
    Beginning test for part 5
    ---------------------------------
    
    -----Beginning test------
    
    problem3_stationary_distribution should be a probability vector of length 5 in the required state order
    You got 0.50 points deduction 
    -----Ending test---------
    
    -----Beginning test------
    
    problem3_stationary_distribution is incorrect
    You got 0.50 points deduction 
    -----Ending test---------
    
    Manual points: -1
    Part 5 free-text interpretation requires manual review by the instructor.
    ---------------------------------
    Beginning test for part 6
    ---------------------------------
    
    -----Beginning test------
    problem3_expected_steps_to_dispatch_from_dock is not close enough; absolute error = 1.9285
    You got 3.00 points deduction 
    -----Ending test---------
    
    
    All tests complete, you got = 2.25 points
    The number of points you have scored for this problem is 2.25 out of 14
     
     
     
    The number of points you have scored in total for this entire set of Problems is 4.250000000000003 out of 40
    

The number of points you have scored in total for this entire set of Problems is 4.250000000000003 out of 40.
