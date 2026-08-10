# Assignment 4 for Course 1MS041
Make sure you pass the `# ... Test` cells and
 submit your solution notebook in the corresponding assignment on the course website. You can submit multiple times before the deadline and your highest score will be used.

---
## Assignment 4, PROBLEM 1
Maximum Points = 24


    This time the assignment only consists of one problem, but we will do a more comprehensive analysis instead.

Consider the dataset `Corona_NLP_train.csv` that you can get from the course website [git](https://github.com/datascience-intro/1MS041-2024/blob/main/notebooks/data/Corona_NLP_train.csv). The data is "Coronavirus tweets NLP - Text Classification" that can be found on [kaggle](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification). The data has several columns, but we will only be working with `OriginalTweet`and `Sentiment`.

1. [3p] Load the data and filter out those tweets that have `Sentiment`=`Neutral`. Let $X$ represent the `OriginalTweet` and let 
    $$
        Y = 
        \begin{cases}
        1 & \text{if sentiment is towards positive}
        \\
        0 & \text{if sentiment is towards negative}.
        \end{cases}
    $$
    Put the resulting arrays into the variables $X$ and $Y$. Split the data into three parts, train/test/validation where train is 60% of the data, test is 15% and validation is 25% of the data. Do not do this randomly, this is to make sure that we all did the same splits (we are in this case assuming the data is IID as presented in the dataset). That is [train,test,validation] is the splitting layout.

2. [4p] There are many ways to solve this classification problem. The first main issue to resolve is to convert the $X$ variable to something that you can feed into a machine learning model. For instance, you can first use [`CountVectorizer`](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) as the first step. The step that comes after should be a `LogisticRegression` model, but for this to work you need to put together the `CountVectorizer` and the `LogisticRegression` model into a [`Pipeline`](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline). Fill in the variable `model` such that it accepts the raw text as input and outputs a number $0$ or $1$, make sure that `model.predict_proba` works for this. **Hint: You might need to play with the parameters of LogisticRegression to get convergence, make sure that it doesn't take too long or the autograder might kill your code**
3. [3p] Use your trained model and calculate the precision and recall on both classes. Fill in the corresponding variables with the answer.
4. [3p] Let us now define a cost function
    * A positive tweet that is classified as negative will have a cost of 1
    * A negative tweet that is classified as positive will have a cost of 5
    * Correct classifications cost 0
    
    complete filling the function `cost` to compute the cost of a prediction model under a certain prediction threshold (recall our precision recall lecture and the `predict_proba` function from trained models). 

5. [4p] Now, we wish to select the threshold of our classifier that minimizes the cost, fill in the selected threshold value in value `optimal_threshold`.
6. [4p] With your newly computed threshold value, compute the cost of putting this model in production by computing the cost using the validation data. Also provide a confidence interval of the cost using Hoeffdings inequality with a 99% confidence.
7. [3p] Let $t$ be the threshold you found and $f$ the model you fitted (one of the outputs of `predict_proba`), if we define the random variable
    $$
        C = (1-1_{f(X)\geq t})Y+5(1-Y)1_{f(X) \geq t}
    $$
    then $C$ denotes the cost of a randomly chosen tweet. In the previous step we estimated $\mathbb{E}[C]$ using the empirical mean. However, since the threshold is chosen to minimize cost it is likely that $C=0$ or $C=1$ than $C=5$ as such it will have a low variance. Compute the empirical variance of $C$ on the validation set. What would be the confidence interval if we used Bennett's inequality instead of Hoeffding in point 6 but with the computed empirical variance as our guess for the variance?


```python

# Part 1

# Load the data from the file specified in the problem definition and make sure that it is loaded using
# the search path `data/Corona_NLP_train.csv`. This is to make sure the autograder and your computer have the same
# file path and can load the data correctly.

# Contrary to how many other problems are structured, this problem actually requires you to
# have X on the shape (n_samples, ) that is a 1-dimensional array. Otherwise it will cause a bunch
# of errors in the autograder or also in for instance CountVectorizer.

# Make sure that all your data is numpy arrays and not pandas dataframes or series.
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv("data/Corona_NLP_train.csv", encoding="latin1")

# Remove Neutral sentiment
data = data[data["Sentiment"] != "Neutral"]

# X = tweets (1D numpy array)
X = data["OriginalTweet"].astype(str).values

# Y = sentiment encoded
Y = np.where(data["Sentiment"].isin(["Positive", "Extremely Positive"]), 1, 0)

# Deterministic splits (NO SHUFFLE)
n = len(X)
n_train = int(0.60 * n)
n_test = int(0.15 * n)
n_valid = n - n_train - n_test

X_train = X[:n_train]
Y_train = Y[:n_train]

X_test = X[n_train:n_train+n_test]
Y_test = Y[n_train:n_train+n_test]

X_valid = X[n_train+n_test:]
Y_valid = Y[n_train+n_test:]

```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Cell In[1], line 16
         13 import numpy as np
         15 # Load data
    ---> 16 data = pd.read_csv("data/Corona_NLP_train.csv", encoding="latin1")
         18 # Remove Neutral sentiment
         19 data = data[data["Sentiment"] != "Neutral"]
    

    File c:\Users\hp\anaconda3\Lib\site-packages\pandas\io\parsers\readers.py:1026, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
       1013 kwds_defaults = _refine_defaults_read(
       1014     dialect,
       1015     delimiter,
       (...)
       1022     dtype_backend=dtype_backend,
       1023 )
       1024 kwds.update(kwds_defaults)
    -> 1026 return _read(filepath_or_buffer, kwds)
    

    File c:\Users\hp\anaconda3\Lib\site-packages\pandas\io\parsers\readers.py:620, in _read(filepath_or_buffer, kwds)
        617 _validate_names(kwds.get("names", None))
        619 # Create the parser.
    --> 620 parser = TextFileReader(filepath_or_buffer, **kwds)
        622 if chunksize or iterator:
        623     return parser
    

    File c:\Users\hp\anaconda3\Lib\site-packages\pandas\io\parsers\readers.py:1620, in TextFileReader.__init__(self, f, engine, **kwds)
       1617     self.options["has_index_names"] = kwds["has_index_names"]
       1619 self.handles: IOHandles | None = None
    -> 1620 self._engine = self._make_engine(f, self.engine)
    

    File c:\Users\hp\anaconda3\Lib\site-packages\pandas\io\parsers\readers.py:1880, in TextFileReader._make_engine(self, f, engine)
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
    

    File c:\Users\hp\anaconda3\Lib\site-packages\pandas\io\common.py:873, in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
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
    

    FileNotFoundError: [Errno 2] No such file or directory: 'data/Corona_NLP_train.csv'



```python

# Part 2

# Train a machine learning model or pipeline that can take the raw strings from X and predict Y=0,1 depending on the
# sentiment of the tweet. Store the trained model in the variable `model`.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("vect", CountVectorizer(max_features=20000)),
    ("clf", LogisticRegression(max_iter=200))
])

model.fit(X_train, Y_train)

```


```python

# Part 3

# Evaluate the model on the test set and calculate precision, and recall on both classes. Store the results in the
# variables `precision_0`, `precision_1`, `recall_0`, `recall_1`.

from sklearn.metrics import precision_score, recall_score

pred_test = model.predict(X_test)

precision_0 = precision_score(Y_test, pred_test, pos_label=0)
precision_1 = precision_score(Y_test, pred_test, pos_label=1)

recall_0 = recall_score(Y_test, pred_test, pos_label=0)
recall_1 = recall_score(Y_test, pred_test, pos_label=1)

```


```python

# Part 4

def cost(model,threshold,X,Y):
    # Hint, make sure that the model has a predict_proba method
    # think about how the decision is made based on the probabilities
    # and how the threshold can be used to make the decision.
    # For reference take a look at the lecture notes "Bayes classifier"
    # which contains how the decision is made based on the probabilities when the threshold is 0.5.
    
    # Fill in what is missing to compute the cost and return it
    # Note that we are interested in average cost
    
 import numpy as np

def cost(model, threshold, X, Y):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    C = np.zeros_like(Y, dtype=float)
    C[(Y==1) & (preds==0)] = 1.0
    C[(Y==0) & (preds==1)] = 5.0

    return C.mean()
```


```python

# Part 5

# Find the optimal threshold for the model on the test set. Store the threshold in the variable `optimal_threshold`
# and the cost at the optimal threshold in the variable `cost_at_optimal_threshold` evaluated on the test set.
thresholds = np.linspace(0.01, 0.99, 99)

best_cost = float("inf")
best_t = None

for t in thresholds:
    c = cost(model, t, X_valid, Y_valid)
    if c < best_cost:
        best_cost = c
        best_t = t

optimal_threshold = best_t
cost_at_optimal_threshold = best_cost

```


```python

# Part 6

delta = 0.01
a, b = 0.0, 5.0
n_valid = len(Y_valid)

cost_at_optimal_threshold_valid = cost(model, optimal_threshold, X_valid, Y_valid)

epsilon = (b - a) * np.sqrt(np.log(2/delta) / (2*n_valid))

cost_interval_valid = (
    cost_at_optimal_threshold_valid - epsilon,
    cost_at_optimal_threshold_valid + epsilon
)


assert(type(cost_interval_valid) == tuple)
assert(len(cost_interval_valid) == 2)
```


```python

# Part 7


probs_valid = model.predict_proba(X_valid)[:, 1]
preds_valid = (probs_valid >= optimal_threshold).astype(int)

C = np.zeros_like(Y_valid, dtype=float)
C[(Y_valid==1) & (preds_valid==0)] = 1.0
C[(Y_valid==0) & (preds_valid==1)] = 5.0

mean_C = C.mean()
variance_of_C = C.var(ddof=1)
n_valid = len(C)

z = 2.5758293035489004  # 99% CI

half_width = z * np.sqrt(variance_of_C / n_valid)

interval_of_C = (mean_C - half_width, mean_C + half_width)


assert(type(interval_of_C) == tuple)
assert(len(interval_of_C) == 2)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 4
          1 # Part 7
    ----> 4 probs_valid = model.predict_proba(X_valid)[:, 1]
          5 preds_valid = (probs_valid >= optimal_threshold).astype(int)
          7 C = np.where((Y_valid == 1) & (preds_valid == 0), 1,
          8      np.where((Y_valid == 0) & (preds_valid == 1), 5, 0))
    

    NameError: name 'model' is not defined

