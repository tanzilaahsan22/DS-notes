# Exam 14th of January 2020 — 1MS041 Introduction to Data Science

This notebook contains the full questions, solved code answers, explanations, and runnable outputs where possible.

Some problems need course data files that were **not included in the uploaded PDF**:
- `data/earthquakes.csv` for Problems 3 and 7.

I added complete runnable code for those problems. Put the data file in the expected folder and re-run the relevant cells.


```python
# Enter your anonymous exam id by replacing XXXX in this cell below
# do NOT delete this cell
MyAnonymousExamID = 'XXX'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy import stats
import math
from collections import Counter
```

---
# PROBLEM 1 — SVD by hand [5p]

Consider the matrix

\[
M =
\begin{bmatrix}
1 & 1 \\
0 & 3 \\
3 & 0
\end{bmatrix}
\]

Manually, by hand, produce:

- **[1p]** Rank: `rank`
- **[1p]** Left singular vectors in matrix form
- **[1p]** Singular values in matrix form
- **[1p]** Right singular vectors in matrix form
- **[1p]** Check that \(U D V^T = M\)

Use exact answers.

## Solution explanation

Compute:

\[
M^T M =
\begin{bmatrix}
10 & 1 \\
1 & 10
\end{bmatrix}
\]

The eigenvalues are \(11\) and \(9\), so the singular values are:

\[
\sigma_1 = \sqrt{11}, \quad \sigma_2 = 3
\]

The right singular vectors are:

\[
v_1 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix},
\quad
v_2 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}
\]

The left singular vectors are \(u_i=Mv_i/\sigma_i\):

\[
u_1 = \frac{1}{\sqrt{22}}\begin{bmatrix}2\\3\\3\end{bmatrix},
\quad
u_2 = \frac{1}{\sqrt{2}}\begin{bmatrix}0\\-1\\1\end{bmatrix}
\]


```python
# PROBLEM 1 solved in numpy

M = np.array([[1, 1],
              [0, 3],
              [3, 0]], dtype=float)

rank = np.linalg.matrix_rank(M)

# U is 3x2, D is 2x2, V is 2x2, so U @ D @ V.T = M
U = np.array([[2/np.sqrt(22), 0],
              [3/np.sqrt(22), -1/np.sqrt(2)],
              [3/np.sqrt(22),  1/np.sqrt(2)]], dtype=float)

D = np.array([[np.sqrt(11), 0],
              [0, 3]], dtype=float)

V = np.array([[1/np.sqrt(2),  1/np.sqrt(2)],
              [1/np.sqrt(2), -1/np.sqrt(2)]], dtype=float)

print("rank =", rank)
print("U =")
print(U)
print("D =")
print(D)
print("V =")
print(V)

print("\nCheck U @ D @ V.T:")
print(U @ D @ V.T)

print("\nIs U @ D @ V.T equal to M?", np.allclose(U @ D @ V.T, M))
```

**Exact Sage-style answer:**

```python
rank = 2
U = matrix([[2/sqrt(22), 0],
            [3/sqrt(22), -1/sqrt(2)],
            [3/sqrt(22),  1/sqrt(2)]])
D = matrix([[sqrt(11), 0],
            [0, 3]])
V = matrix([[1/sqrt(2),  1/sqrt(2)],
            [1/sqrt(2), -1/sqrt(2)]])
```

Then `U*D*V.transpose() == M`.

---
# PROBLEM 2 — Wald test for rescaled Rademacher [5p]

Consider IID rescaled Rademacher random variables:

\[
f(x;\theta)=
\begin{cases}
\theta, & x=+10 \\
1-\theta, & x=-10 \\
0, & \text{otherwise}
\end{cases}
\]

Perform a Wald test of size \(lpha=0.05\) for:

\[
H_0: \theta^*=\theta_0
\quad \text{versus} \quad
H_1: \theta^* \ne \theta_0
\]

with \(	heta_0=0.5\).

## Solution explanation

This is just Bernoulli data where success means `+10`.

So:

\[
\hat{\theta} = \frac{\#(+10)}{n}
\]

The estimated standard error is:

\[
SE(\hat{\theta}) = \sqrt{\frac{\hat{\theta}(1-\hat{\theta})}{n}}
\]

The Wald statistic is:

\[
W = \frac{\hat{\theta}-\theta_0}{SE(\hat{\theta})}
\]

Reject if \(|W|>1.96\), approximately \(|W|>2\).


```python
dataSamples2 = np.array([-10,+10,+10,-10,-10,-10,+10,+10,-10,-10,-10,+10,+10,-10,+10,+10,+10])

print("number of -10s:", sum(dataSamples2 == -10))
print("number of +10s:", sum(dataSamples2 == +10))

n = len(dataSamples2)

thetaHat = np.mean(dataSamples2 == +10)
print("mle thetaHat =", thetaHat)

NullTheta = 0.5
print("Null value of theta under H0 =", NullTheta)

seTheta = np.sqrt(thetaHat * (1 - thetaHat) / n)
print("estimated standard error =", seTheta)

W = (thetaHat - NullTheta) / seTheta
print("Wald statistic =", W)

rejectNull1 = abs(W) > 2.0
if rejectNull1:
    print("we reject the null hypothesis that theta_0=0.5")
else:
    print("we fail to reject the null hypothesis that theta_0=0.5")
```

---
# PROBLEM 3 — Earthquake depth and magnitude regression [5p]

The earthquake data is analyzed further, specifically depth and magnitude.

Tasks:

1. Make a residual plot for the linear fit and briefly discuss the scatter of residuals. Explain what values farthest from the x-axis mean.
2. Conduct a Wald test with alpha at 5% of the null hypothesis:

\[
H_0: \beta_1 = 0
\]

Set `RejectNullHypothesisForProblem4 = True` if you reject and `False` otherwise.

This problem needs `data/earthquakes.csv`, which was not included in the uploaded PDF.


```python
# PROBLEM 3 solved code.
# This cell runs if data/earthquakes.csv is available.

def getLonLatMagDepTimes(NZEQCsvFileName):
    from dateutil.parser import parse
    import time
    with open(NZEQCsvFileName) as f:
        reader = f.read()
    dataList = reader.split('\n')
    myDataAccumulatorList = []
    for data in dataList[1:-1]:
        dataRow = data.split(',')
        try:
            myTimeString = dataRow[2]  # origintime
            myDataString = [dataRow[4], dataRow[5], dataRow[6], dataRow[7]]  # lon, lat, mag, depth
            myTypedTime = time.mktime(parse(myTimeString).timetuple())
            myFloatData = [float(x) for x in myDataString]
            myFloatData.append(myTypedTime)
            myDataAccumulatorList.append(myFloatData)
        except Exception:
            pass
    return myDataAccumulatorList

try:
    myProcessedList = getLonLatMagDepTimes('data/earthquakes.csv')

    eqData = np.array(myProcessedList)[:, [3, 2]]
    eqDepth = eqData[:, 0]
    eqMagnitude = eqData[:, 1]

    # Design matrix with intercept and depth
    X_design = np.column_stack([np.ones_like(eqDepth), eqDepth])

    # Least squares coefficients
    b, residual_sum, rnk, s = lstsq(X_design, eqMagnitude)

    fitted = X_design @ b
    residuals = eqMagnitude - fitted

    # Residual plot
    plt.figure(figsize=(7,4))
    plt.scatter(eqDepth, residuals, s=10)
    plt.axhline(0, linestyle='--')
    plt.xlabel('Earthquake Depth')
    plt.ylabel('Residual = observed magnitude - fitted magnitude')
    plt.title('Residual plot for magnitude ~ depth')
    plt.grid(alpha=0.25)
    plt.show()

    print("beta_hat =", b)

    # Wald test for beta_1 = 0
    n = len(eqMagnitude)
    p = X_design.shape[1]

    sigma2_hat = np.sum(residuals**2) / (n - p)
    cov_beta = sigma2_hat * np.linalg.inv(X_design.T @ X_design)

    se_beta1 = np.sqrt(cov_beta[1, 1])
    W_beta1 = b[1] / se_beta1
    p_value_beta1 = 2 * (1 - stats.norm.cdf(abs(W_beta1)))

    RejectNullHypothesisForProblem4 = bool(abs(W_beta1) > 1.96)

    print("beta_1 =", b[1])
    print("SE(beta_1) =", se_beta1)
    print("Wald statistic =", W_beta1)
    print("p-value =", p_value_beta1)
    print("RejectNullHypothesisForProblem4 =", RejectNullHypothesisForProblem4)

except FileNotFoundError:
    print("data/earthquakes.csv not found.")
    print("Put earthquakes.csv inside a folder named data, then rerun this cell.")
    RejectNullHypothesisForProblem4 = None
```

## Residual interpretation

The residual is:

\[
e_i = y_i - \hat{y}_i
\]

A residual far **above** the x-axis means the earthquake magnitude was much larger than the model predicted for that depth.

A residual far **below** the x-axis means the earthquake magnitude was much smaller than the model predicted for that depth.

If residuals show a pattern rather than random scatter around zero, the linear model may be a poor fit.

---
# PROBLEM 4 — Markov chain from Pride and Prejudice text [5p]

Tasks:

1. Take `prideAndPrejudiceFirstChapter` and split by `' '` into a list of words called `words`.
2. Treat words as states in a Markov chain.
3. Estimate transition matrix \(P\) by maximum likelihood:

\[
\hat p_{i,j} = \frac{n_{i,j}}{\sum_k n_{i,k}}
\]

4. The order of indices should match `unique_words`.


```python
def makeFreqDict(myDataList):
    freqDict = {}
    for res in myDataList:
        if res in freqDict:
            freqDict[res] += 1
        else:
            freqDict[res] = 1
    return freqDict

prideAndPrejudiceFirstChapter = '''It is a truth universally acknowledged, that a single man in
possession of a good fortune, must be in want of a wife.
However little known the feelings or views of such a man may be
on his first entering a neighbourhood, this truth is so well
fixed in the minds of the surrounding families, that he is
considered the rightful property of some one or other of their
daughters.
My dear Mr. Bennet, said his lady to him one day, have you
heard that Netherfield Park is let at last?
Mr. Bennet replied that he had not.
But it is, returned she; for Mrs. Long has just been here, and
she told me all about it.
Mr. Bennet made no answer.
Do you not want to know who has taken it? cried his wife
impatiently.
You want to tell me, and I have no objection to hearing it.
This was invitation enough.
Why, my dear, you must know, Mrs. Long says that Netherfield is
taken by a young man of large fortune from the north of England;
that he came down on Monday in a chaise and four to see the
place, and was so much delighted with it, that he agreed with Mr.
Morris immediately; that he is to take possession before
Michaelmas, and some of his servants are to be in the house by
the end of next week.
What is his name?
Bingley.
Is he married or single?
Oh! Single, my dear, to be sure! A single man of large fortune;
four or five thousand a year. What a fine thing for our girls!
How so? How can it affect them?
My dear Mr. Bennet, replied his wife, how can you be so
tiresome! You must know that I am thinking of his marrying one of
them.
Is that his design in settling here?
Design! Nonsense, how can you talk so! But it is very likely
that he may fall in love with one of them, and therefore you
must visit him as soon as he comes.
I see no occasion for that. You and the girls may go, or you may
send them by themselves, which perhaps will be still better, for
as you are as handsome as any of them, Mr. Bingley may like you
the best of the party.
My dear, you flatter me. I certainly have had my share of
beauty, but I do not pretend to be anything extraordinary now.
When a woman has five grown-up daughters, she ought to give over
thinking of her own beauty.
In such cases, a woman has not often much beauty to think of.
But, my dear, you must indeed go and see Mr. Bingley when he
comes into the neighbourhood.
It is more than I engage for, I assure you.
But consider your daughters. Only think what an establishment it
would be for one of them. Sir William and Lady Lucas are
determined to go, merely on that account, for in general, you
know, they visit no newcomers. Indeed you must go, for it will be
impossible for us to visit him if you do not.
You are over-scrupulous, surely. I dare say Mr. Bingley will be
very glad to see you; and I will send a few lines by you to
assure him of my hearty consent to his marrying whichever he
chooses of the girls; though I must throw in a good word for my
little Lizzy.
I desire you will do no such thing. Lizzy is not a bit better
than the others; and I am sure she is not half so handsome as
Jane, nor half so good-humoured as Lydia. But you are always
giving her the preference.
They have none of them much to recommend them, replied he;
they are all silly and ignorant like other girls; but Lizzy has
something more of quickness than her sisters.
Mr. Bennet, how can you abuse your own children in such a way?
You take delight in vexing me. You have no compassion for my poor
nerves.
You mistake me, my dear. I have a high respect for your nerves.
They are my old friends. I have heard you mention them with
consideration these last twenty years at least.
Ah, you do not know what I suffer.
But I hope you will get over it, and live to see many young men
of four thousand a year come into the neighbourhood.
It will be no use to us, if twenty such should come, since you
will not visit them.
Depend upon it, my dear, that when there are twenty, I will
visit them all.
Mr. Bennet was so odd a mixture of quick parts, sarcastic humour,
reserve, and caprice, that the experience of three-and-twenty
years had been insufficient to make his wife understand his
character. Her mind was less difficult to develop. She was a
woman of mean understanding, little information, and uncertain
temper. When she was discontented, she fancied herself nervous.
The business of her life was to get her daughters married; its
solace was visiting and news.'''.lower()

import re
subs = "_;.,”“?!"
for sub in subs:
    prideAndPrejudiceFirstChapter = prideAndPrejudiceFirstChapter.replace(sub, ' ')
prideAndPrejudiceFirstChapter = re.sub(r'\s+', ' ', prideAndPrejudiceFirstChapter).strip()

# Part 1
words = prideAndPrejudiceFirstChapter.split(' ')
unique_words = sorted(set(words))
n_words = len(unique_words)

print("First 10 words:", words[:10])
print("Number of words:", len(words))
print("Number of unique words:", n_words)

# Part 2
transitions = list(zip(words[:-1], words[1:]))
transition_counts = makeFreqDict(transitions)

indexToWord = {i: word for i, word in enumerate(unique_words)}
wordToIndex = {word: i for i, word in indexToWord.items()}

print("First 10 transitions:", transitions[:10])
print("Count of ('it','is'):", transition_counts.get(('it', 'is'), 0))

# Part 3
transition_matrix = np.zeros((n_words, n_words), dtype=float)

for (w1, w2), count in transition_counts.items():
    i = wordToIndex[w1]
    j = wordToIndex[w2]
    transition_matrix[i, j] = count

row_sums = transition_matrix.sum(axis=1, keepdims=True)
for i in range(n_words):
    if row_sums[i, 0] > 0:
        transition_matrix[i, :] = transition_matrix[i, :] / row_sums[i, 0]
    else:
        transition_matrix[i, i] = 1.0

print("transition_matrix shape:", transition_matrix.shape)
print("First 10 row sums:", transition_matrix.sum(axis=1)[:10])
print("p_it_is =", transition_matrix[wordToIndex['it'], wordToIndex['is']])
```


```python
# Local test: generate Markov-chain text
np.random.seed(1)

start = np.zeros(shape=(n_words, 1))
start[0, 0] = 1
current_pos = start

generated = []
for i in range(50):
    random_word_index = np.random.choice(range(n_words), p=current_pos.reshape(-1))
    generated.append(indexToWord[random_word_index])

    current_pos = np.zeros_like(start)
    current_pos[random_word_index] = 1
    current_pos = (current_pos.T @ transition_matrix).T

print("Generated text:")
print(" ".join(generated))
```

---
# PROBLEM 5 — High-dimensional geometry [5p]

1. Draw a uniform random point \(X\) on the surface of the unit sphere in \(\mathbb R^d\). What is the variance of \(X_1\)?
2. How large must \(\epsilon\) be for 99% of the volume of a \(d\)-dimensional unit-radius ball to lie in the shell of \(\epsilon\)-thickness at the surface of the ball?
3. The volume of the unit ball is:

\[
V(d)=\frac{2\pi^{d/2}}{d\Gamma(d/2)}
=\frac{\pi^{d/2}}{(d/2)!}
\]

What function of \(d\) would the radius need to be for a ball of radius \(r\) to have approximately constant volume as a function of \(d\)? Use Stirling's formula.

## Solution explanation

### Part 1

By symmetry:

\[
X_1^2 + X_2^2 + \dots + X_d^2 = 1
\]

and all coordinates have the same second moment, so:

\[
dE[X_1^2]=1
\]

Also \(E[X_1]=0\), hence:

\[
Var(X_1)=E[X_1^2]=\frac{1}{d}
\]

### Part 2

Volume inside radius \(1-\epsilon\) is \((1-\epsilon)^d\) of the unit ball.

Shell volume fraction is:

\[
1-(1-\epsilon)^d
\]

Set this equal to \(0.99\):

\[
1-(1-\epsilon)^d=0.99
\]

\[
\epsilon = 1 - 0.01^{1/d}
\]

### Part 3

A ball of radius \(r\) has volume \(r^d V(d)\).

Using Stirling, the unit ball volume behaves roughly like:

\[
V(d) pprox \left(\frac{2\pi e}{d}\right)^{d/2}
\]

To keep \(r^d V(d)\) approximately constant:

\[
r^d \left(\frac{2\pi e}{d}\right)^{d/2} pprox C
\]

So the dominant scaling is:

\[
r(d) pprox \sqrt{\frac{d}{2\pi e}}
\]


```python
# PROBLEM 5 answers

# Part 1
# Sage exact expression would be:
# d = var('d')
# variance_x1_problem7 = 1/d
def variance_x1_problem7(d):
    return 1/d

# Part 2
# epsilon = 1 - 0.01^(1/d)
def epsilon_shell_99(d):
    return 1 - 0.01**(1/d)

# Part 3
# r approx sqrt(d/(2*pi*e))
def radius_for_constant_volume(d):
    return np.sqrt(d / (2 * np.pi * np.e))

for dim in [10, 100, 1000]:
    print("d =", dim)
    print("Var(X1) =", variance_x1_problem7(dim))
    print("epsilon for 99% shell =", epsilon_shell_99(dim))
    print("radius scaling approx =", radius_for_constant_volume(dim))
    print()

print("Sage-style exact answers:")
print("variance_x1_problem7 = 1/d")
print("epsilon = 1 - (1/100)^(1/d)")
print("r = sqrt(d/(2*pi*e))")
```

---
# PROBLEM 6 — Perceptron [5p]

Consider data `X` and labels `y`. `X` denotes 20 points in \(\mathbb R^2\), and `y` gives labels.

Tasks:

1. Implement the function `perceptron`.
2. Compute a vector \(\hat w\) with shape `(3,1)` such that:

\[
(\hat w \cdot \hat x_i)l_i > 0, \quad orall i=1,\dots,20
\]

3. Compute \(r\), then use the perceptron convergence theorem to give an upper bound to the number of iterations needed.


```python
X = np.array([
    [0.14774693918368506,0.8537253157278155],
    [-0.1755517430286779,0.8979710703337818],
    [0.5227216475286975,0.7448281947022451],
    [-0.5071170511153492,0.8002027400836075],
    [-0.39436968212400453,1.0177689414422981],
    [-0.3983065780966649,1.0443663197782966],
    [-0.08652771617599643,0.48036820824519255],
    [0.15352541170101042,0.6820807981911706],
    [-0.3303348532791869,1.120673883903539],
    [-0.2656220857139274,0.8526638282828739],
    [0.7259603693529442,0.25428467532034965],
    [0.4577253912481767,-0.2358809079980879],
    [0.9722462145222105,0.13128550836973255],
    [0.4089349951770505,-0.09503914544452634],
    [0.9718156747909192,0.3524307824261209],
    [1.2009353774940565,-0.25004126389987974],
    [1.271791635779178,-0.07571928320750206],
    [0.36784476124502913,-0.23743021661715671],
    [0.8918396050420891,-0.1029336332277948],
    [0.4501578013678095,-0.13188266835015783]
]) + np.array([10,0]).reshape(1,-1)

y = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
              -1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0])

def augment_X(X_in):
    # Add intercept coordinate 1.
    return np.column_stack([np.ones(X_in.shape[0]), X_in])

def perceptron(X_in, labels, max_iter=1000):
    '''Runs the perceptron algorithm on X_in and labels.'''
    X_aug = augment_X(X_in)
    labels = labels.reshape(-1)

    # w has shape (3,)
    w = np.zeros(X_aug.shape[1])

    updates = 0
    for _ in range(max_iter):
        mistake_found = False

        for xi, yi in zip(X_aug, labels):
            if yi * np.dot(w, xi) <= 0:
                w = w + yi * xi
                updates += 1
                mistake_found = True

                if updates >= max_iter:
                    break

        if not mistake_found or updates >= max_iter:
            break

    return w.reshape(-1, 1)

hat_w = perceptron(X, y, max_iter=1000)

print("hat_w shape:", hat_w.shape)
print("hat_w:")
print(hat_w)

# Verify classification
X_aug = augment_X(X)
margins_signed = y * (X_aug @ hat_w).reshape(-1)
print("Minimum signed margin y_i * (w dot x_i):", margins_signed.min())
print("All correctly classified?", np.all(margins_signed > 0))

# Plot data and decision boundary
plt.figure(figsize=(7,4))
plt.scatter(X[y==1,0], X[y==1,1], label="+1")
plt.scatter(X[y==-1,0], X[y==-1,1], label="-1")
xx = np.linspace(X[:,0].min()-0.1, X[:,0].max()+0.1, 100)
# w0 + w1*x + w2*y = 0 => y = -(w0+w1*x)/w2
if abs(hat_w[2,0]) > 1e-12:
    yy = -(hat_w[0,0] + hat_w[1,0]*xx) / hat_w[2,0]
    plt.plot(xx, yy, label="decision boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Perceptron classifier")
plt.show()
```


```python
# Part 2/3: compute margin r and iteration bound.

X_aug = augment_X(X)
w_vec = hat_w.reshape(-1)

# Geometric margin r = min_i y_i * <w,x_i> / ||w||
r = float(np.min(y * (X_aug @ w_vec)) / np.linalg.norm(w_vec))

# R = max_i ||x_i||
R = float(np.max(np.linalg.norm(X_aug, axis=1)))

# Perceptron mistake bound <= (R/r)^2
iteration_bound = int(np.ceil((R / r)**2))

print("r =", r)
print("R =", R)
print("iteration_bound =", iteration_bound)
```

---
# PROBLEM 7 — Bootstrap 99% CI for earthquake inter-event time percentile [5p]

Perform a bootstrap to find the plug-in estimate and 99% CI for the 95th percentile of the inter-earthquake time in minutes.

This problem needs:

```text
data/earthquakes.csv
```

If the file is zipped, unzip it first as the exam says:

```bash
cd data
unzip earthquakes.csv.zip
```


```python
# PROBLEM 7 solved code.
# This cell runs if data/earthquakes.csv is available.

def interQuakeTimes(quakeTimes):
    '''Return inter-earthquake times in seconds from sorted earthquake origin times.'''
    quakeTimes = sorted(quakeTimes)
    return np.diff(quakeTimes)

def makeBootstrappedConfidenceIntervalOfStatisticT(dataset, statT, alpha, B):
    '''
    Build percentile bootstrap CI for statistic statT.
    Returns lower CI, upper CI, bootstrapped statistic values.
    '''
    dataset = np.asarray(dataset)
    n = len(dataset)
    bootstrappedStatisticTs = []

    for b in range(B):
        randIndices = np.random.randint(0, n, size=n)
        bootstrappedDataset = dataset[randIndices]
        bootstrappedStatisticT = statT(bootstrappedDataset)
        bootstrappedStatisticTs.append(bootstrappedStatisticT)

    bootstrappedStatisticTs = np.array(bootstrappedStatisticTs)

    lower = np.percentile(bootstrappedStatisticTs, 100*alpha/2)
    upper = np.percentile(bootstrappedStatisticTs, 100*(1-alpha/2))

    return lower, upper, bootstrappedStatisticTs

try:
    myProcessedList = getLonLatMagDepTimes('data/earthquakes.csv')

    interQuakesSecs = interQuakeTimes([x[4] for x in myProcessedList])
    iQMinutes = np.array(interQuakesSecs) / 60.0

    statT95thPercentile = lambda dataset: np.percentile(dataset, 95)
    alpha = 0.01
    B = 1000

    plugInEstimateOf95thPercentile = statT95thPercentile(iQMinutes)

    lowerCIT95P, upperCIT95P, bootValuesT95P = makeBootstrappedConfidenceIntervalOfStatisticT(
        iQMinutes, statT95thPercentile, alpha, B
    )

    print("The Plug-in Point Estimate of the 95th-Percentile of inter-EQ Times =", plugInEstimateOf95thPercentile)
    print("1-alpha Bootstrapped CI for the 95th-Percentile of inter-EQ Times =", (lowerCIT95P, upperCIT95P))
    print("alpha =", alpha, "bootstrap replicates =", B)

    plt.figure(figsize=(7,4))
    plt.hist(bootValuesT95P, bins=40, density=True)
    plt.axvline(lowerCIT95P, linestyle="--", label="0.5%")
    plt.axvline(upperCIT95P, linestyle="--", label="99.5%")
    plt.xlabel("Bootstrap 95th percentile of inter-EQ time, minutes")
    plt.ylabel("density")
    plt.legend()
    plt.title("Bootstrap distribution")
    plt.show()

except FileNotFoundError:
    print("data/earthquakes.csv not found.")
    print("Put earthquakes.csv inside a folder named data, then rerun this cell.")
```
