# Re-exam 14th of June 2021 — 1MS041 Introduction to Data Science

This notebook contains the full questions, solved code, step-by-step explanations, and outputs where possible.

Some questions depend on the course file `digits.csv`, which was not included in the uploaded PDF.
Put `digits.csv` in the same folder as this notebook before running Problem 5.


```python
MyAnonymousExamID = 'XXX'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```

---
# Problem 1 [8p]

1. Implement a pseudo number generator that produces random numbers from the uniform distribution `[0,1]`.
2. Use that to construct samples from the CDF `F(x)=sin(x)` for `0 < x < pi/2`.
3. Estimate `E[X]` using 1000 samples.
4. Use bootstrap to produce a 95% confidence interval for `E[X]`.

## Problem 1.1 — Uniform pseudo-random number generator

Use a Linear Congruential Generator: `x_next = (a*x + c) mod m`.


```python
def uniform_pseudo_random(n_samples, seed=1):
    m = 2**31
    a = 1103515245
    c = 12345

    x = seed
    out = []

    for _ in range(n_samples):
        x = (a * x + c) % m
        out.append(x / m)

    return out

u_test = uniform_pseudo_random(10, seed=1)
print("First 10 pseudo-random uniform numbers:")
print(u_test)
```

## Problem 1.2 — Sampler from `F(x)=sin(x)`

By inverse transform sampling:

`U = sin(X)`, so `X = arcsin(U)`.


```python
def sampler_problem_1(n_samples, seed=1):
    U = np.array(uniform_pseudo_random(n_samples, seed=seed))
    X = np.arcsin(U)
    return list(X)

samples_test = sampler_problem_1(10, seed=2)
print("First 10 samples from F:")
print(samples_test)

samples_plot = sampler_problem_1(5000, seed=3)
x_grid = np.linspace(0, np.pi/2, 300)
true_density = np.cos(x_grid)

plt.figure(figsize=(7,4))
plt.hist(samples_plot, bins=40, density=True, alpha=0.6, label="Simulated samples")
plt.plot(x_grid, true_density, label="True density f(x)=cos(x)")
plt.xlabel("x")
plt.ylabel("density")
plt.title("Samples from F(x)=sin(x)")
plt.legend()
plt.show()
```

## Problem 1.3 — Estimate `E[X]`

The true mean is `pi/2 - 1`, so the simulation should be close to `0.5708`.


```python
samples = sampler_problem_1(1000, seed=42)
E_X = float(np.mean(samples))
true_E_X = np.pi/2 - 1

print("Estimated E[X] from 1000 samples:", E_X)
print("True E[X] = pi/2 - 1:", true_E_X)
print("Absolute error:", abs(E_X - true_E_X))
```

## Problem 1.4 — Bootstrap confidence interval

Use our own uniform sampler to generate bootstrap indices, then resample with replacement.


```python
def bootstrap_indices(n, B=2000, seed=123):
    U = np.array(uniform_pseudo_random(B * n, seed=seed))
    indices = np.floor(U * n).astype(int)
    indices = np.clip(indices, 0, n - 1)
    return indices.reshape(B, n)

def bootstrap_mean_ci(data, B=2000, alpha=0.05, seed=123):
    data = np.array(data)
    n = len(data)
    idx = bootstrap_indices(n, B=B, seed=seed)

    bootstrap_means = []
    for b in range(B):
        bootstrap_sample = data[idx[b]]
        bootstrap_means.append(np.mean(bootstrap_sample))

    bootstrap_means = np.array(bootstrap_means)
    lower = np.quantile(bootstrap_means, alpha/2)
    upper = np.quantile(bootstrap_means, 1-alpha/2)
    return float(lower), float(upper), bootstrap_means

bootstrap_low, bootstrap_high, bootstrap_means = bootstrap_mean_ci(samples, B=2000, alpha=0.05, seed=10)

print("Bootstrap 95% CI for E[X]:")
print("[%.6f, %.6f]" % (bootstrap_low, bootstrap_high))

plt.figure(figsize=(7,4))
plt.hist(bootstrap_means, bins=40, density=True)
plt.axvline(bootstrap_low, linestyle="--", label="2.5% quantile")
plt.axvline(bootstrap_high, linestyle="--", label="97.5% quantile")
plt.xlabel("Bootstrap mean")
plt.ylabel("density")
plt.title("Bootstrap distribution of sample mean")
plt.legend()
plt.show()
```

---
# Problem 2 [8p]

Let `X_1,...,X_n ~ N(theta,1)`. Define `Y_i=1` if `X_i>0` and `0` otherwise.
Let `psi = P(Y_1=1)`.

Tasks:
- Find MLE of `psi`.
- Find 95% confidence interval.
- Repeat with unknown variance.

## Problem 2.1 — MLE when variance is known

If variance is known and equal to 1:

`theta_hat = mean(X)`

and

`psi_hat = Phi(theta_hat)`.


```python
def mle_psi_problem_2(data):
    data = np.array(data)
    theta_hat = np.mean(data)
    psi_hat = norm.cdf(theta_hat)
    return float(psi_hat)
```

## Problem 2.2 — Apply to given data and compute 95% CI

Known variance CI for theta is `mean +/- 1.96/sqrt(n)`. Transform endpoints using `Phi`.


```python
prob2_samples = [0.88, -0.75, -0.46, -0.13, 0.96, 0.17, 1.24, -1.03, -1.,
                 1.7, 0.34, 1.01, 0.75, 0.58, 0.5, -1.2, -1.45, 1.59, 1.79, -1.32]

mle_psi = mle_psi_problem_2(prob2_samples)

data = np.array(prob2_samples)
n = len(data)
theta_hat = np.mean(data)

z = norm.ppf(0.975)
theta_low = theta_hat - z / np.sqrt(n)
theta_high = theta_hat + z / np.sqrt(n)

low_edge_confidence = float(norm.cdf(theta_low))
high_edge_confidence = float(norm.cdf(theta_high))

print("theta_hat:", theta_hat)
print("mle_psi:", mle_psi)
print("95% CI for psi, known variance:")
print("[%.6f, %.6f]" % (low_edge_confidence, high_edge_confidence))
```

## Problem 2.3 — Unknown variance

Estimate both mean and variance. Then `psi_hat = Phi(theta_hat / sigma_hat)`.
For confidence interval, use bootstrap because the estimator is nonlinear.


```python
def mle_psi_problem_2_uv(data):
    data = np.array(data)
    theta_hat = np.mean(data)
    sigma_hat = np.sqrt(np.mean((data - theta_hat)**2))  # MLE variance uses n
    psi_hat = norm.cdf(theta_hat / sigma_hat)
    return float(psi_hat)

mle_psi_uv = mle_psi_problem_2_uv(prob2_samples)

def bootstrap_psi_unknown_variance(data, B=3000, alpha=0.05, seed=123):
    data = np.array(data)
    n = len(data)
    idx = bootstrap_indices(n, B=B, seed=seed)

    psi_values = []
    for b in range(B):
        boot_sample = data[idx[b]]
        psi_values.append(mle_psi_problem_2_uv(boot_sample))

    psi_values = np.array(psi_values)
    low = float(np.quantile(psi_values, alpha/2))
    high = float(np.quantile(psi_values, 1-alpha/2))
    return low, high, psi_values

low_edge_confidence_uv, high_edge_confidence_uv, psi_boot = bootstrap_psi_unknown_variance(
    prob2_samples, B=3000, alpha=0.05, seed=55
)

print("mle_psi with unknown variance:", mle_psi_uv)
print("Bootstrap 95% CI for psi, unknown variance:")
print("[%.6f, %.6f]" % (low_edge_confidence_uv, high_edge_confidence_uv))
```

---
# Problem 3 [8p]

Gaussian Annulus theorem simulation.

1. For `d=10,20,...,100`, estimate `P(sqrt(d)-1 < ||X|| < sqrt(d)+1)`.
2. Guess constant `c`.
3. For `d=2,20,200,2000`, generate pairs of vectors and compute angles.
4. For `d=100`, find beta such that `P(sqrt(d)-beta < ||X|| < sqrt(d)+beta) approx 0.99`.


```python
rng = np.random.default_rng(123)

dimensions = list(range(10, 101, 10))
probabilities = []

for d in dimensions:
    X = rng.normal(size=(1000, d))
    norms = np.linalg.norm(X, axis=1)

    lower = np.sqrt(d) - 1
    upper = np.sqrt(d) + 1

    prob = np.mean((norms > lower) & (norms < upper))
    probabilities.append(prob)

print("d values:", dimensions)
print("Estimated probabilities:", probabilities)

plt.figure(figsize=(7,4))
plt.plot(dimensions, probabilities, marker="o")
plt.xlabel("dimension d")
plt.ylabel("estimated probability")
plt.title("Gaussian annulus probability")
plt.ylim(0, 1.05)
plt.grid(True)
plt.show()
```


```python
p_avg = float(np.mean(probabilities))
c = float(-np.log(max(1e-12, 1 - p_avg)))

print("Average probability inside shell:", p_avg)
print("Rough simulation-based guess for c:", c)
```


```python
angle_dimensions = [2, 20, 200, 2000]
angle_results = {}

for d in angle_dimensions:
    X = rng.normal(size=(1000, d))
    Y = rng.normal(size=(1000, d))

    dot_products = np.sum(X * Y, axis=1)
    X_norms = np.linalg.norm(X, axis=1)
    Y_norms = np.linalg.norm(Y, axis=1)

    cos_angles = dot_products / (X_norms * Y_norms)
    cos_angles = np.clip(cos_angles, -1, 1)

    angles = np.arccos(cos_angles)
    angle_results[d] = angles

    plt.figure(figsize=(7,4))
    plt.hist(angles, bins=40, density=True)
    plt.xlabel("angle in radians")
    plt.ylabel("density")
    plt.title(f"Angles between Gaussian vectors, d={d}")
    plt.show()

    print(f"d={d}: mean angle={np.mean(angles):.4f}, pi/2={np.pi/2:.4f}")

print("Conclusion: as dimension increases, random Gaussian vectors become almost orthogonal.")
```


```python
d = 100
X = rng.normal(size=(20000, d))
norms = np.linalg.norm(X, axis=1)
center = np.sqrt(d)

deviations = np.abs(norms - center)
beta = float(np.quantile(deviations, 0.99))

estimated_prob = np.mean((norms > center - beta) & (norms < center + beta))

print("beta for about 0.99 probability when d=100:", beta)
print("Estimated probability:", estimated_prob)
```

---
# Problem 4 [8p]

Text Markov chain problem.

1. Split `prideAndPrejudiceFirstChapter` into words.
2. Treat words as states.
3. Estimate transition matrix using MLE: transition count divided by row total.


```python
def makeFreqDict(myDataList):
    freqDict = {}
    for res in myDataList:
        if res in freqDict:
            freqDict[res] = freqDict[res] + 1
        else:
            freqDict[res] = 1
    return freqDict

prideAndPrejudiceFirstChapter = """It is a truth universally acknowledged, that a single man in
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
solace was visiting and news.""".lower()

import re
subs = "_;.,”“?!"
for sub in subs:
    prideAndPrejudiceFirstChapter = prideAndPrejudiceFirstChapter.replace(sub, ' ')
prideAndPrejudiceFirstChapter = re.sub('\s+', ' ', prideAndPrejudiceFirstChapter).strip()
```


```python
words = prideAndPrejudiceFirstChapter.split(' ')
unique_words = sorted(set(words))
n_words = len(unique_words)

print("First 20 words:", words[:20])
print("Number of total words:", len(words))
print("Number of unique words:", n_words)
```


```python
transitions = list(zip(words[:-1], words[1:]))
transition_counts = makeFreqDict(transitions)

indexToWord = {i: word for i, word in enumerate(unique_words)}
wordToIndex = {word: i for i, word in indexToWord.items()}

print("First 10 transitions:", transitions[:10])
print("Example count ('it','is'):", transition_counts.get(('it','is'), 0))
```


```python
transition_matrix = np.zeros((n_words, n_words))

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

print("Transition matrix shape:", transition_matrix.shape)
print("First 10 row sums:", transition_matrix.sum(axis=1)[:10])
print("p_it_is =", transition_matrix[wordToIndex['it'], wordToIndex['is']])
```


```python
np.random.seed(1)

start = np.zeros(shape=(n_words, 1))
start[0, 0] = 1
current_pos = start

generated_words = []
for i in range(30):
    random_word_index = np.random.choice(range(n_words), p=current_pos.reshape(-1))
    generated_words.append(indexToWord[random_word_index])

    current_pos = np.zeros_like(start)
    current_pos[random_word_index] = 1
    current_pos = (current_pos.T @ transition_matrix).T

print("Generated Markov text:")
print(" ".join(generated_words))
```

---
# Problem 5 [8p]

There is a csv file called `digits.csv`.

Tasks:
1. Load digits into shape `1797 x 64` and labels into shape `(1797,)`.
2. Perform PCA using SVD with 2 components.
3. Compute explained variance.
4. Predict labels using a model based on two PCA components.
5. Compare with random projection into R^2.


```python
def load_digits(file_name):
    df = pd.read_csv(file_name)

    possible_label_names = ["label", "target", "y", "class"]
    label_col = None
    for col in df.columns:
        if col.lower() in possible_label_names:
            label_col = col
            break

    if label_col is not None:
        y = df[label_col].values
        X = df.drop(columns=[label_col]).values
    else:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

    X = X.astype(float)
    y = y.astype(int)
    return X, y

def plot_digit(digit):
    assert digit.shape == (64,)
    plt.gray()
    plt.imshow(digit.reshape(8, 8))
    plt.axis("off")
    plt.show()

def pca_svd(X, n_components=2):
    X = np.asarray(X)
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    components = Vt[:n_components]
    X_pca = X_centered @ components.T

    explained_variance = (S**2) / (X.shape[0] - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()

    return X_pca, components, explained_variance_ratio[:n_components], X_mean

try:
    X_digits, y_digits = load_digits("digits.csv")

    print("X_digits shape:", X_digits.shape)
    print("y_digits shape:", y_digits.shape)

    plot_digit(X_digits[0])

    X_pca, components, explained_ratio, X_mean = pca_svd(X_digits, n_components=2)

    print("PCA output shape:", X_pca.shape)
    print("Explained variance ratio first two components:", explained_ratio)
    print("Total explained variance by first 2 PCs:", explained_ratio.sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_digits, test_size=0.3, random_state=1, stratify=y_digits
    )

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    pca_accuracy = accuracy_score(y_test, pred)
    print("KNN accuracy using 2 PCA components:", pca_accuracy)

    rng = np.random.default_rng(1)
    R = rng.normal(size=(X_digits.shape[1], 2))
    R = R / np.sqrt(2)
    X_random_proj = X_digits @ R

    X_train_rp, X_test_rp, y_train_rp, y_test_rp = train_test_split(
        X_random_proj, y_digits, test_size=0.3, random_state=1, stratify=y_digits
    )

    clf_rp = KNeighborsClassifier(n_neighbors=5)
    clf_rp.fit(X_train_rp, y_train_rp)
    pred_rp = clf_rp.predict(X_test_rp)

    rp_accuracy = accuracy_score(y_test_rp, pred_rp)
    print("KNN accuracy using random projection to 2D:", rp_accuracy)

    print("Explanation: PCA usually performs better than a random 2D projection because PCA chooses high-variance directions.")

    plt.figure(figsize=(7,5))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y_digits, s=10)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Digits projected onto first two PCA components")
    plt.colorbar(label="digit label")
    plt.show()

except FileNotFoundError:
    print("digits.csv not found. Put digits.csv in the same folder as this notebook, then run this cell.")
```
