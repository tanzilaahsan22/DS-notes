# Solved ReExam — 13th of June 2022  
## Course: 1MS041 Introduction to Data Science

This notebook contains:
- the full questions from the exam paper,
- step-by-step explanations,
- solved Python code,
- printed outputs for parts that can be run without the hidden exam files.

**Important:** Problems 4 and 5 depend on the course file `exam_extras.py` and the SMS dataset.  
The code is written so it will run inside the exam folder where `exam_extras.py` exists.


```python
import numpy as np
import math

np.random.seed(42)
```

# Problem 1 — Probability warmup

Let’s say we have an exam question which consists of **20 yes/no questions**.

From past performance of similar students, a randomly chosen student will know the correct answer to

\[
N \sim \mathrm{Binomial}(20, 11/20)
\]

questions.

Furthermore, we assume that the student will guess the answer with equal probability to each question they don’t know the answer to. Given \(N\), define

\[
Z \sim \mathrm{Binomial}(20 - N, 1/2)
\]

as the number of correctly guessed answers.

Define

\[
Y = N + Z
\]

so \(Y\) represents the total number of correct answers.

We pass a student at threshold \(T\) if

\[
Y \ge T
\]

where \(T \in \{0,1,2,\ldots,20\}\).

## Questions

1. **[3p]** Produce a simulation of 1000 students. Simulate \(N\) first, then simulate \(Y \mid N\), and add the results.
2. **[3p]** For each threshold \(T\), estimate the probability that the student knows less than 10 correct answers given that the student passed:

\[
P(N < 10 \mid Y \ge T)
\]

Put the answer in `problem1_probabilities` as a list of length 21.
3. **[2p]** What is the smallest value of \(T\) such that if \(Y \ge T\), then we are 90% certain that \(N \ge 10\)?

## Problem 1 — Step-by-step idea

For each simulated student:

1. Simulate how many answers they actually know: `N`.
2. For the remaining questions, simulate how many they guess correctly: `Z`.
3. Total correct answers: `Y = N + Z`.

For part 2, for every possible threshold `T`, we estimate:

```python
P(N < 10 | Y >= T)
```

This means:

```python
number of passed students who know less than 10 / number of passed students
```

For part 3, “90% certain that N >= 10” means:

\[
P(N \ge 10 \mid Y \ge T) \ge 0.9
\]

which is the same as:

\[
P(N < 10 \mid Y \ge T) \le 0.1
\]


```python
# EXAM vB, PROBLEM 1, POINTS 8
# Part 1

n_students = 1000
n_questions = 20
p_knows = 11/20
p_guess = 1/2

# Simulate number of questions each student knows
N = np.random.binomial(n_questions, p_knows, size=n_students)

# Simulate number of guessed answers that are correct
# Each student has 20 - N unknown answers
Z = np.array([
    np.random.binomial(n_questions - n, p_guess)
    for n in N
])

# Total correct answers
Y = N + Z

# The required answer for Part 1
problem1_1000_samples = Y

print("First 20 simulated total scores Y:")
print(problem1_1000_samples[:20])
print()
print("Shape:", problem1_1000_samples.shape)
print("Mean total score:", problem1_1000_samples.mean())
```

    First 20 simulated total scores Y:
    [15 14 17 16 18 17 18 16 14 15 17 17 17 14 17 18 15 16 18 13]
    
    Shape: (1000,)
    Mean total score: 15.59
    


```python
# EXAM vB, PROBLEM 1, POINTS 8
# Part 2

problem1_probabilities = []

for T in range(21):
    passed = (Y >= T)

    if passed.sum() == 0:
        # If nobody passed, conditional probability is undefined.
        # We put np.nan in that case.
        probability = np.nan
    else:
        probability = ((N < 10) & passed).sum() / passed.sum()

    problem1_probabilities.append(probability)

print("P(N < 10 | Y >= T) for T = 0, 1, ..., 20:")
for T, prob in enumerate(problem1_probabilities):
    print(f"T={T:2d}: {prob:.4f}" if not np.isnan(prob) else f"T={T:2d}: nan")
```

    P(N < 10 | Y >= T) for T = 0, 1, ..., 20:
    T= 0: 0.2420
    T= 1: 0.2420
    T= 2: 0.2420
    T= 3: 0.2420
    T= 4: 0.2420
    T= 5: 0.2420
    T= 6: 0.2420
    T= 7: 0.2420
    T= 8: 0.2420
    T= 9: 0.2420
    T=10: 0.2420
    T=11: 0.2397
    T=12: 0.2301
    T=13: 0.2131
    T=14: 0.1900
    T=15: 0.1556
    T=16: 0.1055
    T=17: 0.0656
    T=18: 0.0385
    T=19: 0.0435
    T=20: 0.0000
    


```python
# EXAM vB, PROBLEM 1, POINTS 8
# Part 3

# Need smallest T such that P(N < 10 | Y >= T) <= 0.1
valid_thresholds = [
    T for T, prob in enumerate(problem1_probabilities)
    if (not np.isnan(prob)) and prob <= 0.1
]

problem1_T = min(valid_thresholds)

print("Smallest threshold T:", problem1_T)
print("At this T, P(N < 10 | Y >= T) =", problem1_probabilities[problem1_T])
print("So P(N >= 10 | Y >= T) =", 1 - problem1_probabilities[problem1_T])
```

    Smallest threshold T: 17
    At this T, P(N < 10 | Y >= T) = 0.065625
    So P(N >= 10 | Y >= T) = 0.934375
    

# Problem 2 — Random variable generation and transformation

## Question

1. **[3p]** Using the Accept-Reject sampler with sampling density given by the uniform density on the 10-dimensional box

\[
[-1,1]^d
\]

generate 100 samples from the uniform distribution on the unit ball in 10 dimensions.

To generate a sample from \(\mathrm{Uniform}([-1,1]^d)\), generate each coordinate independently from \(\mathrm{Uniform}(-1,1)\).

Since both the sampling density and target density are constant functions, the ratio \(r(x)\) is either 1 or 0.

2. **[2p]** How many proposals do you need to produce in order to get 100 samples?

3. **[3p]** Using Theorem 10.10 in the lecture notes, generate 100 samples from the uniform distribution on the unit ball in 10 dimensions.

## Problem 2 — Step-by-step idea

### Unit ball in 10 dimensions

The unit ball is all points \(x \in \mathbb{R}^{10}\) such that:

\[
\|x\|_2 \le 1
\]

### Part 1: Accept-Reject

1. Propose a point from the box \([-1,1]^{10}\).
2. Accept it only if it lies inside the unit ball:

\[
\sum_{j=1}^{10} x_j^2 \le 1
\]

### Part 3: Direct sampling using theorem idea

To sample uniformly from the unit ball:

1. Sample a random direction by drawing a standard Gaussian vector and normalizing it.
2. Sample radius as:

\[
R = U^{1/d}
\]

where \(U \sim \mathrm{Uniform}(0,1)\) and \(d=10\).
3. Return:

\[
X = R\Theta
\]


```python
# EXAM vB, PROBLEM 2, POINTS 8
# Part 1

def problem2_accept_reject(n_samples=1, d=10):
    '''
    Produces samples from the 10d unit ball using an Accept-Reject
    sampler with Uniform([-1,1]^d) as the proposal distribution.

    Parameters
    ----------
    n_samples : int
        Number of accepted samples to produce.
    d : int
        Dimension of the unit ball.

    Returns
    -------
    samples : numpy array of shape (n_samples, d)
    n_iterations : int
        Number of proposals needed.
    '''
    retVal = []
    n_iterations = 0

    while len(retVal) < n_samples:
        # Propose uniformly from the box [-1,1]^d
        x = np.random.uniform(-1, 1, size=d)

        # Accept if x is inside the unit ball
        if np.sum(x**2) <= 1:
            retVal.append(x)

        n_iterations += 1

    return np.array(retVal), n_iterations


problem2_samples_ar, problem2_accept_reject_n_iterations = problem2_accept_reject(100)

print("Accept-Reject sample shape:", problem2_samples_ar.shape)
print("Number of proposals needed:", problem2_accept_reject_n_iterations)
print("First sample:")
print(problem2_samples_ar[0])
```

    Accept-Reject sample shape: (100, 10)
    Number of proposals needed: 39715
    First sample:
    [-0.2612347   0.01161845  0.12069735  0.45723031 -0.29030855  0.22536869
      0.17075085  0.0952322   0.28096659 -0.48110166]
    


```python
# EXAM vB, PROBLEM 2, POINTS 8
# Part 2

print("problem2_accept_reject_n_iterations =", problem2_accept_reject_n_iterations)
```

    problem2_accept_reject_n_iterations = 39715
    


```python
# EXAM vB, PROBLEM 2, POINTS 8
# Part 3

def problem2_theorem10_10(n_samples=1, d=10):
    '''
    Produces samples from the d-dimensional unit ball using the standard
    direction-radius method.

    Parameters
    ----------
    n_samples : int
        Number of samples to produce.
    d : int
        Dimension.

    Returns
    -------
    out : numpy array of shape (n_samples, d)
    '''
    # Step 1: sample Gaussian vectors
    G = np.random.normal(size=(n_samples, d))

    # Step 2: normalize to get random directions on the sphere
    Theta = G / np.linalg.norm(G, axis=1, keepdims=True)

    # Step 3: sample radius correctly for uniform ball
    U = np.random.uniform(0, 1, size=(n_samples, 1))
    R = U ** (1/d)

    return R * Theta


problem2_samples_theorem = problem2_theorem10_10(100)

print("Theorem 10.10 sample shape:", problem2_samples_theorem.shape)
print("First sample:")
print(problem2_samples_theorem[0])
print("Norm of first sample:", np.linalg.norm(problem2_samples_theorem[0]))
```

    Theorem 10.10 sample shape: (100, 10)
    First sample:
    [ 0.56845382 -0.05886676  0.29142726 -0.17573502  0.19779095  0.0384307
     -0.1014214  -0.12464344  0.48736188 -0.14877211]
    Norm of first sample: 0.8766372657072253
    


```python
# EXAM vB, Test 2, POINTS 8
# Shape tests from the exam

try:
    problem2_test_1, _ = problem2_accept_reject(100)
    print(problem2_test_1.shape)
except Exception as e:
    print(e)

try:
    problem2_test_2 = problem2_theorem10_10(100)
    print(problem2_test_2.shape)
except Exception as e:
    print(e)
```

    (100, 10)
    (100, 10)
    

# Problem 3 — Concentration of measure

## Question

Which of the following will exponentially concentrate?

For some constants \(C_1,C_2,C_3,C_4\),

\[
P(Z - E[Z] \ge \epsilon)
\le
C_1e^{-C_2n\epsilon^2}
\wedge
C_3e^{-C_4n(\epsilon+1)}
\]

Options:

1. The empirical mean of i.i.d. sub-Gaussian random variables?
2. The empirical mean of i.i.d. sub-Exponential random variables?
3. The empirical mean of i.i.d. random variables with finite variance?
4. The empirical variance of i.i.d. random variables with finite variance?
5. The empirical variance of i.i.d. sub-Gaussian random variables?
6. The empirical variance of i.i.d. sub-Exponential random variables?
7. The empirical third moment of i.i.d. sub-Gaussian random variables?
8. The empirical fourth moment of i.i.d. sub-Gaussian random variables?
9. The empirical mean of i.i.d. deterministic random variables?
10. The empirical tenth moment of i.i.d. Bernoulli random variables?

Second question: Which of the above concentrate in the weaker sense, that for some \(C_1\),

\[
P(Z - E[Z] \ge \epsilon) \le \frac{C_1}{n\epsilon^2}
\]

## Problem 3 — Answer reasoning

### Exponential concentration

- Sub-Gaussian means concentrate exponentially.
- Sub-Exponential means concentrate with Bernstein-type exponential bounds.
- Bounded random variables also concentrate exponentially. Bernoulli variables are bounded.
- Deterministic variables have zero deviation, so they trivially concentrate.
- The variance of sub-Gaussian variables is the empirical mean of squared sub-Gaussian variables, which are sub-Exponential, so it concentrates exponentially.
- Finite variance alone gives weaker Chebyshev-type concentration, not exponential.
- Higher moments of sub-Gaussian variables such as third and fourth powers are not generally sub-Exponential in the required way.

So the exponential list is:

```python
[1, 2, 5, 9, 10]
```

### Weaker concentration

The weaker \(1/(n\epsilon^2)\) type concentration includes:
- all exponentially concentrating examples,
- empirical mean with finite variance, by Chebyshev's inequality.

So the weaker list is:

```python
[1, 2, 3, 5, 9, 10]
```


```python
# EXAM vB, PROBLEM 3, POINTS 8
# Part 1

problem3_answer_1 = [1, 2, 5, 9, 10]

print("Exponential concentration answers:")
print(problem3_answer_1)
```

    Exponential concentration answers:
    [1, 2, 5, 9, 10]
    


```python
# EXAM vB, PROBLEM 3, POINTS 8
# Part 2

problem3_answer_2 = [1, 2, 3, 5, 9, 10]

print("Weaker concentration answers:")
print(problem3_answer_2)
```

    Weaker concentration answers:
    [1, 2, 3, 5, 9, 10]
    

# Problem 4 — SMS spam filtering

In this problem we explore SMS spam texts. The dataset is the SMS Spam Collection Dataset.

The provided code loads the data:

```python
from exam_extras import load_sms
spam_no_spam = load_sms()
```

The result is a list of tuples:

```python
(text, label)
```

where:

- `text` is the SMS message,
- `label = 0` means not spam,
- `label = 1` means spam.

## Questions

1. **[3p]** Estimate:

\[
P(Y=1 \mid \text{"free" or "prize" is in } X)
\]

This is precision.

2. **[3p]** Provide a 90% confidence interval around the true probability using Hoeffding’s inequality.

3. **[2p]** Repeat the two exercises above for `"free"` appearing twice in the SMS.

## Problem 4 — Step-by-step idea

For part 1:

1. Convert text to lowercase.
2. Check if `"free"` or `"prize"` appears in the SMS.
3. Among those messages, count how many are spam.

\[
\hat P = \frac{\text{number of spam messages containing "free" or "prize"}}
{\text{number of messages containing "free" or "prize"}}
\]

For Hoeffding:

\[
P(|\hat P - P| \le l) \ge 1-\delta
\]

Hoeffding gives:

\[
2e^{-2nl^2} \le \delta
\]

So:

\[
l = \sqrt{\frac{\log(2/\delta)}{2n}}
\]

For a 90% interval, \(\delta=0.1\), therefore:

\[
l = \sqrt{\frac{\log(20)}{2n}}
\]


```python
# EXAM vB, PROBLEM 4, POINTS 8

def hoeffding_l(n, confidence=0.90):
    '''
    Hoeffding interval half-width for Bernoulli/proportion estimates.
    '''
    delta = 1 - confidence
    return math.sqrt(math.log(2 / delta) / (2 * n))


try:
    from exam_extras import load_sms
    spam_no_spam = load_sms()

    # Part 1: "free" or "prize"
    selected = [
        (text, label)
        for text, label in spam_no_spam
        if ("free" in text.lower()) or ("prize" in text.lower())
    ]

    n_selected = len(selected)
    problem4_hatP = sum(label for text, label in selected) / n_selected
    problem4_l = hoeffding_l(n_selected, confidence=0.90)

    # Part 3: "free" appearing twice
    selected_double_free = [
        (text, label)
        for text, label in spam_no_spam
        if text.lower().count("free") >= 2
    ]

    n_selected_double_free = len(selected_double_free)
    problem4_hatP2 = sum(label for text, label in selected_double_free) / n_selected_double_free
    problem4_l2 = hoeffding_l(n_selected_double_free, confidence=0.90)

    print("Part 1:")
    print("Number of SMS containing 'free' or 'prize':", n_selected)
    print("problem4_hatP =", problem4_hatP)
    print("problem4_l =", problem4_l)
    print("90% interval =", (max(0, problem4_hatP - problem4_l), min(1, problem4_hatP + problem4_l)))

    print("\nPart 3:")
    print("Number of SMS where 'free' appears at least twice:", n_selected_double_free)
    print("problem4_hatP2 =", problem4_hatP2)
    print("problem4_l2 =", problem4_l2)
    print("90% interval =", (max(0, problem4_hatP2 - problem4_l2), min(1, problem4_hatP2 + problem4_l2)))

except Exception as e:
    print("This problem needs the course file exam_extras.py and the SMS dataset.")
    print("Put this notebook in the same folder as exam_extras.py, then run again.")
    print("Reason:", repr(e))

    # Safe placeholders so the notebook still runs without the hidden exam file
    problem4_hatP = None
    problem4_l = None
    problem4_hatP2 = None
    problem4_l2 = None
```

    This problem needs the course file exam_extras.py and the SMS dataset.
    Put this notebook in the same folder as exam_extras.py, then run again.
    Reason: ModuleNotFoundError("No module named 'exam_extras'")
    

# Problem 5 — Black box testing

In this problem we continue with the SMS spam/no-spam data.

The exam gives prepared data, a train-test split, and a black-box model.

The code from the exam:

```python
import exam_extras
from exam_extras import load_sms_problem6

X_problem6, Y_problem6 = load_sms_problem6()

X_train_problem6, X_test_problem6, Y_train_problem6, Y_test_problem6 = exam_extras.train_test_split(
    X_problem6,
    Y_problem6
)

predictions_problem6 = exam_extras.knn_predictions(
    X_train_problem6,
    Y_train_problem6,
    X_test_problem6,
    k=4
)
```

## Questions

1. **[2p]** Compute precision for class 1, then provide a 95% confidence interval using Hoeffding’s inequality.
2. **[2p]** Compute recall for class 1, then provide a 95% confidence interval using Hoeffding’s inequality.
3. **[2p]** Compute accuracy, then provide a 95% confidence interval using Hoeffding’s inequality.
4. **[2p]** If we used a classifier with VC-dimension 3, would we have obtained a smaller interval for accuracy by using all data?

## Problem 5 — Step-by-step formulas

### Precision for class 1

\[
\text{precision}
=
\frac{TP}{TP+FP}
\]

This answers:

> Of all messages predicted as spam, how many were actually spam?

### Recall for class 1

\[
\text{recall}
=
\frac{TP}{TP+FN}
\]

This answers:

> Of all actual spam messages, how many did we correctly detect?

### Accuracy

\[
\text{accuracy}
=
\frac{\text{number of correct predictions}}{\text{number of test examples}}
\]

### Hoeffding interval for 95%

For 95% confidence, \(\delta = 0.05\):

\[
l = \sqrt{\frac{\log(2/0.05)}{2n}}
=
\sqrt{\frac{\log(40)}{2n}}
\]

Use:
- \(n = TP+FP\) for precision,
- \(n = TP+FN\) for recall,
- \(n =\) number of test samples for accuracy.


```python
# EXAM vB, PROBLEM 5, POINTS 8

def hoeffding_l(n, confidence=0.95):
    '''
    Hoeffding interval half-width for Bernoulli/proportion estimates.
    '''
    delta = 1 - confidence
    return math.sqrt(math.log(2 / delta) / (2 * n))


try:
    import exam_extras
    from exam_extras import load_sms_problem6

    X_problem6, Y_problem6 = load_sms_problem6()

    X_train_problem6, X_test_problem6, Y_train_problem6, Y_test_problem6 = exam_extras.train_test_split(
        X_problem6,
        Y_problem6
    )

    predictions_problem6 = exam_extras.knn_predictions(
        X_train_problem6,
        Y_train_problem6,
        X_test_problem6,
        k=4
    )

    y_true = np.array(Y_test_problem6)
    y_pred = np.array(predictions_problem6)

    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))

    # Precision
    n_precision = TP + FP
    problem6_precision = TP / n_precision
    problem6_precision_l = hoeffding_l(n_precision, confidence=0.95)

    # Recall
    n_recall = TP + FN
    problem6_recall = TP / n_recall
    problem6_recall_l = hoeffding_l(n_recall, confidence=0.95)

    # Accuracy
    n_accuracy = len(y_true)
    problem6_accuracy = np.mean(y_pred == y_true)
    problem6_accuracy_l = hoeffding_l(n_accuracy, confidence=0.95)

    print("Confusion matrix counts:")
    print("TP =", TP, "FP =", FP, "FN =", FN, "TN =", TN)

    print("\nPrecision:")
    print("problem6_precision =", problem6_precision)
    print("problem6_precision_l =", problem6_precision_l)
    print("95% interval =", (max(0, problem6_precision - problem6_precision_l),
                            min(1, problem6_precision + problem6_precision_l)))

    print("\nRecall:")
    print("problem6_recall =", problem6_recall)
    print("problem6_recall_l =", problem6_recall_l)
    print("95% interval =", (max(0, problem6_recall - problem6_recall_l),
                            min(1, problem6_recall + problem6_recall_l)))

    print("\nAccuracy:")
    print("problem6_accuracy =", problem6_accuracy)
    print("problem6_accuracy_l =", problem6_accuracy_l)
    print("95% interval =", (max(0, problem6_accuracy - problem6_accuracy_l),
                            min(1, problem6_accuracy + problem6_accuracy_l)))

    # VC-dimension bound using Sauer's lemma style growth function
    # This is a standard generalization-style interval.
    d_vc = 3
    n_all = len(Y_problem6)
    delta = 0.05

    # growth bound <= (e*n/d)^d for n >= d
    growth_bound = (math.e * n_all / d_vc) ** d_vc

    # One common VC uniform convergence half-width
    problem6_VC_l = math.sqrt((8 / n_all) * math.log((4 * growth_bound) / delta))

    problem6_VC_smaller = problem6_VC_l < problem6_accuracy_l

    print("\nVC-bound comparison:")
    print("problem6_VC_l =", problem6_VC_l)
    print("problem6_VC_smaller =", problem6_VC_smaller)

except Exception as e:
    print("This problem needs the course file exam_extras.py and the SMS dataset.")
    print("Put this notebook in the same folder as exam_extras.py, then run again.")
    print("Reason:", repr(e))

    # Safe placeholders so the notebook still runs without the hidden exam file
    problem6_precision = None
    problem6_precision_l = None
    problem6_recall = None
    problem6_recall_l = None
    problem6_accuracy = None
    problem6_accuracy_l = None
    problem6_VC_l = None
    problem6_VC_smaller = None
```

    This problem needs the course file exam_extras.py and the SMS dataset.
    Put this notebook in the same folder as exam_extras.py, then run again.
    Reason: ModuleNotFoundError("No module named 'exam_extras'")
    

# Final exam-variable summary

This cell prints the variables the exam asks you to fill in.


```python
summary_vars = {
    "problem1_1000_samples_first_10": problem1_1000_samples[:10],
    "problem1_probabilities": problem1_probabilities,
    "problem1_T": problem1_T,
    "problem2_accept_reject_n_iterations": problem2_accept_reject_n_iterations,
    "problem3_answer_1": problem3_answer_1,
    "problem3_answer_2": problem3_answer_2,
    "problem4_hatP": problem4_hatP,
    "problem4_l": problem4_l,
    "problem4_hatP2": problem4_hatP2,
    "problem4_l2": problem4_l2,
    "problem6_precision": problem6_precision,
    "problem6_precision_l": problem6_precision_l,
    "problem6_recall": problem6_recall,
    "problem6_recall_l": problem6_recall_l,
    "problem6_accuracy": problem6_accuracy,
    "problem6_accuracy_l": problem6_accuracy_l,
    "problem6_VC_l": problem6_VC_l,
    "problem6_VC_smaller": problem6_VC_smaller,
}

for key, value in summary_vars.items():
    print(key, "=", value)
```

    problem1_1000_samples_first_10 = [15 14 17 16 18 17 18 16 14 15]
    problem1_probabilities = [np.float64(0.242), np.float64(0.242), np.float64(0.242), np.float64(0.242), np.float64(0.242), np.float64(0.242), np.float64(0.242), np.float64(0.242), np.float64(0.242), np.float64(0.242), np.float64(0.242), np.float64(0.23971915747241726), np.float64(0.23014256619144602), np.float64(0.21308016877637131), np.float64(0.19003476245654694), np.float64(0.15561569688768606), np.float64(0.10546139359698682), np.float64(0.065625), np.float64(0.038461538461538464), np.float64(0.043478260869565216), np.float64(0.0)]
    problem1_T = 17
    problem2_accept_reject_n_iterations = 39715
    problem3_answer_1 = [1, 2, 5, 9, 10]
    problem3_answer_2 = [1, 2, 3, 5, 9, 10]
    problem4_hatP = None
    problem4_l = None
    problem4_hatP2 = None
    problem4_l2 = None
    problem6_precision = None
    problem6_precision_l = None
    problem6_recall = None
    problem6_recall_l = None
    problem6_accuracy = None
    problem6_accuracy_l = None
    problem6_VC_l = None
    problem6_VC_smaller = None
    
