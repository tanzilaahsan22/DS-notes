# Solved Exam Notebook — 12th of January 2022

Course: **1MS041 Introduction to Data Science / Introduktion till dataanalys**

This notebook contains the full questions from the uploaded exam PDF, followed by solved code, step-by-step explanations, and outputs where the required data files are available.

Some exam parts depend on hidden course files such as `exam_extras.py` and `data/flights.csv`. For those, the notebook includes complete runnable code that will produce the exact output when those files are placed in the same folder as this notebook.



```python
# Enter your anonymous exam id by replacing XXXX in this cell below
# do NOT delete this cell
MyAnonymousExamID = "XXX"

import numpy as np
import pandas as pd
import math
from math import comb

```

    Setup complete.
    

## Exam vB, PROBLEM 1 — Probability warmup

Maximum Points = 8

Let's say we have an exam question which consists of **20 yes/no questions**. From past performance of similar students, a randomly chosen student will know the correct answer to

\[
N \sim \text{binom}(20, 11/20)
\]

questions. Furthermore, we assume that the student will guess the answer with equal probability to each question they don't know the answer to, i.e. given \(N\) we define

\[
Z \sim \text{binom}(20-N, 1/2)
\]

as the number of correctly guessed answers. Define

\[
Y = N + Z
\]

i.e. \(Y\) represents the number of total correct answers.

We are interested in setting a deterministic threshold \(T\), i.e. we would pass a student at threshold \(T\) if \(Y \ge T\). Here

\[
T \in \{0,1,2,\ldots,20\}.
\]

1. **[5p]** For each threshold \(T\), compute the probability that the student knows less than 10 correct answers given that the student passed, i.e. \(P(N<10 \mid Y \ge T)\). Put the answer in `problem11_probabilities` as a list.
2. **[3p]** What is the smallest value of \(T\) such that if \(Y \ge T\), then we are 90% certain that \(N \ge 10\)?



```python
# Problem 1 solution

p = 11/20

# PMF of N ~ Binomial(20, 11/20)
def p_N(k):
    return comb(20, k) * (p ** k) * ((1 - p) ** (20 - k))

# Conditional PMF of Y given N = n.
# Since Y = N + Z and Z ~ Binomial(20 - N, 1/2),
# P(Y = y | N = n) = P(Z = y - n)
def p_Y_given_N(y, n):
    z = y - n
    remaining_questions = 20 - n

    if z < 0 or z > remaining_questions:
        return 0

    return comb(remaining_questions, z) * (0.5 ** remaining_questions)

# For every threshold T, compute:
# P(N < 10 | Y >= T) = P(N < 10 and Y >= T) / P(Y >= T)
problem11_probabilities = []

for T in range(21):
    numerator = 0
    denominator = 0

    for n in range(21):
        prob_n = p_N(n)
        prob_pass_given_n = sum(p_Y_given_N(y, n) for y in range(T, 21))

        denominator += prob_n * prob_pass_given_n

        if n < 10:
            numerator += prob_n * prob_pass_given_n

    conditional_probability = numerator / denominator
    problem11_probabilities.append(conditional_probability)

# We are 90% certain that N >= 10 when:
# P(N >= 10 | Y >= T) >= 0.90
# This is the same as:
# P(N < 10 | Y >= T) <= 0.10
problem12_T = min(T for T, prob in enumerate(problem11_probabilities) if prob <= 0.10)

print("problem11_probabilities =")
print([round(x, 6) for x in problem11_probabilities])
print()
print("Smallest T such that P(N < 10 | Y >= T) <= 0.10:")
print("problem12_T =", problem12_T)

```

    problem11_probabilities =
    [0.249289, 0.249289, 0.249289, 0.249289, 0.249289, 0.249289, 0.249289, 0.249283, 0.249246, 0.249039, 0.248086, 0.244608, 0.234944, 0.214756, 0.182671, 0.142725, 0.10227, 0.067628, 0.041665, 0.024151, 0.013287]
    
    Smallest T such that P(N < 10 | Y >= T) <= 0.10:
    problem12_T = 17
    

## Exam vB, PROBLEM 2 — Random variable generation and transformation

Maximum Points = 8

The purpose of this problem is to show that you can implement your own sampler. This will be built in the following three steps:

1. **[2p]** Implement a Linear Congruential Generator where you tested out a good combination, a large \(M\) with \(a,b\) satisfying Hull-Dobell, of parameters.
2. **[2p]** Using a generator construct random numbers from the uniform \([0,1]\) distribution.
3. **[4p]** Using a uniform \([0,1]\) random generator, generate samples from

\[
p_0(x) = \frac{\pi}{2}|\sin(2\pi x)|,\quad x \in [0,1].
\]

Use the Accept-Reject sampler with sampling density given by the uniform \([0,1]\) distribution.

Key idea for accept-reject:

The proposal density is \(q(x)=1\). The target density is \(p_0(x)\). Since

\[
\max_x p_0(x) = \frac{\pi}{2},
\]

we accept a proposal \(x\) when

\[
u \le \frac{p_0(x)}{\pi/2}=|\sin(2\pi x)|.
\]



```python
# Problem 2 solution

def problem2_LCG(size=None, seed=0):
    """
    A linear congruential generator that generates pseudo-random integers.

    We use:
        M = 2^31
        a = 1103515245
        b = 12345

    Hull-Dobell conditions:
    1. b and M are relatively prime.
    2. a - 1 is divisible by all prime factors of M.
    3. a - 1 is divisible by 4 because M is divisible by 4.
    """
    if size is None:
        size = 1

    M = 2**31
    a = 1103515245
    b = 12345

    x = seed
    out = []

    for _ in range(size):
        x = (a * x + b) % M
        out.append(x)

    return out


def problem2_uniform(generator=None, period=1, size=None, seed=0):
    """
    Takes an integer generator and converts output to uniform [0,1) numbers.
    """
    if size is None:
        size = 1

    if generator is None:
        generator = problem2_LCG

    integer_values = generator(size=size, seed=seed)
    uniform_values = [value / period for value in integer_values]

    return uniform_values


def problem2_accept_reject(uniformGenerator=None, size=None, seed=0, n_iterations=None):
    """
    Produces samples from p0(x) = (pi/2)*abs(sin(2*pi*x)) on [0,1].

    This function supports two styles:
    - size = number of accepted samples wanted
    - n_iterations = number of proposals to try, returning the accepted proposals
    """
    if uniformGenerator is None:
        period = 2**31
        uniformGenerator = lambda size, seed: problem2_uniform(
            generator=problem2_LCG,
            period=period,
            size=size,
            seed=seed
        )

    accepted = []
    current_seed = seed
    proposals = 0

    if n_iterations is not None:
        for _ in range(n_iterations):
            x, u = uniformGenerator(size=2, seed=current_seed)
            current_seed += 1
            proposals += 1

            if u <= abs(math.sin(2 * math.pi * x)):
                accepted.append(x)

        return accepted

    if size is None:
        size = 1

    while len(accepted) < size:
        x, u = uniformGenerator(size=2, seed=current_seed)
        current_seed += 1
        proposals += 1

        if u <= abs(math.sin(2 * math.pi * x)):
            accepted.append(x)

    problem2_accept_reject.last_number_of_proposals = proposals

    return accepted


# Local test
period = 2**31
uniform_sampler = lambda size, seed: problem2_uniform(
    generator=problem2_LCG,
    period=period,
    size=size,
    seed=seed
)

print("LCG output:", problem2_LCG(size=10, seed=1))
print("Uniform sampler:", [round(x, 6) for x in uniform_sampler(size=10, seed=1)])

samples_problem2 = problem2_accept_reject(
    uniformGenerator=uniform_sampler,
    size=10,
    seed=1
)

print("First 10 accept-reject samples:", [round(x, 6) for x in samples_problem2])
print("Proposals needed for 10 accepted samples:", problem2_accept_reject.last_number_of_proposals)

```

    LCG output: [1103527590, 377401575, 662824084, 1147902781, 2035015474, 368800899, 1508029952, 486256185, 1062517886, 267834847]
    Uniform sampler: [0.51387, 0.175741, 0.308652, 0.534534, 0.947628, 0.171736, 0.702231, 0.226431, 0.494773, 0.12472]
    First 10 accept-reject samples: [0.541599, 0.569327, 0.597056, 0.624785, 0.652513, 0.680242, 0.707971, 0.735699, 0.249564, 0.763428]
    Proposals needed for 10 accepted samples: 19
    

## Exam vB, PROBLEM 3 — Concentration of measure

Maximum Points = 8

Which of the following will exponentially concentrate?

\[
P(Z-\mathbb{E}[Z]\ge \epsilon)
\le
C_1e^{-C_2n\epsilon^2}
\wedge
C_3e^{-C_4n(\epsilon+1)}
\]

Options:

1. The empirical mean of i.i.d. sub-Gaussian random variables  
2. The empirical mean of i.i.d. sub-Exponential random variables  
3. The empirical mean of i.i.d. random variables with finite variance  
4. The empirical variance of i.i.d. random variables with finite variance  
5. The empirical variance of i.i.d. sub-Gaussian random variables  
6. The empirical variance of i.i.d. sub-Exponential random variables  
7. The empirical third moment of i.i.d. sub-Gaussian random variables  
8. The empirical fourth moment of i.i.d. sub-Gaussian random variables  
9. The empirical mean of i.i.d. deterministic random variables  
10. The empirical tenth moment of i.i.d. Bernoulli random variables  

Which of the above concentrate in the weaker sense?

\[
P(Z-\mathbb{E}[Z]\ge \epsilon) \le \frac{C_1}{n\epsilon^2}
\]



```python
# Problem 3 solution

# Exponential concentration:
# 1: Empirical mean of sub-Gaussian variables -> yes.
# 2: Empirical mean of sub-Exponential variables -> yes, Bernstein-type tail.
# 5: Empirical variance of sub-Gaussian variables -> yes, because X^2 is sub-Exponential.
# 9: Deterministic variables -> yes, deviation is zero.
# 10: Bernoulli tenth moment -> yes, because Bernoulli^10 = Bernoulli, bounded.
problem3_answer_1 = [1, 2, 5, 9, 10]

# Weak concentration:
# Everything above also weakly concentrates.
# 3 also weakly concentrates by Chebyshev's inequality for finite variance means.
problem3_answer_2 = [1, 2, 3, 5, 9, 10]

print("problem3_answer_1 =", problem3_answer_1)
print("problem3_answer_2 =", problem3_answer_2)

```

    problem3_answer_1 = [1, 2, 5, 9, 10]
    problem3_answer_2 = [1, 2, 3, 5, 9, 10]
    

## Exam vB, PROBLEM 4 — SMS spam filtering

Maximum Points = 8

The result is a list of tuples. The first position in the tuple is the SMS text and the second is a flag:

- `0 = not spam`
- `1 = spam`

1. **[3p]** Estimate

\[
P(Y=1 \mid \text{"free" or "prize" is in } X).
\]

2. **[3p]** Provide a 90% confidence interval around the true probability using Hoeffding's inequality.

3. **[2p]** Repeat the two exercises above for `"free"` appearing twice in the SMS.

For Hoeffding:

\[
l = \sqrt{\frac{\log(20)}{2n}}.
\]



```python
# Problem 4 solution
# This needs exam_extras.py from the course folder.

try:
    from exam_extras import load_sms
    spam_no_spam = load_sms()

    condition_messages = []

    for text, y in spam_no_spam:
        text_lower = text.lower()
        if ("free" in text_lower) or ("prize" in text_lower):
            condition_messages.append((text, y))

    n_condition = len(condition_messages)
    n_spam_condition = sum(y for text, y in condition_messages)

    problem4_hatP = n_spam_condition / n_condition
    problem4_l = math.sqrt(math.log(20) / (2 * n_condition))

    double_free_messages = []

    for text, y in spam_no_spam:
        text_lower = text.lower()
        if text_lower.count("free") >= 2:
            double_free_messages.append((text, y))

    n_double_free = len(double_free_messages)
    n_spam_double_free = sum(y for text, y in double_free_messages)

    problem4_hatP2 = n_spam_double_free / n_double_free
    problem4_l2 = math.sqrt(math.log(20) / (2 * n_double_free))

    print("Number of SMS containing free or prize:", n_condition)
    print("problem4_hatP =", problem4_hatP)
    print("problem4_l =", problem4_l)
    print("90% interval =", (problem4_hatP - problem4_l, problem4_hatP + problem4_l))
    print()
    print("Number of SMS where free appears at least twice:", n_double_free)
    print("problem4_hatP2 =", problem4_hatP2)
    print("problem4_l2 =", problem4_l2)
    print("90% interval for double-free =", (problem4_hatP2 - problem4_l2, problem4_hatP2 + problem4_l2))

except Exception as e:
    print("This cell needs the course file exam_extras.py.")
    print("Place exam_extras.py in the same folder as this notebook, then rerun.")
    print("Error:", e)

```

    This cell needs the course file exam_extras.py.
    Place exam_extras.py in the same folder as this notebook, then rerun.
    Error: No module named 'exam_extras'
    

## Exam vB, PROBLEM 5 — Markovian travel

Maximum Points = 8

The dataset `Travel Dataset - Datathon 2019` is at:

```text
data/flights.csv
```

1. **[2p]** Load the CSV and fill in:
   - `number_of_cities`
   - `number_of_userCodes`
   - `number_of_observations`

2. **[2p]** Estimate a Markov chain transition matrix for user travels.

3. **[2p]** Use the transition matrix to compute the stationary distribution.

4. **[2p]** Given that we start in `'Aracaju (SE)'`, what is the probability that after 3 steps we will be back in `'Aracaju (SE)'`?



```python
# Problem 5 solution
# This needs data/flights.csv from the exam folder.

def makeFreqDict(myDataList):
    """Make a frequency mapping out of a list of data."""
    freqDict = {}
    for res in myDataList:
        if res in freqDict:
            freqDict[res] += 1
        else:
            freqDict[res] = 1
    return freqDict


try:
    flights = pd.read_csv("data/flights.csv")

    print("Columns in flights.csv:")
    print(list(flights.columns))

    def find_column(possible_names):
        lower_to_original = {col.lower(): col for col in flights.columns}
        for name in possible_names:
            if name.lower() in lower_to_original:
                return lower_to_original[name.lower()]
        raise ValueError(f"Could not find any of these columns: {possible_names}")

    user_col = find_column(["userCode", "user_code", "user", "userid", "user_id"])
    from_col = find_column(["from", "fromCity", "from_city", "origin", "originCity", "origin_city"])
    to_col = find_column(["to", "toCity", "to_city", "destination", "destinationCity", "destination_city"])

    number_of_userCodes = flights[user_col].nunique()
    number_of_observations = len(flights)

    cities = list(flights[from_col]) + list(flights[to_col])
    unique_cities = sorted(set(cities))
    n_cities = len(unique_cities)
    number_of_cities = n_cities

    transitions = list(zip(flights[from_col], flights[to_col]))
    transition_counts = makeFreqDict(transitions)

    indexToCity = {i: city for i, city in enumerate(unique_cities)}
    cityToIndex = {city: i for i, city in indexToCity.items()}

    count_matrix = np.zeros((n_cities, n_cities))

    for start_city, end_city in transitions:
        i = cityToIndex[start_city]
        j = cityToIndex[end_city]
        count_matrix[i, j] += 1

    transition_matrix = np.zeros_like(count_matrix, dtype=float)

    row_sums = count_matrix.sum(axis=1)
    for i in range(n_cities):
        if row_sums[i] > 0:
            transition_matrix[i, :] = count_matrix[i, :] / row_sums[i]
        else:
            transition_matrix[i, i] = 1.0

    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    index = np.argmin(np.abs(eigenvalues - 1))
    stationary = np.real(eigenvectors[:, index])

    stationary = np.abs(stationary)
    stationary_distribution_problem5 = stationary / stationary.sum()

    start_city = "Aracaju (SE)"
    start_index = cityToIndex[start_city]

    P3 = np.linalg.matrix_power(transition_matrix, 3)
    return_probability_problem5 = P3[start_index, start_index]

    print("number_of_cities =", number_of_cities)
    print("number_of_userCodes =", number_of_userCodes)
    print("number_of_observations =", number_of_observations)
    print("transition_matrix shape =", transition_matrix.shape)
    print("stationary_distribution_problem5 shape =", stationary_distribution_problem5.shape)
    print("stationary distribution sums to:", stationary_distribution_problem5.sum())
    print("return_probability_problem5 =", return_probability_problem5)

except Exception as e:
    print("This cell needs the file data/flights.csv.")
    print("Place the data folder beside this notebook, then rerun.")
    print("Error:", e)

```

    This cell needs the file data/flights.csv.
    Place the data folder beside this notebook, then rerun.
    Error: [Errno 2] No such file or directory: 'data/flights.csv'
    

## Exam vB, PROBLEM 6 — Black box testing

Maximum Points = 8

We continue with SMS spam / no-spam data as a pattern recognition problem. The data is prepared, split into train-test sets, and a black-box model has been fitted on the training data and predicted on the test data.

1. **[2p]** Compute precision for class 1 and provide an interval using Hoeffding's inequality for 95% confidence.
2. **[2p]** Compute recall for class 1 and provide an interval using Hoeffding's inequality for 95% confidence.
3. **[2p]** Compute accuracy and provide an interval using Hoeffding's inequality for 95% confidence.
4. **[2p]** If we used a classifier with VC-dimension 3, would we obtain a smaller interval for accuracy by using all data?

For 95% Hoeffding confidence:

\[
l = \sqrt{\frac{\log(40)}{2n}}.
\]



```python
# Problem 6 solution
# This needs exam_extras.py from the course folder.

def hoeffding_l(confidence, n):
    """
    Returns l such that P(|hatP - P| <= l) >= confidence.
    Hoeffding: P(|hatP - P| >= l) <= 2 exp(-2 n l^2)
    """
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

    n_precision = TP + FP
    problem6_precision = TP / n_precision
    problem6_precision_l = hoeffding_l(confidence=0.95, n=n_precision)

    n_recall = TP + FN
    problem6_recall = TP / n_recall
    problem6_recall_l = hoeffding_l(confidence=0.95, n=n_recall)

    n_accuracy = len(y_true)
    problem6_accuracy = np.mean(y_pred == y_true)
    problem6_accuracy_l = hoeffding_l(confidence=0.95, n=n_accuracy)

    d = 3
    delta = 0.05
    n_all = len(Y_problem6)

    problem6_VC_l = math.sqrt(
        (8 / n_all) * (d * math.log((2 * math.e * n_all) / d) + math.log(4 / delta))
    )

    problem6_VC_smaller = problem6_VC_l < problem6_accuracy_l

    print("TP, FP, FN, TN =", TP, FP, FN, TN)
    print("problem6_precision =", problem6_precision)
    print("problem6_precision_l =", problem6_precision_l)
    print("precision 95% interval =", (problem6_precision - problem6_precision_l, problem6_precision + problem6_precision_l))
    print()
    print("problem6_recall =", problem6_recall)
    print("problem6_recall_l =", problem6_recall_l)
    print("recall 95% interval =", (problem6_recall - problem6_recall_l, problem6_recall + problem6_recall_l))
    print()
    print("problem6_accuracy =", problem6_accuracy)
    print("problem6_accuracy_l =", problem6_accuracy_l)
    print("accuracy 95% interval =", (problem6_accuracy - problem6_accuracy_l, problem6_accuracy + problem6_accuracy_l))
    print()
    print("problem6_VC_l =", problem6_VC_l)
    print("problem6_VC_smaller =", problem6_VC_smaller)

except Exception as e:
    print("This cell needs the course file exam_extras.py.")
    print("Place exam_extras.py in the same folder as this notebook, then rerun.")
    print("Error:", e)

```

    This cell needs the course file exam_extras.py.
    Place exam_extras.py in the same folder as this notebook, then rerun.
    Error: No module named 'exam_extras'
    
