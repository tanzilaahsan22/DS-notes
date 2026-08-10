# Assignment 1 for Course 1MS041
Make sure you pass the `# ... Test` cells and
 submit your solution notebook in the corresponding assignment on the course website. You can submit multiple times before the deadline and your highest score will be used.

---
## Assignment 1, PROBLEM 1
Maximum Points = 3



Given that you are being introduced to data science it is important to bear in mind the true costs of AI, a highly predictive family of algorithms used in data engineering sciences:

Read the 16 pages of [ai-anatomy-publication.pdf](http://www.anatomyof.ai/img/ai-anatomy-publication.pdf) with the highly detailed [ai-anatomy-map.pdf](https://anatomyof.ai/img/ai-anatomy-map.pdf) of [https://anatomyof.ai/](https://anatomyof.ai/), "Anatomy of an AI System" By Kate Crawford and Vladan Joler (2018).  The first problem in ASSIGNMENT 1 is a trivial test of your reading comprehension.


Answer whether each of the following statements is `True` or `False` *according to the authors* by appropriately replacing `Xxxxx` coresponding to `TruthValueOfStatement0a`, `TruthValueOfStatement0b` and `TruthValueOfStatement0c`, respectively, in the next cell to demonstrate your reading comprehension.

1. `Statement0a =` *Each small moment of convenience (provided by Amazon's Echo) – be it answering a question, turning on a light, or playing a song – requires a vast planetary network, fueled by the extraction of non-renewable materials, labor, and data.*
2. `Statement0b =` *The Echo user is simultaneously a consumer, a resource, a worker, and a product* 
3. `Statement0c =` *Many of the assumptions about human life made by machine learning systems are narrow, normative and laden with error. Yet they are inscribing and building those assumptions into a new world, and will increasingly play a role in how opportunities, wealth, and knowledge are distributed.*


```python

# Replace Xxxxx with True or False; Don't modify anything else in this cell!

TruthValueOfStatement0a = True

TruthValueOfStatement0b = True

TruthValueOfStatement0c = True
```

---
#### Local Test for Assignment 1, PROBLEM 1
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.


```python
# Test locally to ensure an acceptable answer, True or False
try:
    assert(isinstance(TruthValueOfStatement0a, bool)) 
    assert(isinstance(TruthValueOfStatement0b, bool)) 
    assert(isinstance(TruthValueOfStatement0c, bool))
except:
    print("Try again. You are not writing True or False for your answers.")
else:
    print("Good, you have answered either True or False. Hopefully they are the correct answers!")
```

    Good, you have answered either True or False. Hopefully they are the correct answers!
    

---
## Assignment 1, PROBLEM 2
Maximum Points = 2


Evaluate the following cells by replacing `X` with the right command-line option to `head` in order to find the first four lines of the csv file `data/final.csv`

```
%%sh
man head

HEAD(1)                   BSD General Commands Manual                  HEAD(1)

NAME
     head -- display first lines of a file

SYNOPSIS
     head [-n count | -c bytes] [file ...]

DESCRIPTION
     This filter displays the first count lines or bytes of each of the speci-
     fied files, or of the standard input if no files are specified.  If count
     is omitted it defaults to 10.

     If more than a single file is specified, each file is preceded by a
     header consisting of the string ``==> XXX <=='' where ``XXX'' is the name
     of the file.

EXIT STATUS
     The head utility exits 0 on success, and >0 if an error occurs.

SEE ALSO
     tail(1)

HISTORY
     The head command appeared in PWB UNIX.

BSD                              June 6, 1993                              BSD
```


```sh
%%sh
head -4 data/final.csv
```

    region,municipality,district,party,votes
    Blekinge län,Karlshamn,0 - Centrala Asarum,S,519
    Blekinge län,Karlshamn,0 - Centrala Asarum,SD,311
    Blekinge län,Karlshamn,0 - Centrala Asarum,M,162
    


```python
line_1_final = "region,municipality,district,party,votes"
line_2_final = "Blekinge län,Karlshamn,0 - Centrala Asarum,S,519"
```

---
#### Local Test for Assignment 1, PROBLEM 2
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.


```python
# Evaluate this cell locally to make sure you have the answer as a string
try:
    assert(type(line_1_final) == str)
    print("Good! You have answered as a string for line 1. Hopefully it is the correct!")
except AssertionError:
    print("Try Again. You should answer with a string.")
try:
    assert(type(line_2_final) == str)
    print("Good! You have answered as a string for line 2. Hopefully it is the correct!")
except AssertionError:
    print("Try Again. You should answer with a string.")
```

    Good! You have answered as a string for line 1. Hopefully it is the correct!
    Good! You have answered as a string for line 2. Hopefully it is the correct!
    

---
## Assignment 1, PROBLEM 3
Maximum Points = 3


In this assignment the goal is to parse the `final.csv` file from the previous problem.

1. Read the file `data/final.csv` and parse it using the `csv` package and store the result as follows

the `header` variable contains a list of names all as strings

the `data` variable should be a list of lists containing all the rows of the csv file


```python
import pandas as pd
df = pd.read_csv('data/final.csv')
header = list(df.columns.astype(str))
data = df.astype(str).values.tolist()
```

---
#### Local Test for Assignment 1, PROBLEM 3
Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.


```python

# Evaluate this cell locally to make sure you have the answer in the right format
try:
    assert(type(header) == list)
    print("Good! You have the header as a list. Hopefully it is the correct!")
except AssertionError:
    print("Try Again. You should answer with a list.")
try:
    types = set([type(a) for a in header])
    assert((len(types) == 1) and (list(types)[0] == str))
    print("Good! You have the header as a list of strings. Hopefully it is the correct!")
except AssertionError:
    print("Try Again. You should answer with a list of strings.")
try:
    assert(type(data) == list)
    print("Good! You have the data as a list. Hopefully it is the correct!")
except AssertionError:
    print("Try Again. You should answer with a list.")
try:
    types = set([type(a) for a in data])
    assert((len(types) == 1) and (list(types)[0] == list))
    print("Good! You have the data as a list of lists. Hopefully it is the correct!")
except AssertionError:
    print("Try Again. You should answer with a list of lists.")
try:
    types = set(sum([[type(d) for d in t] for t in data[:1]],[]))
    assert((len(types) == 1) and (list(types)[0] == str))
    print("Good! You have the data as a list of lists of strings. Hopefully it is the correct!")
except AssertionError:
    print("Try Again. You should answer with a list of lists of strings.")
```

    Good! You have the header as a list. Hopefully it is the correct!
    Good! You have the header as a list of strings. Hopefully it is the correct!
    Good! You have the data as a list. Hopefully it is the correct!
    Good! You have the data as a list of lists. Hopefully it is the correct!
    Good! You have the data as a list of lists of strings. Hopefully it is the correct!
    

---
## Assignment 1, PROBLEM 4
Maximum Points = 8


## Students passing exam (Sample exam problem)
Let's say we have an exam question which consists of $10$ yes/no questions. 
From past performance of similar students, a randomly chosen student will know the correct answer to $N \sim \text{binom}(10,6/10)$ questions. Furthermore, we assume that the student will guess the answer with equal probability to each question they don't know the answer to, i.e. given $N$ we define $Z \sim \text{binom}(10-N,1/2)$ as the number of correctly guessed answers. Define $Y = N + Z$, i.e., $Y$ represents the number of total correct answers.

We are interested in setting a deterministic threshold $T$, i.e., we would pass a student at threshold $T$ if $Y \geq T$. Here $T \in \{0,1,2,\ldots,10\}$.

1. [5p] For each threshold $T$, compute the probability that the student *knows* less than $5$ correct answers given that the student passed, i.e., $N < 5$. Put the answer in `problem11_probabilities` as a list.
2. [3p] What is the smallest value of $T$ such that if $Y \geq T$ then we are 90\% certain that $N \geq 5$?


```python

# Hint the PMF of N is p_N(k) where p_N is
from scipy.special import binom as binomial
p = 6/10
p_N = lambda k: binomial(10,k)*((1-p)**(10-k))*((p)**k)
```


```python

# Part 1: 
problem11_probabilities = [
    0.16623861760000005,
    0.16623853222282572,
    0.16623511712151573,
    0.16617364051358016,
    0.1655173254906334,
    0.16089403080809417,
    0.1444527989145165,
    0.11223025722358064,
    0.0732121231532333,
    0.04058456420898439,
    0.019727706909179698,
]
```


```python

# Part 2:
problem12_T = 8
```

---
## Assignment 1, PROBLEM 5
Maximum Points = 8


## Concentration of measure (Sample exam problem)

As you recall, we said that concentration of measure was simply the phenomenon where we expect that the probability of a large deviation of some quantity becoming smaller as we observe more samples: [0.4 points per correct answer]

1. Which of the following will exponentially concentrate, i.e. for some $C_1,C_2,C_3,C_4 $ 
$$
    P(Z - \mathbb{E}[Z] \geq \epsilon) \leq C_1 e^{-C_2 n \epsilon^2} \vee C_3 e^{-C_4 n (\epsilon+1)} \enspace .
$$

    1. The empirical variance of i.i.d. random variables with finite mean?
    2. The empirical variance of i.i.d. sub-Gaussian random variables?
    3. The empirical variance of i.i.d. sub-Exponential random variables?
    4. The empirical mean of i.i.d. sub-Gaussian random variables?
    5. The empirical mean of i.i.d. sub-Exponential random variables?
    6. The empirical mean of i.i.d. random variables with finite variance?
    7. The empirical third moment of i.i.d. random variables with finite sixth moment?
    8. The empirical fourth moment of i.i.d. sub-Gaussian random variables?
    9. The empirical mean of i.i.d. deterministic random variables?
    10. The empirical tenth moment of i.i.d. Bernoulli random variables?

2. Which of the above will concentrate in the weaker sense, that for some $C_1$
$$
    P(Z - \mathbb{E}[Z] \geq \epsilon) \leq \frac{C_1}{n \epsilon^2}?
$$


```python

problem3_answer_1 = [2, 4, 5, 9, 10]
```


```python
problem3_answer_2 = [2, 3, 4, 5, 6, 7, 8, 9, 10]
```
