

# Page 1

Exam
January 9, 2026
1
ReExam 24th of August 2022, 8.00-13.00 for the course 1MS041
(Introduction to Data Science / Introduktion till dataanalys)
1.1
Instructions:
1. Complete the problems by following instructions.
2. When done, submit this file with your solutions saved, following the instruction sheet.
This exam has 5 problems each worth 8 points for a total of 40 points, to pass you need 15 points.
1.2
Some general hints and information:
• Some problems are similar to the exam in January but changed.
• Try to answer all questions even if you are uncertain.
• Comment your code, so that if you get the wrong answer I can understand how you thought
this can give you some points even though the code does not run.
• Follow the instruction sheet rigorously.
• This exam has no anonymous exam ID due to a technical issue, however this does not mean
that the exam is not anonymous. The grading system will automatically download all the
exams from Studium and it is at this stage that they are anonymized by a randomized ID.
• If there are any questions, please ask the exam guards, they will escalate it to me if necessary.
• I (Benny) will visit the exam room at around 10:30 to see if there are any questions.
1.3
Finally some rules:
• You may not communicate with others during the exam, for example:
– You cannot ask for help in Stack-Overflow or other such help forums during the Exam.
– You may not use encrypted communications
– Your on-line and off-line activity is being monitored according to the examination rules.
1.4
Good luck!
1.5
Exam vB, PROBLEM 1
Maximum Points = 8
1


# Page 2

1.6
Probability warmup
Let’s say we have an exam question which consists of 50 yes/no questions. From past performance
of similar students, a randomly chosen student will know the correct answer to 𝑁∼binom(50, 0.8)
questions. Furthermore, we assume that the student will guess the answer with equal probability
to each question they don’t know the answer to, i.e. given 𝑁we define 𝑍∼binom(50 −𝑁, 1/2) as
the number of correctly guessed answers. Define 𝑌= 𝑁+ 𝑍, i.e., 𝑌represents the number of total
correct answers.
We are interested in setting a deterministic threshold 𝑇, i.e., we would pass a student at threshold
𝑇if 𝑌≥𝑇. Here 𝑇∈{0, 1, 2, … , 50}.
1. [3p] Produce a simulation of 1000 students. Hint: Simulate 𝑁first then simulate 𝑌∣𝑁and
add the results. Numpy has numpy.random.binomial which you can simulate from.
2. [3p] For each threshold 𝑇, produce a simulation as above and estimate the probability that
the student knows less than 40 correct answers given that the student passed, i.e., 𝑁< 40.
Put the answer in problem11_probabilities as a list.
3. [2p] What is the smallest value of 𝑇such that if 𝑌≥𝑇then we are 90% certain that 𝑁≥40?
[ ]: # Part 1:
problem1_1000_samples = XXX
[ ]: # Part 2:
# replace XXX to represent P(N < 40) for T = [0,1,2,...,50], i.e. your answer␣
↪should be a list
# of length 51.
problem1_probabilities = [XXX,XXX,...,XXX]
[ ]: # Part 3: Give an integer between 0 and 50 which is the answer to 2.
problem1_T = XXX
1.7
Exam vB, PROBLEM 2
Maximum Points = 8
In many areas of data science and machine learning we need to produce random samples in different
ways. This can be done to compute diﬀicult integrals or validate algorithms.
1. [2p] Produce 1000 samples from the distribution below using inversion sampling
𝐹[𝑥] =
⎧
{
⎨
{
⎩
0,
𝑥≤0
sin(𝑥),
0 < 𝑥< 𝜋/2
1,
𝑥≥𝜋/2
and show your result with a histogram “You can use sagemath function histogram, or
matplotlib.pyplot hist”. Also what is the true density? Provide a plot of the true density
between 0 and 𝜋/2.
2. [3p] Consider a random variable 𝑋∼𝐹sampled from distribution 𝐹. Your goal is to estimate
𝐸[𝑋]. Do this by producing 1000 different experiments, each sampling 1000 samples from 𝑋
and compute the empirical mean. Provide the 0.025 and the 0.975 quantile of the experiments.
2


# Page 3

3. [3p] Use Hoeffdings inequality to produce a 95% confidence interval for the estimated mean
above?
[ ]: # put your samples in the variable samples
samples = XXX
[ ]: # Produce 1000 experiments, in which each experiment you draw
# 1000 samples from F. Store the value of the empirical mean of each
# experiment and compute the 0.025 and the 0.975 quantiles
means = XXX # the computed empirical means, should be a list of length 1000
quantile_0025 = XXX # the 0.025 quantile
quantile_0975 = XXX # the 0.975 quantile
[ ]: # Put your interval in the form
l_edge = XXX # The left edge of the interval
r_edge = XXX # The right edge of the interval
print("Confidence interval around the mean is [%.2f,%.2f]" % (l_edge,r_edge))
1.8
Exam vB, PROBLEM 3
Maximum Points = 8
1.9
Concentration of measure
As you recall, we said that concentration of measure was simply the phenomenon where we expect
that the probability of a large deviation of some quantity becoming smaller as we observe more
samples: [8/22 points per correct answer]
1. Which of the following will exponentially concentrate, i.e. for some $C_1,C_2,C_3,C_4 $
𝑃(𝑍−𝔼[𝑍] ≥𝜖) ≤𝐶1𝑒−𝐶2𝑛𝜖2 ∧𝐶3𝑒−𝐶4𝑛(𝜖+1) .
1. The empirical mean of i.i.d. sub-Gaussian random variables?
2. The empirical mean of i.i.d. sub-Exponential random variables?
3. The empirical mean of i.i.d. bounded random variables?
4. The empirical variance of i.i.d. bounded random variables?
5. The empirical mean of i.i.d. random variables with finite variance?
6. The empirical variance of i.i.d. random variables with finite variance?
7. The empirical variance of i.i.d. sub-Gaussian random variables?
8. The empirical third moment of i.i.d. bounded random variables?
9. The empirical fourth moment of i.i.d. sub-Gaussian random variables?
10. The empirical mean of i.i.d. deterministic random variables?
11. The empirical tenth moment of i.i.d. Bernoulli random variables?
2. Which of the above will concentrate in the weaker sense, that for some 𝐶1
𝑃(𝑍−𝔼[𝑍] ≥𝜖) ≤𝐶1
𝑛𝜖2 ?
3


# Page 4

[ ]: # Answers to part 1, which of the alternatives exponentially concentrate,␣
↪answer as a list
# i.e. [1,4,5] that is example 1, 4, and 5 concentrate
problem3_answer_1 = [XXX]
[ ]: # Answers to part 2, which of the alternatives concentrate in the weaker sense,␣
↪answer as a list
# i.e. [1,4,5] that is example 1, 4, and 5 concentrate
problem3_answer_2 = [XXX]
1.10
Exam vB, PROBLEM 4
Maximum Points = 8
In this problem you will be working with a text file a_sequence.txt, found in the data folder.
This contains a sequence of numbers that are observations of a Markov chain. The goal of this
exercise is to analyze this sequence in different ways.
1. [2p] Take the file a_sequence.txt and load it as a list of integers. Use bash or something to
figure out how to parse the file.
2. [2p] Define a Markov chain from this list of integers
1. What are the states?
2. How many states are there?
3. [2p] Estimate the transition probability of going from state 42 to state 16?
4. [2p] Find the transition matrix 𝑃and compute the matrix power 𝑃10𝑣where 𝑣= (1, 0, … , 0).
[ ]: # Read the file a_sequence.txt and load it as a list of integers.
# Put your result in the variable "numbers"
numbers = XXX
[ ]: # Construct a Markov chain of this list of integers, that is.
# EXPLAIN in text what are the states are and what
# the transition probabilities mean.
#---------Put your explanation between the lines-------------
#------------------------------------------------------------
[ ]: # put the number of states in the variable n_states
n_states = XXX
# Now fill in the states, stored as a sorted list of integers
states = XXX
4


# Page 5

[ ]: # Estimate the transition probability of going from $42$ to $16$.
# You can use the below function if you want
[ ]: def makeFreqDict(myDataSeq, one = int(1)):
'''Make a frequency mapping out of a sequence of data - list, array, str.
Param myDataList, a list of data.
Return a dictionary mapping each unique data value to its frequency count.
↪'''
freqDict = {} # start with an empty dictionary
for res in myDataSeq:
if res in freqDict: # the data value already exists as a key
freqDict[res] = freqDict[res] + one #int(1) # add 1 to the count
else: # the data value does not exist as a key value
# add a new key-value pair for this new data value, frequency 1
freqDict[res] = one
return freqDict # return the dictionary created
[ ]: # Put your answer here for the transition probability
transition_probability = XXX
[ ]: # Fill in the transition matrix P as a numpy array of
# shape (n_states x n_states)
# Make sure it is a transition matrix by checking the column sum
P =XXX
# If our initial vector is
v = np.zeros(n_states)
v[0] = 1
# What is P^10 v
steady_state_v = XXX
1.11
Exam vB, PROBLEM 5
Maximum Points = 8
1.12
SMS spam filtering [8p]
In the following problem we will explore SMS spam texts. The dataset is the SMS Spam Collection
Dataset and we have provided for you a way to load the data. If you run the appropriate cell
5


# Page 6

below, the result will be in the spam_no_spam variable. The result is a list of tuples with the
first position in the tuple being the SMS text and the second being a flag 0 = not spam and 1 =
spam.
1. [3p] Let 𝑋be the random variable that represents each SMS text (an entry in the list), and
let 𝑌represent whether text is spam or not i.e. 𝑌∈{0, 1}. Thus ℙ(𝑌= 1) is the probability
that we get a spam. The goal is to estimate:
ℙ(𝑌= 1|”free” or ”prize” is in 𝑋) .
That is, the probability that the SMS is spam given that “free” or “prize” occurs in the SMS.
(This is precision) Hint: it is good to remove the upper/lower case of words so that we can
also find “Free” and “Prize”; this can be done with text.lower() if text a string.
2. [3p] Estimate the probability that the word “free” or “prize” is in the text given that it is
spam. (This is recall) I.e. estimate
ℙ(”free” or ”prize” is in 𝑋∣𝑌= 1) .
3. [2p] Provide a “90%” interval of confidence around the true probability from part 1. I.e. use
the Hoeffding inequality to obtain for your estimatê 𝑃. Find 𝑙> 0 such that the following
holds:
ℙ(̂𝑃−𝑙≤𝔼[̂𝑃] ≤̂ 𝑃+ 𝑙) ≥0.9 .
[ ]: # Run this cell to get the SMS text data
from exam_extras import load_sms
spam_no_spam = load_sms()
[ ]: # fill in the estimate for part 1 here (should be a number between 0 and 1)
problem5_hatP = XXX
[ ]: # fill in the estimate for hatP for the double free question in part 2 here␣
↪(should be a number between 0 and 1)
problem5_hatP2 = XXX
[ ]: # fill in the calculated l from part 3 here
problem5_l = XXX
6
