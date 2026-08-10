

# Page 1

ReExamJune
January 9, 2026
1
ReExam 13th of June 2022 for the course 1MS041 (Introduction
to Data Science / Introduktion till dataanalys)
1. Fill in your anonymous exam code in the cell below.
2. Complete the Problems by following instructions.
3. When done, submit this file with your solutions saved, following the instruction sheet.
2
EXAM vB, PROBLEM 1, POINTS 8
2.1
Probability warmup
Let’s say we have an exam question which consists of 20 yes/no questions. From past performance of
similar students, a randomly chosen student will know the correct answer to 𝑁∼binom(20, 11/20)
questions. Furthermore, we assume that the student will guess the answer with equal probability
to each question they don’t know the answer to, i.e. given 𝑁we define 𝑍∼binom(20 −𝑁, 1/2) as
the number of correctly guessed answers. Define 𝑌= 𝑁+ 𝑍, i.e., 𝑌represents the number of total
correct answers.
We are interested in setting a deterministic threshold 𝑇, i.e., we would pass a student at threshold
𝑇if 𝑌≥𝑇. Here 𝑇∈{0, 1, 2, … , 20}.
1. [3p] Produce a simulation of 1000 students. Hint: Simulate 𝑁first then simulate 𝑌∣𝑁and
add the results. Numpy has numpy.random.binomial which you can simulate from.
2. [3p] For each threshold 𝑇, produce a simulation as above and estimate the probability that
the student knows less than 10 correct answers given that the student passed, i.e., 𝑁< 10.
Put the answer in problem11_probabilities as a list.
3. [2p] What is the smallest value of 𝑇such that if 𝑌≥𝑇then we are 90% certain that 𝑁≥10?
[ ]: # EXAM vB, PROBLEM 1, POINTS 8
# Part 1:
problem1_1000_samples = XXX
[ ]: # EXAM vB, PROBLEM 1, POINTS 8
# Part 2:
# replace XXX to represent P(N < 10) for T = [0,1,2,...,20], i.e. your answer␣
↪should be a list
# of length 21.
1


# Page 2

problem1_probabilities = [XXX,XXX,...,XXX]
[ ]: # EXAM vB, PROBLEM 1, POINTS 8
# Part 3: Give an integer between 0 and 20 which is the answer to 2.
problem1_T = XXX
3
EXAM vB, PROBLEM 2, POINTS 8
3.1
Random variable generation and transformation
1. [3p] Using the Accept-Reject sampler (Algorithm 1 in lecture notes) with sampling den-
sity given by the uniform density on the 10 dimensional box [−1, 1]𝑑, i.e. Uniform([−1, 1]𝑑),
generate 100 samples from the uniform distribution on the unit ball in 10 dimensions. Hint,
to generate a sample from Uniform([−1, 1]𝑑) just generate each coordinate as an i.i.d. sam-
ple from Uniform([−1, 1]). Hint since both the sampling density and the target density are
constant functions the ratio 𝑟(𝑥) is either 1 or 0.
2. [2p] How many proposals do you need to produce in order to get 100 samples?
3. [3p] Using Theorem 10.10 in the lecture notes generate 100 samples from the uniform
distribution on the unit ball in 10 dimensions.
[ ]: # EXAM vB, PROBLEM 2, POINTS 8
# Part 1
def problem2_accept_reject(n_samples=1):
"""
Produces samples from the 10d unit ball using an Accept-Reject
sampler with the Uniform([-1,1]^d) as the sampling distribution.
Parameters
-------------
n_samples : an integer denoting how many samples to be produced
Returns
--------------
out : a numpy array of the shape (100,10)
"""
retVal = []
n_iterations=0
while(len(retVal)<n_samples):
x = XXX
if (XXX):
retVal.append(x)
n_iterations+=1
2


# Page 3

print(n_iterations)
return np.array(retVal)
[ ]: # EXAM vB, PROBLEM 2, POINTS 8
# Part 2
problem2_accept_reject_n_iterations = XXX
[ ]: # EXAM vB, PROBLEM 2, POINTS 8
# Part 3
def problem2_theorem10_10(n_samples=1,d=10):
"""
Produces samples from the 10d unit ball using Theorem 10.10.
Parameters
-------------
n_samples : an integer denoting how many samples to be produced
Returns
--------------
out : a numpy array of the shape (100,10)
"""
Theta = XXX
R = XXX
return R*Theta
[146]: # EXAM vB, Test 2, POINTS 8
# Try running the following code to see that you produce
# output of the correct shape. If it says XXX does not have
# attribute 'shape' you are probably not returning a numpy array
# for whatever reason.
try:
problem2_test_1 = problem2_accept_reject(100)
print(problem2_test_1.shape)
except Exception as e:
print(e)
try:
problem2_test_2 = problem2_theorem10_10(100)
print(problem2_test_2.shape)
3


# Page 4

except Exception as e:
print(e)
36313
(100, 10)
(100, 10)
4
EXAM vB, PROBLEM 3, POINTS 8
4.1
Concentration of measure
As you recall, we said that concentration of measure was simply the phenomenon where we expect
that the probability of a large deviation of some quantity becoming smaller as we observe more
samples: [0.4 points per correct answer]
1. Which of the following will exponentially concentrate, i.e. for some $C_1,C_2,C_3,C_4 $
𝑃(𝑍−𝔼[𝑍] ≥𝜖) ≤𝐶1𝑒−𝐶2𝑛𝜖2 ∧𝐶3𝑒−𝐶4𝑛(𝜖+1) .
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
2. Which of the above will concentrate in the weaker sense, that for some 𝐶1
𝑃(𝑍−𝔼[𝑍] ≥𝜖) ≤𝐶1
𝑛𝜖2 ?
Note: If jupyter for whatever reason, renders the options above as A,B,C instead of 1,2,3, you need
to input the answers as numbers anyways A=>1, B=>2 etc.
[100]: # EXAM vB, PROBLEM 3, POINTS 8
# Answers to part 1, which of the alternatives exponentially concentrate,␣
↪answer as a list
# i.e. [1,4,5] that is example 1, 4, and 5 concentrate
problem3_answer_1 = ['XXX']
[101]: # EXAM vB, PROBLEM 3, POINTS 8
# Answers to part 2, which of the alternatives concentrate in the weaker sense,␣
↪answer as a list
4


# Page 5

# i.e. [1,4,5] that is example 1, 4, and 5 concentrate
problem3_answer_2 = ['XXX']
5
EXAM vB, PROBLEM 4, POINTS 8
5.1
SMS spam filtering [8p]
In the following problem we will explore SMS spam texts. The dataset is the SMS Spam Collection
Dataset and we have provided for you a way to load the data. If you run the appropriate cell
below, the result will be in the spam_no_spam variable. The result is a list of tuples with the
first position in the tuple being the SMS text and the second being a flag 0 = not spam and 1 =
spam.
1. [3p] Let 𝑋be the random variable that represents each SMS text (an entry in the list), and
let 𝑌represent whether text is spam or not i.e. 𝑌∈{0, 1}. Thus ℙ(𝑌= 1) is the probability
that we get a spam. The goal is to estimate:
ℙ(𝑌= 1|”free” or ”prize” is in 𝑋) .
That is, the probability that the SMS is spam given that “free” or “prize” occurs in the SMS.
Hint: it is good to remove the upper/lower case of words so that we can also find “Free” and
“Prize”; this can be done with text.lower() if text a string.
2. [3p] Provide a “90%” interval of confidence around the true probability. I.e. use the Hoeffding
inequality to obtain for your estimatê 𝑃of the above quantity. Find 𝑙> 0 such that the
following holds:
ℙ(̂𝑃−𝑙≤𝔼[̂𝑃] ≤̂ 𝑃+ 𝑙) ≥0.9 .
3. [2p] Repeat the two exercises above for “free” appearing twice in the SMS.
[ ]: # EXAM vB, PROBLEM 4, POINTS 8
# Run this cell to get the SMS text data
from exam_extras import load_sms
spam_no_spam = load_sms()
[ ]: # EXAM vB, PROBLEM 4, POINTS 8
# fill in the estimate for part 1 here (should be a number between 0 and 1)
problem4_hatP = XXX
[ ]: # EXAM vB, PROBLEM 4, POINTS 8
# fill in the calculated l from part 2 here
problem4_l = XXX
[ ]: # EXAM vB, PROBLEM 4, POINTS 8
5


# Page 6

# fill in the estimate for hatP for the double free question in part 3 here␣
↪(should be a number between 0 and 1)
problem4_hatP2 = XXX
[ ]: # EXAM vB, PROBLEM 4, POINTS 8
# fill in the estimate for l for the double free question in part 3 here
problem4_l2 = XXX
6
EXAM vB, PROBLEM 5, POINTS 8
6.1
Black box testing
In the following problem we will continue with our SMS spam / nospam data. This time we will
try to approach the problem as a pattern recognition problem. For this particular problem I have
provided you with everything – data is prepared, split into train-test sets and a black-box model
has been fitted on the training data and predicted on the test data. Your goal is to calculate test
metrics and provide guarantees for each metric.
1. [2p] Compute precision for class 1 (see notes 8.3.2 for definition), then provide an interval
using Hoeffding’s inequality for a 95% confidence.
2. [2p] Compute recall for class 1(see notes 8.3.2 for definition), then provide an interval using
Hoeffding’s inequality for a 95% interval.
3. [2p] Compute accuracy (0-1 loss), then provide an interval using Hoeffding’s inequality for a
95% interval.
4. [2p] If we would have used a classifier with VC-dimension 3, would we have obtained a smaller
interval for accuracy by using all data?
[1]: # EXAM vB, PROBLEM 5, POINTS 8
# The code below will load data, split the data into train and test and run a␣
↪"black box" algorithm on it
# the result of the "black box" is stored in predictions_problem6, the true␣
↪values will be stored in
# Y_test_problem6
import exam_extras
from exam_extras import load_sms_problem6
X_problem6, Y_problem6 = load_sms_problem6()
X_train_problem6,X_test_problem6,Y_train_problem6,Y_test_problem6 = exam_extras.
↪train_test_split(X_problem6,Y_problem6)
predictions_problem6 = exam_extras.
↪knn_predictions(X_train_problem6,Y_train_problem6,X_test_problem6,k=4)
[ ]: # EXAM vB, PROBLEM 5, POINTS 8
6


# Page 7

# Compute the precision of predictions_problem6 with respect to Y_test_problem6
problem6_precision = XXX
[ ]: # EXAM vB, PROBLEM 5, POINTS 8
# Compute the interval length l of precision of predictions_problem6 with␣
↪respect to Y_test_problem6, with the same definition of l as in problem 4
problem6_precision_l = XXX
[ ]: # EXAM vB, PROBLEM 5, POINTS 8
# Repeat the same procedure but for recall
problem6_recall = XXX
[ ]: # EXAM vB, PROBLEM 5, POINTS 8
problem6_recall_l = XXX
[ ]: # EXAM vB, PROBLEM 5, POINTS 8
# Repeat the same procedure but for accuracy or 0-1 loss
problem6_accuracy = XXX
[ ]: # EXAM vB, PROBLEM 5, POINTS 8
problem6_accuracy_l = XXX
[ ]: # EXAM vB, PROBLEM 5, POINTS 8
# Below you will calculate the interval parameter l for a classifier running on␣
↪all data with a VC dimension of 3
# put the value in problem6_VC_l and answer problem_VC_smaller as True if the␣
↪interval is smaller than the test-accuracy above
# if not answer False. Make sure you replace XXX with something even if you␣
↪only answer one of them.
problem6_VC_l = XXX # number
problem6_VC_smaller = XXX #True / False
7
