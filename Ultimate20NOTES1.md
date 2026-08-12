```python
The "Template" Method — NO THINKING REQUIRED
Your Cheat Notebook Structure (Copy This Exactly)
Create ONE notebook with 4 clearly labeled sections:

text
SECTION 1: PROBLEM 1 (SVD) — ALWAYS copy this
SECTION 2: PROBLEM 2 (REGRESSION) — ALWAYS copy this
SECTION 3: PROBLEM 2 (COST THRESHOLDING) — ALWAYS copy this
SECTION 4: PROBLEM 3 (MARKOV CHAINS) — ALWAYS copy this

📋 SECTION 1: Problem 1 (SVD) — ONE CODE BLOCK
You copy THIS ENTIRE BLOCK for Problem 1 every time:

python
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt

# --- STEP 1: LOAD DATA (CHANGE ONLY THE FILE NAME) ---
df = pd.read_csv("data/digits.csv")  # <--- CHANGE THIS

# --- STEP 2: SEPARATE X AND y (CHANGE ONLY THE COLUMN NAME) ---
problem1_X = df.drop(columns=['target']).to_numpy()  # <--- CHANGE 'target'
problem1_y = df['target'].to_numpy()  # <--- CHANGE 'target'

# --- STEP 3: CENTER DATA (NEVER CHANGE) ---
problem1_X_centered = problem1_X - np.mean(problem1_X, axis=0)

# --- STEP 4: SVD (NEVER CHANGE) ---
U, d, Vt = linalg.svd(problem1_X_centered, full_matrices=False)
problem1_U = U
problem1_D = np.diag(d)
problem1_V = Vt.T

# --- STEP 5: EXPLAINED VARIANCE (NEVER CHANGE) ---
singular_values_squared = d**2
total_variance = np.sum(singular_values_squared)
problem1_explained_variance = np.cumsum(singular_values_squared) / total_variance

# --- STEP 6: FIND k (CHANGE ONLY THE THRESHOLD) ---
THRESHOLD = 0.90  # <--- CHANGE THIS to 0.90, 0.95, or 0.99
problem1_num_components = np.argmax(problem1_explained_variance >= THRESHOLD) + 1

# --- STEP 7: 2D PROJECTION (NEVER CHANGE) ---
problem1_scores_2d = np.dot(problem1_X_centered, problem1_V[:, :2])

# --- STEP 8: PLOT (NEVER CHANGE) ---
plt.figure(figsize=(10, 8))
plt.scatter(problem1_scores_2d[:, 0], problem1_scores_2d[:, 1], c=problem1_y, cmap='tab10', alpha=0.6)
plt.colorbar()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Plot')
plt.grid(True)
plt.show()

# --- STEP 9: IF THE TASK IS CLASSIFICATION (like June exam) ---
# COPY THE CODE FROM BELOW

k = problem1_num_components
problem1_scores_k = np.dot(problem1_X_centered, problem1_V[:, :k])

num_rows = problem1_scores_k.shape[0]
split_idx = int(num_rows * 0.8)

X_train = problem1_scores_k[:split_idx]
y_train = problem1_y[:split_idx]
X_test = problem1_scores_k[split_idx:]
y_test = problem1_y[split_idx:]

NUM_CLASSES = 10  # <--- CHANGE THIS to the number of classes
problem1_centroids = np.zeros((NUM_CLASSES, k))
for digit in range(NUM_CLASSES):
    digit_mask = (y_train == digit)
    problem1_centroids[digit] = np.mean(X_train[digit_mask], axis=0)

problem1_test_predictions = []
for test_row in X_test:
    distances = np.linalg.norm(problem1_centroids - test_row, axis=1)
    problem1_test_predictions.append(np.argmin(distances))
problem1_test_predictions = np.array(problem1_test_predictions)
problem1_test_accuracy = np.mean(problem1_test_predictions == y_test)

# --- STEP 10: IF THE TASK IS ANOMALY DETECTION (like Jan exam) ---
# COPY THE CODE FROM BELOW INSTEAD OF STEP 9

k = problem1_num_components
problem1_approximation = problem1_U[:, :k] @ problem1_D[:k, :k] @ problem1_V[:, :k].T
problem1_reconstruction_error = np.linalg.norm(problem1_X_centered - problem1_approximation, axis=1)

NUM_OUTLIERS = 100  # <--- CHANGE THIS to the number of outliers
sorted_errors = np.sort(problem1_reconstruction_error)
problem1_threshold = sorted_errors[-NUM_OUTLIERS]
problem1_outliers = problem1_X[problem1_reconstruction_error >= problem1_threshold]

```


```python
📋 SECTION 2: Problem 2 — Linear Regression (ONE CODE BLOCK)
You copy THIS ENTIRE BLOCK if the exam asks for Linear Regression:

python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- STEP 1: LOAD DATA (CHANGE ONLY THE FILE NAME) ---
df = pd.read_csv("auto.csv")  # <--- CHANGE THIS

# --- STEP 2: CLEAN DATA (CHANGE ONLY THE COLUMN NAME) ---
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')  # <--- CHANGE 'horsepower'
df_clean = df.dropna(subset=['horsepower'])  # <--- CHANGE 'horsepower'

# --- STEP 3: CHOOSE FEATURES AND TARGET (CHANGE THESE) ---
feature_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model-year']  # <--- CHANGE
target_col = 'mpg'  # <--- CHANGE

problem2_X = df_clean[feature_cols].to_numpy()
problem2_y = df_clean[target_col].to_numpy()

# --- STEP 4: SPLIT (NEVER CHANGE) ---
n = problem2_X.shape[0]
problem2_split_index = int(0.8 * n)
problem2_X_train = problem2_X[:problem2_split_index]
problem2_X_test = problem2_X[problem2_split_index:]
problem2_y_train = problem2_y[:problem2_split_index]
problem2_y_test = problem2_y[problem2_split_index:]

# --- STEP 5: STANDARDIZE (NEVER CHANGE) ---
train_mean = np.mean(problem2_X_train, axis=0)
train_std = np.std(problem2_X_train, axis=0)
problem2_X_train_standardized = (problem2_X_train - train_mean) / train_std
problem2_X_test_standardized = (problem2_X_test - train_mean) / train_std

# --- STEP 6: FIT MODEL (NEVER CHANGE) ---
model = LinearRegression()
model.fit(problem2_X_train_standardized, problem2_y_train)
problem2_beta = np.concatenate(([model.intercept_], model.coef_))
problem2_y_pred_test = model.predict(problem2_X_test_standardized)
problem2_residuals_test = problem2_y_test - problem2_y_pred_test

# --- STEP 7: METRICS (NEVER CHANGE) ---
problem2_mse_test = np.mean(problem2_residuals_test ** 2)
problem2_mae_test = np.mean(np.abs(problem2_residuals_test))
ss_residual = np.sum(problem2_residuals_test ** 2)
ss_total = np.sum((problem2_y_test - np.mean(problem2_y_test)) ** 2)
problem2_r2_test = 1 - ss_residual / ss_total

# --- STEP 8: BASELINE (NEVER CHANGE) ---
train_mean_target = np.mean(problem2_y_train)
problem2_baseline_mse_test = np.mean((problem2_y_test - train_mean_target) ** 2)
problem2_model_beats_baseline = problem2_mse_test < problem2_baseline_mse_test

# --- STEP 9: HOEFFDING (CHANGE ONLY THE BOUND) ---
BOUND = 50  # <--- CHANGE THIS
absolute_residuals = np.abs(problem2_residuals_test)
clipped = np.clip(absolute_residuals, a_min=0, a_max=BOUND)
sample_mean = np.mean(clipped)
n_test = len(clipped)
alpha = 0.05
epsilon = np.sqrt((BOUND**2 * np.log(2 / alpha)) / (2 * n_test))
problem2_lower_bound = np.clip(sample_mean - epsilon, 0.0, BOUND)
problem2_upper_bound = np.clip(sample_mean + epsilon, 0.0, BOUND)

```


```python
📋 SECTION 3: Problem 2 — Cost Thresholding (ONE CODE BLOCK)
You copy THIS ENTIRE BLOCK if the exam asks for Cost Thresholding (like Jan):

python
import numpy as np

# --- STEP 1: DEFINE COSTS (CHANGE THESE) ---
COST_TP = 80   # <--- CHANGE
COST_TN = 0    # <--- CHANGE
COST_FP = 150  # <--- CHANGE
COST_FN = 900  # <--- CHANGE

def problem2_avg_cost(y_true, y_predict_proba, threshold):
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_predict_proba) >= threshold).astype(int)
    
    tp = (y_true == 1) & (y_pred == 1)
    tn = (y_true == 0) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)
    
    individual_costs = (COST_TP * tp.astype(float) + COST_TN * tn.astype(float) + 
                        COST_FP * fp.astype(float) + COST_FN * fn.astype(float))
    return float(np.mean(individual_costs))

# --- STEP 2: FIND OPTIMAL THRESHOLD (NEVER CHANGE) ---
thresholds = np.arange(0, 1.01, 0.01)
costs = np.array([problem2_avg_cost(y_true_val, y_pred_proba_val, t) for t in thresholds])
min_cost_index = np.argmin(costs)
problem2_threshold = float(thresholds[min_cost_index])
problem2_cost_val = float(costs[min_cost_index])

# --- STEP 3: PRECISION AND RECALL (NEVER CHANGE) ---
from sklearn.metrics import precision_score, recall_score
problem2_y_pred_val = (y_pred_proba_val >= problem2_threshold).astype(int)
problem2_precision_1 = precision_score(y_true_val, problem2_y_pred_val, pos_label=1)
problem2_recall_1 = recall_score(y_true_val, problem2_y_pred_val, pos_label=1)
problem2_precision_0 = precision_score(y_true_val, problem2_y_pred_val, pos_label=0)
problem2_recall_0 = recall_score(y_true_val, problem2_y_pred_val, pos_label=0)

# --- STEP 4: HOEFFDING (CHANGE ONLY THE BOUND) ---
BOUND = 900  # <--- CHANGE THIS
y_pred_test = (y_pred_proba_test >= problem2_threshold).astype(int)
tp = (y_true_test == 1) & (y_pred_test == 1)
tn = (y_true_test == 0) & (y_pred_test == 0)
fp = (y_true_test == 0) & (y_pred_test == 1)
fn = (y_true_test == 1) & (y_pred_test == 0)
individual_costs = (COST_TP * tp.astype(float) + COST_TN * tn.astype(float) + 
                    COST_FP * fp.astype(float) + COST_FN * fn.astype(float))
avg_test_cost = float(np.mean(individual_costs))
n_test = len(y_true_test)
alpha = 0.05
epsilon = np.sqrt((BOUND**2 * np.log(2 / alpha)) / (2 * n_test))
problem2_lower_bound = avg_test_cost - epsilon
problem2_upper_bound = avg_test_cost + epsilon
```


```python
📋 TEMPLATE A: Estimate Transition Matrix FROM DATA
Use this when the exam gives you a CSV file with transitions.

python
import numpy as np
import pandas as pd

# ==========================================
# STEP 1: LOAD DATA (CHANGE THE FILE NAME)
# ==========================================
df = pd.read_csv("warehouse_transitions.csv")  # <--- CHANGE THIS

# ==========================================
# STEP 2: DEFINE STATES (CHANGE THESE)
# ==========================================
states = ['Dock', 'Sort', 'Storage', 'Packing', 'Dispatch']  # <--- CHANGE THESE

# ==========================================
# STEP 3: CREATE TRANSITION MATRIX (NEVER CHANGE)
# ==========================================
counts = pd.crosstab(df['from_zone'], df['to_zone']).reindex(index=states, columns=states, fill_value=0)
counts_array = counts.to_numpy()

# ==========================================
# STEP 4: MAKE ROWS SUM TO 1 (NEVER CHANGE)
# ==========================================
problem3_transition_matrix = counts_array / counts_array.sum(axis=1, keepdims=True)

# ==========================================
# STEP 5: PRINT TO CHECK (OPTIONAL)
# ==========================================
print(problem3_transition_matrix)
📋 TEMPLATE B: Transition Matrix is GIVEN
Use this when the exam gives you the matrix in the question text.

python
import numpy as np

# ==========================================
# STEP 1: COPY THE MATRIX FROM THE EXAM
# ==========================================
# The question will give you a table like this:
# 
#         D     S     C     M
# D     0.25  0.35  0.30  0.10
# S     0.20  0.40  0.30  0.10
# C     0.15  0.35  0.40  0.10
# M     0.00  0.00  0.00  1.00
#
# COPY THESE NUMBERS EXACTLY:

problem3_transition_matrix = np.array([
    [0.25, 0.35, 0.30, 0.10],  # <--- Row 1 (from D)
    [0.20, 0.40, 0.30, 0.10],  # <--- Row 2 (from S)
    [0.15, 0.35, 0.40, 0.10],  # <--- Row 3 (from C)
    [0.00, 0.00, 0.00, 1.00]   # <--- Row 4 (from M)
])

# ==========================================
# STEP 2: THAT'S IT! (NEVER CHANGE)
# ==========================================
📋 TEMPLATE C: N-Step Probability
Use this when the exam asks: "Probability of being in state X after N steps"

python
import numpy as np

# ==========================================
# STEP 1: DEFINE THE TRANSITION MATRIX
# ==========================================
# Use Template A or B first to get:
problem3_transition_matrix = np.array([[...]])  # <--- FROM TEMPLATE A OR B

P = problem3_transition_matrix

# ==========================================
# STEP 2: SET THE NUMBERS FROM THE EXAM
# ==========================================
N_STEPS = 5      # <--- How many steps? (5? 4? 8?)
FROM_STATE = 0   # <--- Starting state index (0 = first state)
TO_STATE = 2     # <--- Target state index (2 = third state)

# ==========================================
# STEP 3: CALCULATE (NEVER CHANGE)
# ==========================================
P_n = np.linalg.matrix_power(P, N_STEPS)
problem3_prob_after_n_steps = float(P_n[FROM_STATE, TO_STATE])

# ==========================================
# EXAMPLE:
# If question says "Starting from Downtown (index 0), 
# probability of being in Countryside (index 2) after 5 steps"
# Then: N_STEPS=5, FROM_STATE=0, TO_STATE=2
# ==========================================
State Index Reference:

Index	State
0	First state (Dock / Downtown / etc.)
1	Second state (Sort / Suburbs / etc.)
2	Third state (Storage / Countryside / etc.)
3	Fourth state (Dispatch / Maintenance / etc.)
📋 TEMPLATE D: Simulation
Use this when the exam asks: "Simulate N chains for M steps"

python
import numpy as np

# ==========================================
# STEP 1: DEFINE THE TRANSITION MATRIX
# ==========================================
# Use Template A or B first to get:
problem3_transition_matrix = np.array([[...]])  # <--- FROM TEMPLATE A OR B

P = problem3_transition_matrix

# ==========================================
# STEP 2: SET THE NUMBERS FROM THE EXAM
# ==========================================
SEED = 20260616      # <--- Use the seed from the exam
N_CHAINS = 20000     # <--- Number of chains to simulate
N_STEPS = 8          # <--- Number of steps per chain
START_STATE = 0      # <--- Starting state index (0 = first state)

# ==========================================
# STEP 3: RUN SIMULATION (NEVER CHANGE)
# ==========================================
rng = np.random.default_rng(SEED)
num_states = P.shape[0]

final_states = np.zeros(N_CHAINS, dtype=int)
for chain in range(N_CHAINS):
    state = START_STATE
    for step in range(N_STEPS):
        state = rng.choice(num_states, p=P[state])
    final_states[chain] = state

counts = np.zeros(num_states)
for state in final_states:
    counts[state] += 1

problem3_simulated_distribution = counts / N_CHAINS

# ==========================================
# STEP 4: PRINT TO CHECK (OPTIONAL)
# ==========================================
print(problem3_simulated_distribution)
📋 TEMPLATE E: Hitting Time (Expected Steps to Reach a State)
Use this when the exam asks: "Expected number of steps to reach state X"

python
import numpy as np

# ==========================================
# STEP 1: DEFINE THE TRANSITION MATRIX
# ==========================================
# Use Template A or B first to get:
problem3_transition_matrix = np.array([[...]])  # <--- FROM TEMPLATE A OR B

P = problem3_transition_matrix

# ==========================================
# STEP 2: SET THE TARGET STATE
# ==========================================
TARGET_STATE = 3   # <--- The absorbing state (Maintenance/Dispatch/etc.)

# ==========================================
# STEP 3: CALCULATE HITTING TIMES (NEVER CHANGE)
# ==========================================
def hitting_time(P, target_state):
    n = P.shape[0]
    idx = list(range(n))
    idx.remove(target_state)
    Q = P[np.ix_(idx, idx)]
    A = np.eye(n - 1) - Q
    b = np.ones(n - 1)
    E = np.linalg.solve(A, b)
    full_E = np.zeros(n)
    for i, state_idx in enumerate(idx):
        full_E[state_idx] = E[i]
    return full_E

E = hitting_time(P, TARGET_STATE)

# ==========================================
# STEP 4: GET ANSWER FOR A SPECIFIC START STATE
# ==========================================
START_STATE = 0   # <--- Starting state index
problem3_expected_steps = float(E[START_STATE])

# ==========================================
# EXAMPLE:
# If question says "Starting from Downtown (index 0), 
# expected steps to reach Maintenance (index 3)"
# Then: TARGET_STATE=3, START_STATE=0
# ==========================================
📋 TEMPLATE F: Irreducible + Aperiodic + Stationary Distribution
Use this when the exam asks: "Is it irreducible? Is it aperiodic? Find stationary distribution."

python
import numpy as np

# ==========================================
# STEP 1: DEFINE THE TRANSITION MATRIX
# ==========================================
# Use Template A or B first to get:
problem3_transition_matrix = np.array([[...]])  # <--- FROM TEMPLATE A OR B

P = problem3_transition_matrix

# ==========================================
# STEP 2: CHECK IRREDUCIBLE (NEVER CHANGE)
# ==========================================
def is_irreducible(P):
    n = P.shape[0]
    total = np.eye(n)
    power = np.eye(n)
    for _ in range(n-1):
        power = power @ P
        total += power
    return np.all(total > 1e-10)

problem3_is_irreducible = is_irreducible(P)   # True or False

# ==========================================
# STEP 3: CHECK APERIODIC (NEVER CHANGE)
# ==========================================
def is_aperiodic(P):
    return np.any(np.diag(P) > 0)

problem3_is_aperiodic = is_aperiodic(P)   # True or False

# ==========================================
# STEP 4: STATIONARY DISTRIBUTION (NEVER CHANGE)
# ==========================================
evals, evecs = np.linalg.eig(P.T)
idx = np.argmin(np.abs(evals - 1))
pi = np.real(evecs[:, idx])
pi = pi / np.sum(pi)

problem3_stationary_distribution = pi

# ==========================================
# STEP 5: PRINT TO CHECK (OPTIONAL)
# ==========================================
print("Irreducible:", problem3_is_irreducible)
print("Aperiodic:", problem3_is_aperiodic)
print("Stationary Distribution:", problem3_stationary_distribution)
📋 TEMPLATE G: Last State Before Absorption
Use this when the exam asks: "Probability that the last state before reaching X is Y"

python
import numpy as np

# ==========================================
# STEP 1: DEFINE THE TRANSITION MATRIX
# ==========================================
# Use Template A or B first to get:
problem3_transition_matrix = np.array([[...]])  # <--- FROM TEMPLATE A OR B

P = problem3_transition_matrix

# ==========================================
# STEP 2: SET THE TARGET STATE
# ==========================================
TARGET_STATE = 3   # <--- The absorbing state (Maintenance/Dispatch/etc.)
SUCCESS_STATE = 1  # <--- The state you want to be the "last" state

# ==========================================
# STEP 3: CALCULATE (NEVER CHANGE)
# ==========================================
# Get transient states (all except TARGET_STATE)
n = P.shape[0]
idx = list(range(n))
idx.remove(TARGET_STATE)

Q = P[np.ix_(idx, idx)]

# b: probability of going to SUCCESS_STATE then to TARGET_STATE
# b[i] = probability of going from state i to SUCCESS_STATE, then to TARGET_STATE
b = np.zeros(len(idx))
for i, state_idx in enumerate(idx):
    # probability of going from state_idx to SUCCESS_STATE, then to TARGET_STATE
    b[i] = P[state_idx, SUCCESS_STATE] * P[SUCCESS_STATE, TARGET_STATE]

# Solve: f = Q f + b
I = np.eye(len(idx))
f = np.linalg.solve(I - Q, b)

# Convert back to full state order
full_f = np.zeros(n)
for i, state_idx in enumerate(idx):
    full_f[state_idx] = f[i]

# ==========================================
# STEP 4: GET ANSWER FOR A SPECIFIC START STATE
# ==========================================
START_STATE = 2   # <--- Starting state index (2 = Countryside)
problem3_prob_last_state = float(full_f[START_STATE])

# ==========================================
# EXAMPLE:
# If question says "Starting from Countryside (index 2), 
# probability that the last state before Maintenance (index 3) is Suburbs (index 1)"
# Then: TARGET_STATE=3, SUCCESS_STATE=1, START_STATE=2
# ==========================================
📝 Summary: Which Template to Use
Exam Question	Template
"Load data and estimate transition matrix"	Template A
"Given the transition matrix..."	Template B
"Probability of being in state X after N steps"	Template C
"Simulate N chains for M steps"	Template D
"Expected number of steps to reach state X"	Template E
"Is it irreducible? Is it aperiodic? Stationary distribution?"	Template F
"Probability that the last state before X is Y"	Template G
⚠️ Quick Reference: State Index
State	Index
First state (Dock/Downtown/etc.)	0
Second state (Sort/Suburbs/etc.)	1
Third state (Storage/Countryside/etc.)	2
Fourth state (Dispatch/Maintenance/etc.)	3

```
