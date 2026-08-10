```python
import pandas as pd
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
```


```python
# Generate fake data (You don't need to memorize this part, it's just to make the practice work)
import pandas as pd
import numpy as np
np.random.seed(42)
math_scores = np.random.normal(65, 15, 200)
reading_scores = np.random.normal(70, 12, 200)
# If they get > 140 combined, they pass (1), else fail (0)
pass_fail = ((math_scores + reading_scores) > 140).astype(int)

# Combine into a Pandas DataFrame
df = pd.DataFrame({
    'Math_Score': math_scores,
    'Reading_Score': reading_scores,
    'Pass_Fail': pass_fail
})
```


```python
X = df[['Math_Score', 'Reading_Score']].to_numpy()
Y = df['Pass_Fail'].to_numpy()

#Calculate the mean (axis=0 means calculate mean for each column)

mean_X = np.mean(X,axis=0)

#Center the data

X_centered = X - mean_X
```


```python
U, d, Vt = linalg.svd(X_centered, full_matrices=False)
```


```python
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.7)
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.title('Student Scores: Pass vs Fail')
plt.colorbar(label='Pass/Fail (0=Fail, 1=Pass)')
plt.show()
```


```python
#Nearest Centroid Classifier

centroids = np.zero((2,2))

# Calculate centroid for Fail (0)
centroids[0] = np.mean(X[y == 0], axis=0)

# Calculate centroid for Pass (1)
centroids[1] = np.mean(X[y == 1], axis=0)

print("Centroids:\n", centroids)


```


```python
#Complete PCA/SVD Solutions - All Common Exam Questions
#Here are complete, ready-to-use solutions for ALL types of PCA/SVD problems you might encounter.

#📦 Import Everything First

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#🔷 TYPE 1: BASIC PCA (Digits Classification)
#Use when: Problem asks about classification, predicting labels, or accuracy.

# ============================================
# PROBLEM: PCA for Digit Classification
# ============================================

# ---------- PART 1: Load Data & SVD ----------
df = pd.read_csv("data/digits.csv")
problem1_X = df.iloc[:, :-1].to_numpy()
problem1_y = df.iloc[:, -1].to_numpy()

# Center the data
problem1_X_centered = problem1_X - np.mean(problem1_X, axis=0)

# Compact SVD
U, d, Vt = np.linalg.svd(problem1_X_centered, full_matrices=False)
problem1_U = U
problem1_D = np.diag(d)
problem1_V = Vt.T

# ---------- PART 2: Explained Variance ----------
singular_values_squared = d ** 2
total_variance = np.sum(singular_values_squared)
cumulative_variance = np.cumsum(singular_values_squared)
problem1_explained_variance = cumulative_variance / total_variance

# For 90% variance
problem1_num_components = np.argmax(problem1_explained_variance >= 0.90) + 1

print(f"Number of components for 90% variance: {problem1_num_components}")

# ---------- PART 3: 2D Visualization ----------
problem1_scores_2d = np.dot(problem1_X_centered, problem1_V[:, :2])

plt.figure(figsize=(12, 10))
scatter = plt.scatter(problem1_scores_2d[:, 0], problem1_scores_2d[:, 1], 
                      c=problem1_y, cmap='tab10', alpha=0.6, s=20)
plt.colorbar(scatter, label='Digit Label')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Handwritten Digits')
plt.grid(True, alpha=0.3)
plt.show()

# ---------- PART 4: Classification ----------
k = problem1_num_components
problem1_scores_k = np.dot(problem1_X_centered, problem1_V[:, :k])

# 80/20 split
n_samples = problem1_scores_k.shape[0]
split_idx = int(0.8 * n_samples)

X_train = problem1_scores_k[:split_idx]
y_train = problem1_y[:split_idx]
X_test = problem1_scores_k[split_idx:]
y_test = problem1_y[split_idx:]

# Nearest centroid classification
problem1_centroids = np.zeros((10, k))
for digit in range(10):
    mask = (y_train == digit)
    problem1_centroids[digit] = np.mean(X_train[mask], axis=0)

# Predict
problem1_test_predictions = []
for row in X_test:
    distances = np.linalg.norm(problem1_centroids - row, axis=1)
    problem1_test_predictions.append(np.argmin(distances))
problem1_test_predictions = np.array(problem1_test_predictions)

problem1_test_accuracy = np.mean(problem1_test_predictions == y_test)
print(f"Test Accuracy: {problem1_test_accuracy:.4f}")
```


```python
#🔷 TYPE 2: PCA for Anomaly Detection
#Use when: Problem asks about outliers, anomalies, or reconstruction error.

# ============================================
# PROBLEM: SVD for Anomaly Detection
# ============================================

# ---------- PART 1: Load Data & SVD ----------
df = pd.read_csv("data/SVD.csv")
problem1_X = df.to_numpy()

# SVD (NO centering for this type!)
U, d, Vt = np.linalg.svd(problem1_X, full_matrices=False)

# Extract first singular vectors
problem1_U = U
problem1_D = np.diag(d)
problem1_V = Vt.T

first_left_singular_vector = U[:, 0]
first_right_singular_vector = Vt[0, :]

# ---------- PART 2: Explained Variance ----------
singular_values_squared = d ** 2
total_variance = np.sum(singular_values_squared)
cumulative_variance = np.cumsum(singular_values_squared)
problem1_explained_variance = cumulative_variance / total_variance

# For 99% variance
problem1_num_components = np.argmax(problem1_explained_variance >= 0.99) + 1

print(f"Number of components for 99% variance: {problem1_num_components}")

# ---------- PART 3: Plot Explained Variance ----------
k_values = np.arange(1, len(problem1_explained_variance) + 1)
plt.figure(figsize=(10, 6))
plt.plot(k_values, problem1_explained_variance, 'bo-', linewidth=2)
plt.axhline(y=0.99, color='r', linestyle='--', label='99% Threshold')
plt.axvline(x=problem1_num_components, color='g', linestyle='--', 
            label=f'k = {problem1_num_components}')
plt.xlabel('Number of Components (k)')
plt.ylabel('Explained Variance EV(k)')
plt.title('Explained Variance vs Number of Components')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ---------- PART 4: Reconstruction & Outliers ----------
k = problem1_num_components

# Rank-k approximation
U_k = U[:, :k]
D_k = np.diag(d[:k])
V_k = Vt[:k, :].T  # or problem1_V[:, :k]

problem1_approximation = U_k @ D_k @ V_k.T

# Row-wise reconstruction error
problem1_reconstruction_error = np.linalg.norm(problem1_X - problem1_approximation, axis=1)

# Plot EDF (Empirical Distribution Function)
def plotEDF(data):
    """Plot Empirical Distribution Function"""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    y = np.arange(1, n + 1) / n
    plt.figure(figsize=(10, 6))
    plt.step(sorted_data, y, where='post')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Cumulative Probability')
    plt.title('Empirical Distribution Function of Reconstruction Errors')
    plt.grid(True, alpha=0.3)
    plt.show()

plotEDF(problem1_reconstruction_error)

# Find threshold for exactly N outliers
N_outliers = 100
sorted_errors = np.sort(problem1_reconstruction_error)[::-1]  # descending
problem1_threshold = sorted_errors[N_outliers - 1]  # Nth largest

# Flag outliers
outlier_mask = problem1_reconstruction_error >= problem1_threshold
problem1_outliers = problem1_X[outlier_mask]

print(f"Threshold for {N_outliers} outliers: {problem1_threshold:.4f}")
print(f"Outliers shape: {problem1_outliers.shape}")
```


```python
#🔷 TYPE 3: PCA for Data Compression
#Use when: Problem asks about compression, reducing storage, or approximation.

# ============================================
# PROBLEM: SVD for Data Compression
# ============================================

# ---------- Load & SVD ----------
df = pd.read_csv("data/data.csv")
X = df.to_numpy()

# Center the data
X_centered = X - np.mean(X, axis=0)

# SVD
U, d, Vt = np.linalg.svd(X_centered, full_matrices=False)

# ---------- Choose Compression Level ----------
# Option A: Keep 95% variance
explained_variance = np.cumsum(d**2) / np.sum(d**2)
k = np.argmax(explained_variance >= 0.95) + 1

# Option B: Keep fixed number of components
# k = 10

# ---------- Compress ----------
U_k = U[:, :k]
D_k = np.diag(d[:k])
V_k = Vt[:k, :].T

# Compressed representation
X_compressed = U_k @ D_k @ V_k.T

# ---------- Compression Metrics ----------
original_size = X.nbytes
compressed_size = U_k.nbytes + D_k.nbytes + V_k.nbytes
compression_ratio = original_size / compressed_size

print(f"Original size: {original_size} bytes")
print(f"Compressed size: {compressed_size} bytes")
print(f"Compression ratio: {compression_ratio:.2f}x")

# ---------- Reconstruction Error ----------
reconstruction_error = np.linalg.norm(X_centered - X_compressed, 'fro')
print(f"Frobenius reconstruction error: {reconstruction_error:.4f}")
```


```python
#TYPE 4: PCA for Feature Importance
#Use when: Problem asks about important features, interpretation, or loadings.

# ============================================
# PROBLEM: Find Most Important Features
# ============================================

# ---------- Load & SVD ----------
df = pd.read_csv("data/data.csv")
X = df.iloc[:, :-1].to_numpy()  # features
feature_names = df.columns[:-1].tolist()  # feature names

# Center
X_centered = X - np.mean(X, axis=0)

# SVD
U, d, Vt = np.linalg.svd(X_centered, full_matrices=False)
V = Vt.T  # right singular vectors as columns

# ---------- Explained Variance ----------
explained_variance = np.cumsum(d**2) / np.sum(d**2)
k = np.argmax(explained_variance >= 0.90) + 1
print(f"k = {k}")

# ---------- Top Features for Each Principal Component ----------
print("\n" + "="*60)
print("TOP FEATURES FOR EACH PRINCIPAL COMPONENT")
print("="*60)

for pc_idx in range(min(5, V.shape[1])):
    loadings = V[:, pc_idx]
    # Get top 5 features with highest absolute loadings
    top_indices = np.argsort(np.abs(loadings))[-5:][::-1]
    
    print(f"\nPC{pc_idx+1} (Explains {explained_variance[pc_idx]*100:.1f}% variance):")
    for idx in top_indices:
        print(f"  {feature_names[idx]:20s}: {loadings[idx]:.4f}")

# ---------- Feature Contribution Plot ----------
plt.figure(figsize=(10, 6))
plt.bar(feature_names, V[:, 0])
plt.xlabel('Features')
plt.ylabel('PC1 Loading')
plt.title('Feature Contributions to First Principal Component')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```


```python
#TYPE 5: Generic Dimensionality Reduction
#Use when: Problem just asks to reduce dimensions with PCA.

# ============================================
# PROBLEM: Reduce Dimensionality with PCA
# ============================================

# ---------- Load & SVD ----------
df = pd.read_csv("data/data.csv")
X = df.to_numpy()

# Center
X_centered = X - np.mean(X, axis=0)

# SVD
U, d, Vt = np.linalg.svd(X_centered, full_matrices=False)

# ---------- Choose k ----------
explained_variance = np.cumsum(d**2) / np.sum(d**2)
k = np.argmax(explained_variance >= 0.95) + 1
print(f"Using k = {k} components")

# ---------- Project to PCA Space ----------
X_pca = X_centered @ Vt[:k, :].T

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_pca.shape}")

# ---------- Explained Variance Plot ----------
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(d)+1), d**2 / np.sum(d**2), alpha=0.6, label='Individual')
plt.plot(range(1, len(d)+1), np.cumsum(d**2) / np.sum(d**2), 'r-', 
         linewidth=2, label='Cumulative')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% Threshold')
plt.axvline(x=k, color='orange', linestyle='--', label=f'k = {k}')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('PCA Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```


```python
#🔷 TYPE 6: SVD for Matrix Factorization
#Use when: Problem asks about recommendation systems or user-item matrices.

# ============================================
# PROBLEM: SVD for Matrix Factorization
# ============================================

# ---------- Load Data ----------
# User-Item matrix (users × movies)
df = pd.read_csv("data/ratings.csv")
X = df.to_numpy()  # ratings matrix

# Fill NaN with mean or 0
X = np.nan_to_num(X, nan=np.nanmean(X))

# ---------- SVD ----------
U, d, Vt = np.linalg.svd(X, full_matrices=False)

# ---------- Choose k ----------
explained_variance = np.cumsum(d**2) / np.sum(d**2)
k = np.argmax(explained_variance >= 0.90) + 1
print(f"k = {k}")

# ---------- Factor Matrices ----------
U_k = U[:, :k]          # User factors (users × latent)
D_k = np.diag(d[:k])    # Singular values
V_k = Vt[:k, :].T       # Movie factors (movies × latent)

# ---------- Reconstruct Ratings ----------
X_approx = U_k @ D_k @ V_k.T

# ---------- Predict Missing Ratings ----------
def predict_ratings(user_ratings):
    # user_ratings: known ratings (1 × movies)
    # Project to latent space
    user_latent = user_ratings @ V_k @ np.linalg.inv(D_k)
    # Predict all ratings
    return user_latent @ D_k @ V_k.T

print(f"Original ratings shape: {X.shape}")
print(f"Reconstructed ratings shape: {X_approx.shape}")
print(f"User factors shape: {U_k.shape}")
print(f"Movie factors shape: {V_k.shape}")
```


```python
#TYPE 7: PCA for Image Denoising
#Use when: Problem asks about denoising, removing noise, or cleaning images.

# ============================================
# PROBLEM: Denoise Images Using PCA
# ============================================

# ---------- Load Noisy Images ----------
df = pd.read_csv("data/noisy_images.csv")
X = df.to_numpy()  # each row is a noisy image

# Center
X_centered = X - np.mean(X, axis=0)

# ---------- SVD ----------
U, d, Vt = np.linalg.svd(X_centered, full_matrices=False)

# ---------- Choose k ----------
explained_variance = np.cumsum(d**2) / np.sum(d**2)
k = np.argmax(explained_variance >= 0.90) + 1
print(f"k = {k} (keeps {explained_variance[k-1]*100:.1f}% variance)")

# ---------- Denoise ----------
# Keep only first k components (remove noise)
X_denoised = U[:, :k] @ np.diag(d[:k]) @ Vt[:k, :]

# Add back mean
X_denoised = X_denoised + np.mean(X, axis=0)

print(f"Original shape: {X.shape}")
print(f"Denoised shape: {X_denoised.shape}")

# ---------- Noise Reduction ----------
noise_original = np.linalg.norm(X - X_denoised, 'fro')
print(f"Removed noise (Frobenius norm): {noise_original:.4f}")
```


```python
#Quick Reference Card

# ============================================
# THE ULTIMATE PCA/SVD CHEAT SHEET
# ============================================

# 1. ALWAYS DO THIS FIRST
U, d, Vt = np.linalg.svd(X_centered, full_matrices=False)
V = Vt.T

# 2. EXPLAINED VARIANCE
ev = np.cumsum(d**2) / np.sum(d**2)
k = np.argmax(ev >= threshold) + 1

# 3. PROJECT TO PCA SPACE
scores = X_centered @ V[:, :k]  # or X_centered @ Vt[:k, :].T

# 4. RECONSTRUCT
X_approx = U[:, :k] @ np.diag(d[:k]) @ Vt[:k, :]

# 5. RECONSTRUCTION ERROR
error = np.linalg.norm(X - X_approx, axis=1)

# 6. PLOT 2D
plt.scatter(scores[:, 0], scores[:, 1], c=labels)

# 7. CLASSIFY
# Train classifier on 'scores', predict on test

# 8. FIND OUTLIERS
threshold = np.sort(error)[::-1][N-1]
outliers = X[error >= threshold]

# Quick Decision Guide 
# Keywords in Problem	Use Solution Type
"classify", "predict", "accuracy", "labels"	Type 1 - Classification
"anomaly", "outlier", "reconstruction error"	Type 2 - Anomaly Detection
"compress", "reduce storage", "approximation"	Type 3 - Compression
"important features", "interpret", "loadings"	Type 4 - Feature Importance
"dimensionality reduction", "reduce dimensions"	Type 5 - Generic Reduction
"recommendation", "matrix factorization", "ratings"	Type 6 - Factorization
"denoise", "remove noise", "clean"	Type 7 - Denoising  
```
