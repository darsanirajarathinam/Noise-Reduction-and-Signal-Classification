import numpy as np
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# --- Step 1: Parameters ---
fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)
f0 = 50.0
Q = 30.0
b, a = signal.iirnotch(f0, Q, fs)

# --- Step 2: Create signals of different frequencies (our "classes") ---
freq_classes = [5, 10, 20]  # 3 signal classes
num_samples_per_class = 100

X = []  # features
y = []  # labels

for label, freq in enumerate(freq_classes):
    for _ in range(num_samples_per_class):
        clean_signal = np.sin(2 * np.pi * freq * t)
        noise = 0.5 * np.sin(2 * np.pi * 50 * t)  # 50 Hz noise
        noisy_signal = clean_signal + noise
        filtered_signal = signal.filtfilt(b, a, noisy_signal)
        
        # --- Step 3: Extract simple features ---
        mean_val = np.mean(filtered_signal)
        std_val = np.std(filtered_signal)
        energy = np.sum(filtered_signal ** 2)
        max_amp = np.max(filtered_signal)
        min_amp = np.min(filtered_signal)
        
        # Collect features
        X.append([mean_val, std_val, energy, max_amp, min_amp])
        y.append(label)

X = np.array(X)
y = np.array(y)

# --- Step 4: Split data into train and test sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 5: Train ML model ---
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# --- Step 6: Evaluate performance ---
acc = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {acc * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Step 7: Visualize feature space (just 2D example) ---
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel('Mean Value')
plt.ylabel('Standard Deviation')
plt.title('Feature Space (Colored by Class)')
plt.show()
