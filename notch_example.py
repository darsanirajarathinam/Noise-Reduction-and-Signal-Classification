import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- SIGNAL SETUP ---
fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)
clean_signal = np.sin(2 * np.pi * 5 * t)
noise = 0.5 * np.sin(2 * np.pi * 50 * t)
noisy_signal = clean_signal + noise

# --- NOTCH FILTER ---
f0 = 50.0
Q = 30.0
b, a = signal.iirnotch(f0, Q, fs)
filtered_signal = signal.filtfilt(b, a, noisy_signal)

# --- SNR CALCULATION (done BEFORE plotting so it prints immediately) ---
def compute_snr_db(clean, noisy_or_filtered):
    sig_power = np.sum(clean ** 2)
    noise_power = np.sum((noisy_or_filtered - clean) ** 2)
    if noise_power == 0:
        return float('inf')
    snr = 10 * np.log10(sig_power / noise_power)
    return snr

snr_before = compute_snr_db(clean_signal, noisy_signal)
snr_after  = compute_snr_db(clean_signal, filtered_signal)
improvement = snr_after - snr_before

print(f"SNR before filtering : {snr_before:.2f} dB")
print(f"SNR after  filtering : {snr_after:.2f} dB")
print(f"Improvement (after - before) : {improvement:.2f} dB")

# --- PLOTTING ---
plt.figure(figsize=(10,6))
plt.subplot(3,1,1); plt.plot(t, clean_signal); plt.title("Clean Signal (5 Hz)")
plt.subplot(3,1,2); plt.plot(t, noisy_signal); plt.title("Noisy Signal (with 50 Hz)")
plt.subplot(3,1,3); plt.plot(t, filtered_signal); plt.title("Filtered Signal (after notch filter)")
plt.tight_layout()
plt.show(block=True)

