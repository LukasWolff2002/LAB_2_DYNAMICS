import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.signal import detrend, butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq

# === 1. Cargar y procesar datos ===

# Cargar archivo completo
data = np.loadtxt("DATA/Armonica_Base.txt")
df_total = pd.DataFrame(data, columns=["tiempo", "dof1", "dof2", "dof3", "suelo"])

# Buscar el máximo de DOF1 y recortar desde ese instante hasta el final
idx_max = df_total["dof1"].idxmax()
df_segment = df_total.loc[idx_max:].copy()
df_segment["tiempo"] -= df_segment["tiempo"].iloc[0]  # reiniciar tiempo desde cero

# Usaremos una única lista con un solo pullback
dfs = [df_segment]

# === 2. Filtro pasa banda ===

def butter_bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

# === 3. Identificación modal y Fourier ===

def identificar_modos_fft_con_picos(dfs, fs=200, n_modos=3, fmin=1.0):
    df_total = pd.concat(dfs, ignore_index=True)
    T = 1 / fs
    t = df_total["tiempo"].values
    N = len(t)

    dofs = ["dof1", "dof2", "dof3"]
    disp = []

    for dof in dofs:
        a = df_total[dof].values
        a = detrend(a)
        a = butter_bandpass_filter(a, lowcut=0.5, highcut=20, fs=fs)

        v = scipy.integrate.cumulative_trapezoid(a, dx=T, initial=0)
        v = detrend(v)
        d = scipy.integrate.cumulative_trapezoid(v, dx=T, initial=0)
        d = detrend(d)
        disp.append(d)

    disp = np.array(disp)
    Y = np.array([fft(d) for d in disp])
    freqs = fftfreq(N, T)
    mask = freqs > fmin
    freqs_pos = freqs[mask]
    Y_mag = np.abs(Y[:, mask])

    # Magnitud combinada
    suma_magnitud = np.sum(Y_mag, axis=0)

    # Buscar picos reales
    peaks, _ = find_peaks(suma_magnitud, distance=fs//2)  # evitar picos muy juntos
    top_peaks = peaks[np.argsort(suma_magnitud[peaks])[-n_modos:]]

    frecs_naturales = []
    formas_modales = []

    for idx in sorted(top_peaks):
        f = freqs_pos[idx]
        frecs_naturales.append(f)

        forma = Y_mag[:, idx]
        forma /= np.max(np.abs(forma))
        formas_modales.append(forma)

    return frecs_naturales, formas_modales, freqs_pos, Y_mag

# === Ejecutar ===
frecs, modos, freqs_pos, Y_mag = identificar_modos_fft_con_picos(dfs, fs=200)

# === Graficar FFT para cada DOF ===
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
dofs = ["DOF 1", "DOF 2", "DOF 3"]
for i in range(3):
    axs[i].plot(freqs_pos, Y_mag[i])
    axs[i].set_ylabel("Magnitud")
    axs[i].set_title(f"FFT - {dofs[i]}")
    axs[i].grid(True)

axs[-1].set_xlabel("Frecuencia (Hz)")
plt.tight_layout()
plt.savefig("INFORME/GRAFICOS/FFT_Modos.png", dpi=300)

print("Frecuencias naturales (Hz):", frecs)