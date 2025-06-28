import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.fft import fft, fftfreq
from scipy.signal import detrend

# === 1. Cargar y procesar datos ===

# Cargar archivo
data = np.loadtxt("DATA/Pullback.txt")
df_total = pd.DataFrame(data, columns=["tiempo", "dof1", "dof2", "dof3", "suelo"])

# Intervalos definidos manualmente (en segundos)
intervalos = [
    (50, 150),
    (150, 200),
    (200, 250),
    (250, 300),
]

# Extraer pullbacks desde el máximo de DOF1 hasta fin de intervalo
dfs = []
for ini, fin in intervalos:
    df_intervalo = df_total[(df_total["tiempo"] >= ini) & (df_total["tiempo"] <= fin)].copy()
    idx_max = df_intervalo["dof1"].idxmax()
    tiempo_max = df_total.loc[idx_max, "tiempo"]
    df_segment = df_total[(df_total["tiempo"] >= tiempo_max) & (df_total["tiempo"] <= fin)].copy()
    df_segment["tiempo"] -= df_segment["tiempo"].iloc[0]  # reiniciar tiempo desde máximo
    dfs.append(df_segment)

df1, df2, df3, df4 = dfs

# === 2. Graficar los pullbacks ===

def graficar_pullbacks_en_grid(dfs):
    """
    Genera una figura por cada pullback con 4 subplots (DOF1, DOF2, DOF3, suelo).
    Guarda cada figura como PNG.
    """
    dof_labels = ["dof1", "dof2", "dof3", "suelo"]
    titulos = ["Acceleration DOF 1", "Acceleration DOF 2",
               "Acceleration DOF 3", "Ground Acceleration"]

    for i, df in enumerate(dfs):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Pullback {i+1}", fontsize=16)

        for j, ax in enumerate(axs.flat):
            ax.plot(df["tiempo"], df[dof_labels[j]])
            ax.set_title(titulos[j])
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Acceleration [m/s²]")
            ax.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"INFORME/GRAFICOS/Pullback_{i+1}.png", dpi=300)
        plt.close(fig)  # cerrar para no saturar memoria

graficar_pullbacks_en_grid(dfs)

# === 3. Identificación modal global ===
import numpy as np
import pandas as pd
import scipy.integrate
from scipy.signal import detrend, butter, filtfilt
from scipy.fft import fft, fftfreq

def butter_bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

def identificar_modos_globales_desplazamiento_filtrado(dfs, fs=200, n_modos=3):
    """
    Estima frecuencias y formas modales a partir de desplazamientos
    integrando aceleraciones filtradas.
    """
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

    disp = np.array(disp)  # shape (3 x N)

    Y = np.array([fft(d) for d in disp])
    freqs = fftfreq(N, T)
    mask = freqs > 0
    freqs_pos = freqs[mask]
    Y_mag = np.abs(Y[:, mask])

    suma_magnitud = np.sum(Y_mag, axis=0)
    idx_top = np.argsort(suma_magnitud)[-n_modos:][::-1]

    frecs_naturales = []
    formas_modales = []

    for idx in idx_top:
        f = freqs_pos[idx]
        frecs_naturales.append(f)

        forma = Y_mag[:, idx]
        forma /= np.max(np.abs(forma))
        formas_modales.append(forma)

    return frecs_naturales, formas_modales

def analizar_fourier_pullbacks(dfs, fs=200):
    """
    Aplica FFT a cada pullback, grafica el espectro de frecuencias (solo DOF1–3)
    y muestra la frecuencia dominante (máxima magnitud) por DOF.
    """
    dofs = ["dof1", "dof2", "dof3"]

from scipy.signal import find_peaks

def analizar_fourier_pullbacks(dfs, fs=200, n_picos=3):
    """
    Aplica FFT a cada pullback, grafica el espectro (solo DOF1–3)
    y muestra los n_picos picos más altos de frecuencia dominante por DOF.
    """
    dofs = ["dof1", "dof2", "dof3"]

    for i, df in enumerate(dfs):
        T = 1 / fs
        N = len(df)
        t = df["tiempo"].values

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Fourier Spectrum - Pullback {i+1}", fontsize=16)

        print(f"\n🎯 Pullback {i+1} - {n_picos} Frecuencias dominantes por DOF:")

        for j, dof in enumerate(dofs):
            a = df[dof].values
            a = detrend(a)

            Y = fft(a)
            freqs = fftfreq(N, T)
            mask = freqs > 0  # Solo frecuencias positivas

            Y_mag = np.abs(Y[mask])
            freqs_pos = freqs[mask]

            # === Detectar picos reales ===
            peaks, _ = find_peaks(Y_mag, distance=fs//2, height=np.max(Y_mag)*0.05)  # filtro por separación y altura
            peak_mags = Y_mag[peaks]
            top_peaks_idx = peaks[np.argsort(peak_mags)[-n_picos:][::-1]]
            frecs_dominantes = freqs_pos[top_peaks_idx]

            # === Imprimir resultados ===
            print(f"  DOF{j+1}: " + ", ".join([f"{f:.2f} Hz" for f in frecs_dominantes]))

            # === Graficar ===
            axs[j].plot(freqs_pos, Y_mag)
            for f in frecs_dominantes:
                axs[j].axvline(f, color='r', linestyle='--')
            axs[j].set_title(f"DOF {j+1}")
            axs[j].set_ylabel("Magnitude")
            axs[j].grid(True)

        axs[-1].set_xlabel("Frequency [Hz]")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"INFORME/GRAFICOS/Fourier_Pullback_{i+1}.png", dpi=300)
        plt.close(fig)



frecs, modos = identificar_modos_globales_desplazamiento_filtrado(dfs, fs=200)
periodos = [1/f for f in frecs]

print("Frecuencias naturales (Hz):", frecs)
print("Periodos naturales (s):", periodos)
print("Formas modales (normalizadas):")
for i, modo in enumerate(modos):
    print(f"Modo {i+1}:", modo.round(3))
analizar_fourier_pullbacks([df1, df2, df3], fs=200, n_picos=3)



