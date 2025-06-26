import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.signal import detrend, butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import os

# === 1. MATRIZ DE MASA Y VECTOR r (copiado de tu parte teórica) ===
g = 9.81  # m/s^2
peso_vigas = 0.075 / 6  # kg
m1 = 0.555 + 0.26  # Kgf
m2 = 0.553 + 0.26  # Kgf
m3 = 0.691 + 0.26  # Kgf

M1 = (m1 + 4 * peso_vigas) / (g * 1000)  # Tonf s2 / m
M2 = (m2 + 8 * peso_vigas) / (g * 1000)
M3 = (m3 + 8 * peso_vigas) / (g * 1000)

M = np.array([M1, M2, M3])  # vector de masas (para multiplicar elemento a elemento)
r = np.array([1, 1, 1])     # vector sísmico

# === 2. LEER DATOS DEL ARCHIVO ===
try:
    data = np.loadtxt('Armonica_Base.txt')
except FileNotFoundError:
    data = np.loadtxt('DATA/Armonica_Base.txt')

# Columnas: tiempo, base, dof1, dof2, dof3
tiempo = data[:, 0]
dof1 = data[:, 2]
dof2 = data[:, 3]
dof3 = data[:, 4]

# === 3. PROCESAR SEÑALES: FILTRO + INTEGRACIÓN ===
def butter_bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

fs = 200  # Hz, frecuencia de muestreo (ajusta si corresponde)
T = 1/fs
dofs = [dof1, dof2, dof3]
disp = []

for a in dofs:
    a = detrend(a)
    a = butter_bandpass_filter(a, lowcut=0.5, highcut=20, fs=fs)
    v = scipy.integrate.cumulative_trapezoid(a, dx=T, initial=0)
    v = detrend(v)
    d = scipy.integrate.cumulative_trapezoid(v, dx=T, initial=0)
    d = detrend(d)
    disp.append(d)

disp = np.array(disp)
N = len(tiempo)

# === 4. FFT E IDENTIFICACIÓN MODAL ===
Y = np.array([fft(d) for d in disp])
freqs = fftfreq(N, T)
fmin = 1.0
mask = freqs > fmin
freqs_pos = freqs[mask]
Y_mag = np.abs(Y[:, mask])
suma_magnitud = np.sum(Y_mag, axis=0)
n_modos = 3
peaks, _ = find_peaks(suma_magnitud, distance=fs//2)
top_peaks = peaks[np.argsort(suma_magnitud[peaks])[-n_modos:]]

frecs_naturales = []
formas_modales = []

for idx in sorted(top_peaks):
    f = freqs_pos[idx]
    frecs_naturales.append(f)
    forma = Y_mag[:, idx]
    # normaliza por el máximo absoluto
    forma = forma / np.max(np.abs(forma))
    # forzar primer piso positivo (por convención)
    if forma[0] < 0:
        forma = -forma
    formas_modales.append(forma)

formas_modales = np.array(formas_modales)

# === 5. NORMALIZACIÓN SÍSMICA DE LOS MODOS EXPERIMENTALES ===
def normalizar_vectores_exp(phi, M, r):
    A = phi.T @ (M * r)
    B = phi.T @ (M * phi)
    return phi * (A/B)

formas_modales_sismicas = []
for modo in formas_modales:
    modo_norm = normalizar_vectores_exp(modo, M, r)
    formas_modales_sismicas.append(modo_norm)
formas_modales_sismicas = np.array(formas_modales_sismicas)

# === 6. GRAFICAR FORMAS MODALES COMO ESTRUCTURA DEFORMADA ===
def graficar_modos_estructura_exp(formas_modales, ancho=1.0, altura=1.0, escalar=0.3, guardar=True):
    """
    Dibuja los modos experimentales de vibración con forma estructural (marco plano deformado).
    - formas_modales: array de shape (n_modos, n_dofs)
    """
    n_modos = formas_modales.shape[0]
    n_dofs = formas_modales.shape[1]
    niveles_y = np.array([altura * i for i in range(n_dofs+1)][::-1])

    for i in range(n_modos):
        modo = formas_modales[i, :].copy()
        
        # === CAMBIO DE SIGNO PEDIDO ===
        if i == 1:
            modo[0] *= -1  # Modo 2, piso 3
        if i == 2:
            modo[1] *= -1  # Modo 3, piso 2

        despl_x = modo * escalar
        x_izq = np.zeros_like(niveles_y)
        x_der = np.full_like(niveles_y, ancho)

        fig, ax = plt.subplots(figsize=(6, 8))

        # Estructura no deformada (rojo)
        for j in range(n_dofs):
            ax.plot([0, ancho], [niveles_y[j]] * 2, color='red', linestyle='--', linewidth=1)
            if j < n_dofs:
                ax.plot([0, 0], [niveles_y[j], niveles_y[j + 1]], color='red', linestyle='--', linewidth=1)
                ax.plot([ancho, ancho], [niveles_y[j], niveles_y[j + 1]], color='red', linestyle='--', linewidth=1)
        y_base = niveles_y[-1]
        ax.plot([0, 0], [0, y_base], color='red', linestyle='--', linewidth=1)
        ax.plot([ancho, ancho], [0, y_base], color='red', linestyle='--', linewidth=1)

        # Estructura deformada
        for j in range(n_dofs):
            ax.plot([x_izq[j] + despl_x[j], x_der[j] + despl_x[j]], [niveles_y[j], niveles_y[j]], color='blue', linewidth=2)
            if j < n_dofs - 1:
                ax.plot([x_izq[j] + despl_x[j], x_izq[j + 1] + despl_x[j + 1]],
                        [niveles_y[j], niveles_y[j + 1]], color='black', linestyle='--')
                ax.plot([x_der[j] + despl_x[j], x_der[j + 1] + despl_x[j + 1]],
                        [niveles_y[j], niveles_y[j + 1]], color='black', linestyle='--')
                ax.plot([0, x_izq[2] + despl_x[2]], [0, niveles_y[2]], color='black', linestyle='--')
                ax.plot([ancho, x_der[2] + despl_x[2]], [0, niveles_y[2]], color='black', linestyle='--')

        ax.set_title(f"Experimental vibration of Mode {i+1}")
        ax.set_xlabel("Desplazamiento horizontal (amplificado)")
        ax.set_ylabel("Altura [m]")
        ax.set_aspect('equal')
        ax.set_xlim(-1.5 * escalar, ancho + 1.5 * escalar)
        ax.set_ylim(-0.1, niveles_y[0] + 0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True)
        if guardar:
            os.makedirs("INFORME/GRAFICOS/", exist_ok=True)
            plt.savefig(f"INFORME/GRAFICOS/Modo_Experimental_Estructura_Sismico_{i+1}.png", dpi=300)
        else:
            plt.show()


graficar_modos_estructura_exp(formas_modales_sismicas, ancho=1.0, altura=1.0, escalar=0.3, guardar=True)

# === 7. IMPRIMIR FRECUENCIAS Y MODOS ===
print("\n=== FRECUENCIAS NATURALES EXPERIMENTALES (Hz) ===")
for i, f in enumerate(frecs_naturales):
    print(f"Modo {i+1}: {f:.2f} Hz | Forma modal sísmicamente normalizada: {formas_modales_sismicas[i]}")

print("\nGráficos guardados en INFORME/GRAFICOS/")

#print normalized modes
print("\n=== MODOS NORMALIZADOS SÍSMICAMENTE ===")
for i, modo in enumerate(formas_modales_sismicas):
    print(f"Modo {i+1}: {modo}")

