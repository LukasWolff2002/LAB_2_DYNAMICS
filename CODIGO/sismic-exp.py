import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === 1. DATOS ESTRUCTURALES EXPERIMENTALES (de tus scripts) ===
g = 9.81 # m/s^2
peso_vigas = 0.075 / 6 # kg
m1 = 0.555 + 0.26 # Kgf
m2 = 0.553 + 0.26 # Kgf
m3 = 0.691 + 0.26 # Kgf

M1 = (m1 + 4*peso_vigas) / (g*1000)
M2 = (m2 + 8*peso_vigas) / (g*1000)
M3 = (m3 + 8*peso_vigas) / (g*1000)
M = np.array([[M1, 0, 0],
              [0, M2, 0],
              [0, 0, M3]])
r = np.array([[1], [1], [1]])

# === 2. Primer modo experimental (de tu análisis armónico/pullback) ===
# Sustituye estos datos por los de tu procesamiento experimental:
# Ejemplo (ajusta según tus resultados experimentales reales):
phi1_exp = np.array([1.24, 0.76, 0.10])         # <- Primer modo experimental normalizado sísmicamente
freq1_exp = 2.00                               # <- Frecuencia experimental del modo 1 (Hz)
T1_exp = 1 / freq1_exp                         # <- Periodo experimental
b_exp = 0.05                                   # <- Damping ratio experimental (ajusta si lo estimaste)

# === 3. Masa modal experimental ===
phi1_exp = phi1_exp.reshape((3, 1))
m_modal_exp = float((phi1_exp.T @ M @ r) ** 2 / (phi1_exp.T @ M @ phi1_exp))
print(f"Experimental modal mass: {m_modal_exp * 1000:.4f} kg")
print(f"Experimental period: {T1_exp:.4f} s")

# === 4. Leer sismo ===
df = pd.read_csv('DATA\Concepción.txt', sep=r'\s+', header=None)
t = df.iloc[:, 0].values
a_base = df.iloc[:, 4].values
a_base = a_base * 9.81 if np.max(np.abs(a_base)) < 2 else a_base
dt = np.mean(np.diff(t))
Fs = 1 / dt

# === 5. Fuerza sísmica modal experimental ===
P_modal_exp = -m_modal_exp * a_base

# === 6. Método de Newmark para SDOF (idéntico a antes) ===
def respnewmark(m, T, b, P, Fs, xo=0.0, vo=0.0, beta=0.25, gama=0.5):
    P = P.flatten()
    N = len(P)
    x = np.zeros(N)
    v = np.zeros(N)
    a = np.zeros(N)
    w = 2 * np.pi / T
    k = m * w ** 2
    c = 2 * m * w * b
    dt = 1 / Fs
    dt2 = dt ** 2
    x[0] = xo
    v[0] = vo
    a[0] = (P[0] - c * v[0] - k * x[0]) / m
    k1 = k + gama * c / (beta * dt) + m / (beta * dt2)
    A = m / (beta * dt) + c * gama / beta
    B = m / (2 * beta) + dt * (gama / (2 * beta) - 1) * c
    for i in range(N - 1):
        deltaP = P[i + 1] - P[i] + A * v[i] + B * a[i]
        deltax = deltaP / k1
        deltav = gama * deltax / (beta * dt) - gama * v[i] / beta + dt * (1 - gama / (2 * beta)) * a[i]
        deltaa = deltax / (beta * dt2) - v[i] / (beta * dt) - a[i] / (2 * beta)
        x[i + 1] = x[i] + deltax
        v[i + 1] = v[i] + deltav
        a[i + 1] = a[i] + deltaa
    at = a - P / m
    return x, v, a, at

# === 7. Respuesta con Newmark usando modo experimental ===
x, v, a_rel, a_abs = respnewmark(m_modal_exp * 1000, T1_exp, b_exp, P_modal_exp, Fs)

# === 8. Gráfica ===
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(t, x, label='Displacement [m]')
plt.ylabel('x [m]')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, v, label='Velocity [m/s]', color='orange')
plt.ylabel('v [m/s]')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, a_abs, label='Absolute Acceleration [m/s²]', color='red')
plt.xlabel('Time [s]')
plt.ylabel('a [m/s²]')
plt.legend()
plt.grid(True)

plt.suptitle('SDOF Experimental Modal Response to Concepción Earthquake\n(Newmark, Parameters from Pull-back Test)')
plt.tight_layout()
plt.show()
