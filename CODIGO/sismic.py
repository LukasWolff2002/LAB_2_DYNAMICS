import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === 1. DATOS ESTRUCTURA (de tus códigos, alineados) ===
g = 9.81 # m/s^2 
E = 200_000_000 / g # Tonf/m2
peso_vigas = 0.075 / 6 # kg
largo_viga = 0.6 / 3 # m
I = ((26.25/1000) * (0.7/1000)**3) / 12 # m^4

m1 = 0.555 + 0.26 # Kgf
m2 = 0.553 + 0.26 # Kgf
m3 = 0.691 + 0.26 # Kgf

M1 = (m1 + 4*peso_vigas) / (g*1000) # Tonf s2 / m
M2 = (m2 + 8*peso_vigas) / (g*1000)
M3 = (m3 + 8*peso_vigas) / (g*1000)

# Matriz de masa (en Tonf·s²/m)
M = np.array([[M1, 0, 0],
              [0, M2, 0],
              [0, 0, M3]])

# Matriz de rigidez (en Tonf/m)
k1 = k2 = k3 = (48 * E * I) / (largo_viga ** 3)
K = np.array([[k1, -k1, 0],
              [-k1, k1 + k2, -k2],
              [0, -k2, k2 + k3]])

# Vector de influencia sísmica
r = np.array([[1], [1], [1]])

# === 2. MODOS Y MASA MODAL DEL PRIMER MODO ===
def eigenvalues(M, K):
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M) @ K)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

def normalizar_vectores(phi, M, r):
    A = phi.T @ M @ r
    B = phi.T @ M @ phi
    return phi * (A / B)

# Valores propios y modos
eigs, vectores = eigenvalues(M, K)
wn = np.sqrt(eigs)               # rad/s
freqs = wn / (2 * np.pi)         # Hz

phi1 = vectores[:, 0]
phi1_norm = normalizar_vectores(phi1, M, r).flatten()

# === 3. MASA MODAL EFECTIVA DEL PRIMER MODO ===
m_modal = float((phi1_norm.T @ M @ r) ** 2 / (phi1_norm.T @ M @ phi1_norm))
T_modal = 1 / freqs[0]
w_modal = wn[0]

print("=== SDOF MODAL PARAMETERS ===")
print(f"Modal mass (kg): {m_modal * 1000:.4f}")
print(f"Modal period (s): {T_modal:.4f}")

# === 4. LECTURA DE SISMO ===
df = pd.read_csv('DATA\Concepción.txt', sep=r'\s+', header=None)
t = df.iloc[:, 0].values
a_base = df.iloc[:, 4].values
a_base = a_base * 9.81 if np.max(np.abs(a_base)) < 2 else a_base

dt = np.mean(np.diff(t))
Fs = 1 / dt
N = len(t)

# === 5. FUERZA SÍSMICA MODAL ===
# Proyección de la aceleración base en el primer modo (phi1_norm)
P_modal = -m_modal * np.dot(phi1_norm, np.ones(3)/3) * a_base  # Simplificación: r=[1,1,1] y uniform

# Si quieres exactamente: P_modal = -m_modal * a_base
# pero la teoría estricta sería usar la participación modal del primer modo
# Puedes dejarlo como: 
P_modal = -m_modal * a_base

# === 6. MÉTODO DE NEWMARK PARA SDOF ===
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

# Amortiguamiento modal (5%)
b = 0.05

# === 7. RESPUESTA CON NEWMARK (SDOF MODAL) ===
x, v, a_rel, a_abs = respnewmark(m_modal * 1000, T_modal, b, P_modal, Fs, xo=0.0, vo=0.0)

# === 8. GRAFICAR RESPUESTA ===
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

plt.suptitle('SDOF Modal Response to Concepción Earthquake (Newmark, ζ = 5%)')
plt.tight_layout()
plt.show()
