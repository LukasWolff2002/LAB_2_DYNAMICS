import numpy as np

#Otros datos
g = 9.81 #m/s^2 
E = 200_000_000/g #Tonf/m2
peso_vigas = 0.075/6 #kg
largo_viga = 0.6/3 #m

I = ((26.25/1000) * (0.7/1000)**3)/12 #m^4, momento de inercia de la viga


#Defino los datos conocidos del sistema
m1 = 0.555 + 0.26 #Kgf
m2 = 0.553 + 0.26 #Kgf
m3 = 0.691 + 0.26 #Kgf

M1 = (m1 + 4*peso_vigas) / (g*1000) #Tonf s2 / m
M2 = (m2 + 8 * peso_vigas) / (g*1000) 
M3 = (m3 + 8 * peso_vigas) / (g*1000)

k1 = (48*E*I)/(largo_viga**3)
k2 = (48*E*I)/(largo_viga**3)
k3 = (48*E*I)/(largo_viga**3)


#Las matrices a ensamblar son:
M = np.array([[M1, 0, 0],
              [0, M2, 0],
              [0, 0, M3]])

K = np.array([[k1, -k1, 0],
             [-k1, k1+k2, -k2],
             [0, -k2, k2+k3]])

r = np.array([[1],
              [1],
              [1]])

#Defino las frecuencias fundamentales del sistema, ademas de los modos de vibracion
#Los cuales son los valores propios de la matriz de masa por la matriz de rigidez

def eigenvalues(M, K):
    # Calculamos los valores propios y vectores propios
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M) @ K)
    
    # Ordenamos los valores propios y vectores propios
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors

def normalizar_vectores(phi):

    A = phi.T@M@r
    B = phi.T@M@phi

    return phi * (A/B)

def normalizar_matrices (phi, matriz):

    return phi.T @ matriz @ phi

valores, vectores = eigenvalues(M, K)

wn = np.sqrt(valores)  # Frecuencias naturales


print("\nFrecuencias naturales (rad/s):", wn)

print("frecuencias naturales (Hz):", wn / (2 * np.pi))


#Ahora normalizo los vectores

phi1, phi2, phi3 = vectores[:, 0], vectores[:, 1], vectores[:, 2]

phi1_norm = normalizar_vectores(phi1)
phi2_norm = normalizar_vectores(phi2)
phi3_norm = normalizar_vectores(phi3)

Phi_norm = np.array([phi1_norm, phi2_norm, phi3_norm])

#Ahora normalizo las matrices
M_norm = normalizar_matrices(Phi_norm, M)
K_norm = normalizar_matrices(Phi_norm, K)

#Bien, ahora import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

def graficar_modos_estructura(phi_norm, ancho=1.0, altura=1.0, escalar=0.3, guardar=False):
    """
    Dibuja los modos de vibración con forma estructural (marco plano deformado).
    - DOF1 es el último piso, DOF3 es el primero.
    - Se agregan columnas desde el suelo.
    - Se dibuja estructura no deformada como referencia en gris.
    """
    n_modos = phi_norm.shape[1]
    n_dofs = phi_norm.shape[0]

    # Invertir el orden de los niveles: DOF3 en el suelo (y=0)
    niveles_y = np.array([altura * i for i in range(n_dofs+1)][::-1])
    print(f"🎯 Niveles de la estructura: {niveles_y}")

    for i in range(n_modos):
        modo = phi_norm[:, i].flatten()
        modo = modo / np.max(np.abs(modo))  # normalizar
        despl_x = modo * escalar

        # Posición horizontal de columnas
        x_izq = np.zeros_like(niveles_y)
        x_der = np.full_like(niveles_y, ancho)

        fig, ax = plt.subplots(figsize=(6, 8))

        # === ESTRUCTURA NO DEFORMADA (gris) ===
        for j in range(n_dofs):
            # Vigas horizontales
            ax.plot([0, ancho], [niveles_y[j]] * 2, color='red', linestyle='--', linewidth=1)
            # Columnas verticales
            if j < n_dofs :
                ax.plot([0, 0], [niveles_y[j], niveles_y[j + 1]], color='red', linestyle='--', linewidth=1)
                ax.plot([ancho, ancho], [niveles_y[j], niveles_y[j + 1]], color='red', linestyle='--', linewidth=1)
        # Columnas desde suelo a primer piso
        y_base = niveles_y[-1]
        ax.plot([0, 0], [0, y_base], color='red', linestyle='--', linewidth=1)
        ax.plot([ancho, ancho], [0, y_base], color='red', linestyle='--', linewidth=1)

        # === ESTRUCTURA DEFORMADA ===
        for j in range(n_dofs):
            # Vigas horizontales deformadas
            ax.plot([x_izq[j] + despl_x[j], x_der[j] + despl_x[j]],
                    [niveles_y[j], niveles_y[j]], color='blue', linewidth=2)
            # Columnas deformadas entre niveles
            if j < n_dofs - 1:
                ax.plot([x_izq[j] + despl_x[j], x_izq[j + 1] + despl_x[j + 1]],
                        [niveles_y[j], niveles_y[j + 1]], color='black', linestyle='--')
                ax.plot([x_der[j] + despl_x[j], x_der[j + 1] + despl_x[j + 1]],
                        [niveles_y[j], niveles_y[j + 1]], color='black', linestyle='--')
                
                ax.plot([0, x_izq[2] + despl_x[2]],
                        [0, niveles_y[2]], color='black', linestyle='--')
                
                ax.plot([ancho, x_der[2] + despl_x[2]],
                        [0, niveles_y[2]], color='black', linestyle='--')


        # === Etiquetas y configuración ===
        ax.set_title(f"Modo {i+1} (estructura deformada)")
        ax.set_xlabel("Desplazamiento horizontal (amplificado)")
        ax.set_ylabel("Altura [m]")
        ax.set_aspect('equal')
        ax.set_xlim(-1.5 * escalar, ancho + 1.5 * escalar)
        ax.set_ylim(-0.1, niveles_y[0] + 0.1)
        ax.yaxis
        
        ax.set_xticks([])  # Oculta los valores en el eje X
        ax.set_yticks([])  # Oculta los valores en el eje Y
        ax.grid(True)
        if guardar:
            plt.savefig(f"INFORME/GRAFICOS/Modo_Estructura_{i+1}.png", dpi=300)
        else:
            plt.show()


graficar_modos_estructura(Phi_norm.T, ancho=1.0, altura=1.0, escalar=0.3, guardar=True)


#print vectores normalizados por piso
print("\nVectores normalizados por piso:")
print("Modo 1:", phi1_norm) 
print("Modo 2:", phi2_norm)
print("Modo 3:", phi3_norm)
