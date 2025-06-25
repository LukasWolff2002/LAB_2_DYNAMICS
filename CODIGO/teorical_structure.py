import numpy as np

#Otros datos
E = 200000 #MPa
peso_vigas = 0.05 #kg
largo_viga = 600/3 #m
g = 9.81 #m/s^2 

I = 0.7503124999999998 


#Defino los datos conocidos del sistema
m1 = 0.26 #Kg
m2 = 0.26 #Kg
m3 = 0.26 #Kg

M1 = (m1 + peso_vigas) * g
M2 = (m2 + 2 * peso_vigas) * g
M3 = (m3 + 3 * peso_vigas) * g

k1 = (48*E*I)/(largo_viga**3)
k2 = 200
k3 = 100


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
Tn = 2 * np.pi / wn  # Periodos naturales

print("Frecuencias naturales (rad/s):", wn)
print("Periodos naturales (s):", Tn)

#Ahora normalizo los vectores

phi1, phi2, phi3 = vectores[:, 0], vectores[:, 1], vectores[:, 2]

phi1_norm = normalizar_vectores(phi1)
phi2_norm = normalizar_vectores(phi2)
phi3_norm = normalizar_vectores(phi3)

Phi_norm = np.array([phi1_norm, phi2_norm, phi3_norm])

#Ahora normalizo las matrices
M_norm = normalizar_matrices(Phi_norm, M)
K_norm = normalizar_matrices(Phi_norm, K)

#Bien, ahora 

