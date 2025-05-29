import numpy as np


#Defino los datos conocidos del sistema
m1 = 0
m2 = 0
m3 = 0

c1 = 0
c2 = 0
c3 = 0

k1 = 0
k2 = 0
k3 = 0


#Las matrices a ensamblar son:
M = np.array([[m1, 0, 0],
              [0, m2, 0],
              [0, 0, m3]])

C = np.array([[c1, -c1, 0],
              [-c1, c1+c2, -c2],
              [0, -c2, c2+c3]])

K = np.array([[k1, -k1, 0],
             [-k1, k1+k2, -k2],
             [0, -k2, k2+k3]])

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