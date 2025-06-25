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

#Bien, ahora 

