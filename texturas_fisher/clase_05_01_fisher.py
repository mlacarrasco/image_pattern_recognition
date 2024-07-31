"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Ejemplo test Fisher en una dimensión
 Autor: Miguel Carrasco (06-08-2021)
 rev.1.0

"""
import matplotlib.pyplot as plt
import numpy as np
from math import pi

def  fisher(X1, Y1, X2, Y2):
 
    n = len(X1)
    # normalizamos la distribucion
    Y1n = Y1/np.sum(Y1)
    Y2n = Y2/np.sum(Y1)
    
    #media mu
    m1 = np.sum(Y1n*X1)
    m2 = np.sum(Y2n*X2)
    
    print(m1)
    print(m2)

    # desviacion estandar
    s1 = np.sqrt( sum( ((X1-m1)**2)*Y1n ))
    s2 = np.sqrt( sum( ((X2-m2)**2)*Y2n ))
    
    # Descriptor de Fisher
    J= ((m1-m2)**2)/ (s1**2+s2**2)
    return J


 
def datg(N, s, mu):
    x = np.linspace(-1,1, N)
    fx = (1/(s*np.sqrt(2*pi)))* np.exp(-0.5*((x-mu)/s)**2)

    return x,fx


X1,Y1 = datg(300,0.15,-0.1)
X2,Y2 = datg(300,0.21,0.1)
 
J = fisher(X1,Y1,X2,Y2)
print(f' J:{J}')
 
plt.figure()
plt.plot(X1,Y1, color='blue')
plt.plot(X2,Y2, color='red')
plt.show()
