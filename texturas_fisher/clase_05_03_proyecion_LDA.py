"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Test Fisher 2D LDA
 Autor: Miguel Carrasco (05-08-2021)
 rev.1.0

"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand, random
from itertools import combinations
from math import pi,radians
from numpy.matlib import repmat
from numpy.linalg import inv

# FUNCIONES

def ploteo_datos_cluster(Gn, Vk, color, mker):
    plt.figure()
    for i in range(Gn.shape[0]):
        plt.scatter(Gn[i,:,0], Gn[i,:,1], marker=mker[i], color=color[i])
        plt.scatter(Vk[i,0],Vk[i,1], marker='*', color='black', s=40)

    plt.title('Datos')
    plt.grid('both', alpha=0.2)
    plt.show()

def datos(m1, s, N):
    # Generacion de datos inventados
    if (N>5):
        g = 1.5
        x1 = np.array([(np.random.rand(N,1)+m1)*s,
                       (np.random.rand(N,1)+m1+g)*s]).reshape(-1,2)
        th = radians(np.random.randint(0,360))
        ra = np.array([[np.cos(th), -np.sin(th)],
                        [np.sin(th), np.cos(th)]])
            
        x1 = np.matmul(x1,ra)
        X = np.array([x1]).reshape(-1,2)
    else:
        print('Ingrese más de 5 puntos')
    
    return X

def datosHard():
    D=[[-0.1306, -1.1811, 1],
    [-0.7306,   -0.9711, 1],
    [-0.8506,   -0.9011, 1],
    [-0.6806,   -0.9611, 1],
    [-0.3806,   -0.3111, 1],
    [-0.2806,   -0.4311, 1], 
    [-0.2706,    0.1489, 2],
    [-0.1106,    0.5089, 2],
    [ 0.0594,    0.9689, 2],
    [-0.0406,    1.4789, 2],
    [-0.0706,    1.0189, 2],
    [ 0.0094,    0.7389, 2],
    [0.9294,    0.2089, 3],
    [0.3094,   -0.0311, 3],
    [0.3594,    0.2989, 3],
    [0.4694,   -0.4011, 3],
    [0.5194,    0.1789, 3],
    [0.8894,   -0.3611, 3]]

    D = np.array(D)
    
    G =  np.zeros((3,6,2))
    for i in range(3):
        idx= D[:,2]==i+1
        G[i,:,:]= D[idx,0:2]

    return G



#******************************
#      PROGRAMA PRINCIPAL     *
#******************************

#  PARAMETROS: Creación de datos aleatorios
clus = 3      # numero de clusters
pts  = 8      # numero de puntos por cluster
opcion = 0    # 0: datos clase no.8
              # 1: datos aleatorios

color = ['blue','red','green','cyan', 'black', 'yellow']
mker = ['o','o','o','s', 's', 'd']

if not(opcion):
    G = datosHard()
    clus = 3
    pts = 6
else:
    # Datos aleatorios
    G =  np.zeros((clus,pts,2))
    for i in range(clus):
        m1 = np.random.rand(1)*20
        G[i,:,:]= datos(m1,0.3, pts)   #% Cluster i

#% combinaciones de pares
combs = np.array(list(combinations(np.arange(clus),2))).reshape(-1,2)

# Buscamos la media
D = G.reshape(-1,2)
Vm= np.mean(D, axis =0)

p = np.zeros((clus,1))
Vk = np.zeros((clus,2))
Gn = np.zeros((clus,pts,2))
# Centrado
for i in range(clus):
    Vk[i,:] = np.mean(G[i,:,:], axis=0)-Vm         # centramos las medias
    Gn[i,:,:]= G[i,:,:]-repmat(Vm,pts,1)   # centramos los puntos
    p[i] = len(G[i,:,:])/ D.shape[0]        # probabilidad de cada cluster

# Inicialización
Cb = np.zeros((2,2))
Cw = np.zeros((2,2))

ploteo_datos_cluster(Gn, Vk, color, mker)

for k in range(clus):
    Cb = Cb + p[k]*np.matmul((Vk[k,:]-Vm).reshape(2,1), (Vk[k,:]-Vm).reshape(1,2))
    Cw = Cw + p[k]*np.cov(Gn[k,:,:].T)


#Calculamos el índice de Fisher
J = np.trace(np.matmul(inv(Cw),Cb))
print(f' Fisher value:{J}')

# Calculamos las rectas
X = np.array([-2, 2])

W = np.zeros((2,combs.shape[0]))
y = np.zeros((2,combs.shape[0]))


plt.figure()
for j in range(combs.shape[0]):

    for i in range(Gn.shape[0]):
        plt.scatter(Gn[i,:,0], Gn[i,:,1], marker=mker[i], color=color[i])
        plt.scatter(Vk[i,0],Vk[i,1], marker='*', color='black', s=40)

    plt.title('Datos')
    plt.grid('both', alpha=0.2)
    a = combs[j,0]
    b = combs[j,1]
    
    W[:,j] = np.matmul(inv(Cw), (Vk[a,:].T - Vk[b,:].T))
    # pendiente
    m = W[0,j]/W[1,j] 
    y[:,j] = m * X
    lbl= f'Combinacion: {a}-{b}, m:{m:2.2}'
    plt.plot(X,y[:,j], color=color[j], label=lbl )
    plt.title('Proyección de líneas W')
    plt.axis([np.min(Gn[:,:,0])*1.5, 
             np.max(Gn[:,:,0])*1.5,
             np.min(Gn[:,:,1])*1.5, 
             np.max(Gn[:,:,1])*1.5])

plt.legend()
plt.show()