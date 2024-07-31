"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Aplicación del algoritmo SBS para selección de características
 Autor:. Miguel Carrasco (31-08-2023)
 rev.1.0
"""
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from numpy.matlib import repmat
from numpy.linalg import inv

#********************************************************************
def sfs_procedure( data, clases, features, sel_features):
    #sfs_procedure: Esta función realiza el algoritmo SFS 

    # Input: data: dataframe con los datos del problema
    #        clases: ndarray con valores de las clases.
    #        features: nombre de las características del dataset
    #        sel_features: numero de características a escojer


    #numero de columnas
    cols = data.shape[1]   
    print('columnas: ',cols)
    
    id_clases = np.unique(clases)      # valores unicos de las clases
    u_clases = len(np.unique(clases))  # número de clases
    
    #% combinaciones caracteristicas 
    combs = np.arange(cols)
    umbral = sel_features
    
    t = 1
    while(t<=umbral):
        J = np.zeros(len(combs))
        
        for i in range(len(combs)):
            feat = combs[i]
            sub_data = []
            for k in id_clases:
                #%Seleccionamos las caracteristicas
                id_class= clases==k
                tmp = data[id_class]
                sub_data.append(tmp[features[feat]].to_numpy())
            
            # Determinamos el indice de Fisher para dichas columnas
            J[i],Gn, Vk = fisher_extraction_list(sub_data, t, u_clases)
            oldJ = J
            old_comb= combs
            
            #% impresion resultados
            print(f'  > feature : ({feat}) >> J[{i}]= \t {J[i]} ')

        
        t = t+1 #incremento para ciclo while
        best= np.argmax(J)
        vector= combs[best]
        new_cmb =  np.setxor1d(np.arange(cols), vector).reshape(-1,1)
        old_cmb = repmat(vector,new_cmb.shape[0],1)
        combs =  np.append(old_cmb, new_cmb, axis=1)
        #fin ciclo while
        

    print(f'\n Mejor Combinacion de {umbral} caracteristicas:')
    best = np.argmax(oldJ)

    feat = old_comb[best]
    feature_selected = features[feat]
    return feature_selected

#********************************************************************
def fisher_extraction_list(data, cols,  clases):
    # función calcula el índice de Fisher para un determinado 
    # conjunto de datos.
    # Input: data: lista con submatrices (una por cada clase)
    #        cols: numero de columnas
    #       clases: lista con valores de las clases.

    # unimos los datos en una sola matriz
   
    if (cols==1):
        D = np.hstack(data)
        total = len(D)
        # Buscamos la media de todos los datos
        Vm = np.mean(D)
    else:
        D = np.vstack(data)
        Vm = np.mean(D, axis =0)
        total = D.shape[0]

    # inicializacion de matrices
    p = np.zeros((clases,1))
    Vk = np.zeros((clases,cols))
    Gn = []
    
    # Centrado
    for i in range(clases):
        Vk[i,:] = np.mean(data[i], axis=0)           # centramos las medias de cada clase
        pts = data[i].shape[0]                          # número de puntos de ese clase
        if cols==1:
            Gn.append(data[i]-Vm)                       # centramos los puntos de cada clase
        else:
            Gn.append(data[i]-repmat(Vm,pts,1))         # centramos los puntos de cada clase

        
        p[i] = data[i].shape[0] / total # probabilidad de cada cluster
        
    # Inicialización
    Cb = np.zeros((cols,cols))
    Cw = np.zeros((cols,cols))

    # construccion de matrices inter e intraclase
    for k in range(clases):
        Cb = Cb + p[k]*np.matmul((Vk[k,:]-Vm).reshape(cols,1), (Vk[k,:]-Vm).reshape(1,cols))
        MGn = np.array(Gn[k])
        Cw = Cw + p[k]*np.cov(MGn.T)

    #Calculamos el índice de Fisher
    J = np.trace(np.matmul(inv(Cw),Cb))
    return J, Gn, Vk


#*********************************
#      PROGRAMA PRINCIPAL  SFS   *
#*********************************
df = load_breast_cancer()

#preparamos el dataset en sus componentes
features = df.feature_names
data = StandardScaler().fit_transform(df.data)
data = pd.DataFrame(df.data, columns=features)

clases = df.target

seleccion = 3 #<--- número de características seleccionadas 

#buscamos las mejores características con SFS
feature_selected  = sfs_procedure( data, clases, features, seleccion)

print(feature_selected)
data_selection= data[feature_selected]

if seleccion>2:
    xx = data_selection.iloc[:,0]
    yy = data_selection.iloc[:,1]
    zz = data_selection.iloc[:,2]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(xx,yy,zz, c=clases, edgecolor='k', s=20)
    plt.show()
