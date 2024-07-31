"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Aplicación del algoritmo OPTUNA
 Autor:. Miguel Carrasco (04-08-2021)
 rev.1.0
"""
import cv2
from skimage.feature import graycomatrix, graycoprops
import numpy as np
from math import radians
from sklearn import preprocessing
from skimage.io import imread_collection
import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations
from math import pi 
from numpy.matlib import repmat
from numpy.linalg import inv
import optuna

def objective(trial):
    level = trial.suggest_int("level", 4, 60)

    no_cols = 3
    df = feature_extraction(level)
    # no empleamos la ultima columna ya que en ella está la clase
    cols = df.shape[1] -1    #numero de columnas
    col_features = df.columns

    # numero de clases
    clases = len(df['clase'].unique())

    #% combinaciones caracteristicas 
    combs = np.array(list(combinations(np.arange(cols),no_cols))).reshape(-1,no_cols)
    J = []

    for i in range(len(combs)):
        a = combs[i]
        sub_data = []
        for k in range(1,clases+1):
            #%Seleccionamos las caracteristicas
            id_class= df['clase']==k
            tmp = df[id_class]
            sub_data.append(tmp[col_features[a]].to_numpy())
        
        # Determinamos el indice de Fisher para dichas columnas
        J.append(fisher_extraction_list(sub_data,clases))

    out_value = np.max(J)
    id_best_comb = np.argmax(J)
    #print(combs[id_best_comb])
    
    #retornamos dos valores
    return out_value


    
def fisher_extraction_list(data, clases):
    # función calcula el índice de Fisher para un determinado 
    # conjunto de datos.
    # Input: data: lista con submatrices (una por cada clase)
    #      clases: lista con valores de las clases.

    # unimos los datos en una sola matriz
    D = np.vstack(data)
    # Buscamos la media de todos los datos
    Vm = np.mean(D, axis =0)
    
    # numero de columnas
    cols = D.shape[1] 
    
    # inicializacion de matrices
    p = np.zeros((clases,1))
    Vk = np.zeros((clases,cols))
    Gn = []
    
    # Centrado
    for i in range(clases):
        Vk[i,:] = np.mean(data[i], axis=0)-Vm    # centramos las medias de cada clase
        pts = data[i].shape[0]                   # numero de puntos de ese clase
        Gn.append(data[i]-repmat(Vm,pts,1))      # centramos los puntos de cada clase
        p[i] = data[i].shape[0] /len(D)          # probabilidad de cada cluster

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
    return J

    
def feature_extraction(level):
    # funcion utiliza el algoritmo de extracción de 
    # características de haralick
    # Input:    level: numero de niveles de matriz de salida que utiliza Haralick

    #caracteristicas de matriz de textura
    features = ['contrast','correlation', 'dissimilarity','homogeneity','ASM','energy']
    
    #clases de las imagenes (clasificación supervisada)
    clase = np.array([1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 3, 3, 3, 3, 3])

    # leemos todas las imagenes que cumplan el siguiente formato
    col_dir = 'texturas_fisher/textura*.tif'
    
    col = imread_collection(col_dir)  #coleccion de imágenes
    col_files = col.files
    
    # vamos a almacenar en la matriz F todos los descriptores
    F = []

    # >> recorremos la lista de archivos 
    
    for i,filename  in enumerate(col_files):
        # lectura de la imagen en formato .tif
        img = cv2.imread(filename)

        # convertimos la imagen a escala de grises
        gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_column = gray.reshape(-1,1) #la definimos como columna       

        # Escalamos los datos en una matriz con menos valores
        new_scale = (0,level)
        new_gray = preprocessing.MinMaxScaler(new_scale).fit_transform(gray_column).astype(int)
        
        # redimensionamos la imagen 
        new_gray = new_gray.reshape(gray.shape)
        
        # --> algoritmo graycomatrix P01
        # numero de niveles de la imagenq
        l = np.max(new_gray)+1  
        P_1_0 = graycomatrix(new_gray, distances=[2], angles=[radians(90)], levels=l, symmetric=False, normed=True)

        # extracción de caracteristicas a traves de greycomatrix
        
        S = []
        # para cada imagen extraemos las caracteristicas definidas en la lista features
        for ft in features:
            sts = graycoprops(P_1_0, ft).squeeze()
            S.append(float(sts))
            
        #agregamos los características en la matriz F
        F.append(S)

    # >> FIN ciclo para cada imagen de textura

    # almacenamos los datos en un dataframe
    df = pd.DataFrame(F, columns=features, index=col_files)
    df['clase'] = clase
    return df

#*********************************
#      PROGRAMA PRINCIPAL  SBS   *
#*********************************
colores = {1:'red', 2:'blue', 3:'green'}
sel_features = 3     # numero de características seleccionadas

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

best_params = study.best_params
found_level = best_params["level"]
print(f'best level {found_level}')
