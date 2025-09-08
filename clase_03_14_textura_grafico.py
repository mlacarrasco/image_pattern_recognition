"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Aplicación de analisis de Texturas en un grupo de imágenes
 Grafico de las propiedades extraidas
 Autor:. Miguel Carrasco (04-08-2021)
 rev.1.0
"""

import cv2
from matplotlib import markers
from numpy.ma import arange
from skimage.feature import graycomatrix, graycoprops
import numpy as np
from math import radians
from sklearn import preprocessing
from skimage.io import imread_collection
import pandas as pd
import matplotlib.pyplot as plt

# leemos todas las imagenes que cumplan el siguiente formato
col_dir = 'data/imagenes/textura*.tif'
col = imread_collection(col_dir)  #coleccion de imagenes
col_files = col.files
print(col_files)

# vamos a almacenar en la matriz F todos los descriptores
F = []

# >> recorremos la lista de archivos 
for filename  in col_files:
    
    # lectura de la imagen en formato .tif
    img = cv2.imread(filename)

    # convertimos la imagen a escala de grises
    gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_column = gray.reshape(-1,1) #la definimos como columna

    # Escalamos los datos en una matriz con menos valores
    new_scale = (0,15)
    new_gray = preprocessing.MinMaxScaler(new_scale).fit_transform(gray_column).astype(int)

    # redimensionamos la imagen 
    new_gray = new_gray.reshape(gray.shape)
    # --> algoritmo graycomatrix P01
    # numero de niveles de la imagen
    l = np.max(new_gray)+1  
    P_1_0 = graycomatrix(new_gray, distances=[1], angles=[radians(90)], levels=l, symmetric=False, normed=False)

    # extracción de caracteristicas a traves de greycomatrix
    features = ['contrast','correlation', 'dissimilarity','homogeneity','ASM','energy']
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
print(df)

# preparamos los datos para el grafico
feature_X = 'correlation'
feature_Y = 'energy'
title = f'{feature_X} versus {feature_Y}'


plt.figure()
for i, texture_file in enumerate(df.index):
    label =  texture_file.rsplit('_')[-1]
    plt.scatter(df[feature_X][texture_file], 
                df[feature_Y][texture_file], 
                marker='o', 
                label=label)
    plt.annotate(label, (df[feature_X][texture_file], df[feature_Y][texture_file]))

plt.title(title)
plt.xlabel(feature_X)
plt.ylabel(feature_Y)
plt.legend()
plt.show()