"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Aplicación del algoritmo LASSO para selección de características
 Autor:. Miguel Carrasco (06-08-2023)
 rev.1.1
"""

import cv2
from skimage.feature import graycomatrix, graycoprops
import numpy as np
from math import radians
from sklearn import preprocessing
from skimage.io import imread_collection
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV

def lasso_procedure(df, features):
    
    #separamos el dataset en entrenamiento y testing
    X = df[features]
    y = df['clase']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=40)
    # normalizamos los datos de test y entrenamiento
    pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('modelo',Lasso())
    ])

    search = GridSearchCV(pipeline,
                      {'modelo__alpha':np.arange(0.01,2,0.01)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )

    search.fit(X_train,y_train)
    print(search.best_params_)

    coefficients = search.best_estimator_.named_steps['modelo'].coef_

    importance = np.abs(coefficients)
    print(importance)

    
    return np.array(features)[importance > 0]




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
    print(col_files)
    
    # vamos a almacenar en la matriz F todos los descriptores
    F = []

    # >> recorremos la lista de archivos 
    fig = plt.figure(figsize=(15, 7))

    for i,filename  in enumerate(col_files):
        # lectura de la imagen en formato .tif
        img = cv2.imread(filename)

        # convertimos la imagen a escala de grises
        gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_column = gray.reshape(-1,1) #la definimos como columna    

        fig.add_subplot(3, 5, i+1)
        plt.imshow(gray, cmap='gray')
        plt.title(f'textura {i}')
        
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
    plt.show()

    # almacenamos los datos en un dataframe
    df = pd.DataFrame(F, columns=features, index=col_files)
    df['clase'] = clase
    print(df)
    return df, features


if __name__=='__main__':
    level = 32            # numero de niveles (Haralick)

    # extracción de caracteristicas (incluye la clase de cada imagen)
    df, feat = feature_extraction (level)
    feat_selected = lasso_procedure(df, feat)
    print(feat_selected)
   