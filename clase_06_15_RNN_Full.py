"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Ejemplo completo de extracción y redes neuronales 

 Características: 
    + Dataset: Imagen con Letras
    + Features: Momentos invariantes de Hu

 Autor:. Miguel Carrasco (09-11-2023)
 rev.1.0
"""

import cv2
import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from sklearn import tree

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

# ////////////////////////////////////////////////
# ////////////  CLASE del modelo NN  ////////////
class ModeloLetras(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 15)
        self.act1 = nn.ReLU()
        self.output = nn.Linear(15, 6)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.output(x)
        return x

# ////////////////////////////////////////////////
def feature_extraction(img, clase):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = (gray< 1)*1

    # transformamos la imagen en objetos y extraemos las estadísticas por region
    regiones = label(bw)
    sts = regionprops(label_image=regiones)

    plt.figure()
    plt.imshow(regiones, cmap='jet')

    # extraemos los descriptores
    data = []
    cont = 1
    for region in sts:
        xy = region.centroid
        hu = region.moments_hu
        so = [region.eccentricity]
        data.append(np.append(hu, so))
        plt.text(xy[1], xy[0], f'{cont}', bbox=dict(facecolor='white', alpha=0.2))
        cont +=1

    plt.show()
    
    # construimos el dataset
    X = np.vstack(data)
    y = clase.to_numpy().ravel()
    
    return X,y

# ////////////////////////////////////////////////
def training_RNN_model(X,y):

    
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y.reshape(-1,1))
    y = ohe.transform(y.reshape(-1,1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = ModeloLetras()
    print(model)

    loss_fn = nn.CrossEntropyLoss()  #modelo para múltiples clases MSELoss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 50
    batch_size = 5
    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')

    return model, epoch, optimizer, X_test, y_test

# ////////////////////////////////////////////////
def evaluacion_RNN(model, epoch, optimizer, X_test, y_test):
    
    # compute accuracy (no_grad is optional)
    # with torch.no_grad():
    y_pred = model(X_test)


    accuracy = (y_pred.round() == y_test).float().mean()
    print(f"Accuracy {accuracy}")

    #guardamos el modelo para no tener que re-entrenarlo.
    print('model saving....')

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, 'data/model/modelo_4.pth')

    return accuracy


# ****************************************************************
# ***************************** MAIN ***************************** 
# ****************************************************************
if __name__ == '__main__':
    
    # Leemos la imagen y la clase
    img = cv2.imread('data/imagenes/sopa_letras.png')
    clase = pd.read_csv('data/clase_letras.csv')
 
    #extracción de características
    X,y = feature_extraction(img, clase)
    
    #entrenamiento del modelo
    model, epoch, optimizer, X_test, y_test= training_RNN_model(X,y)

    acc = evaluacion_RNN(model, epoch, optimizer, X_test, y_test)