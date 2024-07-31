"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Aplicación de la herramienta SMOTE a datos de un formulario
 
 Miguel Carrasco (miguel.carrasco@uai.cl)
 version 1.0 (25/11/2021)

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE


#diccionarios de color
colormap = {'Azul':'b', 'Rojo':'r'}
colorSmote = {0:'b', 1:'r'}

# data normal sample
data = pd.read_csv('data/data_Bootstrap.csv')
print(data.columns)

estatura = np.array(data[data.columns[1]])
peso     = np.array(data[data.columns[2]])
nota     = np.array(data[data.columns[3]])
color    = data[data.columns[4]]

# seleccion de columnas
X = np.vstack((estatura,peso,nota)).T
y = list(map(lambda i:0 if i=='Azul' else 1, color))


#aplicamos SMOTE
smote = SMOTE() 
X_sm, y_sm =   smote.fit_resample(X,y)
y_color_sm = pd.Series(y_sm)

# generamos los gráficos
fig, ax = plt.subplots(1,2, figsize=(10,5))

# primer grafico
ax[0].scatter(X[:,0], X[:,1], c= color.replace(colormap))
ax[0].set_xlabel('Estatura (cm)')
ax[0].set_ylabel('Peso (kg)')
ax[0].set_title('Datos originales (sin SMOTE)')
ax[0].grid(alpha=0.2)

# segundo grafico
ax[1].scatter(X_sm[:,0], X_sm[:,1], c=y_color_sm.replace(colorSmote) )
ax[1].set_xlabel('Estatura (cm)')
ax[1].set_ylabel('Peso (kg)')
ax[1].set_title('con SMOTE')
ax[1].grid(alpha=0.2)

plt.show()
