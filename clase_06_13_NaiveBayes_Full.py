"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Ejemplo completo de extracción y clasificacion con NaiveBayes

  Características: 
    + Dataset: Imagen con Letras
    + Features: Momentos invariantes de Hu
    + Validación Cruzada

 Autor:. Miguel Carrasco (09-11-2021)
 rev.1.0
"""

import cv2
import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Leemos la imagen y la clase
img = cv2.imread('data/imagenes/sopa_letras.png')
clase = pd.read_csv('data/clase_letras.csv')
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
    xy= region.centroid
    hu = region.moments_hu
    data.append(hu)
    cont +=1

# construimos el dataset
X = np.vstack(data)
y = clase.to_numpy().ravel()

# Clasificador Naïve  Bayes
model = GaussianNB()

# k-cross validation
k_fold = KFold(n_splits=5 , shuffle=True, random_state=None)

#no necesario..obtener indices
indices = k_fold.split(X)

# evaluación del modelo
scores =  cross_val_score(model, X, y, cv=k_fold, n_jobs=1)
y_pred =  cross_val_predict(model, X, y, cv=k_fold, n_jobs=1)
print(f'Rendimiento promedio clasificador:{scores.mean()}')

# matriz de confusión
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['a', 'b', 'c', 'e', 'f', 'g'])
disp.plot()

#resultados finales
plt.figure()
plt.imshow(regiones, cmap='jet')
letras ={0:'a', 1:'b', 2:'c', 3:'e', 4:'f', 5:'g'}
for region, clase in zip(sts, y_pred):
    xy= region.centroid
    plt.plot([xy[1],xy[1]+20], [xy[0],xy[0]+10], color='red')
    plt.text(xy[1]+20, xy[0]+20, f'{letras[clase]}', bbox=dict(facecolor='red', alpha=0.4))


# graficos
plt.figure()
plt.bar(x=np.arange(len(scores)), height= scores)
plt.title('Accuracy')
plt.show()