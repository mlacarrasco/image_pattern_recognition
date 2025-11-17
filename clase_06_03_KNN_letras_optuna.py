"""
 Universidad Diego Portales
 Facultad de Ingeniería y Ciencias
 Reconocimiento de Patrones en imágenes
 Ejemplo completo de extracción y clasificación con K Vecinos Cercanos
 Características: 
    + Dataset: Imagen con Letras
    + Features: Momentos invariantes de Hu + Excentricidad
    + Validación Cruzada
    + Optimización de hiperparámetros con Optuna

    Autor: Miguel Carrasco (17-11-2025)
 rev.1.1
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from sklearn import neighbors
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import optuna

def feature_extraction():
    """
    Función que utiliza el algoritmo de extracción de 
    características de Hu y excentricidad
    """
    # Leemos la imagen y la clase
    img = cv2.imread('data/imagenes/sopa_letras.png')
    clase = pd.read_csv('data/clase_letras.csv')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = (gray < 1) * 1
    
    # Transformamos la imagen en objetos y extraemos las estadísticas por región
    regiones = label(bw)
    sts = regionprops(label_image=regiones)
    
    plt.figure()
    plt.imshow(regiones, cmap='jet')
    
    # Extraemos los descriptores
    data = []
    cont = 1
    
    for region in sts:
        xy = region.centroid
        hu = region.moments_hu
        so = [region.eccentricity]
        data.append(np.append(hu, so))
        # plt.text(xy[1], xy[0], f'{cont}', bbox=dict(facecolor='white', alpha=0.2))
        cont += 1
    
    # Construimos el dataset
    X = np.vstack(data)
    y = clase.to_numpy().ravel()
    
    return X, y     

def objective(trial, X, y):
    """
    Función objetivo para optimización con Optuna
    """
    # Aplicamos el algoritmo KNN
    n_neighbors = trial.suggest_int('n_neighbors', 5, 20)
    model = neighbors.KNeighborsClassifier(n_neighbors, weights='distance', algorithm='ball_tree')
    
    # k-cross validation
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluación del modelo
    scores = cross_val_score(model, X, y, cv=k_fold, n_jobs=1)
    accuracy = scores.mean()
    
    # Retornamos el accuracy para maximizar
    return accuracy

#*********************************
#      PROGRAMA PRINCIPAL        *
#*********************************

# Extracción de características
X, y = feature_extraction()

# Optimización con Optuna
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, X, y), n_trials=100, show_progress_bar=True)

# Resultados
best_params = study.best_params
best_n_neighbors = best_params["n_neighbors"]
print(f'Mejor número de vecinos: {best_n_neighbors}')
print(f'Mejor accuracy: {study.best_value:.4f}')

# Entrenamiento con mejores parámetros y predicción
model = neighbors.KNeighborsClassifier(best_n_neighbors, weights='distance', algorithm='ball_tree')
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(model, X, y, cv=k_fold, n_jobs=1)

# Matriz de confusión
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusión - KNN')
plt.show()

print("\nMatriz de Confusión:")
print(cm)