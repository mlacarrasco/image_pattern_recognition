"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 Inteligencia de Negocios- TICS 423
 
 Analisis con componentes principales (PCA) para clasificación
 Miguel Carrasco (miguel.carrasco@uai.cl)
 version 1.0 (12/11/2019)
    
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification


def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    plt.figure(dpi=150)
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
    
    
# Create a simulated feature matrix and output vector with 100 samples,
features, output = make_classification(n_samples = 100,
                                       # ten features
                                       n_features = 10,
                                       # five features that actually predict the output's classes
                                       n_informative = 5,
                                       # five features that are random and unrelated to the output's classes
                                       n_redundant = 5,
                                       # three output classes
                                       n_classes = 2,
                                       # with 20% of observations in the first class, 30% in the second class,
                                       # and 50% in the third class. ('None' makes balanced classes)
                                       weights = [.2, .8],

                                       class_sep=1.5
                                       )

pd.DataFrame(features).head()

pca = PCA(n_components=2)
Z = pca.fit_transform(features)


plot_2d_space(features, output, 'Feature 1-2')
plot_2d_space(Z, output, 'PCA')



