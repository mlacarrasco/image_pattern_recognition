"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 Inteligencia de Negocios- TICS 423
 
 Miguel Carrasco (miguel.carrasco@uai.cl)
 version 1.0 (12/11/2019)
 
 Objetivo:
 
 1) Generar una distribución con datos aleatorios
 2) Generar nuevos datos a través de técnica Bootstrap (con repetición)
 2) Plotear la nueva distribución balanceada

"""

# scikit-learn bootstrap
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt


no_data = 60
no_data_bootstrap = 10000
bin_hist = 15

# distribución de datos normal
data = np.random.normal(0,0.1,[no_data])

print("Promedio de datos reales")
print(data.mean())
print(data.std())

plt.figure()
plt.hist(x=data,bins=bin_hist,color='#8634eb',alpha=0.7,rwidth=0.95)
plt.show()


# prepare bootstrap sample
boot = resample(data, replace=True, n_samples=no_data_bootstrap)

print("Promedio de datos con Bootstrap")
print(boot.mean())
print(boot.std())

# Calculamos el histograma a partir de los datos
bin_height,bin_boundary = np.histogram(boot,bins=10) 

plt.figure()
plt.hist(x=boot,bins=bin_hist,color='#8634eb',alpha=0.7,rwidth=0.95)
plt.show()
