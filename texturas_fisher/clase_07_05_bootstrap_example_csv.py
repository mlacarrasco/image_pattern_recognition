"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 Inteligencia de Negocios- TICS 423
 
 Miguel Carrasco (miguel.carrasco@uai.cl)
 version 1.0 (12/11/2019)
 
 Objetivo:
 
 1) Emplear una técnia de re-muestreo con repetición Bootstrap
 2) Leer datos csv de formulario https://forms.gle/rio9WtVzL5oRi3ny6
 3) Plotear los datos en un gráfico de histograma
    
"""

# scikit-learn bootstrap
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import KernelDensity


# data normal sample
data = pd.pandas.read_csv('data/data_Bootstrap.csv')
print(data.columns)

estatura = np.array(data[data.columns[1]])
peso     = np.array(data[data.columns[2]])
nota     = np.array(data[data.columns[3]])

plt.figure()
plt.scatter(estatura, peso)
plt.show()

print("Promedio de datos reales: ")
print(estatura.mean())
print(peso.mean())
print(nota.mean())

# graficamos los datos originales
fig, ax = plt.subplots(3,gridspec_kw={'hspace': 0.5}, figsize=(4,8))
ax[0].hist(x=estatura,bins=20,color='#8634eb',alpha=0.7,rwidth=0.95) 
ax[1].hist(x=peso,bins=20,color='#8634eb',alpha=0.7,rwidth=0.95)
ax[2].hist(x=nota,bins=20,color='#8634eb',alpha=0.7,rwidth=0.95)  
ax[0].set_title(data.columns[1])
ax[1].set_title(data.columns[2])
ax[2].set_title(data.columns[3])
plt.show()


# Generamos una nuestra distribución a partir de los datos originales
boot = resample(nota, replace=True, n_samples=4000)

#print('Bootstrap Sample: %s' % boot)
print("Promedio de datos con Bootstrap")
print(boot.mean())
print(boot.std())


plt.figure()
plt.title("Histograma (Bootstrap): "+ data.columns[3])
plt.hist(x=boot,bins=10,color='#8634eb',alpha=0.7,rwidth=0.95)
plt.show()


