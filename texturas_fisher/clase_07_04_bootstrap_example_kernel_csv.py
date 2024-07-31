"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 Inteligencia de Negocios- TICS 423
 
 Miguel Carrasco (miguel.carrasco@uai.cl)
 version 1.0 (08/11/2019)

 Semana 12
 
 Objetivo:
 
 1) Emplear una técnia de re-muestreo con repetición Bootstrap
 2) Leer datos csv de formulario https://forms.gle/rio9WtVzL5oRi3ny6
 2 ) http://bit.ly/3US3m86
 3) Emplear un kernel de densidad para analizar la distribución de los datos
"""

# scikit-learn bootstrap
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import KernelDensity


# data normal sample
data = pd.read_csv('data/ejemplo_bootstrap.csv')
print(data.columns)

estatura = np.array(data[data.columns[1]])
peso     = np.array(data[data.columns[2]])
nota     = np.array(data[data.columns[3]])

estatura =  estatura[:, np.newaxis]
peso     =  peso[:, np.newaxis]
nota     =  nota[:, np.newaxis]


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


# prepare bootstrap sample
boot = resample(nota, replace=True, n_samples=400)

#print('Bootstrap Sample: %s' % boot)
print("Promedio de datos con Bootstrap")
print(boot.mean())
print(boot.std())


plt.figure()
plt.hist(x=boot,bins=20,color='#8634eb',alpha=0.7,rwidth=0.95)
plt.show()


#definimos los valores del eje X (luego los ocuparemos para evaluar la función de Kernel)
X_plot=np.linspace(np.min(nota), np.max(nota), 100)[:, np.newaxis]
# tipos de kernel
#['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine'] #
kdd = KernelDensity(kernel='gaussian', bandwidth=1.5).fit(nota)  # ajuste del kernel a los datos
log_dens = kdd.score_samples(X_plot)  

# Calculamos el histograma a partir de los datos
bin_height,bin_boundary = np.histogram(boot,bins=10) 
bin_height = bin_height/float(max(bin_height))         # normalizamos la altura a un maximo de 1.0
bin_height = bin_height*0.2#max(np.exp(log_dens))          # normalizamos  por el maximo del kernel


plt.figure()
plt.title("Histograma (Bootstrap Normalizado): "+ data.columns[1])
plt.plot(boot[:, 0], - 0.1 * np.random.random(boot.shape[0]), '.k')
plt.plot(X_plot[:, 0], np.exp(log_dens), 'r')
plt.show()

