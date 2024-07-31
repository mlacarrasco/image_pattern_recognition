"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Uso de Normalización de datos con StandardScaler
 Autor:. Miguel Carrasco (16-08-2021)
 rev.1.0
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(1)

#generamos datos aleatorios
data = np.random.randint(0,10,(20,1))

#normalizamos los datos
scaled = StandardScaler().fit_transform(data)

#ploteo de figuras
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.plot(data)
plt.title('Data Original')

plt.subplot(122)
plt.plot(scaled)
plt.title('Datos normalizados (media 0, std=1')

plt.show()




