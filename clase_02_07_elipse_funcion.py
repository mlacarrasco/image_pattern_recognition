"""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Ejemplo que interpola una elipse en la region
 Función de skimage
 Autor:. Miguel Carrasco (02-08-2021)
 rev.1.0
 
"""
import numpy as np
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

points = [(1,2),(1,3),
          (2,1),(2,2),(2,3),(2,4),
          (3,1),(3,2),(3,4),
          (4,1),(4,4),(4,5),
          (5,1),(5,5),
          (6,1),(6,5),
          (7,1),(7,2),(7,5),
          (8,2),(8,5),
          (9,2),(9,3),(9,5),
          (10,3),(10,4),(10,5),
          (11,4),(11,5)]
          

a_points = np.array(points)
x = a_points[:, 0]
y = a_points[:, 1]


ell = EllipseModel()
ell.estimate(a_points)
xc, yc, a, b, theta = ell.params


print("centro = ",  (xc, yc))
print("angulo de rotación = ",  theta)
print("ejes = ", (a,b))

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
ax[0].scatter(x,y)

ax[1].scatter(x, y)
ax[1].scatter(xc, yc, color='red', s=100)
ax[1].set_xlim(x.min()-1, x.max()+1)
ax[1].set_ylim(y.min()-1, y.max()+1)

ell_patch = Ellipse((xc, yc), 2*a, 2*b, theta*180/np.pi, edgecolor='red', facecolor='none')

ax[1].add_patch(ell_patch)
plt.show()