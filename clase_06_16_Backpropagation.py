""""
 Universidad Adolfo Ibañez
 Facultad de Ingeniería y Ciencias
 TICS 585 - Reconocimiento de Patrones en imágenes

 Ejemplo de algoritmo Backpropagation
 Basado en 
 https://stackoverflow.com/questions/47577794/impact-of-using-relu-for-gradient-descent
 
 Modificado por Miguel Carrasco (11-11-2022)
 rev.1.0
 """

import numpy as np
import matplotlib.pyplot as plt

# N es el tamaño de la muestra; 
# D_in es el numero de características o features;
# H es el numero de neuronas ocultas; 
# D_out es la domension de salida.
N, D_in, H, D_out = 4, 2, 30, 1

# Creamos datos de ejemplo de entrada 
# (x) y la clase de salida (y)
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inicializamos los pesos en forma aleatorio.
# Observe que esta red solo tiene una capa oculta 
# con H neuronas
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 0.002
loss_col = []
for t in range(200):
    # Forward pass: Calculamos el valor de Sigma
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)  # empleamos la función ReLU
    y_pred = h_relu.dot(w2)

    # Calcula y estima el error (loss)
    loss = np.square(y_pred - y).sum() # loss function
    loss_col.append(loss)
    print(t, loss, y_pred)

    # Realiza el procedimiento de backpropagation.
    #  Note que este procedimiento es empleado solo para una activación tipo ReLU
    grad_y_pred = 2*(y_pred - y) # error de la última capa
    grad_w2 = h_relu.T.dot(grad_y_pred)

    grad_h_relu = grad_y_pred.dot(w2.T) #error de la capa intermedia
    grad_h = grad_h_relu.copy()
    
    grad_h[h < 0] = 0  # derivada de ReLU
    grad_w1 = x.T.dot(grad_h)

    # actualizamos los pesos
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

plt.plot(loss_col)
plt.show()