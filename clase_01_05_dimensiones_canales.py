import cv2

#lectura de imagen
img = cv2.imread('data/imagenes/lena.png')

#dimensiones
dimensions = img.shape
 
height   = img.shape[0]
width    = img.shape[1]
channels = img.shape[2]
 
print('Image Dimension: ',dimensions)
print('Image Height   : ',height)
print('Image Width    : ',width)
print('Channels       : ',channels) 
