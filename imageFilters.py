import cv2 as cv
import numpy as np
import sys

from cv2 import getStructuringElement
from cv2 import MORPH_RECT

nom_imagen = "examen_b.tif"

img = cv.imread(nom_imagen)
cv.imshow("Display window", img)
k = cv.waitKey(0)

#Filtro de Sobel usando matriz 3x3
#https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html
gradX = cv.Sobel(img,-1,1,0,ksize=3)
gradY = cv.Sobel(img,-1,0,1,ksize=3)
grad = cv.addWeighted(gradX, 0.5, gradY, 0.5,0)
cv.imshow("Sobel", grad)
cv.imwrite("Sobel.tif", grad)
k = cv.waitKey(0)

#Rotar la imagen 
#https://pyimagesearch.com/2021/01/20/opencv-rotate-image/
(h, w) = img.shape[:2]
(cX, cY) = (w // 2, h // 2)
M = cv.getRotationMatrix2D((cX, cY), -14, 1.0)
img = cv.warpAffine(img, M, (w, h))
cv.imshow("Rotacion", img)
cv.imwrite("Rotacion.tif", img)
k = cv.waitKey(0)

#Recortar la imagen (1000 x 667)
img = img[82:428, 245:595]
cv.imshow("Recortada", img)
cv.imwrite("Recortada.tif", img)
k = cv.waitKey(0)

#Filtro de dilatacion
kernel = getStructuringElement(MORPH_RECT,(3,3))
img = cv.dilate(img,kernel,iterations=1)
cv.imshow("Dilatacion", img)
cv.imwrite("Dilatacion.tif", img)
k = cv.waitKey(0)

#Filtro laplaciano
filter = np.array([[0, -1, 0], 
                   [-1, 5, -1], 
                   [0, -1, 0]])
img = cv.filter2D(img,-1,filter)
img = cv.filter2D(img,-1,filter)
cv.imshow("Filtro de Laplace", img)
cv.imwrite("Laplace.tif", img)
k = cv.waitKey(0)

#Erosion seguida de dilatacion (Opening)
img = cv.erode(img,kernel,iterations=1)
img = cv.dilate(img,kernel,iterations=1)
cv.imshow("Final con Opening", img)
cv.imwrite("Final.tif", img)
k = cv.waitKey(0)

