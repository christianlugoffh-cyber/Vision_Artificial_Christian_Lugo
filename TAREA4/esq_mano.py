import cv2
import numpy as np

# Cargar imagen
img = cv2.imread("mano.jpeg")

# Escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Suavizado más suave (para no deformar los dedos)
blur = cv2.GaussianBlur(gray,(5,5),0)

# Umbral OTSU
_,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Morfología más ligera (kernel pequeño)
kernel = np.ones((3,3),np.uint8)

# Solo limpieza ligera
th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

# -------- ESQUELETO --------

size = np.size(th)
skel = np.zeros(th.shape,np.uint8)

element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

done = False
img_temp = th.copy()

while not done:

    eroded = cv2.erode(img_temp,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(img_temp,temp)
    skel = cv2.bitwise_or(skel,temp)
    img_temp = eroded.copy()

    zeros = size - cv2.countNonZero(img_temp)
    if zeros==size:
        done = True

# Mostrar resultados
cv2.imshow("Original Mano",img)
cv2.imshow("Preprocesada Mano",th)
cv2.imshow("Esqueleto Mano",skel)

cv2.waitKey(0)
cv2.destroyAllWindows()