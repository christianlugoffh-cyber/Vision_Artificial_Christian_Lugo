import cv2
import numpy as np
np.seterr(over='raise')
def contraste_brillo(imagen, contraste, brillo):
    imagen_procesada = np.zeros_like(imagen)
    h,w = imagen.shape
    for y in range(h):
        for x in range(w):
            try:
                imagen_procesada[y,x] = np.clip(contraste * imagen[y,x] + brillo, 0, 255).astype(np.uint8)
            except FloatingPointError as e:
                imagen_procesada[y,x] = 255
    return imagen_procesada
ruta = "lena.jpeg"
imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
h,w = imagen.shape
imagen_procesada = contraste_brillo(imagen, 1.2, -30)
imagenes_comparadas = np.hstack((imagen, imagen_procesada))
cv2.imshow("Comparacion", imagenes_comparadas)
cv2.waitKey(0)
imagen_color = cv2.imread(ruta)
cv2.imshow("Imagen a color", imagen_color)
cv2.waitKey(0)
def contraste_brillo_mat(imagen, contraste, brillo):
    imagen_p = imagen.astype(np.float32)
    return np.clip(contraste * imagen_p + brillo, 0, 255).astype(np.uint8)
imagen_color_procesada = contraste_brillo_mat(imagen_color, 2,-60)
cv2.imshow("Imagen a color procesada", imagen_color_procesada)
cv2.waitKey(0)
imagen_color_HSV = cv2.cvtColor(cv2.imread(ruta), cv2.COLOR_BGR2HSV)
imagen_color_HSV.shape
imagen_v = imagen_color_HSV[:,:,2]
imagen_color_HSV_procesada = contraste_brillo_mat(imagen_v, 1.2, -30)
imagen_color_HSV[:,:,2] = imagen_color_HSV_procesada
imagen_color_procesada = cv2.cvtColor(imagen_color_HSV, cv2.COLOR_HSV2BGR)

cv2.imshow("Imagen a color procesada HSV", imagen_color_procesada)
cv2.waitKey(0)


