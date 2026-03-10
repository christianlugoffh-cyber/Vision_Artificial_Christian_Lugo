import cv2
import numpy as np

# Cargar imagen
imagen = cv2.imread("tornillos.jpg")
original = imagen.copy()

# Escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Suavizado
blur = cv2.GaussianBlur(gris,(5,5),0)

# Umbral automático
_, binaria = cv2.threshold(blur,0,255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Eliminar ruido pequeño
kernel = np.ones((3,3),np.uint8)
procesada = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)

# Buscar contornos
contornos,_ = cv2.findContours(procesada,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

area_min = 300
conteo = 0

for c in contornos:

    area = cv2.contourArea(c)

    if area > area_min:

        conteo += 1

        x,y,w,h = cv2.boundingRect(c)

        cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,255,0),2)

# Resultados
print("Imagen: tornillos.jpg")
print("Area minima:",area_min)
print("Numero de tornillos:",conteo)

# Mostrar imágenes
cv2.imshow("Original",original)
cv2.imshow("Procesada",procesada)
cv2.imshow("Etiquetada",imagen)

cv2.waitKey(0)
cv2.destroyAllWindows()