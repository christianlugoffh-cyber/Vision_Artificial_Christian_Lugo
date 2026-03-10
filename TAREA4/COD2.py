import cv2
import numpy as np

# Cargar imagen en escala de grises
imagen = cv2.imread("tornillos_2.jpg", cv2.IMREAD_GRAYSCALE)

if imagen is None:
    print("Error: no se pudo cargar la imagen")
    exit()

original = imagen.copy()

# Convertir a imagen binaria (tu código)
ret, imagen_binaria = cv2.threshold(imagen, 120, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("Imagen Binaria", imagen_binaria)
cv2.waitKey(0)

# Operaciones morfológicas para limpiar ruido
kernel = np.ones((5,5),np.uint8)

procesada = cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel, iterations=2)
procesada = cv2.morphologyEx(procesada, cv2.MORPH_OPEN, kernel, iterations=1)

# Buscar contornos
contornos,_ = cv2.findContours(procesada,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

# Convertir imagen a color para dibujar etiquetas
imagen_color = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)

area_min = 150
umbral_largo = 120

buenos = 0
malos = 0

for c in contornos:

    area = cv2.contourArea(c)

    if area > area_min:

        x,y,w,h = cv2.boundingRect(c)

        largo = max(w,h)

        if largo > umbral_largo:

            buenos += 1
            color = (0,255,0)
            texto = "Bueno"

        else:

            malos += 1
            color = (0,0,255)
            texto = "Mal estado"

        cv2.rectangle(imagen_color,(x,y),(x+w,y+h),color,2)

        cv2.putText(imagen_color,texto,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)

# Resultados
print("Imagen: tornillos_2.jpg")
print("Area minima:",area_min)
print("Umbral largo:",umbral_largo)
print("Buenos:",buenos)
print("Malos:",malos)
print("Total:",buenos+malos)

# Mostrar imágenes
cv2.imshow("Original", original)
cv2.imshow("Procesada", procesada)
cv2.imshow("Etiquetada", imagen_color)

cv2.waitKey(0)
cv2.destroyAllWindows()