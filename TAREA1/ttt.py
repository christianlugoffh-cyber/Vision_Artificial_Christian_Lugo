import cv2
import matplotlib.pyplot as plt
import numpy as np

ruta_imagen = "IM1.jpeg"

# Leer imagen (OpenCV la lee en BGR)
imagen_BGR = cv2.imread(ruta_imagen)

if imagen_BGR is None:
    print("Error: No se encontró la imagen")
    exit()

# Convertir formatos
imagen_RGB = cv2.cvtColor(imagen_BGR, cv2.COLOR_BGR2RGB)
imagen_GRAYSCALE = cv2.cvtColor(imagen_BGR, cv2.COLOR_BGR2GRAY)
imagen_HSV = cv2.cvtColor(imagen_BGR, cv2.COLOR_BGR2HSV)

# Mostrar con OpenCV (BGR)
cv2.imshow("Imagen en BGR (OpenCV)", imagen_BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f'Estructura de la imagen RGB: {imagen_RGB.shape}')
print(f'Estructura de la imagen HSV: {imagen_HSV.shape}')
print(f'Estructura de la imagen escala de grises: {imagen_GRAYSCALE.shape}')

# =============================
# HISTOGRAMA
# =============================
histograma = cv2.calcHist([imagen_GRAYSCALE], [0], None, [256], [0,256])

plt.figure()
plt.hist(imagen_GRAYSCALE.flatten(), bins=256, range=(0,255), color="gray")
plt.xlabel("Valores de pixel")
plt.ylabel("Frecuencia")
plt.title("Histograma en escala de grises")
plt.show()

# =============================
# FUNCIÓN ENTROPÍA
# =============================
def calcular_entropia(imagen):

    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([imagen], [0], None, [256], [0,256])
    hist_norm = hist / hist.sum()

    hist_norm = hist_norm[hist_norm > 0]

    entropia = -np.sum(hist_norm * np.log2(hist_norm))

    return entropia


# =============================
# ESTADÍSTICAS
# =============================
imagen_flat = imagen_GRAYSCALE.flatten()

media = np.mean(imagen_flat)
mediana = np.median(imagen_flat)
desviacion_std = np.std(imagen_flat)
contraste = imagen_flat.max() - imagen_flat.min()
entropia = calcular_entropia(imagen_GRAYSCALE)

print(f"Media: {media}")
print(f"Mediana: {mediana}")
print(f"Desviación Estándar: {desviacion_std}")
print(f"Contraste: {contraste}")
print(f"Entropía: {entropia}")

# =============================
# MOSTRAR CON MATPLOTLIB
# =============================

# Imagen RGB correcta
plt.figure()
plt.imshow(imagen_RGB)
plt.title("Imagen en RGB (Matplotlib)")
plt.axis("off")
plt.show()

# Imagen en escala de grises
plt.figure()
plt.imshow(imagen_GRAYSCALE, cmap="gray")
plt.title("Imagen en escala de grises")
plt.axis("off")
plt.show()

