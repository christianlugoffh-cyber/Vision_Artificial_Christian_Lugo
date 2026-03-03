import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------- CARGAR IMAGEN --------
ruta_imagen = "IM4.jpeg"
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print("No se pudo cargar la imagen")
    exit()

# -------- CONVERTIR A YUV --------
imagen_yuv = cv2.cvtColor(imagen, cv2.COLOR_BGR2YUV)

# Obtener canal Y
canal_y_original = imagen_yuv[:, :, 0]

# -------- ECUALIZAR --------
canal_y_ecualizado = cv2.equalizeHist(canal_y_original)

# Reemplazar canal Y
imagen_yuv[:, :, 0] = canal_y_ecualizado

# Convertir nuevamente a BGR
imagen_ecualizada = cv2.cvtColor(imagen_yuv, cv2.COLOR_YUV2BGR)

# -------- MOSTRAR IMÁGENES --------
cv2.imshow("Imagen Original", imagen)
cv2.imshow("Imagen Ecualizada", imagen_ecualizada)

# -------- HISTOGRAMAS --------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Histograma Original (Canal Y)")
plt.hist(canal_y_original.flatten(), bins=256, range=(0,255))

plt.subplot(1,2,2)
plt.title("Histograma Ecualizado (Canal Y)")
plt.hist(canal_y_ecualizado.flatten(), bins=256, range=(0,255))

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
