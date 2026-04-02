import cv2
import numpy as np
import os

# Rutas de las imágenes
ruta_original = "lena.jpeg"
ruta_rotada = "lena_rotada.jpeg"

# Verificar que existan
if not os.path.exists(ruta_original):
    raise FileNotFoundError(f"No se encontró: {ruta_original}")

if not os.path.exists(ruta_rotada):
    raise FileNotFoundError(f"No se encontró: {ruta_rotada}")

# Leer imágenes en color y en escala de grises
img1 = cv2.imread(ruta_original)
img2 = cv2.imread(ruta_rotada)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detector de esquinas y descriptores
orb = cv2.ORB_create(nfeatures=5000)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

if des1 is None or des2 is None:
    raise RuntimeError("No se pudieron calcular descriptores en una o ambas imágenes.")

# Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

if len(matches) < 3:
    raise RuntimeError("No hay suficientes matches para calcular la transformación afín.")

# Ordenar matches por calidad (distancia menor = mejor)
matches = sorted(matches, key=lambda x: x.distance)

# Tomar los 3 mejores matches
mejores = matches[:3]

# Obtener los puntos correspondientes
pts1 = np.float32([kp1[m.queryIdx].pt for m in mejores])
pts2 = np.float32([kp2[m.trainIdx].pt for m in mejores])

# Calcular transformación afín
M = cv2.getAffineTransform(pts1, pts2)

# Aplicar transformación afín
alto, ancho = img2.shape[:2]
img_afine = cv2.warpAffine(img1, M, (ancho, alto))

# Dibujar matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, mejores, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Mostrar resultados
cv2.imshow("Mejores 3 matches", img_matches)
cv2.imshow("Imagen original transformada", img_afine)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar resultados
cv2.imwrite("matches_3_mejores.jpg", img_matches)
cv2.imwrite("lena_transformada_afine.jpg", img_afine)

print("Matriz afín calculada:")
print(M)
print("Resultados guardados como:")
print("- matches_3_mejores.jpg")
print("- lena_transformada_afine.jpg")
