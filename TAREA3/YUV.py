import cv2
import numpy as np

# -------- FUNCION DE PROCESAMIENTO EN YUV --------
def procesamiento_yuv(imagen, brillo=0, contraste=1.0, gamma=1.0):

    # Convertir imagen de BGR a YUV
    imagen_yuv = cv2.cvtColor(imagen, cv2.COLOR_BGR2YUV)

    # Obtener canal Y (luminancia)
    Y = imagen_yuv[:, :, 0].astype(np.float32)

    # ----- Ajuste de brillo y contraste -----
    Y = contraste * Y + brillo
    Y = np.clip(Y, 0, 255)

    # ----- Corrección Gamma -----
    Y = Y / 255.0
    Y = np.power(Y, gamma)
    Y = np.uint8(Y * 255)

    # Reemplazar canal Y modificado
    imagen_yuv[:, :, 0] = Y

    # Convertir nuevamente a BGR
    imagen_resultado = cv2.cvtColor(imagen_yuv, cv2.COLOR_YUV2BGR)

    return imagen_resultado


# -------- CARGAR IMAGEN --------
imagen = cv2.imread("lena.jpeg")

if imagen is None:
    print("Error: no se pudo cargar la imagen")
    exit()

# Parámetros de procesamiento
brillo = 30
contraste = 1.2
gamma = 0.8

# Procesar imagen
imagen_procesada = procesamiento_yuv(imagen, brillo, contraste, gamma)

# Mostrar comparación
comparacion = np.hstack((imagen, imagen_procesada))

cv2.imshow("Original vs Procesada (Canal Y - YUV)", comparacion)

cv2.waitKey(0)
cv2.destroyAllWindows()
