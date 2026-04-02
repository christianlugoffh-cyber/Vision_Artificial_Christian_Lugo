import cv2
import os


def encontrar_esquinas(img):
    detector = cv2.FastFeatureDetector_create(
        threshold=20,
        nonmaxSuppression=True,
        type=cv2.FastFeatureDetector_TYPE_7_12
    )
    return detector.detect(img, mask=None)


def dibujar_puntos(img, pts):
    return cv2.drawKeypoints(img, pts, None, color=(0,0,255))


def procesar_imagenes():

    carpeta_entrada = "TELAS"
    carpeta_salida = "RESULTADOS"

    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    archivos = os.listdir(carpeta_entrada)

    print("\nTABLA: Tela vs Densidad Absoluta\n")
    print("Tela\t\tEsquinas\tDensidad")

    resultados = []

    for archivo in archivos:

        ruta = os.path.join(carpeta_entrada, archivo)

        img = cv2.imread(ruta)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        pts = encontrar_esquinas(gray)

        img_pts = dibujar_puntos(img, pts)

        alto, ancho = gray.shape

        densidad = len(pts)/(alto*ancho)

        resultados.append((archivo, len(pts), densidad))

        print(f"{archivo}\t{len(pts)}\t{densidad:.6f}")

        salida = os.path.join(carpeta_salida, archivo)

        cv2.imwrite(salida, img_pts)


procesar_imagenes()