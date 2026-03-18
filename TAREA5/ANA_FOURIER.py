import cv2
import numpy as np
import os
import glob

# ---------------------------------------
# MOSTRAR IMAGEN REDIMENSIONADA
# ---------------------------------------
def mostrar_redimensionado(nombre, imagen, ancho=600):
    h, w = imagen.shape[:2]
    escala = ancho / w
    nueva_dim = (ancho, int(h * escala))
    img_resized = cv2.resize(imagen, nueva_dim)
    cv2.imshow(nombre, img_resized)

# ---------------------------------------
# CONFIGURACION
# ---------------------------------------
CARPETA_ENTRADA = "IMAGENES"

AREA_MIN_DADO = 15   # más flexible (dados pequeños)
AREA_MAX_DADO = 300000

AREA_MIN_PUNTO = 5
AREA_MAX_PUNTO = 30

# ---------------------------------------
# PREPROCESAMIENTO
# ---------------------------------------
def preprocesar_imagen(gray):

    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binaria = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=2)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)

    return binaria

# ---------------------------------------
# CONTAR PUNTOS
# ---------------------------------------
def contar_puntos_dado(roi_gris):

    roi = cv2.GaussianBlur(roi_gris, (5, 5), 0)

    # puntos negros
    _, bin_negro = cv2.threshold(roi, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # puntos blancos
    _, bin_blanco = cv2.threshold(roi, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    bin_negro = cv2.morphologyEx(bin_negro, cv2.MORPH_OPEN, kernel)
    bin_blanco = cv2.morphologyEx(bin_blanco, cv2.MORPH_OPEN, kernel)

    def contar(binaria):
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
        conteo = 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if AREA_MIN_PUNTO <= area <= AREA_MAX_PUNTO:
                conteo += 1

        return conteo

    puntos_negros = contar(bin_negro)
    puntos_blancos = contar(bin_blanco)

    return max(puntos_negros, puntos_blancos)

# ---------------------------------------
# VALIDAR SI ES DADO
# ---------------------------------------
def es_dado(w, h, area, puntos):

    # 1. Área válida
    if not (AREA_MIN_DADO <= area <= AREA_MAX_DADO):
        return False

    # 2. Forma cuadrada
    relacion = w / h
    if relacion < 0.6 or relacion > 1.4:
        return False

    # 3. Debe tener al menos 1 punto (clave para eliminar círculos)
    if puntos == 0:
        return False

    return True

# ---------------------------------------
# PROCESAR IMAGEN
# ---------------------------------------
def procesar_imagen(ruta):

    img = cv2.imread(ruta)
    if img is None:
        return

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binaria = preprocesar_imagen(gray)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)

    total_dados = 0
    resultados = []

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # recorte
        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)

        roi = gray[y1:y2, x1:x2]

        puntos = contar_puntos_dado(roi)

        # 🔥 FILTRO INTELIGENTE
        if not es_dado(w, h, area, puntos):
            continue

        total_dados += 1

        # Validar valor
        if 1 <= puntos <= 6:
            texto = str(puntos)
            color = (0, 255, 0)
            resultados.append(puntos)
        else:
            texto = "?"
            color = (0, 165, 255)
            resultados.append("X")

        cv2.rectangle(original, (x1, y1), (x2, y2), color, 2)
        cv2.putText(original, texto, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(original, f"Total: {total_dados}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    print("\nImagen:", os.path.basename(ruta))
    print("Total dados:", total_dados)
    print("Valores:", resultados)

    mostrar_redimensionado("Original", img)
    mostrar_redimensionado("Binaria", binaria)
    mostrar_redimensionado("Resultado", original)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------------------------------
# PROCESAR CARPETA
# ---------------------------------------
def procesar_carpeta(carpeta):

    rutas = glob.glob(os.path.join(carpeta, "*.jpg")) + \
            glob.glob(os.path.join(carpeta, "*.png"))

    for r in rutas:
        procesar_imagen(r)

# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    procesar_carpeta(CARPETA_ENTRADA)