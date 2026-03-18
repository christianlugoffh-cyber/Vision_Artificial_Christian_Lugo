import cv2
from pathlib import Path

# ========= CONFIG =========
DIR_IMAGENES = Path("Images")
DIR_LABELS   = Path("Annotations")
# ==========================


# ---------------------------------------
# CONVERTIR FORMATO YOLO A PIXELES
# ---------------------------------------
def yolo_a_pixeles(cx, cy, bw, bh, ancho, alto):

    cx *= ancho
    cy *= alto
    bw *= ancho
    bh *= alto

    x_ini = int(cx - bw / 2)
    y_ini = int(cy - bh / 2)
    x_fin = int(cx + bw / 2)
    y_fin = int(cy + bh / 2)

    return x_ini, y_ini, x_fin, y_fin


# ---------------------------------------
# CARGAR ANOTACIONES
# ---------------------------------------
def cargar_etiquetas(ruta_label, ancho, alto):

    resultados = []

    with open(ruta_label, "r") as archivo:

        for fila in archivo:

            datos = fila.strip().split()

            if len(datos) != 5:
                continue

            clase, cx, cy, bw, bh = datos

            x1, y1, x2, y2 = yolo_a_pixeles(
                float(cx), float(cy),
                float(bw), float(bh),
                ancho, alto
            )

            numero = int(clase) + 1

            resultados.append({
                "bbox": (x1, y1, x2, y2),
                "valor": numero
            })

    return resultados


# ---------------------------------------
# DIBUJAR RESULTADOS
# ---------------------------------------
def dibujar_dados(imagen, lista_datos):

    contador = 0

    for item in lista_datos:

        x1, y1, x2, y2 = item["bbox"]
        valor = item["valor"]

        contador += 1

        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 200, 0), 2)

        etiqueta = f"{valor}"

        cv2.putText(imagen, etiqueta,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0, (0, 200, 0), 2)

    return contador


# ---------------------------------------
# PROCESAR UNA IMAGEN
# ---------------------------------------
def analizar_imagen(ruta_img):

    nombre_base = ruta_img.stem
    ruta_label = DIR_LABELS / f"{nombre_base}.txt"

    if not ruta_label.exists():
        print(f"[!] Sin etiqueta: {ruta_img.name}")
        return

    imagen = cv2.imread(str(ruta_img))

    if imagen is None:
        print(f"[!] Error cargando: {ruta_img.name}")
        return

    alto, ancho = imagen.shape[:2]

    info = cargar_etiquetas(ruta_label, ancho, alto)

    total = dibujar_dados(imagen, info)

    print(f"{ruta_img.name} -> Total detectado: {total}")

    # ajuste de tamaño para mostrar
    factor = 800 / ancho
    vista = cv2.resize(imagen, (800, int(alto * factor)))

    cv2.imshow("Visualizacion", vista)

    tecla = cv2.waitKey(0)
    return tecla


# ---------------------------------------
# MAIN
# ---------------------------------------
def ejecutar():

    extensiones_validas = (".jpg", ".png", ".jpeg")

    for ruta in DIR_IMAGENES.iterdir():

        if ruta.suffix.lower() not in extensiones_validas:
            continue

        tecla = analizar_imagen(ruta)

        if tecla == ord('q'):
            break

    cv2.destroyAllWindows()


# ---------------------------------------
# INICIO
# ---------------------------------------
if __name__ == "__main__":
    ejecutar()