import cv2
import numpy as np

# Abrir video
cap = cv2.VideoCapture("Traffic.mp4")

# Leer primer frame
ret, old_frame = cap.read()

if not ret:
    print("Error al abrir el video")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Crear máscara ROI (zona de vehículos)
mask = np.zeros_like(old_gray)

alto, ancho = old_gray.shape

mask[int(alto*0.35):int(alto*0.75),
     int(ancho*0.25):int(ancho*0.75)] = 255

cv2.imshow("Mascara aplicada", mask)

# Detectar esquinas iniciales
p0 = cv2.goodFeaturesToTrack(
    old_gray,
    mask=mask,
    maxCorners=300,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7
)

mask_dibujo = np.zeros_like(old_frame)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray,
        frame_gray,
        p0,
        None
    )

    if p1 is not None:

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Contar esquinas en movimiento
        num_esquinas = len(good_new)

        # Dibujar trayectorias
        for i, (new, old) in enumerate(zip(good_new, good_old)):

            a, b = new.ravel()
            c, d = old.ravel()

            mask_dibujo = cv2.line(
                mask_dibujo,
                (int(a), int(b)),
                (int(c), int(d)),
                (0, 255, 0),
                2
            )

            frame = cv2.circle(
                frame,
                (int(a), int(b)),
                5,
                (0, 0, 255),
                -1
            )

        img = cv2.add(frame, mask_dibujo)

        # Mostrar contador en pantalla
        cv2.putText(
            img,
            f"Esquinas en movimiento: {num_esquinas}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow("Tracking Vehiculos", img)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()