import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Cargar imagen
# -----------------------------
imagen = cv2.imread("triangulo.jpg", 0)   # <- aqui cambias el nombre de la imagen

if imagen is None:
    print("No se pudo cargar la imagen")
    exit()

img = np.float32(imagen)

# -----------------------------
# DFT
# -----------------------------
dft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitud = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
magnitud = np.log(magnitud + 1)

# -----------------------------
# DCT
# -----------------------------
dct = cv2.dct(img)
dct_mag = np.log(np.abs(dct) + 1)

# -----------------------------
# Mostrar resultados
# -----------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(imagen, cmap='gray')
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(magnitud, cmap='gray')
plt.title("DFT")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(dct_mag, cmap='gray')
plt.title("DCT")
plt.axis("off")

plt.show()


