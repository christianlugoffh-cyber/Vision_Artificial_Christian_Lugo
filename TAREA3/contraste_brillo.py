import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class Ventana(tk.Tk):
    def __init__(self):
        super().__init__(className="Control de brillo y contraste")

        self.title("Control de brillo y contraste")
        self.geometry("1500x850")
        self.protocol("WM_DELETE_WINDOW", self.cierre)

        self.ruta_imagen = "IM4.jpeg"

        # -------- CARGAR IMAGEN --------
        self.imagen_original = cv2.imread(self.ruta_imagen)

        if self.imagen_original is None:
            print("No se pudo cargar la imagen")
            exit()

        # Redimensionar solo para visualización
        self.display_w = 640
        self.display_h = 360

        self.imagen = cv2.resize(
            self.imagen_original,
            (self.display_w, self.display_h)
        )

        self.h, self.w = self.imagen.shape[:2]

        img_rgb = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB)
        self.imagen_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))

        self.imagen_procesada = self.imagen.copy()

        # Histograma más grande
        self.fig, self.ax = plt.subplots(figsize=(6,4))

        self.__check_status = tk.BooleanVar()

        self.mostrar_ventana()

    def cierre(self):
        plt.close("all")
        self.quit()
        self.destroy()

    # ---------------- INTERFAZ ----------------
    def mostrar_ventana(self):

        self.__canvas_imagen = tk.Canvas(
            self,
            width=self.display_w,
            height=self.display_h,
            background="black"
        )

        self.__canvas_imagen_proc = tk.Canvas(
            self,
            width=self.display_w,
            height=self.display_h,
            background="black"
        )

        self.__canvas_imagen.create_image(
            self.w/2, self.h/2,
            image=self.imagen_tk
        )
        self.__canvas_imagen.place(x=30, y=40)

        self.mostrar_imagen_procesada(0, 1, 1)

        # -------- SLIDERS --------
        self.__escala_brillo = tk.Scale(
            self, label="Brillo",
            from_=-100, to=100,
            orient=tk.HORIZONTAL,
            command=self.actualizar_imagen
        )

        self.__escala_contraste = tk.Scale(
            self, label="Contraste",
            from_=0.1, to=3,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            command=self.actualizar_imagen
        )
        self.__escala_contraste.set(1)

        self.__escala_gamma = tk.Scale(
            self, label="Gamma",
            from_=0.1, to=3,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            command=self.actualizar_imagen
        )
        self.__escala_gamma.set(1)

        self.__escala_brillo.place(x=100, y=430)
        self.__escala_contraste.place(x=300, y=430)
        self.__escala_gamma.place(x=100, y=500)

        # -------- HISTOGRAMA --------
        self.__canvas_hist = FigureCanvasTkAgg(self.fig, master=self)
        self.__canvas_hist.get_tk_widget().place(
            x=30, y=600, width=1310, height=220
        )

        self.mostrar_histograma(
            self.obtener_y(self.imagen_procesada)
        )

        self.__canvas_check = tk.Checkbutton(
            self,
            text="Ecualizar Histograma",
            variable=self.__check_status,
            command=self.actualizar_imagen
        )

        self.__canvas_check.place(x=100, y=550)

    # ---------------- MOSTRAR IMAGEN ----------------
    def mostrar_imagen_procesada(self, brillo, contraste, gamma):

        self.procesar_imagen(contraste, brillo, gamma)

        im = cv2.cvtColor(
            self.imagen_procesada,
            cv2.COLOR_BGR2RGB
        )

        img = Image.fromarray(im)
        self.imagen_procesada_tk = ImageTk.PhotoImage(img)

        self.__canvas_imagen_proc.create_image(
            self.w/2,
            self.h/2,
            image=self.imagen_procesada_tk
        )

        self.__canvas_imagen_proc.place(x=700, y=40)

    def actualizar_imagen(self, val=None):

        brillo = self.__escala_brillo.get()
        contraste = self.__escala_contraste.get()
        gamma = self.__escala_gamma.get()

        self.mostrar_imagen_procesada(
            brillo,
            contraste,
            gamma
        )

        self.mostrar_histograma(
            self.obtener_y(self.imagen_procesada)
        )

    # -------- CANAL Y --------
    def obtener_y(self, imagen):
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2YUV)[:,:,0]

    # -------- HISTOGRAMA --------
    def mostrar_histograma(self, datos):

        self.ax.clear()
        self.ax.set_title('Histograma Canal Y (YUV)')
        self.ax.set_xlabel('Nivel de intensidad')
        self.ax.set_ylabel('Frecuencia')

        self.ax.hist(
            datos.flatten(),
            bins=256,
            range=(0,255)
        )

        self.__canvas_hist.draw_idle()

    # -------- PROCESAMIENTO EN YUV --------
    def procesar_imagen(self, contraste, brillo, gamma):

        imagen_yuv = cv2.cvtColor(
            self.imagen,
            cv2.COLOR_BGR2YUV
        )

        canal_y = imagen_yuv[:,:,0]

        canal_y = Procesador.contraste_brillo_centrado(
            canal_y,
            contraste,
            brillo
        )

        canal_y = Procesador.correccion_gamma(
            canal_y,
            gamma
        )

        if self.__check_status.get():
            canal_y = Procesador.ecualizar_hist(canal_y)

        imagen_yuv[:,:,0] = canal_y

        self.imagen_procesada = cv2.cvtColor(
            imagen_yuv,
            cv2.COLOR_YUV2BGR
        )


# ---------------- PROCESADOR ----------------
class Procesador:

    @staticmethod
    def contraste_brillo_centrado(imagen, contraste, brillo):
        imagen = imagen.astype(np.float32)
        return np.clip(
            contraste * (imagen - 128) + 128 + brillo,
            0, 255
        ).astype(np.uint8)

    @staticmethod
    def correccion_gamma(imagen, gamma):
        return np.clip(
            ((imagen.astype(np.float32)/255)**gamma)*255,
            0, 255
        ).astype(np.uint8)

    @staticmethod
    def ecualizar_hist(imagen):
        return cv2.equalizeHist(imagen)


# -------- MAIN --------
if __name__ == "__main__":
    v = Ventana()
    v.mainloop()