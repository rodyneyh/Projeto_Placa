import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==========================================
# CONFIG
# ==========================================
PATH_MODELO_PLACA = r"C:/Users/rafit/Desktop/deteccao-placas-veiculares-main/models/best.pt"
PATH_MODELO_CHARS = r"C:/Users/rafit/Desktop/projeto_i/projetin/treino_chars_top_v22/weights/best.pt"

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


class IdentificadorUniversal:

    def __init__(self):
        self.modelo_placa = YOLO(PATH_MODELO_PLACA)
        self.modelo_chars = YOLO(PATH_MODELO_CHARS)

    # ==========================================
    # DETECTAR PLACA
    # ==========================================
    def detectar_placa(self, img):

        resultado = self.modelo_placa(
            img,
            conf=0.25,
            verbose=False
        )[0]

        placas = []

        for box in resultado.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            placas.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

        return placas

    # ==========================================
    # LER CHARS DIRETO
    # ==========================================
    def ler_chars_yolo(self, placa):

        placa = cv2.resize(
            placa,
            (320, 120),
            interpolation=cv2.INTER_CUBIC
        )

        resultado = self.modelo_chars(
            placa,
            conf=0.15,
            iou=0.25,
            imgsz=320,
            verbose=False
        )[0]

        chars = []

        for box in resultado.boxes:

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            xc = (x1 + x2) / 2

            chars.append({
                "x": xc,
                "char": CLASSES[cls],
                "conf": conf
            })

        chars = sorted(chars, key=lambda x: x["x"])

        # remove duplicados muito próximos
        filtrado = []

        for c in chars:

            if not filtrado:
                filtrado.append(c)
                continue

            ultimo = filtrado[-1]

            if abs(c["x"] - ultimo["x"]) < 12:
                if c["conf"] > ultimo["conf"]:
                    filtrado[-1] = c
            else:
                filtrado.append(c)

        texto = "".join([c["char"] for c in filtrado])

        return texto

    # ==========================================
    # FALLBACK OPENCV
    # ==========================================
    def fallback(self, placa):

        placa = cv2.resize(
            placa,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        _, thresh = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        chars = []

        for cnt in contours:

            x, y, w, h = cv2.boundingRect(cnt)

            if h < 40:
                continue

            if w < 8:
                continue

            roi = placa[y:y+h, x:x+w]

            roi = cv2.copyMakeBorder(
                roi,
                8, 8, 8, 8,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255)
            )

            roi = cv2.resize(
                roi,
                (64, 64),
                interpolation=cv2.INTER_CUBIC
            )

            r = self.modelo_chars(
                roi,
                conf=0.10,
                imgsz=64,
                verbose=False
            )[0]

            if len(r.boxes) > 0:

                best = max(
                    r.boxes,
                    key=lambda b: float(b.conf[0])
                )

                cls = int(best.cls[0])

                chars.append({
                    "x": x,
                    "char": CLASSES[cls]
                })

        chars = sorted(chars, key=lambda x: x["x"])

        texto = "".join([c["char"] for c in chars])

        return texto

    # ==========================================
    # CORREÇÃO INTELIGENTE
    # ==========================================
    def corrigir(self, texto):

        texto = re.sub(r'[^A-Z0-9]', '', texto.upper())

        # remove extra char à esquerda se tiver 8
        if len(texto) == 8:
            texto = texto[1:]

        if len(texto) > 7:
            texto = texto[:7]

        texto = list(texto.ljust(7))

        # posições letras
        letras = [0, 1, 2, 4]

        # posições números
        nums = [3, 5, 6]

        conv_letra = {
            "0": "O",
            "1": "I",
            "2": "Z",
            "5": "S",
            "8": "B"
        }

        conv_num = {
            "O": "0",
            "Q": "0",
            "D": "0",
            "I": "1",
            "L": "1",
            "Z": "2",
            "S": "5",
            "B": "8"
        }

        for i in letras:
            if texto[i] in conv_letra:
                texto[i] = conv_letra[texto[i]]

        for i in nums:
            if texto[i] in conv_num:
                texto[i] = conv_num[texto[i]]

        return "".join(texto).strip()

    # ==========================================
    # LER PLACA
    # ==========================================
    def ler_placa(self, placa):

        direto = self.ler_chars_yolo(placa)
        print("YOLO direto:", direto)

        fb = self.fallback(placa)
        print("Fallback:", fb)

        texto = direto

        if len(fb) > len(direto):
            texto = fb

        final = self.corrigir(texto)

        return final

    # ==========================================
    # MAIN
    # ==========================================
    def identificar_e_mostrar(self, caminho):

        img = cv2.imread(caminho)

        if img is None:
            print("Imagem não encontrada")
            return

        placas = self.detectar_placa(img)

        img_vis = img.copy()

        for i, p in enumerate(placas):

            x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]

            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.08)

            placa = img[
                max(0, y1-pad_y):min(img.shape[0], y2+pad_y),
                max(0, x1-pad_x):min(img.shape[1], x2+pad_x)
            ]

            print(f"\nPlaca {i+1}")

            texto = self.ler_placa(placa)

            print("FINAL:", texto)

            cv2.rectangle(
                img_vis,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                img_vis,
                texto,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

        plt.figure(figsize=(14, 8))
        plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()


# ==========================================
# EXECUTAR
# ==========================================
if __name__ == "__main__":

    app = IdentificadorUniversal()

    app.identificar_e_mostrar(
        r"C:/Users/rafit/Pictures/placa/images.jpg"
    )