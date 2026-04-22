import os
import cv2
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# CONFIG
# ==========================================
OUTPUT = r"C:/Users/rafit/Desktop/projeto_i/projetin/dataset_mercosul"

FONT_PATH = r"C:/Users/rafit/Desktop/projeto_i/FE-FONT.TTF"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "FE-FONT.TTF"

TOTAL      = 5000   # imagens de treino
TOTAL_VAL  = 1000   # imagens de validação (20% do treino)
W, H = 520, 120

# Classes em ordem: A-Z (0-25) e 0-9 (26-35)
classes = list(string.ascii_uppercase + string.digits)

# ==========================================
# PASTAS — estrutura esperada pelo YOLO:
#   dataset_mercosul/
#     train/images/
#     train/labels/
#     val/images/
#     val/labels/
# ==========================================
for split in ("train", "val"):
    os.makedirs(f"{OUTPUT}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT}/{split}/labels", exist_ok=True)

# ==========================================
# AUXILIARES
# ==========================================
def gerar_placa():
    letras = ''.join(random.choices(string.ascii_uppercase, k=3))
    num1   = random.choice(string.digits)
    letra4 = random.choice(string.ascii_uppercase)
    nums   = ''.join(random.choices(string.digits, k=2))
    return letras + num1 + letra4 + nums

def class_id(c):
    return classes.index(c)

# ==========================================
# DESENHAR PLACA
# ==========================================
def criar_imagem(texto):
    img  = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, W, 28), fill=(20, 70, 180))

    try:
        font = ImageFont.truetype(FONT_PATH, 72)
    except OSError:
        print(f"ERRO: Fonte não encontrada em {FONT_PATH}. Verifique o arquivo!")
        font = ImageFont.load_default()

    x_offset = 38
    y_offset = 30
    labels   = []

    for c in texto:
        bbox = draw.textbbox((x_offset, y_offset), c, font=font)
        draw.text((x_offset, y_offset), c, font=font, fill=(0, 0, 0))
        labels.append((c, bbox[0], bbox[1], bbox[2], bbox[3]))
        x_offset += 62

    return np.array(img), labels

# ==========================================
# AUGMENTATION
# ==========================================
def augment(img):
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    if random.random() < 0.3:
        noise = np.random.randint(0, 15, img.shape, dtype='uint8')
        img   = cv2.addWeighted(img, 1.0, noise, 0.5, 0)

    if random.random() < 0.3:
        value    = random.randint(-30, 30)
        hsv      = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v  = cv2.split(hsv)
        v        = np.clip(cv2.add(v, value), 0, 255)
        img      = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)

    return img

# ==========================================
# SALVAR YOLO LABEL
# ==========================================
def salvar_label(path, labels):
    linhas = []
    for c, x1, y1, x2, y2 in labels:
        cls = class_id(c)
        xc  = ((x1 + x2) / 2) / W
        yc  = ((y1 + y2) / 2) / H
        bw  = (x2 - x1) / W
        bh  = (y2 - y1) / H
        linhas.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    with open(path, "w") as f:
        f.write("\n".join(linhas))

# ==========================================
# FUNÇÃO GENÉRICA DE GERAÇÃO
# ==========================================
def gerar_split(split, total):
    print(f"\nGerando {total} imagens para '{split}'...")
    for i in range(total):
        try:
            texto   = gerar_placa()
            img, labels = criar_imagem(texto)
            img_aug = augment(img)
            nome    = f"placa_{i:05d}"

            cv2.imwrite(
                f"{OUTPUT}/{split}/images/{nome}.jpg",
                cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR)
            )
            salvar_label(f"{OUTPUT}/{split}/labels/{nome}.txt", labels)

            if i % 500 == 0:
                print(f"  {split}: {i}/{total}")

        except Exception as e:
            print(f"  Erro na iteração {i} ({split}): {e}")

    print(f"  '{split}' concluído!")

# ==========================================
# LOOP PRINCIPAL
# ==========================================
gerar_split("train", TOTAL)
gerar_split("val",   TOTAL_VAL)

print(f"\nCONCLUÍDO! Dataset salvo em: {OUTPUT}")
print(f"  Treino : {TOTAL}   imagens  →  {OUTPUT}/train/images/")
print(f"  Val    : {TOTAL_VAL} imagens  →  {OUTPUT}/val/images/")