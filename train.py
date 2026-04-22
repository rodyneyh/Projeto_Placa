from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolo11s.pt")

    model.train(
        data=r"C:/Users/rafit/Desktop/projeto_i/projetin/data.yaml",
        project=r"C:/Users/rafit/Desktop/projeto_i/projetin",
        name="treino_chars_top_v2",

        # ── TREINO ────────────────────────────────────────────────
        epochs=150,         # mais épocas + patience maior = mais tempo p/ convergir
        imgsz=320,          # 320 > 256 para caracteres finos (B, 8, 0, D...)
        batch=32,           # RTX 3050 4 GB aguenta 32 com imgsz=320 + amp=True
        patience=40,

        # ── AUGMENTATION OCR ──────────────────────────────────────
        # desligados (correto para OCR de placa)
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.0,
        flipud=0.0,
        shear=0.0,

        # cor: bem conservador — placa tem cores definidas
        hsv_h=0.003,        # variação mínima de matiz
        hsv_s=0.15,
        hsv_v=0.20,         # brilho varia mais (sol, sombra, câmera)

        # geometria: simula câmera não-frontal
        degrees=2.0,        # rotação leve
        translate=0.03,
        scale=0.10,
        perspective=0.0002, # distorção de perspectiva bem suave

        # erosão: desligada — remove detalhes de borda dos chars
        erasing=0.0,
        close_mosaic=0,

        # ── OTIMIZAÇÃO ────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.0008,         # LR inicial menor → convergência mais estável
        lrf=0.005,          # LR final = lr0 * lrf = 4e-6 (decai mais)
        warmup_epochs=5,    # aquecimento mais longo com AdamW
        warmup_momentum=0.8,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,        # cosine scheduler suaviza a descida do LR

        # ── LOSS WEIGHTS ──────────────────────────────────────────
        box=5.0,            # reduz peso do box — o que importa é cls correta
        cls=1.5,            # aumenta peso da classificação (A vs B vs 8...)
        dfl=1.5,

        # ── HARDWARE ──────────────────────────────────────────────
        device=0,
        workers=4,
        amp=True,           # mixed precision — economiza VRAM (crucial p/ 4 GB)
        cache="ram",        # dataset cabe na RAM, acelera muito o treino

        # ── SALVAR / LOG ──────────────────────────────────────────
        save=True,
        save_period=15,
        plots=True,
        verbose=True,
    )