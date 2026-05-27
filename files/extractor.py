#!/usr/bin/env python3
"""
extractor.py — Extracción de 68 características geométricas faciales
Uso    : python extractor.py --dataset ./dataset --output features.csv
Dataset: subcarpetas alegria/ dolor/ neutral/ con imágenes JPG/PNG
Modelo : MediaPipe FaceLandmarker (Tasks API, float16)
Salida : CSV con 68 columnas de features + columna 'label'

68 features = 4 EAR + 6 Boca + 4 COM-MEJ + 6 Cejas + 3 Nariz + 45 distancias
Autores: José Antonio Benavides Ortega — FIME UdC
"""
import argparse
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pathlib import Path
from itertools import combinations
from typing import Optional
import logging
import time

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

GREEN = "\033[92m"; YELLOW = "\033[93m"; CYAN = "\033[96m"
RESET = "\033[0m";  BOLD   = "\033[1m"

# ── Clases válidas ─────────────────────────────────────────────────────────────
CLASES  = ["alegria", "dolor", "neutral"]
EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Tamaño de upscale (imágenes pequeñas como CK+ 48×48) ─────────────────────
TARGET_SIZE  = (224, 224)
CLAHE_CLIP   = 2.0
CLAHE_GRID   = (8, 8)

# ── Índices MediaPipe FaceLandmarker (478 puntos) ─────────────────────────────
# Ojo derecho (perspectiva observador)
R_EYE  = [33, 160, 158, 133, 153, 144]
# Ojo izquierdo
L_EYE  = [263, 387, 385, 362, 373, 380]

MOUTH_L      = 61;  MOUTH_R  = 291
MOUTH_TOP    = 13;  MOUTH_BOT = 14
MOUTH_TOP_L  = 37;  MOUTH_TOP_R = 267
MOUTH_BOT_L  = 84;  MOUTH_BOT_R = 314

L_BROW_IN  = 55;  L_BROW_MID = 107
R_BROW_IN  = 285; R_BROW_MID = 336

NOSE_TIP   = 4
NOSE_L     = 98;  NOSE_R = 327
NOSE_BRIDGE = 6

FACE_L = 234; FACE_R = 454
FACE_TOP = 10; FACE_BOT = 152

# ── 10 puntos clave para 45 distancias par a par ───────────────────────────────
# (índices calculados como promedios o directos; ver función key_points())
KP_NAMES = ["LEC","REC","LBROW","RBROW","NTIP",
            "MLEFT","MRIGHT","MTOP","LCHK","RCHK"]

# Nombres de las 45 distancias (C(10,2))
DIST_NAMES = [f"d_{a}_{b}" for a, b in combinations(range(10), 2)]

# ── Nombres de los 68 features (orden fijo) ───────────────────────────────────
FEATURE_NAMES = (
    # Block A — EAR (4)
    ["ear_left","ear_right","ear_mean","ear_asymmetry"] +
    # Block B — Boca (6)
    ["mar","mouth_width","mouth_height",
     "corner_lift_left","corner_lift_right","corner_asymmetry"] +
    # Block C — COM-MEJ (4)
    ["com_mej_left","com_mej_right","com_mej_mean","com_mej_asymmetry"] +
    # Block D — Cejas (6)
    ["brow_h_left","brow_h_right","brow_h_mean",
     "ba","brow_angle_left","brow_angle_right"] +
    # Block E — Nariz / midface (3)
    ["nose_width","philtrum_len","nasolabial_mean"] +
    # Block F — 45 distancias par a par
    DIST_NAMES
)
assert len(FEATURE_NAMES) == 68, f"Feature count: {len(FEATURE_NAMES)}"


# ── Inicializar detector ───────────────────────────────────────────────────────
def crear_detector(model_path: Path) -> vision.FaceLandmarker:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}\n"
            "Ejecuta: python setup.py"
        )
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.25,
        min_face_presence_confidence=0.25,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.FaceLandmarker.create_from_options(opts)


# ── Preprocesado ───────────────────────────────────────────────────────────────
def preprocesar(img_bgr: np.ndarray) -> np.ndarray:
    """
    BGR → CLAHE → resize a 224×224 → RGB
    Maneja tanto imágenes grandes (RAF-DB) como pequeñas (CK+ 48×48).
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    enhanced = clahe.apply(gray)
    resized  = cv2.resize(enhanced, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
    rgb      = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return rgb


# ── Geometría ──────────────────────────────────────────────────────────────────
def lm_arr(lm, idx: int) -> np.ndarray:
    return np.array([lm[idx].x, lm[idx].y, lm[idx].z])

def midpoint(lm, i: int, j: int) -> np.ndarray:
    return (lm_arr(lm, i) + lm_arr(lm, j)) / 2.0

def dist3(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def dist_lm(lm, i: int, j: int) -> float:
    return dist3(lm_arr(lm, i), lm_arr(lm, j))

def ear(lm, eye: list) -> float:
    p1,p2,p3,p4,p5,p6 = eye
    num = dist_lm(lm,p2,p6) + dist_lm(lm,p3,p5)
    den = 2.0 * dist_lm(lm,p1,p4) + 1e-9
    return num / den

def key_points(lm) -> list:
    """Retorna lista de 10 arrays numpy para los puntos clave."""
    LEC   = midpoint(lm, 362, 263)
    REC   = midpoint(lm, 33, 133)
    LBROW = lm_arr(lm, L_BROW_MID)
    RBROW = lm_arr(lm, R_BROW_MID)
    NTIP  = lm_arr(lm, NOSE_TIP)
    MLEFT = lm_arr(lm, MOUTH_L)
    MRIGHT= lm_arr(lm, MOUTH_R)
    MTOP  = lm_arr(lm, MOUTH_TOP)
    LCHK  = lm_arr(lm, FACE_L)
    RCHK  = lm_arr(lm, FACE_R)
    return [LEC, REC, LBROW, RBROW, NTIP, MLEFT, MRIGHT, MTOP, LCHK, RCHK]


# ── Cálculo de los 68 features ────────────────────────────────────────────────
def extraer_features(lm) -> Optional[dict]:
    kp  = key_points(lm)
    LEC, REC = kp[0], kp[1]
    ipd = dist3(LEC, REC) + 1e-9            # distancia intercantal (normalización)
    fh  = dist_lm(lm, FACE_TOP, FACE_BOT) + 1e-9  # alto facial

    # ── Block A: EAR ──────────────────────────────────────────────────────────
    ear_l = ear(lm, L_EYE)
    ear_r = ear(lm, R_EYE)
    ear_m = (ear_l + ear_r) / 2.0
    ear_a = abs(ear_l - ear_r)

    # ── Block B: Boca ─────────────────────────────────────────────────────────
    mw  = dist_lm(lm, MOUTH_L, MOUTH_R)
    mh  = dist_lm(lm, MOUTH_TOP, MOUTH_BOT)
    mar_v = mh / (mw + 1e-9)
    mw_n  = mw / ipd
    mh_n  = mh / ipd

    lip_mid_y = (lm[MOUTH_TOP].y + lm[MOUTH_BOT].y) / 2.0
    cl_l = (lip_mid_y - lm[MOUTH_L].y) / (fh + 1e-9)   # positivo = comisura baja
    cl_r = (lip_mid_y - lm[MOUTH_R].y) / (fh + 1e-9)
    cl_a = abs(cl_l - cl_r)

    # ── Block C: COM-MEJ ──────────────────────────────────────────────────────
    cm_l  = dist_lm(lm, MOUTH_L, FACE_L) / ipd
    cm_r  = dist_lm(lm, MOUTH_R, FACE_R) / ipd
    cm_m  = (cm_l + cm_r) / 2.0
    cm_a  = abs(cm_l - cm_r)

    # ── Block D: Cejas ────────────────────────────────────────────────────────
    bh_l  = dist3(kp[2], kp[0]) / ipd   # LBROW → LEC
    bh_r  = dist3(kp[3], kp[1]) / ipd   # RBROW → REC
    bh_m  = (bh_l + bh_r) / 2.0
    ba    = abs(bh_l - bh_r)            # Brow Asymmetry

    # Ángulo de inclinación de ceja (rads, positivo = extremo externo más alto)
    def brow_angle(inner_idx, outer_idx):
        dx = lm[outer_idx].x - lm[inner_idx].x
        dy = lm[outer_idx].y - lm[inner_idx].y
        return float(np.arctan2(-dy, abs(dx) + 1e-9))  # -dy: y crece hacia abajo

    ba_l = brow_angle(L_BROW_IN, L_BROW_MID)
    ba_r = brow_angle(R_BROW_IN, R_BROW_MID)

    # ── Block E: Nariz / midface ──────────────────────────────────────────────
    nose_w = dist_lm(lm, NOSE_L, NOSE_R) / ipd
    # Filtro: punta nariz → top labio
    philtrum = dist_lm(lm, NOSE_TIP, MOUTH_TOP) / ipd
    # Distancias nasolabiales (comisura → nariz)
    nasolabial = (dist_lm(lm, MOUTH_L, NOSE_L) + dist_lm(lm, MOUTH_R, NOSE_R)) / (2*ipd)

    # ── Block F: 45 distancias par a par ──────────────────────────────────────
    pair_dists = [dist3(kp[i], kp[j]) / ipd
                  for i, j in combinations(range(10), 2)]

    # ── Ensamblar vector ──────────────────────────────────────────────────────
    values = (
        [ear_l, ear_r, ear_m, ear_a] +                          # A: 4
        [mar_v, mw_n, mh_n, cl_l, cl_r, cl_a] +               # B: 6
        [cm_l,  cm_r, cm_m, cm_a] +                            # C: 4
        [bh_l,  bh_r, bh_m, ba, ba_l, ba_r] +                 # D: 6
        [nose_w, philtrum, nasolabial] +                        # E: 3
        pair_dists                                              # F: 45
    )
    assert len(values) == 68

    return {name: round(val, 7) for name, val in zip(FEATURE_NAMES, values)}


# ── Pipeline principal ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Extractor de features FER")
    parser.add_argument("--dataset", required=True,
                        help="Ruta al dataset (subcarpetas: alegria/ dolor/ neutral/)")
    parser.add_argument("--output", default="features.csv",
                        help="Archivo CSV de salida")
    parser.add_argument("--model", default="face_landmarker.task",
                        help="Ruta al modelo MediaPipe")
    parser.add_argument("--max-per-class", type=int, default=8000,
                        help="Máximo de imágenes por clase (balance)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path  = Path(args.output)
    model_path   = Path(args.model)

    print(f"\n{BOLD}{CYAN}{'='*60}")
    print(f"  Extractor FER — 68 features geométricas")
    print(f"{'='*60}{RESET}\n")

    # Verificar estructura del dataset
    for cls in CLASES:
        cls_dir = dataset_path / cls
        if not cls_dir.exists():
            log.warning(f"Carpeta no encontrada: {cls_dir} — se omitirá")

    # Recopilar imágenes por clase
    imagenes = {}
    for cls in CLASES:
        cls_dir = dataset_path / cls
        if not cls_dir.exists():
            continue
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in EXTS]
        if len(imgs) > args.max_per_class:
            rng = np.random.default_rng(42)
            imgs = list(rng.choice(imgs, args.max_per_class, replace=False))
        imagenes[cls] = imgs
        log.info(f"  {cls}: {len(imgs):,} imágenes")

    total = sum(len(v) for v in imagenes.values())
    log.info(f"Total a procesar: {total:,}")

    # Inicializar detector
    log.info("Inicializando FaceLandmarker...")
    detector = crear_detector(model_path)
    log.info(f"{GREEN}Detector listo{RESET}")

    # Extracción
    rows       = []
    skip_count = 0
    ok_count   = {cls: 0 for cls in CLASES}
    t0 = time.time()
    processed  = 0

    for cls, imgs in imagenes.items():
        for img_path in imgs:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                skip_count += 1
                continue

            rgb = preprocesar(img_bgr)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            if not result.face_landmarks:
                skip_count += 1
            else:
                feats = extraer_features(result.face_landmarks[0])
                if feats:
                    feats["label"] = cls
                    rows.append(feats)
                    ok_count[cls] += 1

            processed += 1
            if processed % 200 == 0:
                elapsed = time.time() - t0
                rate    = processed / elapsed
                eta     = (total - processed) / (rate + 1e-9)
                pct     = processed / total * 100
                bar     = "█" * int(pct/5) + "░" * (20-int(pct/5))
                print(f"\r  [{bar}] {pct:.0f}%  ok={len(rows):,}  "
                      f"skip={skip_count}  ETA={eta/60:.1f}min", end="", flush=True)

    print()

    if not rows:
        log.error("No se extrajo ninguna muestra. Verifica el dataset y el modelo.")
        return

    # Guardar
    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ["label"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    elapsed = time.time() - t0
    print(f"\n{GREEN}{BOLD}Extracción completada en {elapsed/60:.1f} min{RESET}")
    print(f"  Muestras extraídas : {len(rows):,}")
    print(f"  Muestras descartadas (sin rostro detectado): {skip_count}")
    print(f"\n  Distribución final:")
    for cls in CLASES:
        if cls in ok_count:
            print(f"    {cls}: {ok_count[cls]:,}")
    print(f"\n  {CYAN}Features: 68 (4 EAR + 6 Boca + 4 COM-MEJ + 6 Cejas + 3 Nariz + 45 distancias){RESET}")
    print(f"  {CYAN}Guardado: {output_path}{RESET}")
    print(f"\n{BOLD}Siguiente paso:{RESET}")
    print(f"  python clasificador.py --csv {output_path} --output modelo_svm.pkl\n")


if __name__ == "__main__":
    main()
