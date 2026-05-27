#!/usr/bin/env python3
"""
setup.py — Configuración inicial del pipeline FER
Descarga: face_landmarker.task y FER2013
Uso: python setup.py
"""
import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = PIPELINE_DIR / "face_landmarker.task"
DATA_DIR = PIPELINE_DIR / "data"

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def log(msg, color=CYAN):   print(f"{color}{msg}{RESET}")
def ok(msg):                print(f"{GREEN}✓ {msg}{RESET}")
def warn(msg):              print(f"{YELLOW}⚠ {msg}{RESET}")
def err(msg):               print(f"{RED}✗ {msg}{RESET}"); sys.exit(1)

def check_deps():
    log(f"\n{BOLD}[1/4] Verificando dependencias{RESET}")
    required = ["mediapipe", "cv2", "sklearn", "pandas", "numpy",
                "matplotlib", "seaborn", "torch", "opendatasets"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            ok(pkg)
        except ImportError:
            missing.append(pkg)
            warn(f"Falta: {pkg}")

    if missing:
        warn("Instalando paquetes faltantes...")
        names = {
            "cv2": "opencv-python", "sklearn": "scikit-learn",
            "opendatasets": "opendatasets"
        }
        pkgs = [names.get(p, p) for p in missing]
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "--break-system-packages", "-q"] + pkgs)
        ok("Dependencias instaladas")

def download_model():
    log(f"\n{BOLD}[2/4] Descargando modelo face_landmarker.task{RESET}")
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        ok(f"Modelo ya existe ({MODEL_PATH.stat().st_size/1e6:.1f} MB) — omitiendo")
        return

    log(f"Descargando desde Google Storage...")
    try:
        def progress(count, block, total):
            pct = count * block / total * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"\r  [{bar}] {pct:.0f}%", end="", flush=True)

        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=progress)
        print()
        ok(f"Modelo descargado: {MODEL_PATH.stat().st_size/1e6:.1f} MB")
    except Exception as e:
        print()
        warn(f"No se pudo descargar automáticamente: {e}")
        warn("Descarga manual:")
        warn(f"  URL: {MODEL_URL}")
        warn(f"  Destino: {MODEL_PATH}")
        warn("  O desde: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker")

def download_fer2013():
    log(f"\n{BOLD}[3/4] Descargando FER2013{RESET}")
    DATA_DIR.mkdir(exist_ok=True)
    fer_csv = DATA_DIR / "fer2013.csv"

    if fer_csv.exists() and fer_csv.stat().st_size > 10_000_000:
        ok(f"FER2013 ya existe ({fer_csv.stat().st_size/1e6:.0f} MB) — omitiendo")
        return

    # Método 1: opendatasets (solicita credenciales Kaggle interactivamente)
    try:
        import opendatasets as od
        log("Usando opendatasets (necesita usuario y API key de Kaggle)...")
        log("Obtén tu API key en: https://www.kaggle.com/settings → API → Create New Token")
        od.download("https://www.kaggle.com/datasets/msambare/fer2013",
                    data_dir=str(DATA_DIR))
        # Mover CSV si quedó en subdirectorio
        for csv in DATA_DIR.rglob("fer2013.csv"):
            if csv != fer_csv:
                csv.rename(fer_csv)
                break
        ok("FER2013 descargado")
        return
    except Exception as e:
        warn(f"opendatasets falló: {e}")

    # Método 2: kaggle CLI
    kaggle_cfg = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_cfg.exists():
        try:
            log("Intentando con kaggle CLI...")
            subprocess.check_call(
                ["kaggle", "datasets", "download", "-d", "msambare/fer2013",
                 "-p", str(DATA_DIR), "--unzip"],
                stdout=subprocess.DEVNULL
            )
            ok("FER2013 descargado via kaggle CLI")
            return
        except Exception as e:
            warn(f"kaggle CLI falló: {e}")

    warn("No se pudo descargar FER2013 automáticamente.")
    warn("Instrucciones manuales:")
    warn("  1. Ve a https://www.kaggle.com/datasets/msambare/fer2013")
    warn("  2. Descarga 'fer2013.csv'")
    warn(f"  3. Colócalo en: {DATA_DIR}/fer2013.csv")

def verify_setup():
    log(f"\n{BOLD}[4/4] Verificando setup{RESET}")
    issues = 0

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        ok(f"Modelo: {MODEL_PATH}")
    else:
        warn(f"Modelo no encontrado en {MODEL_PATH}")
        issues += 1

    fer_csv = DATA_DIR / "fer2013.csv"
    if fer_csv.exists() and fer_csv.stat().st_size > 1_000_000:
        ok(f"Dataset: {fer_csv}")
    else:
        warn(f"FER2013 no encontrado en {fer_csv}")
        issues += 1

    if issues == 0:
        log(f"\n{GREEN}{BOLD}Setup completo. Ejecuta:{RESET}")
        log(f"  python extractor.py    # ~30-60 min según tu CPU")
        log(f"  python clasificador.py # ~5-15 min")
    else:
        warn(f"\n{issues} problema(s) pendiente(s). Revisa las instrucciones arriba.")

if __name__ == "__main__":
    log(f"{BOLD}{'='*50}")
    log(f"  Pipeline FER — Setup")
    log(f"{'='*50}{RESET}")
    check_deps()
    download_model()
    download_fer2013()
    verify_setup()
