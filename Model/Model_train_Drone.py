#!/usr/bin/env python3
"""
Detección de drones con YOLO26 (cabezal P2) - versión local para tu GPU.

Equivale al notebook de Colab (Dron_detection_3.ipynb), pero:
  - sin Google Drive ni rutas /content
  - sin la API key escrita en el código (se lee de una variable de entorno)
  - dividido en comandos: train, resume, eval, predict

Uso rápido (desde la carpeta del script):
    python drone_yolo.py train                  # entrena yolo26s-p2 a 960 px
    python drone_yolo.py train --size m         # modelo medium
    python drone_yolo.py resume                 # retoma el último entrenamiento
    python drone_yolo.py eval                   # métricas sobre el split test
    python drone_yolo.py predict                # 9 imágenes de test con detecciones

API key de Roboflow (elige una):
    Windows (PowerShell):  $env:ROBOFLOW_API_KEY = "tu_clave"
    Linux / macOS:         export ROBOFLOW_API_KEY="tu_clave"
    (si no existe, el script te la pide por teclado)
"""
from __future__ import annotations

import argparse
import getpass
import os
import random
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Configuración (rutas y dataset)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
RUNS_DIR = ROOT / "runs" / "dron_detection"

ROBOFLOW_WORKSPACE = "isaias-avirod-gmail-com"
ROBOFLOW_PROJECT = "drone-tracking-yageb"
ROBOFLOW_VERSION = 3
ROBOFLOW_FORMAT = "yolo26"
DATASET_FOLDER = "Drone-tracking-3"  # mismo nombre que creaba Roboflow en Colab

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def check_device(device: str) -> None:
    """Verifica que PyTorch vea la GPU antes de gastar tiempo descargando/entrenando."""
    import torch

    if device.lower() == "cpu":
        print("[aviso] Entrenando en CPU: será MUY lento.")
        return

    if not torch.cuda.is_available():
        sys.exit(
            "[error] PyTorch no detecta ninguna GPU CUDA.\n"
            f"        torch {torch.__version__} | build CUDA: {torch.version.cuda}\n"
            "        Si dice 'None' o '+cpu', instalaste la versión de CPU.\n"
            "        Reinstala PyTorch con CUDA desde https://pytorch.org/get-started/locally/\n"
            "        y comprueba también tus drivers con `nvidia-smi`."
        )

    idx = int(device.split(",")[0]) if device.split(",")[0].isdigit() else 0
    props = torch.cuda.get_device_properties(idx)
    print(f"[gpu] {props.name} | {props.total_memory / 1024**3:.1f} GB VRAM | torch {torch.__version__}")


def download_dataset(force: bool = False) -> Path:
    """Descarga el dataset de Roboflow si todavía no está en disco. Devuelve la ruta a data.yaml."""
    location = DATASETS_DIR / DATASET_FOLDER
    data_yaml = location / "data.yaml"

    if data_yaml.exists() and not force:
        print(f"[dataset] Ya existe: {data_yaml}")
        return data_yaml

    from roboflow import Roboflow

    api_key = os.environ.get("ROBOFLOW_API_KEY") or getpass.getpass("Roboflow API key: ")
    rf = Roboflow(api_key=api_key)
    version = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT).version(ROBOFLOW_VERSION)
    version.download(ROBOFLOW_FORMAT, location=str(location))

    if not data_yaml.exists():
        sys.exit(f"[error] La descarga terminó pero no encuentro {data_yaml}")
    return data_yaml


def resolve_weights(value: str) -> Path:
    """Acepta una ruta a .pt o el nombre de una corrida dentro de runs/dron_detection."""
    p = Path(value)
    if p.suffix == ".pt":
        return p
    return RUNS_DIR / value / "weights" / "last.pt"


def pick_best(value: str) -> Path:
    """Para evaluar / predecir: prefiere best.pt de la corrida; acepta también una ruta a un .pt."""
    p = resolve_weights(value)
    if p.name == "last.pt" and (p.parent / "best.pt").exists():
        p = p.parent / "best.pt"
    if not p.exists():
        sys.exit(f"[error] No encuentro {p}")
    return p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detección de drones con YOLO26-P2 (local)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--device", default="0", help='GPU a usar: "0", "1", "0,1" o "cpu" (def. 0)')
        p.add_argument("--imgsz", type=int, default=960)

    # train
    t = sub.add_parser("train", help="Entrena un modelo nuevo")
    common(t)
    t.add_argument("--size", choices=["n", "s", "m", "l"], default="s", help="Tamaño del modelo (def. s)")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--patience", type=int, default=30)
    t.add_argument("--batch", type=int, default=4, help="Tamaño de lote (def. 4, pensado para 12 GB a 960 px con P2). -1 = autobatch, que falló en Colab con este modelo")
    t.add_argument("--workers", type=int, default=4, help="Hilos del dataloader. Baja a 2-4 si hay problemas de RAM")
    t.add_argument("--name", default=None, help="Nombre de la corrida (def. yolo26<size>_p2_<imgsz>)")
    t.add_argument("--weights", default=None, help="Arrancar desde un .pt propio en lugar de los preentrenados")
    t.add_argument("--force-download", action="store_true", help="Volver a descargar el dataset")

    # resume
    r = sub.add_parser("resume", help="Retoma un entrenamiento interrumpido")
    common(r)
    r.add_argument("--run", default="yolo26s_p2_960", help="Nombre de la corrida o ruta a last.pt")

    # eval
    e = sub.add_parser("eval", help="Métricas (P, R, mAP) sobre el split test")
    common(e)
    e.add_argument("--run", default="yolo26s_p2_960", help="Nombre de la corrida o ruta a un .pt")
    e.add_argument("--split", default="test", choices=["val", "test"])

    # predict
    pr = sub.add_parser("predict", help="Inferencia sobre imágenes de ejemplo y muestra una cuadrícula")
    common(pr)
    pr.add_argument("--run", default="yolo26s_p2_960", help="Nombre de la corrida o ruta a un .pt")
    pr.add_argument("--source", default=None, help="Imagen o carpeta (def. test/images del dataset)")
    pr.add_argument("--n", type=int, default=9, help="Cuántas imágenes tomar si source es una carpeta")
    pr.add_argument("--conf", type=float, default=0.25)
    pr.add_argument("--no-show", action="store_true", help="Solo guardar el PNG, sin abrir ventana")

    return parser.parse_args()


# --------------------------------------------------------------------------
# Comandos
# --------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    check_device(args.device)
    data_yaml = download_dataset(force=args.force_download)

    if args.weights:
        model = YOLO(args.weights)
    else:
        # Arquitectura con cabezal P2 + pesos preentrenados (igual que en Colab)
        model = YOLO(f"yolo26{args.size}-p2.yaml").load(f"yolo26{args.size}.pt")

    name = args.name or f"yolo26{args.size}_p2_{args.imgsz}"
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(RUNS_DIR),  # ruta absoluta: evita que Ultralytics la mueva a runs/detect
        name=name,
        save_period=10,
    )
    print(f"\n[ok] Resultados en: {model.trainer.save_dir}")


def cmd_resume(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    check_device(args.device)
    last = resolve_weights(args.run)
    if not last.exists():
        sys.exit(f"[error] No encuentro {last}")

    data_yaml = download_dataset()
    model = YOLO(str(last))
    # data=...: por si el checkpoint recuerda una ruta que ya no existe (p. ej. /content/... de Colab)
    model.train(resume=True, data=str(data_yaml), device=args.device)


def cmd_eval(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    check_device(args.device)
    weights = pick_best(args.run)
    data_yaml = download_dataset()
    model = YOLO(str(weights))
    m = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        project=str(RUNS_DIR),
        name=f"eval_{args.split}",
    )
    print(f"\nPesos: {weights}")
    print(f"Precision : {m.box.mp:.3f}")
    print(f"Recall    : {m.box.mr:.3f}")
    print(f"mAP@50    : {m.box.map50:.3f}")
    print(f"mAP@50-95 : {m.box.map:.3f}")


def cmd_predict(args: argparse.Namespace) -> None:
    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    weights = pick_best(args.run)

    source = Path(args.source) if args.source else DATASETS_DIR / DATASET_FOLDER / "test" / "images"
    if source.is_dir():
        imgs = sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_EXT)
        if not imgs:
            sys.exit(f"[error] No hay imágenes en {source}")
        random.seed(0)
        imgs = random.sample(imgs, min(args.n, len(imgs)))
    elif source.is_file():
        imgs = [source]
    else:
        sys.exit(f"[error] No existe {source}")

    model = YOLO(str(weights))
    results = model.predict([str(p) for p in imgs], imgsz=args.imgsz, conf=args.conf,
                            device=args.device, verbose=False)

    cols = min(3, len(results))
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for ax, path, r in zip(axes.ravel(), imgs, results):
        annotated_rgb = r.plot()[..., ::-1]  # r.plot() devuelve BGR (OpenCV); matplotlib espera RGB
        ax.imshow(annotated_rgb)
        ax.set_title(f"{path.name[:22]}  |  {len(r.boxes)} dron(es)", fontsize=9)

    plt.tight_layout()
    out = RUNS_DIR / "predicciones.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"[ok] Figura guardada en {out}")
    if not args.no_show:
        plt.show()


# --------------------------------------------------------------------------
# Punto de entrada
# --------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    {"train": cmd_train, "resume": cmd_resume, "eval": cmd_eval, "predict": cmd_predict}[args.cmd](args)


# Este "if" es OBLIGATORIO en Windows: el dataloader de PyTorch lanza procesos
# nuevos que reimportan el archivo, y sin esto se relanzaría el entrenamiento.
if __name__ == "__main__":
    main()