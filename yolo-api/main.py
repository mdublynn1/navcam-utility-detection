"""
Field Engineering YOLO API
FastAPI service for training and running YOLO object detection
on field engineering assets: utility poles, hand holes, vaults,
conduit, and pipes.
"""

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import subprocess
import shutil
import os
import json
import uuid
import re
import csv
import zipfile
import cv2
import gpxpy
import gpxpy.gpx
import geojson
import base64
import requests as http_requests
from datetime import datetime, timezone

app = FastAPI(
    title="Field Engineering YOLO API",
    description="Train and run YOLO object detection on field engineering assets",
    version="1.0.0"
)

# ── Paths (set via env vars in docker-compose) ──────────────
DATASETS_PATH = Path(os.getenv("DATASETS_PATH", "/data/datasets/field-engineering"))
RUNS_PATH     = Path(os.getenv("RUNS_PATH",     "/data/runs"))
INCOMING_PATH = Path(os.getenv("INCOMING_PATH", "/data/incoming"))
DATA_YAML     = DATASETS_PATH / "data.yaml"

# ── Training state ───────────────────────────────────────────
training_state = {
    "status":    "idle",       # idle | running | complete | error
    "message":   "Ready",
    "started_at": None,
    "finished_at": None,
    "run_name":  None,
}


# ── Helpers ──────────────────────────────────────────────────

def get_best_model() -> Path | None:
    """Return the best.pt from the most recent training run."""
    candidates = sorted(RUNS_PATH.glob("**/weights/best.pt"))
    return candidates[-1] if candidates else None


def count_images(split: str = "train") -> int:
    d = DATASETS_PATH / "images" / split
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})


def _run_training(epochs: int, model: str, run_name: str):
    global training_state
    training_state.update({
        "status": "running",
        "message": f"Training {model} for {epochs} epochs",
        "started_at": datetime.utcnow().isoformat(),
        "finished_at": None,
        "run_name": run_name,
    })

    cmd = [
        "yolo", "detect", "train",
        f"data={DATA_YAML}",
        f"model={model}",
        f"epochs={epochs}",
        "imgsz=640",
        f"project={RUNS_PATH}",
        f"name={run_name}",
        "device=cpu",
        "workers=2",
        "patience=20",   # early stopping
        "save=True",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    finished = datetime.utcnow().isoformat()

    if result.returncode == 0:
        training_state.update({
            "status": "complete",
            "message": "Training finished successfully",
            "finished_at": finished,
        })
    else:
        training_state.update({
            "status": "error",
            "message": result.stderr[-500:] if result.stderr else "Unknown error",
            "finished_at": finished,
        })


# ── Routes ───────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Field Engineering YOLO API",
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "training": training_state["status"],
        "dataset": {
            "train_images": count_images("train"),
            "val_images": count_images("val"),
            "data_yaml_exists": DATA_YAML.exists(),
        },
        "best_model": str(get_best_model()) if get_best_model() else None,
    }


# ── Training ─────────────────────────────────────────────────

@app.post("/train", tags=["Training"])
def start_training(
    background_tasks: BackgroundTasks,
    epochs: int = 50,
    model: str = "yolo11n.pt",
):
    """
    Start a YOLO training run.

    - **epochs**: number of training epochs (default 50)
    - **model**: base model to fine-tune (yolo11n.pt / yolo11s.pt / yolo11m.pt)
    """
    if training_state["status"] == "running":
        raise HTTPException(409, "Training is already in progress")

    if not DATA_YAML.exists():
        raise HTTPException(400, f"data.yaml not found at {DATA_YAML}. Run /dataset/init first.")

    train_count = count_images("train")
    if train_count == 0:
        raise HTTPException(400, "No training images found. Add images to data/datasets/field-engineering/images/train/")

    run_name = f"field-eng-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    background_tasks.add_task(_run_training, epochs, model, run_name)

    return {
        "message": "Training started",
        "run_name": run_name,
        "epochs": epochs,
        "model": model,
        "train_images": train_count,
    }


@app.get("/train/status", tags=["Training"])
def training_status():
    """Check the current training status."""
    return training_state


@app.get("/train/results", tags=["Training"])
def training_results():
    """List all completed training runs and their metrics."""
    runs = []
    for results_csv in RUNS_PATH.glob("*/results.csv"):
        run_dir = results_csv.parent
        best    = run_dir / "weights" / "best.pt"
        runs.append({
            "run": run_dir.name,
            "path": str(run_dir),
            "best_model": str(best) if best.exists() else None,
            "results_csv": str(results_csv),
        })
    return {"runs": runs}


# ── Inference ────────────────────────────────────────────────

PREDICTIONS_PATH = Path(os.getenv("PREDICTIONS_PATH", "/data/predictions"))

def _run_batch_predict(model: str, conf: float, run_name: str, image_paths: list):
    """Run YOLO detection on a list of images."""
    out_dir = PREDICTIONS_PATH / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results_summary = []
    for img_path in image_paths:
        cmd = [
            "yolo", "detect", "predict",
            f"model={model}",
            f"source={img_path}",
            f"conf={conf}",
            "save=True",
            "save_txt=True",
            "save_conf=True",
            f"project={out_dir}",
            f"name={Path(img_path).stem}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        results_summary.append({
            "image": Path(img_path).name,
            "success": result.returncode == 0,
        })

    return results_summary


@app.post("/predict/incoming", tags=["Inference"])
def predict_incoming(
    background_tasks: BackgroundTasks,
    conf: float = 0.5,
    move_after: bool = False,
):
    """
    Run object detection on ALL images currently in data/incoming/.

    - **conf**: confidence threshold (default 0.5)
    - **move_after**: if True, moves processed images to data/incoming/processed/ after detection
    """
    best = get_best_model()
    if not best or not best.exists():
        raise HTTPException(404, "No trained model found. Run POST /train first.")

    extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = [f for f in INCOMING_PATH.iterdir() if f.suffix in extensions]

    if not images:
        raise HTTPException(400, "No images found in data/incoming/. Drop some images in there first.")

    run_name = f"batch-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    out_dir  = PREDICTIONS_PATH / run_name

    # Move processed images if requested
    if move_after:
        processed_dir = INCOMING_PATH / "processed"
        processed_dir.mkdir(exist_ok=True)
        for img in images:
            shutil.move(str(img), processed_dir / img.name)
        source_paths = [str(processed_dir / img.name) for img in images]
    else:
        source_paths = [str(img) for img in images]

    background_tasks.add_task(_run_batch_predict, str(best), conf, run_name, source_paths)

    return {
        "message": f"Detection started on {len(images)} images",
        "run_name": run_name,
        "images": [img.name for img in images],
        "output_dir": str(out_dir),
        "model_used": str(best),
        "tip": f"Results (annotated images + .txt detections) saved to data/predictions/{run_name}/",
    }


@app.get("/predict/results", tags=["Inference"])
def list_prediction_results():
    """List all previous detection runs and their output folders."""
    PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)
    runs = []
    for run_dir in sorted(PREDICTIONS_PATH.iterdir(), reverse=True):
        if run_dir.is_dir():
            images = list(run_dir.rglob("*.jpg")) + list(run_dir.rglob("*.png"))
            runs.append({
                "run": run_dir.name,
                "output_dir": str(run_dir),
                "annotated_images": len(images),
            })
    return {"prediction_runs": runs}


@app.post("/predict", tags=["Inference"])
async def predict(
    file: UploadFile = File(...),
    conf: float = 0.5,
    model_path: str = None,
):
    """
    Run object detection on an uploaded image.

    - **file**: image file (jpg/png)
    - **conf**: confidence threshold (default 0.5)
    - **model_path**: override model path (uses best trained model by default)
    """
    best = get_best_model()
    model = model_path or (str(best) if best else None)

    if not model or not Path(model).exists():
        raise HTTPException(
            404,
            "No trained model found. Run POST /train first, or provide a model_path."
        )

    # Save uploaded image to temp location
    tmp = Path(f"/tmp/{file.filename}")
    with open(tmp, "wb") as f:
        f.write(await file.read())

    out_name = f"predict-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    cmd = [
        "yolo", "detect", "predict",
        f"model={model}",
        f"source={tmp}",
        f"conf={conf}",
        "save=True",
        "save_txt=True",
        f"project={RUNS_PATH}",
        f"name={out_name}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)

    if result.returncode == 0:
        out_dir = RUNS_PATH / out_name
        return {
            "message": "Detection complete",
            "output_dir": str(out_dir),
            "model_used": model,
            "confidence_threshold": conf,
        }
    else:
        raise HTTPException(500, result.stderr[-500:])


# ── Dataset management ───────────────────────────────────────

@app.post("/dataset/init", tags=["Dataset"])
def init_dataset():
    """
    Create the data.yaml file and folder structure for
    field engineering object classes.
    """
    yaml_content = """# Field Engineering Object Detection Dataset
path: /data/datasets/field-engineering
train: images/train
val:   images/val

nc: 5
names:
  - utility_pole
  - hand_hole
  - vault
  - conduit
  - pipe
"""
    for folder in [
        DATASETS_PATH / "images" / "train",
        DATASETS_PATH / "images" / "val",
        DATASETS_PATH / "labels" / "train",
        DATASETS_PATH / "labels" / "val",
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    DATA_YAML.write_text(yaml_content)

    return {
        "message": "Dataset initialized",
        "data_yaml": str(DATA_YAML),
        "classes": ["utility_pole", "hand_hole", "vault", "conduit", "pipe"],
        "next_step": "Add images to data/datasets/field-engineering/images/train/ then POST /train",
    }


@app.post("/dataset/ingest", tags=["Dataset"])
def ingest_incoming(split: str = "train"):
    """
    Move all images from data/incoming/ into the training or val dataset.

    - **split**: 'train' or 'val'
    """
    if split not in {"train", "val"}:
        raise HTTPException(400, "split must be 'train' or 'val'")

    dest = DATASETS_PATH / "images" / split
    dest.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    moved = []

    for img in INCOMING_PATH.iterdir():
        if img.suffix in extensions:
            target = dest / img.name
            shutil.move(str(img), str(target))
            moved.append(img.name)

    return {
        "moved": moved,
        "count": len(moved),
        "destination": str(dest),
        "message": f"Moved {len(moved)} images to {split} set",
    }


@app.get("/dataset/stats", tags=["Dataset"])
def dataset_stats():
    """Show image counts for training and validation sets."""
    return {
        "train_images": count_images("train"),
        "val_images": count_images("val"),
        "data_yaml_exists": DATA_YAML.exists(),
        "classes": ["utility_pole", "hand_hole", "vault", "conduit", "pipe"],
        "incoming_images": sum(
            1 for f in INCOMING_PATH.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ) if INCOMING_PATH.exists() else 0,
    }


@app.delete("/dataset/clear-incoming", tags=["Dataset"])
def clear_incoming():
    """Remove all files from the incoming/ staging folder."""
    removed = []
    for f in INCOMING_PATH.iterdir():
        f.unlink()
        removed.append(f.name)
    return {"removed": removed, "count": len(removed)}


# ── Models ───────────────────────────────────────────────────

@app.get("/models", tags=["Models"])
def list_models():
    """List all trained model weights available."""
    models = []
    for pt in RUNS_PATH.glob("**/weights/*.pt"):
        models.append({
            "name": pt.name,
            "run": pt.parent.parent.name,
            "path": str(pt),
            "size_mb": round(pt.stat().st_size / 1_048_576, 2),
        })
    return {"models": models, "count": len(models)}


# ── Google Vision OCR ────────────────────────────────────────
_vision_api_key: str | None = os.getenv("GOOGLE_VISION_API_KEY", None)
VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"


def set_vision_api_key(key: str):
    global _vision_api_key
    _vision_api_key = key.strip()


def _vision_read(frame_bgr, region: str = "full") -> str:
    """
    Send frame (or a cropped region) to Google Cloud Vision API, return detected text.
    region: "full" | "bottom" | "top"  — crop to that strip before calling the API.
    Cropping to the GPS overlay band dramatically cuts noise and improves accuracy.
    """
    if not _vision_api_key:
        return ""

    # Crop to the likely GPS overlay band
    if region != "full" and frame_bgr is not None:
        h = frame_bgr.shape[0]
        if region == "bottom":
            frame_bgr = frame_bgr[int(h * 0.75):, :]   # bottom 25 %
        elif region == "top":
            frame_bgr = frame_bgr[:int(h * 0.25), :]   # top 25 %

    try:
        _, buf   = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        b64_img  = base64.b64encode(buf).decode("utf-8")
        payload  = {"requests": [{"image": {"content": b64_img},
                                   "features": [{"type": "TEXT_DETECTION", "maxResults": 1}],
                                   "imageContext": {"languageHints": ["en"]}}]}
        resp     = http_requests.post(f"{VISION_API_URL}?key={_vision_api_key}",
                                      json=payload, timeout=10)
        data     = resp.json()
        if "error" in data:
            return f"API_ERROR: {data['error'].get('message','')}"
        anns = data.get("responses", [{}])[0].get("textAnnotations", [])
        return anns[0].get("description", "") if anns else ""
    except Exception as e:
        return f"EXCEPTION: {e}"


def _clean_ocr_number(text: str) -> str:
    """
    Normalize raw OCR text for GPS coordinate extraction.
    Applied to the full OCR string before pattern matching.

    Handles observed iPhone NavCam overlay OCR artifacts:
      • degree symbol misread as `"` or `\` (e.g. `-086.71239"`)
      • decimal point misread as `/` (e.g. `+030/40972`)
      • decimal point dropped entirely (e.g. `-08671261` → `-086.71261`)
      • space injected inside the decimal portion (e.g. `030.409 68` → `030.40968`)
      • lat/lng glued together without a separator (e.g. `+030.40972-086.71261`)
    """
    s = text
    # Collapse newlines / carriage returns into spaces — Vision API often splits mid-coord
    s = re.sub(r'[\r\n]+', ' ', s)
    # Strip degree/minute/second symbols and common OCR misreads of them (", \)
    s = re.sub(r'[°º˚′″"\\]', ' ', s)
    # Capital O between digits → 0
    s = re.sub(r'(?<=\d)[Oo](?=\d)', '0', s)
    # I or l between digits → 1
    s = re.sub(r'(?<=\d)[Il](?=\d)', '1', s)
    # Pipe character (|) used as slash separator by some OCR engines
    s = re.sub(r'\s*\|\s*', ' / ', s)
    # Slash misread as decimal point inside a coord: "+030/40972" → "+030.40972"
    # (digits on both sides, 2-3 int digits + 4-6 decimal digits — matches overlay format)
    s = re.sub(r'([+-]?)(\d{2,3})/(\d{4,6})(?!\d)', r'\1\2.\3', s)
    # Missing decimal on 8-digit signed coord: "-08671261" → "-086.71261"
    # (iPhone NavCam overlay format is always +/-DDD.DDDDD, 3 int + 5 decimal)
    s = re.sub(r'([+-])(\d{3})(\d{5})(?!\d)', r'\1\2.\3', s)
    # Space injected inside the decimal portion: "030.409 68" → "030.40968"
    s = re.sub(r'(\d{2,3}\.\d{3})\s+(\d{2,3})(?!\d)', r'\1\2', s)
    # "028 54484" or "028,54484" — space/comma between 2-3 int digits and 4-5 decimal digits
    # is almost always a misread decimal point (e.g. +028.54484 read as +028 54484)
    s = re.sub(r'(\d{2,3})[ ,](\d{4,5})(?!\d)', r'\1.\2', s)
    # Glued signed pair (no separator between lat and lng): "+030.40972-086.71261"
    #   → "+030.40972 / -086.71261"
    s = re.sub(r'(\.\d{4,6})([+-]\d{2,3}\.\d{4,6})', r'\1 / \2', s)
    # Collapse runs of whitespace
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()


def preprocess_frame_for_ocr(frame_bgr):
    """
    Minimal preprocessing for OCR.  Google Vision handles the raw JPEG well,
    so we just return a PIL image (for any Tesseract fallback) and the RGB array.
    Returns (pil_image, numpy_rgb).
    """
    try:
        from PIL import Image
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb), rgb
    except Exception:
        return None, None


def _tesseract_read(pil_img) -> str:
    """Tesseract stub — not installed in this container; returns empty string."""
    return ""


# ── Coordinate extraction ─────────────────────────────────────
_GPS_PATTERNS = [
    # "Position: +028.54521 / -080.79476" — exact app overlay format
    re.compile(
        r'[Pp]osition[:\s]+([+-]?\d{2,3}\.\d+)\s*[/|\\]\s*([+-]\d{2,3}\.\d+)'
    ),
    # "Position: 028.54521 / -080.79476" — missing + on lat
    re.compile(
        r'[Pp]osition[:\s]+(\d{2,3}\.\d+)\s*[/|\\]\s*([+-]\d{2,3}\.\d+)'
    ),
    # Signed slash-separated (allow pipe/backslash as slash): +028.54521 / -080.79476
    re.compile(
        r'([+-]\d{2,3}\.\d{4,})\s*[/|\\]\s*([+-]\d{2,3}\.\d{4,})'
    ),
    # Unsigned lat + signed lng: 028.54521 / -080.79476
    re.compile(
        r'(\d{2,3}\.\d{4,})\s*[/|\\]\s*([+-]\d{2,3}\.\d{4,})'
    ),
    # IKE/app format: lat: 27.34393, lng: -82.53857
    re.compile(
        r'lat[itude]*[:\s]+([+-]?\d{1,3}\.\d+)[,\s]+lon[gitude]*[:\s]+([+-]?\d{1,3}\.\d+)',
        re.IGNORECASE
    ),
    # GPS label followed by decimals
    re.compile(
        r'(?:gps|location|coord(?:inates?)?)[:\s]+([+-]?\d{1,3}\.\d+)[,\s]+([+-]?\d{1,3}\.\d+)',
        re.IGNORECASE
    ),
    # Plain signed decimal pair anywhere in text
    re.compile(
        r'([+-]\d{1,3}\.\d{4,})\s*[,/]\s*([+-]\d{1,3}\.\d{4,})'
    ),
    # DMS format: N 27° 20' 38" W 82° 32' 19"
    re.compile(
        r'([NS])\s*(\d{1,3})\s*(\d{1,2})\s*(\d{1,2}(?:\.\d+)?)\s*'
        r'([EW])\s*(\d{1,3})\s*(\d{1,2})\s*(\d{1,2}(?:\.\d+)?)',
        re.IGNORECASE
    ),
]


def _dms_to_decimal(deg: float, mins: float, secs: float, direction: str) -> float:
    """Convert degrees/minutes/seconds to decimal degrees."""
    val = deg + mins / 60 + secs / 3600
    return -val if direction.upper() in ("S", "W") else val


def _validate_coords(lat: float, lng: float) -> bool:
    """Basic sanity check on coordinate range."""
    return -90 <= lat <= 90 and -180 <= lng <= 180


def _extract_coords_from_text(raw_text: str) -> tuple[float, float] | None:
    """
    Extract lat/lng from OCR text using format-anchored patterns.
    Tries both raw and cleaned text.  Falls back to numeric anchor search.
    """
    cleaned = _clean_ocr_number(raw_text)

    # ── Strategy 1: regex patterns on raw and cleaned text ───
    for text_variant in (raw_text, cleaned):
        for pat in _GPS_PATTERNS:
            m = pat.search(text_variant)
            if not m:
                continue
            groups = m.groups()
            if len(groups) == 8:
                # DMS format
                try:
                    lat = _dms_to_decimal(float(groups[1]), float(groups[2]),
                                          float(groups[3]), groups[0])
                    lng = _dms_to_decimal(float(groups[5]), float(groups[6]),
                                          float(groups[7]), groups[4])
                except (ValueError, IndexError):
                    continue
            else:
                try:
                    lat = float(_clean_ocr_number(groups[0]).replace(',', '.'))
                    lng = float(_clean_ocr_number(groups[1]).replace(',', '.'))
                    # Longitude in Americas should be negative; fix if needed
                    if lng > 0 and lat > 0 and lng > lat:
                        lng = -lng
                except ValueError:
                    continue
            if _validate_coords(lat, lng):
                return lat, lng

    # ── Strategy 2: numeric anchor search ────────────────────
    # Scan for anything that looks like DDD.DDDDD and classify by value range.
    # Tuned for US Southeast (Florida ~28 N, 80-82 W).
    anchor_pat = re.compile(r'(\d{2,3})[.,](\d{4,6})')
    candidates = anchor_pat.findall(cleaned)

    lat_val = lng_val = None
    for integer_s, decimal_s in candidates:
        try:
            val = float(f"{integer_s}.{decimal_s[:5]}")   # cap at 5 decimal places
        except ValueError:
            continue
        # Latitude: plausible US range 20–50
        if 20 <= val <= 50 and lat_val is None:
            lat_val = val
        # Longitude: plausible US range (stored as positive, will negate)
        elif 65 <= val <= 130 and lng_val is None:
            lng_val = -val

    if lat_val is not None and lng_val is not None and _validate_coords(lat_val, lng_val):
        return lat_val, lng_val

    return None


def ocr_frame_for_gps(frame_bgr) -> dict | None:
    """
    Extract GPS coordinates from a video frame using Google Vision API.
    Tries bottom strip → top strip → full frame for best accuracy.
    Returns {"lat", "lng", "bearing", "raw_text", "region", "method"}
    where "bearing" is the compass heading in degrees (or None if the overlay
    does not include one). Returns None if no coordinates are found.
    """
    try:
        for region in ("bottom", "top", "full"):
            raw_text = _vision_read(frame_bgr, region=region)
            if not raw_text or raw_text.startswith(("API_ERROR", "EXCEPTION")):
                continue
            result = _extract_coords_from_text(raw_text)
            if result:
                lat, lng = result
                bearing = parse_bearing_deg(raw_text)
                return {
                    "lat":      round(lat, 7),
                    "lng":      round(lng, 7),
                    "bearing":  round(bearing, 2) if bearing is not None else None,
                    "raw_text": raw_text.strip()[:500],
                    "region":   region,
                    "method":   "vision",
                }

        return None

    except Exception:
        return None


# ── Video Pipeline ────────────────────────────────────────────
# Geocoded video + GPX → frame extraction → GPS interpolation
# → YOLO detection → GeoJSON output

PIPELINE_PATH = Path(os.getenv("PIPELINE_PATH", "/data/pipeline"))
pipeline_jobs: dict = {}   # in-memory job tracker

# ── Class names for the current 28-class model (Run 5) ───────
# These must match the order/indices used during training.
# Indices 0..16 are the Run 3 base (preserved exactly).
# Indices 17..27 were added by merging telecom.v3 in Run 5.
CLASS_NAMES = {
    0:  "pole",
    1:  "conduit",
    2:  "manhole",
    3:  "splice_enclosure",
    4:  "transformer",
    5:  "bush",
    6:  "car",
    7:  "chair",
    8:  "crosswalk",
    9:  "curb",
    10: "fire_hydrant",
    11: "obstacle",
    12: "sign",
    13: "stairs",
    14: "table",
    15: "trashcan",
    16: "tree",
    17: "culvert",
    18: "hand_hole",
    19: "power_marks",
    20: "ramp",
    21: "riser",
    22: "sidewalk",
    23: "street_light",
    24: "telecom",
    25: "u_guard",
    26: "valve",
    27: "water_mark",
}

# ── Per-class confidence thresholds ──────────────────────────
# The pipeline passes --conf=<min_of_these> to YOLO (so nothing is
# dropped too early) and then filters in Python with this table.
# Use this to raise the bar ONLY for classes that over-fire.
#
# Calibrated from detection audits:
#   pole:    low-confidence palm trees get called poles. Need >= 0.35 to
#            keep most real poles while dropping the noisiest palms.
#   manhole: textured ground (tactile paving / brick crosswalks) gets
#            called manhole at 0.5–0.6. Need >= 0.60 until we retrain.
#   tree:    low-conf tree detections are mostly fine; keeping at 0.30
#            to catch more negatives we can use for palm-vs-pole training.
#
# Run 5 new-class overrides — raised on purpose:
# These were trained on ~15 images oversampled 50x, so the model is
# highly confident on near-duplicates of training but unreliable on
# novel instances. High thresholds (0.55+) keep only very confident
# calls until we have more labeled data. Revisit after Run 6.
CLASS_CONF_OVERRIDE = {
    "pole":         0.25,   # lowered from 0.35 — field test had 3 poles,
                            # YOLO only caught 2 at 0.35. Trade-off: may
                            # introduce a few extra phantoms which the
                            # cross-class dedup should catch.
    "manhole":      0.60,
    "tree":         0.30,
    "culvert":      0.55,
    "hand_hole":    0.55,
    "power_marks":  0.30,   # lowered from 0.55 — painted ground marks are
                            # faint; need lower bar to fire at all
    "ramp":         0.60,   # only 1 train instance — be very skeptical
    "riser":        0.55,
    "sidewalk":     0.50,
    "street_light": 0.25,   # lowered 0.50 → 0.30 → 0.25. Job eee27029
                            # produced ZERO streetlight detections even
                            # though one is clearly in the scene. Drop
                            # another 0.05 to let borderline frames
                            # through; the cross-class dedup can catch
                            # any phantoms.
    "telecom":      0.60,   # only 1 train instance
    "u_guard":      0.55,
    "valve":        0.60,   # 0 train instances in v3 — won't fire usefully
    "water_mark":   0.30,   # lowered from 0.55 — same reasoning as power_marks
    # everything else falls back to the pipeline's `conf` parameter
}

# ── Monocular geolocation (per-object lat/lon) ───────────────
# iPhone 14 main/wide lens, 1080p video in landscape sensor = ~68° H / ~40° V.
# User records PORTRAIT (phone held vertical) -> the rendered frame is
# tall/narrow and the FOVs swap relative to the image axes:
#   image horizontal span  ≈ 40°
#   image vertical   span  ≈ 68°
# These can be refined per-device via a one-time calibration test.
CAMERA_FOV_H_DEG = 40.0
CAMERA_FOV_V_DEG = 68.0

# Typical real-world heights in meters per class. Used to turn a box's
# pixel height into an estimated camera-to-object distance. Missing
# classes fall back to the camera lat/lon (no triangulation).
#
# Run 5 additions: pole-mounted / upright objects get real heights so
# they can be triangulated. Nearly-flat ground markings (power_marks,
# ramp, sidewalk, water_mark) are intentionally omitted — their pixel
# height is too sensitive to camera angle to give a useful distance
# estimate, so they'll just geolocate to the camera position.
CLASS_HEIGHTS_M = {
    "pole":             11.0,  # ~36ft wood distribution pole
    "transformer":       1.2,  # can body, pole-mount
    "fire_hydrant":      0.8,
    "manhole":           0.6,  # cover diameter (object is ~flat)
    "splice_enclosure":  0.6,
    "conduit":           2.0,  # riser visible on a pole
    "sign":              2.4,  # typical mounting height of face center
    "pedestal":          1.0,
    "car":               1.5,
    "trashcan":          0.9,
    "fire_hydrant ":     0.8,  # (defensive: trailing space variants)
    # Run 5 new classes — upright structures
    "riser":             2.0,  # pole-mounted conduit (same as 'conduit')
    "street_light":      8.0,  # ~25ft mast, taller than distribution pole
    "u_guard":           1.8,  # U-shaped pole guard, ~5-6ft
    "telecom":           1.5,  # ground-mount telecom cabinet
    "hand_hole":         0.5,  # small utility box / cover
    "culvert":           0.6,  # visible opening height
    "valve":             0.2,  # water valve cover
    # Run 5 flat / ground markings — no triangulation (camera lat/lon)
    # "power_marks", "ramp", "sidewalk", "water_mark" intentionally omitted
}


# ── Cross-frame detection dedup ─────────────────────────────────
# Same physical asset (pole, manhole, etc.) shows up in many consecutive
# video frames. We collapse those into a single placemark using spatial +
# temporal clustering:
#   * DBSCAN-with-min_samples=1 equivalent (connected components)
#   * distance metric: great-circle haversine in feet
#   * hard time gate: pairs > DEDUP_MAX_TIME_DIFF_SEC seconds apart never
#     merge (catches the "drove past the same spot twice" case)
#
# Per-class eps is tuned from what makes physical sense (pole spacing,
# sidewalk segment length, etc.). Anything not listed uses the default.
DEDUP_DISTANCE_THRESHOLDS_FT = {
    # Run 3 base classes
    "pole":           20,  # drive-by GPS wobble
    "transformer":    12,
    "cabinet":        10,
    "pedestal":        8,
    "manhole":         6,
    "fire_hydrant":    6,
    "splice_enclosure": 6,
    "tree":           25,  # wide canopies + noisy monocular depth
    # Run 5 new classes
    "culvert":        25,  # bumped from 15 (2.8x collapse was weak)
    "hand_hole":      20,  # 25 over-merged 3 raw pins into 1; need 2.
                           # 20 ft keeps pins ≥20 ft apart distinct while
                           # still collapsing the 13-18 ft duplicate pair.
    "handhole":       20,  # defensive alias
    "riser":          15,  # bumped from 8
    "u_guard":         8,
    "street_light":   30,  # bumped from 15
    "telecom":         8,
    "valve":           4,
    "sidewalk":       30,  # long segment features
    "ramp":           20,
    "water_mark":      6,
    "power_marks":     6,
}
DEDUP_DEFAULT_FT        = 10.0
DEDUP_MAX_TIME_DIFF_SEC = 6.0  # drive-by at 30 mph: 6s = ~260 ft of travel
_EARTH_R_FT             = 20_902_231.0  # mean earth radius, feet


def _haversine_ft(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in feet."""
    import math as _m
    phi1, phi2 = _m.radians(lat1), _m.radians(lat2)
    dphi = _m.radians(lat2 - lat1)
    dlam = _m.radians(lon2 - lon1)
    a = (_m.sin(dphi / 2) ** 2
         + _m.cos(phi1) * _m.cos(phi2) * _m.sin(dlam / 2) ** 2)
    return 2 * _EARTH_R_FT * _m.asin(_m.sqrt(a))


def _dedup_cluster(points: list, eps_ft: float,
                   max_time_diff: float = DEDUP_MAX_TIME_DIFF_SEC) -> list:
    """Union-find clustering on (lat, lon, timestamp) with class-specific eps.

    Equivalent to DBSCAN(eps=eps_ft, min_samples=1, metric='precomputed')
    where the precomputed matrix pins time-gated pairs to +inf so they
    can never merge. Returns a list of clusters; each cluster is a list
    of the original point dicts.
    """
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [points]

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            if abs(pi["timestamp"] - pj["timestamp"]) > max_time_diff:
                continue
            if _haversine_ft(pi["lat"], pi["lon"],
                             pj["lat"], pj["lon"]) <= eps_ft:
                union(i, j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(points[i])
    return list(groups.values())


# Per-class perpendicular offset (meters) — how far off the drive line
# the asset typically sits when the truck is passing it at closest
# approach. Kept tiny on purpose (3-5 m max) because ANYTHING larger
# just means trusting monocular depth again, which we've already seen
# blow up on partial-view objects.
DEDUP_ROADSIDE_OFFSET_M = {
    "sidewalk":     0.0,   # sidewalk is directly under the camera
    "ramp":         0.0,
    "water_mark":   0.0,
    "power_marks":  0.0,
    "hand_hole":    1.5,
    "handhole":     1.5,
    "manhole":      1.5,
    "riser":        2.0,
    "valve":        1.5,
    "fire_hydrant": 3.0,
    "telecom":      3.0,
    "pedestal":     3.0,
    "cabinet":      3.0,
    "culvert":      3.0,
    "pole":         4.0,
    "transformer":  4.0,
    "street_light": 4.0,
    "u_guard":      3.0,
    "tree":         5.0,
    "splice_enclosure": 2.0,
}
DEDUP_DEFAULT_OFFSET_M = 3.0

# Per-class maximum plausible distance-from-camera (meters) used to
# CLAMP monocular distance estimates during the cam+bearing fallback.
# Monocular depth routinely over-estimates 2–4× for partial-view or
# occluded objects; clamping protects us from 70 m "pole" claims that
# are really 8 m poles rendered with a tiny bbox because only the top
# is visible. Values are upper bounds drawn from typical roadside
# setback distances for US utilities.
# Per-class maximum plausible object distance. Used only as a last-resort
# sanity cap on monocular-depth estimates. Assets that are curb-adjacent
# by definition get tight caps; assets that can realistically sit far
# from the camera path (poles across an easement, culverts under a
# cross-road, trees in a field) get looser caps so we don't under-shoot
# when they really are distant.
# Per-class REALISTIC distance-from-camera for a pedestrian walking
# along a sidewalk next to a road. These are the true physical setbacks
# of US roadside utilities, not theoretical ranges. Monocular depth
# estimates from YOLO bboxes are systematically inflated 3–4× (partial
# view bias — the model only sees the bottom of a 10 m pole), so the
# clamp is the dominant accuracy mechanism, not the mono distance
# itself. Over-clamping is far better than over-projecting; a pin 5 m
# out when the pole is actually 3 m out is close enough, whereas a pin
# 25 m out lands on the other side of the road.
DEDUP_MAX_OBJ_DISTANCE_M = {
    # under-foot (you walk ON them) — always very close
    "sidewalk":         3.0,
    "ramp":             3.0,
    "water_mark":       3.0,
    "power_marks":      3.0,
    # in-curb / in-road (can be on the opposite lane of a wide road)
    "hand_hole":       15.0,
    "handhole":        15.0,
    "manhole":         20.0,
    "valve":           12.0,
    "fire_hydrant":    20.0,
    # road-shoulder (can be on near OR far shoulder of divided highway)
    "riser":           30.0,
    "pedestal":        30.0,
    "splice_enclosure":30.0,
    "u_guard":         30.0,
    "telecom":         40.0,
    "cabinet":         40.0,
    # off-shoulder / easement items — far side of wide divided highways
    # can put these 30–65 m off the walker. Ceiling accommodates the
    # stretched triangulation (factor 1.25) for objects at 50 m real.
    "pole":            80.0,
    "street_light":    80.0,
    "culvert":         80.0,
    "transformer":     80.0,
    # set back (trees can be anywhere in yards, lots, medians)
    "tree":            80.0,
}
# Default max for unknown classes: pick a permissive value. The clamp
# is a sanity ceiling to reject obviously-bogus triangulation (500 m,
# 1 km), NOT a tight bound that forces pins close to the walker.
DEDUP_DEFAULT_MAX_DISTANCE_M = 50.0
DEDUP_MIN_OBJ_DISTANCE_M = 0.5
# Single-frame clusters get a harder cap — one detection has no
# cross-check against the rest of the pass, so we don't trust big mono
# distances from it.
DEDUP_SINGLE_FRAME_MAX_FACTOR = 0.65
# Raw YOLO height-based monocular distance vs. real distance — depends
# heavily on whether the object is partially or fully in frame. Field
# measurements:
#   culvert on far shoulder, seen fully in frame throughout pass:
#     raw mono 71.98 m, real 63.1 m → mono over by 14% (factor 0.88)
#   close-up utility pole, only bottom in frame:
#     factor typically 0.3–0.5 (partial view worst when close)
# For fallback projection (used when triangulation can't run), we
# default to 0.85 which handles the far-object case well; close-object
# single-frame clusters will still be rough.
MONO_SHRINK_FACTOR = 0.85

# Triangulation from 2×2 bearing-ray least-squares consistently
# UNDER-shoots real distance. The undershoot depends on the class /
# viewing geometry:
#
#   * PERPENDICULAR-to-path objects (culvert on far shoulder, seen
#     continuously across the pass as the walker moves parallel to
#     it): rays are nearly coplanar, convergence is clean. Field data:
#     raw tri 50.6 m, real 66.4 m → stretch 1.30.
#
#   * PASSING-OBJECTS (pole / street_light / tree the walker walks
#     past): rays from BEFORE-passing converge toward the object;
#     rays from AFTER-passing converge BEHIND the object, pulling the
#     least-squares solution closer to the camera. Field data: pole
#     raw tri ~31 m, real ~48 m → stretch 1.55.
#
# Cause in both: YOLO bbox center biases toward the visible edge of
# a partially-visible object, rotating each ray slightly toward the
# camera side. Passing-objects compound this with the before/after
# geometry problem.
TRI_STRETCH_BY_CLASS = {
    # Passing-objects (vertical assets close to walker path) — larger stretch
    "pole":         1.55,
    "street_light": 1.55,
    "tree":         1.55,
    "transformer":  1.55,
    "riser":        1.55,
    "u_guard":      1.55,
    # Perpendicular-view / across-road assets — modest stretch
    "culvert":      1.30,
    "cabinet":      1.30,
    "telecom":      1.30,
    "pedestal":     1.30,
    "hand_hole":    1.30,
    "manhole":      1.30,
    "fire_hydrant": 1.30,
    "valve":        1.30,
    # Flat / ground markings fall back to default (rarely triangulated)
}
TRI_STRETCH_DEFAULT = 1.30

# Systematic bearing bias observed in ground-truth calibration:
# pipeline projected bearings run ~2–3° counterclockwise of truth
# (culvert +2.9°, streetlight +1.6°, mean +2.24°). Applied CLOCKWISE
# to triangulated bearing (and fallback bearing) before projecting
# the point — i.e. effective bearing = raw + offset.
#
# 2026-04-21 UPDATE: set to -2.5 based on SSE diagnostics from the
# run with job eee27029. Every cluster was still rejected from
# triangulation (100% fallback path), and every fallback pin sat
# systematically east of ground truth:
#   culvert:         4.6 m east of GT
#   pole (22-ray):  14.9 m east of GT
#   pole (59-ray):  40.8 m east of GT
# The east-ward drift scales with distance, which is the signature
# of a constant bearing bias in the fallback projection (d·sin(Δθ)).
# A -2.5° offset rotates every projection CCW (toward NW), pulling
# the pins off the east edge of the track and back onto the GT. This
# applies equally to the triangulation path if/when it starts
# succeeding, since that also rotates the final projected coord.
CALIBRATION_BEARING_OFFSET_DEG = -2.5

# Runtime diagnostic buffer. _triangulate_asset / _dedup_merge_cluster
# push one line per triangulation so we can prove from the API response
# (and a calibration.log in the job folder) which stretch + bearing
# offset the live process actually used. Cleared at job start.
CAL_LOG_LINES: list[str] = []

# ── Triangulation acceptance thresholds ─────────────────────────────
# All gates that can reject a cluster from the least-squares bearing
# intersection. Each one exists because a specific failure mode of
# walk-toward drive-by video makes that kind of solve untrustworthy.
# Lifted out of _triangulate_asset so they're one place to tune.
#
# TRI_MIN_BASELINE_M:  cluster's cameras must span at least this many
#   meters end-to-end, otherwise the rays are almost parallel and
#   tiny bearing noise swings the solved point wildly.
# TRI_MIN_SPREAD_DEG:  widest angular gap between any two rays in the
#   cluster must exceed this. Forward-facing walk-by footage often has
#   20 rays all pointing within 5° of the walk direction — the solve
#   is then mathematically ill-conditioned, and in practice the
#   solution collapses onto the camera track in a meaningless direction.
# TRI_MAX_RESIDUAL_M:  mean perpendicular miss-distance from solved
#   point to the rays. Above ~3 m the rays didn't really intersect;
#   bbox-center noise is dominating the geometry.
# TRI_MAX_FARTHEST_M:  rejects absurd extrapolations (rays barely
#   converging and placing the point 200 m out).
# TRI_MIN_CLOSEST_M:   the degenerate "collapse onto track" case —
#   if the solved point is within this many meters of at least one
#   camera position, it almost certainly snapped onto the camera path.
TRI_MIN_BASELINE_M  = 5.0
TRI_MIN_SPREAD_DEG  = 20.0
TRI_MAX_RESIDUAL_M  = 3.0
TRI_MAX_FARTHEST_M  = 80.0
TRI_MIN_CLOSEST_M   = 2.0


def _empty_tri_diag(cluster: list, reject_reason: str,
                    n_rays: int = 0) -> dict:
    """Helper to build a skeleton diagnostic for a cluster we couldn't
    even start solving (e.g. too few bearings). Keeps the debug CSV
    well-formed so every cluster shows up in it."""
    _cls = (cluster[0].get("class", "?") if cluster else "?")
    # Pick the first two cluster members (if any) as the pair to report
    # so the debug CSV has SOMETHING to show for the rejected case.
    a = cluster[0] if len(cluster) >= 1 else {}
    b = cluster[1] if len(cluster) >= 2 else {}
    return {
        "status":              "rejected",
        "reject_reason":       reject_reason,
        "n_rays":              n_rays,
        "n_cluster":           len(cluster),
        "baseline_m":          None,
        "max_spread_deg":      None,
        "residual_m":          None,
        "closest_dist_m":      None,
        "farthest_dist_m":     None,
        "raw_lat":             None,
        "raw_lon":             None,
        "cls":                 _cls,
        "frame_a":             a.get("frame_id"),
        "frame_b":             b.get("frame_id"),
        "cam_a_lat":           a.get("cam_lat"),
        "cam_a_lng":           a.get("cam_lng"),
        "cam_a_bearing":       a.get("cam_bearing"),
        "obj_bearing_a":       a.get("obj_bearing"),
        "cam_b_lat":           b.get("cam_lat"),
        "cam_b_lng":           b.get("cam_lng"),
        "cam_b_bearing":       b.get("cam_bearing"),
        "obj_bearing_b":       b.get("obj_bearing"),
    }


def _triangulate_asset(cluster: list) -> dict:
    """Least-squares triangulation of an asset from multi-frame bearings.

    Geometry: each frame gives a RAY from camera GPS (cam_lat, cam_lng)
    in the direction of obj_bearing (compass, 0=N, 90=E). Intersection
    of those rays = real asset position. Multi-view drive-by beats any
    single-frame monocular depth estimate because the camera moves
    through known GPS positions while the asset stays fixed.

    Math: minimize Σ ((Q - p_i) · n_i)² where n_i is the unit normal
    to ray_i. Closed form 2x2 linear system solved for Q.

    Returns a dict with ALL diagnostics — baseline, spread, residual,
    closest/farthest-camera distance, raw solved lat/lon (even when a
    gate rejects the solve, so the debug CSV can show "what it would
    have placed"), the two most angularly-divergent rays, and a
    status/reject_reason pair so the caller can decide whether to
    trust the solve.

    Every cluster that enters this function produces a diagnostic row
    — if we can't even get two bearings out, the row shows
    reject_reason='too_few_bearings' with the rest empty.
    """
    import math as _m
    _cls = (cluster[0].get("class", "?") if cluster else "?")
    _n_total = len(cluster)
    pts = []
    # Keep parallel arrays of the source cluster entries so we can
    # recover which two frames constitute the most divergent ray pair
    # (for the debug CSV's "frame_a / frame_b" columns).
    src = []
    for p in cluster:
        cl = p.get("cam_lat")
        co = p.get("cam_lng")
        ob = p.get("obj_bearing")
        if cl is None or co is None or ob is None:
            continue
        pts.append((float(cl), float(co), float(ob)))
        src.append(p)
    if len(pts) < 2:
        CAL_LOG_LINES.append(
            f"[TRI_REJECT] cls={_cls} reason=too_few_bearings "
            f"cluster_n={_n_total} with_bearing={len(pts)}"
        )
        return _empty_tri_diag(cluster, "too_few_bearings", n_rays=len(pts))

    # Local equirectangular frame around the cluster centroid (meters,
    # east/north). Good enough for a 100 m cluster; ~0.001 % error.
    clat_mean = sum(p[0] for p in pts) / len(pts)
    clon_mean = sum(p[1] for p in pts) / len(pts)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * _m.cos(_m.radians(clat_mean))

    xs, ys, thetas = [], [], []
    for lat, lon, brg in pts:
        x = (lon - clon_mean) * m_per_deg_lon
        y = (lat - clat_mean) * m_per_deg_lat
        xs.append(x); ys.append(y)
        thetas.append(_m.radians(brg))

    # Baseline check — if cameras barely moved, triangulation is unstable.
    baseline = max(
        _m.hypot(xs[i] - xs[j], ys[i] - ys[j])
        for i in range(len(pts)) for j in range(i + 1, len(pts))
    )

    # Find the most angularly-divergent pair of rays (by index into
    # pts/src). Drives both the spread gate and the debug CSV's
    # frame_a/frame_b pick. Done before any solve so we can report it
    # even in rejection rows.
    # We compare bearings modulo 360 with wrap-around; the pair with
    # the widest circular gap wins.
    max_spread = 0.0
    idx_a = 0
    idx_b = 1 if len(pts) > 1 else 0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            di = abs((_m.degrees(thetas[i]) % 360.0)
                     - (_m.degrees(thetas[j]) % 360.0))
            if di > 180:
                di = 360 - di
            if di > max_spread:
                max_spread = di
                idx_a, idx_b = i, j
    pa = src[idx_a]; pb = src[idx_b]

    def _pack(status, reject_reason,
              raw_lat=None, raw_lon=None, residual=None,
              closest_dist=None, farthest=None):
        return {
            "status":          status,
            "reject_reason":   reject_reason,
            "n_rays":          len(pts),
            "n_cluster":       _n_total,
            "baseline_m":      round(baseline, 2),
            "max_spread_deg":  round(max_spread, 2),
            "residual_m":      (round(residual, 2)
                                if residual is not None else None),
            "closest_dist_m":  (round(closest_dist, 2)
                                if closest_dist is not None else None),
            "farthest_dist_m": (round(farthest, 2)
                                if farthest is not None else None),
            "raw_lat":         (round(raw_lat, 7)
                                if raw_lat is not None else None),
            "raw_lon":         (round(raw_lon, 7)
                                if raw_lon is not None else None),
            "cls":             _cls,
            "frame_a":         pa.get("frame_id"),
            "frame_b":         pb.get("frame_id"),
            "cam_a_lat":       pa.get("cam_lat"),
            "cam_a_lng":       pa.get("cam_lng"),
            "cam_a_bearing":   pa.get("cam_bearing"),
            "obj_bearing_a":   pa.get("obj_bearing"),
            "cam_b_lat":       pb.get("cam_lat"),
            "cam_b_lng":       pb.get("cam_lng"),
            "cam_b_bearing":   pb.get("cam_bearing"),
            "obj_bearing_b":   pb.get("obj_bearing"),
        }

    # Attempt the 2×2 solve up-front — BEFORE any rejection gates.
    # Historically baseline_too_short / bearing_spread_too_small bailed
    # here, leaving the debug CSV's solved_lat/lon columns blank for
    # those clusters. That hides the most useful diagnostic — "where
    # would the math have put this asset?" — which is exactly what we
    # need to know when deciding whether to loosen a gate. So: run the
    # solve whenever the normal matrix isn't actually singular, and let
    # the gates reject with fully-populated diagnostics. Clusters with
    # a legitimately singular matrix (parallel rays, det ≈ 0) still
    # get rejected with empty coords — they genuinely aren't solvable.
    #
    # Build (Σ n n^T) Q = Σ (p·n) n  — normal of a compass-bearing ray
    # pointing direction (sin θ, cos θ) is (cos θ, -sin θ).
    a11 = a12 = a22 = b1 = b2 = 0.0
    for x, y, th in zip(xs, ys, thetas):
        nx = _m.cos(th)
        ny = -_m.sin(th)
        a11 += nx * nx
        a12 += nx * ny
        a22 += ny * ny
        pn = x * nx + y * ny
        b1 += pn * nx
        b2 += pn * ny
    det = a11 * a22 - a12 * a12
    trace = a11 + a22
    singular = abs(det) < 1e-4 * max(trace * trace, 1.0)

    q_lat_pre = q_lon_pre = None
    residual = None
    closest_dist = None
    farthest = None
    if not singular:
        qx = ( a22 * b1 - a12 * b2) / det
        qy = (-a12 * b1 + a11 * b2) / det
        # Residual (mean perpendicular miss-distance in meters).
        residual = 0.0
        for x, y, th in zip(xs, ys, thetas):
            nx = _m.cos(th); ny = -_m.sin(th)
            residual += abs((qx - x) * nx + (qy - y) * ny)
        residual /= len(pts)
        # Closest/farthest to any camera in the cluster — drives the
        # "collapsed onto track" guard and the "runaway extrapolation"
        # guard. Always computed so the debug CSV shows them even for
        # gate-rejected solves.
        closest_dist = min(_m.hypot(qx - x, qy - y) for x, y in zip(xs, ys))
        farthest = max(_m.hypot(qx - x, qy - y) for x, y in zip(xs, ys))
        q_lat_pre = clat_mean + qy / m_per_deg_lat
        q_lon_pre = clon_mean + qx / m_per_deg_lon

    # Baseline gate — cluster's cameras must span enough real-world
    # distance that tiny bearing noise doesn't swing the solved point
    # wildly. Still returns the raw solve so the debug CSV can show
    # where the math would have placed the asset.
    if baseline < TRI_MIN_BASELINE_M:
        CAL_LOG_LINES.append(
            f"[TRI_REJECT] cls={_cls} reason=baseline_too_short "
            f"baseline={baseline:.2f}m cluster_n={_n_total} rays={len(pts)}"
        )
        return _pack("rejected", "baseline_too_short",
                     raw_lat=q_lat_pre, raw_lon=q_lon_pre,
                     residual=residual,
                     closest_dist=closest_dist, farthest=farthest)

    # Bearing-spread check — the real killer for forward-facing walk-by
    # geometry. If every ray points within ±10° of the same direction,
    # the 2×2 normal matrix is solvable but massively ill-conditioned
    # and the solution snaps onto the camera track with no relation to
    # the asset. Still returns the raw solve so the debug CSV can show
    # the (untrustworthy) coordinates the math produced.
    if max_spread < TRI_MIN_SPREAD_DEG:
        CAL_LOG_LINES.append(
            f"[TRI_REJECT] cls={_cls} reason=bearing_spread_too_small "
            f"spread={max_spread:.1f}deg baseline={baseline:.1f}m rays={len(pts)}"
        )
        return _pack("rejected", "bearing_spread_too_small",
                     raw_lat=q_lat_pre, raw_lon=q_lon_pre,
                     residual=residual,
                     closest_dist=closest_dist, farthest=farthest)

    # Singular matrix — parallel rays. Genuinely unsolvable; no raw
    # coord to report.
    if singular:
        CAL_LOG_LINES.append(
            f"[TRI_REJECT] cls={_cls} reason=singular_matrix "
            f"det={det:.2e} trace={trace:.2e} rays={len(pts)}"
        )
        return _pack("rejected", "singular_matrix")

    # Rays must actually converge — reject solutions where the mean
    # perpendicular miss is worse than TRI_MAX_RESIDUAL_M. That's slop
    # caused by bbox-center noise, not an actual intersection.
    if residual > TRI_MAX_RESIDUAL_M:
        CAL_LOG_LINES.append(
            f"[TRI_REJECT] cls={_cls} reason=residual_too_big "
            f"residual={residual:.2f}m rays={len(pts)} spread={max_spread:.1f}deg"
        )
        return _pack("rejected", "residual_too_big",
                     raw_lat=q_lat_pre, raw_lon=q_lon_pre,
                     residual=residual,
                     closest_dist=closest_dist, farthest=farthest)

    if farthest > TRI_MAX_FARTHEST_M:
        CAL_LOG_LINES.append(
            f"[TRI_REJECT] cls={_cls} reason=too_far_from_cams "
            f"farthest={farthest:.1f}m closest={closest_dist:.1f}m rays={len(pts)}"
        )
        return _pack("rejected", "too_far_from_cams",
                     raw_lat=q_lat_pre, raw_lon=q_lon_pre,
                     residual=residual,
                     closest_dist=closest_dist, farthest=farthest)
    if closest_dist < TRI_MIN_CLOSEST_M:
        CAL_LOG_LINES.append(
            f"[TRI_REJECT] cls={_cls} reason=collapsed_onto_track "
            f"closest={closest_dist:.2f}m farthest={farthest:.1f}m rays={len(pts)}"
        )
        return _pack("rejected", "collapsed_onto_track",
                     raw_lat=q_lat_pre, raw_lon=q_lon_pre,
                     residual=residual,
                     closest_dist=closest_dist, farthest=farthest)

    return _pack("ok", None,
                 raw_lat=q_lat_pre, raw_lon=q_lon_pre,
                 residual=residual,
                 closest_dist=closest_dist, farthest=farthest)


def _dedup_merge_cluster(cluster: list) -> dict:
    """Collapse a cluster into one asset record.

    Coordinate source priority:
      1. Multi-frame triangulation (two or more frames w/ bearings):
         intersection of the bearing rays = real asset position. This
         is the accurate path — sub-meter when the camera moves
         through a good baseline and the model's bbox center is clean.
      2. Fallback: camera-at-closest-approach + small per-class
         directional offset (for single-frame clusters or degenerate
         geometry).

    Image + confidence always come from the highest-scoring frame.
    """
    if not cluster:
        return {}
    areas = [float(p.get("bbox_area") or 0) for p in cluster]
    lo, hi = min(areas), max(areas)
    if hi > lo:
        norm_areas = [(a - lo) / (hi - lo) for a in areas]
    else:
        norm_areas = [1.0] * len(cluster)

    closest_idx = max(range(len(cluster)), key=lambda i: areas[i])
    closest = cluster[closest_idx]
    scores = [p["confidence"] + 0.3 * na for p, na in zip(cluster, norm_areas)]
    best_idx = max(range(len(cluster)), key=lambda i: scores[i])
    best = cluster[best_idx]

    # --- Primary: triangulate from multi-frame bearings ---
    tri_diag = _triangulate_asset(cluster)
    tri_lat = tri_lon = None
    tri_residual = None
    if tri_diag.get("status") == "ok":
        tri_lat = tri_diag.get("raw_lat")
        tri_lon = tri_diag.get("raw_lon")
        tri_residual = tri_diag.get("residual_m")

    # --- Fallback: project along obj_bearing using the closest frame ---
    # When triangulation rejects walk-toward geometry, we still know the
    # direction the object is in (obj_bearing from the closest-approach
    # frame, where the bbox is biggest and the monocular distance is
    # the most trustworthy). Project along that bearing by the cluster's
    # MIN monocular distance (the frame where you were physically
    # closest), then sanity-cap by class.
    cls_name = best.get("class", "")

    # Anchor on closest-approach camera (largest bbox frame).
    fb_base_lat = closest.get("cam_lat")
    fb_base_lon = closest.get("cam_lng")
    if fb_base_lat is None or fb_base_lon is None:
        fb_base_lat = closest.get("lat")
        fb_base_lon = closest.get("lon")

    # Bearing from closest-approach frame (biggest bbox = cleanest
    # angular measurement).
    cl_obj_brg = closest.get("obj_bearing")
    cl_cam_brg = closest.get("cam_bearing")

    # Distance: MINIMUM mono across the whole cluster, not just the
    # closest-frame value. When the camera walks past an object the
    # frame with the smallest estimated distance is the moment of true
    # closest approach — that estimate is the most accurate in the
    # pass, because the bbox is at its largest and the object is
    # least foreshortened.
    cluster_dists = [float(p["distance_m"]) for p in cluster
                     if p.get("distance_m") and p["distance_m"] > 0]

    max_d = DEDUP_MAX_OBJ_DISTANCE_M.get(cls_name, DEDUP_DEFAULT_MAX_DISTANCE_M)
    # Single-frame clusters: no cross-check, so we apply a tighter cap.
    if len(cluster) == 1:
        max_d = max_d * DEDUP_SINGLE_FRAME_MAX_FACTOR
    min_d = DEDUP_MIN_OBJ_DISTANCE_M

    if not cluster_dists:
        # No monocular distance anywhere in the cluster — fall back to
        # a small class-appropriate stand-off.
        proj_dist = DEDUP_ROADSIDE_OFFSET_M.get(cls_name, DEDUP_DEFAULT_OFFSET_M)
    else:
        # Use the MINIMUM mono across the pass (closest-approach frame,
        # biggest bbox = highest resolution). Then correct for the
        # systematic ~1.8× overestimation of YOLO height-based mono
        # (see MONO_SHRINK_FACTOR comment). Finally clamp to the
        # class's plausible survey range — this protects against cases
        # where the walker never got close (min mono ~= far distance)
        # AND cases where a single bad bbox gives an absurd value.
        raw_dist = min(cluster_dists)
        proj_dist = raw_dist * MONO_SHRINK_FACTOR
        proj_dist = max(min_d, min(proj_dist, max_d))

    # Choose the bearing to project along. Prefer the actual
    # object-bearing (compass) from the closest frame. If absent, derive
    # one from the camera heading + a side offset (classic roadside
    # guess — object 90° off the driving direction).
    proj_bearing = cl_obj_brg
    if proj_bearing is None and cl_cam_brg is not None:
        # Without obj_bearing we don't know which side of the road; use
        # +90° as a default (most assets in US survey are to the right).
        proj_bearing = (cl_cam_brg + 90.0) % 360.0

    fb_lat, fb_lon = fb_base_lat, fb_base_lon
    offset_bearing = proj_bearing
    offset_m = proj_dist
    if proj_bearing is not None and fb_base_lat is not None and proj_dist > 0:
        import math as _m
        R_EARTH_M = 6_371_008.8
        phi1 = _m.radians(fb_base_lat); lam1 = _m.radians(fb_base_lon)
        sigma = proj_dist / R_EARTH_M
        # Apply systematic bearing calibration offset
        # (see CALIBRATION_BEARING_OFFSET_DEG — pipeline runs ~+2.24°
        # CCW of truth; rotate CW to correct).
        theta = _m.radians(proj_bearing + CALIBRATION_BEARING_OFFSET_DEG)
        phi2 = _m.asin(
            _m.sin(phi1) * _m.cos(sigma)
            + _m.cos(phi1) * _m.sin(sigma) * _m.cos(theta)
        )
        lam2 = lam1 + _m.atan2(
            _m.sin(theta) * _m.sin(sigma) * _m.cos(phi1),
            _m.cos(sigma) - _m.sin(phi1) * _m.sin(phi2),
        )
        fb_lat = _m.degrees(phi2); fb_lon = _m.degrees(lam2)

    # --- Decide the final placement -----------------------------------
    # Field measurements showed triangulation UNDER-shoots real
    # distance by ~20% (real 63 m, tri 50 m). Fallback (mono × 0.55)
    # was even worse (40 m — way under). So the right order is:
    #   1) triangulation, STRETCHED by a class-specific factor
    #      (TRI_STRETCH_BY_CLASS) to compensate for the systematic
    #      bearing-bias undershoot
    #   2) fallback (bearing projection with mono × MONO_SHRINK_FACTOR)
    #   3) bare camera position
    # Triangulation is still clamped to the class max as a sanity check.
    tri_rejected_reason = None
    tri_stretched_lat = tri_stretched_lon = None
    tri_cam_dist_m = None
    if tri_lat is not None and fb_base_lat is not None and fb_base_lon is not None:
        tri_cam_dist_ft = _haversine_ft(fb_base_lat, fb_base_lon,
                                        tri_lat, tri_lon)
        tri_cam_dist_m = tri_cam_dist_ft / 3.28084
        if tri_cam_dist_m > max_d:
            tri_rejected_reason = f"tri_dist_{tri_cam_dist_m:.1f}m>max_{max_d:.1f}m"
            tri_lat = tri_lon = None
            tri_cam_dist_m = None
        elif tri_cam_dist_m < min_d:
            tri_rejected_reason = f"tri_dist_{tri_cam_dist_m:.2f}m<min_{min_d:.1f}m"
            tri_lat = tri_lon = None
            tri_cam_dist_m = None
        else:
            # Stretch the cam → triangulated-point vector by a
            # class-specific factor to compensate for triangulation
            # undershoot. Passing-objects (pole/street_light/tree) need
            # more stretch than perpendicular objects (culvert). See
            # TRI_STRETCH_BY_CLASS comment above. Clamp the stretched
            # distance against max_d so we never exceed the class's
            # plausible setback.
            import math as _ms
            R_EARTH_M = 6_371_008.8
            _stretch = TRI_STRETCH_BY_CLASS.get(cls_name, TRI_STRETCH_DEFAULT)
            stretched_d = min(tri_cam_dist_m * _stretch, max_d)
            # Runtime proof-of-value so we can confirm via the API
            # response / calibration.log that the new TRI_STRETCH_BY_CLASS
            # and CALIBRATION_BEARING_OFFSET_DEG are actually being read
            # by the running process (stale worker vs. stale file is
            # otherwise indistinguishable from output alone).
            CAL_LOG_LINES.append(
                f"[CAL] cls={cls_name} raw_tri={tri_cam_dist_m:.2f}m "
                f"stretch={_stretch:.2f} stretched={stretched_d:.2f}m "
                f"(max={max_d:.1f}m) bearing_off="
                f"{CALIBRATION_BEARING_OFFSET_DEG:+.2f}deg"
            )
            # bearing from camera to triangulated point
            p1 = _ms.radians(fb_base_lat); p2 = _ms.radians(tri_lat)
            dl = _ms.radians(tri_lon - fb_base_lon)
            y = _ms.sin(dl) * _ms.cos(p2)
            x = (_ms.cos(p1) * _ms.sin(p2)
                 - _ms.sin(p1) * _ms.cos(p2) * _ms.cos(dl))
            tri_bearing = _ms.atan2(y, x)
            # Apply systematic bearing calibration offset
            # (pipeline runs ~+2.24° CCW of truth; rotate CW to
            # correct).
            tri_bearing = tri_bearing + _ms.radians(CALIBRATION_BEARING_OFFSET_DEG)
            # project from cam by stretched_d along tri_bearing
            sigma = stretched_d / R_EARTH_M
            lam1 = _ms.radians(fb_base_lon)
            phi2 = _ms.asin(
                _ms.sin(p1) * _ms.cos(sigma)
                + _ms.cos(p1) * _ms.sin(sigma) * _ms.cos(tri_bearing)
            )
            lam2 = lam1 + _ms.atan2(
                _ms.sin(tri_bearing) * _ms.sin(sigma) * _ms.cos(p1),
                _ms.cos(sigma) - _ms.sin(p1) * _ms.sin(phi2),
            )
            tri_stretched_lat = _ms.degrees(phi2)
            tri_stretched_lon = _ms.degrees(lam2)

    fallback_projected = (
        proj_bearing is not None
        and fb_base_lat is not None
        and proj_dist is not None
        and proj_dist > 0
        and (fb_lat != fb_base_lat or fb_lon != fb_base_lon)
    )

    if tri_stretched_lat is not None:
        coord_lat, coord_lon = tri_stretched_lat, tri_stretched_lon
        coord_src = "triangulated"
    elif fallback_projected:
        coord_lat, coord_lon, coord_src = fb_lat, fb_lon, "cam+offset"
    else:
        # Neither projection could run — pin on camera position.
        coord_lat = fb_base_lat if fb_base_lat is not None else best.get("lat")
        coord_lon = fb_base_lon if fb_base_lon is not None else best.get("lon")
        coord_src = "cam_only"

    merged = dict(best)
    merged["lat"] = round(float(coord_lat), 7)
    merged["lon"] = round(float(coord_lon), 7)
    merged["num_detections_merged"] = len(cluster)
    merged["members_video_sec"]     = sorted({p["timestamp"] for p in cluster})
    merged["coords_from_frame"]     = closest.get("frame_id")
    merged["image_from_frame"]      = best.get("frame_id")
    merged["coord_source"]          = coord_src
    merged["triangulation_residual_m"] = (round(tri_residual, 2)
                                          if tri_residual is not None else None)
    merged["projected_lat"]         = closest.get("lat")
    merged["projected_lon"]         = closest.get("lon")
    merged["offset_m"]              = offset_m
    merged["offset_bearing"]        = (round(offset_bearing, 1)
                                       if offset_bearing is not None else None)
    if tri_rejected_reason:
        merged["triangulation_rejected"] = tri_rejected_reason

    # ── Attach full triangulation diagnostics to the merged asset so
    # the downstream CSV writers can render per-cluster columns (mean
    # confidence, baseline_m, ray_angle_deg, residual, source_frames,
    # raw-vs-final coords). We also persist the ENTIRE diag dict under
    # a single key so the debug CSV can pull frame_a/frame_b and the
    # two cameras' poses without re-solving.
    try:
        cluster_confs = [float(p.get("confidence") or 0.0) for p in cluster]
        mean_conf = (sum(cluster_confs) / len(cluster_confs)
                     if cluster_confs else None)
        source_frames = [p.get("frame_id") for p in cluster
                         if p.get("frame_id") is not None]
    except Exception:
        mean_conf = None
        source_frames = []
    merged["mean_conf"]      = (round(mean_conf, 4)
                                if mean_conf is not None else None)
    merged["source_frames"]  = source_frames
    # Triangulation geometry always present (even for rejected clusters
    # where tri_diag contains baseline/spread = None).
    merged["tri_baseline_m"]        = tri_diag.get("baseline_m")
    merged["tri_spread_deg"]        = tri_diag.get("max_spread_deg")
    merged["tri_closest_dist_m"]    = tri_diag.get("closest_dist_m")
    merged["tri_farthest_dist_m"]   = tri_diag.get("farthest_dist_m")
    merged["tri_status"]            = tri_diag.get("status")
    merged["tri_reject_reason"]     = tri_diag.get("reject_reason")
    merged["tri_n_rays"]            = tri_diag.get("n_rays")
    # Raw (pre-stretch, pre-snap) triangulated point. Lets the final
    # CSV compare the solver's actual output vs where the pin landed
    # after stretch/fallback snapping.
    merged["raw_tri_lat"]           = tri_diag.get("raw_lat")
    merged["raw_tri_lon"]           = tri_diag.get("raw_lon")
    # Keep the full diagnostic dict around for the debug CSV writer.
    merged["_tri_diag"]             = tri_diag

    # ── CRITICAL: recompute distance_m / obj_bearing from the FINAL
    # placed pin position. `merged = dict(best)` above copied the
    # per-frame monocular distance estimate (and the per-frame object
    # bearing) from the best-scoring frame. Those values have no
    # relationship to where the pin ACTUALLY landed — so the KMZ
    # popup's "Distance from camera: 71.98 m" would contradict the
    # pin's real 36 m offset from the displayed camera position.
    # Recompute both from (best.cam_lat, best.cam_lng) → (coord_lat,
    # coord_lon) so the popup numbers match the pin you see on the
    # map.
    try:
        _disp_cam_lat = best.get("cam_lat")
        _disp_cam_lng = best.get("cam_lng")
        if (_disp_cam_lat is not None and _disp_cam_lng is not None
                and coord_lat is not None and coord_lon is not None):
            import math as _mdist
            R_EARTH_M = 6_371_008.8
            p1 = _mdist.radians(float(_disp_cam_lat))
            p2 = _mdist.radians(float(coord_lat))
            dp = _mdist.radians(float(coord_lat) - float(_disp_cam_lat))
            dl = _mdist.radians(float(coord_lon) - float(_disp_cam_lng))
            a = (_mdist.sin(dp/2) ** 2
                 + _mdist.cos(p1) * _mdist.cos(p2) * _mdist.sin(dl/2) ** 2)
            c = 2 * _mdist.atan2(_mdist.sqrt(a), _mdist.sqrt(1 - a))
            actual_dist_m = R_EARTH_M * c
            # Forward azimuth (compass bearing) from cam → pin.
            y = _mdist.sin(dl) * _mdist.cos(p2)
            x = (_mdist.cos(p1) * _mdist.sin(p2)
                 - _mdist.sin(p1) * _mdist.cos(p2) * _mdist.cos(dl))
            actual_brg = (_mdist.degrees(_mdist.atan2(y, x)) + 360.0) % 360.0
            merged["distance_m"]  = round(actual_dist_m, 2)
            merged["obj_bearing"] = round(actual_brg, 1)
    except Exception:
        # If recompute fails for any reason, leave the per-frame
        # fallback in place — a popup with a stale number is strictly
        # better than a crashed pipeline.
        pass

    # Per-cluster resolution-path log: exactly one line per surviving
    # cluster, showing which branch (tri/fallback/cam_only) placed the
    # pin and why triangulation failed when it did. This is what tells
    # us whether the stretch constant even gets a chance to run on the
    # real data; calibration_summary.lines_count==0 across a whole
    # run means every cluster fell through to fallback.
    try:
        bearings_in_cluster = sum(
            1 for p in cluster if p.get("obj_bearing") is not None
        )
        cam_pts = {
            (round(p.get("cam_lat") or 0, 5),
             round(p.get("cam_lng") or 0, 5))
            for p in cluster
            if p.get("cam_lat") is not None and p.get("cam_lng") is not None
        }
        CAL_LOG_LINES.append(
            f"[PATH] cls={cls_name} src={coord_src} "
            f"cluster_n={len(cluster)} bearings={bearings_in_cluster} "
            f"unique_cam_pos={len(cam_pts)} "
            f"tri_ok={tri_lat is not None} "
            f"tri_reject={tri_rejected_reason or '-'} "
            f"proj_dist={proj_dist:.1f}m proj_brg="
            f"{('%.1f' % proj_bearing) if proj_bearing is not None else '-'}"
        )
    except Exception:
        pass

    return merged


# Minimum frames a detection cluster must include to survive into the
# final output, keyed by class. For classes where YOLO produces lots of
# low-confidence false-positives on unrelated background textures
# (culverts seen where there is none, "poles" in trees, street_lights
# in random shadows), a single-frame detection is almost always noise
# — a real object held by the same camera over multiple frames should
# register 3+ times. For curbside / under-foot items that are genuine
# and often brief (a single hand_hole on a walkway), we still accept
# single-frame detections.
DEDUP_MIN_FRAMES = {
    # Long-range off-shoulder classes — require 3+ frames.
    # Ground-truth calibration showed 2-frame clusters (very short
    # camera baseline) triangulate catastrophically at range — e.g.
    # a real 233 ft streetlight pinned at 119 ft (51% short). 3+
    # frames gives enough baseline + redundancy for survey-grade
    # pins. Single and 2-frame clusters for these classes are
    # dropped rather than pinned inaccurately.
    "pole":         3,
    "culvert":      3,
    "street_light": 3,
    "tree":         3,
    "cabinet":      3,
    "telecom":      3,
    "transformer":  3,
    # curbside/under-foot — single-frame OK
    # (default below handles these)
}
# Any class not listed: accept single-frame detections.
DEDUP_MIN_FRAMES_DEFAULT = 1

# Cross-class confusion pairs — when clusters from two visually-similar
# classes overlap in space AND time, YOLO is almost certainly seeing
# the SAME physical object and classifying some frames one way and
# some the other. Keep the higher-confidence cluster, drop the other.
# Distance threshold in feet (spatial overlap), time threshold in
# seconds (temporal overlap of cluster frames).
CROSS_CLASS_CONFUSION_PAIRS = [
    ("pole", "tree"),              # dark vertical trunks look like poles
    ("pole", "street_light"),      # many street lights are pole-mounted
    ("culvert", "sidewalk"),       # concrete edge looks like sidewalk
    ("pedestal", "cabinet"),       # both are short boxes
    ("riser", "pedestal"),         # both are short tubes/boxes
    ("hand_hole", "manhole"),      # both are round covers in ground
]
# Spatial threshold: pole/tree clusters in the sample data land 35–40 ft
# apart even when they're the same physical object — triangulation + bearing
# noise drifts each cluster toward its own "consensus" point. 50 ft catches
# this while still rejecting genuinely separate objects one house-width apart.
CROSS_CLASS_DISTANCE_FT = 50.0
CROSS_CLASS_TIME_SEC    = 8.0


def _cross_class_dedup(merged: list[dict]) -> list[dict]:
    """Drop the lower-confidence cluster when two merged assets of
    visually-similar classes land within CROSS_CLASS_DISTANCE_FT of
    each other AND their frame-timestamps overlap within
    CROSS_CLASS_TIME_SEC. Handles the pole/tree, pole/street_light,
    etc. mis-classification that happens in the YOLO step."""
    confusion_map: dict = {}
    for a, b in CROSS_CLASS_CONFUSION_PAIRS:
        confusion_map.setdefault(a, set()).add(b)
        confusion_map.setdefault(b, set()).add(a)

    # Mark each cluster by its best video_sec (for time comparison).
    def _time_range(c):
        ts = c.get("members_video_sec") or [c.get("timestamp")]
        ts = [t for t in ts if t is not None]
        return (min(ts), max(ts)) if ts else (0.0, 0.0)

    drop = set()
    n = len(merged)
    for i in range(n):
        if i in drop:
            continue
        ci = merged[i]
        cls_i = ci.get("class", "")
        partners = confusion_map.get(cls_i, set())
        if not partners:
            continue
        ti_min, ti_max = _time_range(ci)
        for j in range(n):
            if j == i or j in drop:
                continue
            cj = merged[j]
            cls_j = cj.get("class", "")
            if cls_j not in partners:
                continue
            # Spatial overlap
            d_ft = _haversine_ft(ci["lat"], ci["lon"], cj["lat"], cj["lon"])
            if d_ft > CROSS_CLASS_DISTANCE_FT:
                continue
            # Temporal overlap
            tj_min, tj_max = _time_range(cj)
            if not (ti_min - CROSS_CLASS_TIME_SEC <= tj_max
                    and tj_min - CROSS_CLASS_TIME_SEC <= ti_max):
                continue
            # Keep whichever has higher confidence (tiebreak: more frames)
            ci_score = (float(ci.get("confidence", 0)),
                        ci.get("num_detections_merged", 1))
            cj_score = (float(cj.get("confidence", 0)),
                        cj.get("num_detections_merged", 1))
            if cj_score > ci_score:
                drop.add(i)
                break
            else:
                drop.add(j)
    return [c for k, c in enumerate(merged) if k not in drop]


def dedup_per_frame_detections(entries: list[dict]) -> list[dict]:
    """Group entries by class, cluster each group, return collapsed assets.

    Drops noise clusters that don't meet DEDUP_MIN_FRAMES for the class,
    then runs cross-class confusion dedup (pole/tree etc.).
    """
    from collections import defaultdict
    by_cls: dict = defaultdict(list)
    for e in entries:
        by_cls[e["class"]].append(e)
    out = []
    for cls, pts in by_cls.items():
        eps_ft = DEDUP_DISTANCE_THRESHOLDS_FT.get(cls, DEDUP_DEFAULT_FT)
        min_frames = DEDUP_MIN_FRAMES.get(cls, DEDUP_MIN_FRAMES_DEFAULT)
        for cl in _dedup_cluster(pts, eps_ft):
            if len(cl) < min_frames:
                continue  # drop single-frame noise for this class
            out.append(_dedup_merge_cluster(cl))
    return _cross_class_dedup(out)


# ── GPS-overlay bearing parsing ──────────────────────────────
# Matches strings like "Azimuth/Bearing: 339° N21W 6027mils (True)"
# or "Bearing: 339" or "Heading 339°". Returns bearing in degrees 0..360
# where 0=N, 90=E, 180=S, 270=W (navigational / compass convention).
_BEARING_PATTERNS = [
    re.compile(r"(?:azimuth|bearing|heading)[^\d\-+]*([0-3]?\d{1,2}(?:\.\d+)?)", re.IGNORECASE),
]


def parse_bearing_deg(text: str):
    """Extract a compass bearing (deg true) from OCR text; None if absent."""
    if not text:
        return None
    for pat in _BEARING_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        try:
            deg = float(m.group(1))
        except ValueError:
            continue
        if 0 <= deg <= 360:
            return deg % 360.0
    return None


def estimate_distance_m(class_name: str, box_h_px: float, image_h_px: float,
                        vfov_deg: float = CAMERA_FOV_V_DEG):
    """Pinhole-camera known-height distance:
          d ≈ (H_real * image_h_px) / (2 * box_h_px * tan(vfov/2))
    Returns distance in meters or None if we don't know the class's height.
    """
    H = CLASS_HEIGHTS_M.get(class_name)
    if not H or not box_h_px or box_h_px <= 0 or image_h_px <= 0:
        return None
    import math as _m
    return (H * image_h_px) / (2.0 * box_h_px * _m.tan(_m.radians(vfov_deg / 2.0)))


def offset_latlon(lat: float, lon: float, bearing_deg: float, distance_m: float):
    """Great-circle destination point (haversine forward formula).
    bearing 0=N, 90=E. Returns (lat, lon) in degrees.
    """
    import math as _m
    R = 6_371_000.0
    br = _m.radians(bearing_deg)
    d_over_R = distance_m / R
    phi1 = _m.radians(lat)
    lam1 = _m.radians(lon)
    phi2 = _m.asin(_m.sin(phi1) * _m.cos(d_over_R)
                   + _m.cos(phi1) * _m.sin(d_over_R) * _m.cos(br))
    lam2 = lam1 + _m.atan2(_m.sin(br) * _m.sin(d_over_R) * _m.cos(phi1),
                            _m.cos(d_over_R) - _m.sin(phi1) * _m.sin(phi2))
    return _m.degrees(phi2), _m.degrees(lam2)


def project_detection_latlon(
    cam_lat: float, cam_lon: float, cam_bearing_deg: float,
    box_cx_norm: float, box_h_norm: float,
    class_name: str,
    image_w_px: int, image_h_px: int,
    hfov_deg: float = CAMERA_FOV_H_DEG, vfov_deg: float = CAMERA_FOV_V_DEG,
):
    """Given camera pose + YOLO box (normalized), return (lat, lon, distance_m,
    obj_bearing_deg). If we can't estimate distance, returns (cam_lat, cam_lon,
    None, None) so the point still plots at the camera's position.
    """
    box_h_px = box_h_norm * image_h_px
    distance_m = estimate_distance_m(class_name, box_h_px, image_h_px, vfov_deg)

    # Horizontal angular offset from image center (left = negative, right = positive)
    # box_cx_norm is in [0,1]; 0.5 means center column.
    angular_offset = (box_cx_norm - 0.5) * hfov_deg
    obj_bearing = (cam_bearing_deg + angular_offset) % 360.0

    if distance_m is None or distance_m <= 0 or distance_m > 200:
        # Unknown class or implausible distance — keep the camera's coords.
        return cam_lat, cam_lon, None, obj_bearing

    obj_lat, obj_lon = offset_latlon(cam_lat, cam_lon, obj_bearing, distance_m)
    return obj_lat, obj_lon, distance_m, obj_bearing


def parse_gpx_track(gpx_path: Path) -> list:
    """Parse a GPX file and return a list of (datetime, lat, lng, ele) tuples."""
    with open(gpx_path) as f:
        gpx = gpxpy.parse(f)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                if pt.time:
                    t = pt.time.replace(tzinfo=timezone.utc) if pt.time.tzinfo is None else pt.time
                    points.append((t, pt.latitude, pt.longitude, pt.elevation or 0.0))
    # Also check waypoints
    for wpt in gpx.waypoints:
        if wpt.time:
            t = wpt.time.replace(tzinfo=timezone.utc) if wpt.time.tzinfo is None else wpt.time
            points.append((t, wpt.latitude, wpt.longitude, wpt.elevation or 0.0))
    points.sort(key=lambda x: x[0])
    return points


def interpolate_gps(gpx_points: list, query_time: datetime):
    """
    Linearly interpolate GPS position at query_time between GPX track points.
    Extrapolates up to 60 s beyond track ends using last velocity vector.
    Returns (lat, lng, ele) or None if too far out of range.
    """
    if not gpx_points:
        return None

    EXTRAPOLATE_LIMIT = 60   # seconds — extrapolate up to 1 min beyond track ends
    HARD_CUTOFF       = 300  # seconds — give up entirely beyond 5 min

    gpx_start_ts = gpx_points[0][0].timestamp()
    gpx_end_ts   = gpx_points[-1][0].timestamp()
    query_ts     = query_time.timestamp()

    # Hard cutoff — completely out of range
    if query_ts < gpx_start_ts - HARD_CUTOFF or query_ts > gpx_end_ts + HARD_CUTOFF:
        return None

    # Before track start — extrapolate backward using first velocity vector
    if query_ts < gpx_start_ts:
        overshoot = gpx_start_ts - query_ts
        if overshoot > EXTRAPOLATE_LIMIT or len(gpx_points) < 2:
            return gpx_points[0][1], gpx_points[0][2], gpx_points[0][3]
        t0, lat0, lng0, ele0 = gpx_points[0]
        t1, lat1, lng1, ele1 = gpx_points[1]
        span = max((t1 - t0).total_seconds(), 0.001)
        ratio = -overshoot / span
        return (
            lat0 + ratio * (lat1 - lat0),
            lng0 + ratio * (lng1 - lng0),
            ele0 + ratio * (ele1 - ele0),
        )

    # After track end — extrapolate forward using last velocity vector
    if query_ts > gpx_end_ts:
        overshoot = query_ts - gpx_end_ts
        if overshoot > EXTRAPOLATE_LIMIT or len(gpx_points) < 2:
            return gpx_points[-1][1], gpx_points[-1][2], gpx_points[-1][3]
        t0, lat0, lng0, ele0 = gpx_points[-2]
        t1, lat1, lng1, ele1 = gpx_points[-1]
        span = max((t1 - t0).total_seconds(), 0.001)
        ratio = overshoot / span
        return (
            lat1 + ratio * (lat1 - lat0),
            lng1 + ratio * (lng1 - lng0),
            ele1 + ratio * (ele1 - ele0),
        )

    # Within track — normal linear interpolation
    for i in range(len(gpx_points) - 1):
        t0, lat0, lng0, ele0 = gpx_points[i]
        t1, lat1, lng1, ele1 = gpx_points[i + 1]
        if t0 <= query_time <= t1:
            span = (t1 - t0).total_seconds()
            ratio = (query_time - t0).total_seconds() / span if span > 0 else 0
            return (
                lat0 + ratio * (lat1 - lat0),
                lng0 + ratio * (lng1 - lng0),
                ele0 + ratio * (ele1 - ele0),
            )
    return None


def get_video_rotation(video_path: Path) -> int:
    """Detect video rotation angle from metadata (0, 90, 180, 270)."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", str(video_path)
        ], capture_output=True, text=True)
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                tags = stream.get("tags", {})
                rotate = tags.get("rotate", stream.get("rotate", "0"))
                return int(rotate)
    except Exception:
        pass
    return 0


def apply_rotation(frame, rotation: int):
    """Rotate a frame to correct for video rotation metadata."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270 or rotation == -90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def get_video_start_time(video_path: Path) -> datetime | None:
    """Try to extract video creation timestamp from metadata via ffprobe."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", str(video_path)
        ], capture_output=True, text=True)
        meta = json.loads(result.stdout)
        tags = meta.get("format", {}).get("tags", {})
        for key in ("creation_time", "date", "com.apple.quicktime.creationdate"):
            if key in tags:
                raw = tags[key].replace("Z", "+00:00")
                return datetime.fromisoformat(raw)
    except Exception:
        pass
    return None


def _image_exif_datetime(path: Path) -> datetime | None:
    """Read EXIF DateTimeOriginal from a JPEG/PNG. Returns UTC datetime or None.
    Used by /pipeline/run-frames to pull per-frame timestamps from phone photos.
    """
    try:
        from PIL import Image, ExifTags
        with Image.open(path) as im:
            exif = im.getexif() or {}
            tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            dt = tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")
            if dt:
                # EXIF format: "YYYY:MM:DD HH:MM:SS"
                return datetime.strptime(dt, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return None


def _run_pipeline(job_id: str, video_path: Path | None, gpx_path: Path,
                  frame_interval: float, conf: float, video_start_override: str | None,
                  use_ocr: bool = True, imgsz: int = 1280):
    """Full pipeline: extract frames → GPS interpolation → YOLO → GeoJSON.

    If `video_path` is None, frames are assumed to already be in
    `job_dir/frames/` (the `/pipeline/run-frames` path unpacks a zip there
    before calling this).
    """
    job_dir = PIPELINE_PATH / job_id
    frames_dir = job_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Reset the runtime calibration log so each job's lines are its own.
    # _triangulate_asset appends one [CAL] line per triangulated cluster.
    CAL_LOG_LINES.clear()

    def update(status, message, progress=None):
        pipeline_jobs[job_id].update({
            "status": status,
            "message": message,
            **({"progress": progress} if progress is not None else {}),
        })

    try:
        # ── Step 1: Parse GPX (optional) ─────────────────────
        update("running", "Parsing GPS source...", 5)
        gpx_points = []
        if gpx_path and Path(gpx_path).exists():
            gpx_points = parse_gpx_track(gpx_path)

        # ── Step 2: Get video start time ─────────────────────
        frames_mode = video_path is None
        if frames_mode:
            update("running", "Frames mode — using uploaded frame directory...", 10)
            fps = 1.0 / max(frame_interval, 0.001)   # used only to synthesize frame_num
            # If override given, use as anchor; otherwise EXIF (per-frame) will
            # be preferred below, falling back to "now" for sequence ordering.
            if video_start_override:
                video_start = datetime.fromisoformat(video_start_override).replace(tzinfo=timezone.utc)
            else:
                video_start = datetime.now(timezone.utc)
        else:
            update("running", "Reading video metadata...", 10)
            if video_start_override:
                video_start = datetime.fromisoformat(video_start_override).replace(tzinfo=timezone.utc)
            else:
                video_start = get_video_start_time(video_path)
                if not video_start:
                    # Fall back: assume video started at GPX start
                    video_start = gpx_points[0][0] if gpx_points else datetime.now(timezone.utc)

            # ── Step 3: Extract frames via ffmpeg (auto-applies rotation) ──
            update("running", "Extracting frames from video...", 15)

            # Get FPS from ffprobe for timestamp calculation
            fps_result = subprocess.run([
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", str(video_path)
            ], capture_output=True, text=True)
            fps = 30.0
            try:
                for s in json.loads(fps_result.stdout).get("streams", []):
                    if s.get("codec_type") == "video":
                        r = s.get("r_frame_rate", "30/1").split("/")
                        fps = float(r[0]) / float(r[1]) if len(r) == 2 else 30.0
                        break
            except Exception:
                pass

            # ffmpeg extracts frames at the requested interval AND auto-rotates
            ffmpeg_cmd = [
                "ffmpeg", "-i", str(video_path),
                "-vf", f"fps=1/{frame_interval}",
                "-q:v", "2",
                str(frames_dir / "frame_%06d.jpg"),
                "-y"
            ]
            subprocess.run(ffmpeg_cmd, capture_output=True)

        # Accept any common image extension (covers both video-extracted
        # `frame_000001.jpg` and user-uploaded `IMG_1234.jpg` / `.png`).
        # rglob (not glob) — users sometimes zip a folder that contains
        # a nested directory, e.g. `test_pics.zip` → `frames/test_pics/*.jpg`.
        # Recursive search means the pipeline works regardless of whether
        # the zip is flat or has a top-level folder inside.
        frame_files = sorted(
            [*frames_dir.rglob("*.jpg"),
             *frames_dir.rglob("*.jpeg"),
             *frames_dir.rglob("*.JPG"),
             *frames_dir.rglob("*.png"),
             *frames_dir.rglob("*.PNG")]
        )
        extracted  = []
        ocr_hits   = 0
        gpx_hits   = 0
        ocr_misses = []      # list of frame filenames where OCR returned no coords
        ocr_samples = []     # list of {frame, text[:200]} for the first few misses (diag)

        for idx, frame_file in enumerate(frame_files):
            frame      = cv2.imread(str(frame_file))
            if frame is None:
                continue
            frame_num  = idx * max(int(fps * frame_interval), 1)
            offset_sec = round(idx * frame_interval, 2)

            # Prefer EXIF DateTimeOriginal (phone photos / frames-mode),
            # fall back to the synthetic video_start + offset.
            frame_time = (
                _image_exif_datetime(frame_file) if frames_mode else None
            ) or datetime.fromtimestamp(
                video_start.timestamp() + offset_sec, tz=timezone.utc
            )

            # ── GPS source priority ───────────────────────────
            # OCR first. NavCam burns a GPS-accurate lat/lon into the
            # frame at the moment the frame was shot, so there is zero
            # time-alignment risk. GPX is the fallback for frames where
            # OCR missed — useful, but only safe when the video clock
            # matches the GPX clock. We've seen ~60 m drift when the
            # video EXIF start-time is off from the GPS clock, which
            # makes GPX-first visibly worse despite being "higher-res".
            #
            # Until we have auto-calibration of video_start → GPX time
            # (by matching OCR-read positions against the GPX track),
            # OCR wins any frame where it succeeds.
            gps_source = None
            lat = lng = ele = None
            bearing = None
            image_h_px, image_w_px = frame.shape[:2]

            if use_ocr:
                ocr_result = ocr_frame_for_gps(frame)
                if ocr_result:
                    lat        = ocr_result["lat"]
                    lng        = ocr_result["lng"]
                    bearing    = ocr_result.get("bearing")
                    ele        = 0.0
                    gps_source = "ocr"
                    ocr_hits  += 1
                else:
                    ocr_misses.append(frame_file.name)
                    if len(ocr_samples) < 8:
                        try:
                            sample_txt = _vision_read(frame, region="top") or ""
                        except Exception:
                            sample_txt = ""
                        ocr_samples.append({
                            "frame": frame_file.name,
                            "top_text": sample_txt.strip()[:200],
                        })

            if lat is None and gpx_points:
                gps = interpolate_gps(gpx_points, frame_time)
                if gps:
                    lat, lng, ele = gps
                    gps_source    = "gpx"
                    gpx_hits     += 1

            extracted.append({
                "frame_num":   frame_num,
                "offset_sec":  offset_sec,
                "frame_time":  frame_time,
                "frame_file":  frame_file,
                "lat":         lat,
                "lng":         lng,
                "ele":         ele,
                "bearing":     bearing,
                "image_w_px":  image_w_px,
                "image_h_px":  image_h_px,
                "gps_source":  gps_source,
            })

        update("running",
               f"Extracted {len(extracted)} frames — OCR GPS: {ocr_hits}, GPX fallback: {gpx_hits}. "
               f"Running detection...", 40)

        if not extracted:
            source = (
                "uploaded frames archive"
                if video_path is None
                else "video"
            )
            raise ValueError(
                f"No frames could be extracted from {source}. "
                f"frames_dir={frames_dir}, files_found={len(frame_files)}. "
                "If you uploaded a zip, make sure the images are jpg/jpeg/png."
            )

        update("running", f"Extracted {len(extracted)} frames. Running detection...", 40)

        # ── Step 4: YOLO detection + annotated frames ────────
        best_model = get_best_model()
        if not best_model:
            raise ValueError("No trained YOLO model found. Run POST /train first.")

        images_dir = job_dir / "images"   # annotated frames go here for KMZ
        images_dir.mkdir(exist_ok=True)

        features = []
        # Flat list of every kept detection, in the exact schema expected by
        # dedupe_detections.py (Option B hook). Written to detections_raw.json
        # alongside the KMZ so we can run the offline deduper and validate the
        # collapse ratio before wiring dedup into the pipeline itself.
        raw_dedup_entries: list[dict] = []
        # YOLO runs at the LOWEST threshold in CLASS_CONF_OVERRIDE (or the
        # pipeline's `conf`, whichever is smaller) so nothing gets dropped
        # too early. We then raise the bar per-class in Python below.
        yolo_conf = min([conf] + list(CLASS_CONF_OVERRIDE.values()))
        raw_detections_total  = 0   # boxes YOLO returned (pre-filter)
        kept_detections_total = 0   # boxes that survived CLASS_CONF_OVERRIDE
        no_gps_drops          = 0   # kept boxes dropped because their frame has no lat/lng
        for i, frm in enumerate(extracted):
            lat         = frm.get("lat")
            lng         = frm.get("lng")
            ele         = frm.get("ele") or 0.0
            bearing     = frm.get("bearing")
            image_w_px  = frm.get("image_w_px") or 0
            image_h_px  = frm.get("image_h_px") or 0
            gps_source  = frm.get("gps_source", "none")

            # Run YOLO — save=True so we get annotated frames with boxes drawn
            detect_dir = job_dir / "detections" / f"frame_{frm['frame_num']:06d}"
            cmd = [
                "yolo", "detect", "predict",
                f"model={best_model}",
                f"source={frm['frame_file']}",
                f"conf={yolo_conf}",
                f"imgsz={imgsz}",     # iPhone portrait is 1080x1920; default 640 is way too small
                "save=True",          # annotated image with bounding boxes
                "save_txt=True",
                "save_conf=True",
                f"project={detect_dir}",
                "name=out",
            ]
            subprocess.run(cmd, capture_output=True)

            # Parse detection .txt output.
            # YOLO save_txt format (save_conf=True):
            #   class_id  cx_norm  cy_norm  w_norm  h_norm  confidence
            # All coordinates are normalized 0..1 relative to the image.
            txt_file = detect_dir / "out" / "labels" / frm["frame_file"].with_suffix(".txt").name
            detections = []
            # Per-detection crop support. We lazily open the raw frame
            # with PIL only if we end up keeping at least one detection
            # from this frame — crops are written as
            # frame_XXXXXX_det_NN.jpg in images_dir and the per-detection
            # filename is threaded through raw_dedup_entries so the final
            # KMZ placemark's <img src=""> points at a clean crop of ONLY
            # the object this placemark represents, not the full frame.
            from PIL import Image as _PILImage, ImageDraw as _PILDraw
            _frame_pil = None
            kept_idx_in_frame = 0  # NN counter for crop filename
            # Padding around each bbox when cropping. We use TWO rules and
            # take whichever produces the larger crop on each axis:
            #
            #   (1) CROP_PAD is a fraction of bbox-dim on each side,
            #       so a 400px-wide truck with CROP_PAD=0.75 gets a
            #       crop 1000px wide (400 + 300 + 300).
            #   (2) MIN_CROP_FRAC_* is a minimum fraction of the FRAME
            #       on each axis. This matters for tall-narrow objects
            #       like poles, risers, streetlights where rule (1)
            #       alone would still give a skinny crop that hides
            #       all environmental context.
            #
            # With MIN_CROP_FRAC_W=0.45 and MIN_CROP_FRAC_H=0.55, every
            # crop is AT LEAST 45% of frame width and 55% of frame
            # height, so you always see the road, horizon, and adjacent
            # objects — not just a close-up of the thing itself.
            CROP_PAD = 0.75
            MIN_CROP_FRAC_W = 0.45
            MIN_CROP_FRAC_H = 0.55
            if txt_file.exists():
                for line in txt_file.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_id     = int(parts[0])
                        cx_norm    = float(parts[1])
                        cy_norm    = float(parts[2])
                        w_norm     = float(parts[3])
                        h_norm     = float(parts[4])
                        confidence = float(parts[5]) if len(parts) > 5 else 0.0
                    except ValueError:
                        continue

                    raw_detections_total += 1
                    cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                    # ── Per-class confidence filter ───────────────
                    # CLASS_CONF_OVERRIDE lets us raise the bar on over-firing
                    # classes (e.g. pole=0.55) without hurting rare classes.
                    min_conf = CLASS_CONF_OVERRIDE.get(cls_name, conf)
                    if confidence < min_conf:
                        continue
                    kept_detections_total += 1

                    # ── Per-detection crop ─────────────────────────
                    # Save a crop of just this detection so the KMZ
                    # popup shows only the object detected (not the
                    # full frame with every other pin's bbox drawn on
                    # it). Crop is taken from the RAW frame, then a
                    # single green rectangle is drawn around this
                    # detection's bbox so it's obvious where YOLO
                    # triggered.
                    img_filename_per_det = None
                    try:
                        if _frame_pil is None:
                            _frame_pil = _PILImage.open(frm["frame_file"]).convert("RGB")
                        W, H = _frame_pil.size
                        cx_px = cx_norm * W; cy_px = cy_norm * H
                        w_px  = w_norm  * W; h_px  = h_norm  * H
                        # Rule 1: fractional pad around the bbox.
                        pad_w = w_px * CROP_PAD
                        pad_h = h_px * CROP_PAD
                        # Rule 2: floor the total crop size at a
                        # minimum fraction of the frame. If the
                        # fractional-pad rule already exceeds this
                        # floor, it wins — otherwise we expand the
                        # pad outward from the bbox center.
                        min_crop_w = W * MIN_CROP_FRAC_W
                        min_crop_h = H * MIN_CROP_FRAC_H
                        if (w_px + 2 * pad_w) < min_crop_w:
                            pad_w = (min_crop_w - w_px) / 2.0
                        if (h_px + 2 * pad_h) < min_crop_h:
                            pad_h = (min_crop_h - h_px) / 2.0
                        x0 = max(0, int(cx_px - w_px/2 - pad_w))
                        y0 = max(0, int(cy_px - h_px/2 - pad_h))
                        x1 = min(W, int(cx_px + w_px/2 + pad_w))
                        y1 = min(H, int(cy_px + h_px/2 + pad_h))
                        if x1 > x0 and y1 > y0:
                            crop = _frame_pil.crop((x0, y0, x1, y1)).copy()
                            # Draw the bbox in crop-local coordinates.
                            draw = _PILDraw.Draw(crop)
                            bx0 = int(cx_px - w_px/2) - x0
                            by0 = int(cy_px - h_px/2) - y0
                            bx1 = int(cx_px + w_px/2) - x0
                            by1 = int(cy_px + h_px/2) - y0
                            draw.rectangle([bx0, by0, bx1, by1],
                                           outline=(0, 220, 0), width=4)
                            img_filename_per_det = (
                                f"frame_{frm['frame_num']:06d}"
                                f"_det_{kept_idx_in_frame:02d}.jpg"
                            )
                            crop.save(images_dir / img_filename_per_det,
                                      format="JPEG", quality=85)
                        kept_idx_in_frame += 1
                    except Exception:
                        # If crop fails for any reason, leave img_filename_per_det
                        # as None — the KMZ will simply render without an image
                        # for that placemark rather than break the whole pass.
                        pass

                    # ── Per-object geolocation ────────────────────
                    # Project the bbox into world coords using camera pose +
                    # pixel position + class-known height. Falls back to camera
                    # lat/lon if we don't know the class's real-world height
                    # or if bearing is missing from the OCR overlay.
                    obj_lat = lat
                    obj_lng = lng
                    distance_m = None
                    obj_bearing = None
                    if (lat is not None and lng is not None
                            and bearing is not None
                            and image_w_px > 0 and image_h_px > 0):
                        obj_lat, obj_lng, distance_m, obj_bearing = project_detection_latlon(
                            cam_lat=lat, cam_lon=lng, cam_bearing_deg=bearing,
                            box_cx_norm=cx_norm, box_h_norm=h_norm,
                            class_name=cls_name,
                            image_w_px=image_w_px, image_h_px=image_h_px,
                        )

                    detections.append({
                        "class_id":    cls_id,
                        "class_name":  cls_name,
                        "confidence":  round(confidence, 3),
                        "cx_norm":     round(cx_norm, 4),
                        "cy_norm":     round(cy_norm, 4),
                        "w_norm":      round(w_norm, 4),
                        "h_norm":      round(h_norm, 4),
                        "obj_lat":     round(obj_lat, 7) if obj_lat is not None else None,
                        "obj_lng":     round(obj_lng, 7) if obj_lng is not None else None,
                        "distance_m":  round(distance_m, 2) if distance_m is not None else None,
                        "obj_bearing": round(obj_bearing, 1) if obj_bearing is not None else None,
                        "image_file":  img_filename_per_det,
                    })

                    # ── Option A: flat record per kept detection, fed into
                    # the cross-frame deduper after the loop. The core six
                    # fields (frame_id/timestamp/class/confidence/lat/lon
                    # /bbox_area) are the deduper's contract; extra fields
                    # are passed through unchanged on the best-scoring
                    # detection so the final KML placemark has the right
                    # popup (image, distance, bearings, timestamp).
                    ded_lat = obj_lat if obj_lat is not None else lat
                    ded_lon = obj_lng if obj_lng is not None else lng
                    if ded_lat is not None and ded_lon is not None:
                        raw_dedup_entries.append({
                            # --- deduper contract ---
                            "frame_id":   frm["frame_num"],
                            "timestamp":  float(frm["offset_sec"]),
                            "class":      cls_name,
                            "confidence": round(confidence, 3),
                            "lat":        round(ded_lat, 7),
                            "lon":        round(ded_lon, 7),
                            "bbox_area":  round(w_norm * h_norm * image_w_px * image_h_px, 1),
                            # --- metadata carried through for KML ---
                            "class_id":     cls_id,
                            "image_file":   img_filename_per_det,
                            "timestamp_iso": frm["frame_time"].isoformat(),
                            "elevation_m":  round(ele, 2) if ele is not None else 0.0,
                            "distance_m":   round(distance_m, 2) if distance_m is not None else None,
                            "obj_bearing":  round(obj_bearing, 1) if obj_bearing is not None else None,
                            "cam_lat":      round(lat, 7) if lat is not None else None,
                            "cam_lng":      round(lng, 7) if lng is not None else None,
                            "cam_bearing":  round(bearing, 1) if bearing is not None else None,
                            "gps_source":   gps_source,
                        })

            # Per-detection crops (saved above, one per kept detection)
            # replace the old per-frame annotated image. The full
            # annotated frame is no longer copied into images_dir — each
            # KMZ placemark now references ONLY the bbox crop for its
            # own detection, so the user sees the specific object YOLO
            # found, not a frame-wide view with every other bbox drawn.

            # If this frame had detections but no GPS, nothing will be plotted —
            # count them so the final message tells the user how many kept
            # detections were lost for missing coords.
            if lat is None and detections:
                no_gps_drops += len(detections)

            # Build GeoJSON features
            if lat is not None:
                if detections:
                    for det in detections:
                        # Object coords from monocular projection (or fall back
                        # to the camera's lat/lon if we couldn't estimate).
                        plot_lat = det.get("obj_lat") if det.get("obj_lat") is not None else lat
                        plot_lng = det.get("obj_lng") if det.get("obj_lng") is not None else lng
                        feat = geojson.Feature(
                            geometry=geojson.Point((plot_lng, plot_lat, ele)),
                            properties={
                                "type":        "detection",
                                "class_id":    det["class_id"],
                                "class_name":  det["class_name"],
                                "confidence":  det["confidence"],
                                "frame_num":   frm["frame_num"],
                                "video_sec":   frm["offset_sec"],
                                "timestamp":   frm["frame_time"].isoformat(),
                                # Object (estimated) position
                                "latitude":    round(plot_lat, 7),
                                "longitude":   round(plot_lng, 7),
                                "elevation_m": round(ele, 2),
                                "distance_m":  det.get("distance_m"),
                                "obj_bearing": det.get("obj_bearing"),
                                # Camera pose when this frame was captured
                                "cam_lat":     round(lat, 7),
                                "cam_lng":     round(lng, 7),
                                "cam_bearing": round(bearing, 1) if bearing is not None else None,
                                # Raw bbox (for debugging / downstream tools)
                                "bbox_cx":     det.get("cx_norm"),
                                "bbox_cy":     det.get("cy_norm"),
                                "bbox_w":      det.get("w_norm"),
                                "bbox_h":      det.get("h_norm"),
                                "gps_source":  gps_source,
                                "image_file":  det.get("image_file"),
                            }
                        )
                        features.append(feat)
                else:
                    features.append(geojson.Feature(
                        geometry=geojson.Point((lng, lat, ele)),
                        properties={
                            "type":        "gps_track",
                            "frame_num":   frm["frame_num"],
                            "video_sec":   frm["offset_sec"],
                            "timestamp":   frm["frame_time"].isoformat(),
                            "latitude":    round(lat, 7),
                            "longitude":   round(lng, 7),
                            "elevation_m": round(ele, 2),
                            "cam_bearing": round(bearing, 1) if bearing is not None else None,
                            "gps_source":  gps_source,
                            "image_file":  None,
                        }
                    ))

            progress = 40 + int((i / len(extracted)) * 55)
            update("running", f"Processed {i+1}/{len(extracted)} frames...", progress)

        # ── Step 5: Write GeoJSON output ─────────────────────
        update("running", "Writing outputs...", 97)
        collection = geojson.FeatureCollection(features)
        geojson_path = job_dir / "detections.geojson"
        with open(geojson_path, "w") as f:
            geojson.dump(collection, f, indent=2)

        # ── Step 5b: Dump raw per-frame detections (unchanged; debug aid) ──
        raw_dedup_path = job_dir / "detections_raw.json"
        with open(raw_dedup_path, "w") as f:
            json.dump(raw_dedup_entries, f, indent=2)

        # ── Step 5c: Cross-frame dedup (Option A) ────────────────────
        # Collapse the flat per-frame detection list into one asset per
        # real-world object, using per-class distance thresholds + a
        # 3-second time gate. This is what the KMZ placemarks are built
        # from below — so the user sees one pole, not 20 copies of it.
        deduped_assets = dedup_per_frame_detections(raw_dedup_entries)

        # Write a Deduped-only GeoJSON (detections.geojson stays as the
        # full per-frame audit trail)
        deduped_features = [
            geojson.Feature(
                geometry=geojson.Point(
                    (a["lon"], a["lat"], a.get("elevation_m") or 0.0)
                ),
                properties={
                    "type":                  "detection",
                    "class_id":              a.get("class_id"),
                    "class_name":            a["class"],
                    "confidence":            a["confidence"],
                    "num_detections_merged": a.get("num_detections_merged", 1),
                    "latitude":              a["lat"],
                    "longitude":             a["lon"],
                    "elevation_m":           a.get("elevation_m") or 0.0,
                    "distance_m":            a.get("distance_m"),
                    "obj_bearing":           a.get("obj_bearing"),
                    "cam_lat":               a.get("cam_lat"),
                    "cam_lng":               a.get("cam_lng"),
                    "cam_bearing":           a.get("cam_bearing"),
                    "gps_source":            a.get("gps_source"),
                    "image_file":            a.get("image_file"),
                    "timestamp":             a.get("timestamp_iso", ""),
                    "video_sec":             a.get("timestamp"),
                    "members_video_sec":     a.get("members_video_sec", []),
                }
            )
            for a in deduped_assets
        ]
        deduped_path = job_dir / "detections_deduped.geojson"
        with open(deduped_path, "w") as f:
            geojson.dump(geojson.FeatureCollection(deduped_features), f, indent=2)

        # ── Step 5d: CSV of solved asset coordinates ─────────────────
        # One row per final pin — the triangulated (or fallback) asset.
        # This is the "assets_triangulated" deliverable: everything the
        # pipeline decided is a real-world object, with the provenance
        # of how its coord was placed (triangulated / cam+offset /
        # cam_only) and the camera pose that anchored it. Use this CSV
        # when you want to audit pin placement outside Google Earth —
        # or to re-solve asset positions from the raw numbers.
        assets_csv_path = job_dir / "detections_deduped.csv"
        # Column order matches the user-requested schema at the front
        # (asset_id, class, solved_lat, solved_lon, solve_method,
        # num_observations, mean_conf, baseline_m, ray_angle_deg,
        # residual_m, rejection_reason, source_frames) and then
        # carries the richer pipeline fields after for full auditability.
        csv_fields = [
            # ── Survey-grade fields (user-requested) ───────────────
            "asset_id",
            "class",
            "solved_lat",
            "solved_lon",
            "solve_method",
            "num_observations",
            "mean_conf",
            "baseline_m",
            "ray_angle_deg",
            "residual_m",
            "rejection_reason",
            "source_frames",
            # ── Raw pre-snap / pre-stretch triangulation output ───
            "raw_tri_lat",
            "raw_tri_lon",
            "tri_closest_dist_m",
            "tri_farthest_dist_m",
            # ── Richer pipeline fields (provenance) ───────────────
            "class_id",
            "confidence",
            "elevation_m",
            "distance_m",
            "obj_bearing",
            "cam_lat",
            "cam_lng",
            "cam_bearing",
            "offset_m",
            "offset_bearing",
            "gps_source",
            "timestamp_iso",
            "video_sec",
            "members_video_sec",
            "image_file",
            "coords_from_frame",
            "image_from_frame",
        ]
        try:
            with open(assets_csv_path, "w", newline="") as _cf:
                w = csv.DictWriter(_cf, fieldnames=csv_fields,
                                   extrasaction="ignore")
                w.writeheader()
                for a in deduped_assets:
                    # asset_id: use coords_from_frame (the frame that
                    # anchored the pin), falling back to image_from_frame
                    # or the timestamp. Stable across re-runs.
                    asset_id = (a.get("coords_from_frame")
                                or a.get("image_from_frame")
                                or a.get("timestamp_iso"))
                    # rejection_reason: preserve the tri gate that
                    # fired even when we fell back to cam+offset.
                    # Priority: tri_reject_reason (from diag) >
                    # triangulation_rejected (post-solve clamp) > None.
                    reject_reason = (a.get("tri_reject_reason")
                                     or a.get("triangulation_rejected"))
                    row = {
                        # User-requested schema
                        "asset_id":               asset_id,
                        "class":                  a.get("class"),
                        "solved_lat":             a.get("lat"),
                        "solved_lon":             a.get("lon"),
                        "solve_method":           a.get("coord_source"),
                        "num_observations":       a.get("num_detections_merged", 1),
                        "mean_conf":              a.get("mean_conf"),
                        "baseline_m":             a.get("tri_baseline_m"),
                        "ray_angle_deg":          a.get("tri_spread_deg"),
                        "residual_m":             a.get("triangulation_residual_m"),
                        "rejection_reason":       reject_reason,
                        "source_frames":          "|".join(
                            str(x) for x in (a.get("source_frames") or [])
                        ),
                        # Raw pre-snap/pre-stretch triangulation
                        "raw_tri_lat":            a.get("raw_tri_lat"),
                        "raw_tri_lon":            a.get("raw_tri_lon"),
                        "tri_closest_dist_m":     a.get("tri_closest_dist_m"),
                        "tri_farthest_dist_m":    a.get("tri_farthest_dist_m"),
                        # Richer pipeline fields
                        "class_id":               a.get("class_id"),
                        "confidence":             a.get("confidence"),
                        "elevation_m":            a.get("elevation_m"),
                        "distance_m":             a.get("distance_m"),
                        "obj_bearing":            a.get("obj_bearing"),
                        "cam_lat":                a.get("cam_lat"),
                        "cam_lng":                a.get("cam_lng"),
                        "cam_bearing":            a.get("cam_bearing"),
                        "offset_m":               a.get("offset_m"),
                        "offset_bearing":         a.get("offset_bearing"),
                        "gps_source":             a.get("gps_source"),
                        "timestamp_iso":          a.get("timestamp_iso"),
                        "video_sec":              a.get("timestamp"),
                        "members_video_sec":      "|".join(
                            str(x) for x in (a.get("members_video_sec") or [])
                        ),
                        "image_file":             a.get("image_file"),
                        "coords_from_frame":      a.get("coords_from_frame"),
                        "image_from_frame":       a.get("image_from_frame"),
                    }
                    w.writerow(row)
        except Exception as _e:
            # Don't let CSV formatting break the whole job; log and
            # continue so GeoJSON / KMZ still publish.
            print(f"[assets_csv] failed to write: {_e}")

        # ── Step 5e: Triangulation debug CSV (per-cluster diagnostics)
        # One row per cluster — accepted OR rejected. The "why did
        # asset X end up at pin Y" deliverable. Each row shows the two
        # most angularly-divergent rays in the cluster (frame_a /
        # frame_b), both cameras' poses, the raw triangulated point
        # BEFORE any stretch/snap, the distance from that point to the
        # nearest camera, and the accept/reject decision with reason.
        # Rejected clusters still show what the solve would have
        # placed — so you can see which gate killed each one.
        tri_debug_path = job_dir / "detections_triangulation_debug.csv"
        tri_debug_fields = [
            "class",
            "frame_a",
            "frame_b",
            "cam_a_lat",
            "cam_a_lng",
            "cam_a_bearing",
            "cam_b_lat",
            "cam_b_lng",
            "cam_b_bearing",
            "obj_bearing_a",
            "obj_bearing_b",
            "baseline_m",
            "intersection_angle_deg",
            "solved_lat",
            "solved_lon",
            "distance_from_track_m",
            "farthest_from_track_m",
            "residual_m",
            "n_rays",
            "n_cluster",
            "accept",
            "reject_reason",
        ]
        try:
            with open(tri_debug_path, "w", newline="") as _df:
                dw = csv.DictWriter(_df, fieldnames=tri_debug_fields,
                                    extrasaction="ignore")
                dw.writeheader()
                for a in deduped_assets:
                    d = a.get("_tri_diag") or {}
                    dw.writerow({
                        "class":                  d.get("cls") or a.get("class"),
                        "frame_a":                d.get("frame_a"),
                        "frame_b":                d.get("frame_b"),
                        "cam_a_lat":              d.get("cam_a_lat"),
                        "cam_a_lng":              d.get("cam_a_lng"),
                        "cam_a_bearing":          d.get("cam_a_bearing"),
                        "cam_b_lat":              d.get("cam_b_lat"),
                        "cam_b_lng":              d.get("cam_b_lng"),
                        "cam_b_bearing":          d.get("cam_b_bearing"),
                        "obj_bearing_a":          d.get("obj_bearing_a"),
                        "obj_bearing_b":          d.get("obj_bearing_b"),
                        "baseline_m":             d.get("baseline_m"),
                        "intersection_angle_deg": d.get("max_spread_deg"),
                        "solved_lat":             d.get("raw_lat"),
                        "solved_lon":             d.get("raw_lon"),
                        "distance_from_track_m":  d.get("closest_dist_m"),
                        "farthest_from_track_m":  d.get("farthest_dist_m"),
                        "residual_m":             d.get("residual_m"),
                        "n_rays":                 d.get("n_rays"),
                        "n_cluster":              d.get("n_cluster"),
                        "accept":                 (d.get("status") == "ok"),
                        "reject_reason":          d.get("reject_reason"),
                    })
        except Exception as _e:
            print(f"[tri_debug_csv] failed to write: {_e}")

        # Drop the internal diag blob from each asset before the
        # deduped_features GeoJSON is dumped — it's not JSON-safe and
        # the GeoJSON consumers don't need it.
        for a in deduped_assets:
            a.pop("_tri_diag", None)

        # ── Step 6: Extract video thumbnail for KMZ overlay ──
        # Frames-mode has no video; use the first frame as the thumbnail.
        thumb_path = images_dir / "thumbnail.jpg"
        thumb_name = None
        if not frames_mode:
            thumb_cmd = [
                "ffmpeg", "-y", "-ss", "1", "-i", str(video_path),
                "-frames:v", "1", "-q:v", "3", str(thumb_path)
            ]
            if subprocess.run(thumb_cmd, capture_output=True).returncode == 0 and thumb_path.exists():
                thumb_name = "thumbnail.jpg"
        elif frame_files:
            try:
                shutil.copy2(frame_files[0], thumb_path)
                thumb_name = "thumbnail.jpg"
            except Exception:
                pass

        # ── Step 7: Write KML — track line + detection placemarks ──
        kml_path = job_dir / "detections.kml"

        kml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2">',
            '<Document>',
            f'<name>Field Engineering — {job_id}</name>',

            # ── Styles ────────────────────────────────────────
            '<Style id="trackLine">',
            '  <LineStyle><color>ffff7800</color><width>3</width></LineStyle>',
            '</Style>',
            '<Style id="detection">',
            '  <IconStyle><color>ff0000ff</color><scale>1.3</scale>',
            '    <Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon>',
            '  </IconStyle>',
            '</Style>',
            '<Style id="gpstrack">',
            '  <IconStyle><color>ff888888</color><scale>0.5</scale>',
            '    <Icon><href>http://maps.google.com/mapfiles/kml/shapes/dot.png</href></Icon>',
            '  </IconStyle>',
            '</Style>',
            '<Style id="start">',
            '  <IconStyle><scale>1.2</scale>',
            '    <Icon><href>http://maps.google.com/mapfiles/kml/paddle/grn-circle.png</href></Icon>',
            '  </IconStyle>',
            '</Style>',
            '<Style id="end">',
            '  <IconStyle><scale>1.2</scale>',
            '    <Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-square.png</href></Icon>',
            '  </IconStyle>',
            '</Style>',
        ]

        # ── GPS track LineString ──────────────────────────────
        # Draw the line from the *camera* path, one point per extracted
        # frame — NOT from detection features (which now sit at projected
        # object positions and would zigzag the track).
        gps_coords = [
            (frm["lng"], frm["lat"], frm.get("ele") or 0.0)
            for frm in extracted
            if frm.get("lat") is not None and frm.get("lng") is not None
        ]
        if len(gps_coords) >= 2:
            coord_str = " ".join(f"{lng_c},{lat_c},{ele_c}" for lng_c, lat_c, ele_c in gps_coords)
            kml_lines += [
                '<Placemark>',
                '  <name>GPS Track</name>',
                '  <styleUrl>#trackLine</styleUrl>',
                '  <LineString>',
                '    <tessellate>1</tessellate>',
                f'    <coordinates>{coord_str}</coordinates>',
                '  </LineString>',
                '</Placemark>',
            ]
            # Start / End markers — source timestamps from the frame track,
            # not from `features` (detection features are now at projected
            # object positions, so their timestamps are per-object, not
            # per-frame ordered).
            s_lng, s_lat, s_ele = gps_coords[0]
            e_lng, e_lat, e_ele = gps_coords[-1]
            track_frames = [f for f in extracted if f.get("lat") is not None]
            first_ts = track_frames[0]["frame_time"].isoformat() if track_frames else ""
            last_ts  = track_frames[-1]["frame_time"].isoformat() if track_frames else ""
            kml_lines += [
                '<Placemark><name>Start</name><styleUrl>#start</styleUrl>',
                f'  <description><![CDATA[{first_ts}]]></description>',
                f'  <Point><coordinates>{s_lng},{s_lat},{s_ele}</coordinates></Point>',
                '</Placemark>',
                '<Placemark><name>End</name><styleUrl>#end</styleUrl>',
                f'  <description><![CDATA[{last_ts}]]></description>',
                f'  <Point><coordinates>{e_lng},{e_lat},{e_ele}</coordinates></Point>',
                '</Placemark>',
            ]

        # ── Detection + GPS track placemarks ─────────────────
        # Detections come from the DEDUPED feature list (one per real-world
        # asset). GPS track points still come from the per-frame features
        # list so the interpolated Track @Ns breadcrumbs are preserved.
        track_features = [f for f in features
                          if f["properties"].get("type") == "gps_track"]
        placemark_source = deduped_features + track_features

        for feat in placemark_source:
            lng_f, lat_f, ele_f = feat["geometry"]["coordinates"]
            props     = feat["properties"]
            feat_type = props.get("type", "detection")
            img_file  = props.get("image_file")

            if feat_type == "detection":
                cls_name  = props.get("class_name", f'class_{props.get("class_id","?")}')
                conf_pct  = f'{float(props.get("confidence", 0)) * 100:.1f}%'
                n_merged  = props.get("num_detections_merged", 1)
                suffix    = f" ×{n_merged}" if n_merged > 1 else ""
                name      = f'{cls_name} ({conf_pct}){suffix}'
                style_url = "#detection"
                dist_m    = props.get("distance_m")
                obj_br    = props.get("obj_bearing")
                cam_lat_p = props.get("cam_lat")
                cam_lng_p = props.get("cam_lng")
                cam_brg   = props.get("cam_bearing")
                desc_html = (
                    f"<b>{cls_name}</b> — {conf_pct}"
                )
                if n_merged > 1:
                    desc_html += f" &nbsp;(merged from {n_merged} frames)"
                desc_html += (
                    f"<br/>"
                    f"Video: {props.get('video_sec','?')}s &nbsp;|&nbsp; "
                    f"GPS: {props.get('gps_source','?')}<br/>"
                    f"<b>Object:</b> {props.get('latitude','?')}, {props.get('longitude','?')}<br/>"
                )
                if dist_m is not None:
                    desc_html += f"Distance from camera: {dist_m} m"
                    if obj_br is not None:
                        desc_html += f" &nbsp;|&nbsp; Bearing to object: {obj_br}°"
                    desc_html += "<br/>"
                if cam_lat_p is not None and cam_lng_p is not None:
                    desc_html += f"<i>Camera pos:</i> {cam_lat_p}, {cam_lng_p}"
                    if cam_brg is not None:
                        desc_html += f" &nbsp;(heading {cam_brg}°)"
                    desc_html += "<br/>"
                desc_html += f"{props.get('timestamp','')}<br/>"
                if img_file:
                    desc_html += f'<br/><img src="images/{img_file}" width="480"/>'
            else:
                name      = f'Track @{props.get("video_sec","?")}s'
                style_url = "#gpstrack"
                desc_html = (
                    f"GPS track point<br/>"
                    f"Source: {props.get('gps_source','?')}<br/>"
                    f"{props.get('timestamp','')}"
                )

            kml_lines += [
                "<Placemark>",
                f"  <name>{name}</name>",
                f"  <styleUrl>{style_url}</styleUrl>",
                f"  <TimeStamp><when>{props.get('timestamp','')}</when></TimeStamp>",
                f"  <description><![CDATA[{desc_html}]]></description>",
                "  <Point>",
                f"    <coordinates>{lng_f},{lat_f},{ele_f}</coordinates>",
                "  </Point>",
                "</Placemark>",
            ]

        # (video thumbnail ScreenOverlay removed — was visual clutter in KMZ)

        kml_lines += ["</Document>", "</kml>"]
        kml_path.write_text("\n".join(kml_lines))

        # ── Step 8: Package KMZ (KML + images in a zip) ──────
        kmz_path = job_dir / "detections.kmz"
        with zipfile.ZipFile(kmz_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(kml_path, "doc.kml")          # Google Earth expects doc.kml as root
            for img in images_dir.glob("*.jpg"):
                zf.write(img, f"images/{img.name}")

        # ── Step 8b: Write calibration.log for runtime diagnostics ───
        # _triangulate_asset / _dedup_merge_cluster push one [CAL],
        # [TRI_REJECT], [PATH] line each so we can prove — from the
        # status response alone — which branch placed each pin and why
        # triangulation was rejected when it was.
        cal_log_path = job_dir / "calibration.log"
        try:
            header = [
                f"# calibration log for job {job_id}",
                f"# TRI_STRETCH_DEFAULT={TRI_STRETCH_DEFAULT}",
                f"# CALIBRATION_BEARING_OFFSET_DEG={CALIBRATION_BEARING_OFFSET_DEG:+.2f}",
                "# TRI_STRETCH_BY_CLASS:",
            ] + [f"#   {k}: {v}" for k, v in TRI_STRETCH_BY_CLASS.items()] + [""]
            cal_log_path.write_text("\n".join(header + CAL_LOG_LINES))
            with zipfile.ZipFile(kmz_path, "a", zipfile.ZIP_DEFLATED) as zf:
                zf.write(cal_log_path, "calibration.log")
            cal_err = None
        except Exception as _cal_e:
            cal_log_path = None
            cal_err = str(_cal_e)

        # Detection counts — per-frame vs deduped (unique real-world assets)
        det_count_raw     = len([f for f in features if f["properties"].get("type") == "detection"])
        det_count_deduped = len(deduped_assets)
        # Build a diagnostic hint so zero-detection runs tell you *why*.
        if raw_detections_total == 0:
            hint = " — YOLO returned no boxes. Try higher imgsz (1920) or lower conf."
        elif kept_detections_total == 0:
            hint = (f" — YOLO returned {raw_detections_total} boxes but "
                    f"per-class filter dropped all. Lower `conf`.")
        elif no_gps_drops:
            hint = (f" — raw={raw_detections_total}, kept={kept_detections_total}, "
                    f"deduped={det_count_deduped}, "
                    f"but {no_gps_drops} were dropped because OCR couldn't read GPS "
                    f"on their frame (see ocr_misses / ocr_samples below).")
        else:
            hint = (f" — YOLO raw={raw_detections_total}, kept={kept_detections_total} "
                    f"after class filter, collapsed to {det_count_deduped} unique assets.")

        pipeline_jobs[job_id].update({
            "status":           "complete",
            "message":          (f"Done — {det_count_deduped} unique assets "
                                 f"({det_count_raw} per-frame detections across "
                                 f"{len(extracted)} frames){hint}"),
            "progress":         100,
            "finished_at":      datetime.utcnow().isoformat(),
            "detections":            det_count_deduped,
            "detections_per_frame":  det_count_raw,
            "raw_detections":   raw_detections_total,
            "kept_detections":  kept_detections_total,
            "no_gps_drops":     no_gps_drops,
            "ocr_hits":         ocr_hits,
            "ocr_misses_count": len(ocr_misses),
            "ocr_misses":       ocr_misses[:20],
            "ocr_samples":      ocr_samples,
            "frames_processed": len(extracted),
            "imgsz":            imgsz,
            "yolo_conf":        yolo_conf,
            "geojson_path":     str(geojson_path),
            "kml_path":         str(kml_path),
            "kmz_path":         str(kmz_path),
            "download_geojson": f"/pipeline/{job_id}/download/geojson",
            "download_kml":     f"/pipeline/{job_id}/download/kml",
            "download_kmz":     f"/pipeline/{job_id}/download/kmz",
            "assets_csv_path":  str(assets_csv_path),
            "download_assets_csv": f"/pipeline/{job_id}/download/assets_csv",
            "tri_debug_csv_path":  str(tri_debug_path),
            "download_tri_debug_csv": f"/pipeline/{job_id}/download/tri_debug_csv",
            "calibration_log_path":     str(cal_log_path) if cal_log_path else None,
            "download_calibration_log": f"/pipeline/{job_id}/download/calibration_log",
            "calibration_summary": {
                "TRI_STRETCH_DEFAULT":            TRI_STRETCH_DEFAULT,
                "CALIBRATION_BEARING_OFFSET_DEG": CALIBRATION_BEARING_OFFSET_DEG,
                "TRI_STRETCH_BY_CLASS":           TRI_STRETCH_BY_CLASS,
                "lines_count":                    len(CAL_LOG_LINES),
                "lines_tail":                     CAL_LOG_LINES[-40:],
                "error":                          cal_err,
            },
        })

    except Exception as e:
        pipeline_jobs[job_id].update({
            "status":      "error",
            "message":     f"Pipeline failed: {e}",
            "finished_at": datetime.utcnow().isoformat(),
        })


# ── Pipeline routes ───────────────────────────────────────────

@app.post("/pipeline/run", tags=["Video Pipeline"])
async def run_pipeline(
    background_tasks:   BackgroundTasks,
    video:              UploadFile = File(..., description="MP4/MOV video file from phone"),
    gpx:                UploadFile = File(None, description="GPX GPS track file (optional — not needed if OCR is enabled)"),
    frame_interval:     float = 1.0,
    conf:               float = 0.25,
    imgsz:              int   = 1280,
    video_start_time:   str   = None,
    use_ocr:            bool  = True,
):
    """
    Full geocoded video detection pipeline.

    Upload your phone video. GPX file is optional — if Google Vision API
    is configured, GPS coordinates are read directly off each frame.
    """
    job_id = uuid.uuid4().hex[:8]
    job_dir = PIPELINE_PATH / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Persist the uploaded video to the job folder.
    video_path = job_dir / video.filename
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Optional GPX.
    gpx_path = None
    if gpx is not None:
        gpx_path = job_dir / gpx.filename
        with open(gpx_path, "wb") as f:
            shutil.copyfileobj(gpx.file, f)

    pipeline_jobs[job_id] = {
        "job_id":         job_id,
        "status":         "queued",
        "message":        "Queued for processing",
        "progress":       0,
        "started_at":     datetime.utcnow().isoformat(),
        "finished_at":    None,
        "video_file":     video.filename,
        "gpx_file":       gpx.filename if gpx is not None else "none (OCR mode)",
        "gps_mode":       "ocr" if use_ocr else "gpx",
        "frame_interval": frame_interval,
        "conf":           conf,
        "imgsz":          imgsz,
    }

    background_tasks.add_task(
        _run_pipeline,
        job_id=job_id,
        video_path=video_path,
        gpx_path=gpx_path,
        frame_interval=frame_interval,
        conf=conf,
        video_start_override=video_start_time,
        use_ocr=use_ocr,
        imgsz=imgsz,
    )
    return pipeline_jobs[job_id]


@app.post("/pipeline/run-frames", tags=["Video Pipeline"])
async def run_pipeline_frames(
    background_tasks:   BackgroundTasks,
    frames_zip:         UploadFile = File(..., description="ZIP of pre-extracted frames"),
    gpx:                UploadFile = File(None, description="GPX GPS track file (optional)"),
    frame_interval:     float = 1.0,
    conf:               float = 0.25,
    imgsz:              int   = 1280,
    video_start_time:   str   = None,
    use_ocr:            bool  = True,
):
    """Run the pipeline against a ZIP of already-extracted frames."""
    job_id = uuid.uuid4().hex[:8]
    job_dir = PIPELINE_PATH / job_id
    frames_dir = job_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Save the zip, then unpack into frames_dir.
    zip_path = job_dir / frames_zip.filename
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(frames_zip.file, f)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(frames_dir)

    gpx_path = None
    if gpx is not None:
        gpx_path = job_dir / gpx.filename
        with open(gpx_path, "wb") as f:
            shutil.copyfileobj(gpx.file, f)

    pipeline_jobs[job_id] = {
        "job_id":         job_id,
        "status":         "queued",
        "message":        "Queued for processing (frames mode)",
        "progress":       0,
        "started_at":     datetime.utcnow().isoformat(),
        "finished_at":    None,
        "video_file":     "(frames mode)",
        "gpx_file":       gpx.filename if gpx is not None else "none (OCR mode)",
        "gps_mode":       "ocr" if use_ocr else "gpx",
        "frame_interval": frame_interval,
        "conf":           conf,
        "imgsz":          imgsz,
    }

    background_tasks.add_task(
        _run_pipeline,
        job_id=job_id,
        video_path=None,
        gpx_path=gpx_path,
        frame_interval=frame_interval,
        conf=conf,
        video_start_override=video_start_time,
        use_ocr=use_ocr,
        imgsz=imgsz,
    )
    return pipeline_jobs[job_id]


@app.get("/pipeline/{job_id}/status", tags=["Video Pipeline"])
def pipeline_status(job_id: str):
    """Get the status of a pipeline job."""
    if job_id not in pipeline_jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    return pipeline_jobs[job_id]


@app.get("/pipeline/{job_id}/download/geojson", tags=["Video Pipeline"])
def download_geojson(job_id: str):
    """Download the GeoJSON detection output for a completed pipeline run."""
    if job_id not in pipeline_jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    path = PIPELINE_PATH / job_id / "detections.geojson"
    if not path.exists():
        raise HTTPException(404, "GeoJSON not ready yet. Check /pipeline/{job_id}/status")
    return FileResponse(path, media_type="application/geo+json",
                        filename=f"detections-{job_id}.geojson")


@app.get("/pipeline/{job_id}/download/kml", tags=["Video Pipeline"])
def download_kml(job_id: str):
    """Download the KML detection output for a completed pipeline run."""
    if job_id not in pipeline_jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    path = PIPELINE_PATH / job_id / "detections.kml"
    if not path.exists():
        raise HTTPException(404, "KML not ready yet. Check /pipeline/{job_id}/status")
    return FileResponse(path, media_type="application/vnd.google-earth.kml+xml",
                        filename=f"detections-{job_id}.kml")


@app.get("/pipeline/{job_id}/download/kmz", tags=["Video Pipeline"])
def download_kmz(job_id: str):
    """
    Download the KMZ package for a completed pipeline run.

    KMZ is a zip of the KML + per-detection crop images.
    Open in Google Earth Pro to see each detection with its photo pop-up.
    """
    if job_id not in pipeline_jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    path = PIPELINE_PATH / job_id / "detections.kmz"
    if not path.exists():
        raise HTTPException(404, "KMZ not ready yet. Check /pipeline/{job_id}/status")
    return FileResponse(path, media_type="application/vnd.google-earth.kmz",
                        filename=f"detections-{job_id}.kmz")


@app.get("/pipeline/{job_id}/download/assets_csv", tags=["Video Pipeline"])
def download_assets_csv(job_id: str):
    """
    Download the deduped assets as a CSV — one row per final pin, with
    the camera pose + bearings + coord_source provenance that placed
    the pin. This is the "solved asset coordinates" deliverable;
    matches detections_deduped.geojson 1:1 but in tabular form.
    """
    if job_id not in pipeline_jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    path = PIPELINE_PATH / job_id / "detections_deduped.csv"
    if not path.exists():
        raise HTTPException(404, "detections_deduped.csv not ready yet. "
                                 "Check /pipeline/{job_id}/status")
    return FileResponse(path, media_type="text/csv",
                        filename=f"assets-{job_id}.csv")


@app.get("/pipeline/{job_id}/download/tri_debug_csv", tags=["Video Pipeline"])
def download_tri_debug_csv(job_id: str):
    """
    Download the per-cluster triangulation debug CSV. One row per
    cluster (accepted OR rejected), showing the two most divergent
    rays, their cameras' poses, the raw triangulated point before
    stretch/snap, and the accept/reject decision + reason. Pair this
    with calibration.log to diagnose why specific assets fail.
    """
    if job_id not in pipeline_jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    path = PIPELINE_PATH / job_id / "detections_triangulation_debug.csv"
    if not path.exists():
        raise HTTPException(404, "detections_triangulation_debug.csv not "
                                 "ready yet. Check /pipeline/{job_id}/status")
    return FileResponse(path, media_type="text/csv",
                        filename=f"tri-debug-{job_id}.csv")


@app.get("/pipeline/{job_id}/download/calibration_log", tags=["Video Pipeline"])
def download_calibration_log(job_id: str):
    """
    Download the runtime calibration log for a completed pipeline run.

    Each surviving cluster emits one [PATH] line showing which branch
    (triangulated / cam+offset / cam_only) placed the pin; triangulation
    failures emit one [TRI_REJECT] line with the exact reason. Useful
    for diagnosing why assets land where they do.
    """
    if job_id not in pipeline_jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    path = PIPELINE_PATH / job_id / "calibration.log"
    if not path.exists():
        raise HTTPException(404, "calibration.log not ready yet. Check /pipeline/{job_id}/status")
    return FileResponse(path, media_type="text/plain",
                        filename=f"calibration-{job_id}.log")


@app.get("/pipeline/jobs", tags=["Video Pipeline"])
def list_pipeline_jobs():
    """List all pipeline jobs and their current status."""
    return {
        jid: {
            "status":      j.get("status"),
            "message":     j.get("message"),
            "progress":    j.get("progress"),
            "started_at":  j.get("started_at"),
            "finished_at": j.get("finished_at"),
            "detections":  j.get("detections"),
        }
        for jid, j in pipeline_jobs.items()
    }


@app.post("/ocr/set-key", tags=["GPS Calibration"])
def set_ocr_api_key(api_key: str):
    """
    Set the Google Cloud Vision API key at runtime — no rebuild needed.

    The key is stored in memory for the life of the container.
    To persist it across restarts, add GOOGLE_VISION_API_KEY to your .env file.

    Get a free key at: https://console.cloud.google.com
    Enable: Cloud Vision API → Credentials → Create API Key
    Free tier: 1000 image requests/month
    """
    set_vision_api_key(api_key)
    return {
        "message": "API key set successfully",
        "vision_api_configured": True,
    }


@app.get("/ocr/status", tags=["GPS Calibration"])
def ocr_status():
    """Report whether Google Cloud Vision API is configured."""
    return {
        "vision_api_configured": bool(os.getenv("GOOGLE_VISION_API_KEY")),
        "note": "OCR reads GPS directly from iPhone NavCam burned-in overlays.",
    }
