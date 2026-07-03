"""
NatureDex AI — Congressional App Challenge
An AI-powered wildlife identification and learning platform.
"""

import sys
import os
import json
import datetime
import threading
import urllib.request
import urllib.parse
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QTextEdit, QLineEdit,
    QSplitter, QStackedWidget, QGraphicsOpacityEffect, QComboBox,
    QMenu, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation,
    QEasingCurve, QRect, QSize, pyqtProperty, QObject
)
from PyQt6.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush,
    QLinearGradient, QPalette, QFontDatabase, QIcon
)

import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from openai import OpenAI

# PyTorch — custom NC wildlife model
import torch
import torch.nn.functional as F
from torchvision import transforms, models as torch_models
from PIL import Image as PILImage

# ─── Constants ────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_FILE   = Path.home() / ".naturedex_collection.json"
ACHIEVEMENTS_FILE = Path.home() / ".naturedex_achievements.json"
CORRECTIONS_FILE  = Path.home() / ".naturedex_corrections.json"

# Custom NC model — sits in models/ folder next to naturedex.py
_SCRIPT_DIR      = Path(__file__).parent
CUSTOM_MODEL_PTH = _SCRIPT_DIR / "models" / "naturedex_nc_v1.pth"
CUSTOM_MODEL_THRESHOLD = 60.0   # % — use custom model if confidence above this

# ── The Ranger's Field Guide Color Palette ───────────────────────────────────
# 60% Base: Deep desaturated forest — organic, not pitch black
# 30% Structure: Dark forest/charcoal for cards and nav containers
# 10% Accent: Sunset orange for all action items, badges, active states
C_BG         = "#1a1f14"       # 60% — deep forest floor
C_PANEL      = "#222918"       # 60% — slightly lifted, warmer forest panel
C_CARD       = "#2a3320"       # 30% — dark moss card background
C_BORDER     = "#3a4a2a"       # structural border, used very sparingly

C_ACCENT     = "#e8720c"       # 10% — sunset orange, all interactive/action elements
C_ACCENT_DIM = "#c45e08"       # dimmer orange for hover states

C_TEXT       = "#f0ead8"       # cream/canvas — warm, readable on dark green
C_SUBTEXT    = "#7a9060"       # muted sage for secondary labels

# Semantic
C_GREEN      = "#6abf5e"       # earthy green — common species, success
C_YELLOW     = "#d4a017"       # warm amber — uncommon
C_RED        = "#c0392b"       # deep red — rare/errors
C_PURPLE     = "#8e5fb5"       # forest violet — very rare / legendary
C_GOLD       = "#e8a020"       # warm gold — achievements
C_PINK       = "#c45e8a"       # muted rose — fun facts

C_ACCENT2    = C_ACCENT_DIM
C_SCREEN     = "#0f1409"       # camera area — darkest forest
C_SCAN_LINE  = C_ACCENT

GROQ_MODEL   = "llama-3.3-70b-versatile"

# ─── Achievement Definitions ──────────────────────────────────────────────────
# Each badge is checked after every scan. "type" determines how it's evaluated:
#   "count"    -> unlocked when len(collection) >= threshold
#   "category" -> unlocked when number of UNIQUE categories seen >= threshold
ACHIEVEMENTS = [
    {"id": "first_scan",   "type": "count",    "threshold": 1,  "icon": "🔍", "name": "First Discovery",     "desc": "Scan your first object"},
    {"id": "count_5",      "type": "count",    "threshold": 5,  "icon": "🌱", "name": "Budding Naturalist",  "desc": "Discover 5 species"},
    {"id": "count_10",     "type": "count",    "threshold": 10, "icon": "🌿", "name": "Field Explorer",      "desc": "Discover 10 species"},
    {"id": "count_25",     "type": "count",    "threshold": 25, "icon": "🌳", "name": "Wildlife Tracker",    "desc": "Discover 25 species"},
    {"id": "count_50",     "type": "count",    "threshold": 50, "icon": "🏆", "name": "Master Naturalist",   "desc": "Discover 50 species"},
    {"id": "cats_3",       "type": "category", "threshold": 3,  "icon": "🎯", "name": "Well-Rounded",        "desc": "Discover 3 different categories"},
    {"id": "cats_5",       "type": "category", "threshold": 5,  "icon": "🧭", "name": "Diverse Explorer",    "desc": "Discover 5 different categories"},
    {"id": "cats_7",       "type": "category", "threshold": 7,  "icon": "🌍", "name": "Renaissance Scout",   "desc": "Discover 7 different categories"},
]

# ─── iNaturalist API Lookup ───────────────────────────────────────────────────

# North Carolina's iNaturalist place ID — used to get NC-specific observation counts
NC_PLACE_ID = 51

def inat_lookup(label: str) -> dict:
    """Query the iNaturalist taxa API with a label from MobileNetV2 and return
    enriched species data: scientific name, conservation status, NC observation
    count, taxon rank, iconic taxon, and Wikipedia summary if available.

    Returns an empty dict on any error so callers can always safely use .get().
    This function is intentionally synchronous — it runs inside AnalysisWorker's
    thread so it never blocks the UI.
    """
    try:
        # ── Step 1: taxa search by name ───────────────────────────────────────
        params = urllib.parse.urlencode({
            "q": label,
            "per_page": 1,
            "rank": "species,genus,family",   # skip kingdom/class level hits
            "is_active": "true",
        })
        url = f"https://api.inaturalist.org/v1/taxa?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "NatureDexAI/1.0"})

        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())

        results = data.get("results", [])
        if not results:
            return {}

        taxon = results[0]
        taxon_id = taxon.get("id")

        # Pull the fields we care about
        inat_data = {
            "taxon_id":            taxon_id,
            "scientific_name":     taxon.get("name", ""),
            "common_name":         taxon.get("preferred_common_name", ""),
            "rank":                taxon.get("rank", ""),
            "iconic_taxon":        taxon.get("iconic_taxon_name", ""),   # Bird, Reptilia, Plantae etc.
            "conservation_status": _parse_conservation(taxon),
            "wikipedia_summary":   taxon.get("wikipedia_summary", ""),
            "observations_count":  taxon.get("observations_count", 0),
        }

        # ── Step 2: NC observation count for rarity context ───────────────────
        if taxon_id:
            inat_data["nc_observations"] = _get_nc_count(taxon_id)

        return inat_data

    except Exception as e:
        # Network down, timeout, bad JSON — degrade silently
        print(f"[iNat lookup] {e}")
        return {}


def _parse_conservation(taxon: dict) -> str:
    """Extract a human-readable conservation status from the taxon dict.
    iNaturalist nests this in conservation_status.status_name or as a
    direct iucn_status_name field depending on API version."""
    # Try nested conservation_status object first (v1 API)
    cs = taxon.get("conservation_status", {})
    if isinstance(cs, dict):
        name = cs.get("status_name", "")
        if name:
            return name.replace("_", " ").title()

    # Fallback: top-level field present in some responses
    iucn = taxon.get("iucn_status_name", "")
    if iucn:
        return iucn.replace("_", " ").title()

    return ""   # Unknown — let Groq fill this in


def _get_nc_count(taxon_id: int) -> int:
    """Return the number of research-grade iNaturalist observations of this
    taxon in North Carolina. Used for rarity scoring later."""
    try:
        params = urllib.parse.urlencode({
            "taxon_id": taxon_id,
            "place_id": NC_PLACE_ID,
            "quality_grade": "research",
            "per_page": 0,   # we only need total_results, not actual records
        })
        url = f"https://api.inaturalist.org/v1/observations?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "NatureDexAI/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        return data.get("total_results", 0)
    except Exception:
        return 0


def _nc_rarity_label(nc_count: int) -> str:
    """Convert a raw NC observation count to a human-readable rarity label.
    Thresholds are rough but reasonable for NC wildlife; tune later with real data."""
    if nc_count == 0:
        return "Not Recorded in NC"
    elif nc_count < 10:
        return "Very Rare in NC"
    elif nc_count < 100:
        return "Rare in NC"
    elif nc_count < 1000:
        return "Uncommon in NC"
    elif nc_count < 10000:
        return "Common in NC"
    else:
        return "Very Common in NC"


# ─── Custom NC Model ──────────────────────────────────────────────────────────

# Image transform matching what train_model.py used for validation
_custom_transform = transforms.Compose([
    transforms.Resize(int(224 * 1.1)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_custom_model():
    """Load the trained EfficientNetV2-S NC wildlife model.
    Returns (model, label_map, device) or (None, None, None) if not found."""
    if not CUSTOM_MODEL_PTH.exists():
        print(f"[Custom model] Not found at {CUSTOM_MODEL_PTH} — using MobileNetV2 only")
        return None, None, None

    label_map_path = CUSTOM_MODEL_PTH.parent / "label_map.json"
    if not label_map_path.exists():
        print("[Custom model] label_map.json missing — skipping")
        return None, None, None

    try:
        device = torch.device("cpu")   # Mac inference runs on CPU

        checkpoint = torch.load(CUSTOM_MODEL_PTH, map_location=device, weights_only=False)
        num_classes = checkpoint["num_classes"]

        model = torch_models.efficientnet_v2_s(weights=None)
        import torch.nn as nn
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(device)

        with open(label_map_path) as f:
            label_map = json.load(f)

        print(f"[Custom model] Loaded — {num_classes} NC species, "
              f"val acc {checkpoint.get('val_accuracy', '?'):.1f}%")
        return model, label_map, device

    except Exception as e:
        print(f"[Custom model] Failed to load: {e}")
        return None, None, None


@torch.no_grad()
def run_custom_model(model, label_map, device, image_path: str):
    """Run the custom model on an image. Returns (label, confidence, alternatives)
    or None if model is unavailable."""
    if model is None:
        return None

    try:
        img = PILImage.open(image_path).convert("RGB")
        tensor = _custom_transform(img).unsqueeze(0).to(device)
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

        top5_probs, top5_idx = torch.topk(probs, min(5, len(label_map)))

        top_idx   = top5_idx[0].item()
        top_prob  = top5_probs[0].item() * 100
        top_info  = label_map.get(str(top_idx), {})
        top_label = top_info.get("common_name", f"Species {top_idx}")

        alternatives = []
        for prob, idx in zip(top5_probs[1:4], top5_idx[1:4]):
            info = label_map.get(str(idx.item()), {})
            alternatives.append({
                "name":       info.get("common_name", f"Species {idx.item()}"),
                "confidence": prob.item() * 100,
            })

        return top_label, top_prob, alternatives, top_info

    except Exception as e:
        print(f"[Custom model] Inference error: {e}")
        return None


# ─── Worker Threads ────────────────────────────────────────────────────────────

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._running = True
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            self.msleep(33)  # ~30fps

    def stop(self):
        self._running = False
        if self.cap:
            self.cap.release()
        self.wait()

    def capture_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return frame if ret else None
        return None


class AnalysisWorker(QThread):
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, frame, model, client, custom_model=None, custom_label_map=None, custom_device=None):
        super().__init__()
        self.frame            = frame
        self.model            = model        # MobileNetV2 (TensorFlow fallback)
        self.client           = client
        self.custom_model     = custom_model
        self.custom_label_map = custom_label_map
        self.custom_device    = custom_device

    def run(self):
        try:
            # Save frame
            tmp_path = "/tmp/naturedex_scan.jpg"
            cv2.imwrite(tmp_path, self.frame)

            # ── Step 1: Try custom NC model first ────────────────────────────
            used_custom = False
            custom_result = run_custom_model(
                self.custom_model, self.custom_label_map,
                self.custom_device, tmp_path
            )

            if custom_result and custom_result[1] >= CUSTOM_MODEL_THRESHOLD:
                # Custom model is confident — use it
                label, confidence, alternatives, top_info = custom_result
                raw_label = label.lower().replace(" ", "_")
                used_custom = True
                model_source = "NatureDex NC Model (89% accuracy)"
            else:
                # ── Step 1b: Fall back to MobileNetV2 ────────────────────────
                img = keras_image.load_img(tmp_path, target_size=(224, 224))
                arr = keras_image.img_to_array(img)
                arr = np.expand_dims(arr, axis=0)
                arr = preprocess_input(arr)
                preds = self.model.predict(arr, verbose=0)
                decoded = decode_predictions(preds, top=5)[0]

                top = decoded[0]
                label       = top[1].replace("_", " ").title()
                confidence  = top[2] * 100
                raw_label   = top[1]
                alternatives = [
                    {"name": d[1].replace("_", " ").title(), "confidence": d[2] * 100}
                    for d in decoded[1:4]
                ]
                model_source = "MobileNetV2 (ImageNet)"

                # If custom model ran but wasn't confident, still note its top guess
                if custom_result:
                    print(f"[Model] Custom model low confidence ({custom_result[1]:.1f}%) "
                          f"— using MobileNetV2 fallback")

            # ── Step 2: iNaturalist lookup ────────────────────────────────────
            inat_data = inat_lookup(label)

            # If custom model ran, use its taxon_id for more accurate iNat lookup
            if used_custom and top_info.get("taxon_id"):
                inat_data["taxon_id"]    = top_info["taxon_id"]
                inat_data["iconic_taxon"] = top_info.get("iconic_group", "")
                if not inat_data.get("nc_observations"):
                    inat_data["nc_observations"] = _get_nc_count(top_info["taxon_id"])

            # ── Step 3: Groq entry generation ────────────────────────────────
            entry = self._generate_entry(label, confidence, inat_data)

            nc_count = inat_data.get("nc_observations", 0)
            rarity   = _nc_rarity_label(nc_count)

            result = {
                "name":           inat_data.get("common_name") or label,
                "raw_label":      raw_label,
                "confidence":     confidence,
                "alternatives":   alternatives,
                "entry":          entry,
                "inat":           inat_data,
                "rarity":         rarity,
                "nc_observations":nc_count,
                "model_source":   model_source,
                "used_custom_model": used_custom,
                "timestamp":      datetime.datetime.now().isoformat(),
                "image_path":     tmp_path,
            }
            self.result_ready.emit(result)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _generate_entry(self, label: str, confidence: float, inat_data: dict) -> dict:
        # Build a context block from iNaturalist data so Groq uses real facts
        # rather than hallucinating conservation status, scientific names, etc.
        inat_context = ""
        if inat_data:
            lines = []
            if inat_data.get("scientific_name"):
                lines.append(f"Scientific name: {inat_data['scientific_name']}")
            if inat_data.get("common_name"):
                lines.append(f"Common name: {inat_data['common_name']}")
            if inat_data.get("rank"):
                lines.append(f"Taxonomic rank: {inat_data['rank']}")
            if inat_data.get("iconic_taxon"):
                lines.append(f"Taxonomic group: {inat_data['iconic_taxon']}")
            if inat_data.get("conservation_status"):
                lines.append(f"Conservation status (verified): {inat_data['conservation_status']}")
            if inat_data.get("observations_count"):
                lines.append(f"Global iNaturalist observations: {inat_data['observations_count']:,}")
            if inat_data.get("nc_observations") is not None:
                lines.append(f"North Carolina observations: {inat_data['nc_observations']:,}")
            if inat_data.get("wikipedia_summary"):
                # Trim long summaries — we just need enough for factual grounding
                summary = inat_data["wikipedia_summary"][:600]
                lines.append(f"Wikipedia summary: {summary}")
            if lines:
                inat_context = "\n\nVerified data from iNaturalist (use this — do NOT contradict it):\n" + "\n".join(lines)

        prompt = f"""You are NatureDex AI, an educational wildlife identification system.
A user has scanned an object identified as: {label} (confidence: {confidence:.1f}%){inat_context}

Generate a structured NatureDex entry. You MUST respond with ONLY valid JSON — no markdown, no code fences, no explanation.

IMPORTANT rules:
- Use the verified iNaturalist data above where provided — do NOT invent a different scientific name or conservation status
- If conservation_status is provided above, use it exactly
- If scientific_name is provided above, use it exactly
- For north_carolina_context, use the NC observation count above to describe how common it is in NC

The JSON must have exactly these keys:
{{
  "common_name": "...",
  "scientific_name": "...",
  "category": "Animal / Plant / Insect / Bird / Fish / Reptile / Object / Food / etc",
  "type_tags": ["tag1", "tag2"],
  "habitat": "...",
  "diet": "...",
  "behavior": "...",
  "conservation_status": "Least Concern / Near Threatened / Vulnerable / Endangered / Critically Endangered / N/A",
  "north_carolina_context": "Is this found in NC? How common? When is it seen? Any NC-specific facts?",
  "fun_fact": "...",
  "description": "2-3 sentence Pokédex-style description"
}}

If this is a non-living object, adapt the fields creatively in Pokédex style.
Return ONLY the JSON object. No other text."""

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,   # raised from 800 — truncation was causing JSON parse failures
        )
        raw = response.choices[0].message.content.strip()

        # Strategy: try increasingly aggressive extraction methods so we
        # never fall through to displaying raw text in the UI.

        # 1. Strip markdown code fences if present
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        # 2. If there's preamble text before the JSON, slice from first { to last }
        if not raw.startswith("{"):
            start = raw.find("{")
            end   = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = raw[start:end + 1]

        raw = raw.strip()

        # 3. If JSON was token-truncated mid-response, it won't have a closing }.
        #    Try to close it so json.loads has a chance.
        if raw.startswith("{") and not raw.endswith("}"):
            # Close any open string (remove trailing partial value)
            last_quote = raw.rfind('"')
            last_comma = raw.rfind(",")
            # Trim back to the last complete key-value pair
            cut = max(last_comma, 0)
            raw = raw[:cut].rstrip().rstrip(",") + "\n}"

        try:
            return json.loads(raw)
        except Exception:
            # Genuine parse failure — return a clean fallback with whatever
            # iNat data we have so at least the header renders correctly
            return {
                "common_name": inat_data.get("common_name") or label,
                "scientific_name": inat_data.get("scientific_name") or "Unknown",
                "category": inat_data.get("iconic_taxon") or "Unknown",
                "type_tags": [],
                "habitat": "Unknown",
                "diet": "Unknown",
                "behavior": "Unknown",
                "conservation_status": inat_data.get("conservation_status") or "N/A",
                "north_carolina_context": "Unknown",
                "fun_fact": "Analysis unavailable.",
                "description": "Entry generation failed — try scanning again.",
            }


class ChatWorker(QThread):
    reply_ready = pyqtSignal(str)

    def __init__(self, client, messages):
        super().__init__()
        self.client = client
        self.messages = messages

    def run(self):
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=self.messages,
                temperature=0.7,
                max_tokens=400,
            )
            self.reply_ready.emit(response.choices[0].message.content.strip())
        except Exception as e:
            self.reply_ready.emit(f"Error: {str(e)}")


# ─── UI Components ─────────────────────────────────────────────────────────────

class ScanButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("⬤  SCAN")
        self.setFixedSize(160, 52)
        self._scanning = False
        self._dot_count = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._pulse)
        self.setStyleSheet(self._normal_style())

    def _normal_style(self):
        return f"""
            QPushButton {{
                background: {C_ACCENT};
                color: #1a1f14;
                border: none;
                border-radius: 26px;
                font-size: 15px;
                font-weight: 900;
                letter-spacing: 3px;
            }}
            QPushButton:hover {{ background: #f08020; }}
            QPushButton:pressed {{ background: {C_ACCENT_DIM}; }}
        """

    def _scanning_style(self):
        return f"""
            QPushButton {{
                background: {C_CARD};
                color: {C_ACCENT};
                border: 2px solid {C_ACCENT};
                border-radius: 26px;
                font-size: 13px;
                font-weight: 700;
                letter-spacing: 2px;
            }}
        """

    def start_scanning(self):
        self._scanning = True
        self._dot_count = 0
        self.setEnabled(False)
        self.setStyleSheet(self._scanning_style())
        self._timer.start(400)

    def stop_scanning(self):
        self._scanning = False
        self._timer.stop()
        self.setText("⬤  SCAN")
        self.setEnabled(True)
        self.setStyleSheet(self._normal_style())

    def _pulse(self):
        dots = "." * (self._dot_count % 4)
        self.setText(f"SCANNING{dots}")
        self._dot_count += 1


class ScanOverlay(QWidget):
    """Animated scan line overlay drawn on top of camera feed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._active = False
        self._y = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        self._corner_flash = 0

    def start(self):
        self._active = True
        self._y = 0
        self._corner_flash = 0
        self._timer.start(16)
        self.show()

    def stop(self):
        self._active = False
        self._timer.stop()
        self.update()

    def _update(self):
        self._y = (self._y + 4) % max(self.height(), 1)
        self._corner_flash = (self._corner_flash + 1) % 30
        self.update()

    def paintEvent(self, event):
        if not self._active:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Subtle dark vignette
        painter.fillRect(0, 0, w, h, QColor(0, 0, 0, 40))

        # Sunset orange scan line
        grad = QLinearGradient(0, self._y - 20, 0, self._y + 20)
        grad.setColorAt(0.0, QColor(232, 114, 12, 0))
        grad.setColorAt(0.5, QColor(232, 114, 12, 160))
        grad.setColorAt(1.0, QColor(232, 114, 12, 0))
        painter.fillRect(0, self._y - 20, w, 40, grad)

        # Orange corner brackets
        pen = QPen(QColor(C_ACCENT), 3)
        painter.setPen(pen)
        corner = 26
        gap = 16
        for x, y in [(gap, gap), (w - gap, gap), (gap, h - gap), (w - gap, h - gap)]:
            dx = corner if x == gap else -corner
            dy = corner if y == gap else -corner
            painter.drawLine(x, y, x + dx, y)
            painter.drawLine(x, y, x, y + dy)

        # Center crosshair
        cx, cy = w // 2, h // 2
        pen2 = QPen(QColor(232, 114, 12, 80), 1)
        pen2.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(pen2)
        painter.drawLine(cx - 12, cy, cx + 12, cy)
        painter.drawLine(cx, cy - 12, cx, cy + 12)
        painter.end()


class ToastNotification(QFrame):
    """A small auto-dismissing popup used to announce unlocked achievements."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {C_CARD}, stop:1 #1f3f1f);
                border: 2px solid {C_GOLD};
                border-radius: 14px;
            }}
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(57, 255, 20, 100))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 16, 10)
        layout.setSpacing(12)

        self._icon_lbl = QLabel("🏆")
        self._icon_lbl.setStyleSheet("font-size: 26px;")
        layout.addWidget(self._icon_lbl)

        text_box = QVBoxLayout()
        text_box.setSpacing(1)
        header = QLabel("ACHIEVEMENT UNLOCKED")
        header.setStyleSheet(f"color: {C_GOLD}; font-size: 9px; font-weight: 800; letter-spacing: 1.5px;")
        self._name_lbl = QLabel("")
        self._name_lbl.setStyleSheet(f"color: {C_TEXT}; font-size: 13px; font-weight: 700;")
        text_box.addWidget(header)
        text_box.addWidget(self._name_lbl)
        layout.addLayout(text_box)

        self.setFixedWidth(280)
        self.adjustSize()
        self.hide()

        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self._fade_out)

        self._opacity_effect = None

    def show_achievement(self, icon, name):
        self._icon_lbl.setText(icon)
        self._name_lbl.setText(name)
        self.setWindowOpacity(1.0)
        self.show()
        self.raise_()
        self._dismiss_timer.start(3000)

    def _fade_out(self):
        self.hide()


class CollectionCard(QFrame):
    clicked_signal = pyqtSignal(dict)
    delete_signal = pyqtSignal(str)  # emits the entry's timestamp (used as unique id)

    def __init__(self, entry_data, parent=None):
        super().__init__(parent)
        self.entry_data = entry_data
        self._category = entry_data.get("entry", {}).get("category", "Unknown") or "Unknown"
        self._search_text = (
            entry_data.get("name", "") + " " +
            entry_data.get("entry", {}).get("common_name", "") + " " +
            self._category
        ).lower()
        self.setFixedHeight(64)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setStyleSheet(f"""
            QFrame {{
                background: {C_CARD};
                border: none;
                border-radius: 8px;
            }}
            QFrame:hover {{
                background: #254225;
                border-left: 3px solid {C_ACCENT};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 6, 6)
        layout.setSpacing(10)

        # Confidence indicator dot
        conf = entry_data.get("confidence", 0)
        dot_color = C_GREEN if conf >= 75 else C_YELLOW if conf >= 50 else C_RED
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {dot_color}; font-size: 10px;")
        dot.setFixedWidth(14)
        layout.addWidget(dot)

        # Name + timestamp
        info = QVBoxLayout()
        info.setSpacing(1)
        name_lbl = QLabel(entry_data.get("name", "Unknown"))
        name_lbl.setStyleSheet(f"color: {C_TEXT}; font-size: 12px; font-weight: 600;")
        ts = entry_data.get("timestamp", "")[:10]
        ts_lbl = QLabel(ts)
        ts_lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 10px;")
        info.addWidget(name_lbl)
        info.addWidget(ts_lbl)
        layout.addLayout(info)
        layout.addStretch()

        conf_lbl = QLabel(f"{conf:.0f}%")
        conf_lbl.setStyleSheet(f"color: {dot_color}; font-size: 11px; font-weight: 700;")
        layout.addWidget(conf_lbl)

        # Delete button - hidden until hover
        self._delete_btn = QPushButton("✕")
        self._delete_btn.setFixedSize(22, 22)
        self._delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {C_SUBTEXT};
                border: none;
                border-radius: 11px;
                font-size: 12px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background: {C_RED};
                color: white;
            }}
        """)
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        self._delete_btn.hide()
        layout.addWidget(self._delete_btn)

    def _on_delete_clicked(self):
        self.delete_signal.emit(self.entry_data.get("timestamp", ""))

    def enterEvent(self, event):
        self._delete_btn.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._delete_btn.hide()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        # Don't trigger card click if the delete button was clicked
        if self._delete_btn.geometry().contains(event.pos()):
            return
        self.clicked_signal.emit(self.entry_data)


# ─── Main Window ───────────────────────────────────────────────────────────────

class NatureDexWindow(QMainWindow):

    # Signal for safely updating the loading label from the background thread
    model_status_signal = pyqtSignal(str)  # "ready" | "error: ..."

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NatureDex AI")

        # Clamp window size to whatever screen we're actually on, so the
        # bottom controls can never end up pushed off-screen (e.g. smaller
        # laptop displays, scaled resolutions, or a menu bar eating space).
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen else QRect(0, 0, 1280, 800)
        target_w = min(1280, available.width() - 40)
        target_h = min(800, available.height() - 40)
        min_w = min(1000, target_w)
        min_h = min(640, target_h)
        self.setMinimumSize(min_w, min_h)
        self.resize(target_w, target_h)
        self.move(available.x() + 20, available.y() + 20)

        self._collection = self._load_collection()
        self._unlocked_achievements = self._load_achievements()
        self._current_result = None
        self._chat_history = []
        self._camera_thread = None
        self._analysis_worker = None
        self._last_frame = None
        self._scan_overlay = None
        self._toast = None

        # Load AI models
        self._model = None           # MobileNetV2 fallback
        self._client = None
        self._custom_model = None    # custom NC EfficientNetV2 model
        self._custom_label_map = None
        self._custom_device = None
        self._models_loaded = False

        self._setup_style()
        self._build_ui()
        self._start_camera()

        # Wire the signal BEFORE starting the loader so it's ready to receive
        self.model_status_signal.connect(self._on_model_status)
        self._load_models_async()

        # Toast lives on top of the whole window so it can appear regardless
        # of which tab/panel is active
        self._toast = ToastNotification(self)
        self._toast.move(self.width() - 300, 70)

    # ── Style ──────────────────────────────────────────────────────────────────

    def _setup_style(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background: {C_BG}; }}
            QWidget {{
                background: {C_BG};
                color: {C_TEXT};
                font-family: 'SF Pro Display', 'Segoe UI', sans-serif;
            }}
            QScrollArea {{ border: none; background: transparent; }}
            QScrollBar:vertical {{
                background: {C_PANEL};
                width: 5px;
                border-radius: 2px;
            }}
            QScrollBar::handle:vertical {{
                background: {C_BORDER};
                border-radius: 2px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
            QToolTip {{
                background: {C_CARD};
                color: {C_TEXT};
                border: 1px solid {C_ACCENT};
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 11px;
            }}
            QComboBox QAbstractItemView {{
                background: {C_CARD};
                color: {C_TEXT};
                selection-background-color: {C_BORDER};
                border: 1px solid {C_ACCENT};
                outline: none;
            }}
        """)

    # ── UI Build ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Left sidebar (collection)
        sidebar = self._build_sidebar()
        root_layout.addWidget(sidebar)

        # Main area: camera + info
        main = self._build_main()
        root_layout.addWidget(main, stretch=1)

    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(270)
        sidebar.setStyleSheet(f"background: {C_PANEL};")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header — flat, no gradient, no button feel
        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet(f"background: {C_PANEL};")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(18, 0, 14, 0)

        title = QLabel("NatureDex")
        title.setStyleSheet(f"""
            color: {C_TEXT};
            font-size: 20px;
            font-weight: 800;
            letter-spacing: 1px;
        """)
        h_layout.addWidget(title)
        h_layout.addStretch()

        # Trophy — plain text label, no box, no button styling
        self._badges_btn = QLabel("🏆")
        self._badges_btn.setStyleSheet(f"color: {C_GOLD}; font-size: 18px;")
        self._badges_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._badges_btn.setToolTip("Achievements")
        self._badges_btn.mousePressEvent = lambda e: self._show_badges_panel()
        h_layout.addWidget(self._badges_btn)

        layout.addWidget(header)

        # Thin divider — one line only, no stacked borders
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background: {C_BORDER};")
        layout.addWidget(divider)

        # Stats — inline with header feel, no separate block
        stats_frame = QFrame()
        stats_frame.setFixedHeight(36)
        stats_frame.setStyleSheet(f"background: {C_PANEL};")
        s_layout = QHBoxLayout(stats_frame)
        s_layout.setContentsMargins(18, 0, 18, 0)
        self._species_count_lbl = QLabel(f"{len(self._collection)} discovered")
        self._species_count_lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 11px;")
        s_layout.addWidget(self._species_count_lbl)
        layout.addWidget(stats_frame)

        # Search + filter — no box around them, just inputs floating on panel bg
        search_frame = QFrame()
        search_frame.setStyleSheet(f"background: {C_PANEL};")
        sf_layout = QVBoxLayout(search_frame)
        sf_layout.setContentsMargins(12, 4, 12, 10)
        sf_layout.setSpacing(6)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search discoveries...")
        self._search_input.setStyleSheet(f"""
            QLineEdit {{
                background: {C_CARD};
                color: {C_TEXT};
                border: none;
                border-radius: 8px;
                padding: 7px 12px;
                font-size: 12px;
            }}
            QLineEdit:focus {{ border: 1px solid {C_ACCENT}; }}
        """)
        self._search_input.textChanged.connect(self._apply_filters)
        sf_layout.addWidget(self._search_input)

        self._category_filter = QComboBox()
        self._category_filter.addItem("All Categories")
        self._category_filter.setStyleSheet(f"""
            QComboBox {{
                background: {C_CARD};
                color: {C_TEXT};
                border: none;
                border-radius: 8px;
                padding: 5px 10px;
                font-size: 11px;
            }}
            QComboBox::drop-down {{ border: none; width: 24px; }}
        """)
        self._category_filter.currentTextChanged.connect(self._apply_filters)
        sf_layout.addWidget(self._category_filter)

        layout.addWidget(search_frame)

        # Section label
        col_label = QLabel("DISCOVERIES")
        col_label.setStyleSheet(f"""
            color: {C_SUBTEXT};
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 2px;
            padding: 8px 18px 4px 18px;
        """)
        layout.addWidget(col_label)

        # Empty-state label, shown when filters match nothing
        self._empty_filter_lbl = QLabel("No matches found")
        self._empty_filter_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_filter_lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 12px; padding: 20px;")
        self._empty_filter_lbl.hide()
        layout.addWidget(self._empty_filter_lbl)

        # Scroll area for collection cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")

        self._collection_container = QWidget()
        self._collection_container.setStyleSheet("background: transparent;")
        self._collection_layout = QVBoxLayout(self._collection_container)
        self._collection_layout.setContentsMargins(10, 4, 10, 10)
        self._collection_layout.setSpacing(4)
        self._collection_layout.addStretch()

        scroll.setWidget(self._collection_container)
        layout.addWidget(scroll)

        for entry in reversed(self._collection):
            self._add_collection_card(entry, prepend=False)
        self._refresh_category_filter_options()

        return sidebar

    def _build_main(self):
        main = QWidget()
        layout = QHBoxLayout(main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Camera panel
        cam_panel = self._build_camera_panel()
        layout.addWidget(cam_panel, stretch=5)

        # Info panel
        info_panel = self._build_info_panel()
        layout.addWidget(info_panel, stretch=4)

        return main

    def _build_camera_panel(self):
        panel = QFrame()
        panel.setStyleSheet(f"background: {C_SCREEN};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        cam_header = QFrame()
        cam_header.setFixedHeight(40)
        cam_header.setStyleSheet(f"background: {C_PANEL};")
        ch_layout = QHBoxLayout(cam_header)
        ch_layout.setContentsMargins(16, 0, 16, 0)
        cam_lbl = QLabel("● LIVE SCANNER")
        cam_lbl.setStyleSheet(f"color: {C_ACCENT}; font-size: 11px; font-weight: 700; letter-spacing: 2px;")
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 11px;")
        ch_layout.addWidget(cam_lbl)
        ch_layout.addStretch()
        ch_layout.addWidget(self._status_lbl)
        layout.addWidget(cam_header)

        # Camera feed — pure black screen
        cam_container = QFrame()
        cam_container.setStyleSheet("background: #000;")
        cam_layout = QVBoxLayout(cam_container)
        cam_layout.setContentsMargins(0, 0, 0, 0)

        self._camera_label = QLabel()
        self._camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._camera_label.setStyleSheet("background: #000;")
        self._camera_label.setMinimumHeight(220)
        cam_layout.addWidget(self._camera_label)

        # Overlay (scan animation)
        self._scan_overlay = ScanOverlay(cam_container)
        self._scan_overlay.hide()

        layout.addWidget(cam_container, stretch=1)

        # Controls
        controls = QFrame()
        controls.setFixedHeight(88)
        controls.setStyleSheet(f"background: {C_PANEL}; border-top: 1px solid {C_BORDER};")
        c_layout = QHBoxLayout(controls)
        c_layout.setContentsMargins(24, 0, 24, 0)
        c_layout.setSpacing(16)

        self._scan_btn = ScanButton()
        self._scan_btn.clicked.connect(self._on_scan)

        hint = QLabel("Point camera at any\nplant, animal, or object")
        hint.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 11px; line-height: 1.5;")
        hint.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._loading_lbl = QLabel("Loading AI models...")
        self._loading_lbl.setStyleSheet(f"color: {C_ACCENT}; font-size: 11px;")

        c_layout.addWidget(self._scan_btn)
        c_layout.addWidget(hint)
        c_layout.addStretch()
        c_layout.addWidget(self._loading_lbl)

        layout.addWidget(controls)

        return panel

    def _build_info_panel(self):
        panel = QFrame()
        panel.setStyleSheet(f"background: {C_BG};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Tab header
        tab_bar = QFrame()
        tab_bar.setFixedHeight(44)
        tab_bar.setStyleSheet(f"background: {C_PANEL};")
        tab_layout = QHBoxLayout(tab_bar)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)

        self._tab_entry_btn = self._make_tab_btn("ENTRY", True)
        self._tab_chat_btn  = self._make_tab_btn("ASK AI", False)
        self._tab_entry_btn.clicked.connect(lambda: self._switch_tab(0))
        self._tab_chat_btn.clicked.connect(lambda: self._switch_tab(1))

        tab_layout.addWidget(self._tab_entry_btn)
        tab_layout.addWidget(self._tab_chat_btn)
        tab_layout.addStretch()
        layout.addWidget(tab_bar)

        # Stacked content
        self._tab_stack = QStackedWidget()
        self._tab_stack.addWidget(self._build_entry_tab())
        self._tab_stack.addWidget(self._build_chat_tab())
        layout.addWidget(self._tab_stack, stretch=1)

        return panel

    def _make_tab_btn(self, text, active):
        btn = QPushButton(text)
        btn.setFixedHeight(44)
        btn.setFixedWidth(110)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._set_tab_style(btn, active)
        return btn

    def _set_tab_style(self, btn, active):
        if active:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {C_BG};
                    color: {C_ACCENT};
                    border: none;
                    border-bottom: 2px solid {C_ACCENT};
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 2px;
                }}
            """)
        else:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    color: {C_SUBTEXT};
                    border: none;
                    font-size: 11px;
                    font-weight: 600;
                    letter-spacing: 2px;
                }}
                QPushButton:hover {{ color: {C_TEXT}; }}
            """)

    def _switch_tab(self, idx):
        self._tab_stack.setCurrentIndex(idx)
        self._set_tab_style(self._tab_entry_btn, idx == 0)
        self._set_tab_style(self._tab_chat_btn,  idx == 1)

    def _build_entry_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._entry_content = QWidget()
        self._entry_content.setStyleSheet("background: transparent;")
        self._entry_inner = QVBoxLayout(self._entry_content)
        self._entry_inner.setContentsMargins(20, 20, 20, 20)
        self._entry_inner.setSpacing(14)

        self._placeholder_lbl = QLabel("Scan an object to generate\na NatureDex entry.")
        self._placeholder_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_lbl.setStyleSheet(f"""
            color: {C_SUBTEXT};
            font-size: 15px;
            line-height: 1.8;
            padding: 60px 20px;
        """)
        self._entry_inner.addWidget(self._placeholder_lbl)
        self._entry_inner.addStretch()

        scroll.setWidget(self._entry_content)
        layout.addWidget(scroll, stretch=1)

        # ── Correction bar — hidden by default, shown inline when report clicked
        self._correction_bar = QFrame()
        self._correction_bar.setFixedHeight(50)
        self._correction_bar.setStyleSheet(f"background: {C_CARD};")
        cb_layout = QHBoxLayout(self._correction_bar)
        cb_layout.setContentsMargins(16, 8, 16, 8)
        cb_layout.setSpacing(8)

        cb_prompt = QLabel("Correct name:")
        cb_prompt.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 11px;")
        cb_layout.addWidget(cb_prompt)

        self._correction_input = QLineEdit()
        self._correction_input.setPlaceholderText("e.g. Eastern Bluebird")
        self._correction_input.setStyleSheet(f"""
            QLineEdit {{
                background: {C_BG};
                color: {C_TEXT};
                border: none;
                border-bottom: 1px solid {C_ACCENT};
                border-radius: 0;
                padding: 4px 6px;
                font-size: 12px;
            }}
        """)
        cb_layout.addWidget(self._correction_input, stretch=1)

        cb_submit = QPushButton("Submit")
        cb_submit.setCursor(Qt.CursorShape.PointingHandCursor)
        cb_submit.setStyleSheet(f"""
            QPushButton {{
                background: {C_ACCENT};
                color: #1a1f14;
                border: none;
                border-radius: 6px;
                padding: 4px 14px;
                font-size: 11px;
                font-weight: 700;
            }}
            QPushButton:hover {{ background: #f08020; }}
        """)
        cb_cancel = QPushButton("Cancel")
        cb_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cb_cancel.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {C_SUBTEXT};
                border: none;
                font-size: 11px;
            }}
            QPushButton:hover {{ color: {C_TEXT}; }}
        """)
        cb_layout.addWidget(cb_submit)
        cb_layout.addWidget(cb_cancel)
        self._correction_bar.hide()
        layout.addWidget(self._correction_bar)

        cb_submit.clicked.connect(self._on_correction_submit)
        cb_cancel.clicked.connect(self._on_correction_cancel)
        self._correction_input.returnPressed.connect(self._on_correction_submit)

        # ── Footer — report button always visible at bottom-right
        footer = QFrame()
        footer.setFixedHeight(40)
        footer.setStyleSheet(f"background: {C_BG};")
        f_layout = QHBoxLayout(footer)
        f_layout.setContentsMargins(16, 0, 16, 0)

        self._report_btn = QPushButton("⚑  Wrong ID? Report it")
        self._report_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._report_btn.setEnabled(False)
        self._report_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {C_SUBTEXT};
                border: none;
                font-size: 11px;
                text-align: right;
                padding: 0;
            }}
            QPushButton:enabled:hover {{ color: {C_ACCENT}; }}
            QPushButton:disabled {{ color: {C_BORDER}; }}
        """)
        self._report_btn.clicked.connect(self._on_report_wrong_id)

        f_layout.addStretch()
        f_layout.addWidget(self._report_btn)
        layout.addWidget(footer)

        return widget

    def _build_chat_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Chat messages
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._chat_scroll = scroll

        self._chat_content = QWidget()
        self._chat_inner = QVBoxLayout(self._chat_content)
        self._chat_inner.setContentsMargins(16, 16, 16, 8)
        self._chat_inner.setSpacing(10)

        intro = QLabel("Ask follow-up questions about\nyour last scan.")
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        intro.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 13px; padding: 40px 0;")
        self._chat_intro = intro
        self._chat_inner.addWidget(intro)
        self._chat_inner.addStretch()

        scroll.setWidget(self._chat_content)
        layout.addWidget(scroll, stretch=1)

        # Input bar
        input_bar = QFrame()
        input_bar.setFixedHeight(58)
        input_bar.setStyleSheet(f"background: {C_PANEL}; border-top: 1px solid {C_BORDER};")
        i_layout = QHBoxLayout(input_bar)
        i_layout.setContentsMargins(12, 10, 12, 10)
        i_layout.setSpacing(8)

        self._chat_input = QLineEdit()
        self._chat_input.setPlaceholderText("Ask about this species...")
        self._chat_input.setStyleSheet(f"""
            QLineEdit {{
                background: {C_CARD};
                color: {C_TEXT};
                border: 1px solid {C_BORDER};
                border-radius: 18px;
                padding: 6px 16px;
                font-size: 13px;
            }}
            QLineEdit:focus {{ border: 1px solid {C_ACCENT}; }}
        """)
        self._chat_input.returnPressed.connect(self._on_chat_send)

        send_btn = QPushButton("➤")
        send_btn.setFixedSize(38, 38)
        send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        send_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C_ACCENT};
                color: {C_BG};
                border: none;
                border-radius: 19px;
                font-size: 14px;
                font-weight: 700;
            }}
            QPushButton:hover {{ background: #33ddff; }}
        """)
        send_btn.clicked.connect(self._on_chat_send)

        i_layout.addWidget(self._chat_input)
        i_layout.addWidget(send_btn)
        layout.addWidget(input_bar)

        return widget

    # ── Camera ─────────────────────────────────────────────────────────────────

    def _start_camera(self):
        self._camera_thread = CameraThread()
        self._camera_thread.frame_ready.connect(self._on_frame)
        self._camera_thread.start()

    def _on_frame(self, frame):
        self._last_frame = frame
        h, w, ch = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        lbl_size = self._camera_label.size()
        scaled = pixmap.scaled(
            lbl_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self._camera_label.setPixmap(scaled)

        # Keep overlay sized to match label widget
        if self._scan_overlay:
            self._scan_overlay.setGeometry(self._camera_label.geometry())

    # ── Model Loading ──────────────────────────────────────────────────────────

    def _load_models_async(self):
        def _load():
            try:
                # Load custom NC model first (faster than MobileNetV2 to load)
                self._custom_model, self._custom_label_map, self._custom_device = \
                    load_custom_model()

                # Always load MobileNetV2 as fallback
                self._model = MobileNetV2(weights="imagenet")

                self._client = OpenAI(
                    api_key=GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1",
                )
                self._models_loaded = True
                self.model_status_signal.emit("ready")
            except Exception as e:
                self.model_status_signal.emit(f"error:{e}")

        t = threading.Thread(target=_load, daemon=True)
        t.start()

    def _on_model_status(self, status: str):
        """Slot — always runs on the main Qt thread via signal."""
        if status == "ready":
            self._loading_lbl.setText("✓  Models ready")
            self._loading_lbl.setStyleSheet(f"color: {C_GREEN}; font-size: 11px;")
            QTimer.singleShot(2500, self._loading_lbl.hide)
        elif status.startswith("error:"):
            msg = status[6:]
            self._loading_lbl.setText(f"⚠  {msg[:40]}")
            self._loading_lbl.setStyleSheet(f"color: {C_RED}; font-size: 11px;")

    # ── Scan ───────────────────────────────────────────────────────────────────

    def _on_scan(self):
        if not self._models_loaded:
            self._status_lbl.setText("Still loading models...")
            return
        if self._last_frame is None:
            self._status_lbl.setText("No camera frame available")
            return

        frame = self._last_frame.copy()
        self._scan_btn.start_scanning()
        self._status_lbl.setText("Classifying + looking up species data...")
        if self._scan_overlay:
            self._scan_overlay.setGeometry(self._camera_label.geometry())
            self._scan_overlay.show()
            self._scan_overlay.start()

        self._analysis_worker = AnalysisWorker(
            frame, self._model, self._client,
            custom_model=self._custom_model,
            custom_label_map=self._custom_label_map,
            custom_device=self._custom_device,
        )
        self._analysis_worker.result_ready.connect(self._on_result)
        self._analysis_worker.error_occurred.connect(self._on_error)
        self._analysis_worker.start()

    def _on_result(self, result):
        self._scan_btn.stop_scanning()
        self._status_lbl.setText(f"Identified: {result['name']}")
        if self._scan_overlay:
            self._scan_overlay.stop()
            self._scan_overlay.hide()

        self._current_result = result
        self._chat_history = []
        self._collection.append(result)
        self._save_collection()
        self._add_collection_card(result, prepend=True)
        self._species_count_lbl.setText(f"{len(self._collection)} discovered")
        self._refresh_category_filter_options()
        self._apply_filters()
        self._check_achievements()

        self._render_entry(result)
        self._switch_tab(0)
        self._reset_chat()
        self._report_btn.setEnabled(True)

    def _on_error(self, msg):
        self._scan_btn.stop_scanning()
        self._status_lbl.setText(f"Error: {msg}")
        if self._scan_overlay:
            self._scan_overlay.stop()
            self._scan_overlay.hide()

    # ── Entry Rendering ────────────────────────────────────────────────────────

    def _clear_entry(self):
        while self._entry_inner.count():
            item = self._entry_inner.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _render_entry(self, result):
        self._clear_entry()
        entry = result.get("entry", {})

        # ── Name + confidence header
        name_frame = QFrame()
        name_frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {C_CARD}, stop:1 #172917);
                border: none;
                border-radius: 10px;
            }}
        """)
        nf_layout = QVBoxLayout(name_frame)
        nf_layout.setContentsMargins(16, 14, 16, 14)
        nf_layout.setSpacing(4)

        common = entry.get("common_name", result["name"])
        sci    = entry.get("scientific_name", "")
        cat    = entry.get("category", "")

        name_lbl = QLabel(common.upper())
        name_lbl.setStyleSheet(f"""
            color: {C_TEXT};
            font-size: 20px;
            font-weight: 900;
            letter-spacing: 2px;
        """)
        nf_layout.addWidget(name_lbl)

        if sci and sci != "Unknown":
            sci_lbl = QLabel(sci)
            sci_lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 12px; font-style: italic; letter-spacing: 0.5px;")
            nf_layout.addWidget(sci_lbl)

        # Confidence bar row
        conf = result["confidence"]
        conf_color = C_GREEN if conf >= 75 else C_YELLOW if conf >= 50 else C_RED
        conf_row = QHBoxLayout()
        conf_row.setSpacing(10)

        conf_lbl = QLabel(f"CONFIDENCE  {conf:.1f}%")
        conf_lbl.setStyleSheet(f"color: {conf_color}; font-size: 11px; font-weight: 700; letter-spacing: 1px;")
        conf_row.addWidget(conf_lbl)

        # Show which model identified this — custom NC model or MobileNetV2 fallback
        used_custom = result.get("used_custom_model", False)
        model_badge = QLabel("🌿 NatureDex Model" if used_custom else "⚙ MobileNetV2")
        model_badge.setStyleSheet(f"""
            color: {C_ACCENT if used_custom else C_SUBTEXT};
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 0.5px;
        """)

        if cat:
            cat_badge = QLabel(f"  {cat}  ")
            cat_badge.setStyleSheet(f"""
                background: {C_BORDER};
                color: {C_ACCENT};
                font-size: 10px;
                font-weight: 700;
                border-radius: 4px;
                padding: 2px 6px;
                letter-spacing: 1px;
            """)
            conf_row.addWidget(cat_badge)
        conf_row.addWidget(model_badge)
        conf_row.addStretch()
        nf_layout.addLayout(conf_row)

        # Rarity row — shown when iNaturalist data is available
        rarity = result.get("rarity", "")
        nc_obs = result.get("nc_observations", None)
        inat = result.get("inat", {})
        global_obs = inat.get("observations_count", 0)

        if rarity:
            rarity_row = QHBoxLayout()
            rarity_row.setSpacing(8)

            # Color-code rarity: rare = warm, common = cool
            if "Very Rare" in rarity or "Not Recorded" in rarity:
                rarity_color = C_PURPLE
            elif "Rare" in rarity:
                rarity_color = C_RED
            elif "Uncommon" in rarity:
                rarity_color = C_YELLOW
            else:
                rarity_color = C_ACCENT   # neon lime for common

            rarity_lbl = QLabel(f"◈  {rarity}")
            rarity_lbl.setStyleSheet(f"color: {rarity_color}; font-size: 11px; font-weight: 700;")
            rarity_row.addWidget(rarity_lbl)

            if global_obs:
                global_lbl = QLabel(f"·  {global_obs:,} global observations")
                global_lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 10px;")
                rarity_row.addWidget(global_lbl)

            rarity_row.addStretch()
            nf_layout.addLayout(rarity_row)

        # Type tags
        tags = entry.get("type_tags", [])
        if tags:
            tag_row = QHBoxLayout()
            tag_row.setSpacing(6)
            for tag in tags[:4]:
                t_lbl = QLabel(tag)
                t_lbl.setStyleSheet(f"""
                    background: #1a3a2a;
                    color: {C_GREEN};
                    font-size: 10px;
                    font-weight: 600;
                    border-radius: 3px;
                    padding: 2px 8px;
                """)
                tag_row.addWidget(t_lbl)
            tag_row.addStretch()
            nf_layout.addLayout(tag_row)

        self._entry_inner.addWidget(name_frame)

        # ── Description
        desc = entry.get("description", "")
        if desc:
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setStyleSheet(f"""
                color: {C_TEXT};
                font-size: 13px;
                line-height: 1.6;
                padding: 4px 0;
            """)
            self._entry_inner.addWidget(desc_lbl)

        # ── Info grid
        # Build the NC context string with real observation count appended
        nc_context = entry.get("north_carolina_context", "")
        if nc_obs is not None and nc_obs > 0 and nc_context and nc_context not in ("Unknown", "N/A"):
            nc_context = f"{nc_context}  ({nc_obs:,} research-grade iNaturalist observations in NC)"
        elif nc_obs == 0 and nc_context not in ("Unknown", "N/A", ""):
            nc_context = f"{nc_context}  (No research-grade iNaturalist observations recorded in NC)"

        fields = [
            ("🌿  HABITAT",        entry.get("habitat", "")),
            ("🍃  DIET",           entry.get("diet", "")),
            ("🐾  BEHAVIOR",       entry.get("behavior", "")),
            ("🔴  CONSERVATION",   entry.get("conservation_status", "")),
            ("📍  NORTH CAROLINA", nc_context),
            ("⚡  FUN FACT",       entry.get("fun_fact", "")),
        ]
        for icon_label, value in fields:
            if value and value not in ("Unknown", "N/A", ""):
                card = self._make_info_card(icon_label, value)
                self._entry_inner.addWidget(card)

        # ── Alternatives
        alts = result.get("alternatives", [])
        if alts:
            alt_frame = QFrame()
            alt_frame.setStyleSheet(f"""
                QFrame {{
                    background: {C_CARD};
                    border: 1px solid {C_BORDER};
                    border-radius: 8px;
                }}
            """)
            alt_layout = QVBoxLayout(alt_frame)
            alt_layout.setContentsMargins(14, 12, 14, 12)
            alt_layout.setSpacing(6)

            alt_title = QLabel("OTHER POSSIBILITIES")
            alt_title.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 10px; font-weight: 700; letter-spacing: 2px;")
            alt_layout.addWidget(alt_title)

            for a in alts:
                a_conf = a["confidence"]
                a_color = C_GREEN if a_conf >= 20 else C_SUBTEXT
                row = QHBoxLayout()
                n_lbl = QLabel(f"• {a['name']}")
                n_lbl.setStyleSheet(f"color: {C_TEXT}; font-size: 12px;")
                c_lbl = QLabel(f"{a_conf:.1f}%")
                c_lbl.setStyleSheet(f"color: {a_color}; font-size: 11px; font-weight: 600;")
                row.addWidget(n_lbl)
                row.addStretch()
                row.addWidget(c_lbl)
                alt_layout.addLayout(row)

            self._entry_inner.addWidget(alt_frame)

        self._entry_inner.addStretch()

    def _make_info_card(self, label, value):
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background: {C_CARD};
                border: none;
                border-left: 3px solid {C_ACCENT};
                border-radius: 6px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 9, 14, 9)
        layout.setSpacing(3)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {C_ACCENT}; font-size: 9px; font-weight: 800; letter-spacing: 1.5px;")
        val = QLabel(value)
        val.setWordWrap(True)
        val.setStyleSheet(f"color: {C_TEXT}; font-size: 12px;")

        layout.addWidget(lbl)
        layout.addWidget(val)
        return card

    # ── Collection ─────────────────────────────────────────────────────────────

    def _add_collection_card(self, entry_data, prepend=True):
        card = CollectionCard(entry_data)
        card.clicked_signal.connect(self._on_collection_click)
        card.delete_signal.connect(self._on_delete_entry)
        if prepend:
            self._collection_layout.insertWidget(0, card)
        else:
            count = self._collection_layout.count()
            self._collection_layout.insertWidget(count - 1, card)

    def _on_collection_click(self, entry_data):
        self._current_result = entry_data
        self._chat_history = []
        self._render_entry(entry_data)
        self._switch_tab(0)
        self._reset_chat()
        self._report_btn.setEnabled(True)

    def _on_delete_entry(self, timestamp):
        if not timestamp:
            return
        # Remove from the underlying collection list
        self._collection = [e for e in self._collection if e.get("timestamp") != timestamp]
        self._save_collection()

        # Remove the matching card widget from the sidebar
        for i in range(self._collection_layout.count()):
            item = self._collection_layout.itemAt(i)
            widget = item.widget() if item else None
            if isinstance(widget, CollectionCard) and widget.entry_data.get("timestamp") == timestamp:
                widget.deleteLater()
                break

        self._species_count_lbl.setText(f"◈  {len(self._collection)} discovered")
        self._refresh_category_filter_options()

        # If the deleted entry was being shown in the main panel, clear it
        if self._current_result and self._current_result.get("timestamp") == timestamp:
            self._current_result = None
            self._clear_entry()
            self._placeholder_lbl = QLabel("Scan an object to generate\na NatureDex entry.")
            self._placeholder_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._placeholder_lbl.setStyleSheet(f"""
                color: {C_SUBTEXT};
                font-size: 15px;
                line-height: 1.8;
                padding: 60px 20px;
            """)
            self._entry_inner.addWidget(self._placeholder_lbl)
            self._entry_inner.addStretch()
            self._reset_chat()

    # ── Search & Filter ────────────────────────────────────────────────────────

    def _refresh_category_filter_options(self):
        """Repopulate the category dropdown with whatever categories actually
        exist in the collection, preserving the current selection if possible."""
        current = self._category_filter.currentText()
        categories = sorted({
            e.get("entry", {}).get("category", "Unknown") or "Unknown"
            for e in self._collection
        })

        self._category_filter.blockSignals(True)
        self._category_filter.clear()
        self._category_filter.addItem("All Categories")
        self._category_filter.addItems(categories)
        idx = self._category_filter.findText(current)
        self._category_filter.setCurrentIndex(idx if idx >= 0 else 0)
        self._category_filter.blockSignals(False)

    def _apply_filters(self):
        """Show/hide collection cards based on the current search text and
        category selection. Hiding (rather than rebuilding) keeps this fast
        and avoids losing scroll position while typing."""
        query = self._search_input.text().strip().lower()
        category = self._category_filter.currentText()

        visible_count = 0
        for i in range(self._collection_layout.count()):
            item = self._collection_layout.itemAt(i)
            widget = item.widget() if item else None
            if not isinstance(widget, CollectionCard):
                continue

            matches_search = (query in widget._search_text) if query else True
            matches_category = (category == "All Categories") or (widget._category == category)
            should_show = matches_search and matches_category

            widget.setVisible(should_show)
            if should_show:
                visible_count += 1

        self._empty_filter_lbl.setVisible(visible_count == 0 and len(self._collection) > 0)

    # ── Achievements ───────────────────────────────────────────────────────────

    def _load_achievements(self):
        if ACHIEVEMENTS_FILE.exists():
            try:
                return set(json.loads(ACHIEVEMENTS_FILE.read_text()))
            except Exception:
                return set()
        return set()

    def _save_achievements(self):
        try:
            ACHIEVEMENTS_FILE.write_text(json.dumps(sorted(self._unlocked_achievements)))
        except Exception as e:
            print(f"Could not save achievements: {e}")

    def _check_achievements(self):
        """Compare current collection stats against achievement thresholds.
        Any newly-crossed thresholds get unlocked and (the first one) toasted."""
        count = len(self._collection)
        categories = {
            e.get("entry", {}).get("category", "Unknown") or "Unknown"
            for e in self._collection
        }
        num_categories = len(categories)

        newly_unlocked = []
        for badge in ACHIEVEMENTS:
            if badge["id"] in self._unlocked_achievements:
                continue
            if badge["type"] == "count" and count >= badge["threshold"]:
                newly_unlocked.append(badge)
            elif badge["type"] == "category" and num_categories >= badge["threshold"]:
                newly_unlocked.append(badge)

        if not newly_unlocked:
            return

        for badge in newly_unlocked:
            self._unlocked_achievements.add(badge["id"])
        self._save_achievements()

        # Toast the first new badge now; if multiple unlocked at once, queue
        # the rest with a slight delay so they don't all overlap visually.
        for idx, badge in enumerate(newly_unlocked):
            QTimer.singleShot(idx * 3500, lambda b=badge: self._toast.show_achievement(b["icon"], b["name"]))

    def _show_badges_panel(self):
        panel = QFrame(self)
        panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 12px;")
        panel.setFixedSize(360, 460)
        panel.move((self.width() - 360) // 2, (self.height() - 460) // 2)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(0)

        # Header — title + close, no box
        header_row = QHBoxLayout()
        title = QLabel("Achievements")
        title.setStyleSheet(f"color: {C_TEXT}; font-size: 16px; font-weight: 800;")
        close_btn = QLabel("✕")
        close_btn.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 16px;")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.mousePressEvent = lambda e: panel.deleteLater()
        header_row.addWidget(title)
        header_row.addStretch()
        header_row.addWidget(close_btn)
        layout.addLayout(header_row)
        layout.addSpacing(14)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        c_layout = QVBoxLayout(content)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(0)

        for badge in ACHIEVEMENTS:
            unlocked = badge["id"] in self._unlocked_achievements

            row = QWidget()
            row.setStyleSheet("background: transparent;")
            r_layout = QHBoxLayout(row)
            r_layout.setContentsMargins(0, 8, 0, 8)
            r_layout.setSpacing(14)

            icon = QLabel(badge["icon"] if unlocked else "🔒")
            icon.setStyleSheet(f"font-size: 18px; color: {'inherit' if unlocked else C_SUBTEXT};")
            icon.setFixedWidth(26)
            r_layout.addWidget(icon)

            text_box = QVBoxLayout()
            text_box.setSpacing(2)
            name_lbl = QLabel(badge["name"])
            name_lbl.setStyleSheet(
                f"color: {C_TEXT}; font-size: 12px; font-weight: 700;"
                if unlocked else
                f"color: {C_SUBTEXT}; font-size: 12px; font-weight: 600;"
            )
            desc_lbl = QLabel(badge["desc"])
            desc_lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 10px;")
            text_box.addWidget(name_lbl)
            text_box.addWidget(desc_lbl)
            r_layout.addLayout(text_box)
            r_layout.addStretch()

            if unlocked:
                check = QLabel("✓")
                check.setStyleSheet(f"color: {C_ACCENT}; font-size: 14px; font-weight: 800;")
                r_layout.addWidget(check)

            c_layout.addWidget(row)

            # Hairline divider between rows — one pixel, barely visible
            if badge != ACHIEVEMENTS[-1]:
                line = QFrame()
                line.setFixedHeight(1)
                line.setStyleSheet(f"background: {C_BORDER};")
                c_layout.addWidget(line)

        c_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)

        panel.show()
        panel.raise_()

    # ── Chat ───────────────────────────────────────────────────────────────────

    def _reset_chat(self):
        while self._chat_inner.count():
            item = self._chat_inner.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self._current_result:
            entry = self._current_result.get("entry", {})
            name = entry.get("common_name", self._current_result.get("name", "this organism"))
            intro = QLabel(f'Ask me anything about\n"{name}"')
            intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
            intro.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 13px; padding: 30px 0;")
            self._chat_inner.addWidget(intro)
        else:
            intro = QLabel("Scan something first to\nstart a conversation.")
            intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
            intro.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 13px; padding: 40px 0;")
            self._chat_inner.addWidget(intro)

        self._chat_inner.addStretch()

    def _on_chat_send(self):
        text = self._chat_input.text().strip()
        if not text or not self._current_result:
            if not self._current_result:
                self._add_chat_bubble("Scan something first, then ask me about it!", is_user=False)
            return
        if not self._client:
            self._add_chat_bubble("AI model is still loading — try again in a moment.", is_user=False)
            return
        self._chat_input.clear()

        # User bubble
        self._add_chat_bubble(text, is_user=True)

        # Build system context
        entry = self._current_result.get("entry", {})
        system = f"""You are NatureDex AI, a knowledgeable wildlife educator.
The user has just scanned: {self._current_result.get('name', 'an organism')}.
Here is what you know about it:
{json.dumps(entry, indent=2)}

Answer questions in an engaging, educational tone — like a Pokédex that can converse.
Keep responses concise (2-4 sentences). Focus on the organism or object scanned.
If asked about North Carolina specifically, provide NC-relevant context."""

        messages = [{"role": "system", "content": system}]
        messages += self._chat_history
        messages.append({"role": "user", "content": text})

        thinking = self._add_chat_bubble("Thinking...", is_user=False)

        # Store worker on self so it isn't garbage-collected while the thread
        # is still running — that was causing the window to crash and close.
        self._chat_worker = ChatWorker(self._client, messages)

        def on_reply(reply):
            # Guard against the thinking label having been deleted (e.g. if
            # the user deleted the entry or cleared the chat mid-request).
            try:
                thinking.setText(reply)
            except RuntimeError:
                pass  # widget was deleted before reply came back — safe to ignore
            self._chat_history.append({"role": "user", "content": text})
            self._chat_history.append({"role": "assistant", "content": reply})

        self._chat_worker.reply_ready.connect(on_reply)
        self._chat_worker.start()

    def _add_chat_bubble(self, text, is_user):
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setMaximumWidth(320)
        if is_user:
            bubble.setStyleSheet(f"""
                background: {C_ACCENT};
                color: {C_BG};
                border-radius: 14px;
                padding: 8px 14px;
                font-size: 13px;
                font-weight: 500;
            """)
            align = Qt.AlignmentFlag.AlignRight
        else:
            bubble.setStyleSheet(f"""
                background: {C_CARD};
                color: {C_TEXT};
                border: 1px solid {C_BORDER};
                border-radius: 14px;
                padding: 8px 14px;
                font-size: 13px;
            """)
            align = Qt.AlignmentFlag.AlignLeft

        wrapper = QWidget()
        w_layout = QHBoxLayout(wrapper)
        w_layout.setContentsMargins(0, 0, 0, 0)
        if is_user:
            w_layout.addStretch()
        w_layout.addWidget(bubble)
        if not is_user:
            w_layout.addStretch()

        count = self._chat_inner.count()
        self._chat_inner.insertWidget(count - 1, wrapper)

        # Scroll to bottom
        QTimer.singleShot(50, lambda: self._chat_scroll.verticalScrollBar().setValue(
            self._chat_scroll.verticalScrollBar().maximum()
        ))
        return bubble

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_collection(self):
        if COLLECTION_FILE.exists():
            try:
                return json.loads(COLLECTION_FILE.read_text())
            except Exception:
                return []
        return []

    def _save_collection(self):
        try:
            COLLECTION_FILE.write_text(json.dumps(self._collection, indent=2))
        except Exception as e:
            print(f"Could not save collection: {e}")

    # ── Correction / Feedback System ───────────────────────────────────────────

    def _on_report_wrong_id(self):
        if not self._current_result:
            return
        self._correction_input.clear()
        self._correction_input.setPlaceholderText("e.g. Eastern Bluebird")
        self._correction_bar.show()
        self._correction_input.setFocus()

    def _on_correction_submit(self):
        text = self._correction_input.text().strip()
        if not text:
            self._correction_input.setPlaceholderText("Please enter a name")
            return
        self._save_correction(text)
        self._correction_bar.hide()
        self._correction_input.clear()
        self._report_btn.setText("⚑  Thanks for the correction!")
        QTimer.singleShot(3000, lambda: self._report_btn.setText("⚑  Wrong ID? Report it"))

    def _on_correction_cancel(self):
        self._correction_bar.hide()
        self._correction_input.clear()

    def _save_correction(self, correct_name: str):
        """Append one correction record to the local corrections log.
        Each record has everything needed for Phase 4 model training:
        the original model label, user's correction, confidence, and timestamp."""
        result = self._current_result or {}
        entry = result.get("entry", {})

        record = {
            "timestamp":        datetime.datetime.now().isoformat(),
            "original_label":   result.get("raw_label", ""),
            "original_name":    result.get("name", ""),
            "confidence":       result.get("confidence", 0),
            "correct_name":     correct_name,
            "scientific_name":  entry.get("scientific_name", ""),
            "category":         entry.get("category", ""),
            "inat_taxon_id":    result.get("inat", {}).get("taxon_id"),
            "image_path":       result.get("image_path", ""),
        }

        try:
            corrections = []
            if CORRECTIONS_FILE.exists():
                corrections = json.loads(CORRECTIONS_FILE.read_text())
            corrections.append(record)
            CORRECTIONS_FILE.write_text(json.dumps(corrections, indent=2))
            print(f"[Correction saved] '{record['original_name']}' → '{correct_name}'")
        except Exception as e:
            print(f"Could not save correction: {e}")

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._toast:
            self._toast.move(self.width() - 300, 70)

    def closeEvent(self, event):
        if self._camera_thread:
            self._camera_thread.stop()
        super().closeEvent(event)


# ─── Entry Point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("NatureDex AI")
    win = NatureDexWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()