"""
NatureDex AI — Congressional App Challenge
An AI-powered wildlife identification and learning platform.
"""

import sys
import os
import json
import datetime
import threading
import subprocess
import struct
import wave
import tempfile
import math
import urllib.request
import urllib.parse
import http.server
import socket
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QTextEdit, QLineEdit,
    QSplitter, QStackedWidget, QGraphicsOpacityEffect, QComboBox,
    QMenu, QGraphicsDropShadowEffect, QProgressBar, QLayout, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation,
    QEasingCurve, QRect, QSize, pyqtProperty, QObject, QPoint
)
from PyQt6.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush,
    QLinearGradient, QPalette, QFontDatabase, QIcon
)
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEngineSettings
    _HAS_WEBENGINE = True
except ImportError:
    _HAS_WEBENGINE = False

import cv2
from openai import OpenAI

import torch
import torch.nn.functional as F
from torchvision import transforms, models as torch_models
from PIL import Image as PILImage

# ─── Constants ────────────────────────────────────────────────────────────────

GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
COLLECTION_FILE   = Path.home() / ".naturedex_collection.json"
ACHIEVEMENTS_FILE = Path.home() / ".naturedex_achievements.json"
ONBOARD_FILE      = Path.home() / ".naturedex_onboarded"
CHALLENGES_FILE   = Path.home() / ".naturedex_challenges.json"


def _species_group(entry_dict: dict) -> str:
    """Map an entry's taxonomic group to a simple category name."""
    g = (entry_dict.get("entry", {}).get("iconic_group", "")
         or entry_dict.get("entry", {}).get("category", "") or "").lower()
    if "av" in g or "bird" in g:                     return "bird"
    if "mamm" in g or "beast" in g:                  return "mammal"
    if "insect" in g or "arachn" in g or "bug" in g: return "insect"
    if "plant" in g or "flora" in g:                 return "plant"
    if "rept" in g or "scale" in g:                  return "reptile"
    if "amphib" in g:                                return "amphibian"
    if "fung" in g:                                  return "fungus"
    return ""


def generate_daily_challenges(date_str: str, has_location: bool = True) -> list:
    """Deterministically pick 3 mixed challenges for a given date."""
    import random
    rng = random.Random(int(date_str.replace("-", "")))
    cats = [("bird", "a bird"), ("mammal", "a mammal"), ("insect", "an insect"),
            ("plant", "a plant"), ("reptile", "a reptile"), ("fungus", "a fungus")]
    cat = rng.choice(cats)
    count_target = rng.choice([2, 3])
    category_ch = {"id": f"cat_{cat[0]}", "type": "category", "param": cat[0],
                   "target": 1, "icon": "🎯", "text": f"Scan {cat[1]} today"}
    pool = [
        {"id": "rarity",  "type": "rarity",  "param": "", "target": 1, "icon": "💎",
         "text": "Find something Uncommon or rarer"},
        {"id": "count",   "type": "count",   "param": "", "target": count_target, "icon": "🔢",
         "text": f"Make {count_target} discoveries today"},
        {"id": "new",     "type": "new",     "param": "", "target": 1, "icon": "✨",
         "text": "Scan a species new to your collection"},
        {"id": "sticker", "type": "sticker", "param": "", "target": 1, "icon": "🎴",
         "text": "Create a new sticker"},
    ]
    if has_location:
        pool.append({"id": "native", "type": "native", "param": "", "target": 1,
                     "icon": "📍", "text": "Find a species native to your area"})
    rng.shuffle(pool)
    return [category_ch] + pool[:2]


def challenge_progress(ch: dict, today_scans: list, all_coll: list, today_str: str) -> int:
    t, p = ch["type"], ch.get("param", "")
    if t == "category":
        return sum(1 for e in today_scans if _species_group(e) == p)
    if t == "rarity":
        return sum(1 for e in today_scans
                   if any(w in (e.get("local_rarity", "") or "")
                          for w in ("Uncommon near", "Rare near", "Very Rare near")))
    if t == "count":
        return len(today_scans)
    if t == "native":
        return sum(1 for e in today_scans if e.get("native_nc"))
    if t == "sticker":
        return sum(1 for e in today_scans if e.get("sticker_path"))
    if t == "new":
        cnt = 0
        for e in today_scans:
            nm = (e.get("name", "") or "").lower()
            earlier = any((o.get("name", "") or "").lower() == nm
                          and (o.get("timestamp", "") or "")[:10] < today_str
                          for o in all_coll)
            if not earlier:
                cnt += 1
        return cnt
    return 0
CORRECTIONS_FILE  = Path.home() / ".naturedex_corrections.json"

_SCRIPT_DIR           = Path(__file__).parent
CUSTOM_MODEL_PTH      = _SCRIPT_DIR / "models" / "naturedex_nc_v1.pth"
CUSTOM_MODEL_THRESHOLD = 60.0
BIOCLIP_TOPK           = 5      # how many candidates BioCLIP returns per scan
LOW_CONFIDENCE_THRESHOLD = 10.0 # below this %, flag the ID as uncertain

INVASIVE_REPORTS_FILE = Path.home() / ".naturedex_invasive_reports.json"

C_BG         = "#1a1f14"
C_PANEL      = "#222918"
C_CARD       = "#2a3320"
C_BORDER     = "#3a4a2a"
C_ACCENT     = "#e8720c"
C_ACCENT_DIM = "#c45e08"
C_TEXT       = "#f0ead8"
C_SUBTEXT    = "#7a9060"
C_GREEN      = "#6abf5e"
C_YELLOW     = "#d4a017"
C_RED        = "#c0392b"
C_PURPLE     = "#8e5fb5"
C_GOLD       = "#e8a020"
C_PINK       = "#c45e8a"
C_ACCENT2    = C_ACCENT_DIM
C_SCREEN     = "#0f1409"
C_SCAN_LINE  = C_ACCENT

GROQ_MODEL = "openai/gpt-oss-120b"   # Groq deprecated llama-3.3-70b-versatile (Jun 2026)

# Cesium Ion token — only used for the star-field skybox assets, not imagery.
# Imagery is served locally through the proxy below (no Ion dependency).
CESIUM_TOKEN = os.getenv("CESIUM_ION_TOKEN", "")

ACHIEVEMENTS = [
    # ── Discovery count ────────────────────────────────────────────────────────
    {"id": "first_scan", "type": "count",    "threshold": 1,   "icon": "🔍", "name": "First Discovery",      "desc": "Scan your first species"},
    {"id": "count_5",    "type": "count",    "threshold": 5,   "icon": "🌱", "name": "Budding Naturalist",   "desc": "Discover 5 species"},
    {"id": "count_10",   "type": "count",    "threshold": 10,  "icon": "🌿", "name": "Field Explorer",       "desc": "Discover 10 species"},
    {"id": "count_25",   "type": "count",    "threshold": 25,  "icon": "🌳", "name": "Wildlife Tracker",     "desc": "Discover 25 species"},
    {"id": "count_50",   "type": "count",    "threshold": 50,  "icon": "🦅", "name": "Master Naturalist",    "desc": "Discover 50 species"},
    {"id": "count_100",  "type": "count",    "threshold": 100, "icon": "🏆", "name": "NatureDex Legend",     "desc": "Discover 100 species"},

    # ── Category diversity ─────────────────────────────────────────────────────
    {"id": "cats_3",     "type": "category", "threshold": 3,   "icon": "🎯", "name": "Well-Rounded",         "desc": "Find 3 different categories"},
    {"id": "cats_5",     "type": "category", "threshold": 5,   "icon": "🧭", "name": "Diverse Explorer",     "desc": "Find 5 different categories"},
    {"id": "cats_7",     "type": "category", "threshold": 7,   "icon": "🌍", "name": "Renaissance Scout",    "desc": "Find 7 different categories"},
    {"id": "sweep",      "type": "sweep",    "threshold": 4,   "icon": "🃏", "name": "Full House",           "desc": "Find a bird, mammal, insect, AND plant"},

    # ── Rarity finds ───────────────────────────────────────────────────────────
    {"id": "rare_1",     "type": "rarity",   "threshold": 1,   "icon": "💎", "name": "Rare Find",            "desc": "Discover a Rare or Very Rare species near you"},
    {"id": "rare_5",     "type": "rarity",   "threshold": 5,   "icon": "🔮", "name": "Rarity Hunter",        "desc": "Discover 5 Rare or Very Rare species near you"},

    # ── Conservation ───────────────────────────────────────────────────────────
    {"id": "endanger_1", "type": "endangered","threshold": 1,  "icon": "🚨", "name": "Conservationist",      "desc": "Scan a threatened species"},
    {"id": "endanger_3", "type": "endangered","threshold": 3,  "icon": "🛡️", "name": "Species Guardian",     "desc": "Find 3 threatened species"},

    # ── Local to your area ─────────────────────────────────────────────────────
    {"id": "local_1",    "type": "local",    "threshold": 1,   "icon": "📍", "name": "Local Spotter",        "desc": "Find a species native to your area"},
    {"id": "local_5",    "type": "local",    "threshold": 5,   "icon": "🏡", "name": "Neighborhood Naturalist","desc": "Find 5 species native to your area"},
    {"id": "local_15",   "type": "local",    "threshold": 15,  "icon": "🗺️", "name": "Local Legend",         "desc": "Find 15 species native to your area"},

    # ── Sticker collection ─────────────────────────────────────────────────────
    {"id": "sticker_3",  "type": "sticker",  "threshold": 3,   "icon": "✨", "name": "Sticker Starter",      "desc": "Collect 3 cut-out stickers"},
    {"id": "sticker_15", "type": "sticker",  "threshold": 15,  "icon": "🎴", "name": "Sticker Collector",    "desc": "Collect 15 cut-out stickers"},

    # ── Streaks (different days) ────────────────────────────────────────────────
    {"id": "streak_3",   "type": "streak",   "threshold": 3,   "icon": "📅", "name": "Daily Explorer",       "desc": "Scan on 3 different days"},
    {"id": "streak_7",   "type": "streak",   "threshold": 7,   "icon": "🔥", "name": "Devoted Naturalist",   "desc": "Scan on 7 different days"},

    # ── Skill & travel ─────────────────────────────────────────────────────────
    {"id": "highconf",   "type": "highconf", "threshold": 1,   "icon": "✅", "name": "Crystal Clear",        "desc": "Get a high-confidence identification"},
    {"id": "region_3",   "type": "region",   "threshold": 3,   "icon": "🧳", "name": "Traveler",             "desc": "Discover species in 3 different regions"},

    # ── Corrections ────────────────────────────────────────────────────────────
    {"id": "correct_1",  "type": "correct",  "threshold": 1,   "icon": "✏️", "name": "Fact Checker",         "desc": "Submit your first correction"},
    {"id": "correct_5",  "type": "correct",  "threshold": 5,   "icon": "📚", "name": "Data Contributor",     "desc": "Submit 5 corrections"},
]

NC_PLACE_ID = 51

# ─── Sound Helpers ────────────────────────────────────────────────────────────
# Generates WAV files using stdlib only, plays via macOS afplay.
# No extra pip installs needed.

def _write_wav(freqs, duration, volume=0.22, sample_rate=44100) -> str:
    """Generate a chord WAV and return its temp file path."""
    n = int(sample_rate * duration)
    fade = int(sample_rate * 0.04)
    buf = []
    for i in range(n):
        v = sum(math.sin(2 * math.pi * f * i / sample_rate) for f in freqs)
        v = v / len(freqs) * volume
        if i < fade:       v *= i / fade
        elif i > n - fade: v *= (n - i) / fade
        buf.append(max(-32767, min(32767, int(v * 32767))))
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *buf))
    return tmp.name

# Pre-generate once at startup
_WAV_SCAN    = _write_wav([880], 0.08, 0.28)           # short beep
_WAV_SUCCESS = _write_wav([523, 659, 784], 0.35, 0.20)  # C-E-G chord
_WAV_BOOT    = _write_wav([261, 329, 392, 523], 0.55, 0.16)  # warm chord
_WAV_UNLOCK  = _write_wav([659, 831, 988, 1319], 0.55, 0.22)  # bright fanfare chord
_WAV_AMBIENT = _write_wav([131, 165, 196, 262], 8.0, 0.10)    # soft low pad, looped

# Sound preferences (persisted)
SOUND_CONFIG_FILE = Path.home() / ".naturedex_sound.json"
_SOUND_ENABLED = True
_AMBIENT_ON    = False
_ambient_proc  = None

def _load_sound_prefs():
    global _SOUND_ENABLED, _AMBIENT_ON
    try:
        if SOUND_CONFIG_FILE.exists():
            d = json.loads(SOUND_CONFIG_FILE.read_text())
            _SOUND_ENABLED = d.get("enabled", True)
            _AMBIENT_ON    = d.get("ambient", False)
    except Exception:
        pass

def _save_sound_prefs():
    try:
        SOUND_CONFIG_FILE.write_text(
            json.dumps({"enabled": _SOUND_ENABLED, "ambient": _AMBIENT_ON}))
    except Exception:
        pass

def _play(path: str):
    """Play a WAV file non-blocking via afplay (macOS built-in)."""
    if not _SOUND_ENABLED:
        return
    try:
        subprocess.Popen(["afplay", path],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except Exception:
        pass  # silently skip if afplay not available

def _start_ambient():
    """Loop the ambient pad in a background thread while ambient is on."""
    global _ambient_proc
    _stop_ambient()
    def _loop():
        global _ambient_proc
        while _AMBIENT_ON and _SOUND_ENABLED:
            try:
                _ambient_proc = subprocess.Popen(
                    ["afplay", _WAV_AMBIENT],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                _ambient_proc.wait()
            except Exception:
                return
    threading.Thread(target=_loop, daemon=True).start()

def _stop_ambient():
    global _ambient_proc
    if _ambient_proc:
        try:
            _ambient_proc.terminate()
        except Exception:
            pass
        _ambient_proc = None

_load_sound_prefs()

# ─── iNaturalist API Lookup ───────────────────────────────────────────────────

def inat_lookup(label: str) -> dict:
    try:
        params = urllib.parse.urlencode({
            "q": label, "per_page": 1,
            "rank": "species,genus,family", "is_active": "true",
        })
        req = urllib.request.Request(
            f"https://api.inaturalist.org/v1/taxa?{params}",
            headers={"User-Agent": "NatureDexAI/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())
        results = data.get("results", [])
        if not results:
            return {}
        taxon = results[0]; taxon_id = taxon.get("id")
        inat_data = {
            "taxon_id":           taxon_id,
            "scientific_name":    taxon.get("name", ""),
            "common_name":        taxon.get("preferred_common_name", ""),
            "rank":               taxon.get("rank", ""),
            "iconic_taxon":       taxon.get("iconic_taxon_name", ""),
            "conservation_status":_parse_conservation(taxon),
            "wikipedia_summary":  taxon.get("wikipedia_summary", ""),
            "observations_count": taxon.get("observations_count", 0),
        }
        if taxon_id:
            inat_data["nc_observations"] = _get_nc_count(taxon_id)
        return inat_data
    except Exception as e:
        print(f"[iNat lookup] {e}")
        return {}

def _parse_conservation(taxon: dict) -> str:
    cs = taxon.get("conservation_status", {})
    if isinstance(cs, dict):
        name = cs.get("status_name", "")
        if name:
            return name.replace("_", " ").title()
    iucn = taxon.get("iucn_status_name", "")
    if iucn:
        return iucn.replace("_", " ").title()
    return ""

def _get_nc_count(taxon_id: int) -> int:
    try:
        params = urllib.parse.urlencode({
            "taxon_id": taxon_id, "place_id": NC_PLACE_ID,
            "quality_grade": "research", "per_page": 0,
        })
        req = urllib.request.Request(
            f"https://api.inaturalist.org/v1/observations?{params}",
            headers={"User-Agent": "NatureDexAI/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode()).get("total_results", 0)
    except Exception:
        return 0

def _nc_rarity_label(nc_count: int) -> str:
    if nc_count == 0:      return "Not Recorded in NC"
    elif nc_count < 10:    return "Very Rare in NC"
    elif nc_count < 100:   return "Rare in NC"
    elif nc_count < 1000:  return "Uncommon in NC"
    elif nc_count < 10000: return "Common in NC"
    else:                  return "Very Common in NC"


def _get_local_count(taxon_id: int, lat: float, lng: float, radius_km: int = 150) -> int:
    """Count research-grade observations of a taxon within radius_km of the user.
    Works anywhere on Earth — no place_id needed."""
    try:
        params = urllib.parse.urlencode({
            "taxon_id": taxon_id, "lat": lat, "lng": lng, "radius": radius_km,
            "quality_grade": "research", "per_page": 0,
        })
        req = urllib.request.Request(
            f"https://api.inaturalist.org/v1/observations?{params}",
            headers={"User-Agent": "NatureDexAI/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            return json.loads(resp.read().decode()).get("total_results", 0)
    except Exception:
        return 0


def _local_rarity_label(count: int, area: str = "your area") -> str:
    if count == 0:       return f"Not Recorded near {area}"
    elif count < 10:     return f"Very Rare near {area}"
    elif count < 100:    return f"Rare near {area}"
    elif count < 1000:   return f"Uncommon near {area}"
    elif count < 10000:  return f"Common near {area}"
    else:                return f"Very Common near {area}"


def get_nearby_species(lat: float, lng: float, n: int = 24, radius_km: int = 150) -> list:
    """Most-observed species near a location — works anywhere on Earth.
    Powers the location-scoped discovery index."""
    species, page = [], 1
    try:
        while len(species) < n and page <= 3:
            params = urllib.parse.urlencode({
                "lat": lat, "lng": lng, "radius": radius_km,
                "quality_grade": "research", "photos": "true",
                "per_page": 100, "page": page,
            })
            req = urllib.request.Request(
                f"https://api.inaturalist.org/v1/observations/species_counts?{params}",
                headers={"User-Agent": "NatureDexAI/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())
            results = data.get("results", [])
            if not results:
                break
            for row in results:
                t = row.get("taxon") or {}
                if t.get("rank") != "species":
                    continue
                species.append({
                    "taxon_id":        t.get("id"),
                    "common_name":     t.get("preferred_common_name") or t.get("name", ""),
                    "scientific_name": t.get("name", ""),
                    "iconic":          t.get("iconic_taxon_name", ""),
                    "count":           row.get("count", 0),
                })
                if len(species) >= n:
                    break
            page += 1
    except Exception as e:
        print(f"[Nearby] {e}")
    return species


class NearbyWorker(QThread):
    done = pyqtSignal(list)

    def __init__(self, lat, lng):
        super().__init__()
        self.lat, self.lng = lat, lng

    def run(self):
        self.done.emit(get_nearby_species(self.lat, self.lng))


# ─── Sticker Generation (cut-out) ─────────────────────────────────────────────
# Turns a scan photo into a transparent-background "sticker" using U^2-Net via
# onnxruntime. onnxruntime has clean macOS wheels (no compiler, no numba) and
# works with numpy<2, so it coexists with TensorFlow. The ~5MB model downloads
# once to ~/.naturedex_models/.
#   pip install onnxruntime

_U2NET_SESSION = None
_U2NET_PATH = Path.home() / ".naturedex_models" / "u2netp.onnx"
# Lightweight (~5MB) model. For higher quality, swap "u2netp.onnx" -> "u2net.onnx"
# in both the filename and URL below (that model is ~176MB).
_U2NET_URL  = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"

def _get_u2net():
    global _U2NET_SESSION
    if _U2NET_SESSION is None:
        import onnxruntime as ort
        if not _U2NET_PATH.exists():
            _U2NET_PATH.parent.mkdir(parents=True, exist_ok=True)
            print("[Sticker] downloading cut-out model (one time, ~5MB)…")
            urllib.request.urlretrieve(_U2NET_URL, _U2NET_PATH)
        _U2NET_SESSION = ort.InferenceSession(
            str(_U2NET_PATH), providers=["CPUExecutionProvider"])
    return _U2NET_SESSION


def make_sticker(input_path: str, output_path: str) -> str:
    """Cut the subject out of a photo → transparent PNG. Returns path or ""."""
    try:
        sess = _get_u2net()
        orig = PILImage.open(input_path).convert("RGB")

        # Preprocess: 320x320, scale, ImageNet-style normalize, CHW batch
        inp = orig.resize((320, 320), PILImage.LANCZOS)
        arr = np.array(inp).astype(np.float32)
        arr = arr / (arr.max() if arr.max() > 0 else 1.0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)

        # Inference → saliency mask, then normalize 0..1
        out = sess.run(None, {sess.get_inputs()[0].name: arr})[0]
        mask = out[0, 0]
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        # Resize mask back to the original and apply it as the alpha channel
        mask_img = PILImage.fromarray((mask * 255).astype("uint8")).resize(
            orig.size, PILImage.LANCZOS)
        rgba = orig.convert("RGBA")
        rgba.putalpha(mask_img)
        rgba.save(output_path)
        return output_path
    except Exception as e:
        print(f"[Sticker] cut-out unavailable/failed ({e}) — using raw photo")
        return ""


# ─── Shareable Finds Card ─────────────────────────────────────────────────────
# Composes a square-ish PNG a user can post to Instagram/etc. Pure PIL — no deps.

def _hex_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _card_font(size: int, bold: bool = False):
    from PIL import ImageFont
    candidates = ([
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ] if bold else [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]) + ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    from PIL import ImageFont as _IF
    return _IF.load_default()


def build_share_card(result: dict, out_path: str) -> str:
    """Build a Pokémon-style trading-card PNG for a discovery. Returns path or ""."""
    try:
        from PIL import Image, ImageDraw
        W, H = 1080, 1500

        entry  = result.get("entry", {}) or {}
        name   = (entry.get("common_name") or result.get("name") or "Unknown")
        sci    = entry.get("scientific_name", "") or ""
        group  = (entry.get("iconic_group") or entry.get("category")
                  or (result.get("inat", {}) or {}).get("iconic_taxon", "") or "").strip()
        rarity = result.get("local_rarity") or result.get("rarity") or ""
        habitat = entry.get("habitat", "") or ""
        fun     = entry.get("fun_fact", "") or entry.get("description", "") or ""
        cons    = entry.get("conservation_status", "") or ""
        area    = (result.get("inat", {}) or {}).get("local_area", "")
        ts      = (result.get("timestamp", "") or "")[:10]

        # Species group -> card "type" colour + tag
        TYPES = {
            "aves": ("#7fb3ff", "SKY"),      "mammalia": ("#d9a066", "BEAST"),
            "insecta": ("#8bc34a", "BUG"),   "arachnida": ("#6b8e23", "BUG"),
            "plantae": ("#57b95a", "FLORA"), "fungi": ("#b085f5", "FUNGAL"),
            "reptilia": ("#c0a838", "SCALE"),"amphibia": ("#4dd0c0", "AQUA"),
            "actinopterygii": ("#4aa3df", "AQUA"), "mollusca": ("#b085f5", "AQUA"),
        }
        tcol_hex, ttag = TYPES.get(group.lower(), ("#00d4ff", "WILD"))
        tcol = _hex_rgb(tcol_hex)

        # HP from rarity tier (rarer = more legendary)
        rl = rarity.lower()
        hp = (180 if ("very rare" in rl or "not recorded" in rl)
              else 150 if "rare" in rl
              else 120 if "uncommon" in rl
              else 80  if "very common" in rl
              else 100)

        gold_l, gold_d = _hex_rgb("#f4d47a"), _hex_rgb("#a9822f")
        dark, darker   = _hex_rgb("#12203a"), _hex_rgb("#0a0e1a")
        white, red, ink = _hex_rgb("#f7fafc"), _hex_rgb("#e23b3b"), _hex_rgb("#101828")
        body_col       = _hex_rgb("#c8d2e0")

        img  = Image.new("RGB", (W, H), gold_d)
        draw = ImageDraw.Draw(img)

        # ── Holographic foil background (rainbow diagonal + shimmer streaks) ──
        xs = np.linspace(0, 1, W); ys = np.linspace(0, 1, H)
        gx, gy = np.meshgrid(xs, ys)
        hue  = ((((gx + gy) * 0.5 + 0.12) % 1.0) * 255).astype("uint8")
        band = (np.sin((gx - gy) * 46) * 0.5 + 0.5)          # diagonal foil streaks
        val  = (44 + band * 82).astype("uint8")               # kept dark for readability
        sat  = np.full((H, W), 205, "uint8")
        holo = Image.fromarray(np.stack([hue, sat, val], axis=-1), "HSV").convert("RGB")
        img.paste(holo, (0, 0))
        draw = ImageDraw.Draw(img)

        # Gold double frame + a type-coloured accent line, holo showing through
        draw.rounded_rectangle((0, 0, W - 1, H - 1), radius=46, outline=gold_l, width=12)
        draw.rounded_rectangle((16, 16, W - 16, H - 16), radius=40, outline=gold_d, width=6)
        draw.rounded_rectangle((30, 30, W - 30, H - 30), radius=34, outline=tcol, width=4)

        # Translucent panel behind the lower text so it stays readable on the foil
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(ov).rounded_rectangle((60, 858, W - 60, H - 84),
                                             radius=24, fill=(8, 12, 22, 165))
        img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
        draw = ImageDraw.Draw(img)

        def wrap(text, font, maxw):
            words, lines, cur = text.split(), [], ""
            for w in words:
                test = (cur + " " + w).strip()
                if draw.textlength(test, font=font) <= maxw:
                    cur = test
                else:
                    if cur:
                        lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            return lines

        def energy(cx, cy, color):
            draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), fill=color, outline=gold_d, width=3)

        # ── Name banner ──
        draw.rounded_rectangle((72, 76, W - 72, 178), radius=20, fill=darker, outline=gold_d, width=2)
        draw.text((96, 92), name.upper()[:18], font=_card_font(60, True), fill=white)
        hp_txt = f"HP {hp}"
        hf = _card_font(46, True); hb = draw.textbbox((0, 0), hp_txt, font=hf)
        draw.text((W - 96 - (hb[2] - hb[0]), 100), hp_txt, font=hf, fill=red)
        tf = _card_font(20, True); tb = draw.textbbox((0, 0), ttag, font=tf)
        draw.rounded_rectangle((W - 96 - (tb[2] - tb[0]) - 26, 150, W - 96, 150 + 32),
                               radius=16, fill=tcol)
        draw.text((W - 96 - (tb[2] - tb[0]) - 13, 154), ttag, font=tf, fill=ink)

        # ── Art window (the scan photo as card art) ──
        art = (80, 198, W - 80, 762)
        draw.rounded_rectangle(art, radius=14, fill=darker, outline=gold_l, width=6)
        pic = result.get("saved_image_path") or result.get("sticker_path") or ""
        if pic and Path(pic).exists():
            pim = Image.open(pic).convert("RGB")
            aw, ah = art[2] - art[0] - 16, art[3] - art[1] - 16
            sr, dr = pim.width / pim.height, aw / ah
            if sr > dr:
                nh, nw = ah, int(ah * sr)
            else:
                nw, nh = aw, int(aw / sr)
            pim = pim.resize((nw, nh))
            lx, ty = (nw - aw) // 2, (nh - ah) // 2
            pim = pim.crop((lx, ty, lx + aw, ty + ah))
            img.paste(pim, (art[0] + 8, art[1] + 8))

        # ── Scientific-name strip ──
        draw.rounded_rectangle((80, 776, W - 80, 842), radius=12, fill=darker, outline=gold_d, width=2)
        draw.text((100, 790), (sci if sci and sci != "Unknown" else name),
                  font=_card_font(30, True), fill=gold_l)
        if area:
            an = f"RANGE: {area.upper()}"; af = _card_font(20, True)
            ab = draw.textbbox((0, 0), an, font=af)
            draw.text((W - 100 - (ab[2] - ab[0]), 798), an, font=af, fill=white)

        # ── "Attacks" ──
        y = 884
        if habitat:
            energy(110, y + 14, tcol)
            draw.text((146, y - 6), "HABITAT", font=_card_font(34, True), fill=white)
            yy = y + 44
            for ln in wrap(habitat, _card_font(26), W - 210)[:2]:
                draw.text((146, yy), ln, font=_card_font(26), fill=body_col); yy += 34
            y = yy + 22
            draw.line([(92, y - 14), (W - 92, y - 14)], fill=_hex_rgb("#2a3a52"), width=2)

        if fun:
            energy(110, y + 14, tcol); energy(148, y + 14, white)
            draw.text((186, y - 6), "SIGNATURE TRAIT", font=_card_font(34, True), fill=white)
            dmg = str(hp - 10); df = _card_font(46, True)
            db = draw.textbbox((0, 0), dmg, font=df)
            draw.text((W - 100 - (db[2] - db[0]), y - 8), dmg, font=df, fill=white)
            yy = y + 48
            for ln in wrap(fun, _card_font(26), W - 210)[:3]:
                draw.text((146, yy), ln, font=_card_font(26), fill=body_col); yy += 34

        # ── Bottom stat bar ──
        by = H - 210
        draw.line([(92, by), (W - 92, by)], fill=gold_d, width=2)
        if rarity:
            rf = _card_font(24, True); rb = draw.textbbox((0, 0), rarity, font=rf)
            draw.rounded_rectangle((92, by + 20, 92 + (rb[2] - rb[0]) + 40, by + 66),
                                   radius=23, fill=tcol)
            draw.text((112, by + 28), rarity, font=rf, fill=ink)
        if cons and cons not in ("N/A", ""):
            cf = _card_font(24, True); cb = draw.textbbox((0, 0), cons.upper(), font=cf)
            draw.text((W - 100 - (cb[2] - cb[0]), by + 28), cons.upper(), font=cf, fill=_hex_rgb("#ffd77a"))

        # ── Footer ──
        draw.text((96, H - 110), "Identified with NatureDex AI",
                  font=_card_font(24, True), fill=_hex_rgb("#8895aa"))
        if ts:
            nf = _card_font(24, True); nb = draw.textbbox((0, 0), ts, font=nf)
            draw.text((W - 100 - (nb[2] - nb[0]), H - 110), ts, font=nf, fill=_hex_rgb("#8895aa"))

        img.save(out_path)
        return out_path
    except Exception as e:
        print(f"[Share] card build failed: {e}")
        return ""


# ─── Invasive Species (North Carolina) ────────────────────────────────────────
# A curated list of species that are invasive / of concern in North Carolina.
# "severity" drives the banner color:
#   "Emerging Threat" (red)  — report immediately, limited spread, high priority
#   "Established"     (orange)— widespread, still worth documenting
#   "Watch"           (yellow)— monitor / coastal / localized
# "report_to" tells the user the correct real-world reporting channel.

_EDDMAPS   = "Report at EDDMapS.org or the EDDMapS mobile app"
_NCDA_PEST = "Report to NCDA&CS Plant Industry: 1-800-206-9333 / newpest@ncagr.gov"
_NCWRC     = "Report to the NC Wildlife Resources Commission (ncwildlife.org)"

NC_INVASIVE_SPECIES = [
    # ── Insects / pests (highest priority — emerging or destructive) ──────────
    {"common": "Spotted Lanternfly", "scientific_name": "Lycorma delicatula",
     "type": "Insect", "severity": "Emerging Threat", "report_to": _NCDA_PEST,
     "why": "A planthopper that damages grapevines, fruit trees, and hardwoods. NC is actively tracking its spread — early reports matter.",
     "match_keywords": ["spotted lanternfly", "lanternfly", "lycorma"]},
    {"common": "Emerald Ash Borer", "scientific_name": "Agrilus planipennis",
     "type": "Insect", "severity": "Emerging Threat", "report_to": _NCDA_PEST,
     "why": "A metallic-green beetle whose larvae have killed millions of ash trees. Look for D-shaped exit holes and canopy dieback.",
     "match_keywords": ["emerald ash borer", "agrilus planipennis"]},
    {"common": "Asian Longhorned Beetle", "scientific_name": "Anoplophora glabripennis",
     "type": "Insect", "severity": "Emerging Threat", "report_to": _NCDA_PEST,
     "why": "A large black-and-white beetle that bores into hardwoods. Not yet established in NC — sightings are urgent.",
     "match_keywords": ["asian longhorned beetle", "longhorned beetle", "anoplophora"]},
    {"common": "Hemlock Woolly Adelgid", "scientific_name": "Adelges tsugae",
     "type": "Insect", "severity": "Established", "report_to": _NCDA_PEST,
     "why": "Tiny sap-feeding insect (white woolly masses on hemlock needles) devastating NC's mountain hemlock forests.",
     "match_keywords": ["hemlock woolly adelgid", "woolly adelgid", "adelges"]},
    {"common": "Brown Marmorated Stink Bug", "scientific_name": "Halyomorpha halys",
     "type": "Insect", "severity": "Established", "report_to": _EDDMAPS,
     "why": "An agricultural pest that damages crops and invades homes in fall. Widespread across NC.",
     "match_keywords": ["brown marmorated stink bug", "marmorated stink bug", "halyomorpha"]},

    # ── Aquatic / coastal ─────────────────────────────────────────────────────
    {"common": "Red Lionfish", "scientific_name": "Pterois volitans",
     "type": "Fish", "severity": "Established", "report_to": _NCWRC,
     "why": "A venomous Indo-Pacific reef fish now off the NC coast. Preys on native reef species and has no natural predators here.",
     "match_keywords": ["lionfish", "pterois"]},
    {"common": "Northern Snakehead", "scientific_name": "Channa argus",
     "type": "Fish", "severity": "Emerging Threat", "report_to": _NCWRC,
     "why": "A predatory air-breathing fish that can survive out of water. Do NOT release it — report and remove.",
     "match_keywords": ["snakehead", "channa"]},
    {"common": "Nutria", "scientific_name": "Myocastor coypus",
     "type": "Mammal", "severity": "Established", "report_to": _NCWRC,
     "why": "A large semi-aquatic rodent that destroys wetland vegetation and erodes marsh banks.",
     "match_keywords": ["nutria", "coypu", "myocastor"]},
    {"common": "Feral Hog", "scientific_name": "Sus scrofa",
     "type": "Mammal", "severity": "Established", "report_to": _NCWRC,
     "why": "Wild pigs root up soil, destroy crops, and spread disease. A major problem across NC.",
     "match_keywords": ["feral hog", "wild boar", "feral pig", "wild pig"]},

    # ── Plants (widespread, still valuable to map) ────────────────────────────
    {"common": "Japanese Stiltgrass", "scientific_name": "Microstegium vimineum",
     "type": "Plant", "severity": "Established", "report_to": _EDDMAPS,
     "why": "A fast-spreading annual grass that carpets forest floors and crowds out native plants.",
     "match_keywords": ["stiltgrass", "microstegium"]},
    {"common": "Kudzu", "scientific_name": "Pueraria montana",
     "type": "Plant", "severity": "Established", "report_to": _EDDMAPS,
     "why": "\"The vine that ate the South\" — smothers trees and structures, growing up to a foot per day.",
     "match_keywords": ["kudzu", "pueraria"]},
    {"common": "Tree of Heaven", "scientific_name": "Ailanthus altissima",
     "type": "Plant", "severity": "Established", "report_to": _EDDMAPS,
     "why": "A fast-growing tree that crowds out natives and is the preferred host of the spotted lanternfly.",
     "match_keywords": ["tree of heaven", "ailanthus"]},
    {"common": "Chinese Privet", "scientific_name": "Ligustrum sinense",
     "type": "Plant", "severity": "Established", "report_to": _EDDMAPS,
     "why": "A shrub that forms dense thickets along streams and forest edges, shading out native seedlings.",
     "match_keywords": ["chinese privet", "ligustrum"]},
    {"common": "Autumn Olive", "scientific_name": "Elaeagnus umbellata",
     "type": "Plant", "severity": "Established", "report_to": _EDDMAPS,
     "why": "A shrub with silvery leaves and red berries that spreads aggressively into fields and roadsides.",
     "match_keywords": ["autumn olive", "elaeagnus"]},
    {"common": "Multiflora Rose", "scientific_name": "Rosa multiflora",
     "type": "Plant", "severity": "Established", "report_to": _EDDMAPS,
     "why": "A thorny shrub forming impenetrable thickets in pastures and woodlands.",
     "match_keywords": ["multiflora rose"]},
    {"common": "Oriental Bittersweet", "scientific_name": "Celastrus orbiculatus",
     "type": "Plant", "severity": "Established", "report_to": _EDDMAPS,
     "why": "A woody vine that girdles and topples trees under its weight.",
     "match_keywords": ["oriental bittersweet", "celastrus"]},
    {"common": "Japanese Honeysuckle", "scientific_name": "Lonicera japonica",
     "type": "Plant", "severity": "Established", "report_to": _EDDMAPS,
     "why": "A vine that blankets shrubs and forest edges, outcompeting native understory plants.",
     "match_keywords": ["japanese honeysuckle"]},
    {"common": "Callery (Bradford) Pear", "scientific_name": "Pyrus calleryana",
     "type": "Plant", "severity": "Established", "report_to": _EDDMAPS,
     "why": "An ornamental tree gone wild — forms dense thorny stands and displaces native trees.",
     "match_keywords": ["callery pear", "bradford pear", "pyrus calleryana"]},
    {"common": "Giant Hogweed", "scientific_name": "Heracleum mantegazzianum",
     "type": "Plant", "severity": "Emerging Threat", "report_to": _NCDA_PEST,
     "why": "Sap causes severe burns and blisters. Rare in NC — do NOT touch; report immediately.",
     "match_keywords": ["giant hogweed", "heracleum"]},
]


def check_invasive(common_name: str, scientific_name: str, raw_label: str = "") -> dict:
    """Return an invasive-species info dict if the scan matches a known NC invasive,
    else {}. Matches on the full scientific binomial first (most reliable), then on
    distinctive common-name keywords. Genus-only matching is deliberately avoided
    to prevent false positives (e.g. a garden rose being flagged as multiflora rose)."""
    sci = (scientific_name or "").strip().lower()
    hay = " ".join([common_name or "", raw_label or "",
                    scientific_name or ""]).lower()
    for inv in NC_INVASIVE_SPECIES:
        inv_sci = inv["scientific_name"].lower()
        if sci and sci == inv_sci:
            return inv
        for kw in inv["match_keywords"]:
            if kw in hay:
                return inv
    return {}


def _invasive_severity_color(severity: str) -> str:
    return {"Emerging Threat": C_RED,
            "Established":      C_ACCENT,
            "Watch":           C_YELLOW}.get(severity, C_ACCENT)


def _fetch_observation_coords(taxon_id: int, max_obs: int = 80) -> list[dict]:
    """Fetch research-grade observation coordinates for a taxon from iNaturalist."""
    if not taxon_id:
        return []
    try:
        params = urllib.parse.urlencode({
            "taxon_id":     taxon_id,
            "quality_grade":"research",
            "per_page":     max_obs,
            "order":        "votes",
            "order_by":     "votes",
            "geo":          "true",
        })
        req = urllib.request.Request(
            f"https://api.inaturalist.org/v1/observations?{params}",
            headers={"User-Agent": "NatureDexAI/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        coords = []
        for obs in data.get("results", []):
            loc = obs.get("location")
            if not loc:
                continue
            try:
                lat, lng = map(float, loc.split(","))
            except Exception:
                continue
            coords.append({
                "lat":         lat,
                "lng":         lng,
                "place":       obs.get("place_guess", ""),
                "observed_on": obs.get("observed_on", ""),
                "quality":     obs.get("quality_grade", ""),
            })
        return coords
    except Exception as e:
        print(f"[Map] Coord fetch error: {e}")
        return []


# Cache so we only look up location once per session
_user_location_cache: dict = {}

def _get_user_location() -> dict:
    """Get approximate user location using IP geolocation (ipinfo.io, free tier).
    Returns dict with lat, lng, city, region, country or empty dict on failure.
    Result is cached for the session so we only call the API once."""
    if _user_location_cache:
        return _user_location_cache.copy()
    try:
        req = urllib.request.Request(
            "https://ipinfo.io/json",
            headers={"User-Agent": "NatureDexAI/1.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        loc = data.get("loc", "")
        if loc and "," in loc:
            lat, lng = map(float, loc.split(","))
            result = {
                "lat":     lat,
                "lng":     lng,
                "city":    data.get("city", ""),
                "region":  data.get("region", ""),
                "country": data.get("country", ""),
            }
            _user_location_cache.update(result)
            return result
    except Exception as e:
        print(f"[Location] {e}")
    return {}


# ─── Custom NC Model ──────────────────────────────────────────────────────────

_custom_transform = transforms.Compose([
    transforms.Resize(int(224 * 1.1)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_custom_model():
    if not CUSTOM_MODEL_PTH.exists():
        print(f"[Custom model] Not found at {CUSTOM_MODEL_PTH} — using BioCLIP only")
        return None, None, None
    label_map_path = CUSTOM_MODEL_PTH.parent / "label_map.json"
    if not label_map_path.exists():
        print("[Custom model] label_map.json missing — skipping")
        return None, None, None
    try:
        device = torch.device("cpu")
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
        model.eval().to(device)
        model.naturedex_meta = {
            "val_accuracy": checkpoint.get("val_accuracy", 0),
            "num_classes":  num_classes,
        }
        with open(label_map_path) as f:
            label_map = json.load(f)
        print(f"[Custom model] Loaded — {num_classes} NC species, "
              f"val acc {checkpoint.get('val_accuracy', 0):.1f}%")
        return model, label_map, device
    except Exception as e:
        print(f"[Custom model] Failed to load: {e}")
        return None, None, None

@torch.no_grad()
def run_custom_model(model, label_map, device, image_path: str):
    if model is None:
        return None
    try:
        img = PILImage.open(image_path).convert("RGB")
        tensor = _custom_transform(img).unsqueeze(0).to(device)
        probs = F.softmax(model(tensor), dim=1)[0]
        top5_probs, top5_idx = torch.topk(probs, min(5, len(label_map)))
        top_idx  = top5_idx[0].item()
        top_prob = top5_probs[0].item() * 100
        top_info = label_map.get(str(top_idx), {})
        top_label = top_info.get("common_name", f"Species {top_idx}")
        alternatives = [
            {"name": label_map.get(str(idx.item()), {}).get("common_name", f"Species {idx.item()}"),
             "confidence": prob.item() * 100}
            for prob, idx in zip(top5_probs[1:4], top5_idx[1:4])
        ]
        return top_label, top_prob, alternatives, top_info
    except Exception as e:
        print(f"[Custom model] Inference error: {e}")
        return None

# ─── Global Model (BioCLIP) ───────────────────────────────────────────────────
# BioCLIP is a pretrained "tree of life" model covering ~950k taxa worldwide.
# It handles anything the NC specialist model isn't confident about, making the
# app global. No training required — install with:  pip install pybioclip

def load_bioclip():
    """Load the global BioCLIP classifier. Returns None if unavailable."""
    try:
        from bioclip import TreeOfLifeClassifier
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        classifier = TreeOfLifeClassifier(device=device)
        print(f"[BioCLIP] Global model loaded on {device}")
        return classifier
    except Exception as e:
        print(f"[BioCLIP] Not available ({e}) — global fallback disabled")
        return None


def _bioclip_names(pred: dict):
    """Extract (common_name, scientific_name) from one BioCLIP prediction dict."""
    genus = (pred.get("genus") or "").strip()
    sp    = (pred.get("species") or "").strip()
    # 'species' is sometimes the epithet, sometimes the full binomial
    sci = sp if (genus and genus.lower() in sp.lower()) else f"{genus} {sp}".strip()
    common = (pred.get("common_name") or pred.get("species_common_name")
              or sci or "Unknown species")
    return common, sci


@torch.no_grad()
def run_bioclip(classifier, image_path: str):
    """Identify with the global model. Returns (label, conf%, alternatives, info)."""
    if classifier is None:
        return None
    try:
        from bioclip import Rank
        preds = classifier.predict(image_path, Rank.SPECIES)
        if not preds:
            return None
        preds = preds[:BIOCLIP_TOPK]
        top = preds[0]
        common, sci = _bioclip_names(top)
        confidence  = float(top.get("score", 0)) * 100
        alternatives = []
        for p in preds[1:4]:
            c, _ = _bioclip_names(p)
            alternatives.append({"name": c,
                                 "confidence": float(p.get("score", 0)) * 100})
        top_info = {
            "scientific_name": sci,
            "iconic_group":    top.get("class") or top.get("phylum") or "",
        }
        return common, confidence, alternatives, top_info
    except Exception as e:
        print(f"[BioCLIP] Inference error: {e}")
        return None

# ─── Flow Layout (wraps its children to the next line when narrow) ──────────────

class FlowLayout(QLayout):
    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self._items = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _do_layout(self, rect, test_only):
        x, y = rect.x(), rect.y()
        line_height = 0
        spacing = self.spacing()
        for item in self._items:
            w = item.sizeHint().width()
            h = item.sizeHint().height()
            if x + w > rect.right() and line_height > 0:
                x = rect.x()
                y += line_height + spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            x += w + spacing
            line_height = max(line_height, h)
        return y + line_height - rect.y()


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
            self.msleep(33)

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
    result_ready   = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, frame, model, client,
                 custom_model=None, custom_label_map=None, custom_device=None,
                 bioclip=None):
        super().__init__()
        self.frame            = frame
        self.model            = model
        self.client           = client
        self.custom_model     = custom_model
        self.custom_label_map = custom_label_map
        self.custom_device    = custom_device
        self.bioclip          = bioclip

    def run(self):
        try:
            tmp_path = "/tmp/naturedex_scan.jpg"
            cv2.imwrite(tmp_path, self.frame)
            sticker_tmp = make_sticker(tmp_path, "/tmp/naturedex_sticker.png")

            used_custom   = False   # custom NC model retired from the pipeline
            native_nc     = False   # set later from iNaturalist location data

            # Primary identification: BioCLIP global model (~950k species worldwide)
            bioclip_result = run_bioclip(self.bioclip, tmp_path)
            if bioclip_result:
                label, confidence, alternatives, top_info = bioclip_result
                raw_label    = label.lower().replace(" ", "_")
                model_source = "BioCLIP — global model (950k+ species)"
            else:
                # BioCLIP unavailable — return a clear "couldn't identify" result
                # rather than a wrong guess.
                label        = "Unidentified"
                confidence   = 0.0
                raw_label    = "unidentified"
                alternatives = []
                model_source = "No model available"
                top_info     = {}

            inat_data = inat_lookup(label)

            # ── "Local to your area" detection (global, via iNaturalist) ──
            # Works anywhere on Earth: count observations of this species near
            # the user's location. Found nearby → it's local/native to them.
            local_rarity = ""
            taxon_id = inat_data.get("taxon_id") or top_info.get("taxon_id")
            loc = _get_user_location()
            if taxon_id and loc.get("lat") is not None:
                local_count = _get_local_count(taxon_id, loc["lat"], loc["lng"])
                area = loc.get("region") or loc.get("city") or "your area"
                inat_data["local_observations"] = local_count
                inat_data["local_area"]         = area
                local_rarity = _local_rarity_label(local_count, area)
                native_nc    = local_count > 0

            entry    = self._generate_entry(label, confidence, inat_data)
            nc_count = inat_data.get("nc_observations", 0)
            rarity   = _nc_rarity_label(nc_count)

            # Generate phonetic pronunciation for any real scientific name
            sci_name = (entry.get("scientific_name") or
                        inat_data.get("scientific_name") or "")
            phonetic, phonetic_speech = (
                self._get_phonetic(sci_name)
                if sci_name and sci_name not in ("Unknown", "N/A", "")
                else ("", ""))

            # Flag invasive species of concern in North Carolina
            invasive = check_invasive(
                inat_data.get("common_name") or label,
                sci_name, raw_label)

            self.result_ready.emit({
                "name":              inat_data.get("common_name") or label,
                "raw_label":         raw_label,
                "confidence":        confidence,
                "alternatives":      alternatives,
                "entry":             entry,
                "inat":              inat_data,
                "rarity":            rarity,
                "local_rarity":      local_rarity,
                "nc_observations":   nc_count,
                "model_source":      model_source,
                "used_custom_model": used_custom,
                "native_nc":         native_nc,
                "low_confidence":    confidence < LOW_CONFIDENCE_THRESHOLD,
                "phonetic":          phonetic,
                "phonetic_speech":   phonetic_speech,
                "invasive":          invasive,
                "timestamp":         datetime.datetime.now().isoformat(),
                "image_path":        tmp_path,
                "sticker_tmp":       sticker_tmp,
                "scan_location":     _get_user_location(),
            })
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _generate_entry(self, label: str, confidence: float, inat_data: dict) -> dict:
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
                lines.append(f"Wikipedia summary: {inat_data['wikipedia_summary'][:600]}")
            if lines:
                inat_context = (
                    "\n\nVerified data from iNaturalist "
                    "(use this — do NOT contradict it):\n" + "\n".join(lines))

        prompt = f"""You are NatureDex AI, an educational wildlife identification system.
A user has scanned an object identified as: {label} (confidence: {confidence:.1f}%){inat_context}

Generate a structured NatureDex entry. You MUST respond with ONLY valid JSON — no markdown, no code fences, no explanation.

IMPORTANT rules:
- Use the verified iNaturalist data above where provided
- Do NOT invent a different scientific name or conservation status
- For north_carolina_context, use the NC observation count above

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

        raw = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        ).choices[0].message.content.strip()

        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part; break

        if not raw.startswith("{"):
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e > s:
                raw = raw[s:e + 1]

        raw = raw.strip()
        if raw.startswith("{") and not raw.endswith("}"):
            cut = max(raw.rfind(","), 0)
            raw = raw[:cut].rstrip().rstrip(",") + "\n}"

        try:
            return json.loads(raw)
        except Exception:
            return {
                "common_name":           inat_data.get("common_name") or label,
                "scientific_name":       inat_data.get("scientific_name") or "Unknown",
                "category":              inat_data.get("iconic_taxon") or "Unknown",
                "type_tags":             [],
                "habitat":               "Unknown",
                "diet":                  "Unknown",
                "behavior":              "Unknown",
                "conservation_status":   inat_data.get("conservation_status") or "N/A",
                "north_carolina_context":"Unknown",
                "fun_fact":              "Analysis unavailable.",
                "description":           "Entry generation failed — try scanning again.",
            }

    def _get_phonetic(self, scientific_name: str):
        """Return (display, speech) pronunciations for a Latin scientific name.
        `display` is a hyphenated guide with the stressed syllable capitalized.
        `speech` is respelled for a US-English text-to-speech engine so it says
        the Latin correctly instead of reading it with English spelling rules."""
        if not scientific_name or scientific_name.lower().strip() in ("unknown", "n/a", ""):
            return "", ""
        try:
            prompt = (
                "You pronounce Latin biological scientific names. For the name below, "
                "give its pronunciation in TWO formats.\n"
                "Rules: 'ae'/'oe' = ee, 'c' before e/i = s, 'ch' = k, 'g' before e/i = j, "
                "'ph' = f, final 'a' = ah, final 'us' = us, final 'is' = iss, 'ii' = ee-eye. "
                "Stress the second-to-last syllable if it's heavy, else third-to-last.\n"
                "Reply with EXACTLY two lines, nothing else:\n"
                "DISPLAY: <syllables hyphenated, CAPITALIZE the stressed syllable>\n"
                "SPEECH: <the same sounds respelled as simple lowercase English syllables "
                "separated by spaces, no hyphens, no capitals — spelled so a US English "
                "text-to-speech voice pronounces it correctly>\n\n"
                f"Scientific name: {scientific_name}\n"
                "Examples:\n"
                "Sialia sialis →\nDISPLAY: sy-AY-lee-ah sy-AY-lis\nSPEECH: sigh ay lee ah, sigh ay liss\n"
                "Danaus plexippus →\nDISPLAY: DAN-ay-us plek-SIP-us\nSPEECH: dan ay us, plek sip us\n"
                "Laetiporus gilbertsonii →\nDISPLAY: lay-TIP-oh-rus gil-bert-SOH-nee-eye\n"
                "SPEECH: lay tip oh rus, gil bert soh nee eye"
            )
            resp = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=120,
            )
            raw = resp.choices[0].message.content.strip()
            display, speech = "", ""
            for line in raw.splitlines():
                s = line.strip().strip('"\'')
                if s.upper().startswith("DISPLAY:"):
                    display = s.split(":", 1)[1].strip()
                elif s.upper().startswith("SPEECH:"):
                    speech = s.split(":", 1)[1].strip()
            # Fallbacks
            if not display and "-" in raw:
                display = raw.strip()
            if not speech and display:
                speech = display.replace("-", " ")
            if display and "-" in display:
                return display, (speech or display.replace("-", " "))
            return "", ""
        except Exception:
            return "", ""


class ChatWorker(QThread):
    reply_ready = pyqtSignal(str)

    def __init__(self, client, messages):
        super().__init__()
        self.client   = client
        self.messages = messages

    def run(self):
        try:
            self.reply_ready.emit(
                self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=self.messages,
                    temperature=0.7,
                    max_tokens=400,
                ).choices[0].message.content.strip()
            )
        except Exception as e:
            self.reply_ready.emit(f"Error: {str(e)}")


# ─── Map HTTP Server ──────────────────────────────────────────────────────────
# Cesium needs an http:// origin (not file://) to resolve its built-in assets
# and load tile imagery. We spin up a tiny single-file server on localhost that
# also proxies Esri World Imagery + reference-label tiles (avoids all CORS/CDN
# blocking inside QWebEngineView).

_MAP_HTML_CONTENT = ""   # set each time before loading
_MAP_SERVER_PORT  = None


class _MapHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/map" or self.path == "/":
            # Serve the map HTML
            content = _MAP_HTML_CONTENT.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type",   "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)

        elif self.path.startswith("/tiles/"):
            # Proxy tile requests to Esri World Imagery (satellite basemap).
            # Path format: /tiles/{z}/{y}/{x}
            try:
                tile_path = self.path[len("/tiles/"):]
                esri_url  = (f"https://server.arcgisonline.com/ArcGIS/rest/services/"
                             f"World_Imagery/MapServer/tile/{tile_path}")
                req = urllib.request.Request(
                    esri_url,
                    headers={"User-Agent": "NatureDexAI/1.0",
                             "Referer":    "https://server.arcgisonline.com"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                self.send_response(200)
                self.send_header("Content-Type",   "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_response(404)
                self.end_headers()

        elif self.path.startswith("/labels/"):
            # Proxy Esri reference label tiles (place names + boundaries, PNG w/ alpha)
            try:
                tile_path = self.path[len("/labels/"):]
                esri_url  = (f"https://server.arcgisonline.com/ArcGIS/rest/services/"
                             f"Reference/World_Boundaries_and_Places/MapServer/tile/{tile_path}")
                req = urllib.request.Request(
                    esri_url,
                    headers={"User-Agent": "NatureDexAI/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                self.send_response(200)
                self.send_header("Content-Type",   "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(data)
            except Exception:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


def _get_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ensure_map_server() -> int:
    """Start the map server if not already running. Returns the port."""
    global _MAP_SERVER_PORT
    if _MAP_SERVER_PORT is not None:
        return _MAP_SERVER_PORT
    port = _get_free_port()

    class _ThreadedServer(http.server.ThreadingHTTPServer):
        pass

    server = _ThreadedServer(("127.0.0.1", port), _MapHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    _MAP_SERVER_PORT = port
    print(f"[Map server] Listening on http://127.0.0.1:{port}")
    return port


# ─── Boot Screen ──────────────────────────────────────────────────────────────

class BootScreen(QWidget):
    """Full-screen splash shown while models load, then fades out."""
    boot_complete = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"background: {C_BG};")
        self._progress = 0.0
        self._ready    = False
        self._dc       = 0

        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.setSpacing(0)

        # Icon
        icon = QLabel("🌿")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("font-size: 72px; background: transparent;")
        outer.addWidget(icon)
        outer.addSpacing(16)

        # App name
        name = QLabel("NatureDex")
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name.setStyleSheet(
            f"color:{C_TEXT};font-size:48px;font-weight:900;"
            f"letter-spacing:4px;background:transparent;")
        outer.addWidget(name)
        outer.addSpacing(8)

        # Tagline
        tag = QLabel("AI Wildlife Identification")
        tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tag.setStyleSheet(
            f"color:{C_ACCENT};font-size:14px;font-weight:600;"
            f"letter-spacing:3px;background:transparent;")
        outer.addWidget(tag)
        outer.addSpacing(56)

        # Progress container
        prog_wrap = QWidget()
        prog_wrap.setFixedWidth(320)
        prog_wrap.setStyleSheet("background:transparent;")
        pw_layout = QVBoxLayout(prog_wrap)
        pw_layout.setSpacing(10)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFixedHeight(4)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(f"""
            QProgressBar {{
                background: {C_CARD};
                border-radius: 2px;
                border: none;
            }}
            QProgressBar::chunk {{
                background: {C_ACCENT};
                border-radius: 2px;
            }}
        """)
        pw_layout.addWidget(self._bar)

        self._status = QLabel("Initializing...")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet(
            f"color:{C_SUBTEXT};font-size:11px;background:transparent;")
        pw_layout.addWidget(self._status)
        outer.addWidget(prog_wrap, alignment=Qt.AlignmentFlag.AlignCenter)
        outer.addSpacing(40)

        # Credit
        credit = QLabel("Congressional App Challenge  ·  Tejo Mukkamala")
        credit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credit.setStyleSheet(f"color:{C_BORDER};font-size:10px;background:transparent;")
        outer.addWidget(credit)

        # Tick timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(55)

        # Play boot sound after short delay
        QTimer.singleShot(300, lambda: _play(_WAV_BOOT))

    def mark_ready(self):
        """Called from model-load thread (via signal) when loading is done."""
        self._ready = True

    def _tick(self):
        if self._ready:
            self._bar.setValue(100)
            self._status.setText("Ready!")
            self._timer.stop()
            QTimer.singleShot(500, self._finish)
            return

        # Simulate progress that slows near 90%
        if   self._progress < 30:  self._progress += 2.8
        elif self._progress < 65:  self._progress += 1.1
        elif self._progress < 88:  self._progress += 0.3

        self._bar.setValue(int(self._progress))
        self._dc = (self._dc + 1) % 4
        dots = "." * self._dc
        if   self._progress < 35: msg = f"Loading vision models{dots}"
        elif self._progress < 65: msg = f"Loading NC wildlife model{dots}"
        else:                      msg = f"Preparing AI systems{dots}"
        self._status.setText(msg)

    def _finish(self):
        """Fade the boot screen out, then signal completion."""
        eff  = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(eff)
        anim = QPropertyAnimation(eff, b"opacity")
        anim.setDuration(500)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(self.boot_complete.emit)
        anim.start()
        self._anim = anim  # keep alive


# ─── UI Components ─────────────────────────────────────────────────────────────

class ScanButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("⬤  SCAN")
        self.setFixedSize(160, 52)
        self._scanning  = False
        self._dot_count = 0
        self._timer     = QTimer(self)
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
        self._scanning  = True
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._active = False
        self._y      = 0
        self._timer  = QTimer(self)
        self._timer.timeout.connect(self._update)
        self._corner_flash = 0

    def start(self):
        self._active = True
        self._y      = 0
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

        painter.fillRect(0, 0, w, h, QColor(0, 0, 0, 40))

        grad = QLinearGradient(0, self._y - 20, 0, self._y + 20)
        grad.setColorAt(0.0, QColor(232, 114, 12, 0))
        grad.setColorAt(0.5, QColor(232, 114, 12, 160))
        grad.setColorAt(1.0, QColor(232, 114, 12, 0))
        painter.fillRect(0, self._y - 20, w, 40, grad)

        pen = QPen(QColor(C_ACCENT), 3)
        painter.setPen(pen)
        corner, gap = 26, 16
        for x, y in [(gap, gap), (w - gap, gap), (gap, h - gap), (w - gap, h - gap)]:
            dx = corner if x == gap else -corner
            dy = corner if y == gap else -corner
            painter.drawLine(x, y, x + dx, y)
            painter.drawLine(x, y, x, y + dy)

        cx, cy = w // 2, h // 2
        pen2 = QPen(QColor(232, 114, 12, 80), 1)
        pen2.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(pen2)
        painter.drawLine(cx - 12, cy, cx + 12, cy)
        painter.drawLine(cx, cy - 12, cx, cy + 12)
        painter.end()


class AchievementOverlay(QWidget):
    """Full-window celebration: badge springs in and confetti bursts outward."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._particles = []
        self._t = 0.0
        self._icon = "🏆"
        self._name = ""
        self._badge_scale = 0.0
        self._active = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def show_achievement(self, icon, name):
        import random, math
        self._icon, self._name = icon, name
        if self.parent():
            self.setGeometry(self.parent().rect())
        self._t = 0.0
        self._badge_scale = 0.0
        self._active = True
        cx, cy = self.width() // 2, self.height() // 2 - 30
        palette = ["#00d4ff", "#ffd43b", "#39ff14", "#ff6b35",
                   "#ff4fa3", "#a06bff", "#ffffff"]
        self._particles = []
        for _ in range(160):
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(7, 23)
            self._particles.append({
                "x": float(cx), "y": float(cy),
                "vx": math.cos(ang) * spd,
                "vy": math.sin(ang) * spd - random.uniform(3, 9),
                "col": QColor(random.choice(palette)),
                "w": random.uniform(7, 15), "h": random.uniform(9, 18),
                "rot": random.uniform(0, 360), "vrot": random.uniform(-16, 16),
                "life": 1.0,
            })
        self.show()
        self.raise_()
        self._timer.start(16)

    def _tick(self):
        self._t += 0.016
        if self._badge_scale < 1.0:
            self._badge_scale = min(1.0, self._badge_scale + 0.10)
        for p in self._particles:
            p["vy"] += 0.45
            p["vx"] *= 0.99
            p["x"] += p["vx"]; p["y"] += p["vy"]
            p["rot"] += p["vrot"]
            p["life"] -= 0.006
        self._particles = [p for p in self._particles if p["life"] > 0]
        self.update()
        if self._t > 3.6:
            self._finish()

    def _finish(self):
        self._timer.stop()
        self._active = False
        self.hide()

    def mousePressEvent(self, e):
        self._finish()

    @staticmethod
    def _ease_out_back(x):
        c1 = 1.70158; c3 = c1 + 1
        return 1 + c3 * ((x - 1) ** 3) + c1 * ((x - 1) ** 2)

    def paintEvent(self, e):
        if not self._active:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        if self._t < 0.3:
            a = int(150 * (self._t / 0.3))
        elif self._t > 3.0:
            a = int(150 * max(0.0, (3.6 - self._t) / 0.6))
        else:
            a = 150
        p.fillRect(self.rect(), QColor(5, 8, 16, a))

        p.setPen(Qt.PenStyle.NoPen)
        for pt in self._particles:
            p.save()
            p.translate(pt["x"], pt["y"])
            p.rotate(pt["rot"])
            c = QColor(pt["col"])
            c.setAlphaF(max(0.0, min(1.0, pt["life"])))
            p.setBrush(c)
            p.drawRect(int(-pt["w"] / 2), int(-pt["h"] / 2), int(pt["w"]), int(pt["h"]))
            p.restore()

        sc = self._ease_out_back(self._badge_scale)
        cw, ch = 340, 220
        cx, cy = w // 2, h // 2 - 30
        bw, bh = int(cw * sc), int(ch * sc)
        if bw > 4 and bh > 4:
            rect = QRect(cx - bw // 2, cy - bh // 2, bw, bh)
            p.setBrush(QColor(C_CARD))
            pen = QPen(QColor(C_GOLD)); pen.setWidth(4)
            p.setPen(pen)
            p.drawRoundedRect(rect, 22, 22)
            if self._badge_scale > 0.55:
                p.setPen(QColor(C_GOLD))
                fh = QFont(); fh.setPointSize(11); fh.setBold(True); p.setFont(fh)
                p.drawText(QRect(cx - bw // 2, cy - bh // 2 + 16, bw, 24),
                           int(Qt.AlignmentFlag.AlignHCenter), "★  ACHIEVEMENT UNLOCKED  ★")
                fi = QFont(); fi.setPointSize(46); p.setFont(fi)
                p.drawText(QRect(cx - bw // 2, cy - bh // 2 + 42, bw, 96),
                           int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                           self._icon)
                p.setPen(QColor(C_TEXT))
                fn = QFont(); fn.setPointSize(17); fn.setBold(True); p.setFont(fn)
                p.drawText(QRect(cx - bw // 2 + 10, cy + bh // 2 - 52, bw - 20, 40),
                           int(Qt.AlignmentFlag.AlignHCenter), self._name)
        p.end()


class ToastNotification(QFrame):
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
        header.setStyleSheet(
            f"color: {C_GOLD}; font-size: 9px; font-weight: 800; letter-spacing: 1.5px;")
        self._name_lbl = QLabel("")
        self._name_lbl.setStyleSheet(
            f"color: {C_TEXT}; font-size: 13px; font-weight: 700;")
        text_box.addWidget(header)
        text_box.addWidget(self._name_lbl)
        layout.addLayout(text_box)

        self.setFixedWidth(280)
        self.adjustSize()
        self.hide()

        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self.hide)

    def show_achievement(self, icon, name):
        self._icon_lbl.setText(icon)
        self._name_lbl.setText(name)
        self.show()
        self.raise_()
        self._dismiss_timer.start(3000)


class GalleryCard(QFrame):
    """A thumbnail card in the scan gallery with a hover-reveal delete button."""
    clicked_signal = pyqtSignal(dict)
    delete_signal  = pyqtSignal(dict)   # emits entry dict so caller can remove image + entry

    THUMB = 120

    def __init__(self, entry: dict, parent=None):
        super().__init__(parent)
        self._entry = entry
        self.setFixedSize(self.THUMB, self.THUMB + 36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setStyleSheet(f"""
            QFrame {{
                background: {C_CARD};
                border-radius: 8px;
                border: none;
            }}
            QFrame:hover {{
                background: #254225;
                border: 1px solid {C_ACCENT};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Image container — stacked so delete button overlays it
        img_container = QWidget()
        img_container.setFixedSize(self.THUMB, self.THUMB)
        img_container.setStyleSheet("background: transparent;")

        # ── Bottom layer: the real scan photo (shown by default) ──
        self._photo_lbl = QLabel(img_container)
        self._photo_lbl.setFixedSize(self.THUMB, self.THUMB)
        self._photo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._photo_lbl.setStyleSheet(f"border-radius: 8px 8px 0 0; background: {C_BG};")
        raw = entry.get("saved_image_path", "")
        try:
            if raw and Path(raw).exists():
                px = QPixmap(raw)
                if not px.isNull():
                    px = px.scaled(self.THUMB, self.THUMB,
                                   Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                   Qt.TransformationMode.SmoothTransformation)
                    if px.width() > self.THUMB or px.height() > self.THUMB:
                        x = (px.width()  - self.THUMB) // 2
                        y = (px.height() - self.THUMB) // 2
                        px = px.copy(x, y, self.THUMB, self.THUMB)
                    self._photo_lbl.setPixmap(px)
                else:
                    self._photo_lbl.setText("📷")
            else:
                self._photo_lbl.setText("📷")
        except Exception:
            self._photo_lbl.setText("📷")

        # ── Top layer: cut-out sticker on a checkerboard, revealed on click ──
        self._has_sticker = False
        self._sticker_shown = False
        self._reveal_anim = None
        self._sticker_lbl = QLabel(img_container)
        self._sticker_lbl.setFixedSize(self.THUMB, self.THUMB)
        self._sticker_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sticker_lbl.setStyleSheet("border-radius: 8px 8px 0 0; background: transparent;")
        sticker = entry.get("sticker_path", "")
        if sticker and Path(sticker).exists():
            comp = self._build_sticker_pixmap(sticker)
            if comp is not None:
                self._sticker_lbl.setPixmap(comp)
                self._has_sticker = True
        self._sticker_effect = QGraphicsOpacityEffect(self._sticker_lbl)
        self._sticker_effect.setOpacity(0.0)
        self._sticker_lbl.setGraphicsEffect(self._sticker_effect)

        self._spark = None
        if self._has_sticker:
            self._spark = QLabel("✨", img_container)
            self._spark.setStyleSheet(
                "background: rgba(10,14,26,0.65); color: #ffd77a;"
                " border-radius: 7px; font-size: 10px; padding: 1px 4px;")
            self._spark.move(6, self.THUMB - 24)
            self._spark.setToolTip("Tap to reveal sticker")

        # Delete button — top-right corner of image, hidden until hover
        self._del_btn = QPushButton("✕", img_container)
        self._del_btn.setFixedSize(22, 22)
        self._del_btn.move(self.THUMB - 26, 4)
        self._del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._del_btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(26,31,20,0.85);
                color: {C_SUBTEXT};
                border: 1px solid {C_BORDER};
                border-radius: 11px;
                font-size: 11px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background: {C_RED};
                color: white;
                border-color: {C_RED};
            }}
        """)
        self._del_btn.hide()
        self._del_btn.clicked.connect(lambda: self.delete_signal.emit(self._entry))

        layout.addWidget(img_container)

        name_lbl = QLabel(entry.get("name", "Unknown")[:14])
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet(
            f"color: {C_TEXT}; font-size: 9px; font-weight: 600; padding: 4px 4px 2px 4px;")
        date_lbl = QLabel(entry.get("timestamp", "")[:10])
        date_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        date_lbl.setStyleSheet(
            f"color: {C_SUBTEXT}; font-size: 8px; padding: 0 4px 4px 4px;")
        layout.addWidget(name_lbl)
        layout.addWidget(date_lbl)

    def enterEvent(self, event):
        self._del_btn.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._del_btn.hide()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        # Don't fire if the delete button was pressed
        if self._del_btn.geometry().contains(event.pos()):
            return
        self._toggle_sticker()

    def _toggle_sticker(self):
        if not self._has_sticker:
            return
        self._sticker_shown = not self._sticker_shown
        self._reveal_anim = QPropertyAnimation(self._sticker_effect, b"opacity")
        self._reveal_anim.setDuration(300)
        self._reveal_anim.setStartValue(self._sticker_effect.opacity())
        self._reveal_anim.setEndValue(1.0 if self._sticker_shown else 0.0)
        self._reveal_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._reveal_anim.start()
        if self._spark is not None:
            self._spark.setVisible(not self._sticker_shown)

    def _build_sticker_pixmap(self, sticker_path):
        """Composite the cut-out onto a classic transparency checkerboard."""
        try:
            st = QPixmap(sticker_path)
            if st.isNull():
                return None
            size = self.THUMB
            canvas = QPixmap(size, size)
            canvas.fill(Qt.GlobalColor.transparent)
            p = QPainter(canvas)
            sq = 10
            light, dark = QColor("#7c8ba0"), QColor("#586780")
            for yy in range(0, size, sq):
                for xx in range(0, size, sq):
                    p.fillRect(xx, yy, sq, sq,
                               light if ((xx // sq + yy // sq) % 2 == 0) else dark)
            st = st.scaled(size - 10, size - 10,
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
            p.drawPixmap((size - st.width()) // 2, (size - st.height()) // 2, st)
            p.end()
            return canvas
        except Exception:
            return None


class CollectionCard(QFrame):
    clicked_signal = pyqtSignal(dict)
    delete_signal  = pyqtSignal(str)

    def __init__(self, entry_data, parent=None):
        super().__init__(parent)
        self.entry_data   = entry_data
        self._category    = entry_data.get("entry", {}).get("category", "Unknown") or "Unknown"
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

        conf      = entry_data.get("confidence", 0)
        dot_color = C_GREEN if conf >= 75 else C_YELLOW if conf >= 50 else C_RED
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {dot_color}; font-size: 10px;")
        dot.setFixedWidth(14)
        layout.addWidget(dot)

        info = QVBoxLayout()
        info.setSpacing(1)
        name_lbl = QLabel(entry_data.get("name", "Unknown"))
        name_lbl.setStyleSheet(
            f"color: {C_TEXT}; font-size: 12px; font-weight: 600;")
        ts_lbl = QLabel(entry_data.get("timestamp", "")[:10])
        ts_lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 10px;")
        info.addWidget(name_lbl)
        info.addWidget(ts_lbl)
        layout.addLayout(info)
        layout.addStretch()

        conf_lbl = QLabel(f"{conf:.0f}%")
        conf_lbl.setStyleSheet(
            f"color: {dot_color}; font-size: 11px; font-weight: 700;")
        layout.addWidget(conf_lbl)

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
        if self._delete_btn.geometry().contains(event.pos()):
            return
        self.clicked_signal.emit(self.entry_data)


# ─── Main Window ───────────────────────────────────────────────────────────────

class OnboardingOverlay(QWidget):
    """First-run welcome screen: what NatureDex does + how to use it."""

    STEPS = [
        ("🔍", "Point & Scan", "Aim at any plant, animal, or bug and press SCAN."),
        ("📖", "Learn", "Get an instant ID plus habitat, diet, rarity, and fun facts."),
        ("🧭", "Nearby Index", "See which species live around you and go find them."),
        ("🎴", "Collect", "Every catch becomes a cut-out sticker and a shareable card."),
        ("🏆", "Earn Awards", "Unlock badges and confetti as your collection grows."),
    ]

    def __init__(self, parent, on_done):
        super().__init__(parent)
        self._on_done = on_done
        self.setGeometry(parent.rect())

        self._card = QFrame(self)
        self._card.setObjectName("onbCard")
        self._card.setFixedSize(480, 560)
        self._card.setStyleSheet(f"""
            QFrame#onbCard {{
                background: {C_PANEL};
                border: 1px solid {C_BORDER};
                border-radius: 18px;
            }}
            QLabel {{ background: transparent; border: none; }}
        """)
        cl = QVBoxLayout(self._card)
        cl.setContentsMargins(30, 26, 30, 26)
        cl.setSpacing(6)

        title = QLabel("Welcome to NatureDex")
        title.setStyleSheet(f"color: {C_ACCENT}; font-size: 24px; font-weight: 800;")
        cl.addWidget(title)
        tag = QLabel("Turn the world into a living Pokédex.")
        tag.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 13px;")
        cl.addWidget(tag)
        cl.addSpacing(14)

        for icon, name, desc in self.STEPS:
            row = QHBoxLayout(); row.setSpacing(14)
            ic = QLabel(icon); ic.setStyleSheet("font-size: 26px;")
            ic.setFixedWidth(38)
            row.addWidget(ic)
            tb = QVBoxLayout(); tb.setSpacing(1)
            nm = QLabel(name)
            nm.setStyleSheet(f"color: {C_TEXT}; font-size: 14px; font-weight: 700;")
            ds = QLabel(desc); ds.setWordWrap(True)
            ds.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 12px;")
            tb.addWidget(nm); tb.addWidget(ds)
            row.addLayout(tb)
            cl.addLayout(row)
            cl.addSpacing(8)

        cl.addStretch()
        btn = QPushButton("Get Started  →")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedHeight(46)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {C_ACCENT}; color: {C_BG};
                border: none; border-radius: 23px;
                font-size: 15px; font-weight: 800; letter-spacing: 1px;
            }}
            QPushButton:hover {{ background: #33ddff; }}
        """)
        btn.clicked.connect(self._finish)
        cl.addWidget(btn)

        self._center()

    def _center(self):
        self._card.move((self.width() - self._card.width()) // 2,
                        (self.height() - self._card.height()) // 2)

    def resizeEvent(self, e):
        self.setGeometry(self.parent().rect())
        self._center()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(5, 8, 16, 205))
        p.end()

    def _finish(self):
        try:
            self._on_done()
        finally:
            self.deleteLater()


class NatureDexWindow(QMainWindow):

    model_status_signal = pyqtSignal(str)
    map_html_signal     = pyqtSignal(str)  # emits full HTML from background thread

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NatureDex AI")

        screen    = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen else QRect(0, 0, 1280, 800)
        target_w  = min(1280, available.width()  - 40)
        target_h  = min(800,  available.height() - 40)
        self.setMinimumSize(min(1000, target_w), min(640, target_h))
        self.resize(target_w, target_h)
        self.move(available.x() + 20, available.y() + 20)

        self._collection            = self._load_collection()
        self._unlocked_achievements = self._load_achievements()
        self._challenge_state = self._load_challenges()
        self._current_result        = None
        self._chat_history          = []
        self._camera_thread         = None
        self._analysis_worker       = None
        self._scan_cancelled        = False  # set True on cancel, checked in _on_result
        self._last_frame            = None
        self._scan_overlay          = None
        self._toast                 = None
        self._chat_worker           = None
        self._entry_anims           = []  # keeps fade-in animation refs alive

        self._model            = None
        self._client           = None
        self._custom_model     = None
        self._custom_label_map = None
        self._custom_device    = None
        self._bioclip          = None
        self._models_loaded    = False

        self._setup_style()
        self._build_ui()
        self._refresh_sound_icons()
        self._refresh_streak_label()

        # ── Boot screen overlay ──────────────────────────────────────────────
        self._boot = BootScreen(self)
        self._boot.setGeometry(self.rect())
        self._boot.boot_complete.connect(self._on_boot_done)
        self._boot.show()
        self._boot.raise_()

        self._start_camera()
        self.model_status_signal.connect(self._on_model_status)
        self.map_html_signal.connect(self._on_map_html)
        _ensure_map_server()  # start early so port is known before first scan
        self._load_models_async()

        self._toast = AchievementOverlay(self)
        self._toast.move(self.width() - 300, 70)

    # ── Boot ───────────────────────────────────────────────────────────────────

    def _on_boot_done(self):
        self._boot.hide()
        self._boot.deleteLater()
        if not ONBOARD_FILE.exists():
            QTimer.singleShot(200, self._show_onboarding)

    def _show_onboarding(self):
        def _done():
            try:
                ONBOARD_FILE.write_text("1")
            except Exception:
                pass
        self._onboarding = OnboardingOverlay(self, _done)
        self._onboarding.show()
        self._onboarding.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_boot"):
            try:
                self._boot.setGeometry(self.rect())
            except RuntimeError:
                pass
        if self._toast:
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
        root_layout.addWidget(self._build_sidebar())
        root_layout.addWidget(self._build_main(), stretch=1)

    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(270)
        sidebar.setStyleSheet(f"background: {C_PANEL};")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setFixedHeight(78)
        header.setStyleSheet(f"background: {C_PANEL};")
        header_v = QVBoxLayout(header)
        header_v.setContentsMargins(18, 9, 12, 7)
        header_v.setSpacing(7)

        title = QLabel("NatureDex")
        title.setStyleSheet(
            f"color: {C_TEXT}; font-size: 20px; font-weight: 800; letter-spacing: 1px;")
        header_v.addWidget(title)

        icon_row = QHBoxLayout()
        icon_row.setContentsMargins(0, 0, 0, 0)
        icon_row.setSpacing(16)

        self._nearby_btn = QLabel("🧭")
        self._nearby_btn.setStyleSheet(f"color: {C_ACCENT}; font-size: 17px;")
        self._nearby_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nearby_btn.setToolTip("Nearby Index — species to discover near you")
        self._nearby_btn.mousePressEvent = lambda e: self._show_nearby_index()
        icon_row.addWidget(self._nearby_btn)

        self._badges_btn = QLabel("🏆")
        self._badges_btn.setStyleSheet(f"color: {C_GOLD}; font-size: 17px;")
        self._badges_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._badges_btn.setToolTip("Achievements")
        self._badges_btn.mousePressEvent = lambda e: self._show_badges_panel()
        icon_row.addWidget(self._badges_btn)

        self._challenge_btn = QLabel("🔥")
        self._challenge_btn.setStyleSheet(f"color: {C_ACCENT2}; font-size: 17px;")
        self._challenge_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._challenge_btn.setToolTip("Daily challenges & streak")
        self._challenge_btn.mousePressEvent = lambda e: self._show_challenges_panel()
        icon_row.addWidget(self._challenge_btn)

        self._sound_btn = QLabel("🔊")
        self._sound_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sound_btn.setToolTip("Sound effects")
        self._sound_btn.mousePressEvent = lambda e: self._toggle_sound()
        icon_row.addWidget(self._sound_btn)

        self._help_btn = QLabel("❔")
        self._help_btn.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 16px;")
        self._help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._help_btn.setToolTip("How to use NatureDex")
        self._help_btn.mousePressEvent = lambda e: self._show_onboarding()
        icon_row.addWidget(self._help_btn)

        icon_row.addStretch()
        header_v.addLayout(icon_row)
        layout.addWidget(header)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background: {C_BORDER};")
        layout.addWidget(divider)

        stats_frame = QFrame()
        stats_frame.setFixedHeight(36)
        stats_frame.setStyleSheet(f"background: {C_PANEL};")
        s_layout = QHBoxLayout(stats_frame)
        s_layout.setContentsMargins(18, 0, 18, 0)
        self._species_count_lbl = QLabel(f"{len(self._collection)} discovered")
        self._species_count_lbl.setStyleSheet(
            f"color: {C_SUBTEXT}; font-size: 11px;")
        s_layout.addWidget(self._species_count_lbl)
        s_layout.addStretch()
        self._streak_lbl = QLabel("")
        self._streak_lbl.setStyleSheet(f"color: {C_ACCENT2}; font-size: 11px; font-weight: 700;")
        s_layout.addWidget(self._streak_lbl)
        layout.addWidget(stats_frame)

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

        col_label = QLabel("DISCOVERIES")
        col_label.setStyleSheet(f"""
            color: {C_SUBTEXT};
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 2px;
            padding: 8px 18px 4px 18px;
        """)
        layout.addWidget(col_label)

        self._empty_filter_lbl = QLabel("No matches found")
        self._empty_filter_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_filter_lbl.setStyleSheet(
            f"color: {C_SUBTEXT}; font-size: 12px; padding: 20px;")
        self._empty_filter_lbl.hide()
        layout.addWidget(self._empty_filter_lbl)

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
        layout.addWidget(self._build_camera_panel(), stretch=5)
        layout.addWidget(self._build_info_panel(),   stretch=4)
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
        cam_lbl.setStyleSheet(
            f"color: {C_ACCENT}; font-size: 11px; font-weight: 700; letter-spacing: 2px;")
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 11px;")
        ch_layout.addWidget(cam_lbl)
        ch_layout.addStretch()
        ch_layout.addWidget(self._status_lbl)
        layout.addWidget(cam_header)

        cam_container = QFrame()
        cam_container.setStyleSheet("background: #000;")
        cam_layout = QVBoxLayout(cam_container)
        cam_layout.setContentsMargins(0, 0, 0, 0)
        self._camera_label = QLabel()
        self._camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._camera_label.setStyleSheet("background: #000;")
        self._camera_label.setMinimumHeight(220)
        cam_layout.addWidget(self._camera_label)
        self._scan_overlay = ScanOverlay(cam_container)
        self._scan_overlay.hide()
        layout.addWidget(cam_container, stretch=1)

        controls = QFrame()
        controls.setFixedHeight(88)
        controls.setStyleSheet(
            f"background: {C_PANEL}; border-top: 1px solid {C_BORDER};")
        c_layout = QHBoxLayout(controls)
        c_layout.setContentsMargins(24, 0, 24, 0)
        c_layout.setSpacing(16)

        self._scan_btn = ScanButton()
        self._scan_btn.clicked.connect(self._on_scan)

        self._cancel_btn = QPushButton("✕  Cancel")
        self._cancel_btn.setFixedSize(100, 38)
        self._cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_btn.hide()
        self._cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {C_SUBTEXT};
                border: 1px solid {C_BORDER};
                border-radius: 19px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                color: {C_RED};
                border-color: {C_RED};
            }}
        """)
        self._cancel_btn.clicked.connect(self._on_cancel_scan)

        hint = QLabel("Point camera at any\nplant, animal, or object")
        hint.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 11px; line-height: 1.5;")
        hint.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._loading_lbl = QLabel("Loading AI models...")
        self._loading_lbl.setStyleSheet(f"color: {C_ACCENT}; font-size: 11px;")

        c_layout.addWidget(self._scan_btn)
        c_layout.addWidget(self._cancel_btn)
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

        tab_bar = QFrame()
        tab_bar.setFixedHeight(44)
        tab_bar.setStyleSheet(f"background: {C_PANEL};")
        tab_layout = QHBoxLayout(tab_bar)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)

        self._tab_entry_btn   = self._make_tab_btn("ENTRY",   True)
        self._tab_chat_btn    = self._make_tab_btn("ASK AI",  False)
        self._tab_map_btn     = self._make_tab_btn("MAP",     False)
        self._tab_gallery_btn = self._make_tab_btn("GALLERY", False)
        self._tab_entry_btn.clicked.connect(lambda: self._switch_tab(0))
        self._tab_chat_btn.clicked.connect(lambda: self._switch_tab(1))
        self._tab_map_btn.clicked.connect(lambda: self._switch_tab(2))
        self._tab_gallery_btn.clicked.connect(lambda: self._switch_tab(3))

        tab_layout.addWidget(self._tab_entry_btn)
        tab_layout.addWidget(self._tab_chat_btn)
        tab_layout.addWidget(self._tab_map_btn)
        tab_layout.addWidget(self._tab_gallery_btn)
        tab_layout.addStretch()
        layout.addWidget(tab_bar)

        self._tab_stack = QStackedWidget()
        self._tab_stack.addWidget(self._build_entry_tab())
        self._tab_stack.addWidget(self._build_chat_tab())
        self._tab_stack.addWidget(self._build_map_tab())
        self._tab_stack.addWidget(self._build_gallery_tab())
        layout.addWidget(self._tab_stack, stretch=1)
        return panel

    def _make_tab_btn(self, text, active):
        btn = QPushButton(text)
        btn.setFixedHeight(44)
        btn.setFixedWidth(88)
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
        self._set_tab_style(self._tab_entry_btn,   idx == 0)
        self._set_tab_style(self._tab_chat_btn,    idx == 1)
        self._set_tab_style(self._tab_map_btn,     idx == 2)
        self._set_tab_style(self._tab_gallery_btn, idx == 3)
        if idx == 2 and self._current_result:
            self._update_map(self._current_result)
        if idx == 3:
            self._refresh_gallery()

    def _build_entry_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._entry_content = QWidget()
        self._entry_content.setStyleSheet(f"background: {C_BG};")
        self._entry_content.setAutoFillBackground(True)
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

        self._share_btn = QPushButton("📤  Share")
        self._share_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._share_btn.setEnabled(False)
        self._share_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C_ACCENT};
                color: {C_BG};
                border: none;
                border-radius: 13px;
                font-size: 11px;
                font-weight: 700;
                padding: 5px 16px;
            }}
            QPushButton:disabled {{ background: {C_BORDER}; color: {C_SUBTEXT}; }}
            QPushButton:enabled:hover {{ background: #33ddff; }}
        """)
        self._share_btn.clicked.connect(self._on_share_card)

        f_layout.addWidget(self._share_btn)
        f_layout.addStretch()
        f_layout.addWidget(self._report_btn)
        layout.addWidget(footer)
        return widget

    def _build_map_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if _HAS_WEBENGINE:
            self._map_view = QWebEngineView()
            s = self._map_view.settings()
            # Allow the localhost-served page to load remote CDN scripts (Cesium)
            s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
            s.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            s.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, False)
            s.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
            s.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
            self._map_view.setStyleSheet(f"background: {C_BG};")
            layout.addWidget(self._map_view)
            self._map_view.setHtml(self._map_placeholder_html())
        else:
            # Fallback if WebEngine not available
            lbl = QLabel("Map requires PyQt6-WebEngine.\nRun: pip3 install PyQt6-WebEngine")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 13px;")
            layout.addWidget(lbl)
            self._map_view = None

        return widget

    def _map_placeholder_html(self) -> str:
        return f"""<!DOCTYPE html><html><body style="margin:0;background:{C_BG};
            display:flex;align-items:center;justify-content:center;height:100vh;
            font-family:sans-serif;color:{C_SUBTEXT};font-size:14px;">
            <div style="text-align:center">
                <div style="font-size:40px;margin-bottom:12px">🗺️</div>
                <div>Scan a species to see its distribution map</div>
            </div></body></html>"""

    def _update_map(self, result: dict):
        """Build and load the CesiumJS 3D globe for the current result."""
        if not _HAS_WEBENGINE or not self._map_view:
            return

        taxon_id      = result.get("inat", {}).get("taxon_id")
        name          = result.get("name", "Unknown")
        sci           = result.get("entry", {}).get("scientific_name", "")
        rarity        = result.get("rarity", "")
        scan_location = result.get("scan_location", {})

        self._map_view.setHtml(f"""<!DOCTYPE html><html>
<body style="margin:0;background:{C_BG};display:flex;align-items:center;
justify-content:center;height:100vh;font-family:sans-serif;
color:{C_SUBTEXT};font-size:14px;">
<div style="text-align:center">
    <div style="font-size:32px;margin-bottom:12px">🌍</div>
    <div>Loading globe for <b style="color:{C_TEXT}">{name}</b>...</div>
</div></body></html>""")

        def _fetch():
            coords = _fetch_observation_coords(taxon_id) if taxon_id else []
            html   = self._build_map_html(name, sci, rarity, coords, scan_location)
            self.map_html_signal.emit(html)

        threading.Thread(target=_fetch, daemon=True).start()

    def _on_map_html(self, html: str):
        """Slot — always runs on main Qt thread.
        Serve the map HTML via the localhost HTTP server so Cesium gets a proper
        http:// origin and can load its skybox assets + our proxied tile imagery."""
        global _MAP_HTML_CONTENT
        if not self._map_view:
            return
        _MAP_HTML_CONTENT = html
        port = _ensure_map_server()
        from PyQt6.QtCore import QUrl
        self._map_view.setUrl(QUrl(f"http://127.0.0.1:{port}/map"))

    # ── Globe Builder ──────────────────────────────────────────────────────────
    # Realistic CesiumJS 3D globe. Key fixes vs. the old version:
    #   • Imagery is attached AFTER viewer creation via imageryLayers
    #     (Cesium 1.115 removed the constructor `imageryProvider` option — that
    #      is why the old globe showed only a flat teal ellipsoid).
    #   • Satellite basemap + place labels are served through our localhost proxy
    #     (Esri World Imagery) so there are zero CORS / CDN issues in WebEngine.
    #   • Adds ground + sky atmosphere, star-field skybox, subtle pin bloom,
    #     a pulsing user-location pin, and gentle auto-rotation until interaction.

    def _build_map_html(self, name, sci, rarity, coords, scan_location=None):
        scan_location = scan_location or {}
        port = _ensure_map_server()
        base = "http://127.0.0.1:" + str(port)

        obs_js = json.dumps([{"lat": c["lat"], "lng": c["lng"],
            "place": c.get("place", "Unknown"), "date": c.get("observed_on", "")}
            for c in coords])

        rc = {"Very Rare in NC": "#8e5fb5", "Rare in NC": "#c0392b",
              "Uncommon in NC": "#d4a017", "Common in NC": "#e8720c",
              "Very Common in NC": "#6abf5e"}.get(rarity, "#e8720c")
        sci_html = ('<div style="color:#7a9060;font-style:italic;font-size:11px;margin-bottom:4px">' + sci + '</div>'
                    if sci and sci not in ("Unknown", "N/A", "") else "")
        loc_dot = ('<br/><span class="dot" style="background:' + C_ACCENT + '"></span>Your location'
                   if scan_location.get("lat") else "")

        # Camera centre + altitude
        if scan_location.get("lat"):
            cam_lon, cam_lat, cam_alt = scan_location["lng"], scan_location["lat"], 14000000
        elif coords:
            cam_lon = sum(c["lng"] for c in coords) / len(coords)
            cam_lat = sum(c["lat"] for c in coords) / len(coords)
            cam_alt = 20000000
        else:
            cam_lon, cam_lat, cam_alt = -79.0, 35.5, 20000000

        # Pulsing user-location pin
        user_js = ""
        if scan_location.get("lat"):
            city  = scan_location.get("city", "")
            state = scan_location.get("region", "")
            label = (city + ", " + state) if city else "Your scan location"
            ulat, ulng = scan_location["lat"], scan_location["lng"]
            user_js = (
                "var _t0=viewer.clock.currentTime;"
                "viewer.entities.add({"
                "name:'Your Scan Location',"
                "position:Cesium.Cartesian3.fromDegrees(" + str(ulng) + "," + str(ulat) + "),"
                "point:{pixelSize:new Cesium.CallbackProperty(function(time){"
                "  var s=Cesium.JulianDate.secondsDifference(time,_t0);"
                "  return 13+4*Math.sin(s*3.0);},false),"
                "color:Cesium.Color.fromCssColorString('" + C_ACCENT + "'),"
                "outlineColor:Cesium.Color.WHITE,outlineWidth:2,"
                "heightReference:Cesium.HeightReference.CLAMP_TO_GROUND,"
                "disableDepthTestDistance:Number.POSITIVE_INFINITY},"
                "label:{text:'" + label + "',font:'bold 13px sans-serif',"
                "fillColor:Cesium.Color.fromCssColorString('" + C_TEXT + "'),"
                "outlineColor:Cesium.Color.fromCssColorString('" + C_BG + "'),"
                "outlineWidth:3,style:Cesium.LabelStyle.FILL_AND_OUTLINE,"
                "pixelOffset:new Cesium.Cartesian2(0,-22),"
                "disableDepthTestDistance:Number.POSITIVE_INFINITY},"
                "description:'<div style=\"color:#f0ead8;font-family:sans-serif;"
                "font-size:14px;line-height:1.6;padding:4px 2px\">Your scan location:<br/>"
                "<span style=\"font-weight:700;color:#e8720c\">" + label + "</span></div>',"
                "});"
            )

        cver = "1.115"
        cbase = "https://cesium.com/downloads/cesiumjs/releases/" + cver + "/Build/Cesium/"

        L = []
        L.append("<!DOCTYPE html><html><head><meta charset=\"utf-8\"/>")
        L.append("<script>window.CESIUM_BASE_URL='" + cbase + "';</script>")
        L.append("<script src=\"" + cbase + "Cesium.js\"></script>")
        L.append("<link href=\"" + cbase + "Widgets/widgets.css\" rel=\"stylesheet\"/>")
        L.append("<style>*{margin:0;padding:0;box-sizing:border-box}")
        L.append("html,body,#c{width:100%;height:100%;overflow:hidden;background:#000}")
        L.append(".cesium-widget-credits,.cesium-credit-logoContainer{display:none!important}")
        L.append(".lg{position:fixed;top:14px;left:14px;background:rgba(26,31,20,.92);"
                 "border:1px solid " + C_BORDER + ";border-radius:10px;padding:12px 16px;"
                 "color:" + C_TEXT + ";font-size:11px;line-height:1.9;z-index:900;"
                 "max-width:230px;font-family:sans-serif}")
        L.append(".nm{font-size:14px;font-weight:800;color:" + C_TEXT + ";display:block;margin-bottom:3px}")
        L.append(".dot{display:inline-block;width:10px;height:10px;border-radius:50%;"
                 "margin-right:6px;vertical-align:middle}")
        L.append(".hint{position:fixed;bottom:12px;left:14px;color:" + C_SUBTEXT + ";"
                 "font-size:10px;font-family:sans-serif;z-index:1000;opacity:.75}</style>")
        L.append("</head><body>")
        L.append("<div id=\"c\"></div>")
        L.append("<div class=\"lg\">")
        L.append("  <span class=\"nm\">" + name + "</span>" + sci_html)
        L.append("  <span style=\"color:" + rc + "\">\u25c8 " + (rarity or "\u2014") + "</span><br/>")
        L.append("  <span class=\"dot\" style=\"background:#6abf5e\"></span>" +
                 str(len(coords)) + " iNaturalist obs" + loc_dot)
        L.append("</div>")
        L.append("<div class=\"hint\">Drag to rotate \u00b7 scroll to zoom \u00b7 click a dot for details</div>")
        L.append("<script>")
        if CESIUM_TOKEN:
            L.append("Cesium.Ion.defaultAccessToken='" + CESIUM_TOKEN + "';")
        # Viewer WITHOUT constructor imagery (that option was removed in 1.115).
        L.append("var viewer=new Cesium.Viewer('c',{")
        L.append("  baseLayer:false,")
        L.append("  baseLayerPicker:false,geocoder:false,homeButton:false,")
        L.append("  sceneModePicker:false,navigationHelpButton:false,")
        L.append("  animation:false,timeline:false,fullscreenButton:false,")
        L.append("  infoBox:true,selectionIndicator:true,")
        L.append("  contextOptions:{webgl:{alpha:false}},")
        L.append("});")
        L.append("var scene=viewer.scene;")
        # Realistic satellite basemap via local proxy (Esri World Imagery)
        L.append("viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({")
        L.append("  url:'" + base + "/tiles/{z}/{y}/{x}',")
        L.append("  tilingScheme:new Cesium.WebMercatorTilingScheme(),")
        L.append("  minimumLevel:0,maximumLevel:18,")
        L.append("  credit:'Esri, Maxar, Earthstar Geographics'")
        L.append("}));")
        # Semi-transparent place-name / boundary label overlay
        L.append("var _lbl=viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({")
        L.append("  url:'" + base + "/labels/{z}/{y}/{x}',")
        L.append("  tilingScheme:new Cesium.WebMercatorTilingScheme(),")
        L.append("  minimumLevel:0,maximumLevel:18")
        L.append("}));")
        L.append("_lbl.alpha=0.8;")
        # Atmosphere / lighting / stars for realism
        L.append("scene.globe.showGroundAtmosphere=true;")
        L.append("scene.globe.enableLighting=false;")
        L.append("scene.globe.baseColor=Cesium.Color.fromCssColorString('#0a1626');")
        L.append("scene.backgroundColor=Cesium.Color.BLACK;")
        L.append("scene.skyAtmosphere.show=true;")
        L.append("try{scene.skyAtmosphere.atmosphereLightIntensity=12.0;}catch(e){}")
        L.append("scene.skyBox.show=true;")
        L.append("scene.sun.show=true;")
        L.append("scene.moon.show=false;")
        L.append("scene.fog.enabled=true;")
        L.append("scene.highDynamicRange=true;")
        # Subtle bloom so the observation dots glow against the Earth
        L.append("try{var bl=scene.postProcessStages.bloom;bl.enabled=true;")
        L.append("bl.uniforms.glowOnly=false;bl.uniforms.contrast=118;")
        L.append("bl.uniforms.brightness=-0.45;bl.uniforms.delta=1.2;")
        L.append("bl.uniforms.sigma=2.2;bl.uniforms.stepSize=1.0;}catch(e){}")
        # Observation pins
        L.append("var obsData=" + obs_js + ";")
        L.append("obsData.forEach(function(obs){")
        L.append("  viewer.entities.add({")
        L.append("    name:'iNaturalist Observation',")
        L.append("    position:Cesium.Cartesian3.fromDegrees(obs.lng,obs.lat),")
        L.append("    point:{pixelSize:8,color:Cesium.Color.fromCssColorString('#7CFF6B').withAlpha(0.92),")
        L.append("      outlineColor:Cesium.Color.fromCssColorString('#0a1a06'),outlineWidth:1.5,")
        L.append("      heightReference:Cesium.HeightReference.CLAMP_TO_GROUND},")
        L.append("    description:'<div style=\"color:#f0ead8;font-family:sans-serif;"
                 "font-size:14px;line-height:1.6;padding:4px 2px\">"
                 "<div style=\"font-weight:700;color:#7CFF6B;margin-bottom:4px\">'+obs.place+'</div>"
                 "<div style=\"color:#b9c6a5\">Observed: '+obs.date+'</div></div>',")
        L.append("  });")
        L.append("});")
        L.append(user_js)
        # Initial framed view
        L.append("viewer.camera.setView({")
        L.append("  destination:Cesium.Cartesian3.fromDegrees(" +
                 str(cam_lon) + "," + str(cam_lat) + "," + str(cam_alt) + "),")
        L.append("  orientation:{heading:0.0,pitch:Cesium.Math.toRadians(-90),roll:0.0},")
        L.append("});")
        # Gentle auto-rotation until the user interacts
        L.append("var _spin=true;")
        L.append("scene.preRender.addEventListener(function(){")
        L.append("  if(_spin){viewer.camera.rotate(Cesium.Cartesian3.UNIT_Z,-0.0008);}")
        L.append("});")
        L.append("['pointerdown','wheel','touchstart'].forEach(function(ev){")
        L.append("  scene.canvas.addEventListener(ev,function(){_spin=false;});")
        L.append("});")
        L.append("</script></body></html>")
        return "\n".join(L)

    def _build_gallery_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setFixedHeight(40)
        header.setStyleSheet(f"background: {C_PANEL};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 0, 16, 0)
        title = QLabel("SCAN GALLERY")
        title.setStyleSheet(
            f"color: {C_SUBTEXT}; font-size: 10px; font-weight: 700; letter-spacing: 2px;")
        self._gallery_count_lbl = QLabel("")
        self._gallery_count_lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 10px;")
        hl.addWidget(title)
        hl.addStretch()
        hl.addWidget(self._gallery_count_lbl)
        layout.addWidget(header)

        # Scrollable grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")

        self._gallery_container = QWidget()
        self._gallery_container.setStyleSheet("background: transparent;")
        self._gallery_grid = QVBoxLayout(self._gallery_container)
        self._gallery_grid.setContentsMargins(12, 12, 12, 12)
        self._gallery_grid.setSpacing(8)

        placeholder = QLabel("Scans will appear here after you scan species")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 13px; padding: 60px 20px;")
        self._gallery_grid.addWidget(placeholder)
        self._gallery_grid.addStretch()

        scroll.setWidget(self._gallery_container)
        layout.addWidget(scroll, stretch=1)

        self._gallery_scroll = scroll
        return widget

    def _refresh_gallery(self):
        """Rebuild the gallery grid from saved scan images."""
        # Clear existing
        while self._gallery_grid.count():
            item = self._gallery_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Get all entries that have a saved image
        entries_with_images = [
            e for e in reversed(self._collection)
            if e.get("saved_image_path") and
               Path(e["saved_image_path"]).exists()
        ]

        if not entries_with_images:
            msg = QLabel("No scan images yet.\nScan a species to build your gallery!")
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            msg.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 13px; padding: 60px 20px;")
            self._gallery_grid.addWidget(msg)
            self._gallery_grid.addStretch()
            self._gallery_count_lbl.setText("")
            return

        self._gallery_count_lbl.setText(f"{len(entries_with_images)} scans")

        # Build rows of 3 thumbnails
        COLS = 3
        THUMB = GalleryCard.THUMB

        row_widget = None
        row_layout = None

        for i, entry in enumerate(entries_with_images):
            if i % COLS == 0:
                row_widget = QWidget()
                row_widget.setStyleSheet("background: transparent;")
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)
                self._gallery_grid.addWidget(row_widget)

            # Thumbnail card
            card = GalleryCard(entry)
            card.clicked_signal.connect(lambda e: (self._on_collection_click(e),
                                                    self._switch_tab(0)))
            card.delete_signal.connect(self._on_gallery_delete)
            row_layout.addWidget(card)

        # Pad last row if needed
        remainder = len(entries_with_images) % COLS
        if remainder and row_layout:
            for _ in range(COLS - remainder):
                spacer = QWidget()
                spacer.setFixedSize(THUMB, THUMB + 36)
                spacer.setStyleSheet("background: transparent;")
                row_layout.addWidget(spacer)

        self._gallery_grid.addStretch()

    def _on_gallery_delete(self, entry: dict):
        """Delete a scan image and remove saved_image_path from the collection."""
        ts = entry.get("timestamp", "")
        if not ts:
            return

        # Delete the image file
        img_path = entry.get("saved_image_path", "")
        if img_path:
            try:
                Path(img_path).unlink(missing_ok=True)
            except Exception as e:
                print(f"[Gallery delete] {e}")

        # Remove saved_image_path from collection entry (keep the entry itself)
        for e in self._collection:
            if e.get("timestamp") == ts:
                e.pop("saved_image_path", None)
                break
        self._save_collection()

        # Refresh the gallery to reflect the deletion
        self._refresh_gallery()

    def _build_chat_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._chat_scroll = scroll

        self._chat_content = QWidget()
        self._chat_inner   = QVBoxLayout(self._chat_content)
        self._chat_inner.setContentsMargins(16, 16, 16, 8)
        self._chat_inner.setSpacing(10)

        intro = QLabel("Ask follow-up questions about\nyour last scan.")
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        intro.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 13px; padding: 40px 0;")
        self._chat_inner.addWidget(intro)
        self._chat_inner.addStretch()

        scroll.setWidget(self._chat_content)
        layout.addWidget(scroll, stretch=1)

        input_bar = QFrame()
        input_bar.setFixedHeight(58)
        input_bar.setStyleSheet(
            f"background: {C_PANEL}; border-top: 1px solid {C_BORDER};")
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
            QPushButton:hover {{ background: #f08020; }}
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
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(
            self._camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self._camera_label.setPixmap(scaled)
        if self._scan_overlay:
            self._scan_overlay.setGeometry(self._camera_label.geometry())

    # ── Model Loading ──────────────────────────────────────────────────────────

    def _load_models_async(self):
        def _load():
            try:
                self._custom_model, self._custom_label_map, self._custom_device = \
                    load_custom_model()
                self._bioclip = load_bioclip()
                self._client = OpenAI(
                    api_key=GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1")
                self._models_loaded = True
                self.model_status_signal.emit("ready")
            except Exception as e:
                self.model_status_signal.emit(f"error:{e}")
        threading.Thread(target=_load, daemon=True).start()

    def _on_model_status(self, status: str):
        if status == "ready":
            # Tell boot screen models are done → it will finish its animation
            if hasattr(self, "_boot"):
                try:
                    self._boot.mark_ready()
                except RuntimeError:
                    pass
            self._loading_lbl.setText("✓  Models ready")
            self._loading_lbl.setStyleSheet(f"color: {C_GREEN}; font-size: 11px;")
            QTimer.singleShot(2500, self._loading_lbl.hide)
        elif status.startswith("error:"):
            msg = status[6:]
            self._loading_lbl.setText(f"⚠  {msg[:40]}")
            self._loading_lbl.setStyleSheet(f"color: {C_RED}; font-size: 11px;")

    # ── Scan ───────────────────────────────────────────────────────────────────

    def _refresh_sound_icons(self):
        if hasattr(self, "_sound_btn"):
            on = _SOUND_ENABLED
            self._sound_btn.setText("🔊" if on else "🔇")
            self._sound_btn.setStyleSheet(
                f"color: {C_ACCENT if on else C_SUBTEXT}; font-size: 17px;")

    def _toggle_sound(self):
        global _SOUND_ENABLED
        _SOUND_ENABLED = not _SOUND_ENABLED
        _save_sound_prefs()
        self._refresh_sound_icons()
        if _SOUND_ENABLED:
            _play(_WAV_SUCCESS)

    def _on_scan(self):
        if not self._models_loaded:
            self._status_lbl.setText("Still loading models...")
            return
        if self._last_frame is None:
            self._status_lbl.setText("No camera frame available")
            return

        _play(_WAV_SCAN)
        self._scan_cancelled = False  # reset for new scan
        frame = self._last_frame.copy()
        self._scan_btn.start_scanning()
        self._cancel_btn.show()
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
            bioclip=self._bioclip)
        self._analysis_worker.result_ready.connect(self._on_result)
        self._analysis_worker.error_occurred.connect(self._on_error)
        self._analysis_worker.start()

    def _on_cancel_scan(self):
        """Cancel an in-progress scan safely.
        We NEVER call terminate() on AnalysisWorker — it runs TensorFlow/PyTorch
        inference in C++ native code and terminating mid-inference causes a segfault.
        Instead we set a flag so the result is silently discarded when it arrives."""
        self._scan_cancelled = True
        # Don't touch the worker thread — let it finish naturally
        self._scan_btn.stop_scanning()
        self._cancel_btn.hide()
        if self._scan_overlay:
            self._scan_overlay.stop()
            self._scan_overlay.hide()
        self._status_lbl.setText("Scan cancelled")

    def _on_result(self, result):
        # If user cancelled while the worker was finishing, discard the result
        if self._scan_cancelled:
            self._scan_cancelled = False
            return

        self._scan_btn.stop_scanning()
        self._cancel_btn.hide()
        self._status_lbl.setText(f"Identified: {result['name']}")
        if self._scan_overlay:
            self._scan_overlay.stop()
            self._scan_overlay.hide()

        _play(_WAV_SUCCESS)

        # Save scan image permanently to ~/.naturedex_images/
        saved_path = self._save_scan_image(result.get("image_path", ""),
                                           result.get("timestamp", ""))
        if saved_path:
            result["saved_image_path"] = saved_path
        sticker_saved = self._save_sticker(result.get("sticker_tmp", ""),
                                           result.get("timestamp", ""))
        if sticker_saved:
            result["sticker_path"] = sticker_saved

        self._current_result = result
        self._chat_history   = []
        self._collection.append(result)
        self._save_collection()
        self._add_collection_card(result, prepend=True)
        self._species_count_lbl.setText(f"{len(self._collection)} discovered")
        self._refresh_category_filter_options()
        self._apply_filters()
        self._check_achievements()
        self._update_streak_on_scan()
        self._evaluate_challenges(fire=True)
        self._render_entry(result)
        self._switch_tab(0)
        self._reset_chat()
        self._report_btn.setEnabled(True)
        self._share_btn.setEnabled(True)
        # Refresh gallery in background so it's ready when user navigates to it
        QTimer.singleShot(500, self._refresh_gallery)

    def _save_scan_image(self, tmp_path: str, timestamp: str) -> str:
        """Copy the temp scan image to a permanent location.
        Returns the saved path, or empty string on failure."""
        try:
            import shutil
            images_dir = Path.home() / ".naturedex_images"
            images_dir.mkdir(exist_ok=True)
            # Use timestamp as filename so each scan is unique
            safe_ts = timestamp.replace(":", "-").replace(".", "-")[:19]
            dest = images_dir / f"scan_{safe_ts}.jpg"
            if tmp_path and Path(tmp_path).exists():
                shutil.copy2(tmp_path, dest)
                return str(dest)
        except Exception as e:
            print(f"[Image save] {e}")
        return ""

    def _save_sticker(self, tmp_sticker: str, timestamp: str) -> str:
        """Copy the temp cut-out sticker to a permanent PNG. Returns path or ""."""
        try:
            import shutil
            images_dir = Path.home() / ".naturedex_images"
            images_dir.mkdir(exist_ok=True)
            safe_ts = timestamp.replace(":", "-").replace(".", "-")[:19]
            dest = images_dir / f"sticker_{safe_ts}.png"
            if tmp_sticker and Path(tmp_sticker).exists():
                shutil.copy2(tmp_sticker, dest)
                return str(dest)
        except Exception as e:
            print(f"[Sticker save] {e}")
        return ""

    def _on_error(self, msg):
        if self._scan_cancelled:
            self._scan_cancelled = False
            return
        self._scan_btn.stop_scanning()
        self._cancel_btn.hide()
        self._status_lbl.setText(f"Error: {msg}")
        if self._scan_overlay:
            self._scan_overlay.stop()
            self._scan_overlay.hide()

    # ── Entry Rendering (with fade-in animations) ──────────────────────────────

    def _clear_entry(self):
        self._entry_anims.clear()  # release old animation references
        while self._entry_inner.count():
            item = self._entry_inner.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _fade_in(self, widget, delay_ms: int):
        """Fade a widget from invisible to fully visible after delay_ms."""
        eff = QGraphicsOpacityEffect(widget)
        eff.setOpacity(0.0)
        widget.setGraphicsEffect(eff)
        anim = QPropertyAnimation(eff, b"opacity")
        anim.setDuration(280)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._entry_anims.append((anim, eff))  # keep alive
        QTimer.singleShot(delay_ms, anim.start)

    def _render_entry(self, result):
        self._clear_entry()
        entry = result.get("entry", {})

        # ── Low-confidence warning (uncertain identification) ──
        if result.get("low_confidence"):
            warn = QFrame()
            warn.setStyleSheet(f"""
                QFrame {{
                    background: #2a2410;
                    border: 1px solid {C_YELLOW};
                    border-radius: 10px;
                }}
            """)
            wl = QVBoxLayout(warn)
            wl.setContentsMargins(14, 10, 14, 10)
            wl.setSpacing(2)
            wt = QLabel(f"⚠  Not sure about this one ({result.get('confidence', 0):.0f}%)")
            wt.setStyleSheet(f"color: {C_YELLOW}; font-size: 12px; font-weight: 700;")
            ws = QLabel("The lighting or framing may be throwing it off. Try a clearer, "
                        "closer, well-lit shot — or check the other possibilities below.")
            ws.setWordWrap(True)
            ws.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 11px;")
            wl.addWidget(wt)
            wl.addWidget(ws)
            self._entry_inner.addWidget(warn)

        # ── Hero sticker — the cut-out of the user's own photo
        sticker = result.get("sticker_path", "")
        if sticker and Path(sticker).exists():
            hero = QLabel()
            hero.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hero.setFixedHeight(190)
            hero.setStyleSheet("background: transparent;")
            _px = QPixmap(sticker)
            if not _px.isNull():
                _px = _px.scaled(260, 180,
                                 Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
                hero.setPixmap(_px)
                self._entry_inner.addWidget(hero)

        # ── Name header
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

        # Show speaker button for any real scientific name (genus-only or binomial)
        has_real_sci = bool(sci and sci not in ("Unknown", "N/A", "")
                            and len(sci.strip()) > 2)

        if sci and sci not in ("Unknown", "N/A", ""):
            sci_widget = QWidget()
            sci_widget.setStyleSheet("background: transparent;")
            sci_flow = FlowLayout(sci_widget, spacing=8)
            _sp = sci_widget.sizePolicy(); _sp.setHeightForWidth(True)
            sci_widget.setSizePolicy(_sp)

            sci_lbl = QLabel(sci)
            sci_lbl.setStyleSheet(
                f"color: {C_TEXT}; font-size: 12px; font-style: italic; letter-spacing: 0.5px;")
            sci_flow.addWidget(sci_lbl)

            # Speaker button — show for any real scientific name
            if has_real_sci:
                speak_btn = QPushButton("🔊")
                speak_btn.setFixedSize(24, 22)
                speak_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                speak_btn.setToolTip(f"Hear pronunciation of '{sci}'")
                speak_btn.setStyleSheet(f"""
                    QPushButton {{
                        background: transparent; border: none;
                        font-size: 13px; padding: 0;
                    }}
                    QPushButton:hover {{ background: {C_CARD}; border-radius: 4px; }}
                """)
                speak_str = result.get("phonetic_speech", "") or sci
                speak_btn.clicked.connect(
                    lambda checked, s=speak_str: subprocess.Popen(
                        ["say", "-v", "Samantha", "-r", "105", s],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
                sci_flow.addWidget(speak_btn)

            phonetic = result.get("phonetic", "") if has_real_sci else ""
            if phonetic:
                phon_lbl = QLabel(phonetic)
                phon_lbl.setStyleSheet(
                    f"color: {C_SUBTEXT}; font-size: 10px; letter-spacing: 0.3px;")
                sci_flow.addWidget(phon_lbl)

            nf_layout.addWidget(sci_widget)

        conf       = result["confidence"]
        conf_color = C_GREEN if conf >= 75 else C_YELLOW if conf >= 50 else C_RED

        # All metadata badges live in a flow layout so they wrap on narrow panels
        meta_widget = QWidget()
        meta_widget.setStyleSheet("background: transparent;")
        meta_flow = FlowLayout(meta_widget, spacing=8)
        _mp = meta_widget.sizePolicy(); _mp.setHeightForWidth(True)
        meta_widget.setSizePolicy(_mp)

        conf_lbl = QLabel(f"CONFIDENCE  {conf:.1f}%")
        conf_lbl.setStyleSheet(
            f"color: {conf_color}; font-size: 11px; font-weight: 700; letter-spacing: 1px;")
        meta_flow.addWidget(conf_lbl)

        if cat:
            cat_badge = QLabel(f"  {cat}  ")
            cat_badge.setStyleSheet(
                f"background: {C_BORDER}; color: {C_ACCENT}; font-size: 10px;"
                f" font-weight: 700; border-radius: 4px; padding: 2px 6px;"
                f" letter-spacing: 1px;")
            meta_flow.addWidget(cat_badge)

        used_custom = result.get("used_custom_model", False)
        src         = result.get("model_source", "")
        if "BioCLIP" in src:
            badge_txt, badge_hot = "🌍 BioCLIP · global", True
        elif used_custom:
            badge_txt, badge_hot = "🌿 NatureDex Model", True
        else:
            badge_txt, badge_hot = "⚙ Unidentified", False
        model_badge = QLabel(badge_txt)
        model_badge.setStyleSheet(
            f"color: {C_ACCENT if badge_hot else C_SUBTEXT}; font-size: 9px;"
            f" font-weight: 700; letter-spacing: 0.5px;")
        meta_flow.addWidget(model_badge)

        if result.get("native_nc"):
            local_badge = QLabel("📍 Local species")
            local_badge.setStyleSheet(
                f"color: {C_ACCENT}; font-size: 9px; font-weight: 700; letter-spacing: 0.5px;")
            meta_flow.addWidget(local_badge)

        rarity     = result.get("local_rarity") or result.get("rarity", "")
        inat       = result.get("inat", {})
        global_obs = inat.get("observations_count", 0)
        if rarity:
            if "Very Rare" in rarity or "Not Recorded" in rarity:
                rarity_color = C_PURPLE
            elif "Rare" in rarity:
                rarity_color = C_RED
            elif "Uncommon" in rarity:
                rarity_color = C_YELLOW
            else:
                rarity_color = C_ACCENT
            rarity_lbl = QLabel(f"◈  {rarity}")
            rarity_lbl.setStyleSheet(
                f"color: {rarity_color}; font-size: 11px; font-weight: 700;")
            meta_flow.addWidget(rarity_lbl)
            if global_obs:
                global_lbl = QLabel(f"·  {global_obs:,} obs")
                global_lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 10px;")
                meta_flow.addWidget(global_lbl)

        for tag in entry.get("type_tags", [])[:4]:
            t_lbl = QLabel(tag)
            t_lbl.setStyleSheet(
                f"background: #1a3a2a; color: {C_GREEN}; font-size: 10px;"
                f" font-weight: 600; border-radius: 3px; padding: 2px 8px;")
            meta_flow.addWidget(t_lbl)

        nf_layout.addWidget(meta_widget)

        self._entry_inner.addWidget(name_frame)
        self._fade_in(name_frame, 0)           # ← fade in immediately

        # ── Invasive species alert ────────────────────────────────────────────
        inv = result.get("invasive")
        if not inv:  # older scans (pre-feature) — detect on the fly
            inv = check_invasive(result.get("name", ""), sci,
                                 result.get("raw_label", ""))
        if inv:
            banner = self._make_invasive_banner(inv, result)
            self._entry_inner.addWidget(banner)
            self._fade_in(banner, 40)

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
            self._fade_in(desc_lbl, 80)        # ← fade in 80ms later

        nc_context = entry.get("north_carolina_context", "")
        nc_obs = result.get("nc_observations", None)
        if nc_obs is not None and nc_obs > 0 and nc_context and nc_context not in ("Unknown", "N/A"):
            nc_context = f"{nc_context}  ({nc_obs:,} research-grade iNaturalist observations in NC)"
        elif nc_obs == 0 and nc_context not in ("Unknown", "N/A", ""):
            nc_context = f"{nc_context}  (No research-grade iNaturalist observations recorded in NC)"

        # Determine if this is a non-living object (so we can adjust labels)
        category  = entry.get("category", "").lower()
        is_object = any(w in category for w in
                        ("object", "food", "vehicle", "furniture",
                         "tool", "device", "clothing", "instrument"))

        fields = [
            ("🌿  HABITAT",        entry.get("habitat", "")),
            ("🍃  DIET",           entry.get("diet", "")),
            ("🐾  BEHAVIOR",       entry.get("behavior", "")),
            ("🔴  CONSERVATION",   entry.get("conservation_status", "")),
            ("📍  NORTH CAROLINA", nc_context),
            ("⚡  FUN FACT",       entry.get("fun_fact", "")),
        ]
        delay = 160
        for icon_label, value in fields:
            # Skip blank, Unknown, and bare N/A values
            if not value:
                continue
            cleaned = value.strip()
            if cleaned.lower() in ("unknown", "n/a", ""):
                continue
            # For conservation on non-living objects, skip entirely
            if "CONSERVATION" in icon_label and is_object:
                continue
            card = self._make_info_card(icon_label, cleaned)
            self._entry_inner.addWidget(card)
            self._fade_in(card, delay)
            delay += 60

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
            alt_title.setStyleSheet(
                f"color: {C_SUBTEXT}; font-size: 10px; font-weight: 700; letter-spacing: 2px;")
            alt_layout.addWidget(alt_title)
            for a in alts:
                a_conf  = a["confidence"]
                a_color = C_GREEN if a_conf >= 20 else C_SUBTEXT
                row     = QHBoxLayout()
                n_lbl   = QLabel(f"• {a['name']}")
                n_lbl.setStyleSheet(f"color: {C_TEXT}; font-size: 12px;")
                c_lbl   = QLabel(f"{a_conf:.1f}%")
                c_lbl.setStyleSheet(
                    f"color: {a_color}; font-size: 11px; font-weight: 600;")
                row.addWidget(n_lbl)
                row.addStretch()
                row.addWidget(c_lbl)
                alt_layout.addLayout(row)
            self._entry_inner.addWidget(alt_frame)
            self._fade_in(alt_frame, delay)

        self._entry_inner.addStretch()

    def _make_invasive_banner(self, inv: dict, result: dict) -> QFrame:
        """Build the prominent invasive-species alert card with a Report button."""
        sev   = inv.get("severity", "Established")
        color = _invasive_severity_color(sev)

        banner = QFrame()
        banner.setObjectName("invBanner")
        banner.setStyleSheet(f"""
            QFrame#invBanner {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {C_CARD}, stop:1 #2a1810);
                border: 1px solid {color};
                border-left: 4px solid {color};
                border-radius: 8px;
            }}
        """)
        v = QVBoxLayout(banner)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(7)

        # Header row: warning icon + "INVASIVE SPECIES" + severity pill
        head = QHBoxLayout()
        head.setSpacing(8)
        icon = QLabel("⚠")
        icon.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: 900;")
        head.addWidget(icon)
        title = QLabel("INVASIVE SPECIES")
        title.setStyleSheet(
            f"color: {color}; font-size: 12px; font-weight: 900; letter-spacing: 2px;")
        head.addWidget(title)
        head.addStretch()
        pill = QLabel(f"  {sev.upper()}  ")
        pill.setStyleSheet(f"""
            background: {color};
            color: #14100a;
            font-size: 9px;
            font-weight: 800;
            border-radius: 4px;
            padding: 2px 6px;
            letter-spacing: 1px;
        """)
        head.addWidget(pill)
        v.addLayout(head)

        # Why it matters
        why = QLabel(inv.get("why", ""))
        why.setWordWrap(True)
        why.setStyleSheet(f"color: {C_TEXT}; font-size: 12px; line-height: 1.5;")
        v.addWidget(why)

        # Report row: button + channel guidance
        report_row = QHBoxLayout()
        report_row.setSpacing(10)

        report_btn = QPushButton("⚑  Report This Sighting")
        report_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        report_btn.setStyleSheet(f"""
            QPushButton {{
                background: {color};
                color: #14100a;
                border: none;
                border-radius: 7px;
                padding: 7px 16px;
                font-size: 12px;
                font-weight: 800;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{ background: {C_TEXT}; }}
        """)
        report_btn.clicked.connect(
            lambda _=False, i=inv, r=result, b=report_btn: self._on_report_invasive(i, r, b))
        report_row.addWidget(report_btn)
        report_row.addStretch()
        v.addLayout(report_row)

        # Where to report
        chan = QLabel(inv.get("report_to", ""))
        chan.setWordWrap(True)
        chan.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 10px; font-style: italic;")
        v.addWidget(chan)

        return banner

    def _on_report_invasive(self, inv: dict, result: dict, btn: QPushButton):
        """Log an invasive-species report locally and confirm to the user."""
        self._save_invasive_report(inv, result)
        try:
            btn.setText("✓  Sighting Reported — Thank You!")
            btn.setEnabled(False)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {C_GREEN};
                    color: #14100a;
                    border: none;
                    border-radius: 7px;
                    padding: 7px 16px;
                    font-size: 12px;
                    font-weight: 800;
                }}
            """)
        except RuntimeError:
            pass
        # Fire a toast for that satisfying feedback in a demo
        if self._toast:
            self._toast.show_achievement("🛡️", "Invasive Species Reported")

    def _save_invasive_report(self, inv: dict, result: dict):
        record = {
            "timestamp":       datetime.datetime.now().isoformat(),
            "common_name":     inv.get("common", result.get("name", "")),
            "scientific_name": inv.get("scientific_name", ""),
            "severity":        inv.get("severity", ""),
            "type":            inv.get("type", ""),
            "report_to":       inv.get("report_to", ""),
            "scan_location":   result.get("scan_location", {}),
            "image_path":      result.get("saved_image_path")
                               or result.get("image_path", ""),
        }
        try:
            reports = []
            if INVASIVE_REPORTS_FILE.exists():
                reports = json.loads(INVASIVE_REPORTS_FILE.read_text())
            reports.append(record)
            INVASIVE_REPORTS_FILE.write_text(json.dumps(reports, indent=2))
            print(f"[Invasive report] Logged: {record['common_name']}")
        except Exception as e:
            print(f"Could not save invasive report: {e}")

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
        lbl.setStyleSheet(
            f"color: {C_ACCENT}; font-size: 9px; font-weight: 800; letter-spacing: 1.5px;")
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
        self._chat_history   = []
        self._render_entry(entry_data)
        self._switch_tab(0)
        self._reset_chat()
        self._report_btn.setEnabled(True)
        self._share_btn.setEnabled(True)

    def _on_delete_entry(self, timestamp):
        if not timestamp:
            return
        self._collection = [e for e in self._collection
                            if e.get("timestamp") != timestamp]
        self._save_collection()
        for i in range(self._collection_layout.count()):
            item   = self._collection_layout.itemAt(i)
            widget = item.widget() if item else None
            if isinstance(widget, CollectionCard) and \
               widget.entry_data.get("timestamp") == timestamp:
                widget.deleteLater()
                break
        self._species_count_lbl.setText(f"{len(self._collection)} discovered")
        self._refresh_category_filter_options()
        if self._current_result and \
           self._current_result.get("timestamp") == timestamp:
            self._current_result = None
            self._clear_entry()
            self._placeholder_lbl = QLabel(
                "Scan an object to generate\na NatureDex entry.")
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
        query    = self._search_input.text().strip().lower()
        category = self._category_filter.currentText()
        visible  = 0
        for i in range(self._collection_layout.count()):
            item   = self._collection_layout.itemAt(i)
            widget = item.widget() if item else None
            if not isinstance(widget, CollectionCard):
                continue
            ms = (query in widget._search_text) if query else True
            mc = (category == "All Categories") or (widget._category == category)
            widget.setVisible(ms and mc)
            if ms and mc:
                visible += 1
        self._empty_filter_lbl.setVisible(
            visible == 0 and len(self._collection) > 0)

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
            ACHIEVEMENTS_FILE.write_text(
                json.dumps(sorted(self._unlocked_achievements)))
        except Exception as e:
            print(f"Could not save achievements: {e}")

    def _check_achievements(self):
        coll   = self._collection
        count  = len(coll)
        n_cats = len({
            e.get("entry", {}).get("category", "Unknown") or "Unknown"
            for e in coll
        })

        # Rare / Very Rare near the user (falls back to NC rarity for old entries)
        n_rare = sum(
            1 for e in coll
            if "Rare near" in (e.get("local_rarity", "") or "")
            or any(w in e.get("rarity", "")
                   for w in ("Rare in NC", "Very Rare in NC"))
        )

        endangered_statuses = {"vulnerable", "endangered", "critically endangered"}
        n_endangered = sum(
            1 for e in coll
            if e.get("entry", {}).get("conservation_status", "").lower()
            in endangered_statuses
        )

        # Native to the user's area
        n_local = sum(1 for e in coll if e.get("native_nc"))

        # Cut-out stickers collected
        n_sticker = sum(1 for e in coll if e.get("sticker_path"))

        # Bird / mammal / insect / plant sweep
        groups = set()
        for e in coll:
            g = (e.get("entry", {}).get("iconic_group", "")
                 or e.get("entry", {}).get("category", "") or "").lower()
            if "av" in g or "bird" in g:            groups.add("bird")
            elif "mamm" in g or "beast" in g:       groups.add("mammal")
            elif "insect" in g or "arachn" in g or "bug" in g: groups.add("insect")
            elif "plant" in g or "flora" in g or "fung" in g:  groups.add("plant")
        n_sweep = len({"bird", "mammal", "insect", "plant"} & groups)

        # Distinct scan days / regions; best confidence
        n_days = len({(e.get("timestamp", "") or "")[:10] for e in coll if e.get("timestamp")})
        n_regions = len({
            (e.get("scan_location", {}) or {}).get("region", "")
            for e in coll if (e.get("scan_location", {}) or {}).get("region")
        })
        n_highconf = sum(1 for e in coll if (e.get("confidence", 0) or 0) >= 50)

        n_correct = 0
        if CORRECTIONS_FILE.exists():
            try:
                n_correct = len(json.loads(CORRECTIONS_FILE.read_text()))
            except Exception:
                pass

        counters = {
            "count": count, "category": n_cats, "rarity": n_rare,
            "endangered": n_endangered, "local": n_local, "sticker": n_sticker,
            "sweep": n_sweep, "streak": n_days, "region": n_regions,
            "highconf": n_highconf, "correct": n_correct,
        }

        newly = []
        for badge in ACHIEVEMENTS:
            if badge["id"] in self._unlocked_achievements:
                continue
            if counters.get(badge["type"], 0) >= badge["threshold"]:
                newly.append(badge)

        if not newly:
            return
        for badge in newly:
            self._unlocked_achievements.add(badge["id"])
        self._save_achievements()
        for idx, badge in enumerate(newly):
            def _fire(b=badge):
                _play(_WAV_UNLOCK)
                self._toast.show_achievement(b["icon"], b["name"])
            QTimer.singleShot(idx * 3500, _fire)

    def _show_nearby_index(self):
        # Which species has the user already caught? (match by taxon id or name)
        caught_taxa, caught_names = set(), set()
        for e in self._collection:
            inat = e.get("inat", {}) or {}
            if inat.get("taxon_id"):
                caught_taxa.add(inat["taxon_id"])
            for nm in (inat.get("scientific_name", ""),
                       e.get("entry", {}).get("scientific_name", ""),
                       e.get("name", "")):
                if nm:
                    caught_names.add(nm.strip().lower())

        def _is_caught(sp):
            if sp.get("taxon_id") in caught_taxa:
                return True
            for nm in (sp.get("scientific_name", ""), sp.get("common_name", "")):
                if nm and nm.strip().lower() in caught_names:
                    return True
            return False

        # ── Panel shell (mirrors the achievements panel) ──
        panel = QFrame(self)
        panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 12px;")
        panel.setFixedSize(480, 560)
        panel.move((self.width() - 480) // 2, (self.height() - 560) // 2)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(0)

        header_row = QHBoxLayout()
        title = QLabel("Nearby Index")
        title.setStyleSheet(f"color: {C_TEXT}; font-size: 16px; font-weight: 800;")
        close_btn = QLabel("✕")
        close_btn.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 16px;")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.mousePressEvent = lambda e: panel.deleteLater()
        header_row.addWidget(title)
        header_row.addStretch()
        header_row.addWidget(close_btn)
        layout.addLayout(header_row)

        subtitle = QLabel("Species to discover near you — 🔒 = still out there")
        subtitle.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 11px;")
        layout.addWidget(subtitle)
        layout.addSpacing(12)

        status = QLabel("Locating you and loading nearby species…")
        status.setStyleSheet(f"color: {C_ACCENT}; font-size: 12px; font-weight: 700;")
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status)
        layout.addSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        grid = QVBoxLayout(content)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)

        panel.show()

        loc = _get_user_location()
        if not loc.get("lat"):
            status.setText("Couldn't detect your location. Check your connection.")
            return
        area = loc.get("region") or loc.get("city") or "you"

        def _make_card(sp):
            caught = _is_caught(sp)
            card = QFrame()
            card.setFixedSize(138, 74)
            if caught:
                card.setStyleSheet(
                    "QFrame { background: #16321f; border: none; border-radius: 8px; }"
                    " QLabel { border: none; background: transparent; }")
            else:
                card.setStyleSheet(
                    f"QFrame {{ background: {C_BG}; border: 1px solid {C_BORDER};"
                    f" border-radius: 8px; }}"
                    " QLabel { border: none; background: transparent; }")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(10, 8, 10, 8)
            cl.setSpacing(3)
            top = QLabel("✓" if caught else "🔒")
            top.setStyleSheet(
                f"color: {C_GREEN if caught else C_SUBTEXT};"
                f" font-size: 13px; font-weight: 700;")
            cl.addWidget(top)
            name_lbl = QLabel(sp.get("common_name", "???"))
            name_lbl.setWordWrap(True)
            name_lbl.setStyleSheet(
                f"color: {C_GREEN if caught else C_SUBTEXT};"
                f" font-size: 11px; font-weight: {'700' if caught else '500'};")
            cl.addWidget(name_lbl)
            return card

        def _populate(species):
            if not species:
                status.setText("No nearby data found. Try again shortly.")
                return
            caught_n = sum(1 for sp in species if _is_caught(sp))
            status.setText(f"📍 {area}  ·  {caught_n}/{len(species)} discovered")
            row = None
            for i, sp in enumerate(species):
                if i % 3 == 0:
                    row_w = QWidget(); row_w.setStyleSheet("background: transparent;")
                    row = QHBoxLayout(row_w)
                    row.setContentsMargins(0, 0, 0, 0); row.setSpacing(8)
                    grid.addWidget(row_w)
                row.addWidget(_make_card(sp))
            if row is not None:
                while row.count() < 3:
                    filler = QWidget(); filler.setFixedSize(138, 74)
                    filler.setStyleSheet("background: transparent;")
                    row.addWidget(filler)
            grid.addStretch()

        self._nearby_worker = NearbyWorker(loc["lat"], loc["lng"])
        self._nearby_worker.done.connect(_populate)
        self._nearby_worker.start()

    def _load_challenges(self):
        if CHALLENGES_FILE.exists():
            try:
                return json.loads(CHALLENGES_FILE.read_text())
            except Exception:
                pass
        return {"streak": 0, "best": 0, "last_active": "", "date": "", "notified": []}

    def _save_challenges(self):
        try:
            CHALLENGES_FILE.write_text(json.dumps(self._challenge_state))
        except Exception as e:
            print(f"[Challenges] {e}")

    @staticmethod
    def _today_str():
        return datetime.date.today().isoformat()

    def _today_challenges(self):
        has_loc = True
        try:
            has_loc = _get_user_location().get("lat") is not None
        except Exception:
            pass
        return generate_daily_challenges(self._today_str(), has_loc)

    def _streak_is_valid(self):
        last = self._challenge_state.get("last_active", "")
        d = datetime.date.today()
        return last in (d.isoformat(),
                        (d - datetime.timedelta(days=1)).isoformat())

    def _refresh_streak_label(self):
        if hasattr(self, "_streak_lbl"):
            s = self._challenge_state.get("streak", 0) if self._streak_is_valid() else 0
            self._streak_lbl.setText(f"🔥 {s}" if s > 0 else "")

    def _update_streak_on_scan(self):
        today = datetime.date.today()
        st = self._challenge_state
        last = st.get("last_active", "")
        if last == today.isoformat():
            return
        yesterday = (today - datetime.timedelta(days=1)).isoformat()
        st["streak"] = st.get("streak", 0) + 1 if last == yesterday else 1
        st["last_active"] = today.isoformat()
        st["best"] = max(st.get("best", 0), st["streak"])
        self._save_challenges()

    def _evaluate_challenges(self, fire=True):
        today = self._today_str()
        st = self._challenge_state
        if st.get("date") != today:
            st["date"] = today
            st["notified"] = []
            self._save_challenges()
        today_scans = [e for e in self._collection
                       if (e.get("timestamp", "") or "")[:10] == today]
        challenges = self._today_challenges()
        newly = []
        for ch in challenges:
            done = challenge_progress(ch, today_scans, self._collection, today) >= ch["target"]
            if done and ch["id"] not in st["notified"]:
                st["notified"].append(ch["id"])
                newly.append(ch)
        if newly:
            self._save_challenges()
            if fire:
                all_done = all(
                    challenge_progress(c, today_scans, self._collection, today) >= c["target"]
                    for c in challenges)
                t = 0
                for i, ch in enumerate(newly):
                    def _fire(c=ch):
                        _play(_WAV_UNLOCK)
                        self._toast.show_achievement("🔥", "Challenge: " + c["text"])
                    QTimer.singleShot(i * 3500, _fire)
                    t = (i + 1) * 3500
                if all_done:
                    streak = st.get("streak", 0)
                    def _fire_all():
                        _play(_WAV_UNLOCK)
                        self._toast.show_achievement(
                            "🔥", f"All challenges done — {streak}-day streak!")
                    QTimer.singleShot(t, _fire_all)
        self._refresh_streak_label()

    def _show_challenges_panel(self):
        today = self._today_str()
        today_scans = [e for e in self._collection
                       if (e.get("timestamp", "") or "")[:10] == today]
        challenges = self._today_challenges()
        st = self._challenge_state
        streak = st.get("streak", 0) if self._streak_is_valid() else 0
        best = st.get("best", 0)

        panel = QFrame(self)
        panel.setObjectName("chalPanel")
        panel.setFixedSize(460, 470)
        panel.move((self.width() - 460) // 2, (self.height() - 470) // 2)
        panel.setStyleSheet(
            f"QFrame#chalPanel {{ background: {C_PANEL}; border: 1px solid {C_BORDER};"
            f" border-radius: 14px; }} QLabel {{ background: transparent; border: none; }}")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(6)

        hdr = QHBoxLayout()
        title = QLabel("Daily Challenges")
        title.setStyleSheet(f"color: {C_TEXT}; font-size: 17px; font-weight: 800;")
        close = QLabel("✕")
        close.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 16px;")
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.mousePressEvent = lambda e: panel.deleteLater()
        hdr.addWidget(title); hdr.addStretch(); hdr.addWidget(close)
        lay.addLayout(hdr)

        streak_lbl = QLabel(
            f"🔥 {streak}-day streak" + (f"    ·    best {best}" if best else ""))
        streak_lbl.setStyleSheet(f"color: {C_ACCENT2}; font-size: 14px; font-weight: 700;")
        lay.addWidget(streak_lbl)
        sub = QLabel("Complete all three today to keep your streak going.")
        sub.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 11px;")
        lay.addWidget(sub)
        lay.addSpacing(12)

        for ch in challenges:
            prog = challenge_progress(ch, today_scans, self._collection, today)
            done = prog >= ch["target"]
            pct = max(0, min(100, int(prog / ch["target"] * 100)))

            row = QFrame()
            row.setObjectName("chalRow")
            row.setStyleSheet(
                f"QFrame#chalRow {{ background: {'#16321f' if done else C_CARD};"
                f" border-radius: 10px; }} QLabel {{ background: transparent; border: none; }}")
            rl = QVBoxLayout(row)
            rl.setContentsMargins(14, 11, 14, 12)
            rl.setSpacing(8)

            top = QHBoxLayout()
            ic = QLabel("✅" if done else ch["icon"])
            ic.setStyleSheet("font-size: 18px;"); ic.setFixedWidth(30)
            txt = QLabel(ch["text"])
            txt.setStyleSheet(f"color: {C_TEXT}; font-size: 13px; font-weight: 600;")
            cnt = QLabel(f"{min(prog, ch['target'])}/{ch['target']}")
            cnt.setStyleSheet(
                f"color: {C_GREEN if done else C_SUBTEXT}; font-size: 12px; font-weight: 700;")
            top.addWidget(ic); top.addWidget(txt); top.addStretch(); top.addWidget(cnt)
            rl.addLayout(top)

            bar = QWidget()
            bar.setFixedHeight(6)
            bl = QHBoxLayout(bar); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(0)
            if pct > 0:
                fill = QFrame()
                fill.setStyleSheet(
                    f"background: {C_GREEN if done else C_ACCENT}; border-radius: 3px;")
                bl.addWidget(fill, pct)
            if pct < 100:
                rest = QFrame()
                rest.setStyleSheet(f"background: {C_BORDER}; border-radius: 3px;")
                bl.addWidget(rest, 100 - pct)
            rl.addWidget(bar)
            lay.addWidget(row)
            lay.addSpacing(6)

        lay.addStretch()
        panel.show()
        panel.raise_()

    def _show_badges_panel(self):
        # ── Compute current stats for progress bars ────────────────────────────
        coll   = self._collection
        count  = len(coll)
        n_cats = len({
            e.get("entry", {}).get("category", "Unknown") or "Unknown"
            for e in coll
        })
        n_rare = sum(
            1 for e in coll
            if "Rare near" in (e.get("local_rarity", "") or "")
            or any(w in e.get("rarity", "") for w in ("Rare in NC", "Very Rare in NC"))
        )
        endangered_statuses = {"vulnerable", "endangered", "critically endangered"}
        n_endangered = sum(
            1 for e in coll
            if e.get("entry", {}).get("conservation_status", "").lower()
            in endangered_statuses
        )
        n_local   = sum(1 for e in coll if e.get("native_nc"))
        n_sticker = sum(1 for e in coll if e.get("sticker_path"))
        groups = set()
        for e in coll:
            g = (e.get("entry", {}).get("iconic_group", "")
                 or e.get("entry", {}).get("category", "") or "").lower()
            if "av" in g or "bird" in g:            groups.add("bird")
            elif "mamm" in g or "beast" in g:       groups.add("mammal")
            elif "insect" in g or "arachn" in g or "bug" in g: groups.add("insect")
            elif "plant" in g or "flora" in g or "fung" in g:  groups.add("plant")
        n_sweep   = len({"bird", "mammal", "insect", "plant"} & groups)
        n_days    = len({(e.get("timestamp", "") or "")[:10] for e in coll if e.get("timestamp")})
        n_regions = len({
            (e.get("scan_location", {}) or {}).get("region", "")
            for e in coll if (e.get("scan_location", {}) or {}).get("region")
        })
        n_highconf = sum(1 for e in coll if (e.get("confidence", 0) or 0) >= 50)
        n_correct = 0
        if CORRECTIONS_FILE.exists():
            try:
                n_correct = len(json.loads(CORRECTIONS_FILE.read_text()))
            except Exception:
                pass

        _counters = {
            "count": count, "category": n_cats, "rarity": n_rare,
            "endangered": n_endangered, "local": n_local, "sticker": n_sticker,
            "sweep": n_sweep, "streak": n_days, "region": n_regions,
            "highconf": n_highconf, "correct": n_correct,
        }

        def _progress(badge: dict) -> tuple[int, int]:
            """Return (current, max) for a badge's progress bar."""
            thr = badge["threshold"]
            return min(_counters.get(badge["type"], 0), thr), thr

        # ── Build panel ────────────────────────────────────────────────────────
        panel = QFrame(self)
        panel.setStyleSheet(f"background: {C_PANEL}; border-radius: 12px;")
        panel.setFixedSize(400, 520)
        panel.move((self.width() - 400) // 2, (self.height() - 520) // 2)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(0)

        # Header
        header_row = QHBoxLayout()
        n_unlocked = len(self._unlocked_achievements)
        title = QLabel(f"Achievements  {n_unlocked}/{len(ACHIEVEMENTS)}")
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

        for i, badge in enumerate(ACHIEVEMENTS):
            unlocked = badge["id"] in self._unlocked_achievements
            cur, mx  = _progress(badge)

            row = QWidget()
            row.setStyleSheet("background: transparent;")
            r_layout = QVBoxLayout(row)
            r_layout.setContentsMargins(0, 10, 0, 10)
            r_layout.setSpacing(5)

            # Top line: icon + name + checkmark or pct
            top = QHBoxLayout()
            top.setSpacing(12)

            icon_lbl = QLabel(badge["icon"] if unlocked else "🔒")
            icon_lbl.setFixedWidth(26)
            icon_lbl.setStyleSheet("font-size: 18px;")
            top.addWidget(icon_lbl)

            info = QVBoxLayout()
            info.setSpacing(1)
            name_lbl = QLabel(badge["name"])
            name_lbl.setStyleSheet(
                f"color: {C_TEXT}; font-size: 12px; font-weight: 700;"
                if unlocked else
                f"color: {C_SUBTEXT}; font-size: 12px; font-weight: 600;"
            )
            desc_lbl = QLabel(badge["desc"])
            desc_lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 10px;")
            info.addWidget(name_lbl)
            info.addWidget(desc_lbl)
            top.addLayout(info)
            top.addStretch()

            if unlocked:
                check = QLabel("✓")
                check.setStyleSheet(
                    f"color: {C_ACCENT}; font-size: 14px; font-weight: 900;")
                top.addWidget(check)
            else:
                pct_lbl = QLabel(f"{cur}/{mx}")
                pct_lbl.setStyleSheet(
                    f"color: {C_SUBTEXT}; font-size: 10px; font-weight: 600;")
                top.addWidget(pct_lbl)

            r_layout.addLayout(top)

            # Progress bar — always shown (full orange if unlocked)
            bar = QProgressBar()
            bar.setRange(0, mx)
            bar.setValue(cur if not unlocked else mx)
            bar.setFixedHeight(4)
            bar.setTextVisible(False)
            bar_color = C_ACCENT if unlocked else C_BORDER
            bar.setStyleSheet(f"""
                QProgressBar {{
                    background: {C_CARD};
                    border-radius: 2px;
                    border: none;
                    margin-left: 38px;
                }}
                QProgressBar::chunk {{
                    background: {bar_color};
                    border-radius: 2px;
                }}
            """)
            r_layout.addWidget(bar)
            c_layout.addWidget(row)

            if i < len(ACHIEVEMENTS) - 1:
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
            name  = entry.get(
                "common_name",
                self._current_result.get("name", "this organism"))
            intro = QLabel(f'Ask me anything about\n"{name}"')
            intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
            intro.setStyleSheet(
                f"color: {C_SUBTEXT}; font-size: 13px; padding: 30px 0;")
            self._chat_inner.addWidget(intro)
        else:
            intro = QLabel("Scan something first to\nstart a conversation.")
            intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
            intro.setStyleSheet(
                f"color: {C_SUBTEXT}; font-size: 13px; padding: 40px 0;")
            self._chat_inner.addWidget(intro)
        self._chat_inner.addStretch()

    def _on_chat_send(self):
        text = self._chat_input.text().strip()
        if not text or not self._current_result:
            if not self._current_result:
                self._add_chat_bubble(
                    "Scan something first, then ask me about it!", is_user=False)
            return
        if not self._client:
            self._add_chat_bubble(
                "AI model is still loading — try again in a moment.", is_user=False)
            return
        self._chat_input.clear()
        self._add_chat_bubble(text, is_user=True)

        entry = self._current_result.get("entry", {})
        system = f"""You are NatureDex AI, a knowledgeable wildlife educator.
The user has just scanned: {self._current_result.get('name', 'an organism')}.
Here is what you know about it:
{json.dumps(entry, indent=2)}

Answer questions in an engaging, educational tone — like a Pokédex that can converse.
Keep responses concise (2-4 sentences). Focus on the organism or object scanned.
If asked about North Carolina specifically, provide NC-relevant context."""

        messages = ([{"role": "system", "content": system}]
                    + self._chat_history
                    + [{"role": "user", "content": text}])
        thinking = self._add_chat_bubble("Thinking...", is_user=False)
        self._chat_worker = ChatWorker(self._client, messages)

        def on_reply(reply):
            try:
                thinking.setText(reply)
            except RuntimeError:
                pass
            self._chat_history.append({"role": "user",      "content": text})
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
        else:
            bubble.setStyleSheet(f"""
                background: {C_CARD};
                color: {C_TEXT};
                border: 1px solid {C_BORDER};
                border-radius: 14px;
                padding: 8px 14px;
                font-size: 13px;
            """)
        wrapper  = QWidget()
        w_layout = QHBoxLayout(wrapper)
        w_layout.setContentsMargins(0, 0, 0, 0)
        if is_user:
            w_layout.addStretch()
        w_layout.addWidget(bubble)
        if not is_user:
            w_layout.addStretch()
        self._chat_inner.insertWidget(self._chat_inner.count() - 1, wrapper)
        QTimer.singleShot(
            50,
            lambda: self._chat_scroll.verticalScrollBar().setValue(
                self._chat_scroll.verticalScrollBar().maximum()))
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

    # ── Correction System ──────────────────────────────────────────────────────

    def _on_share_card(self):
        if not self._current_result:
            return
        self._share_btn.setText("Creating…")
        self._share_btn.setEnabled(False)
        try:
            ts = (self._current_result.get("timestamp", "") or "")
            ts = ts.replace(":", "-").replace(".", "-")[:19] or "card"
            out_dir = Path.home() / ".naturedex_images"
            out_dir.mkdir(parents=True, exist_ok=True)
            out = str(out_dir / f"share_{ts}.png")
            path = build_share_card(self._current_result, out)
            if path:
                try:
                    import subprocess
                    subprocess.run(["open", path])   # opens the card in Preview
                except Exception:
                    print(f"[Share] saved to {path}")
                self._share_btn.setText("📤  Shared ✓")
            else:
                self._share_btn.setText("📤  Share")
        except Exception as e:
            print(f"[Share] {e}")
            self._share_btn.setText("📤  Share")
        finally:
            self._share_btn.setEnabled(True)
            QTimer.singleShot(2500, lambda: self._share_btn.setText("📤  Share"))

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
        QTimer.singleShot(
            3000, lambda: self._report_btn.setText("⚑  Wrong ID? Report it"))

    def _on_correction_cancel(self):
        self._correction_bar.hide()
        self._correction_input.clear()

    def _save_correction(self, correct_name: str):
        result = self._current_result or {}
        entry  = result.get("entry", {})
        record = {
            "timestamp":      datetime.datetime.now().isoformat(),
            "original_label": result.get("raw_label", ""),
            "original_name":  result.get("name", ""),
            "confidence":     result.get("confidence", 0),
            "correct_name":   correct_name,
            "scientific_name":entry.get("scientific_name", ""),
            "category":       entry.get("category", ""),
            "inat_taxon_id":  result.get("inat", {}).get("taxon_id"),
            "image_path":     result.get("image_path", ""),
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

    def closeEvent(self, event):
        _stop_ambient()
        if self._camera_thread:
            self._camera_thread.stop()
        super().closeEvent(event)


# ─── Entry Point ───────────────────────────────────────────────────────────────

def main():
    # Enable WebGL and hardware acceleration for CesiumJS globe rendering
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS",
                          "--enable-unsafe-webgpu --ignore-gpu-blocklist "
                          "--enable-gpu-rasterization --use-angle=metal")
    app = QApplication(sys.argv)
    app.setApplicationName("NatureDex AI")
    win = NatureDexWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()