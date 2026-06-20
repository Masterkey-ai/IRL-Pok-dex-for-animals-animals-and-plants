"""
NatureDex AI — Congressional App Challenge
An AI-powered wildlife identification and learning platform.
"""

import sys
import os
import json
import datetime
import threading
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QTextEdit, QLineEdit,
    QSplitter, QStackedWidget, QGraphicsOpacityEffect
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

# ─── Constants ────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_FILE = Path.home() / ".naturedex_collection.json"

# Pokédex color palette
C_BG         = "#0a0e1a"       # deep navy black
C_PANEL      = "#111827"       # dark panel
C_BORDER     = "#1e3a5f"       # blue border
C_ACCENT     = "#00d4ff"       # electric cyan
C_ACCENT2    = "#ff6b35"       # scan orange
C_TEXT       = "#e2e8f0"       # soft white
C_SUBTEXT    = "#64748b"       # muted slate
C_GREEN      = "#22c55e"       # confidence green
C_YELLOW     = "#eab308"       # medium confidence
C_RED        = "#ef4444"       # low confidence
C_CARD       = "#1a2744"       # card background

GROQ_MODEL   = "llama-3.3-70b-versatile"

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

    def __init__(self, frame, model, client):
        super().__init__()
        self.frame = frame
        self.model = model
        self.client = client

    def run(self):
        try:
            # Save frame
            tmp_path = "/tmp/naturedex_scan.jpg"
            cv2.imwrite(tmp_path, self.frame)

            # Classify
            img = keras_image.load_img(tmp_path, target_size=(224, 224))
            arr = keras_image.img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)
            preds = self.model.predict(arr, verbose=0)
            decoded = decode_predictions(preds, top=5)[0]

            top = decoded[0]
            label = top[1].replace("_", " ").title()
            confidence = top[2] * 100
            alternatives = [
                {"name": d[1].replace("_", " ").title(), "confidence": d[2] * 100}
                for d in decoded[1:4]
            ]

            # Generate structured entry
            entry = self._generate_entry(label, confidence)

            result = {
                "name": label,
                "raw_label": top[1],
                "confidence": confidence,
                "alternatives": alternatives,
                "entry": entry,
                "timestamp": datetime.datetime.now().isoformat(),
                "image_path": tmp_path,
            }
            self.result_ready.emit(result)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _generate_entry(self, label, confidence):
        prompt = f"""You are NatureDex AI, an educational wildlife identification system.
A user has scanned an object and it has been identified as: {label} (confidence: {confidence:.1f}%)

Generate a structured NatureDex entry. You MUST respond with ONLY valid JSON — no markdown, no code fences, no explanation.

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
  "north_carolina_context": "Is this found in North Carolina? When is it seen? Any NC-specific facts?",
  "fun_fact": "...",
  "description": "2-3 sentence Pokédex-style description"
}}

If this is a non-living object, adapt the fields creatively (habitat = where it's found, diet = what it runs on, etc.) in Pokédex style.
Return ONLY the JSON object. No other text."""

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
        )
        raw = response.choices[0].message.content.strip()

        # Strip possible markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            return json.loads(raw)
        except Exception:
            return {
                "common_name": label,
                "scientific_name": "Unknown",
                "category": "Unknown",
                "type_tags": [],
                "habitat": "Unknown",
                "diet": "Unknown",
                "behavior": "Unknown",
                "conservation_status": "N/A",
                "north_carolina_context": "Unknown",
                "fun_fact": "Analysis unavailable.",
                "description": raw[:300] if raw else "No description generated.",
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
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {C_ACCENT2}, stop:1 #ff8c42);
                color: white;
                border: none;
                border-radius: 26px;
                font-size: 15px;
                font-weight: 700;
                letter-spacing: 3px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff8043, stop:1 #ffa060);
            }}
            QPushButton:pressed {{
                background: #cc5520;
            }}
        """

    def _scanning_style(self):
        return f"""
            QPushButton {{
                background: {C_BORDER};
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

        # Dark vignette overlay
        painter.fillRect(0, 0, w, h, QColor(0, 0, 0, 60))

        # Scan line
        grad = QLinearGradient(0, self._y - 30, 0, self._y + 30)
        grad.setColorAt(0.0, QColor(0, 212, 255, 0))
        grad.setColorAt(0.5, QColor(0, 212, 255, 180))
        grad.setColorAt(1.0, QColor(0, 212, 255, 0))
        painter.fillRect(0, self._y - 30, w, 60, grad)

        # Corner brackets
        pen = QPen(QColor(C_ACCENT), 3)
        painter.setPen(pen)
        corner = 24
        gap = 20
        for x, y in [(gap, gap), (w - gap, gap), (gap, h - gap), (w - gap, h - gap)]:
            dx = corner if x == gap else -corner
            dy = corner if y == gap else -corner
            painter.drawLine(x, y, x + dx, y)
            painter.drawLine(x, y, x, y + dy)

        painter.end()


class CollectionCard(QFrame):
    clicked_signal = pyqtSignal(dict)

    def __init__(self, entry_data, parent=None):
        super().__init__(parent)
        self.entry_data = entry_data
        self.setFixedHeight(68)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QFrame {{
                background: {C_CARD};
                border: 1px solid {C_BORDER};
                border-radius: 8px;
            }}
            QFrame:hover {{
                border: 1px solid {C_ACCENT};
                background: #1e3050;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
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

    def mousePressEvent(self, event):
        self.clicked_signal.emit(self.entry_data)


# ─── Main Window ───────────────────────────────────────────────────────────────

class NatureDexWindow(QMainWindow):

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
        # Make sure the window's top-left is actually on screen too
        self.move(available.x() + 20, available.y() + 20)

        self._collection = self._load_collection()
        self._current_result = None
        self._chat_history = []
        self._camera_thread = None
        self._analysis_worker = None
        self._last_frame = None
        self._scan_overlay = None

        # Load AI models
        self._model = None
        self._client = None
        self._models_loaded = False

        self._setup_style()
        self._build_ui()
        self._start_camera()
        self._load_models_async()

    # ── Style ──────────────────────────────────────────────────────────────────

    def _setup_style(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background: {C_BG}; }}
            QWidget {{ background: {C_BG}; color: {C_TEXT}; font-family: 'SF Pro Display', 'Segoe UI', sans-serif; }}
            QScrollArea {{ border: none; background: transparent; }}
            QScrollBar:vertical {{
                background: {C_PANEL}; width: 6px; border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {C_BORDER}; border-radius: 3px; min-height: 30px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
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
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet(f"""
            QFrame {{
                background: {C_PANEL};
                border-right: 1px solid {C_BORDER};
            }}
        """)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setFixedHeight(64)
        header.setStyleSheet(f"background: {C_BG}; border-bottom: 1px solid {C_BORDER};")
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(16, 12, 16, 12)
        title = QLabel("NATUREDEX")
        title.setStyleSheet(f"""
            color: {C_ACCENT};
            font-size: 17px;
            font-weight: 800;
            letter-spacing: 3px;
        """)
        subtitle = QLabel("AI Wildlife Identifier")
        subtitle.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 10px; letter-spacing: 1px;")
        h_layout.addWidget(title)
        h_layout.addWidget(subtitle)
        layout.addWidget(header)

        # Stats bar
        stats_frame = QFrame()
        stats_frame.setFixedHeight(44)
        stats_frame.setStyleSheet(f"background: {C_CARD}; border-bottom: 1px solid {C_BORDER};")
        s_layout = QHBoxLayout(stats_frame)
        s_layout.setContentsMargins(16, 0, 16, 0)
        self._species_count_lbl = QLabel(f"◈  {len(self._collection)} discovered")
        self._species_count_lbl.setStyleSheet(f"color: {C_ACCENT}; font-size: 11px; font-weight: 600;")
        s_layout.addWidget(self._species_count_lbl)
        layout.addWidget(stats_frame)

        # Collection label
        col_label = QLabel("RECENT DISCOVERIES")
        col_label.setStyleSheet(f"""
            color: {C_SUBTEXT};
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 2px;
            padding: 12px 16px 6px 16px;
        """)
        layout.addWidget(col_label)

        # Scroll area for collection cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")

        self._collection_container = QWidget()
        self._collection_container.setStyleSheet("background: transparent;")
        self._collection_layout = QVBoxLayout(self._collection_container)
        self._collection_layout.setContentsMargins(10, 4, 10, 10)
        self._collection_layout.setSpacing(6)
        self._collection_layout.addStretch()

        scroll.setWidget(self._collection_container)
        layout.addWidget(scroll)

        # Populate with existing collection
        for entry in reversed(self._collection[-30:]):
            self._add_collection_card(entry, prepend=False)

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
        panel.setStyleSheet(f"background: #050810; border-right: 1px solid {C_BORDER};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Camera header
        cam_header = QFrame()
        cam_header.setFixedHeight(44)
        cam_header.setStyleSheet(f"background: {C_PANEL}; border-bottom: 1px solid {C_BORDER};")
        ch_layout = QHBoxLayout(cam_header)
        ch_layout.setContentsMargins(16, 0, 16, 0)
        cam_lbl = QLabel("◉  LIVE SCANNER")
        cam_lbl.setStyleSheet(f"color: {C_ACCENT2}; font-size: 11px; font-weight: 700; letter-spacing: 2px;")
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 11px;")
        ch_layout.addWidget(cam_lbl)
        ch_layout.addStretch()
        ch_layout.addWidget(self._status_lbl)
        layout.addWidget(cam_header)

        # Camera feed
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
        tab_bar.setStyleSheet(f"background: {C_PANEL}; border-bottom: 1px solid {C_BORDER};")
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

        # Placeholder
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
        layout.addWidget(scroll)
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
                self._model = MobileNetV2(weights="imagenet")
                self._client = OpenAI(
                    api_key=GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1",
                )
                self._models_loaded = True
                QTimer.singleShot(0, lambda: self._loading_lbl.setText("✓  Models ready"))
                QTimer.singleShot(2000, lambda: self._loading_lbl.setText(""))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._loading_lbl.setText(f"Error: {e}"))

        t = threading.Thread(target=_load, daemon=True)
        t.start()

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
        self._status_lbl.setText("Analyzing...")
        if self._scan_overlay:
            self._scan_overlay.setGeometry(self._camera_label.geometry())
            self._scan_overlay.show()
            self._scan_overlay.start()

        self._analysis_worker = AnalysisWorker(frame, self._model, self._client)
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
        self._species_count_lbl.setText(f"◈  {len(self._collection)} discovered")

        self._render_entry(result)
        self._switch_tab(0)
        self._reset_chat()

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
                    stop:0 {C_CARD}, stop:1 #0f1e3a);
                border: 1px solid {C_BORDER};
                border-radius: 10px;
                padding: 4px;
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
            color: {C_ACCENT};
            font-size: 20px;
            font-weight: 800;
            letter-spacing: 2px;
        """)
        nf_layout.addWidget(name_lbl)

        if sci and sci != "Unknown":
            sci_lbl = QLabel(sci)
            sci_lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 12px; font-style: italic;")
            nf_layout.addWidget(sci_lbl)

        # Confidence bar row
        conf = result["confidence"]
        conf_color = C_GREEN if conf >= 75 else C_YELLOW if conf >= 50 else C_RED
        conf_row = QHBoxLayout()
        conf_row.setSpacing(10)

        conf_lbl = QLabel(f"CONFIDENCE  {conf:.1f}%")
        conf_lbl.setStyleSheet(f"color: {conf_color}; font-size: 11px; font-weight: 700; letter-spacing: 1px;")
        conf_row.addWidget(conf_lbl)

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
        conf_row.addStretch()
        nf_layout.addLayout(conf_row)

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
        fields = [
            ("🌿  HABITAT",      entry.get("habitat", "")),
            ("🍃  DIET",         entry.get("diet", "")),
            ("🐾  BEHAVIOR",     entry.get("behavior", "")),
            ("🔴  CONSERVATION", entry.get("conservation_status", "")),
            ("📍  NORTH CAROLINA", entry.get("north_carolina_context", "")),
            ("⚡  FUN FACT",     entry.get("fun_fact", "")),
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
                border: 1px solid {C_BORDER};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {C_SUBTEXT}; font-size: 10px; font-weight: 700; letter-spacing: 1.5px;")
        val = QLabel(value)
        val.setWordWrap(True)
        val.setStyleSheet(f"color: {C_TEXT}; font-size: 12px; line-height: 1.5;")

        layout.addWidget(lbl)
        layout.addWidget(val)
        return card

    # ── Collection ─────────────────────────────────────────────────────────────

    def _add_collection_card(self, entry_data, prepend=True):
        card = CollectionCard(entry_data)
        card.clicked_signal.connect(self._on_collection_click)
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

        worker = ChatWorker(self._client, messages)

        def on_reply(reply):
            thinking.setText(reply)
            self._chat_history.append({"role": "user", "content": text})
            self._chat_history.append({"role": "assistant", "content": reply})

        worker.reply_ready.connect(on_reply)
        worker.start()

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

    # ── Cleanup ────────────────────────────────────────────────────────────────

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