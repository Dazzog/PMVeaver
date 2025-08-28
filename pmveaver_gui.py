# pmveaver_gui.py
import os, sys, re, time, shutil
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
import qdarktheme

__version__ = "1.0.0"

APP_TITLE = "PMVeaver"

# Progress-Gewichte
STEP_WEIGHTS = {
    "collecting clips": (0.0, 0.1),
    "building video": (0.1, 0.125),
    "writing audio": (0.125, 0.15),
    "writing video": (0.15, 1.00),
}
STEP_ORDER = list(STEP_WEIGHTS.keys())

# ---------------- Codec-Presets / Profile (aus deinem Tkinter-Original übernommen/vereinfacht) ------------
CODEC_PRESETS = {
    "libx264":     ["placebo","veryslow","slower","slow","medium","fast","faster","veryfast","superfast","ultrafast"],
    "libx265":     ["placebo","veryslow","slower","slow","medium","fast","faster","veryfast","superfast","ultrafast"],
    "h264_nvenc":  ["slow","medium","fast","hp","hq","ll","llhq","llhp","lossless","losslesshp"],
    "hevc_nvenc":  ["slow","medium","fast","hp","hq","ll","llhq","llhp","lossless","losslesshp"],
    "h264_qsv":    ["veryslow","slower","slow","medium","fast","faster","veryfast"],
    "h264_amf":    ["balanced","speed","quality"],
    "prores_ks":   [],
    "libvpx-vp9":  [],
}
DEFAULT_PRESET_BY_CODEC = {
    "libx264": "medium", "libx265": "medium",
    "h264_nvenc": "hq",  "hevc_nvenc": "hq",
    "h264_qsv": "medium",
    "h264_amf": "balanced",
    "prores_ks": "", "libvpx-vp9": ""
}
RENDER_PROFILES = {
    "CPU (x264)":      {"codec": "libx264",   "preset": "medium",  "threads": "8", "bitrate": "8M"},
    "NVIDIA GPU":      {"codec": "h264_nvenc","preset": "hq",      "threads": "",  "bitrate": "8M"},
    "AMD GPU":         {"codec": "h264_amf",  "preset": "quality", "threads": "",  "bitrate": "8M"},
    "Intel GPU (QSV)": {"codec": "h264_qsv",  "preset": "medium",  "threads": "",  "bitrate": "8M"},
}

ERROR_CSS = "QLineEdit{border:1px solid #d9534f; border-radius:4px;}"  # rot
OK_CSS    = ""  # Default-Style

def _norm(p: str) -> str:
    return os.path.normpath(os.path.expandvars(os.path.expanduser(p or "")))

def _base_dir() -> Path:
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent

def _find_cli_candidate():
    """Bevorzuge pmveaver.exe neben der GUI, sonst pmveaver.py (im selben Ordner)."""
    base = _base_dir()
    cand_exe = base / "pmveaver.exe"
    if cand_exe.exists():
        return str(cand_exe), True
    cand_py = base / "pmveaver.py"
    if cand_py.exists():
        return str(cand_py), False
    # Fallback: im Arbeitsverzeichnis suchen
    if Path("pmveaver.exe").exists():
        return "pmveaver.exe", True
    if Path("pmveaver.py").exists():
        return "pmveaver.py", False
    return None, None

def hms(seconds: float) -> str:
    if seconds is None or seconds < 0:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600); m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

# --- In pmveaver_gui.py, z.B. oberhalb von PMVeaverQt hinzufügen ---
class TriangularDistWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._min_beats = 2
        self._max_beats = 8
        self._mode_pos = 0.25  # 0..1
        self.setToolTip("Triangular weighting over even beats")

    def setParams(self, min_beats: int, max_beats: int, mode_pos: float):
        changed = (self._min_beats != min_beats) or (self._max_beats != max_beats) or (abs(self._mode_pos - mode_pos) > 1e-6)
        self._min_beats = min_beats
        self._max_beats = max_beats
        self._mode_pos = max(0.0, min(1.0, mode_pos))
        if changed:
            self.update()

    @staticmethod
    def _even_list(lo:int, hi:int):
        lo, hi = sorted((int(lo), int(hi)))
        evens = [b for b in range(lo, hi+1) if b % 2 == 0 and b > 0]
        if not evens:
            evens = [2]
        return lo, hi, evens

    def _weights(self):
        # Mirror the core logic from pmveaver.choose_even_beats
        lo, hi, evens = self._even_list(self._min_beats, self._max_beats)
        mode = lo + self._mode_pos * (hi - lo)

        eps = 1e-9
        span = max(hi - lo, eps)
        MIN_WEIGHT = 0.1

        ws = []
        if mode <= lo + eps:
            for b in evens:
                w = 1.0 - (b - lo) / span
                ws.append(max(w, MIN_WEIGHT))
        elif mode >= hi - eps:
            for b in evens:
                w = 1.0 - (hi - b) / span
                ws.append(max(w, MIN_WEIGHT))
        else:
            left = max(mode - lo, eps)
            right = max(hi - mode, eps)
            denom = max(left, right)
            for b in evens:
                w = 1.0 - abs(b - mode) / denom
                ws.append(max(w, MIN_WEIGHT))

        # Normieren für die Darstellung (Y=0..1)
        mx = max(ws) if ws else 1.0
        ws = [w / mx for w in ws]
        return evens, ws

    def paintEvent(self, e: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect().adjusted(6, 6, -6, -6)

        # Hintergrund/ Rahmen
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(self.palette().base())
        p.drawRect(rect)

        # Achsen
        p.setPen(self.palette().mid().color())
        p.drawRect(rect)

        evens, ws = self._weights()
        if not evens:
            return

        # Balkenbreite
        n = len(evens)
        gap = 4
        bar_w = max(6, (rect.width() - gap*(n+1)) // max(1, n))

        # Textfarbe
        txt_pen = QtGui.QPen(self.palette().text().color())

        # Bars
        x = rect.left() + gap
        max_h = rect.height()  # Platz für Labels
        bar_brush = QtGui.QBrush(self.palette().highlight().color())

        for i, (b, w) in enumerate(zip(evens, ws)):
            h = int(max_h * w)
            bar_rect = QtCore.QRect(x, rect.bottom() - h - 24, bar_w, h)
            p.fillRect(bar_rect, bar_brush)

            # Beat-Label
            p.setPen(txt_pen)
            lbl = str(b) + ' beats'
            metrics = p.fontMetrics()
            tw = metrics.horizontalAdvance(lbl)
            p.drawText(x + (bar_w - tw)//2, rect.bottom() - 8, lbl)

            x += bar_w + gap

        # Mode-Marker (vertikale Linie)
        lo, hi, _ = self._even_list(self._min_beats, self._max_beats)
        if hi > lo:
            rel = (self._mode_pos)  # 0..1
            x_mode = rect.left() + int(rect.width() * rel)
            pen = QtGui.QPen(self.palette().highlight().color())
            pen.setStyle(QtCore.Qt.DashLine)
            p.setPen(pen)
            p.drawLine(x_mode, rect.top(), x_mode, rect.bottom())


class PMVeaverQt(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(APP_TITLE)

        # --- System-Dark/Light automatisch
        qdarktheme.setup_theme(theme="auto", custom_colors={"primary": "#8571c9"})

        # ---------- State ----------
        self.proc: QtCore.QProcess | None = None
        self.running = False
        self._aborted = False
        self._current_line = ""
        self._last_step_pct = None
        self._start_time = None
        self._phase = "—"
        self._overall_progress = 0.0
        self._last_preview_check = 0.0
        self._run_output = None
        self._suspend_validation = True


        # ---------- UI ----------
        self._build_ui()
        self._apply_profile()        # initiale Profileinstellungen
        self._sync_preset_choices()  # Preset-Auswahl zum Codec
        self._sync_bpm_ui()

        # Ticker für ETA/Preview
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(200)

    # ===================== UI =====================
    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        main_split = QtWidgets.QHBoxLayout()
        root.addLayout(main_split, stretch=1)
        main_split.addSpacing(16)

        left_box = QtWidgets.QVBoxLayout()

        # ----- Sources / Output
        src_group = QtWidgets.QGroupBox("Sources / Output")
        src_form = QtWidgets.QGridLayout(src_group)

        self.ed_audio  = QtWidgets.QLineEdit()
        self.ed_audio.editingFinished.connect(self._autofill_output_from_audio)

        self.ed_output = QtWidgets.QLineEdit()

        src_form.addWidget(QtWidgets.QLabel("Source folders:"), 1, 0)
        self.videos_container = QtWidgets.QWidget()
        self.videos_layout = QtWidgets.QVBoxLayout(self.videos_container)
        self.videos_layout.setContentsMargins(0, 0, 0, 0)
        self.videos_layout.setSpacing(6)

        self.video_rows: list[dict] = []
        self._add_video_row()

        btn_audio  = QtWidgets.QPushButton("Browse…")
        btn_audio.clicked.connect(self._browse_audio)
        btn_output = QtWidgets.QPushButton("Browse…")
        btn_output.clicked.connect(self._browse_output)

        src_form.addWidget(QtWidgets.QLabel("Audio:"),        0, 0)
        src_form.addWidget(self.ed_audio,                     0, 1)
        src_form.addWidget(btn_audio,                         0, 2)

        src_form.addWidget(self.videos_container,             1, 1, 1, 2)

        self.chk_trim = QtWidgets.QCheckBox("Trim long clips")
        src_form.addWidget(self.chk_trim, 2, 1)


        src_form.addWidget(QtWidgets.QLabel("Output file:"),  3, 0)
        src_form.addWidget(self.ed_output,                    3, 1)
        src_form.addWidget(btn_output,                        3, 2)
        left_box.addWidget(src_group)

        left_box.addStretch(1)

        # ----- Progress
        prog_group = QtWidgets.QGroupBox("Progress")
        pg = QtWidgets.QHBoxLayout(prog_group)
        pg.setSpacing(16)

        # Preview
        self.lbl_preview = QtWidgets.QLabel("No preview")
        self.lbl_preview.setFixedSize(180, 100)
        self.lbl_preview.setAlignment(QtCore.Qt.AlignCenter)

        self._preview_pix = None
        self.lbl_preview.installEventFilter(self)

        pg.addWidget(self.lbl_preview)

        # Step/Elapsed/ETA + Progressbars
        right_box = QtWidgets.QVBoxLayout()

        # Current step + Elapsed/ETA des aktuellen Steps
        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("▼ Current step:"))
        self.lbl_step = QtWidgets.QLabel("—")
        row1.addWidget(self.lbl_step)
        row1.addStretch()

        self.lbl_elapsed_step = QtWidgets.QLabel("Elapsed (Current step): —")
        self.lbl_eta_step = QtWidgets.QLabel("ETA (Current step): —")
        row1.addWidget(self.lbl_elapsed_step)
        row1.addSpacing(10)
        row1.addWidget(self.lbl_eta_step)
        right_box.addLayout(row1)

        # Zeile 2: Step-Progress
        row2 = QtWidgets.QHBoxLayout()
        self.pb_step = QtWidgets.QProgressBar()
        self.pb_step.setRange(0, 100)
        self.pb_step.setValue(0)
        row2.addWidget(self.pb_step, stretch=1)
        right_box.addLayout(row2)

        # Zeile 3: Total-Progress
        row3 = QtWidgets.QHBoxLayout()
        self.pb_total = QtWidgets.QProgressBar()
        self.pb_total.setRange(0, 100)
        self.pb_total.setValue(0)
        row3.addWidget(self.pb_total, stretch=1)
        right_box.addLayout(row3)

        # Total
        row_total = QtWidgets.QHBoxLayout()
        row_total.addWidget(QtWidgets.QLabel("▲ Total"))
        row_total.addStretch()
        self.lbl_elapsed_total = QtWidgets.QLabel("Elapsed (Total): —")
        row_total.addWidget(self.lbl_elapsed_total)
        right_box.addLayout(row_total)

        pg.addLayout(right_box)

        left_box.addWidget(prog_group)

        # ----- Buttons + FFmpeg-Status
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 8, 0, 0)

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.clicked.connect(self.start)
        self.btn_start.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_start.setMinimumHeight(44)  # << größer
        self.btn_start.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                     QtWidgets.QSizePolicy.Fixed)

        # größere, fette Schrift
        f = self.btn_start.font()
        f.setBold(True)
        f.setPointSize(int(f.pointSize() * 1.15))
        self.btn_start.setFont(f)
        # Icon (systemweit) – optional, aber wertig
        self.btn_start.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.btn_start.setIconSize(QtCore.QSize(24, 24))
        # Primär-Style (farblich zum Theme passend)
        self.btn_start.setStyleSheet("""
        QPushButton#btnStart {
            border-radius: 8px;
            padding: 8px 18px;
        }
        """)

        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaStop))
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)

        btn_row.addWidget(self.btn_start); btn_row.addWidget(self.btn_stop); btn_row.addStretch(1)

        self.lbl_ffmpeg = QtWidgets.QLabel("FFmpeg: …")
        btn_row.addWidget(self.lbl_ffmpeg)
        left_box.addLayout(btn_row)

        main_split.addLayout(left_box, stretch=3)
        main_split.addSpacing(16)

        # ----- Accordion (QToolBox) für alle restlichen Frames
        acc = QtWidgets.QToolBox()
        acc.addItem(self._panel_frame_render(), "Frame / Render")
        acc.addItem(self._panel_audio_mix(),    "Audio Mix")
        acc.addItem(self._panel_bpm(),          "BPM / Beat lengths")
        acc.addItem(self._panel_time_fallback(),"Time-based fallback (when BPM disabled)")
        acc.addItem(self._panel_codecs(),       "Codecs / Performance")

        acc_container = QtWidgets.QWidget()
        acc_layout = QtWidgets.QVBoxLayout(acc_container)
        acc_layout.setContentsMargins(8, 8, 8, 8)
        acc_layout.addWidget(acc)

        main_split.addWidget(acc_container, stretch=2)

        # Auto-Größe
        self.resize(1360, 640)
        self.setMinimumWidth(1360)

        self._check_ffmpeg()

        self._autofill_video_folder()

        self._suspend_validation = False

        for le in (self.ed_audio, self.ed_output):
            le.textChanged.connect(lambda _=None: self._validate_inputs(False))

    # ----------------- Panels -----------------
    def _panel_frame_render(self):
        w = QtWidgets.QWidget()
        w.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        g = QtWidgets.QGridLayout(w)
        g.setSpacing(16)

        self.sb_w = QtWidgets.QSpinBox()
        self.sb_w.setRange(16, 16384)
        self.sb_w.setValue(1920)
        self.sb_w.setSuffix(" px")
        self.sb_h = QtWidgets.QSpinBox()
        self.sb_h.setRange(16, 16384)
        self.sb_h.setValue(1080)
        self.sb_h.setSuffix(" px")
        self.sb_fps = QtWidgets.QDoubleSpinBox()
        self.sb_fps.setDecimals(2)
        self.sb_fps.setRange(1.0, 240.0)
        self.sb_fps.setValue(30.0)
        self.sb_fps.setSuffix(" fps")

        # Alle drei gleichmäßig dehnbar machen
        for sb in (self.sb_w, self.sb_h, self.sb_fps):
            sb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            sb.setMinimumWidth(80)

        lab_w = QtWidgets.QLabel("Width");
        lab_w.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lab_h = QtWidgets.QLabel("Height");
        lab_h.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        g.addWidget(lab_w, 0, 0);
        g.addWidget(self.sb_w, 0, 1)
        g.addWidget(lab_h, 0, 2);
        g.addWidget(self.sb_h, 0, 3)

        self.ds_triptych_carry = QtWidgets.QSpinBox()
        self.ds_triptych_carry.setRange(0, 100)
        self.ds_triptych_carry.setSingleStep(5)
        self.ds_triptych_carry.setValue(30)
        self.ds_triptych_carry.setSuffix(" %")
        g.addWidget(QtWidgets.QLabel("Triptych carry"), 1, 0)
        g.addWidget(self.ds_triptych_carry, 1, 1)

        lab_f = QtWidgets.QLabel("FPS");
        lab_f.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        g.addWidget(lab_f, 1, 2);
        g.addWidget(self.sb_fps, 1, 3)

        self.chk_pulse = QtWidgets.QCheckBox("Beat pulse effect")
        g.addWidget(self.chk_pulse, 2, 0, 1, 2)

        lab_fadeout = QtWidgets.QLabel("Fade out")
        lab_fadeout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.ds_fadeout = QtWidgets.QDoubleSpinBox()
        self.ds_fadeout.setRange(0, 5.0)
        self.ds_fadeout.setDecimals(2)
        self.ds_fadeout.setSingleStep(0.10)
        self.ds_fadeout.setSuffix(" s")
        g.addWidget(lab_fadeout, 2, 2)
        g.addWidget(self.ds_fadeout, 2, 3)

        # alle drei Spinbox-Spalten gleich stretchen
        for col in (1, 3):
            g.setColumnStretch(col, 1)
        # Labels schmal halten
        for col in (0, 2):
            g.setColumnMinimumWidth(col, 80)

        return w

    def _panel_audio_mix(self):
        w = QtWidgets.QWidget()
        w.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        g = QtWidgets.QGridLayout(w)
        g.setSpacing(16)

        # Slider (intern 0..200 bzw. 0..100)
        def slider(init, to=200):
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(0, to)
            s.setValue(init)
            return s

        self.sl_bg = slider(100)  # 1.00x
        self.sl_clip = slider(80)  # 0.80x
        self.sl_rev = slider(20, to=100)  # 0.20

        # Editierbare Zahlenfelder (anzeigen + tippen erlaubt)
        def dspin(minv, maxv, init, step=1, suffix="%"):
            ds = QtWidgets.QDoubleSpinBox()
            ds.setRange(minv, maxv)
            ds.setDecimals(0)
            ds.setSingleStep(step)
            ds.setValue(init)
            ds.setSuffix(suffix)
            ds.setFixedWidth(72)
            return ds

        self.ds_bg = dspin(0, 200, 100)
        self.ds_clip = dspin(0, 200, 80)
        self.ds_rev = dspin(0, 100, 20)

        # Bidirektionale Verdrahtung (Slider <-> SpinBox)
        self.sl_bg.valueChanged.connect(lambda v: self.ds_bg.setValue(v))
        self.ds_bg.valueChanged.connect(lambda val: self.sl_bg.setValue(int(round(val))))

        self.sl_clip.valueChanged.connect(lambda v: self.ds_clip.setValue(v))
        self.ds_clip.valueChanged.connect(lambda val: self.sl_clip.setValue(int(round(val))))

        self.sl_rev.valueChanged.connect(lambda v: self.ds_rev.setValue(v))
        self.ds_rev.valueChanged.connect(lambda val: self.sl_rev.setValue(int(round(val))))

        # Layout: [Label | Slider | Zahl] x 3 – Slider-Spalten gleich stretchen
        g.addWidget(QtWidgets.QLabel("Audio Volume"), 0, 0)
        g.addWidget(self.sl_bg, 0, 1)
        g.addWidget(self.ds_bg, 0, 2)

        g.addWidget(QtWidgets.QLabel("Clip Volume"), 1, 0)
        g.addWidget(self.sl_clip, 1, 1)
        g.addWidget(self.ds_clip, 1, 2)

        g.addWidget(QtWidgets.QLabel("Clip Reverb"), 2, 0)
        g.addWidget(self.sl_rev, 2, 1)
        g.addWidget(self.ds_rev, 2, 2)

        g.setColumnMinimumWidth(0, 90)

        return w

    def _panel_bpm(self):
        w = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(w)
        g.setSpacing(16)

        self.chk_bpm = QtWidgets.QCheckBox("Automatically detect BPM (librosa)")
        self.chk_bpm.setChecked(True)
        self.chk_bpm.toggled.connect(self._sync_bpm_ui)
        g.addWidget(self.chk_bpm, 0, 0, 1, 2)

        self.ed_bpm = QtWidgets.QLineEdit()
        self.ed_bpm.setPlaceholderText("BPM (manual)")
        g.addWidget(QtWidgets.QLabel("BPM (manual)"), 1, 0)
        g.addWidget(self.ed_bpm, 1, 1)

        self.sb_min_beats = QtWidgets.QSpinBox()
        self.sb_min_beats.setRange(1, 64)
        self.sb_min_beats.setSingleStep(1)
        self.sb_min_beats.setValue(2)

        self.sb_max_beats = QtWidgets.QSpinBox()
        self.sb_max_beats.setRange(2, 64)
        self.sb_max_beats.setSingleStep(2)
        self.sb_max_beats.setValue(8)

        g.addWidget(QtWidgets.QLabel("Min beats"), 0, 2)
        g.addWidget(self.sb_min_beats, 0, 3)
        g.addWidget(QtWidgets.QLabel("Max beats"), 1, 2)
        g.addWidget(self.sb_max_beats, 1, 3)

        # --- Clip length probabilities panel ---
        frame = QtWidgets.QGroupBox("Clip length probabilities")
        frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        vbox = QtWidgets.QVBoxLayout(frame)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        self.dist_widget = TriangularDistWidget()
        self.dist_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        vbox.addWidget(self.dist_widget, stretch=1)

        self.sl_beat_mode = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_beat_mode.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.sl_beat_mode.setRange(0, 100)
        self.sl_beat_mode.setValue(25)
        vbox.addWidget(self.sl_beat_mode)

        g.addWidget(frame, 2, 0, 1, 4)

        # Hidden storage for beat mode value (so core logic still works)
        self.ds_beat_mode = QtWidgets.QDoubleSpinBox()
        self.ds_beat_mode.setRange(0.0, 1.0)
        self.ds_beat_mode.setDecimals(3)
        self.ds_beat_mode.setSingleStep(0.01)
        self.ds_beat_mode.setValue(0.25)
        self.ds_beat_mode.hide()

        # Wire slider <-> hidden spinbox
        def _sync_from_slider(v):
            f = v / 100.0
            self.ds_beat_mode.blockSignals(True)
            self.ds_beat_mode.setValue(f)
            self.ds_beat_mode.blockSignals(False)
            self._update_dist_plot()

        self.sl_beat_mode.valueChanged.connect(_sync_from_slider)

        # Änderungen an Min/Max aktualisieren die Grafik
        self.sb_min_beats.valueChanged.connect(lambda _: self._update_dist_plot())
        self.sb_max_beats.valueChanged.connect(lambda _: self._update_dist_plot())

        QtCore.QTimer.singleShot(0, self._update_dist_plot)

        # alle drei Spinbox-Spalten gleich stretchen
        for col in (1, 3):
            g.setColumnStretch(col, 1)
        # Labels schmal halten
        for col in (0, 2):
            g.setColumnMinimumWidth(col, 80)

        return w

    def _update_dist_plot(self):
        self.dist_widget.setParams(self.sb_min_beats.value(), self.sb_max_beats.value(), self.ds_beat_mode.value())

    def _panel_time_fallback(self):
        w = QtWidgets.QWidget();
        w.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        g = QtWidgets.QGridLayout(w)
        g.setSpacing(16)

        self.ds_min_seconds = QtWidgets.QDoubleSpinBox()
        self.ds_min_seconds.setRange(0.10, 60.0)
        self.ds_min_seconds.setDecimals(2)
        self.ds_min_seconds.setSingleStep(0.10)
        self.ds_min_seconds.setValue(2.00)
        self.ds_min_seconds.setSuffix(" s")

        self.ds_max_seconds = QtWidgets.QDoubleSpinBox()
        self.ds_max_seconds.setRange(0.10, 90.0)
        self.ds_max_seconds.setDecimals(2)
        self.ds_max_seconds.setSingleStep(0.10)
        self.ds_max_seconds.setValue(5.00)
        self.ds_max_seconds.setSuffix(" s")

        # gleichmäßig dehnbar
        for sb in (self.ds_min_seconds, self.ds_max_seconds):
            sb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            sb.setMinimumWidth(80)

        lab_min = QtWidgets.QLabel("Min seconds")
        lab_max = QtWidgets.QLabel("Max seconds")
        lab_min.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lab_max.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        g.addWidget(lab_min, 0, 0)
        g.addWidget(self.ds_min_seconds, 0, 1)
        g.addWidget(lab_max, 0, 2)
        g.addWidget(self.ds_max_seconds, 0, 3)

        # beide Spinbox-Spalten gleich stretchen
        for col in (1, 3):
            g.setColumnStretch(col, 1)
        for col in (0, 2):  # Labels schmal halten
            g.setColumnMinimumWidth(col, 90)

        # Konsistenz: min ≤ max
        self.ds_min_seconds.valueChanged.connect(
            lambda v: self.ds_max_seconds.setValue(max(self.ds_max_seconds.value(), v))
        )
        self.ds_max_seconds.valueChanged.connect(
            lambda v: self.ds_min_seconds.setValue(min(self.ds_min_seconds.value(), v))
        )

        return w

    def _panel_codecs(self):
        w = QtWidgets.QWidget()
        w.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        g = QtWidgets.QGridLayout(w)
        g.setSpacing(16)

        self.cb_profile = QtWidgets.QComboBox(); self.cb_profile.addItems(list(RENDER_PROFILES.keys()))
        self.cb_codec   = QtWidgets.QComboBox(); self.cb_codec.addItems(list(CODEC_PRESETS.keys()))
        self.cb_audio   = QtWidgets.QComboBox(); self.cb_audio.addItems(["aac","libopus","libmp3lame"])
        self.cb_preset  = QtWidgets.QComboBox()

        self.ed_bitrate = QtWidgets.QLineEdit(); self.ed_bitrate.setPlaceholderText("Bitrate (e.g., 8M)")
        self.ed_threads = QtWidgets.QLineEdit(); self.ed_threads.setPlaceholderText("Threads")
        self.chk_preview = QtWidgets.QCheckBox("Generate Preview"); self.chk_preview.setChecked(True)

        self.cb_profile.currentIndexChanged.connect(self._apply_profile)
        self.cb_codec.currentIndexChanged.connect(self._sync_preset_choices)

        g.addWidget(QtWidgets.QLabel("Hardware profile"), 0,0); g.addWidget(self.cb_profile, 0,1)

        g.addWidget(QtWidgets.QLabel("Video codec"),      1,0); g.addWidget(self.cb_codec,   1,1)
        g.addWidget(QtWidgets.QLabel("Audio codec"),      1,2); g.addWidget(self.cb_audio,   1,3)

        g.addWidget(QtWidgets.QLabel("Codec Preset"),     2,0); g.addWidget(self.cb_preset,  2,1)

        g.addWidget(QtWidgets.QLabel("Bitrate"),          3,0); g.addWidget(self.ed_bitrate, 3,1)
        g.addWidget(QtWidgets.QLabel("Threads"),          3,2); g.addWidget(self.ed_threads, 3,3)

        g.addWidget(self.chk_preview,                     4,0,1,2)
        g.setColumnStretch(5,1)
        return w

    # ===================== Start/Stop & Prozess =====================
    def start(self):
        if self.running: return

        if not self._validate_inputs(show_message=True): return

        args = self._build_args()
        if not args:
            QtWidgets.QMessageBox.warning(self, "Missing input", "Bitte Audio/Video/Output prüfen.")
            return

        cli, is_exe = _find_cli_candidate()
        if cli is None:
            QtWidgets.QMessageBox.critical(self, "Not found",
                                           "pmveaver.exe / pmveaver.py wurde nicht gefunden.\nLege es neben diese GUI.")
            return

        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        # Environment
        env = QtCore.QProcessEnvironment.systemEnvironment()
        if not env.contains("PYTHONUNBUFFERED"): env.insert("PYTHONUNBUFFERED", "1")
        if not env.contains("TQDM_MININTERVAL"): env.insert("TQDM_MININTERVAL", "0.1")
        self.proc.setProcessEnvironment(env)

        # Signale
        self.proc.readyReadStandardOutput.connect(self._on_ready_read)
        self.proc.finished.connect(self._on_finished)

        if is_exe:
            program = cli
            full_args = args
        else:
            program = sys.executable
            full_args = [cli] + args

        self.proc.start(program, full_args)
        if not self.proc.waitForStarted(5000):
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to start process.")
            self.proc = None
            return

        print("GUI> Starting:", " ".join(self._quote(a) for a in ([program] + full_args)), flush=True)

        # UI-Status
        self._reset_progress()
        self.running = True
        self._aborted = False
        self._start_time = time.time()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_phase("Started")
        self._run_output = _norm(self.ed_output.text().strip())

    def stop(self):
        if not (self.proc and self.running):
            return
        self._aborted = True
        self._set_phase("Aborting")

        print("GUI> Abort requested…", flush=True)

        # Phase A: höflich bitten (Token über STDIN)
        try:
            self.proc.write(b"__PMVEAVER_EXIT__\n")
            self.proc.flush()
        except Exception:
            pass

        # In 3s prüfen – wenn noch läuft: terminate()
        QtCore.QTimer.singleShot(3000, self._try_terminate_then_kill)

    def _try_terminate_then_kill(self):
        if not self.proc:
            return
        if self.proc.state() == QtCore.QProcess.NotRunning:
            return  # already done → finished-Signal räumt UI auf
        # Phase B: terminate (liefert Signal / WM_CLOSE)
        self.proc.terminate()

        # In weiteren 3s prüfen – wenn immer noch läuft: kill()
        QtCore.QTimer.singleShot(3000, self._force_kill)

    def _force_kill(self):
        if not self.proc:
            return
        if self.proc.state() != QtCore.QProcess.NotRunning:
            print("GUI> Forcing kill…", flush=True)
            self.proc.kill()

    def _on_ready_read(self):
        if not self.proc:
            return
        data = bytes(self.proc.readAllStandardOutput())
        if not data:
            return
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            text = data.decode(errors="ignore")
        for ch in text:
            if ch in ("\n", "\r"):
                if self._current_line:
                    self._handle_cli_line(self._current_line)
                    self._current_line = ""
            else:
                self._current_line += ch

        # pass-through log
        try:
            sys.stdout.write(text); sys.stdout.flush()
        except Exception:
            pass

    def _on_finished(self, exit_code: int, status: QtCore.QProcess.ExitStatus):
        if self._aborted:
            self._set_phase("Aborted")
        else:
            if status == QtCore.QProcess.NormalExit and exit_code == 0:
                self._set_phase("Finished")
                self._update_progress(pct=100)  # jetzt ist 100% korrekt
                if self._run_output and Path(self._run_output).exists():
                    self.lbl_preview.setPixmap(QtGui.QPixmap())
                    self.lbl_preview.setText("✅ PMV ready – click to open")
                    self.lbl_preview.setCursor(QtCore.Qt.PointingHandCursor)
            else:
                self._set_phase(f"Failed (exit {exit_code})")
                self.lbl_preview.setCursor(QtCore.Qt.ArrowCursor)

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.running = False
        self.proc = None

    # ===================== Parser / Fortschritt =====================
    def _handle_cli_line(self, line: str):
        s = line.strip()
        if not s:
            return

        if "Finished" in line or "All done" in line:
            if not self._aborted:
                self._set_phase("Finished")
                self._update_progress(pct=100)
            return

        if self._aborted:
            return

        phase_title = self._phase_from_line(s)
        if phase_title:
            self._set_phase(phase_title)

        if self._is_tqdm_line(s):
            if self._phase not in STEP_WEIGHTS:
                self._set_phase("Writing video")
            self._parse_tqdm_progress(s)

    def _phase_from_line(self, s: str) -> str | None:
        sl = s.lower()

        # 1) Direkte CLI-Logs
        if sl.startswith("collecting clips") or "collecting clips:" in sl:
            return "Collecting clips"

        # 2) MoviePy-/Moviepy-Logs
        #    (verschiedene Schreibweisen je nach Version)
        if "moviepy" in sl or "moviepy" in s:  # tolerant
            if "building video" in sl:
                return "Building video"
            if "writing audio" in sl:
                return "Writing audio"
            if "writing video" in sl:
                return "Writing video"

        # 3) ffmpeg-typische oder neutrale Zeilen, die auf den Render-Schritt deuten
        if "writing video" in sl:
            return "Writing video"

        return None

    def _is_tqdm_line(self, s: str) -> bool:
        # sehr tolerant: Prozent + Balken + Klammerblock
        return ("%|" in s) and ("[" in s and "]" in s)

    def _parse_tqdm_progress(self, text: str):
        # Prozent: " 14%|"
        m = re.search(r"(\d+)%\|", text)
        if m:
            pct = int(m.group(1))
            # clamp 0..100, falls tqdm mal "101%|" spuckt
            pct = max(0, min(100, pct))
            self._update_progress(pct=pct)

        # Zeiten: [elapsed<eta, ...] (mm:ss oder hh:mm:ss)
        m2 = re.search(
            r"\[(?P<elapsed>\d{1,2}:\d{2}(?::\d{2})?)(?:<(?P<eta>\d{1,2}:\d{2}(?::\d{2})?))?",
            text
        )
        if m2:
            self.lbl_elapsed_step.setText(f"Elapsed (Current step): {m2.group('elapsed')}")
            eta = m2.group('eta') or "—"
            self.lbl_eta_step.setText(f"ETA (Current step): {eta}")

    def _set_phase(self, name: str):
        key = name.lower()

        if key == self._phase:
            return

        self._phase = key
        self.lbl_step.setText(name)
        self.lbl_step.setStyleSheet("")
        if key.startswith("finished"):
            self.lbl_step.setStyleSheet("color: #2aa52a;")
        elif key.startswith("aborted") or key.startswith("failed"):
            self.lbl_step.setStyleSheet("color: #d12f2f;")

        # Step-Reset (nur Step, nicht Total)
        self._last_step_pct = 0
        self.pb_step.setValue(0)
        self.lbl_elapsed_step.setText("Elapsed (Current step): —")
        self.lbl_eta_step.setText("ETA (Current step): —")

    def _reset_progress(self):
        self._overall_progress = 0.0
        self._last_step_pct = None
        self._start_time = None
        self.pb_step.setValue(0)
        self.pb_total.setValue(0)
        self.lbl_elapsed_step.setText("Elapsed (Current step): —")
        self.lbl_eta_step.setText("ETA (Current step): —")
        self.lbl_elapsed_total.setText("Elapsed (Total): —")

        self._last_preview_check = 0.0
        self._preview_pix = None
        self.lbl_preview.setPixmap(QtGui.QPixmap())
        self.lbl_preview.setText("No preview")
        self._run_output = None

    def _update_progress(self, pct=None, frac=None):
        if pct is None and frac is None:
            return
        if frac is not None:
            a, b = frac
            if a and b:
                pct = max(0.0, min(100.0, 100.0 * a / b))
        if pct is None:
            return

        self._last_step_pct = pct
        self.pb_step.setValue(int(pct))

        # in Gesamtfortschritt mappen per STEP_WEIGHTS
        start, end = STEP_WEIGHTS.get(self._phase, (0.0, 1.0))
        total = start*100.0 + (end-start)*pct
        total = max(0.0, min(100.0, total))
        self._overall_progress = total
        self.pb_total.setValue(int(total))

        if self.chk_preview.isChecked():
            now = time.time()
            # alle 1.0 s oder wenn wir ~fertig sind → neu laden
            if (now - self._last_preview_check) > 1.0 or (pct is not None and float(pct) >= 99.0):
                self._try_load_preview(force=True)
                self._last_preview_check = now

    def _tick(self):
        # Nur Gesamtzeit seit Start anzeigen
        if self.running:
            if self._start_time is None:
                self._start_time = time.time()
            elapsed = time.time() - self._start_time
            self.lbl_elapsed_total.setText(f"Elapsed (Total): {hms(elapsed)}")

        # Preview-Check
        self._try_load_preview()

    def _try_load_preview(self, force=False):
        if self.running and self._run_output:
            out = self._run_output
        else:
            out = _norm(self.ed_output.text().strip())

        if not out or out in (".", "./", ".\\"):
            return

        p = Path(out)
        if p.is_dir():
            return

        preview = p.with_suffix("")  # "output"
        preview = preview.parent / f"{preview.name}.preview.jpg"

        if not preview.exists():
            if force:
                self._preview_pix = None
                self.lbl_preview.setPixmap(QtGui.QPixmap())
                self.lbl_preview.setText("No preview")
            return

        reader = QtGui.QImageReader(str(preview))
        img = reader.read()
        if img.isNull():
            print(f"Preview load failed for {preview}: {reader.errorString()}")
            return

        self._preview_pix = QtGui.QPixmap.fromImage(img)
        self.lbl_preview.setText("")
        self._apply_preview_pixmap()

    # ===================== Helper =====================
    def _sync_bpm_ui(self):
        detect = self.chk_bpm.isChecked()
        self.ed_bpm.setEnabled(not detect)

    def _apply_profile(self):
        prof = self.cb_profile.currentText()
        cfg = RENDER_PROFILES.get(prof, {})
        if "codec" in cfg:
            idx = self.cb_codec.findText(cfg["codec"])
            if idx >= 0:
                self.cb_codec.setCurrentIndex(idx)
        if "bitrate" in cfg:
            self.ed_bitrate.setText(cfg["bitrate"])
        if "threads" in cfg:
            self.ed_threads.setText(cfg["threads"])
        if "preset" in cfg:
            # preset wird nach _sync_preset_choices gesetzt
            pass

    def _sync_preset_choices(self):
        codec = self.cb_codec.currentText()
        presets = CODEC_PRESETS.get(codec, [])
        self.cb_preset.clear()
        if presets:
            self.cb_preset.addItems(presets)
            default = DEFAULT_PRESET_BY_CODEC.get(codec, presets[0])
            idx = self.cb_preset.findText(default)
            self.cb_preset.setCurrentIndex(idx if idx >= 0 else 0)
            self.cb_preset.setEnabled(True)
        else:
            self.cb_preset.setEnabled(False)

    def _browse_audio(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select audio", "",
                                                     "Audio (*.mp3 *.wav *.flac *.m4a);;All files (*.*)")
        if f:
            self.ed_audio.setText(_norm(f))
            self._autofill_output_from_audio()

        self._validate_inputs(False)

    def _add_video_row(self, path: str = "", weight_text: str = ""):
        """
        Fügt eine Zeile hinzu: [Pfad-QLineEdit][Browse…][Gewicht-QLineEdit ('1' implizit)]
        Auto-Append: Sobald die letzte Zeile einen Pfad bekommt, wird eine neue leere Zeile erzeugt.
        """
        row_w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row_w);
        h.setContentsMargins(0, 0, 0, 0);
        h.setSpacing(6)

        ed_path = QtWidgets.QLineEdit()
        ed_path.setPlaceholderText("Video-Ordner auswählen…")
        if path: ed_path.setText(_norm(path))

        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.setCursor(QtCore.Qt.PointingHandCursor)

        ed_weight = QtWidgets.QLineEdit()
        ed_weight.setPlaceholderText("1")  # leere Eingabe ⇒ Gewicht = 1
        ed_weight.setFixedWidth(60)
        # Nur positive Integer zulassen (optional)
        int_validator = QtGui.QIntValidator(1, 99999, self)
        ed_weight.setValidator(int_validator)
        if weight_text:
            ed_weight.setText(weight_text.strip())

        # Stretch so verteilen, dass Pfad schön breit ist
        ed_path.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        h.addWidget(ed_path, 1)
        h.addWidget(QtWidgets.QLabel("Weight:"), 0)
        h.addWidget(ed_weight, 0)
        h.addWidget(btn_browse, 0)

        # Zeilen-Objekt in Liste halten
        row_obj = {"w": row_w, "path": ed_path, "weight": ed_weight, "btn": btn_browse}
        self.video_rows.append(row_obj)
        self.videos_layout.addWidget(row_w)

        # Browse-Handler (für diese Zeile)
        def _browse_this_row():
            start_dir = _norm(ed_path.text().strip()) or _norm(os.getcwd())
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select video folder", start_dir)
            if d:
                ed_path.setText(_norm(d))
            self._validate_inputs(False)

        btn_browse.clicked.connect(_browse_this_row)

        # Auto-Append, wenn letzte Zeile befüllt wird
        def _maybe_append_new(v: str):
            # nur reagieren, wenn dies die *letzte* Zeile ist
            if row_obj is self.video_rows[-1]:
                if v.strip():
                    # aber nur, wenn noch keine leere Abschlusszeile existiert
                    self._add_video_row()
            self._validate_inputs(False)

        ed_path.textChanged.connect(_maybe_append_new)

    def _iter_filled_video_rows(self):
        """
        Generator über befüllte Pfad-Zeilen (Pfad ≠ leer).
        Liefert Tupel (path_norm, weight_int_or_1, original_weight_text)
        """
        for r in self.video_rows:
            p = r["path"].text().strip()
            if not p:
                continue
            p = _norm(p)
            wt = r["weight"].text().strip()
            w = int(wt) if wt else 1
            yield (p, w, wt)

    def _build_videos_arg(self) -> str:
        """
        Baut den *einen* String für --videos:
        "DIR[:weight],DIR[:weight],..."
        Weight wird nur angehängt, wenn != 1.
        """
        parts = []
        for p, w, _wt in self._iter_filled_video_rows():
            if w == 1:
                parts.append(p)
            else:
                parts.append(f"{p}:{w}")
        return ",".join(parts)

    def _check_video_rows_valid(self) -> bool:
        """
        Validiert: Jeder gefüllte Pfad muss ein existierendes Verzeichnis sein.
        Gewicht (falls angegeben) muss gültig (≥1) sein – das stellt der Validator sicher.
        Mindestens *eine* Zeile muss gefüllt sein.
        """
        any_filled = False
        for r in self.video_rows:
            path_txt = r["path"].text().strip()
            if not path_txt:
                continue
            any_filled = True
            p = _norm(path_txt)
            if not Path(p).is_dir():
                return False
            # weight leer ⇒ 1; falls gesetzt, ist es dank Validator ok
        return any_filled

    def _set_video_rows_from_guess(self, guess_path: str | None):
        """
        Wird beim Start/Autofill verwendet: setzt die *erste* Zeile auf guess_path,
        wenn diese noch leer ist.
        """
        if not guess_path:
            return
        if self.video_rows and not self.video_rows[0]["path"].text().strip():
            self.video_rows[0]["path"].setText(_norm(guess_path))

    def _normalize_videos_text(self, txt: str) -> str:
        """
        Normalisiert die Eingabe:
        - trimmt Leerzeichen um Kommas/Colon,
        - entfernt leere Segmente,
        - lässt Gewicht (falls vorhanden) unverändert.
        Gibt einen einzigen String zurück, der direkt an --videos geht.
        """
        parts = []
        for raw in txt.split(","):
            raw = raw.strip()
            if not raw:
                continue
            if ":" in raw:
                path, weight = raw.split(":", 1)
                path = _norm(path.strip())
                weight = weight.strip()
                parts.append(f"{path}:{weight}" if weight else path)  # leeres Gewicht vermeiden
            else:
                parts.append(_norm(raw))
        return ",".join(parts)

    def _check_video_dirs(self, txt: str) -> bool:
        """
        Prüft, ob jeder vor dem ':' stehende Teil ein existierendes Verzeichnis ist.
        (Gewicht wird ignoriert, kann beliebig sein – CLI prüft Semantik.)
        """
        if not txt.strip():
            return False
        ok = True
        for part in txt.split(","):
            part = part.strip()
            if not part:
                ok = False;
                break
            path = part.split(":", 1)[0].strip()
            if not path or not Path(_norm(path)).is_dir():
                ok = False;
                break
        return ok

    def _browse_output(self):
        f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select output", "",
                                                     "MP4 (*.mp4);;All files (*.*)")
        if f: self.ed_output.setText(_norm(f))

        self._validate_inputs(False)

    def _build_args(self) -> list[str]:
        audio = _norm(self.ed_audio.text())
        videos = self._build_videos_arg()
        output = _norm(self.ed_output.text())
        if not (audio and videos and output):
            return []

        args = [
            "--audio", audio,
            "--videos", videos,
            "--output", output,
            "--width", str(self.sb_w.value()),
            "--height", str(self.sb_h.value()),
            "--fps", str(self.sb_fps.value()),
            "--bg-volume", f"{self.sl_bg.value() / 100.0:.2f}",
            "--clip-volume", f"{self.sl_clip.value() / 100.0:.2f}",
            "--clip-reverb", f"{self.sl_rev.value() / 100.0:.2f}",
            "--codec", self.cb_codec.currentText(),
            "--audio-codec", self.cb_audio.currentText(),
            "--triptych-carry", str(self.ds_triptych_carry.value() / 100.0)
        ]

        if self.cb_preset.isEnabled() and self.cb_preset.currentText():
            args += ["--preset", self.cb_preset.currentText()]
        if self.ed_bitrate.text().strip():
            args += ["--bitrate", self.ed_bitrate.text().strip()]
        if self.ed_threads.text().strip():
            args += ["--threads", self.ed_threads.text().strip()]

        # BPM / Sekunden Handling
        if self.chk_bpm.isChecked():
            # Automatische BPM-Erkennung
            args += ["--bpm-detect"]
            args += [
                "--min-beats", str(self.sb_min_beats.value()),
                "--max-beats", str(self.sb_max_beats.value()),
                "--beat-mode", f"{self.ds_beat_mode.value():.2f}"
            ]
        elif self.ed_bpm.text().strip():
            # Manuelle BPM-Eingabe
            args += ["--bpm", self.ed_bpm.text().strip()]
            args += [
                "--min-beats", str(self.sb_min_beats.value()),
                "--max-beats", str(self.sb_max_beats.value()),
                "--beat-mode", f"{self.ds_beat_mode.value():.2f}"
            ]
        else:
            # Fallback: Sekundenwerte
            args += [
                "--min-seconds", f"{self.ds_min_seconds.value():.2f}",
                "--max-seconds", f"{self.ds_max_seconds.value():.2f}"
            ]

        # Preview explizit mit true/false
        args += ["--preview", "true" if self.chk_preview.isChecked() else "false"]

        if self.chk_pulse.isChecked(): args += ["--pulse-effect"]
        if self.chk_trim.isChecked(): args += ["--trim-large-clips"]

        if self.ds_fadeout.value() > 0: args += ["--fade-out-seconds", f"{self.ds_fadeout.value():.2f}"]

        return args

    def _autofill_video_folder(self):
        """
        Versucht sinnvolle Standard-Ordner zu finden und setzt *erste Zeile* darauf,
        falls sie noch leer ist.
        """
        names = ["videos", "clips", "input", "inputs", "source", "sources", "footage"]
        bases = []
        try:
            bases.append(Path(os.getcwd()))
        except Exception:
            pass
        try:
            bases.append(_base_dir())
        except Exception:
            pass

        seen = set()
        for base in bases:
            if not base or not base.exists():
                continue
            for name in names:
                p = (base / name).resolve()
                if p in seen:
                    continue
                seen.add(p)
                if p.exists() and p.is_dir():
                    self._set_video_rows_from_guess(str(p))
                    return

    def _autofill_output_from_audio(self):
        """
        Falls ein Audiofile gesetzt ist, Output automatisch auf
        denselben Namen mit .mp4 im selben Ordner stellen,
        wenn Output noch leer ist oder gerade überschrieben werden soll.
        """
        audio = _norm(self.ed_audio.text().strip())
        if not audio:
            return

        p = Path(audio)
        if p.exists() and p.is_file():
            candidate = p.with_suffix(".mp4")
            # Nur setzen, wenn das Output-Feld leer ist oder noch Standard enthält
            if not self.ed_output.text().strip():
                self.ed_output.setText(str(candidate))

    @staticmethod
    def _quote(s: str) -> str:
        return f"\"{s}\"" if " " in s else s

    @staticmethod
    def _which_ffmpeg_bins():
        """Suche ffmpeg/ffprobe in PATH, ansonsten neben der EXE/GUI."""
        ffm, ffp = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if not ffm or not ffp:
            base = _base_dir()
            cand_ffm = base / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            cand_ffp = base / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
            if cand_ffm.exists(): ffm = str(cand_ffm)
            if cand_ffp.exists(): ffp = str(cand_ffp)
        return ffm, ffp

    def _check_ffmpeg(self):
        ffm,ffp=self._which_ffmpeg_bins()
        if ffm and ffp:
            self.lbl_ffmpeg.setText(f"FFmpeg OK ✅  (ffmpeg: {Path(ffm).name}, ffprobe: {Path(ffp).name})")
            self.lbl_ffmpeg.setStyleSheet("color: #2aa52a;")
        else:
            self.lbl_ffmpeg.setText("FFmpeg/ffprobe not found ⚠️ – please install and / or add to PATH")
            self.lbl_ffmpeg.setStyleSheet("color: #d12f2f;")

    def eventFilter(self, obj, event):
        if obj is self.lbl_preview and event.type() == QtCore.QEvent.Resize and self._preview_pix:
            self._apply_preview_pixmap()

        if obj is self.lbl_preview:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                if (event.button() == QtCore.Qt.LeftButton
                    and not self.running
                    and self._run_output):
                    p = Path(self._run_output)
                    if p.exists():
                        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(p)))
                        return True

        return super().eventFilter(obj, event)

    def _apply_preview_pixmap(self):
        # skaliert das aktuell geladene Bild passend zur Labelgröße
        area = self.lbl_preview.size()
        if self._preview_pix and area.width() > 0 and area.height() > 0:
            self.lbl_preview.setPixmap(
                self._preview_pix.scaled(area, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            )

    def _set_field_state(self, widget: QtWidgets.QLineEdit, ok: bool, tip_ok: str, tip_err: str):
        widget.setStyleSheet(OK_CSS if ok else ERROR_CSS)
        widget.setToolTip(tip_ok if ok else tip_err)

    def _collect_inputs(self):
        return {
            "audio": self.ed_audio.text().strip(),
            "video_dir": self._build_videos_arg(),
            "output": self.ed_output.text().strip(),
        }

    def _validate_inputs(self, show_message: bool = False) -> bool:
        if self._suspend_validation:
            return False

        inp = self._collect_inputs()
        ok_audio = bool(inp["audio"])
        ok_out = bool(inp["output"])
        ok_vdir = self._check_video_rows_valid()

        self._set_field_state(
            self.ed_audio, ok_audio,
            "Audio-Datei ist gesetzt.",
            "Bitte eine existierende Audio-Datei wählen."
        )
        self._set_field_state(
            self.ed_output, ok_out,
            "Ausgabedatei ist gesetzt.",
            "Bitte einen gültigen Ausgabepfad angeben."
        )

        if ok_vdir:
            self.videos_container.setStyleSheet("")
            self.videos_container.setToolTip("Video-Ordner/Liste ist gültig.")
        else:
            # Leichte rote Outline als Hinweis
            self.videos_container.setStyleSheet("QWidget { border: 1px solid #d9534f; border-radius: 4px; }")
            self.videos_container.setToolTip("Bitte gültige(n) Ordner angeben. Leere Gewichtung bedeutet 1.")

        all_ok = ok_audio and ok_vdir and ok_out

        # Start-Button nur aktivieren, wenn alles passt
        self.btn_start.setEnabled(all_ok and not self.running)

        if show_message and not all_ok:
            missing = []
            if not ok_audio: missing.append("Audio-Datei")
            if not ok_vdir:  missing.append("Video-Ordner")
            if not ok_out:   missing.append("Ausgabedatei")
            QtWidgets.QMessageBox.warning(
                self, "Pflichtfelder fehlen",
                "Bitte folgende Felder prüfen:\n• " + "\n• ".join(missing)
            )
        return all_ok

def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", Path(__file__).parent)
    return str(Path(base, rel))

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(resource_path("assets/icon.ico")))

    w = PMVeaverQt()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
