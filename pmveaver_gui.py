# pmveaver_gui.py
import os, sys, re, time, shutil, json
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
import qdarktheme
import builtins

__version__ = "1.4.0"

APP_TITLE = "PMVeaver v" + __version__

# Progress weights
STEP_WEIGHTS = {
    "downloading redgifs": (0.0, 0.0),
    "collecting clips": (0.0, 0.1),
    "building video": (0.1, 0.125),
    "writing audio": (0.125, 0.15),
    "writing video": (0.15, 1.00),
}
STEP_ORDER = list(STEP_WEIGHTS.keys())

# ---------------- Codec presets / profiles (taken from your Tkinter original, simplified) ------------
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

ERROR_CSS = "QLineEdit{border:1px solid #d9534f; border-radius:4px;}"
OK_CSS    = ""  # Default style


def _norm(p: str) -> str:
    return os.path.normpath(os.path.expandvars(os.path.expanduser(p or "")))

def _base_dir() -> Path:
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent

def _find_cli_candidate():
    """Prefer pmveaver.exe next to the GUI, otherwise pmveaver.py (in the same folder)."""
    base = _base_dir()
    cand_exe = base / "pmveaver.exe"
    if cand_exe.exists():
        return str(cand_exe), True
    cand_py = base / "pmveaver.py"
    if cand_py.exists():
        return str(cand_py), False
    # Fallback: search in working directory
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

        # Normalize for visualization (Y=0..1)
        mx = max(ws) if ws else 1.0
        ws = [w / mx for w in ws]
        return evens, ws

    def paintEvent(self, e: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect().adjusted(6, 6, -6, -6)

        # Background / Frame
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(self.palette().base())
        p.drawRect(rect)

        # Axes
        p.setPen(self.palette().mid().color())
        p.drawRect(rect)

        evens, ws = self._weights()
        if not evens:
            return

        # Bar width
        n = len(evens)
        gap = 4
        bar_w = max(6, (rect.width() - gap*(n+1)) // max(1, n))

        # Text color
        txt_pen = QtGui.QPen(self.palette().text().color())

        # Bars
        x = rect.left() + gap
        max_h = rect.height()  # Space for labels
        bar_brush = QtGui.QBrush(self.palette().highlight().color())

        for i, (b, w) in enumerate(zip(evens, ws)):
            h = int(max_h * w)
            bar_rect = QtCore.QRect(x, rect.bottom() - h - 24, bar_w, h)
            p.fillRect(bar_rect, bar_brush)

            # Beat label
            p.setPen(txt_pen)
            lbl = str(b) + ' beats'
            metrics = p.fontMetrics()
            tw = metrics.horizontalAdvance(lbl)
            p.drawText(x + (bar_w - tw)//2, rect.bottom() - 8, lbl)

            x += bar_w + gap

        # Mode marker (vertical line)
        lo, hi, _ = self._even_list(self._min_beats, self._max_beats)
        if hi > lo:
            rel = (self._mode_pos)  # 0..1
            x_mode = rect.left() + int(rect.width() * rel)
            pen = QtGui.QPen(self.palette().highlight().color())
            pen.setStyle(QtCore.Qt.DashLine)
            p.setPen(pen)
            p.drawLine(x_mode, rect.top(), x_mode, rect.bottom())

class ConsoleWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(None)
        self.setWindowTitle("PMVeaver – Console")
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.resize(800, 500)

        layout = QtWidgets.QVBoxLayout(self)
        self.edit = QtWidgets.QPlainTextEdit(self)
        self.edit.setReadOnly(True)
        # Monospace
        candidates = ["Cascadia Mono", "Consolas"]
        for fam in candidates:
            if fam in QtGui.QFontDatabase.families():
                f = QtGui.QFont(fam)
                break
        else:
            f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        f.setPointSize(10)
        self.edit.setFont(f)
        layout.addWidget(self.edit)

        btn_row = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("Copy all")
        btn_clear = QtWidgets.QPushButton("Clear")
        btn_copy.clicked.connect(self._copy_all)
        btn_clear.clicked.connect(self.clear)

        btn_row.addStretch(1)
        btn_row.addWidget(btn_copy)
        btn_row.addWidget(btn_clear)

        layout.addLayout(btn_row)

        self._cr_active = False
        # Filter out ANSI CSI/OSC sequences (colors, cursor control)
        self._ansi_re = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]|\x1b\]0;.*?\x07')

    # --- Helpers
    def _cursor_end(self):
        cur = self.edit.textCursor()
        cur.movePosition(QtGui.QTextCursor.End)
        return cur

    def _replace_current_line(self, text: str, fmt):
        cur = self._cursor_end()
        cur.movePosition(QtGui.QTextCursor.StartOfBlock, QtGui.QTextCursor.KeepAnchor)
        cur.insertText(text, fmt)

    def append_text(self, s: str, gui=False):
        if not s:
            return

        # Remove ANSI; normalize Windows CRLF to \n
        s = s.replace('\r\n', '\n')
        s = self._ansi_re.sub('', s)

        fmt = QtGui.QTextCharFormat()
        if gui:
            fmt.setForeground(QtGui.QBrush(QtGui.QColor("#8571c9")))
        else:
            fmt.setForeground(QtGui.QBrush(self.palette().text().color()))

        # Split, keep separators -> ['chunk', '\r', 'chunk', '\n', ...]
        parts = re.split(r'(\r|\n)', s)

        for tok in parts:
            if tok == '':
                continue
            if tok == '\r':
                # Next text replaces current line (tqdm update)
                self._cr_active = True
                continue
            if tok == '\n':
                # Force new line
                self._cursor_end().insertText('\n')
                self._cr_active = False
                continue

            # normal text
            if self._cr_active:
                self._replace_current_line(tok, fmt)
                self._cr_active = False
            else:
                self._cursor_end().insertText(tok, fmt)

        self.edit.ensureCursorVisible()

    def clear(self):
        self.edit.setPlainText("")

    def _copy_all(self):
        self.edit.selectAll()
        self.edit.copy()
        # Reset selection to end
        self.edit.moveCursor(QtGui.QTextCursor.End)


class PMVeaverQt(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        fid = QtGui.QFontDatabase.addApplicationFont(resource_path("assets/MaterialSymbolsOutlined.ttf"))
        family = QtGui.QFontDatabase.applicationFontFamilies(fid)[0]

        self.icon_font = QtGui.QFont(family)
        self.icon_font.setPointSize(18)

        self.console_win = ConsoleWindow()

        def gui_print(*args, **kwargs):
            text = " ".join(str(a) for a in args)

            self._orig_print(text, **kwargs)
            if self.console_win:
                self.console_win.append_text(text + "\n", gui=True)

        self._orig_print = builtins.print
        builtins.print = gui_print

        self.setWindowTitle(APP_TITLE)

        # --- Automatic system dark/light mode
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
        self._apply_profile()        # initial profile settings
        self._sync_preset_choices()  # preset selection for codec
        self._sync_bpm_ui()

        # Timer for ETA/preview
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(200)

    # ===================== UI =====================
    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 16, 24)

        main_split = QtWidgets.QHBoxLayout()
        root.addLayout(main_split, stretch=1)
        main_split.addSpacing(16)

        left_box = QtWidgets.QVBoxLayout()

        # ----- Sources / Output
        src_group = QtWidgets.QGroupBox("Sources / Output")
        src_form = QtWidgets.QGridLayout(src_group)

        self.ed_audio = FileDropLineEdit()
        self.ed_audio.editingFinished.connect(self._autofill_output_from_audio)

        btn_audio = self.IconButton("\ueb82")
        btn_audio.clicked.connect(self._browse_audio)

        src_form.addWidget(QtWidgets.QLabel("Audio:"),        0, 0)
        src_form.addWidget(self.ed_audio,                     0, 1)
        src_form.addWidget(btn_audio,                         0, 2)

        src_form.addItem(QtWidgets.QSpacerItem(0, 20), 1, 0)

        src_form.addWidget(QtWidgets.QLabel("Source folders:"), 2, 0)
        self.videos_container = QtWidgets.QWidget()
        self.videos_layout = QtWidgets.QVBoxLayout(self.videos_container)
        self.videos_layout.setContentsMargins(0, 0, 0, 0)
        self.videos_layout.setSpacing(6)

        self.video_rows: list[dict] = []
        self._add_video_row()

        src_form.addWidget(self.videos_container,             2, 1, 1, 2)

        self.chk_trim = QtWidgets.QCheckBox("Trim long clips")
        src_form.addWidget(self.chk_trim, 3, 1)

        src_form.addItem(QtWidgets.QSpacerItem(0, 20), 4, 0)


        src_form.addWidget(QtWidgets.QLabel("Output file:"),  5, 0)
        self.ed_output = FileDropLineEdit()
        src_form.addWidget(self.ed_output,                    5, 1)
        btn_output = self.IconButton("\uf17f")
        btn_output.clicked.connect(self._browse_output)
        src_form.addWidget(btn_output,                        5, 2)

        src_form.addItem(QtWidgets.QSpacerItem(0, 20), 6, 0)

        src_form.addWidget(QtWidgets.QLabel("Intro file:"),  7, 0)
        self.ed_intro = FileDropLineEdit()
        src_form.addWidget(self.ed_intro,                    7, 1)
        btn_intro = self.IconButton("\ueb87")
        btn_intro.clicked.connect(self._browse_intro)
        src_form.addWidget(btn_intro,                        7, 2)

        src_form.addWidget(QtWidgets.QLabel("Outro file:"),  8, 0)
        self.ed_outro = FileDropLineEdit()
        src_form.addWidget(self.ed_outro,                    8, 1)
        btn_outro = self.IconButton("\ueb87")
        btn_outro.clicked.connect(self._browse_outro)
        src_form.addWidget(btn_outro,                        8, 2)

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

        # Current step + Elapsed/ETA of the current step
        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("▼ Current step:"))
        self.lbl_step = QtWidgets.QLabel("—")
        row1.addWidget(self.lbl_step)
        row1.addStretch()

        self.lbl_elapsed_step = QtWidgets.QLabel("Elapsed: —")
        self.lbl_eta_step = QtWidgets.QLabel("ETA: —")
        row1.addWidget(self.lbl_elapsed_step)
        row1.addSpacing(10)
        row1.addWidget(self.lbl_eta_step)
        right_box.addLayout(row1)

        # Step progress
        row2 = QtWidgets.QHBoxLayout()
        self.pb_step = QtWidgets.QProgressBar()
        self.pb_step.setRange(0, 100)
        self.pb_step.setValue(0)
        row2.addWidget(self.pb_step, stretch=1)
        right_box.addLayout(row2)

        # Total progress
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
        btn_row.setSpacing(8)

        self.btn_start = QtWidgets.QPushButton(" Start")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.clicked.connect(self.start)
        self.btn_start.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_start.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                     QtWidgets.QSizePolicy.Fixed)

        # larger, bold font
        f = self.btn_start.font()
        f.setBold(True)
        f.setPointSize(int(f.pointSize() * 1.25))
        self.btn_start.setFont(f)
        self.btn_start.setIcon(self.qicon_from_glyph("\ue1c4"))
        self.btn_start.setIconSize(QtCore.QSize(32, 32))

        self.btn_start.setStyleSheet("""
        QPushButton#btnStart {
            border-radius: 8px;
            padding: 8px 18px;
        }
        """)
        btn_row.addWidget(self.btn_start)

        self.btn_stop = QtWidgets.QPushButton(" Stop")
        self.btn_stop.setIcon(self.qicon_from_glyph("\ue5c9"))
        self.btn_stop.setIconSize(QtCore.QSize(24, 24))
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_stop, 0, QtCore.Qt.AlignBottom)

        self.btn_console = QtWidgets.QPushButton(" Console…")
        self.btn_console.setIcon(self.qicon_from_glyph("\ueb8e"))
        self.btn_console.setIconSize(QtCore.QSize(24, 24))
        self.btn_console.setEnabled(True)
        self.btn_console.clicked.connect(lambda: (self.console_win.show(),
                                                  self.console_win.raise_(),
                                                  self.console_win.activateWindow()))
        btn_row.addWidget(self.btn_console, 0, QtCore.Qt.AlignBottom)

        btn_row.addStretch(1)

        self.lbl_ffmpeg = QtWidgets.QLabel("FFmpeg: …")
        btn_row.addWidget(self.lbl_ffmpeg, 0, QtCore.Qt.AlignBottom)
        left_box.addLayout(btn_row)

        main_split.addLayout(left_box, stretch=3)
        main_split.addSpacing(16)

        acc_container = QtWidgets.QWidget()
        acc_layout = QtWidgets.QVBoxLayout(acc_container)
        acc_layout.setContentsMargins(8, 8, 8, 8)

        presets_container = QtWidgets.QWidget()
        presets_layout = QtWidgets.QHBoxLayout(presets_container)
        presets_layout.setContentsMargins(0, 0, 0, 8)
        presets_layout.setSpacing(8)
        presets_layout.addWidget(QtWidgets.QLabel("Preset: "), stretch=0)

        self.cb_preset = QtWidgets.QComboBox()
        self.cb_preset.currentIndexChanged.connect(self._load_preset)
        presets_layout.addWidget(self.cb_preset, stretch=1)
        QtCore.QTimer.singleShot(0, self._scan_presets)  # initially populate list

        btn_presets_reload = self.IconButton("\ue5d5")
        btn_presets_reload.clicked.connect(self._scan_presets)
        presets_layout.addWidget(btn_presets_reload)

        self.btn_preset_delete = self.IconButton("\ue92b")
        self.btn_preset_delete.clicked.connect(self._delete_preset)
        presets_layout.addWidget(self.btn_preset_delete)

        btn_preset_save = self.IconButton("\ue161")
        btn_preset_save.clicked.connect(self._save_preset)
        presets_layout.addWidget(btn_preset_save)

        acc_layout.addWidget(presets_container)

        # ----- Accordion (QToolBox) for settings
        acc = QtWidgets.QToolBox()
        acc.addItem(self._panel_frame_render(), "Frame / Render")
        acc.addItem(self._panel_seed(),         "Seed")
        acc.addItem(self._panel_effects(),      "Filters / Effects")
        acc.addItem(self._panel_audio_mix(),    "Audio Mix")
        acc.addItem(self._panel_bpm(),          "BPM / Beat lengths")
        acc.addItem(self._panel_time_fallback(),"Time-based fallback (when BPM disabled)")
        acc.addItem(self._panel_codecs(),       "Codecs / Performance")
        acc_layout.addWidget(acc)

        settings_group = QtWidgets.QGroupBox("Settings")
        pg = QtWidgets.QHBoxLayout(settings_group)
        pg.setSpacing(8)

        pg.addWidget(acc_container)

        main_split.addWidget(settings_group, stretch=2)

        main_split.addItem(QtWidgets.QSpacerItem(8, 0))

        # automatic size
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

        # Make all three equally stretchable
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

        # Stretch all three spinbox columns equally
        for col in (1, 3):
            g.setColumnStretch(col, 1)
        # Keep labels narrow
        for col in (0, 2):
            g.setColumnMinimumWidth(col, 80)

        return w
    
    def _panel_seed(self):
        w = QtWidgets.QWidget()
        w.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        g = QtWidgets.QGridLayout(w)
        g.setSpacing(16)

        self.chk_seed = QtWidgets.QCheckBox("Use fixed seed")
        self.chk_seed.setChecked(False)

        self.ed_seed = QtWidgets.QSpinBox()
        self.ed_seed.setMinimum(0)
        self.ed_seed.setMaximum((2 ** 31) - 1)
        self.ed_seed.setEnabled(False)

        self.chk_seed.toggled.connect(self.ed_seed.setEnabled)

        g.addWidget(self.chk_seed, 0, 0)
        g.addWidget(self.ed_seed, 0, 1)

        return w

    def _panel_effects(self):
        w = QtWidgets.QWidget()
        w.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        g = QtWidgets.QGridLayout(w)
        g.setSpacing(16)

        # Sliders
        def slider(init, to=300):
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(0, to)
            s.setValue(init)
            return s

        self.sl_contrast = slider(100)
        self.sl_saturation = slider(100)

        # Editable numeric fields (display + typing allowed)
        def dspin(minv, maxv, init, step=1, suffix="%"):
            ds = QtWidgets.QDoubleSpinBox()
            ds.setRange(minv, maxv)
            ds.setDecimals(0)
            ds.setSingleStep(step)
            ds.setValue(init)
            ds.setSuffix(suffix)
            ds.setFixedWidth(72)
            return ds

        self.ds_contrast = dspin(0, 300, 100)
        self.ds_saturation = dspin(0, 300, 100)

        # Bidirectional binding (Slider <-> SpinBox)
        self.sl_contrast.valueChanged.connect(lambda v: self.ds_contrast.setValue(v))
        self.ds_contrast.valueChanged.connect(lambda val: self.sl_contrast.setValue(int(round(val))))
        self.sl_contrast.valueChanged.connect(lambda _: self._update_lut_preview())
        self.ds_contrast.valueChanged.connect(lambda _: self._update_lut_preview())

        self.sl_saturation.valueChanged.connect(lambda v: self.ds_saturation.setValue(v))
        self.ds_saturation.valueChanged.connect(lambda val: self.sl_saturation.setValue(int(round(val))))
        self.sl_saturation.valueChanged.connect(lambda _: self._update_lut_preview())
        self.ds_saturation.valueChanged.connect(lambda _: self._update_lut_preview())

        # Layout: [Label | Slider | Zahl] x 3 – Slider-Spalten gleich stretchen
        g.addWidget(QtWidgets.QLabel("Contrast"), 0, 0)
        g.addWidget(self.sl_contrast, 0, 1, 1, 3)
        g.addWidget(self.ds_contrast, 0, 4)

        g.addWidget(QtWidgets.QLabel("Saturation"), 1, 0)
        g.addWidget(self.sl_saturation, 1, 1, 1, 3)
        g.addWidget(self.ds_saturation, 1, 4)

        # --- 3D LUT selection ---
        self.cb_lut = QtWidgets.QComboBox()
        btn_lut_reload = QtWidgets.QPushButton(" Reload")
        btn_lut_reload.setIcon(self.qicon_from_glyph("\ue5d5"))
        btn_lut_reload.setIconSize(QtCore.QSize(24, 24))
        btn_lut_reload.setCursor(QtCore.Qt.PointingHandCursor)

        g.addWidget(QtWidgets.QLabel("3D LUT"), 2, 0)
        g.addWidget(self.cb_lut, 2, 1, 1, 3)
        g.addWidget(btn_lut_reload, 2, 4)

        btn_lut_reload.clicked.connect(lambda: (self._scan_luts(), self._update_lut_preview()))
        QtCore.QTimer.singleShot(0, self._scan_luts)  # populate initially once UI is ready

        self.chk_pulse = QtWidgets.QCheckBox("Beat pulse effect")
        g.addWidget(self.chk_pulse, 3, 0, 1, 2)

        lab_fadeout = QtWidgets.QLabel("Fade out")
        lab_fadeout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.ds_fadeout = QtWidgets.QDoubleSpinBox()
        self.ds_fadeout.setRange(0, 5.0)
        self.ds_fadeout.setDecimals(2)
        self.ds_fadeout.setSingleStep(0.10)
        self.ds_fadeout.setSuffix(" s")
        g.addWidget(lab_fadeout, 3, 3)
        g.addWidget(self.ds_fadeout, 3, 4)

        # --- LUT Preview ---
        self.lbl_lut_preview = QtWidgets.QLabel()
        self.lbl_lut_preview.setFixedHeight(110)
        self.lbl_lut_preview.setFrameShape(QtWidgets.QFrame.Panel)
        self.lbl_lut_preview.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.lbl_lut_preview.setAlignment(QtCore.Qt.AlignCenter)
        g.addWidget(self.lbl_lut_preview, 5, 0, 1, 5)

        # Signale
        self.cb_lut.currentIndexChanged.connect(self._update_lut_preview)
        QtCore.QTimer.singleShot(0, self._scan_luts)  # initially populate list
        QtCore.QTimer.singleShot(50, self._update_lut_preview)  # and show a preview immediately

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

        # Sliders (internally 0..200 or 0..100)
        def slider(init, to=200):
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(0, to)
            s.setValue(init)
            return s

        self.sl_bg = slider(100)  # 1.00x
        self.sl_clip = slider(80)  # 0.80x
        self.sl_rev = slider(20, to=100)  # 0.20

        # Editable numeric fields (display + manual typing allowed)
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

        # Bidirectional binding (Slider <-> SpinBox)
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

        # Changing min/max updates the graph
        self.sb_min_beats.valueChanged.connect(lambda _: self._update_dist_plot())
        self.sb_max_beats.valueChanged.connect(lambda _: self._update_dist_plot())

        QtCore.QTimer.singleShot(0, self._update_dist_plot)

        # stretch all three spinbox columns equally
        for col in (1, 3):
            g.setColumnStretch(col, 1)
        # keep labels narrow
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

        # Evenly stretchable
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

        # Consistency: min ≤ max
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
        self.cb_codec_preset  = QtWidgets.QComboBox()

        self.ed_bitrate = QtWidgets.QLineEdit(); self.ed_bitrate.setPlaceholderText("Bitrate (e.g., 8M)")
        self.ed_threads = QtWidgets.QLineEdit(); self.ed_threads.setPlaceholderText("Threads")
        self.chk_preview = QtWidgets.QCheckBox("Generate Preview"); self.chk_preview.setChecked(True)

        self.cb_profile.currentIndexChanged.connect(self._apply_profile)
        self.cb_codec.currentIndexChanged.connect(self._sync_preset_choices)

        g.addWidget(QtWidgets.QLabel("Hardware profile"), 0,0); g.addWidget(self.cb_profile, 0,1)

        g.addWidget(QtWidgets.QLabel("Video codec"),      1,0); g.addWidget(self.cb_codec,   1,1)
        g.addWidget(QtWidgets.QLabel("Audio codec"),      1,2); g.addWidget(self.cb_audio,   1,3)

        g.addWidget(QtWidgets.QLabel("Codec Preset"),     2,0); g.addWidget(self.cb_codec_preset,  2,1)

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
            QtWidgets.QMessageBox.warning(self, "Missing input", "Please check audio/video/output.")
            return

        cli, is_exe = _find_cli_candidate()
        if cli is None:
            QtWidgets.QMessageBox.critical(self, "Not found",
                                           "pmveaver.exe / pmveaver.py not found.\nPlace it next to this GUI.")
            return

        out_txt = _norm(self.ed_output.text().strip())
        if not self._handle_output_conflict(Path(out_txt)):
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

        if hasattr(self, "console_win") and self.console_win:
            self.console_win.clear()

        print("PMVeaver GUI - Starting:", " ".join(self._quote(a) for a in ([program] + full_args)), flush=True)

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

        if hasattr(self, "console_win") and self.console_win:
            self.console_win.append_text(text)

        for line in text.splitlines():
            if "random seed:" in line:
                m = re.search(r"random seed:\s*(\d+)", line)
                if m:
                    seed_val = m.group(1)
                    self.ed_seed.setValue(int(seed_val))

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

        if "downloading redgifs" in sl:
            return "Downloading redgifs"

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
        # Very tolerant: percent + bar + bracket block
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
            self.lbl_elapsed_step.setText(f"Elapsed: {m2.group('elapsed')}")
            eta = m2.group('eta') or "—"
            self.lbl_eta_step.setText(f"ETA: {eta}")

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

        # Step reset (only step, not total)
        self._last_step_pct = 0
        self.pb_step.setValue(0)
        self.lbl_elapsed_step.setText("Elapsed: —")
        self.lbl_eta_step.setText("ETA: —")

    def _reset_progress(self):
        self._overall_progress = 0.0
        self._last_step_pct = None
        self._start_time = None
        self.pb_step.setValue(0)
        self.pb_total.setValue(0)
        self.lbl_elapsed_step.setText("Elapsed: —")
        self.lbl_eta_step.setText("ETA: —")
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

        # Map into overall progress using STEP_WEIGHTS
        start, end = STEP_WEIGHTS.get(self._phase, (0.0, 1.0))
        total = start*100.0 + (end-start)*pct
        total = max(0.0, min(100.0, total))
        self._overall_progress = total
        self.pb_total.setValue(int(total))

        if self.chk_preview.isChecked():
            now = time.time()
            # Reload every 1.0 s or when almost finished
            if (now - self._last_preview_check) > 1.0 or (pct is not None and float(pct) >= 99.0):
                self._try_load_preview(force=True)
                self._last_preview_check = now

    def _tick(self):
        # Show only total elapsed time since start
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
        self.cb_codec_preset.clear()
        if presets:
            self.cb_codec_preset.addItems(presets)
            default = DEFAULT_PRESET_BY_CODEC.get(codec, presets[0])
            idx = self.cb_codec_preset.findText(default)
            self.cb_codec_preset.setCurrentIndex(idx if idx >= 0 else 0)
            self.cb_codec_preset.setEnabled(True)
        else:
            self.cb_codec_preset.setEnabled(False)

    def _browse_audio(self):
        start_dir = _norm(self.ed_audio.text().strip()) or _norm(os.getcwd())
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select audio", start_dir,
                                                     "Audio (*.mp3 *.wav *.flac *.m4a);;All files (*.*)")
        if f:
            self.ed_audio.setText(_norm(f))
            self._autofill_output_from_audio()

        self._validate_inputs(False)

    def _add_video_row(self, path: str = "", weight_text: str = ""):
        """
        Adds a row: [Path-QLineEdit][Browse…][Weight-QLineEdit ('1' implicit)]
        Auto-append: As soon as the last row gets a path, a new empty row is created.
        """
        row_w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row_w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        ed_path = DirDropLineEdit()
        ed_path.setPlaceholderText("Video folder path or Redgifs search query")
        if path: ed_path.setText(_norm(path))

        btn_browse = self.IconButton("\ue2c7")

        ed_weight = QtWidgets.QLineEdit()
        ed_weight.setPlaceholderText("1")  # leere Eingabe ⇒ Gewicht = 1
        ed_weight.setFixedWidth(60)
        # Allow only positive integers (optional)
        int_validator = QtGui.QIntValidator(1, 99999, self)
        ed_weight.setValidator(int_validator)
        if weight_text:
            ed_weight.setText(weight_text.strip())

        # Distribute stretch so that path field is wide
        ed_path.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        h.addWidget(ed_path, 1)
        h.addWidget(QtWidgets.QLabel("Weight:"), 0)
        h.addWidget(ed_weight, 0)
        h.addWidget(btn_browse, 0)

        # Keep row object in list
        row_obj = {"w": row_w, "path": ed_path, "weight": ed_weight, "btn": btn_browse}
        self.video_rows.append(row_obj)
        self.videos_layout.addWidget(row_w)

        # Browse handler (for this row)
        def _browse_this_row():
            start_dir = _norm(ed_path.text().strip()) or _norm(os.getcwd())
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select video folder", start_dir)
            if d:
                ed_path.setText(_norm(d))
            self._validate_inputs(False)

        btn_browse.clicked.connect(_browse_this_row)

        # Auto-append when last row is filled
        def _maybe_append_new(v: str):
            # react only if this is the *last* row
            if row_obj is self.video_rows[-1]:
                if v.strip():
                    # but only if no empty trailing row exists yet
                    self._add_video_row()
            self._validate_inputs(False)

        ed_path.textChanged.connect(_maybe_append_new)

    def _iter_filled_video_rows(self):
        """
        Generator over filled path rows (path ≠ empty).
        Yields tuples (path_norm, weight_int_or_1, original_weight_text)
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
        Builds the single string for --videos:
        "DIR[:weight],DIR[:weight],..."
        Weight is only appended if != 1.
        """
        parts = []
        for p, w, _wt in self._iter_filled_video_rows():
            if w == 1:
                parts.append(p)
            else:
                parts.append(f"{p}:{w}")
        return ",".join(parts)

    def _set_video_rows_from_guess(self, guess_path: str | None):
        """
        Used at startup/autofill: sets the *first* row to guess_path,
        if it is still empty.
        """
        if not guess_path:
            return
        if self.video_rows and not self.video_rows[0]["path"].text().strip():
            self.video_rows[0]["path"].setText(_norm(guess_path))

    def _normalize_videos_text(self, txt: str) -> str:
        """
        Normalizes the input:
        - trims spaces around commas/colon,
        - removes empty segments,
        - keeps weight (if present) unchanged.
        Returns a single string passed directly to --videos.
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
        Checks if every part before ':' is an existing directory.
        (Weight is ignored, can be arbitrary – CLI checks semantics.)
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
        start_dir = _norm(self.ed_output.text().strip()) or _norm(os.getcwd())
        f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select output", start_dir,
                                                     "MP4 (*.mp4);;All files (*.*)")
        if f: self.ed_output.setText(_norm(f))

        self._validate_inputs(False)

    def _browse_intro(self):
        start_dir = _norm(self.ed_intro.text().strip()) or _norm(os.getcwd())
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select intro", start_dir,
                                                     "Video file (*.mp4 *.mov *.m4v *.mkv *.avi *.webm *.mpg *.gif);;Image file (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*.*)")
        if f:
            self.ed_intro.setText(_norm(f))

        self._validate_inputs(False)

    def _browse_outro(self):
        start_dir = _norm(self.ed_outro.text().strip()) or _norm(os.getcwd())
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select outro", start_dir,
                                                     "Video file (*.mp4 *.mov *.m4v *.mkv *.avi *.webm *.mpg *.gif);;Image file (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*.*)")
        if f:
            self.ed_outro.setText(_norm(f))

        self._validate_inputs(False)

    def _scan_luts(self):
        """List ./luts/ in working directory and populate LUT selection."""
        lut_dir = Path(_norm(os.path.join(os.getcwd(), "luts")))
        items: list[str] = []

        if lut_dir.exists() and lut_dir.is_dir():
            exts = [".cube", ".3dl", ".lut"]
            for ext in exts:
                for p in sorted(lut_dir.glob(f"*{ext}")):
                    if p.is_file():
                        items.append(str(p))

        self.cb_lut.clear()
        self.cb_lut.addItem("(none)")
        for it in items:
            name = os.path.basename(it)
            self.cb_lut.addItem(name, it)

        # Helpful hints
        tip = "3D LUTs are loaded from the ./luts subfolder of the working directory."
        if not items:
            tip += " (No files found – supported extensions: .cube, .3dl, .lut)"
        self.cb_lut.setToolTip(tip)

    def _lut_sample_path(self) -> str:
        """Generate a colored test image (PNG) for LUT previews and return the path."""
        if hasattr(self, "_lut_sample_file") and self._lut_sample_file and Path(self._lut_sample_file).exists():
            return self._lut_sample_file

        tmp = Path(os.path.join(Path.cwd(), ".pmveaver_cache"))
        tmp.mkdir(exist_ok=True)
        p = tmp / "lut_sample.png"

        # Color blocks + gradients
        w, h = 384, 107
        img = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
        painter = QtGui.QPainter(img)

        # 1) Color grid (top)
        cell_w, cell_h = w // 8, h // 3
        for i in range(8):
            for j in range(1):
                r = int(255 * (i / 7))
                g = int(255 * (j / 2))
                b = int(255 * ((7 - i) / 7))
                painter.fillRect(i * cell_w, j * cell_h, cell_w, cell_h, QtGui.QColor(r, g, b))

        # 2) Horizontal RGB gradient (middle)
        y0 = cell_h
        for x in range(w):
            t = x / (w - 1)
            color = QtGui.QColor(int(255 * t), int(255 * (1 - t)), int(255 * (0.5 + 0.5 * t)))
            painter.setPen(color)
            painter.drawLine(x, y0, x, y0 + cell_h - 1)

        # 3) Saturation gradient (bottom)
        y1 = 2 * cell_h
        for y in range(cell_h):
            sat = y / max(1, cell_h - 1)
            for x in range(w):
                hue = x / max(1, w - 1)
                qc = QtGui.QColor.fromHsvF(hue, sat, 1.0)
                painter.setPen(qc)
                painter.drawPoint(x, y1 + y)

        painter.end()
        img.save(str(p), "PNG", quality=95)
        self._lut_sample_file = str(p)
        return self._lut_sample_file

    def _lut_samples(self) -> list[str]:
        grid = self._lut_sample_path()
        paths = [grid] if grid else []

        photo = Path(Path.cwd() / "sample.jpg")
        if photo.exists():
            paths.insert(0, str(photo))
        return paths

    def _update_lut_preview(self):
        """Generate a LUT preview as a collage (photo + color grid) via FFmpeg and display it."""
        # --- Get sources ---
        try:
            samples = self._lut_samples()  # optional: Foto + Grid
        except AttributeError:
            samples = [self._lut_sample_path()]  # nur Grid

        samples = [s for s in samples if s and Path(s).exists()]
        if not samples:
            self.lbl_lut_preview.setText("No sample")
            return

        # --- Find FFmpeg ---
        ffm, _ = self._which_ffmpeg_bins()
        if not ffm:
            self.lbl_lut_preview.setText("FFmpeg not found")
            return

        # --- Check LUT selection ---
        use_lut = False
        lut_esc = ""
        if hasattr(self, "cb_lut"):
            idx = self.cb_lut.currentIndex()
            if idx > 0:
                lut_path = str(self.cb_lut.itemData(idx) or "").strip()
                if lut_path:
                    lut_esc = lut_path.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
                    use_lut = True

        # --- Contrast / Saturation aus GUI ---
        contrast = getattr(self, "ds_contrast", None).value() / 100.0 if hasattr(self, "ds_contrast") else 1.0
        saturation = getattr(self, "ds_saturation", None).value() / 100.0 if hasattr(self, "ds_saturation") else 1.0
        eq_filter = f",eq=contrast={contrast:.2f}:saturation={saturation:.2f}"

        # --- Ziel & Größe ---
        out_dir = Path(os.path.join(Path.cwd(), ".pmveaver_cache"))
        out_dir.mkdir(exist_ok=True)
        out_png = out_dir / "lut_preview.png"
        tile_w, tile_h = 256, 144

        # --- FFmpeg-Argumente vorbereiten ---
        args = ["-y", "-hide_banner", "-nostats", "-loglevel", "error"]
        for s in samples:
            args += ["-i", s]

        chains = []
        outs = []
        for i in range(len(samples)):
            if use_lut:
                chain = (
                    f"[{i}:v]"
                    f"format=gbrp,"
                    f"lut3d=file='{lut_esc}':interp=tetrahedral,"
                    f"format=rgb24,"
                    f"eq=contrast={contrast:.2f}:saturation={saturation:.2f},"
                    f"scale={tile_w}:{tile_h}:flags=bicubic"
                    f"[v{i}]"
                )
            else:
                chain = (
                    f"[{i}:v]"
                    f"format=rgb24,"
                    f"eq=contrast={contrast:.2f}:saturation={saturation:.2f},"
                    f"scale={tile_w}:{tile_h}:flags=bicubic"
                    f"[v{i}]"
                )
            chains.append(chain)
            outs.append(f"[v{i}]")

        fc = ";".join(chains) + ";" + "".join(outs) + f"hstack=inputs={len(samples)}[out]"
        args += ["-filter_complex", fc, "-map", "[out]", "-frames:v", "1", str(out_png)]

        # --- Abort previous preview jobs ---
        if hasattr(self, "_lut_prev_proc") and self._lut_prev_proc:
            try:
                self._lut_prev_proc.kill()
            except Exception:
                pass
            try:
                self._lut_prev_proc.deleteLater()
            except Exception:
                pass

        # --- QProcess starten ---
        self._lut_prev_proc = QtCore.QProcess(self)
        self._lut_prev_proc.finished.connect(lambda *_: self._on_lut_preview_done(str(out_png)))
        self._lut_prev_proc.setProgram(ffm)
        self._lut_prev_proc.setArguments(args)

        self.lbl_lut_preview.setText("rendering…")
        self._lut_prev_proc.start()

    def _on_lut_preview_done(self, out_png: str):
        p = Path(out_png)
        if p.exists():
            pix = QtGui.QPixmap(str(p)).scaled(self.lbl_lut_preview.size(),
                                               QtCore.Qt.KeepAspectRatio,
                                               QtCore.Qt.SmoothTransformation)
            self.lbl_lut_preview.setPixmap(pix)
        else:
            self.lbl_lut_preview.setText("preview failed")

    def _scan_presets(self):
        """List ./presets/ in working directory and populate preset selection."""
        lut_dir = Path(_norm(os.path.join(os.getcwd(), "presets")))
        items: list[str] = []

        if lut_dir.exists() and lut_dir.is_dir():
            exts = [".json"]
            for ext in exts:
                for p in sorted(lut_dir.glob(f"*{ext}")):
                    if p.is_file():
                        items.append(str(p))

        self.cb_preset.clear()

        self.cb_preset.addItem("(Default)", None)

        for it in items:
            name = os.path.basename(it)
            self.cb_preset.addItem(name, it)

        # Helpful hints
        tip = "Presets are loaded from the ./presets subfolder of the working directory."
        if not items:
            tip += " (No files found – supported extensions: .json)"
        self.cb_preset.setToolTip(tip)

    def _delete_preset(self):
        idx = self.cb_preset.currentIndex()
        if idx < 0:
            return

        preset_path = self.cb_preset.itemData(idx)
        if preset_path is None:
            return

        p = Path(preset_path)
        if not p.exists() or not p.is_file():
            self._scan_presets()
            return

        ret = QtWidgets.QMessageBox.question(
            self,
            "Delete preset",
            f"Are you sure you want to delete the preset '{self.cb_preset.currentText()}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if ret == QtWidgets.QMessageBox.Yes:
            try:
                p.unlink()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Preset could ne be deleted:\n{e}")
            self._scan_presets()

    def _save_preset(self):
        preset_data = {
            "width": self.sb_w.value(),
            "height": self.sb_h.value(),
            "triptych-carry": self.ds_triptych_carry.value(),
            "fps": self.sb_fps.value(),

            "contrast": self.sl_contrast.value(),
            "saturation": self.sl_saturation.value(),
            "lut": self.cb_lut.itemData(self.cb_lut.currentIndex()),
            "pulse-effect": self.chk_pulse.isChecked(),
            "fade-out-seconds": self.ds_fadeout.value(),

            "bg-volume": self.sl_bg.value(),
            "clip-volume": self.sl_clip.value(),
            "clip-reverb": self.sl_rev.value(),

            "bpm-detect": self.chk_bpm.isChecked(),
            "bpm": self.ed_bpm.text(),
            "min-beats": self.sb_min_beats.value(),
            "max-beats": self.sb_max_beats.value(),
            "beat-mode": self.ds_beat_mode.value(),

            "min-seconds": self.ds_min_seconds.value(),
            "max-seconds": self.ds_max_seconds.value(),

            "codec": self.cb_codec.currentText(),
            "preset": self.cb_codec_preset.currentText(),
            "bitrate": self.ed_bitrate.text(),
            "threads": self.ed_threads.text(),
            "preview": self.chk_preview.isChecked(),
        }

        preset_name, ok = QtWidgets.QInputDialog.getText(
            None,
            "Name preset",
            "Please enter a name for your preset:",
        )

        if ok and preset_name.strip():
            preset_name = preset_name.strip() + ".json"
            preset_dir = "./presets"
            preset_path = os.path.join(preset_dir, preset_name)

            os.makedirs(preset_dir, exist_ok=True)

            with open(preset_path, "w", encoding="utf-8") as f:
                json.dump(preset_data, f, ensure_ascii=False, indent=4)

            self._scan_presets()

            index = self.cb_preset.findText(preset_name)
            if index != -1:
                self.cb_preset.setCurrentIndex(index)

    def _load_preset(self):
        idx = self.cb_preset.currentIndex()

        if idx < 0:
            self.btn_preset_delete.setEnabled(False)
            return

        preset_path = self.cb_preset.itemData(idx)

        if preset_path is None:
            # Set default values
            self.sb_w.setValue(1920)
            self.sb_h.setValue(1080)
            self.ds_triptych_carry.setValue(30)
            self.sb_fps.setValue(30.0)
            self.sl_contrast.setValue(100)
            self.sl_saturation.setValue(100)
            self.chk_pulse.setChecked(False)
            self.ds_fadeout.setValue(0.0)
            self.sl_bg.setValue(100)
            self.sl_clip.setValue(80)
            self.sl_rev.setValue(20)
            self.chk_bpm.setChecked(True)
            self.ed_bpm.setText("")
            self.sb_min_beats.setValue(2)
            self.sb_max_beats.setValue(8)
            self.ds_beat_mode.setValue(0.25)
            self.ds_min_seconds.setValue(2.0)
            self.ds_max_seconds.setValue(5.0)
            self.cb_codec.setCurrentText("libx264")
            self.cb_codec_preset.setCurrentText("medium")
            self.ed_bitrate.setText("8M")
            self.ed_threads.setText("8")
            self.chk_preview.setChecked(True)
            self.ed_seed.setValue(0)
            self.cb_audio.setCurrentText("aac")
            self.btn_preset_delete.setEnabled(False)
            return

        if preset_path:
            with open(preset_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.sb_w.setValue(int(data["width"]))
            self.sb_h.setValue(int(data["height"]))
            self.ds_triptych_carry.setValue(float(data["triptych-carry"]))
            self.sb_fps.setValue(float(data["fps"]))

            self.sl_contrast.setValue(int(data["contrast"]))
            self.sl_saturation.setValue(int(data["saturation"]))

            lut_target = data["lut"]
            for i in range(self.cb_lut.count()):
                if self.cb_lut.itemData(i) == lut_target:
                    self.cb_lut.setCurrentIndex(i)
                    break
            self.chk_pulse.setChecked(bool(data["pulse-effect"]))
            self.ds_fadeout.setValue(float(data["fade-out-seconds"]))

            self.sl_bg.setValue(int(data["bg-volume"]))
            self.sl_clip.setValue(int(data["clip-volume"]))
            self.sl_rev.setValue(int(data["clip-reverb"]))

            self.chk_bpm.setChecked(bool(data["bpm-detect"]))
            self.ed_bpm.setText(str(data["bpm"]))
            self.sb_min_beats.setValue(int(data["min-beats"]))
            self.sb_max_beats.setValue(int(data["max-beats"]))
            self.ds_beat_mode.setValue(float(data["beat-mode"]))
            self.ds_min_seconds.setValue(float(data["min-seconds"]))
            self.ds_max_seconds.setValue(float(data["max-seconds"]))

            idx = self.cb_codec.findText(str(data["codec"]))
            if idx != -1:
                self.cb_codec.setCurrentIndex(idx)

            idx = self.cb_codec_preset.findText(str(data["preset"]))
            if idx != -1:
                self.cb_codec_preset.setCurrentIndex(idx)

            self.ed_bitrate.setText(str(data["bitrate"]))
            self.ed_threads.setText(str(data["threads"]))
            self.chk_preview.setChecked(bool(data["preview"]))

            self.btn_preset_delete.setEnabled(True)

    def _autofill_video_folder(self):
        """
        Try to find sensible default folders and set the *first row* to one,
        if it is still empty.
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
        If an audio file is set, automatically set output to
        the same name with .mp4 in the same folder,
        if output is still empty or about to be overwritten.
        """
        audio = _norm(self.ed_audio.text().strip())
        if not audio:
            return

        p = Path(audio)
        if p.exists() and p.is_file():
            candidate = p.with_suffix(".mp4")
            # Only set if the output field is empty or still contains the default
            if not self.ed_output.text().strip():
                self.ed_output.setText(str(candidate))

    @staticmethod
    def _quote(s: str) -> str:
        return f"\"{s}\"" if " " in s else s

    @staticmethod
    def _which_ffmpeg_bins():
        """Search for ffmpeg/ffprobe in PATH, otherwise next to EXE/GUI."""
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
        # scales the currently loaded image to the label size
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

        self._set_field_state(
            self.ed_audio, ok_audio,
            "Audio file is set.",
            "Please select an existing audio file."
        )
        self._set_field_state(
            self.ed_output, ok_out,
            "Output file is set.",
            "Please specify a valid output path."
        )

        all_ok = ok_audio and ok_out

        # Start-Button nur aktivieren, wenn alles passt
        self.btn_start.setEnabled(all_ok and not self.running)

        if show_message and not all_ok:
            missing = []
            if not ok_audio: missing.append("Audio file")
            if not ok_out:   missing.append("Output filder")
            QtWidgets.QMessageBox.warning(
                self, "Required fields are missing",
                "Please check the following fields:\n• " + "\n• ".join(missing)
            )
        return all_ok

    def _next_numbered_path(self, p: Path) -> Path:
        """Generate 'name (1).ext', 'name (2).ext', ... until free."""
        stem, suf = p.stem, p.suffix
        i = 1
        while True:
            cand = p.with_name(f"{stem} ({i}){suf}")
            if not cand.exists():
                return cand
            i += 1

    def _handle_output_conflict(self, p: Path) -> bool:
        """
        Show a choice:
          - Overwrite
          - Rename ➜ name (1).ext
          - Cancel
        Return: True = continue, False = abort
        """
        if not p.exists():
            return True
        if p.is_dir():
            QtWidgets.QMessageBox.warning(self, "Invalid path",
                                          "The specified output path is a folder. Please select a file.")
            return False

        cand = self._next_numbered_path(p)

        m = QtWidgets.QMessageBox(self)
        m.setWindowTitle("File already exists")
        m.setIcon(QtWidgets.QMessageBox.Warning)
        m.setText(f"The output file already exists:\n{p}\n\nWhat do you want to do?")
        btn_over = m.addButton("Overwrite", QtWidgets.QMessageBox.DestructiveRole)
        btn_ren = m.addButton("Rename", QtWidgets.QMessageBox.ActionRole)
        btn_cancel = m.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        m.setStyleSheet(
            f"QPushButton {{ min-width: 80px; }}"
        )
        m.exec()


        clicked = m.clickedButton()
        if clicked is btn_over:
            return True
        if clicked is btn_ren:
            self.ed_output.setText(str(cand))  # neuen Namen ins Feld übernehmen
            return True
        return False

    def IconButton(
            self,
            label: str | None = None,
    ) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(label)
        btn.setFont(self.icon_font)
        btn.setCursor(QtCore.Qt.PointingHandCursor)

        btn.setStyleSheet("QPushButton { padding: 2px 4px; }")

        return btn

    def qicon_from_glyph(self, glyph: str) -> QtGui.QIcon:
        # Take color from palette if not explicitly specified
        pal = QtWidgets.QApplication.palette()
        col_norm = pal.buttonText().color()
        col_dis = pal.color(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText)
        point_size = 20
        icon_font = self.icon_font

        def _render(col: QtGui.QColor) -> QtGui.QPixmap:
            fm = QtGui.QFontMetrics(icon_font)
            w = max(fm.horizontalAdvance(glyph), point_size) + 2
            h = max(fm.height(), point_size) + 2

            dpr = QtGui.QGuiApplication.primaryScreen().devicePixelRatio() or 1
            pm = QtGui.QPixmap(int(w * dpr), int(h * dpr))
            pm.fill(QtCore.Qt.transparent)
            pm.setDevicePixelRatio(dpr)

            p = QtGui.QPainter(pm)
            p.setRenderHint(QtGui.QPainter.Antialiasing, True)
            p.setFont(icon_font)
            p.setPen(col)
            p.drawText(QtCore.QRectF(0, 0, w, h), QtCore.Qt.AlignCenter, glyph)
            p.end()
            return pm

        icon = QtGui.QIcon()
        icon.addPixmap(_render(col_norm), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon.addPixmap(_render(col_dis), QtGui.QIcon.Disabled, QtGui.QIcon.Off)
        return icon

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
            "--triptych-carry", str(self.ds_triptych_carry.value() / 100.0),
            "--contrast", f"{self.sl_contrast.value() / 100.0:.2f}",
            "--saturation", f"{self.sl_saturation.value() / 100.0:.2f}",
        ]

        if self.cb_codec_preset.isEnabled() and self.cb_codec_preset.currentText():
            args += ["--preset", self.cb_codec_preset.currentText()]
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

        # Explicitly set preview to true/false
        args += ["--preview", "true" if self.chk_preview.isChecked() else "false"]

        if self.chk_pulse.isChecked(): args += ["--pulse-effect"]
        if self.chk_trim.isChecked(): args += ["--trim-large-clips"]

        if self.ds_fadeout.value() > 0: args += ["--fade-out-seconds", f"{self.ds_fadeout.value():.2f}"]

        intro_txt = self.ed_intro.text().strip()
        if intro_txt:
            intro_path = _norm(intro_txt)
            if Path(intro_path).is_file():
                args += ["--intro", intro_path]

        outro_txt = self.ed_outro.text().strip()
        if outro_txt:
            outro_path = _norm(outro_txt)
            if Path(outro_path).is_file():
                args += ["--outro", outro_path]

        if hasattr(self, "cb_lut"):
            idx = self.cb_lut.currentIndex()
            if idx > 0:  # 0 = "(none)"
                lut_path = self.cb_lut.itemData(idx)
                if lut_path:
                    args += ["--lut", _norm(lut_path)]

        if self.chk_seed.isChecked():
            args += ["--seed", str(self.ed_seed.value())]

        return args

    def closeEvent(self, event):
        if self.console_win is not None:
            self.console_win.close()

        super().closeEvent(event)


def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", Path(__file__).parent)
    return str(Path(base, rel))

class FileDropLineEdit(QtWidgets.QLineEdit):
    """"QLineEdit that accepts file drops (optional: extension whitelist)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            for url in e.mimeData().urls():
                p = Path(url.toLocalFile())
                if p.is_file():
                    e.acceptProposedAction()
                    return
        e.ignore()

    def dropEvent(self, e: QtGui.QDropEvent):
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_file():
                self.setText(_norm(str(p)))
                self.editingFinished.emit()
                break
        e.acceptProposedAction()

class DirDropLineEdit(QtWidgets.QLineEdit):
    """QLineEdit that accepts folder drops."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            for url in e.mimeData().urls():
                p = Path(url.toLocalFile())
                if p.is_dir():
                    e.acceptProposedAction()
                    return
        e.ignore()

    def dropEvent(self, e: QtGui.QDropEvent):
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_dir():
                self.setText(_norm(str(p)))
                self.editingFinished.emit()
                break
        e.acceptProposedAction()

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(resource_path("assets/icon.ico")))
    app.setFont(QtGui.QFont("Segoe UI", 10))

    w = PMVeaverQt()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
