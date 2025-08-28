#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PMVeaver
# --------------------------------------------------
# - Landscape: center-crop to cover (no bars).
# - Portrait: triptych A | B | (A mirrored).
# - Robust ffprobe (rotation-aware).
# - Independent volumes: --bg-volume (music), --clip-volume (clips).
# - Optional beat-based durations: random EVEN beats between --min-beats and --max-beats,
#   drawn from a triangular distribution with adjustable mode via --beat-mode (default 0.25 = lower bias).
# - Reverb on clip audio via FFmpeg 'areverb' (natural room), with automatic fallback to tuned 'aecho'.
# - Automatic loudness normalization to ~ -20 LUFS (+ true-peak limit), optional noise reduction for very quiet sources.
# - Automatic export of stems:
#     <output_stem>.clips.norm.wav (normalized/denoised)
#     <output_stem>.clips.norm_reverb.wav (normalized/denoised + reverb, only if reverb > 0)
# - During rendering, save a small preview image every 5 seconds to a constant filename:
#     <output_stem>.preview.jpg  (overwritten repeatedly)
#   This preview file is deleted after the video is finished.
#
# CLI:
#   --audio PATH            background track (required)
#   --videos DIR            folder(s) with videos/gifs (required)
#   --output PATH           output video file (required)
#   --width INT             resize output (default: keep source sizes)
#   --height INT            resize output (default: keep source sizes)
#   --fps FLOAT             output fps (default 30)
#   --bg-volume FLOAT       volume multiplier for background audio (default 1.0)
#   --clip-volume FLOAT     volume multiplier for clip audio (default 0.8)
#   --clip-reverb FLOAT     multiplier for clip reverb (default 0.2)
#   --bpm-detect            use librosa to detect BPM of --audio
#   --min-beats INT         min beats per segment (default 2)
#   --max-beats INT         max beats per segment (default 8)
#   --beat-mode FLOAT       multiplier for beat length (default 1.0, e.g. 0.5=8th, 0.25=16th)
#   --min-seconds FLOAT     min seconds per segment (default 2.0)
#   --max-seconds FLOAT     max seconds per segment (default 5.0)
#   --codec STR             ffmpeg vcodec (default libx264)
#   --audio-codec STR       ffmpeg acodec (default aac)
#   --preset STR            ffmpeg preset (default medium)
#   --bitrate STR           video bitrate (e.g. 8M) (optional)
#   --threads INT           ffmpeg threads (optional)
#
# Usage example:
#     python pmveaver.py --audio music.mp3 --videos ./vids --output out.mp4 \
#         --bpm 120 --min-beats 2 --max-beats 8 --beat-mode 0.25 \
#         --width 1920 --height 1080 --bg-volume 1.0 --clip-volume 0.8 --clip-reverb 0.25

import argparse
import json
import random
import subprocess
import sys
import shutil
import tempfile
import atexit, signal
import math
import threading
import numpy as np, time, os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set


from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    ImageClip,
    concatenate_videoclips,
    CompositeVideoClip,
    CompositeAudioClip,
)
from moviepy.video.fx import all as vfx
from tqdm import tqdm
from PIL import Image as _PIL_Image
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mpg", ".gif"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_EXTS = VIDEO_EXTS | IMAGE_EXTS

PREVIEW_INTERVAL = 5.0

# Global random
_rng = random.Random()

# ==== Robustness shims =======================================================

# Pillow ANTIALIAS shim
try:
    _Resampling = getattr(_PIL_Image, "Resampling", None)
    if _Resampling and "ANTIALIAS" not in _PIL_Image.__dict__:
        _PIL_Image.ANTIALIAS = _Resampling.LANCZOS
        _PIL_Image.BICUBIC = _Resampling.BICUBIC
        _PIL_Image.BILINEAR = _Resampling.BILINEAR
except Exception:
    pass

# SciPy signal.hann shim (librosa sometimes expects it)
try:
    import scipy.signal as _scisig
    if not hasattr(_scisig, "hann"):
        try:
            from scipy.signal import windows as _scisig_windows
            _scisig.hann = _scisig_windows.hann  # type: ignore
        except Exception:
            try:
                import numpy as _np
                _scisig.hann = _np.hanning  # type: ignore
            except Exception:
                pass
except Exception:
    pass


# ---------------- ffprobe helpers ----------------

@dataclass(frozen=True)
class ProbeInfo:
    width: int
    height: int
    rotation: int
    duration: float

    @property
    def is_portrait(self) -> bool:
        # rotation-bereinigte Orientierung ist bereits in width/height enthalten
        return self.height > self.width

def ffprobe_video_info(path: Path) -> ProbeInfo:
    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError("ffprobe not found. Please install FFmpeg and ensure ffprobe is in PATH.")
    abs_path = str(path.resolve())
    # Wir fragen selektiv ab, um Parsing-Overhead zu sparen:
    # - width,height,rotation (side_data_list/tags.rotate), duration (aus stream oder format)
    args = [
        exe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,side_data_list:stream_tags=rotate:format=duration",
        "-of", "json",
        abs_path
    ]
    res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    data = json.loads(res.stdout.decode("utf-8", errors="replace"))

    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream in {path}")
    s = streams[0]

    width  = int(s.get("width", 0) or 0)
    height = int(s.get("height", 0) or 0)

    rotation = 0
    # side_data_list rotation
    sdl = s.get("side_data_list") or []
    for sd in sdl:
        if sd.get("rotation") is not None:
            try: rotation = int(sd["rotation"])
            except Exception: rotation = 0
            break
    # tag rotate fallback
    if rotation == 0:
        tags = s.get("tags") or {}
        rot_tag = tags.get("rotate") or tags.get("ROTATE")
        if rot_tag is not None:
            try: rotation = int(rot_tag)
            except Exception: rotation = 0
    rotation %= 360
    if rotation in (90, 270):
        width, height = height, width

    # Dauer (Format-Dauer ist zuverlässiger bei manchen Containern)
    dur = 0.0
    try:
        dur_str = (data.get("format") or {}).get("duration")
        if dur_str: dur = float(dur_str)
    except Exception:
        pass
    # Fallback: aus Stream (falls vorhanden)
    if dur <= 0:
        try:
            dur = float(s.get("duration") or 0.0)
        except Exception:
            dur = 0.0

    return ProbeInfo(width=width, height=height, rotation=rotation, duration=dur)

def probe_image_info(path: Path) -> ProbeInfo:
    with _PIL_Image.open(path) as img:
        w, h = img.size
    return ProbeInfo(width=w, height=h, rotation=0, duration=10.0)

def build_probe_cache(files: List[Path], max_workers: Optional[int] = None) -> Dict[Path, ProbeInfo]:
    cache: Dict[Path, ProbeInfo] = {}
    if not files:
        return cache

    workers = max_workers or max(1, (min(16, os.cpu_count()) or 4))

    def _probe(path: Path) -> ProbeInfo:
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            return probe_image_info(path)
        return ffprobe_video_info(path)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(_probe, p): p for p in files}
        for fut in as_completed(future_map):
            p = future_map[fut]
            try:
                cache[p] = fut.result()
            except Exception as e:
                # Nicht hart abbrechen – Datei einfach überspringen
                print(f"Warning: ffprobe failed for {p}: {e}", file=sys.stderr)
    return cache


# ---------------- core helpers ----------------

def find_video_files(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]


def pick_segment_bounds_random_seconds(duration: float, min_s: float, max_s: float) -> Tuple[float, float]:
    # Fenster bestimmen: <120s → [0,1], sonst [0.10, 0.95]
    lo_frac, hi_frac = (0.0, 1.0) if duration < 120.0 else (0.10, 0.95)

    if duration <= 0:
        return 0.0, 0.0

    seg_len = random.uniform(min_s, max_s)
    # Versuchen, komplett im Fenster zu bleiben
    s, e = _choose_start_in_window(duration, seg_len, lo_frac, hi_frac)
    if e > s:
        return s, e

    # Fallback: nimm das ganze erlaubte Fenster (falls min_s..max_s größer als Fenster)
    lo = duration * lo_frac
    hi = duration * hi_frac
    window_len = max(0.0, hi - lo)
    if window_len <= 0:
        return 0.0, 0.0
    return lo, lo + window_len

def pick_segment_bounds_fixed(duration: float, seg_len: float) -> Tuple[float, float]:
    lo_frac, hi_frac = (0.0, 1.0) if duration < 120.0 else (0.10, 0.95)
    if duration <= 0 or seg_len <= 0:
        return 0.0, 0.0

    s, e = _choose_start_in_window(duration, seg_len, lo_frac, hi_frac)
    if e > s:
        return s, e

    # Kein Platz im Fenster → wenn Segment >= Gesamtdauer, gib ganzen Clip zurück, sonst verwerfen
    if seg_len >= duration:
        return 0.0, duration
    return 0.0, 0.0

def _choose_start_in_window(duration: float, seg_len: float, lo_frac: float, hi_frac: float) -> Tuple[float, float]:
    """
    Wählt einen Start gleichverteilt innerhalb [lo..hi - seg_len], sodass das Segment komplett im Fenster liegt.
    Gibt (0.0, 0.0) zurück, wenn kein gültiger Bereich existiert.
    """
    if duration <= 0 or seg_len <= 0:
        return 0.0, 0.0

    lo = max(0.0, duration * lo_frac)
    hi = min(duration, duration * hi_frac)
    window_len = max(0.0, hi - lo)

    # Segment passt nicht vollständig ins Fenster
    if seg_len > window_len + 1e-9:
        return 0.0, 0.0

    # Grenzfall: exakt passend → Start = lo
    if window_len - seg_len <= 1e-9:
        start = lo
        return start, start + seg_len

    start = random.uniform(lo, hi - seg_len)
    return start, start + seg_len



def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def cover_scale_and_crop(clip: VideoFileClip, target_w: int, target_h: int) -> VideoFileClip:
    src_w, src_h = clip.size
    if src_w == 0 or src_h == 0:
        return clip

    # Early exit: size already fitting
    if src_w == target_w and src_h == target_h:
        return clip

    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    resized = vfx.resize(clip, newsize=(new_w, new_h))

    # Precise, symmetrical center crops with clamps
    # (avoids 1px drift with odd differences)
    dx = max(0, new_w - target_w)
    dy = max(0, new_h - target_h)
    x1 = int(round(dx / 2))
    y1 = int(round(dy / 2))
    x2 = x1 + target_w
    y2 = y1 + target_h

    # Safeguards (numerical edges)
    x1 = max(0, min(x1, max(0, new_w - target_w)))
    y1 = max(0, min(y1, max(0, new_h - target_h)))
    x2 = min(new_w, max(target_w, x2))
    y2 = min(new_h, max(target_h, y2))

    return vfx.crop(resized, x1=x1, y1=y1, x2=x2, y2=y2)


def make_triptych(clipA: VideoFileClip, clipB: VideoFileClip, target_w: int, target_h: int) -> CompositeVideoClip:
    panel_w = math.ceil(target_w / 3)

    B_mid = cover_scale_and_crop(clipB, panel_w, target_h).set_position((panel_w, 0))

    A_scaled = cover_scale_and_crop(clipA, panel_w, target_h)
    A_mirrored = A_scaled.fx(vfx.mirror_x)

    if random.choice([True, False]):
        A_left, A_right = A_scaled, A_mirrored
    else:
        A_left, A_right = A_mirrored, A_scaled

    A_left = A_left.set_position((0, 0))
    A_right = A_right.without_audio().set_position((2 * panel_w, 0))

    return CompositeVideoClip([A_left, B_mid, A_right], size=(target_w, target_h))


# ---------------- BPM helpers ----------------

def detect_bpm_with_librosa(audio_path: Path) -> float:
    import librosa

    y, sr = librosa.load(str(audio_path), sr=None, mono=True, duration=90.0)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if tempo is None or tempo <= 0:
        raise RuntimeError("Could not detect BPM from audio.")
    return round(float(tempo))


def _even_bounds(lo: int, hi: int) -> Tuple[int, int, List[int]]:
    lo_i, hi_i = sorted((int(lo), int(hi)))
    evens = [b for b in range(lo_i, hi_i + 1) if b % 2 == 0 and b > 0]
    if not evens:
        if lo_i <= 2 <= hi_i:
            evens = [2]
        else:
            evens = [max(2, lo_i + (lo_i % 2))]
    return lo_i, hi_i, evens


def choose_even_beats(min_beats: int, max_beats: int, mode_pos: float = 0.25) -> int:
    """
    Draw an even beat count from [min_beats, max_beats] using a triangular weighting.
    mode_pos in [0..1]: 0 = peak at min, 1 = peak at max. No weight ever becomes zero.
    """
    lo, hi, evens = _even_bounds(min_beats, max_beats)
    if not evens:
        return 2

    mode_pos = clamp(mode_pos, 0.0, 1.0)
    mode = lo + mode_pos * (hi - lo)

    eps = 1e-9
    span = max(hi - lo, eps)
    MIN_WEIGHT = 0.1  # floor so nothing ever drops to zero

    weights: List[float] = []

    if mode <= lo + eps:
        # Peak exactly at left edge
        for b in evens:
            w = 1.0 - (b - lo) / span
            weights.append(max(w, MIN_WEIGHT))
    elif mode >= hi - eps:
        # Peak exactly at right edge
        for b in evens:
            w = 1.0 - (hi - b) / span
            weights.append(max(w, MIN_WEIGHT))
    else:
        # Peak inside: symmetric triangle around `mode`
        left  = max(mode - lo, eps)
        right = max(hi - mode, eps)
        denom = max(left, right)
        for b in evens:
            w = 1.0 - abs(b - mode) / denom
            weights.append(max(w, MIN_WEIGHT))

    return random.choices(evens, weights=weights, k=1)[0]

def estimate_beat_offset_with_librosa(audio_path: Path, bpm_hint: Optional[float] = None) -> Tuple[float, float]:
    """
    Liefert (offset_in_s, tempo_from_librosa).
    offset_in_s = Zeit bis zum ersten erkannten Beat ab t=0.
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=None, mono=True, duration=60.0)
    oenv = librosa.onset.onset_strength(y=y, sr=sr)

    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, onset_envelope=oenv, start_bpm=(bpm_hint or 120.0)
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if beat_times.size == 0:
        # Fallback: kein Beat gefunden
        return 0.0, float(tempo) if tempo else float(bpm_hint or 0.0)
    return float(beat_times[0]), float(tempo if tempo else (bpm_hint or 0.0))


# ---------------- Audio helpers: normalize / denoise / reverb / stems ----------------

def _ffmpeg_path_or_raise() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found. Install FFmpeg and ensure ffmpeg is in your PATH.")
    return exe


def analyze_mean_volume_w_dbfs(wav_path: Path) -> Optional[float]:
    """
    Use FFmpeg 'volumedetect' to estimate mean_volume in dBFS.
    Returns e.g. -28.3 or None on failure.
    """
    exe = _ffmpeg_path_or_raise()
    cmd = [exe, "-hide_banner", "-nostats", "-i", str(wav_path), "-af", "volumedetect", "-f", "null", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    text = (proc.stderr or proc.stdout).decode("utf-8", errors="replace")
    mv = None
    for line in text.splitlines():
        line = line.strip().lower()
        if "mean_volume:" in line:
            try:
                mv = float(line.split("mean_volume:")[1].split(" db")[0].strip())
                break
            except Exception:
                pass
    return mv


def normalize_and_maybe_denoise_audio_clip(
    audio_clip: AudioFileClip,
    target_I: float = -20.0,
    target_LRA: float = 7.0,
    target_TP: float = -1.5,
    quiet_threshold_dbfs: float = -32.0
) -> AudioFileClip:
    """
    1) Normalize to target loudness via 'loudnorm' (single pass).
    2) If original clip is very quiet (mean_volume < quiet_threshold_dbfs), apply gentle 'afftdn' denoise.
    Returns a new AudioFileClip; temp dir path stored on attribute _temp_norm_dir for later cleanup.
    """
    exe = _ffmpeg_path_or_raise()
    tmpdir = Path(tempfile.mkdtemp(prefix="montage_norm_"))
    raw_wav  = tmpdir / "clip_raw.wav"
    norm_wav = tmpdir / "clip_norm.wav"
    den_wav  = tmpdir / "clip_norm_denoise.wav"

    # Export AudioClip -> WAV
    audio_clip.write_audiofile(str(raw_wav), fps=44100, nbytes=2, codec="pcm_s16le", verbose=False, logger=None)

    # Pre-analysis: mean volume
    mean_dbfs = analyze_mean_volume_w_dbfs(raw_wav)

    # Normalize
    loudnorm = f"loudnorm=I={target_I}:LRA={target_LRA}:TP={target_TP}:dual_mono=true"
    cmd_norm = [exe, "-y", "-i", str(raw_wav), "-af", loudnorm, "-c:a", "pcm_s16le", str(norm_wav)]
    r1 = subprocess.run(cmd_norm, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if r1.returncode != 0 or not norm_wav.exists():
        # Fallback: return original clip if normalization failed
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
        return audio_clip

    # Optional denoise if original was very quiet
    need_denoise = (mean_dbfs is not None) and (mean_dbfs < quiet_threshold_dbfs)
    if need_denoise:
        afftdn = "afftdn=nf=-25"
        cmd_den = [exe, "-y", "-i", str(norm_wav), "-af", afftdn, "-c:a", "pcm_s16le", str(den_wav)]
        r2 = subprocess.run(cmd_den, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        final_path = den_wav if (r2.returncode == 0 and den_wav.exists()) else norm_wav
    else:
        final_path = norm_wav

    processed = AudioFileClip(str(final_path))
    processed._temp_norm_dir = str(tmpdir)  # type: ignore[attr-defined]
    return processed


def apply_reverb_to_audio_clip(audio_clip: AudioFileClip, strength: float, samplerate: int = 44100) -> AudioFileClip:
    """
    Apply a natural-sounding reverb using FFmpeg 'areverb'.
    If 'areverb' is unavailable in the FFmpeg build, fall back to a tuned 'aecho' chain.
    strength: 0.0 (off) .. 1.0 (strong)
    """
    strength = clamp(strength, 0.0, 1.0)
    if strength <= 1e-6:
        return audio_clip  # no-op

    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg not found. Install FFmpeg and ensure ffmpeg is in your PATH.")

    tmpdir = tempfile.mkdtemp(prefix="montage_reverb_")
    raw_path = Path(tmpdir) / "clip_raw.wav"
    wet_path = Path(tmpdir) / "clip_reverb.wav"

    # Export original clip audio as wav
    audio_clip.write_audiofile(str(raw_path), fps=samplerate, nbytes=2, codec="pcm_s16le", verbose=False, logger=None)

    # ---- Primary: areverb ----
    room = 20.0 + 70.0 * strength       # 20..90
    revb = 20.0 + 70.0 * strength       # 20..90
    areverb_expr = f"areverb={room:.1f}:{revb:.1f}"

    cmd = [ffmpeg_exe, "-y", "-i", str(raw_path), "-af", areverb_expr, "-c:a", "pcm_s16le", str(wet_path)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        # ---- Fallback: tuned aecho ----
        delays = "180|360|540|720"
        base = 0.28 * strength
        decays = f"{min(base,0.45):.3f}|{min(base*0.85,0.40):.3f}|{min(base*0.7,0.35):.3f}|{min(base*0.55,0.30):.3f}"
        aecho_expr = f"aecho=0.7:0.8:{delays}:{decays}"
        cmd_fb = [ffmpeg_exe, "-y", "-i", str(raw_path), "-af", aecho_expr, "-c:a", "pcm_s16le", str(wet_path)]
        fb = subprocess.run(cmd_fb, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if fb.returncode != 0:
            # give up → return dry
            shutil.rmtree(tmpdir, ignore_errors=True)
            return audio_clip

    wet_audio = AudioFileClip(str(wet_path))
    wet_audio._temp_reverb_dir = tmpdir  # type: ignore[attr-defined]
    return wet_audio


def _derive_clip_audio_paths(out_path: Path) -> Tuple[Path, Path]:
    parent = out_path.parent
    stem = out_path.stem
    norm = parent / f"{stem}.clips.norm.wav"
    wet  = parent / f"{stem}.clips.norm_reverb.wav"
    return norm, wet

class _ClipCache:
    def __init__(self, probe_cache: Dict[Path, ProbeInfo]):
        self._cache: Dict[Path, VideoFileClip] = {}
        self._probes = probe_cache

    def get(self, path: Path) -> VideoFileClip:
        clip = self._cache.get(path)
        if clip is None:
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                dur = max(0.1, (self._probes.get(path) or ProbeInfo(0, 0, 0, 3.0)).duration or 3.0)
                clip = ImageClip(str(path)).set_duration(dur)
            else:
                clip = VideoFileClip(str(path))
            self._cache[path] = clip
        return clip

    def close_all(self):
        for clip in self._cache.values():
            try:
                clip.close()
            except Exception:
                pass
        self._cache.clear()

# --- 1-beat run state (aktiv, wenn min_beats == 1) --------------------------

# Anzahl noch ausstehender 1-Beat-Segmente in der aktuellen Folge.
# 0 = kein Lauf aktiv.
_one_beat_run_remaining = 0

def _maybe_start_one_beat_run() -> int:
    """
    Starte mit kleiner Wahrscheinlichkeit einen 1-Beat-Lauf (2/4/6 Segmente).
    Liefert die Anzahl der noch *zusätzlich* zu produzierenden 1er (also Restlauf).
    """
    # Du kannst die Startwahrscheinlichkeit gerne anpassen:
    if random.random() < 0.0:  # 5% Chance, einen 1-Beat-Lauf zu beginnen
        run_len = random.choice([4, 8, 16])
        # Wir geben "Rest" zurück, denn den *ersten* 1er liefern wir direkt
        return run_len - 1
    return 0

def compute_segment_bounds(src_dur: float,
                           bpm: Optional[float],
                           min_beats: int, max_beats: int, beat_mode: float,
                           min_s: float, max_s: float, fps: float, trim_large_clips: bool) -> tuple[float, float]:
    window_lo_frac, window_hi_frac = (0.0, 1.0) if src_dur < 120.0 or not trim_large_clips else (0.10, 0.95)
    window_len = max(0.0, (window_hi_frac - window_lo_frac) * src_dur)

    if bpm and bpm > 0:
        beat_len = 60.0 / bpm

        global _one_beat_run_remaining
        use_beats = None

        if min_beats == 1:
            # 1) Falls ein 1-Beat-Lauf aktiv ist, als nächstes wieder 1 Beat
            if _one_beat_run_remaining > 0:
                use_beats = 1
            else:
                # 2) Sonst normal „gerade Beats“ ziehen
                #    (min=2, damit 1 nicht zufällig auftaucht)
                even_beats = choose_even_beats(2, max_beats, mode_pos=beat_mode)

                # 3) Optional einen 1-Beat-Lauf starten (setzt die nächsten N-1 Segmente auf 1)
                maybe_rest = _maybe_start_one_beat_run()
                if maybe_rest > 0:
                    _one_beat_run_remaining = maybe_rest
                    use_beats = 1  # erster 1er sofort
                else:
                    use_beats = even_beats
        else:
            # klassisch: nur gerade Beats
            use_beats = choose_even_beats(min_beats, max_beats, mode_pos=beat_mode)

        # Länge auf Frames runden
        seg_len = round((beat_len * use_beats) * fps) / fps

        # Passt das in den Quell-Clip?
        if src_dur + 1e-6 >= seg_len:
            # Commit: Wenn wir tatsächlich einen 1er verbrauchen, Run herunterzählen
            if min_beats == 1 and use_beats == 1 and _one_beat_run_remaining > 0:
                _one_beat_run_remaining -= 1
            s, e = _choose_start_in_window(src_dur, seg_len, window_lo_frac, window_hi_frac)
            if e > s:
                return s, e

        # --- Quellclip zu kurz: Fallback-Regeln -------------------------------
        max_beats_fit = int(math.floor((window_len + 1e-9) / beat_len))

        if min_beats == 1:
            # a) Wenn mindestens 1 Beat passt und wir *gerade* einen 1er liefern sollen,
            #    nehmen wir den 1er trotzdem (und zählen Run herunter).
            if use_beats == 1 and max_beats_fit >= 1:
                snapped_len = round((beat_len * 1) * fps) / fps
                if _one_beat_run_remaining > 0:
                    _one_beat_run_remaining -= 1
                s, e = _choose_start_in_window(src_dur, snapped_len, window_lo_frac, window_hi_frac)
                return (s, e) if e > s else (0.0, 0.0)

            # b) Sonst versuchen wir den größten *geraden* Fit (2,4,6,…)
            even_fit = max_beats_fit - (max_beats_fit % 2)
            if even_fit >= 2:
                snapped_len = round((beat_len * even_fit) * fps) / fps
                s, e = _choose_start_in_window(src_dur, snapped_len, window_lo_frac, window_hi_frac)
                return (s, e) if e > s else (0.0, 0.0)

            # c) Weniger als 2 Beats passen insgesamt – wenn wenigstens 1 Beat passt,
            #    und wir *in einem Lauf* sind, geben wir 1 Beat zurück (um den Lauf nicht zu „brechen“).
            if max_beats_fit >= 1 and _one_beat_run_remaining > 0:
                snapped_len = round((beat_len * 1) * fps) / fps
                _one_beat_run_remaining -= 1
                s, e = _choose_start_in_window(src_dur, snapped_len, window_lo_frac, window_hi_frac)
                return (s, e) if e > s else (0.0, 0.0)

            # d) Sonst verwerfen
            return (0.0, 0.0)

        else:
            # Klassischer Fallback: größtes gerades Multiple (>=2), sonst verwerfen
            even_fit = max_beats_fit - (max_beats_fit % 2)  # 7->6, 5->4, 3->2, 2->2, 1->0
            if even_fit >= 2:
                snapped_len = round((beat_len * even_fit) * fps) / fps
                return pick_segment_bounds_fixed(src_dur, snapped_len)

            return (0.0, 0.0)

    else:
        # Zeit-basierter Modus (unverändert)
        return pick_segment_bounds_random_seconds(src_dur, min_s, max_s)

# ---------------- main build ----------------

def build_montage(
    audio_path: Path,
    videos_spec: str,
    out_path: Path,
    min_seconds: float,
    max_seconds: float,
    fps: float,
    width: Optional[int],
    height: Optional[int],
    codec: str,
    audio_codec: str,
    preset: str,
    bitrate: Optional[str],
    threads: Optional[int],
    bg_volume: float,
    clip_volume: float,
    clip_reverb: float,
    bpm: Optional[float],
    bpm_detect: bool,
    min_beats: int,
    max_beats: int,
    beat_mode: float,
    preview: bool,
    triptych_carry: float,
    trim_large_clips: bool,
    pulse_effect: bool,
    fade_out_seconds: float,
):
    setup_tempfile_cleanup(out_path)

    bg_volume = max(0.0, bg_volume)
    clip_volume = max(0.0, clip_volume)
    clip_reverb = clamp(clip_reverb, 0.0, 1.0)
    beat_mode = clamp(beat_mode, 0.0, 1.0)

    bg_audio = AudioFileClip(str(audio_path))
    target_duration = bg_audio.duration

    # BPM logic
    effective_bpm: Optional[float] = None
    if bpm is not None:
        effective_bpm = float(bpm)
        print(f"PMVeaver - Using BPM: {effective_bpm:.2f}")
    elif bpm_detect:
        effective_bpm = detect_bpm_with_librosa(audio_path)
        print(f"PMVeaver - Detected BPM: {effective_bpm:.2f}")

    # Beat-Offset ermitteln und Ziel-Dauer anpassen
    beat_offset = 0.0
    if effective_bpm:
        try:
            off, tempo_from_offset = estimate_beat_offset_with_librosa(audio_path, effective_bpm)
            # Wenn wir auto-detectet haben, übernehmen wir zur Konsistenz das gleiche Tempo
            if bpm_detect and tempo_from_offset > 0:
                effective_bpm = tempo_from_offset
            beat_offset = max(0.0, float(off))
            print(f"PMVeaver - Beat offset: {beat_offset:.3f}s")
        except Exception as e:
            print(f"PMVeaver - Beat offset detection failed: {e}")

    # WICHTIG: Ziel-Dauer nach Offset neu setzen (sonst sammeln wir zu lange)
    if effective_bpm and beat_offset > 0.0:
        target_duration = max(0.0, bg_audio.duration - beat_offset)
    else:
        target_duration = bg_audio.duration

    specs = _parse_videos_spec(videos_spec)

    unique_files: Set[Path] = set()
    for folder, _w in specs:
        unique_files.update(find_video_files(folder))
    if not unique_files:
        raise RuntimeError("No video files found")

    print(f"PMVeaver - Found {len(list(unique_files))} video file(s) total")

    probe_cache = build_probe_cache(list(unique_files))
    clip_cache = _ClipCache(probe_cache)

    next_portrait = _make_epoch_picker(specs, probe_cache, "portrait")
    next_landscape = _make_epoch_picker(specs, probe_cache, "landscape")

    portrait_unique_count = sum(
        1 for p, info in probe_cache.items()
        if info and info.is_portrait and info.width > 0 and info.height > 0 and info.duration > 0
    )

    print(f"PMVeaver - Landscape clips: {len(list(unique_files)) - portrait_unique_count}")
    print(f"PMVeaver - Portrait clips:  {portrait_unique_count}")

    segments: List[VideoFileClip] = []
    total = 0.0
    carry_portrait_path: Optional[Path] = None  # last triptych's side (A) becomes next center (B)

    pbar = tqdm(total=target_duration, desc="PMVeaver - Collecting clips", unit="s")
    default_w = width or 1920
    default_h = height or 1080

    while total < target_duration + 0.01:
        try:
            choice = _rng.choice(["landscape", "portrait"])

            if choice == "landscape":
                src_path = next_landscape()
                if src_path is None:
                    continue

                info = probe_cache.get(src_path)
                if not info or info.width <= 0 or info.height <= 0 or info.duration <= 0:
                    continue
                try:
                    src = clip_cache.get(src_path)
                except Exception as e:
                    print(f"Warning: failed to open {src_path}: {e}", file=sys.stderr)
                    continue

                start, end = compute_segment_bounds(
                    src.duration, effective_bpm, min_beats, max_beats, beat_mode, min_seconds, max_seconds, fps, trim_large_clips
                )
                if end <= start:
                    continue
                sub = src.subclip(start, end)

                if isinstance(sub, ImageClip):
                    filled = kenburns_cover(sub, default_w, default_h, dur=sub.duration, rng=_rng)
                else:
                    filled = cover_scale_and_crop(sub, default_w, default_h)

                segments.append(filled)
                total += filled.duration

            else:  # choice == "portrait"
                # Reuse previous side as current center with given probability
                if carry_portrait_path is not None and random.random() < triptych_carry:
                    pathA = next_portrait()
                    if pathA is None:
                        continue

                    # avoid A == B (otherwise all three panels would be the same source)
                    if pathA == carry_portrait_path:
                        alt = next_portrait()
                        if alt is not None:
                            pathA = alt
                    pathB = carry_portrait_path

                else:
                    pathA = next_portrait()
                    if pathA is None:
                        continue
                    pathB = next_portrait()
                    if pathB is None:
                        continue

                infoA = probe_cache.get(pathA)
                infoB = probe_cache.get(pathB)
                if not infoA or not infoB:
                    continue
                if (infoA.width <= 0 or infoA.height <= 0 or infoA.duration <= 0 or
                        infoB.width <= 0 or infoB.height <= 0 or infoB.duration <= 0):
                    continue

                try:
                    srcA = clip_cache.get(pathA)
                    srcB = clip_cache.get(pathB)
                except Exception as e:
                    print(f"Warning: failed to open portrait clip(s) {pathA} / {pathB}: {e}", file=sys.stderr)
                    continue

                sA, eA = compute_segment_bounds(
                    srcA.duration, effective_bpm, min_beats, max_beats, beat_mode, min_seconds, max_seconds, fps, trim_large_clips
                )
                sB, eB = compute_segment_bounds(
                    srcB.duration, effective_bpm, min_beats, max_beats, beat_mode, min_seconds, max_seconds, fps, trim_large_clips
                )
                if eA <= sA or eB <= sB:
                    continue

                subA = srcA.subclip(sA, eA)
                subB = srcB.subclip(sB, eB)
                dur = min(subA.duration, subB.duration)
                if dur <= 0:
                    subA.close();
                    subB.close()
                    continue

                subA = subA.subclip(0, dur)
                subB = subB.subclip(0, dur)

                if isinstance(subA, ImageClip):
                    subA = kenburns_cover(subA, default_w // 3, default_h, dur=subA.duration, rng=_rng)
                if isinstance(subB, ImageClip):
                    subB = kenburns_cover(subB, default_w // 3, default_h, dur=subB.duration, rng=_rng)

                trip = make_triptych(subA, subB, default_w, default_h)

                segments.append(trip)
                total += trip.duration

                # prepare carry-over for next triptych
                if triptych_carry:
                    carry_portrait_path = pathA

            if segments:
                pbar.update(segments[-1].duration)

        except Exception as e:
            print(f"Warning: building segment failed: {e}", file=sys.stderr)
            continue

    pbar.close()

    # Fit to exact target length
    over = total - target_duration
    if over > 1e-3 and segments:
        last = segments[-1]
        keep = max(0.0, last.duration - over)
        segments[-1] = last.subclip(0, keep)

    # Concatenate (no crossfade)
    montage = concatenate_videoclips(segments, method="chain")
    montage = montage.subclip(0, target_duration).set_fps(fps)

    # --- Beat-Pulse (Zoom + Blur) ----------------------------------------------
    PULSE_ZOOM_MAX = 0.06  # bis zu +6% Zoom am Beat
    PULSE_BLUR_MAX = 4.2  # bis zu 1.8 px GaussianBlur am Beat
    PULSE_DECAY = 6.0  # exponentieller Abfall innerhalb eines Beats

    if pulse_effect and effective_bpm and effective_bpm > 0:
        import numpy as _np
        from PIL import Image, ImageFilter
        import librosa

        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        # RMS Energie pro Frame (z. B. 2048 Samples Fenster)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

        # Normalisieren 0..1
        rms /= rms.max()

        def volume_at(t: float) -> float:
            tt = t + (beat_offset or 0.0)
            idx = np.searchsorted(times, tt)
            if idx <= 0: return float(rms[0])
            if idx >= len(rms): return float(rms[-1])
            return float(rms[idx])

        beat_len = 60.0 / effective_bpm

        def _pulse_intensity(t: float) -> float:
            ph = (max(0.0, t) % beat_len) / beat_len
            return math.exp(-PULSE_DECAY * ph)

        def _pulse_frame(get_frame, t: float):
            frame = get_frame(t)  # np.uint8 (H,W,3) oder (H,W,4) / Maske wäre (H,W)

            beat_factor = _pulse_intensity(t)  # 0..1 aus BPM-Phase
            vol_factor = volume_at(t)  # 0..1 aus Audio
            inten = beat_factor * (0.2 + vol_factor * 2)

            if inten <= 1e-6:
                return frame

            # Maße des *aktuellen* Frames verwenden
            h, w = frame.shape[:2]
            scale = 1.0 + PULSE_ZOOM_MAX * inten
            crop_w = max(1, int(round(w / scale)))
            crop_h = max(1, int(round(h / scale)))

            # zentrierte Crop-Box, strikt einklemmen
            x1 = max(0, (w - crop_w) // 2)
            y1 = max(0, (h - crop_h) // 2)
            x2 = min(w, x1 + crop_w)
            y2 = min(h, y1 + crop_h)
            if x2 <= x1 or y2 <= y1:
                return frame  # Sicherheitsnetz

            # Slicing + contiguous machen (PIL liebt contiguous)
            cropped = _np.ascontiguousarray(frame[y1:y2, x1:x2])

            # Resize zurück auf die Original-Framegröße
            img = Image.fromarray(cropped)
            img = img.resize((w, h), resample=Image.BICUBIC)

            # leichter Blur am Beat
            blur_radius = PULSE_BLUR_MAX * inten
            if blur_radius > 0.05:
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            return _np.array(img)

        montage = montage.fl(_pulse_frame)

    # ----- Audio: normalize (+ optional denoise) -> (optional reverb) -> export stems -> mix with bg -----
    clip_audio = montage.audio
    temp_dirs_to_cleanup: List[str] = []

    if clip_audio is not None and (clip_volume <= 1e-6 and clip_reverb <= 1e-6):
        clip_audio = None  # komplette Audio-Verarbeitung auslassen

    if clip_audio:
        # 1) Normalize + ggf. Denoise
        norm = normalize_and_maybe_denoise_audio_clip(clip_audio)
        tmpdir_norm = getattr(norm, "_temp_norm_dir", None)
        if tmpdir_norm:
            temp_dirs_to_cleanup.append(tmpdir_norm)

        # 2) Export "dry" stem
        norm_out, wet_out = _derive_clip_audio_paths(out_path)
        try:
            norm.set_duration(target_duration).write_audiofile(
                str(norm_out), fps=44100, nbytes=2, codec="pcm_s16le",
                verbose=False, logger=None
            )
        except Exception as e:
            print(f"Warning: failed to write normalized clip audio: {e}", file=sys.stderr)

        clip_audio = norm

        # 3) Optional Reverb
        if clip_reverb > 0:
            wet = apply_reverb_to_audio_clip(clip_audio, clip_reverb)
            tmpdir_rev = getattr(wet, "_temp_reverb_dir", None)
            if tmpdir_rev:
                temp_dirs_to_cleanup.append(tmpdir_rev)

            # 3a) Export "wet" stem
            try:
                wet.set_duration(target_duration).write_audiofile(
                    str(wet_out), fps=44100, nbytes=2, codec="pcm_s16le",
                    verbose=False, logger=None
                )
            except Exception as e:
                print(f"Warning: failed to write reverb clip audio: {e}", file=sys.stderr)

            clip_audio = wet

    # scale volumes
    clip_audio = clip_audio.volumex(clip_volume) if clip_audio else None
    bg_track = bg_audio.subclip(beat_offset).set_duration(target_duration).volumex(bg_volume)

    if clip_audio is not None:
        composite_audio = CompositeAudioClip([bg_track, clip_audio.set_duration(target_duration)])
    else:
        composite_audio = bg_track

    montage = montage.set_audio(composite_audio)

    # --- Optionales Fade-Out am Ende ---
    fade_s = max(0.0, min(float(fade_out_seconds or 0.0), max(0.0, target_duration - 0.05)))
    if fade_s >= 0.05:
        try:
            montage = montage.fx(vfx.fadeout, fade_s)
        except Exception:
            pass
        try:
            montage = montage.audio_fadeout(fade_s)
        except Exception:
            pass

    # --- Live preview via frame pipeline: save every 5 seconds to a constant file
    preview_path = out_path.parent / f"{out_path.stem}.preview.jpg"

    last_ts = 0.0
    def _preview_writer(frame):
        nonlocal last_ts
        now = time.monotonic()
        if now - last_ts < PREVIEW_INTERVAL:
            return frame
        last_ts = now

        try:
            # Ensure we have a uint8 RGB array
            if isinstance(frame, np.ndarray):
                arr = frame
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8, copy=False)
                img = _PIL_Image.fromarray(arr)
            else:
                img = _PIL_Image.fromarray(np.asarray(frame, dtype=np.uint8))

            # Fit to bounding box 180x100 (no upscaling)
            max_w, max_h = 180, 100
            w, h = img.size
            scale = min(1.0, max_w / w, max_h / h)
            if scale < 1.0:
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                # BICUBIC ist schneller als LANCZOS und für Preview gut genug
                img = img.resize(new_size, resample=_PIL_Image.BICUBIC)

            # Write atomically: first .tmp, then os.replace
            tmp_path = str(preview_path) + ".tmp"
            img.save(
                tmp_path,
                format="JPEG",
                quality=85,
                optimize=True,
                progressive=True,
                subsampling=2
            )
            os.replace(tmp_path, str(preview_path))
        except Exception:
            # No hard error message in the render loop
            pass

        return frame

    if preview:
        montage_with_preview = montage.fl_image(_preview_writer)
    else:
        montage_with_preview = montage

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        montage_with_preview.write_videofile(
            str(out_path),
            codec=codec,
            audio_codec=audio_codec,
            fps=fps,
            preset=preset,
            bitrate=bitrate,
            threads=threads,
        )
    finally:
        # Remove preview after finishing
        try:
            if preview_path.exists():
                preview_path.unlink()
        except Exception:
            pass

    # Cleanup
    try:
        # Falls CompositeAudioClip verwendet wurde, schließen
        try:
            if montage.audio is not None and hasattr(montage.audio, "close"):
                montage.audio.close()
        except Exception:
            pass

        try:
            if montage_with_preview is not montage and hasattr(montage_with_preview, "close"):
                montage_with_preview.close()
        except Exception:
            pass

        try:
            montage.close()
        except Exception:
            pass

        # Hintergrund-Audio explizit schließen
        try:
            bg_audio.close()
        except Exception:
            pass

        # Eventuell erzeugtes Clip-Audio schließen (nass/normalisiert)
        try:
            if clip_audio is not None and hasattr(clip_audio, "close"):
                clip_audio.close()
        except Exception:
            pass
    except Exception:
        pass

    for seg in segments:
        try: seg.close()
        except Exception: pass

    clip_cache.close_all()

    try:
        for d in temp_dirs_to_cleanup:
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass

def setup_tempfile_cleanup(out_path):
    def _cleanup():
        try:
            # Delete preview file
            preview = out_path.parent / f"{out_path.stem}.preview.jpg"
            if preview.exists():
                preview.unlink()

            # Delete MoviePy temp files
            for f in out_path.parent.glob(f"{out_path.stem}TEMP_MPY_*"):
                try:
                    f.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    # on normal exit
    atexit.register(_cleanup)

    # in case of abort signals
    def _handler(signum, frame):
        _cleanup()
        sys.exit(1)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handler)

def start_stdin_exit_watcher(token: str = "__PMVEAVER_EXIT__"):
    def _runner():
        try:
            if sys.stdin is None or sys.stdin.closed:
                return
            for line in sys.stdin:
                if not line:
                    break
                if line.strip() == token:
                    # Instead of sys.exit(0): send a signal to the main process.
                    # This ensures that your signal/atexit handlers will work.
                    try:
                        # Windows: SIGBREAK exists; if not, use SIGINT
                        if hasattr(signal, "SIGBREAK"):
                            signal.raise_signal(signal.SIGBREAK)
                        else:
                            signal.raise_signal(signal.SIGINT)
                    except Exception:
                        # Fallbacks
                        try:
                            os.kill(os.getpid(), getattr(signal, "SIGINT", 2))
                        except Exception:
                            os.kill(os.getpid(), getattr(signal, "SIGTERM", 15))
                    return
        except Exception:
            # At least trigger a gentle termination
            try:
                os.kill(os.getpid(), getattr(signal, "SIGINT", 2))
            except Exception:
                pass
    threading.Thread(target=_runner, daemon=True).start()

def _parse_videos_spec(spec: str) -> List[Tuple[Path, int]]:
    """
    Akzeptiert entweder:
      - einen einzelnen Ordnerpfad (Rückgabe [(Path, 1)]), oder
      - eine Liste "dir[:w],dir[:w],..." wobei w eine ganze Zahl ≥ 1 ist.
    Windows-Pfade mit Laufwerksbuchstaben (z. B. "C:\...") werden korrekt behandelt,
    weil nur das *letzte* ':' als potentieller Trenner betrachtet wird.
    """
    # Einzelner Ordnerpfad?
    p0 = Path(spec).expanduser()
    if p0.is_dir():
        out_single = [(p0, 1)]
        print("CLI> Parsed --videos entries (single):")
        print(f"  {p0} (weight=1)")
        return out_single

    out: List[Tuple[Path, int]] = []
    for raw in (spec or "").split(","):
        part = raw.strip()
        if not part:
            continue

        # optionale Quotes entfernen, Whitespaces trimmen
        s = part.strip().strip('"').strip("'")

        # Standard: ganzes Segment ist Pfad, Gewicht = 1
        d, w = s, 1

        # Gewicht NUR erkennen, wenn der *rechte* Teil eine Ganzzahl ist
        if ":" in s:
            head, tail = s.rsplit(":", 1)
            t = tail.strip()
            if t.isdigit():
                d, w = head, int(t)
            else:
                d, w = s, 1  # kein gültiges Gewicht → kompletter String ist Pfad

        d_path = Path(d).expanduser()
        if not d_path.is_dir():
            raise ValueError(f"--videos: directory not found: {d_path}")
        if w <= 0:
            raise ValueError(f"--videos: weight must be >=1 in entry '{raw}'")

        out.append((d_path, w))

    if not out:
        raise ValueError("--videos: no valid directories")

    return out

def _build_weighted_pools(
    specs: List[Tuple[Path, int]],
    probe_cache: Dict[Path, "ProbeInfo"],  # ProbeInfo: width,height,duration,is_portrait
) -> Tuple[List[Path], List[Path]]:
    """
    Erzeugt zwei Pools (Portrait/Landscape), in denen die Dateien gemäß Ordnergewicht
    *dupliziert* sind. Danach werden die Pools gemischt.
    """
    portrait_pool: List[Path] = []
    landscape_pool: List[Path] = []

    for folder, w in specs:
        files = find_video_files(folder)  # vorhandene Helper: rekursiv + Extension-Filter
        for p in files:
            info = probe_cache.get(p)
            if not info or info.width <= 0 or info.height <= 0 or info.duration <= 0:
                continue
            target = portrait_pool if info.is_portrait else landscape_pool
            target.extend([p] * w)

    _rng.shuffle(portrait_pool)
    _rng.shuffle(landscape_pool)
    return portrait_pool, landscape_pool

def _make_epoch_picker(
    specs: List[Tuple[Path, int]],
    probe_cache: Dict[Path, "ProbeInfo"],
    orientation: str,  # "portrait" | "landscape"
):
    """
    Liefert eine Funktion next_clip() -> Optional[Path],
    die aus einem gewichteten Pool zieht und *sämtliche Duplikate*
    des gewählten Clips aus dem Pool entfernt. Wenn der Pool leer ist,
    startet automatisch eine neue Epoche.
    """
    used: Set[Path] = set()

    def _fresh_pool() -> List[Path]:
        pool_p, pool_l = _build_weighted_pools(specs, probe_cache)
        return pool_p if orientation == "portrait" else pool_l

    pool: List[Path] = _fresh_pool()

    def draw_unique_from_pool() -> Optional[Path]:
        nonlocal pool, used
        # bereits verwendete Clips dieser Epoche entfernen (amortisiert O(n))
        if used:
            pool[:] = [x for x in pool if x not in used]
            if not pool:
                print(f"Warning: Ran out of {orientation} clips - duplicates will be used")
                return None

        # Index ziehen
        idx = _rng.randrange(len(pool))
        picked = pool[idx]

        # gezogenen Eintrag entfernen (swap-pop) und *alle* verbleibenden Duplikate filtern
        pool[idx] = pool[-1]
        pool.pop()
        if pool:
            pool[:] = [x for x in pool if x != picked]

        used.add(picked)
        return picked

    def next_clip() -> Optional[Path]:
        nonlocal pool, used
        clip = draw_unique_from_pool()
        if clip is not None:
            return clip

        # Epoche erschöpft → neue starten
        pool = _fresh_pool()
        used.clear()
        # Falls es in dieser Orientierung gar keine Dateien gibt, kann der Pool leer sein
        if not pool:
            return None
        return draw_unique_from_pool()

    return next_clip

from moviepy.video.VideoClip import ImageClip
import numpy as _np
from PIL import Image as _KB_PIL_Image

def kenburns_cover(
    img_clip: ImageClip,
    target_w: int,
    target_h: int,
    dur: float,
    direction: Optional[str] = None,
    zoom_in: bool = True,
    zoom_amount: float = 0.08,
    rng: Optional[random.Random] = None,
) -> ImageClip:
    rng = rng or _rng
    if direction is None:
        direction = rng.choice(['left','right','up','down','diag1','diag2'])

    src_w, src_h = img_clip.size
    if src_w == 0 or src_h == 0:
        return img_clip.set_duration(dur)

    # Basis-Scale, plus Extra-Spielraum fürs Panning/Zoom
    base_scale = max(target_w / src_w, target_h / src_h)
    extra = 1.0 + abs(zoom_amount) + 0.02
    scale0 = base_scale * extra
    scaled_w = int(round(src_w * scale0))
    scaled_h = int(round(src_h * scale0))

    clip = img_clip.resize(newsize=(scaled_w, scaled_h)).set_duration(dur)

    max_x = max(0, scaled_w - target_w)
    max_y = max(0, scaled_h - target_h)

    # Start/Ende der Pan-Route
    if direction == 'left':
        x0, x1 = max_x, 0; y0 = y1 = max_y // 2
    elif direction == 'right':
        x0, x1 = 0, max_x; y0 = y1 = max_y // 2
    elif direction == 'up':
        y0, y1 = max_y, 0; x0 = x1 = max_x // 2
    elif direction == 'down':
        y0, y1 = 0, max_y; x0 = x1 = max_x // 2
    elif direction == 'diag1':
        x0, y0 = 0, 0; x1, y1 = max_x, max_y
    else:  # diag2
        x0, y0 = max_x, max_y; x1, y1 = 0, 0

    def _lerp(a, b, t): return a + (b - a) * t
    def _nt(t):
        if dur <= 1e-6: return 1.0
        return 0.0 if t <= 0 else 1.0 if t >= dur else (t / dur)

    # Optional: leichtes Zoom-in während des Pans (auf das Cropping angewandt)
    # Wir simulieren’s, indem wir die Pan-Geschwindigkeit minimal verzerren
    # (kein zusätzliches Resizing nötig; die Cropposition wandert etwas stärker).
    zoom_bias = abs(zoom_amount) if zoom_in else 0.0

    def _kb_frame(get_frame, t):
        p = _nt(t)
        # Minimale nonlineare Verzerrung für "Zoom-Gefühl"
        pz = min(1.0, max(0.0, p + zoom_bias * (p * (1 - p))))

        cx = int(round(_lerp(x0, x1, pz)))
        cy = int(round(_lerp(y0, y1, pz)))
        # Klemmen, damit das Fenster immer im Bild liegt
        cx = 0 if max_x == 0 else min(max(cx, 0), max_x)
        cy = 0 if max_y == 0 else min(max(cy, 0), max_y)

        frame = get_frame(t)  # ndarray (H,W,3 or 4)
        # Croppen
        crop = frame[cy:cy + target_h, cx:cx + target_w]
        # Falls an Kanten mal 1px fehlt (Rundung): sauber nachziehen
        if crop.shape[0] != target_h or crop.shape[1] != target_w:
            img = _KB_PIL_Image.fromarray(crop)
            img = img.resize((target_w, target_h), resample=_PIL_Image.BICUBIC)
            crop = _np.asarray(img, dtype=frame.dtype, order="C")
        return crop

    animated = clip.fl(_kb_frame, apply_to=['mask'])
    return animated.set_duration(dur)

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument(
        "--videos",
        type = str,
        required = True,
        help = "Ordner mit Videos. Auch mehrere mit Gewichtung möglich: "
        '"DIR[:weight],DIR[:weight],..."  (z.B. "cats:1,dogs:2"). '
        "Ohne ':weight' = 1. Abwärtskompatibel: einzelner Ordnerpfad bleibt gültig."
        )
    p.add_argument("--output", type=Path, required=True)

    # Time-based fallback (used when no BPM is provided/detected)
    p.add_argument("--min-seconds", type=float, default=2.0)
    p.add_argument("--max-seconds", type=float, default=5.0)

    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--codec", default="libx264")
    p.add_argument("--audio-codec", default="aac")
    p.add_argument("--preset", default="medium")
    p.add_argument("--bitrate")
    p.add_argument("--threads", type=int)

    # Volumes
    p.add_argument("--bg-volume", type=float, default=1.0, help="Lautstärke der Hintergrundmusik (1.0 = 100%).")
    p.add_argument("--clip-volume", type=float, default=0.8, help="Lautstärke der Clip-Audios (1.0 = 100%).")

    # Reverb on clip audio
    p.add_argument("--clip-reverb", type=float, default=0.2,
                   help="Nachhall-Stärke für Clip-Audio (0.0 = aus, 1.0 = stark; default 0.2).")

    # BPM-driven even-beat cuts (triangular weighting)
    p.add_argument("--bpm", type=float, help="BPM des Audios (aktiviert Beat-basiertes Schneiden).")
    p.add_argument("--bpm-detect", action="store_true", help="BPM automatisch aus Audiodatei ermitteln (benötigt librosa).")
    p.add_argument("--min-beats", type=int, default=2, help="Minimale Beatanzahl pro Clip (gerade Zahl, inkl.).")
    p.add_argument("--max-beats", type=int, default=8, help="Maximale Beatanzahl pro Clip (gerade Zahl, inkl.).")
    p.add_argument("--beat-mode", type=float, default=0.25,
                   help="Position des Wahrscheinlichkeitsmaximums zwischen min/max (0..1). Default 0.25 (=unterer Bereich).")

    p.add_argument("--preview", choices=["true", "false"], default="true",
                   help="Während des Renderns Preview-JPGs schreiben (true/false, default: true).")
    p.add_argument("--triptych-carry", type=float, default=0.3,
                   help="Wahrscheinlichkeit (0.0–1.0), dass das Mittel-Panel des Triptychons vom Seiten-Panel des vorherigen übernommen wird (default: 0.3).")
    p.add_argument("--pulse-effect", action="store_true", help="Beat-Pulse-Effekt aktivieren.")
    p.add_argument("--trim-large-clips", action="store_true", help="Lange Clips nur segmentweise nutzen (default: ganz verwenden).")
    p.add_argument("--fade-out-seconds", type=float, default=0.0, help="Länge des Video-/Audio-Fade-Outs am Ende (Sekunden, 0=aus).")

    args = p.parse_args(argv)

    # --- Clamping & Sanitizing ---

    def clamp(v, lo, hi):
        if v is None:
            return None
        return lo if v < lo else hi if v > hi else v

    # Seconds
    args.min_seconds = clamp(args.min_seconds, 0.2, 30.0)
    args.max_seconds = clamp(args.max_seconds, 0.3, 60.0)
    if args.max_seconds < args.min_seconds:
        args.min_seconds, args.max_seconds = args.max_seconds, args.min_seconds

    # FPS & dimensions
    args.fps = clamp(args.fps, 1.0, 240.0)
    args.width = clamp(args.width, 16, 8192)
    args.height = clamp(args.height, 16, 8192)

    # Threads
    args.threads = clamp(args.threads, 1, 64)

    # Volumes & reverb
    args.bg_volume = clamp(args.bg_volume, 0.0, 4.0)
    args.clip_volume = clamp(args.clip_volume, 0.0, 4.0)
    args.clip_reverb = clamp(args.clip_reverb, 0.0, 1.0)

    # BPM / Beat-Mode
    if args.bpm is not None:
        args.bpm = clamp(args.bpm, 30.0, 300.0)
    args.beat_mode = clamp(args.beat_mode, 0.0, 1.0)

    # Even-beats erzwingen und Min/Max konsistent halten
    def make_even(n):
        if n is None:
            return None
        return n if n % 2 == 0 else (n - 1 if n > 0 else 0)

    if args.max_beats < args.min_beats:
        args.min_beats, args.max_beats = args.max_beats, args.min_beats

    if args.min_beats == 1:
        # min bleibt 1; max bleibt gerade (>=2)
        args.max_beats = max(2, make_even(args.max_beats))
        # Falls jemand max < 2 gesetzt hat, heben wir es auf 2 an (bereits durch max(..,2) erledigt)
    else:
        # Standardmodus: nur gerade Beats (min/max >= 2, gerade)
        args.min_beats = max(2, make_even(args.min_beats))
        # max muss >= min und gerade sein
        args.max_beats = max(args.min_beats, make_even(args.max_beats))

    args.preview = (args.preview.lower() == "true")

    args.triptych_carry = max(0.0, min(1.0, args.triptych_carry))

    args.fade_out_seconds = clamp(args.fade_out_seconds, 0.0, 5.0)

    return args


def main(argv=None):
    args = parse_args(argv)
    start_stdin_exit_watcher("__PMVEAVER_EXIT__")
    build_montage(
        audio_path=args.audio,
        videos_spec=args.videos,
        out_path=args.output,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        fps=args.fps,
        width=args.width,
        height=args.height,
        codec=args.codec,
        audio_codec=args.audio_codec,
        preset=args.preset,
        bitrate=args.bitrate,
        threads=args.threads,
        bg_volume=args.bg_volume,
        clip_volume=args.clip_volume,
        clip_reverb=args.clip_reverb,
        bpm=args.bpm,
        bpm_detect=args.bpm_detect,
        min_beats=args.min_beats,
        max_beats=args.max_beats,
        beat_mode=args.beat_mode,
        preview=args.preview,
        triptych_carry=args.triptych_carry,
        pulse_effect=args.pulse_effect,
        trim_large_clips = args.trim_large_clips,
        fade_out_seconds=args.fade_out_seconds,
    )


if __name__ == "__main__":
    main()