#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Random Clip Montage
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
# - During rendering, save a preview image every 120 frames to a constant filename:
#     <output_stem>.preview.jpg  (overwritten repeatedly)
#   This preview file is deleted after the video is finished.
#
# CLI:
#   --audio PATH            background track (required)
#   --videos DIR            folder with videos/gifs (required)
#   --output PATH           output video file (required)
#   --width INT             resize output (default: keep source sizes)
#   --height INT            resize output (default: keep source sizes)
#   --fps INT               output fps (default 30)
#   --bg-volume FLOAT       volume multiplier for background audio (default 1.0)
#   --clip-volume FLOAT     volume multiplier for clip audio (default 0.8)
#   --clip-reverb FLOAT     placeholder (kept for compatibility, no-op)
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
    CompositeAudioClip,
)
from moviepy.video.fx.resize import resize
from moviepy.video.fx.all import margin, crop, mirror_x
from tqdm import tqdm
from PIL import Image as _PIL_Image

SUPPORTED_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mpg"}

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

def ffprobe_streams(path: Path) -> Dict:
    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError(
            "ffprobe not found. Please install FFmpeg and ensure ffprobe is in your PATH."
        )
    abs_path = str(path.resolve())
    args = [
        exe, "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        abs_path,
    ]
    try:
        res = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="replace") if e.stderr else str(e)
        raise RuntimeError(f"ffprobe failed for {path}: {err}") from e
    out = res.stdout.decode("utf-8", errors="replace")
    return json.loads(out)


def probe_video_dimensions(path: Path) -> Tuple[int, int, int]:
    data = ffprobe_streams(path)
    vstreams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        raise RuntimeError(f"No video stream found in {path}")
    s = vstreams[0]
    width = int(s.get("width", 0) or 0)
    height = int(s.get("height", 0) or 0)

    rotation = 0
    side_data = s.get("side_data_list") or []
    for sd in side_data:
        if sd.get("rotation") is not None:
            try:
                rotation = int(sd["rotation"])
            except Exception:
                rotation = 0
            break
    if rotation == 0:
        tags = s.get("tags") or {}
        rot_tag = tags.get("rotate") or tags.get("ROTATE")
        if rot_tag is not None:
            try:
                rotation = int(rot_tag)
            except Exception:
                rotation = 0
    rotation = rotation % 360
    if rotation in (90, 270):
        width, height = height, width
    return width, height, rotation


def is_portrait_file(path: Path) -> bool:
    w, h, _ = probe_video_dimensions(path)
    return h > w


# ---------------- core helpers ----------------

def find_video_files(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]


def pick_segment_bounds_random_seconds(duration: float, min_s: float, max_s: float) -> Tuple[float, float]:
    seg_len = random.uniform(min_s, max_s)
    if duration <= 0:
        return 0.0, 0.0
    if seg_len >= duration:
        return 0.0, duration
    start = random.uniform(0, duration - seg_len)
    return start, start + seg_len


def pick_segment_bounds_fixed(duration: float, seg_len: float) -> Tuple[float, float]:
    if duration <= 0:
        return 0.0, 0.0
    if seg_len >= duration:
        return 0.0, duration
    start = random.uniform(0, max(0.0, duration - seg_len))
    return start, start + seg_len


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def cover_scale_and_crop(clip: VideoFileClip, target_w: int, target_h: int) -> VideoFileClip:
    src_w, src_h = clip.size
    if src_w == 0 or src_h == 0:
        return clip
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = resize(clip, newsize=(new_w, new_h))
    x1 = max(0, (new_w - target_w) // 2)
    y1 = max(0, (new_h - target_h) // 2)
    return crop(resized, x1=x1, y1=y1, x2=x1 + target_w, y2=y1 + target_h)


def letterbox(clip: VideoFileClip, target_w: int, target_h: int) -> VideoFileClip:
    src_w, src_h = clip.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = resize(clip, newsize=(new_w, new_h))
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    return margin(
        resized,
        left=pad_x,
        right=target_w - new_w - pad_x,
        top=pad_y,
        bottom=target_h - new_h - pad_y,
        color=(0, 0, 0),
    )


def make_triptych(clipA: VideoFileClip, clipB: VideoFileClip, target_w: int, target_h: int) -> CompositeVideoClip:
    panel_w = target_w // 3
    A_left  = cover_scale_and_crop(clipA, panel_w, target_h).set_position((0, 0))
    B_mid   = cover_scale_and_crop(clipB, panel_w, target_h).set_position((panel_w, 0))
    A_right = mirror_x(cover_scale_and_crop(clipA, panel_w, target_h)).set_position((2 * panel_w, 0))
    return CompositeVideoClip([A_left, B_mid, A_right], size=(3 * panel_w, target_h))


# ---------------- BPM helpers ----------------

def detect_bpm_with_librosa(audio_path: Path) -> float:
    try:
        import librosa  # type: ignore
    except Exception:
        raise RuntimeError(
            "BPM auto-detect requires librosa. Install with: pip install librosa==0.10.1"
        )
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if tempo is None or tempo <= 0:
        raise RuntimeError("Could not detect BPM from audio.")
    return float(tempo)


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
    Choose an even beat count in [min_beats, max_beats] using a triangular weighting:
    mode_pos in [0..1] sets where the maximum probability lies between min and max.
    """
    lo, hi, evens = _even_bounds(min_beats, max_beats)
    if not evens:
        return 2
    mode_pos = clamp(mode_pos, 0.0, 1.0)
    mode = lo + mode_pos * (hi - lo)
    eps = 1e-9
    denom_left  = max(mode - lo, eps)
    denom_right = max(hi - mode, eps)
    weights: List[float] = []
    for b in evens:
        if b <= mode:
            w = (b - lo) / denom_left
        else:
            w = (hi - b) / denom_right
        weights.append(max(w, 0.05))  # small floor so extremes never vanish
    return random.choices(evens, weights=weights, k=1)[0]


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


# ---------------- main build ----------------

def build_montage(
    audio_path: Path,
    videos_folder: Path,
    out_path: Path,
    min_seconds: float,
    max_seconds: float,
    fps: int,
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
    elif bpm_detect:
        effective_bpm = detect_bpm_with_librosa(audio_path)

    all_files = find_video_files(videos_folder)
    if not all_files:
        raise RuntimeError("No video files found")

    # categorize by orientation (rotation-aware)
    portrait_files: List[Path] = []
    landscape_files: List[Path] = []
    for p in all_files:
        try:
            if is_portrait_file(p):
                portrait_files.append(p)
            else:
                landscape_files.append(p)
        except Exception as e:
            print(f"Warning: ffprobe failed for {p}: {e}", file=sys.stderr)

    if not portrait_files and not landscape_files:
        raise RuntimeError("No valid video files after probing")

    # Randomize clip order every run
    random.shuffle(portrait_files)
    random.shuffle(landscape_files)

    segments: List[VideoFileClip] = []
    source_clips: List[VideoFileClip] = []
    total = 0.0

    pbar = tqdm(total=target_duration, desc="Collecting clips", unit="s")
    default_w = width or 1920
    default_h = height or 1080

    idx_land = 0
    idx_port = 0
    use_land_next = True

    while total < target_duration + 0.01:
        try:
            pick_land = bool(landscape_files) and (use_land_next or not portrait_files)
            if pick_land and landscape_files:
                src_path = landscape_files[idx_land % len(landscape_files)]
                idx_land += 1
                use_land_next = False

                src = VideoFileClip(str(src_path))
                source_clips.append(src)

                # Segment duration per mode
                if effective_bpm and effective_bpm > 0:
                    beats = choose_even_beats(min_beats, max_beats, mode_pos=beat_mode)
                    seg_len = (60.0 / effective_bpm) * beats
                    start, end = pick_segment_bounds_fixed(src.duration, seg_len)
                else:
                    start, end = pick_segment_bounds_random_seconds(src.duration, min_seconds, max_seconds)

                if end <= start:
                    src.close()
                    continue

                sub = src.subclip(start, end).set_fps(fps)
                filled = cover_scale_and_crop(sub, default_w, default_h)
                segments.append(filled)
                total += filled.duration

            else:
                if not portrait_files:
                    continue

                pathA = portrait_files[idx_port % len(portrait_files)]
                idx_port += 1

                if len(portrait_files) > 1:
                    pathB = portrait_files[idx_port % len(portrait_files)]
                    if pathB == pathA:
                        pathB = portrait_files[(idx_port + 1) % len(portrait_files)]
                else:
                    pathB = pathA

                srcA = VideoFileClip(str(pathA))
                srcB = VideoFileClip(str(pathB))
                source_clips.extend([srcA, srcB])

                if effective_bpm and effective_bpm > 0:
                    beats = choose_even_beats(min_beats, max_beats, mode_pos=beat_mode)
                    seg_len = (60.0 / effective_bpm) * beats
                    sA, eA = pick_segment_bounds_fixed(srcA.duration, seg_len)
                    sB, eB = pick_segment_bounds_fixed(srcB.duration, seg_len)
                else:
                    sA, eA = pick_segment_bounds_random_seconds(srcA.duration, min_seconds, max_seconds)
                    sB, eB = pick_segment_bounds_random_seconds(srcB.duration, min_seconds, max_seconds)

                if eA <= sA or eB <= sB:
                    srcA.close(); srcB.close()
                    continue

                subA = srcA.subclip(sA, eA).set_fps(fps)
                subB = srcB.subclip(sB, eB).set_fps(fps)

                dur = min(subA.duration, subB.duration)
                if dur <= 0:
                    subA.close(); subB.close()
                    continue

                subA = subA.subclip(0, dur)
                subB = subB.subclip(0, dur)

                trip = make_triptych(subA, subB, default_w, default_h)
                segments.append(trip)
                total += trip.duration
                use_land_next = True

        except Exception as e:
            print(f"Warning: building segment failed: {e}", file=sys.stderr)
            continue

        if segments:
            pbar.update(segments[-1].duration)

    pbar.close()

    # Fit to exact target length
    over = total - target_duration
    if over > 1e-3 and segments:
        last = segments[-1]
        keep = max(0.0, last.duration - over)
        segments[-1] = last.subclip(0, keep)

    # Concatenate (no crossfade)
    montage = concatenate_videoclips(segments, method="compose")
    montage = montage.subclip(0, target_duration).set_fps(fps)

    # ----- Audio: normalize (+ optional denoise) -> (optional reverb) -> export stems -> mix with bg -----
    clip_audio = montage.audio
    temp_dirs_to_cleanup: List[str] = []

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
    bg_track = AudioFileClip(str(audio_path)).set_duration(target_duration).volumex(bg_volume)

    if clip_audio is not None:
        composite_audio = CompositeAudioClip([bg_track, clip_audio.set_duration(target_duration)])
    else:
        composite_audio = bg_track

    montage = montage.set_audio(composite_audio)

    # --- Live preview via frame pipeline: save every 120th frame to a constant file
    preview_path = out_path.parent / f"{out_path.stem}.preview.jpg"
    frame_counter = {"i": -1}

    def _preview_writer(frame):
        frame_counter["i"] += 1
        if frame_counter["i"] % 120 == 0:
            try:
                _PIL_Image.fromarray(frame).save(str(preview_path), format="JPEG", quality=90)
            except Exception:
                pass
        return frame

    montage_with_preview = montage.fl_image(_preview_writer)

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
    montage.close()
    for seg in segments:
        try: seg.close()
        except Exception: pass
    for src in source_clips:
        try: src.close()
        except Exception: pass
    try:
        for d in temp_dirs_to_cleanup:
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass

def setup_tempfile_cleanup(out_path):
    def _cleanup():
        try:
            # preview-Datei löschen
            preview = out_path.parent / f"{out_path.stem}.preview.jpg"
            if preview.exists():
                preview.unlink()

            # MoviePy-Tempfiles löschen
            for f in out_path.parent.glob(f"{out_path.stem}TEMP_MPY_*"):
                try:
                    f.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    # bei normalem Exit
    atexit.register(_cleanup)

    # bei Abbruch-Signalen
    def _handler(signum, frame):
        _cleanup()
        sys.exit(1)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--videos", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    # Time-based fallback (used when no BPM is provided/detected)
    p.add_argument("--min-seconds", type=float, default=2.0)
    p.add_argument("--max-seconds", type=float, default=5.0)

    p.add_argument("--fps", type=int, default=30)
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

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    build_montage(
        audio_path=args.audio,
        videos_folder=args.videos,
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
    )


if __name__ == "__main__":
    main()
