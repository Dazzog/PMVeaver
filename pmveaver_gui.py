#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, io, time, threading, subprocess, shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import OrderedDict

try:
    from PIL import Image, ImageTk
    _PIL_OK = True
except Exception:
    _PIL_OK = False

APP_TITLE = "Random Clip Montage – GUI"

def hms(seconds: float) -> str:
    if seconds is None or seconds < 0: return "—"
    s = int(seconds); h, rem = divmod(s, 3600); m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def _base_dir() -> Path:
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent

def _find_cli_candidate():
    """Bevorzuge pmveaver.exe neben der GUI, sonst pmveaver.py."""
    base = _base_dir()
    cand_exe = base / "pmveaver.exe"
    if cand_exe.exists():
        return cand_exe, True
    cand_py = base / "pmveaver.py"
    if cand_py.exists():
        return cand_py, False
    # Fallback: neben der Quelle
    return Path(__file__).with_name("pmveaver.py"), False

def which_ffmpeg_bins():
    """Suche ffmpeg/ffprobe in PATH, ansonsten neben der EXE/GUI."""
    ffm, ffp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffm or not ffp:
        base = _base_dir()
        cand_ffm = base / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        cand_ffp = base / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if cand_ffm.exists(): ffm = str(cand_ffm)
        if cand_ffp.exists(): ffp = str(cand_ffp)
    return ffm, ffp

def _norm(p: str) -> str:
    try: return os.path.normpath(p)
    except Exception: return p

class PMVeaverGUI(tk.Tk):
    RE_TIME = re.compile(r"time=(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)")
    RE_PCT  = re.compile(r'(\d{1,3}(?:\.\d+)?)\s*%')
    RE_FRAC = re.compile(r'(\d+(?:\.\d+)?)[ ]*/[ ]*(\d+(?:\.\d+)?)')

    STEP_WEIGHTS = OrderedDict([
        ("collecting clips", (0.00, 0.10)),
        ("building", (0.10, 0.125)),
        ("writing audio", (0.125, 0.15)),
        ("writing video", (0.15, 1.00)),
    ])

    STEP_ORDER = list(STEP_WEIGHTS.keys())

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)

        # state
        self.proc=None; self.running=False; self._aborted=False
        self._current_line=""; self._last_step_pct=None; self._last_frac=(None,None)
        self._target_total_secs=None; self._start_time=None
        self._phase="—"; self._phase_idx=-1
        self._overall_progress=0.0

        # preview state
        self._preview_candidates=[]
        self._preview_path=None
        self._preview_mtime=None
        self._preview_tkimg=None
        self._preview_target_h = 100
        self._preview_max_w    = 180

        # form vars
        self.var_audio=tk.StringVar(); self.var_videos=tk.StringVar(); self.var_output=tk.StringVar()
        self.var_width=tk.StringVar(value="1920"); self.var_height=tk.StringVar(value="1080")
        self.var_fps=tk.StringVar(value="30")
        self.var_bg_vol=tk.DoubleVar(value=1.0)   # „Audio Volume“
        self.var_clip_vol=tk.DoubleVar(value=0.8)
        self.var_clip_reverb=tk.DoubleVar(value=0.2)
        self.var_min_seconds=tk.StringVar(value="2.0"); self.var_max_seconds=tk.StringVar(value="5.0")
        self.var_bpm_detect=tk.BooleanVar(value=True)  # default ON
        self.var_bpm=tk.StringVar(value="")
        self.var_min_beats=tk.IntVar(value=2); self.var_max_beats=tk.IntVar(value=8); self.var_beat_mode=tk.DoubleVar(value=0.25)
        self.var_codec=tk.StringVar(value="libx264"); self.var_audio_codec=tk.StringVar(value="aac")
        self.var_preset=tk.StringVar(value="medium"); self.var_bitrate=tk.StringVar(value=""); self.var_threads=tk.StringVar(value="")

        # Standard-Ordner für Videos erraten
        try:
            cwd = os.getcwd()
            entries = {entry.lower(): entry for entry in os.listdir(cwd)}
            for name in ["videos", "clips", "input", "inputs", "source", "sources", "footage"]:
                if name.lower() in entries:
                    candidate = os.path.join(cwd, entries[name.lower()])
                    if os.path.isdir(candidate):
                        self.var_videos.set(_norm(candidate))
                        break
        except:
            pass

        self.entry_bpm=None  # widget handle

        self._build_ui()
        self._check_ffmpeg()

        # BPM-UI-Verhalten
        self.var_bpm_detect.trace_add("write", lambda *_: self._sync_bpm_ui())
        self.var_bpm.trace_add("write", lambda *_: self._on_bpm_text_change())
        self._sync_bpm_ui()

        self.after(50, self._autosize_window)
        self.after(200, self._tick)
        self.after(300, self._preview_tick)

    # ---------- UI ----------
    def _build_ui(self):
        pad={"padx":8,"pady":6}

        # --- Sources / Output (Grid -> saubere Ausrichtung) ---
        frm_top=ttk.LabelFrame(self,text="Sources / Output")
        frm_top.pack(fill="x",**pad)
        src = ttk.Frame(frm_top); src.pack(fill="x",padx=8,pady=4)

        ttk.Label(src, text="Audio:", anchor="e").grid(row=0, column=0, sticky="e")
        self.entry_audio = ttk.Entry(src, textvariable=self.var_audio)
        self.entry_audio.grid(row=0, column=1, sticky="we", padx=(8,8))
        ttk.Button(src, text="Browse…", command=self._browse_audio).grid(row=0, column=2, sticky="e")

        ttk.Label(src, text="Video folder:", anchor="e").grid(row=1, column=0, sticky="e", pady=(6,0))
        self.entry_videos = ttk.Entry(src, textvariable=self.var_videos)
        self.entry_videos.grid(row=1, column=1, sticky="we", padx=(8,8), pady=(6,0))
        ttk.Button(src, text="Browse…", command=self._browse_videos).grid(row=1, column=2, sticky="e", pady=(6,0))

        ttk.Label(src, text="Output file:", anchor="e").grid(row=2, column=0, sticky="e", pady=(6,0))
        self.entry_output = ttk.Entry(src, textvariable=self.var_output)
        self.entry_output.grid(row=2, column=1, sticky="we", padx=(8,8), pady=(6,0))
        ttk.Button(src, text="Browse…", command=self._browse_output).grid(row=2, column=2, sticky="e", pady=(6,0))

        src.columnconfigure(1, weight=1)

        # --- Frame & Render ---
        frm_render=ttk.LabelFrame(self,text="Frame & Render")
        frm_render.pack(fill="x",**pad)
        grid=ttk.Frame(frm_render); grid.pack(fill="x",padx=8,pady=4)
        self._labeled_entry(grid,"Width", self.var_width,0,0,10)
        self._labeled_entry(grid,"Height",self.var_height,0,2,10)
        self._labeled_entry(grid,"FPS",   self.var_fps,  0,4,10)

        # --- Audio Mix ---
        frm_audio=ttk.LabelFrame(self,text="Audio Mix")
        frm_audio.pack(fill="x",**pad)
        vg=ttk.Frame(frm_audio); vg.pack(fill="x",padx=8,pady=4)
        self._labeled_scale(vg, "Audio Volume", self.var_bg_vol,      0, 0, 0.0, 2.0, 0.01, colspan=1)
        self._labeled_scale(vg, "Clip Volume",  self.var_clip_vol,    0, 3, 0.0, 2.0, 0.01, colspan=1)
        self._labeled_scale(vg, "Clip Reverb",  self.var_clip_reverb, 0, 6, 0.0, 1.0, 0.01, colspan=1)

        # --- BPM ---
        frm_bpm=ttk.LabelFrame(self,text="BPM / Beat lengths")
        frm_bpm.pack(fill="x",**pad)
        bg=ttk.Frame(frm_bpm); bg.pack(fill="x",padx=8,pady=4)
        self.chk_bpm = ttk.Checkbutton(bg,text="Automatically detect BPM (librosa)",variable=self.var_bpm_detect)
        self.chk_bpm.grid(row=0,column=0,columnspan=4,sticky="w",pady=(0,6))

        ttk.Label(bg,text="BPM (manual)").grid(row=1,column=0,sticky="w")
        self.entry_bpm = ttk.Entry(bg,textvariable=self.var_bpm,width=10)
        self.entry_bpm.grid(row=1,column=1,sticky="w",padx=(6,18))

        self._labeled_spin(bg,"Min beats",self.var_min_beats,1,2,2,64,2)
        self._labeled_spin(bg,"Max beats",self.var_max_beats,1,4,2,64,2)

        self._labeled_scale(bg, "Beat mode (0..1, peak position)", self.var_beat_mode,
                            2, 0, 0.0, 1.0, 0.01, colspan=5, fmt="{:.2f}")
        bg.columnconfigure(1, weight=1)

        # --- Time-based fallback ---
        frm_time=ttk.LabelFrame(self,text="Time-based fallback (when BPM disabled)")
        frm_time.pack(fill="x",**pad)
        tg=ttk.Frame(frm_time); tg.pack(fill="x",padx=8,pady=4)
        self._labeled_entry(tg,"Min seconds",self.var_min_seconds,0,0,10)
        self._labeled_entry(tg,"Max seconds",self.var_max_seconds,0,2,10)

        # --- Codecs & Performance ---
        frm_codec=ttk.LabelFrame(self,text="Codecs & Performance")
        frm_codec.pack(fill="x",**pad)
        cg=ttk.Frame(frm_codec); cg.pack(fill="x",padx=8,pady=4)
        self._labeled_combo(cg,"Video codec",self.var_codec,0,0,
                            ["libx264","libx265","h264_nvenc","hevc_nvenc","prores_ks","libvpx-vp9"],
                            pady=(0,8))
        self._labeled_combo(cg,"Audio codec",self.var_audio_codec,0,2,
                            ["aac","libopus","libmp3lame"],
                            pady=(0,8))
        self._labeled_combo(cg,"Preset",self.var_preset,0,4,
                            ["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"],
                            pady=(0,8))
        self._labeled_entry(cg,"Bitrate (e.g., 8M)",self.var_bitrate,1,0,12).grid_configure(pady=(0,0))
        self._labeled_entry(cg,"Threads",self.var_threads,1,2,8).grid_configure(pady=(0,0))
        cg.columnconfigure(5,weight=1)

        # --- Progress + Preview ---
        frm_prog=ttk.LabelFrame(self,text="Progress")
        frm_prog.pack(fill="x",**pad)
        inner=ttk.Frame(frm_prog); inner.pack(fill="x",padx=8,pady=6)

        # links: Preview
        left = ttk.Frame(inner)
        left.grid(row=0, column=0, sticky="nw", padx=(0,12))
        self.lbl_preview = ttk.Label(left)
        self.lbl_preview.pack()

        # rechts: Progress
        right = ttk.Frame(inner)
        right.grid(row=0, column=1, sticky="nsew")
        inner.columnconfigure(1, weight=1)

        ttk.Label(right,text="Current step:").grid(row=0,column=0,sticky="w")
        self.var_step=tk.StringVar(value="—")
        ttk.Label(right,textvariable=self.var_step).grid(row=0,column=1,columnspan=2,sticky="w",padx=(6,0))
        self.lbl_elapsed=ttk.Label(right,text="Elapsed: —"); self.lbl_elapsed.grid(row=0,column=3,sticky="e")
        self.lbl_eta=ttk.Label(right,text="ETA: —"); self.lbl_eta.grid(row=0,column=4,sticky="e",padx=(8,0))

        self.pb_step=ttk.Progressbar(right,orient="horizontal",mode="determinate",maximum=100,value=0)
        self.pb_step.grid(row=1,column=0,columnspan=4,sticky="we",pady=(4,2))
        self.lbl_step_pct=ttk.Label(right,text="0%"); self.lbl_step_pct.grid(row=1,column=4,sticky="e",padx=(8,0))
        right.columnconfigure(1, weight=1)
        right.columnconfigure(2, weight=1)

        ttk.Label(right,text="Total progress:").grid(row=2,column=0,sticky="w",pady=(6,0))
        self.pb_total=ttk.Progressbar(right,orient="horizontal",mode="determinate",maximum=100,value=0)
        self.pb_total.grid(row=3,column=0,columnspan=4,sticky="we")
        self.lbl_total_pct=ttk.Label(right,text="0%"); self.lbl_total_pct.grid(row=3,column=4,sticky="e",padx=(8,0))

        # --- Buttons + FFmpeg-Check ganz unten ---
        frm_btns=ttk.Frame(self)
        frm_btns.pack(fill="x",**pad)
        self.btn_start=ttk.Button(frm_btns,text="Start",command=self.start)
        self.btn_stop =ttk.Button(frm_btns,text="Stop", command=self.stop,state="disabled")
        self.btn_start.pack(side="left")
        self.btn_stop.pack(side="left",padx=(6,0))
        self.lbl_ffmpeg=ttk.Label(frm_btns,text="FFmpeg: …")
        self.lbl_ffmpeg.pack(side="right")

    # ----- autosize -----
    def _autosize_window(self):
        self.update_idletasks()
        req_w=self.winfo_reqwidth(); req_h=self.winfo_reqheight()
        max_w=int(self.winfo_screenwidth()*0.9); max_h=int(self.winfo_screenheight()*0.9)
        final_w=min(req_w+20, max_w); final_h=min(req_h+20, max_h)
        self.geometry(f"{final_w}x{final_h}")
        self.minsize(final_w, final_h)

    # ---- labeled helpers ----
    def _labeled_entry(self,parent,label,var,r,c,width=10):
        ttk.Label(parent,text=label).grid(row=r,column=c,sticky="w")
        e=ttk.Entry(parent,textvariable=var,width=width)
        e.grid(row=r,column=c+1,sticky="w",padx=(6,18))
        return e

    def _labeled_spin(self,parent,label,var,r,c,from_,to,increment):
        ttk.Label(parent,text=label).grid(row=r,column=c,sticky="w")
        s=ttk.Spinbox(parent,textvariable=var,from_=from_,to=to,increment=increment,width=8)
        s.grid(row=r,column=c+1,sticky="w",padx=(6,18))
        return s

    def _labeled_combo(self,parent,label,var,r,c,values,pady=(0,0)):
        ttk.Label(parent,text=label).grid(row=r,column=c,sticky="w",pady=pady)
        cb=ttk.Combobox(parent,textvariable=var,values=values,width=16,state="readonly")
        cb.grid(row=r,column=c+1,sticky="w",padx=(6,18),pady=pady)
        return cb

    def _labeled_scale(self, parent, label, var, r, c, from_, to, resolution, colspan=1, fmt="{:.2f}"):
        # Textlabel
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w")
        # Slider
        sc = ttk.Scale(parent, from_=from_, to=to, orient="horizontal", variable=var)
        sc.grid(row=r, column=c+1, columnspan=colspan, sticky="we", padx=(6, 6))
        # Zahlenlabel rechts neben dem Slider – mit zusätzlichem rechten Abstand
        lbl = ttk.Label(parent, text="")
        lbl.grid(row=r, column=c+1+colspan, sticky="w", padx=(0, 18))
        # Slider-Spalte dehnbar
        parent.columnconfigure(c+1, weight=1)

        def upd(*_):
            try:
                lbl.config(text=fmt.format(float(var.get())))
            except Exception:
                lbl.config(text=str(var.get()))
        var.trace_add("write", lambda *_: upd())
        upd()
        return sc

    # ---------- FFmpeg check ----------
    def _check_ffmpeg(self):
        ffm,ffp=which_ffmpeg_bins()
        if ffm and ffp: self.lbl_ffmpeg.config(text=f"FFmpeg OK ✅  (ffmpeg: {Path(ffm).name}, ffprobe: {Path(ffp).name})")
        else: self.lbl_ffmpeg.config(text="FFmpeg/ffprobe not found ⚠️ – please install and add to PATH")

    # ---------- Start/Stop ----------
    def start(self):
        if self.running: return
        args=self._build_args()
        if not args: return
        print("GUI> Starting:", " ".join(self._quote(a) for a in args), flush=True)

        # Preview-Kandidaten vorbereiten
        self._init_preview_candidates(self.var_output.get().strip())

        env=dict(os.environ); env.setdefault("PYTHONUNBUFFERED","1"); env.setdefault("TQDM_MININTERVAL","0.1")
        try:
            self.proc=subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=False, bufsize=0, env=env)
        except Exception as e:
            messagebox.showerror("Start failed", str(e)); return
        self.running=True; self._aborted=False; self._start_time=time.time()
        self._reset_progress()
        self.btn_start.config(state="disabled"); self.btn_stop.config(state="normal")
        threading.Thread(target=self._reader_thread, daemon=True).start()

    def stop(self):
        if self.proc and self.running:
            print("GUI> Abort requested…", flush=True)
            self._aborted=True
            try: self.proc.terminate()
            except Exception: pass
            self._reset_progress()
            self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")
        self.running=False

    # ---------- Reader / Parser ----------
    def _reader_thread(self):
        try:
            stream=self.proc.stdout
            while True:
                chunk=stream.read(256)
                if not chunk: break
                # pass-through logs
                try: sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
                except Exception:
                    try: sys.stdout.write(chunk.decode("utf-8",errors="ignore")); sys.stdout.flush()
                    except Exception: pass
                # parse
                try: text=chunk.decode("utf-8",errors="ignore")
                except Exception: text=chunk.decode(errors="ignore")
                for ch in text:
                    if ch=="\r":
                        self._handle_progress_line(self._current_line, True); self._current_line=""
                    elif ch=="\n":
                        if not self._handle_progress_line(self._current_line, False): pass
                        self._current_line=""
                    else:
                        self._current_line+=ch
            if self._current_line.strip():
                self._handle_progress_line(self._current_line, False)
        finally:
            code=self.proc.wait(); self.running=False
            self.after(0, lambda:(self.btn_start.config(state="normal"), self.btn_stop.config(state="disabled")))
            if self._aborted:
                print("\n⛔ Aborted by user.\n", flush=True)
            else:
                def done_ui():
                    self.pb_step.config(value=100); self.pb_total.config(value=100)
                    self.lbl_step_pct.config(text="100%"); self.lbl_total_pct.config(text="100%")
                    if code==0:
                        messagebox.showinfo("Finished", f"Render complete:\n{self.var_output.get()}")
                    else:
                        messagebox.showerror("Failed", f"Process exited with code {code}")
                self.after(0, done_ui)
                print("\n✅ Done without errors.\n" if code==0 else f"\n❌ Process exited with code {code}\n", flush=True)

    # ---------- Progress parsing ----------
    def _detect_phase_from_line(self, s: str):
        sl = s.lower()
        if re.search(r'\bcollecting\b.*\bclips\b', sl): return "Collecting clips"
        if "writing video" in sl: return "Writing video"
        if "writing audio" in sl: return "Writing audio"
        if "building video" in sl or "building audio" in sl: return "Building"
        return None

    def _phase_order_idx(self, name: str) -> int:
        key = (name or "").strip().lower()
        try: return self.STEP_ORDER.index(key)
        except ValueError: return -1

    def _handle_progress_line(self, line:str, from_cr:bool)->bool:
        s=line.strip()
        if not s: return False

        phase = self._detect_phase_from_line(s)
        if phase: self._schedule_set_phase(phase)

        pct=self._extract_percent(s); frac=self._extract_fraction(s)
        if frac and frac[1] and not self._target_total_secs: self._target_total_secs=float(frac[1])

        handled=False
        if pct is not None or frac is not None:
            self._schedule_progress_update(pct, frac); handled=True

        if "time=" in s:
            m=self.RE_TIME.search(s)
            if m:
                hh,mm,ss=int(m.group(1)),int(m.group(2)),float(m.group(3))
                cur=hh*3600+mm*60+ss
                if self._target_total_secs and self._target_total_secs>0:
                    pct_ff=max(0.0,min(100.0,100.0*cur/self._target_total_secs))
                    self._schedule_set_phase("Writing video")
                    self._schedule_progress_update(pct_ff, None)
                    handled=True
        return handled and from_cr

    def _extract_percent(self,s:str):
        m=self.RE_PCT.search(s)
        if not m: return None
        try: return float(m.group(1))
        except Exception: return None

    def _extract_fraction(self,s:str):
        m=self.RE_FRAC.search(s)
        if not m: return None
        try: return (float(m.group(1)), float(m.group(2)))
        except Exception: return None

    # ---------- Thread-safe UI scheduling ----------
    def _schedule_progress_update(self,pct=None,frac=None): self.after(0, lambda:self._update_progress(pct,frac))
    def _schedule_set_phase(self,name:str): self.after(0, lambda:self._set_phase(name))

    # ---------- Progress / times ----------
    def _reset_progress(self):
        for pb,label in ((self.pb_step,self.lbl_step_pct),(self.pb_total,self.lbl_total_pct)):
            pb.config(maximum=100,value=0); label.config(text="0%")
        self._phase="—"; self._phase_idx=-1; self.var_step.set("—")
        self.lbl_elapsed.config(text="Elapsed: —"); self.lbl_eta.config(text="ETA: —")
        self._current_line=""; self._last_step_pct=None; self._last_frac=(None,None)
        self._target_total_secs=None; self._overall_progress=0.0
        # Preview bleibt stehen

    def _set_phase(self,name:str):
        new_idx = self._phase_order_idx(name)
        if new_idx < 0: return
        if self._phase_idx != -1 and new_idx < self._phase_idx:
            return
        if new_idx == self._phase_idx:
            self._phase=name; self.var_step.set(name); return

        prev_key = (self._phase or "").strip().lower()
        if prev_key in self.STEP_WEIGHTS:
            a,b = self.STEP_WEIGHTS[prev_key]
            self._overall_progress = max(self._overall_progress, b*100.0)

        self._phase=name; self._phase_idx=new_idx; self.var_step.set(name)
        self._last_step_pct=None
        self.pb_step.config(value=0); self.lbl_step_pct.config(text="0%")

        key = (name or "").strip().lower()
        if key in self.STEP_WEIGHTS:
            a,_ = self.STEP_WEIGHTS[key]
            self._overall_progress = max(self._overall_progress, a*100.0)
            self.pb_total.config(value=self._overall_progress)
            self.lbl_total_pct.config(text=f"{self._overall_progress:5.1f}%")

    def _overall_from_step_pct(self, step_pct: float, phase_name: str) -> float:
        key = (phase_name or "").strip().lower()
        if key in self.STEP_WEIGHTS:
            a,b = self.STEP_WEIGHTS[key]
            r = max(0.0, min(1.0, (step_pct or 0.0)/100.0))
            val = (a + r*(b-a)) * 100.0
            return max(self._overall_progress, max(a*100.0, min(val, b*100.0)))
        return max(self._overall_progress, max(0.0, min(100.0, float(step_pct or 0.0))))

    def _update_progress(self,pct=None,frac=None):
        if pct is None and frac and frac[1]:
            try: pct=100.0*frac[0]/frac[1]
            except Exception: pct=None
        if pct is None: return
        pct=max(0.0,min(100.0,float(pct)))
        self._last_step_pct=pct
        if frac: self._last_frac=frac

        self.pb_step.config(value=pct); self.lbl_step_pct.config(text=f"{pct:5.1f}%")
        overall = self._overall_from_step_pct(pct, self._phase)
        self._overall_progress = max(self._overall_progress, overall)
        self.pb_total.config(value=self._overall_progress)
        self.lbl_total_pct.config(text=f"{self._overall_progress:5.1f}%")
        self._refresh_time_labels()

    def _refresh_time_labels(self):
        if self._start_time is None: return
        elapsed=time.time()-self._start_time
        self.lbl_elapsed.config(text=f"Elapsed: {hms(elapsed)}")
        pct_total = self._overall_progress
        if pct_total and pct_total>0.1:
            eta=elapsed*(100.0-pct_total)/pct_total
            self.lbl_eta.config(text=f"ETA: {hms(eta)}")
        else:
            self.lbl_eta.config(text="ETA: —")

    def _tick(self):
        if self.running: self._refresh_time_labels()
        self.after(200, self._tick)

    # ---------- Preview ----------
    def _compute_preview_candidates(self, output_path: str):
        try:
            p = Path(output_path)
            stem = p.stem
            folder = p.parent
            return [
                folder / f"{stem}.preview.jpg",
                folder / f"{stem}.preview.jpeg",
                folder / f"{stem}.preview.png",
            ]
        except Exception:
            return []

    def _init_preview_candidates(self, out: str):
        out = _norm(out or "")
        self._preview_candidates = self._compute_preview_candidates(out) if out else []
        self._preview_path = None
        self._preview_mtime = None
        self._preview_tkimg = None

    def _preview_tick(self):
        try:
            if not self.running or not _PIL_OK: return
            if not self._preview_candidates: return
            chosen = next((c for c in self._preview_candidates if c.exists()), None)
            if chosen is None: return
            if (self._preview_path is None) or (str(chosen) != self._preview_path):
                self._preview_path = str(chosen); self._preview_mtime = None
            mt = chosen.stat().st_mtime
            if self._preview_mtime is None or mt > self._preview_mtime:
                self._preview_mtime = mt
                self._load_preview_image(chosen)
        except Exception:
            pass
        finally:
            self.after(300, self._preview_tick)

    def _load_preview_image(self, path: Path):
        try:
            with open(path, "rb") as f:
                data = f.read()
            im = Image.open(io.BytesIO(data)); im.load()
            # Höhe 100, Max-Breite 180
            if im.height > 0:
                ratio_h = self._preview_target_h / float(im.height)
                w_h = int(round(im.width * ratio_h)); h_h = self._preview_target_h
                if w_h > self._preview_max_w:
                    ratio_w = self._preview_max_w / float(w_h)
                    new_w = max(1, int(round(w_h * ratio_w)))
                    new_h = max(1, int(round(h_h * ratio_w)))
                else:
                    new_w, new_h = max(1, w_h), max(1, h_h)
                im = im.resize((new_w, new_h), Image.LANCZOS)
            self._preview_tkimg = ImageTk.PhotoImage(im)
            self.lbl_preview.config(image=self._preview_tkimg)
        except Exception:
            self.lbl_preview.config(image="")
            self._preview_tkimg = None

    # ---------- BPM UI Logic ----------
    def _sync_bpm_ui(self):
        detect = bool(self.var_bpm_detect.get())
        if self.entry_bpm:
            self.entry_bpm.config(state=("disabled" if detect else "normal"))

    def _on_bpm_text_change(self):
        txt = (self.var_bpm.get() or "").strip()
        if txt and self.var_bpm_detect.get():
            self.var_bpm_detect.set(False)
        self._sync_bpm_ui()

    # ---------- misc ----------
    def _quote(self,s:str)->str:
        if any(ch.isspace() for ch in s) or any(ch in s for ch in ('"', "'", "(", ")", "&", "!", "|", ";")):
            return f"\"{s}\""
        return s

    # Args + Pfad-Validation
    def _build_args(self):
        script_path, is_exe = _find_cli_candidate()
        if not script_path.exists():
            messagebox.showerror("Missing script", f"Could not find {script_path.name} next to the GUI."); return None

        audio=_norm(self.var_audio.get().strip())
        videos=_norm(self.var_videos.get().strip())
        output=_norm(self.var_output.get().strip())

        if not audio:
            messagebox.showerror("Missing path","Please choose an audio file."); return None
        if not Path(audio).is_file():
            messagebox.showerror("Invalid audio","Selected audio file does not exist."); return None

        if not videos:
            messagebox.showerror("Missing folder","Please choose the video folder."); return None
        if not Path(videos).is_dir():
            messagebox.showerror("Invalid folder","Selected video folder does not exist."); return None

        if not output:
            messagebox.showerror("Missing output","Please choose an output file."); return None
        parent = Path(output).parent
        if not parent.exists():
            messagebox.showerror("Invalid output","Output folder does not exist."); return None
        try:
            if not os.access(parent, os.W_OK):
                messagebox.showerror("No permission","Output folder is not writable."); return None
        except Exception:
            pass

        if is_exe:
            args=[str(script_path), "--audio", audio, "--videos", videos, "--output", output]
        else:
            args=[sys.executable or "python", str(script_path), "--audio", audio, "--videos", videos, "--output", output]

        if self.var_width.get().strip():   args+=["--width", self.var_width.get().strip()]
        if self.var_height.get().strip():  args+=["--height", self.var_height.get().strip()]
        if self.var_fps.get().strip():     args+=["--fps", self.var_fps.get().strip()]

        args+=["--bg-volume", f"{self.var_bg_vol.get():.3f}",
               "--clip-volume", f"{self.var_clip_vol.get():.3f}",
               "--clip-reverb", f"{self.var_clip_reverb.get():.3f}"]

        # BPM-Args exklusiv
        if self.var_bpm_detect.get():
            args+=["--bpm-detect"]
        elif self.var_bpm.get().strip():
            args+=["--bpm", self.var_bpm.get().strip()]

        args+=["--min-beats", str(self.var_min_beats.get()),
               "--max-beats", str(self.var_max_beats.get()),
               "--beat-mode", f"{self.var_beat_mode.get():.3f}"]

        if self.var_min_seconds.get().strip(): args+=["--min-seconds", self.var_min_seconds.get().strip()]
        if self.var_max_seconds.get().strip(): args+=["--max-seconds", self.var_max_seconds.get().strip()]
        if self.var_codec.get().strip():       args+=["--codec", self.var_codec.get().strip()]
        if self.var_audio_codec.get().strip(): args+=["--audio-codec", self.var_audio_codec.get().strip()]
        if self.var_preset.get().strip():      args+=["--preset", self.var_preset.get().strip()]
        if self.var_bitrate.get().strip():     args+=["--bitrate", self.var_bitrate.get().strip()]
        if self.var_threads.get().strip():     args+=["--threads", self.var_threads.get().strip()]
        return args

    # ---------- File pickers ----------
    def _browse_audio(self):
        f=filedialog.askopenfilename(title="Select audio",
            filetypes=[("Audio","*.mp3 *.wav *.flac *.m4a"),("All files","*.*")])
        if f:
            f=_norm(f); self.var_audio.set(f)
            if not self.var_output.get():
                self.var_output.set(_norm(os.path.splitext(f)[0]+".mp4"))

    def _browse_videos(self):
        current = self.var_videos.get()
        if not current or not os.path.isdir(current):
            default_videos = os.path.join(os.getcwd())
            if os.path.isdir(default_videos):
                current = default_videos
            else:
                current = os.getcwd()

        d = filedialog.askdirectory(
            title="Select video folder",
            initialdir=current
        )
        if d:
            self.var_videos.set(_norm(d))

    def _browse_output(self):
        f=filedialog.asksaveasfilename(title="Select output", defaultextension=".mp4",
            filetypes=[("MP4","*.mp4"),("All files","*.*")])
        if f: self.var_output.set(_norm(f))

if __name__ == "__main__":
    app=PMVeaverGUI()
    app.mainloop()
