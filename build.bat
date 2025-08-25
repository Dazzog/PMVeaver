@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ---- Python-Launcher bestimmen ----
where py >nul 2>&1
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  set "PY=python"
)

REM ---- .venv erzeugen, falls nicht vorhanden ----
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Erstelle Virtualenv unter .venv ...
  %PY% -m venv .venv || goto :error
)

REM ---- venv aktivieren ----
call ".venv\Scripts\activate" || goto :error

REM ---- Tools & Deps aktualisieren ----
python -m ensurepip -U >nul 2>&1
python -m pip install -q -U pip wheel setuptools || goto :error

REM ---- Projekt-Abhaengigkeiten ----
if exist requirements.txt (
  python -m pip install -q -r requirements.txt || goto :error
) else (
  echo [WARN] requirements.txt nicht gefunden – ueberspringe Installation.
)

REM ---- PyInstaller installieren ----
python -m pip install -q "pyinstaller>=6,<7" || goto :error

REM ---- Builds (sauber, ohne Rueckfragen) ----
pyinstaller -y --clean --log-level WARN random_clip_montage.spec || goto :error
pyinstaller -y --clean --log-level WARN montage_gui.spec || goto :error

echo.
echo [OK] Build fertig. EXEs liegen in .\dist\
exit /b 0

:error
echo.
echo [ERROR] Build fehlgeschlagen. Siehe Ausgabe oben.
exit /b 1
