@echo off
setlocal
set VENV=.venv-build

REM venv anlegen/verwenden
if not exist "%VENV%\Scripts\python.exe" (
  py -3.10 -m venv "%VENV%"
)
call "%VENV%\Scripts\activate"

python -m pip install -U pip wheel setuptools
python -m pip install -r requirements.txt "PyInstaller>=6.0"

REM einmalig cleanen, dann BEIDE bauen (ohne zwischendurch dist löschen!)
rmdir /s /q build dist 2>nul

echo === Building pmveaver.spec ===
python -m PyInstaller -y --clean --log-level=WARN pmveaver.spec       || goto :error

echo === Building pmveaver_gui.spec ===
python -m PyInstaller -y --clean --log-level=WARN pmveaver_gui.spec   || goto :error

REM --- ffmpeg/ffprobe ins dist kopieren ---
copy /Y ffmpeg.exe  dist
copy /Y ffprobe.exe dist

echo.
echo [OK] Build fertig. EXEs liegen in .\dist\
exit /b 0

:error
echo.
echo [ERROR] Build fehlgeschlagen. Siehe Ausgabe oben.
exit /b 1
