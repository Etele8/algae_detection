
@echo off
title Build Algae Detector (MINIMAL)
setlocal

echo 1) Checking for Python...
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [ERROR] No Python found on PATH. Install Python 3.10+.
  pause
  exit /b 1
)
%PY% --version
echo.

echo 2) Checking that algae_detect_app.pyw is here...
if not exist "algae_detect_app.pyw" (
  echo [ERROR] algae_detect_app.pyw not found in this folder.
  echo Put this .bat in the SAME folder as algae_detect_app.pyw and run again.
  pause
  exit /b 1
)
echo Found algae_detect_app.pyw
echo.

echo 3) Creating local venv .venv ...
%PY% -m venv .venv
if errorlevel 1 (
  echo [ERROR] Failed to create venv.
  pause
  exit /b 1
)
call .venv\Scripts\activate
if errorlevel 1 (
  echo [ERROR] Failed to activate venv.
  pause
  exit /b 1
)
echo.

echo 4) Upgrading pip/setuptools/wheel ...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo [ERROR] pip upgrade failed.
  pause
  exit /b 1
)
echo.

echo 5) Installing dependencies (PyInstaller, Torch CPU, OpenCV, NumPy) ...
pip install pyinstaller==6.6.0 torch torchvision opencv-python numpy
if errorlevel 1 (
  echo [ERROR] Dependency install failed.
  pause
  exit /b 1
)
echo.

echo 6) Building ONEDIR app with PyInstaller ...
set "NAME=Algae Detector"
pyinstaller --noconfirm --noconsole --name "%NAME%" --onedir algae_detect_app.pyw
if errorlevel 1 (
  echo [ERROR] PyInstaller failed.
  pause
  exit /b 1
)
echo.

echo DONE!
echo Your app is at: dist\%NAME%\%NAME%.exe
echo Zip and share the whole folder "dist\%NAME%".
pause
