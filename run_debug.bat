@echo off
setlocal
REM Show Qt plugin loading messages (very helpful for PySide6 issues)
set QT_DEBUG_PLUGINS=1
REM Avoid MKL/OMP duplicate warnings killing the process
set KMP_DUPLICATE_LIB_OK=TRUE

cd /d "%~dp0\dist\AlgaeCounter"
AlgaeCounter.exe
echo.
echo ---- Exit code: %ERRORLEVEL% ----
pause
