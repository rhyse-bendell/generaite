@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY_EXE=.venv\Scripts\python.exe"
) else (
  set "PY_EXE=python"
)

%PY_EXE% gui\analysis_launcher.py
if errorlevel 1 (
  echo.
  echo Launcher failed. Ensure Python and required packages are installed.
  pause
)
