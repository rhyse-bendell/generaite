@echo off
setlocal
cd /d "%~dp0"

echo [Gener-AI-te] Bootstrapping local environment...
call setup_generaite_env.bat --no-pause
if errorlevel 1 (
  echo.
  echo [ERROR] Environment setup failed. GUI was not launched.
  pause
  exit /b 1
)

set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo [ERROR] Expected virtual environment Python not found at %VENV_PY%.
  pause
  exit /b 1
)

echo [Gener-AI-te] Launching GUI...
call "%VENV_PY%" -m gui.analysis_launcher
if errorlevel 1 (
  echo.
  echo [ERROR] GUI launch failed.
  pause
  exit /b 1
)
