@echo off
setlocal
cd /d "%~dp0"

set "NO_PAUSE="
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"

echo [Gener-AI-te] Preparing local Python environment...

if not exist "requirements.txt" (
  echo [ERROR] requirements.txt not found in repo root.
  goto :fail
)

if exist ".venv\Scripts\python.exe" (
  set "BOOTSTRAP_PY=.venv\Scripts\python.exe"
) else (
  set "BOOTSTRAP_PY="
  where py >nul 2>nul && set "BOOTSTRAP_PY=py -3"
  if not defined BOOTSTRAP_PY (
    where python >nul 2>nul && set "BOOTSTRAP_PY=python"
  )
)

if not exist ".venv\Scripts\python.exe" (
  if not defined BOOTSTRAP_PY (
    echo [ERROR] Could not find a Python interpreter to create .venv.
    echo         Install Python 3 and rerun this script.
    goto :fail
  )
  echo [Gener-AI-te] Creating .venv...
  call %BOOTSTRAP_PY% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create .venv.
    goto :fail
  )
)

set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo [ERROR] .venv was expected but %VENV_PY% was not found.
  goto :fail
)

echo [Gener-AI-te] Upgrading pip in .venv...
call "%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip.
  goto :fail
)

echo [Gener-AI-te] Installing required packages from requirements.txt...
call "%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Failed to install requirements.
  goto :fail
)

echo [Gener-AI-te] Environment setup complete.
goto :end

:fail
echo.
echo Setup failed.
if not defined NO_PAUSE pause
exit /b 1

:end
if not defined NO_PAUSE pause
exit /b 0
