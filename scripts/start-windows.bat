@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0..\"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "REQUIREMENTS_FILE=%SCRIPT_DIR%requirements.txt"
set "SWINGSIGHT_OPEN_BROWSER=true"
cd /d "%SCRIPT_DIR%"

if not exist "%REQUIREMENTS_FILE%" (
    echo SwingSight could not find requirements.txt.
    pause
    exit /b 1
)

if not exist "%VENV_PYTHON%" (
    echo Setting up SwingSight for the first time...
    call :create_venv
    if errorlevel 1 (
        pause
        exit /b 1
    )
)

if not exist "%VENV_PYTHON%" (
    echo SwingSight could not create its Python environment.
    pause
    exit /b 1
)

echo Checking required Python packages...
"%VENV_PYTHON%" -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 (
    echo SwingSight could not update pip.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -m pip install --disable-pip-version-check -r "%REQUIREMENTS_FILE%"
if errorlevel 1 (
    echo SwingSight could not install its required packages.
    pause
    exit /b 1
)

echo Starting SwingSight...
"%VENV_PYTHON%" "%SCRIPT_DIR%src\run.py"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%

:create_venv
where py >nul 2>&1
if not errorlevel 1 (
    py -3 -m venv "%VENV_DIR%"
    if not errorlevel 1 exit /b 0
)

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info.major == 3 else 1)" >nul 2>&1
    if not errorlevel 1 (
        python -m venv "%VENV_DIR%"
        if not errorlevel 1 exit /b 0
    )
)

echo Python 3 is required. Install it from https://www.python.org/downloads/
exit /b 1
