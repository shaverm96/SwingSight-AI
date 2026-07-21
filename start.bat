@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "VENV_PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
cd /d "%SCRIPT_DIR%"

if not exist "%VENV_PYTHON%" (
    echo Setting up SwingSight for the first time...

    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv "%SCRIPT_DIR%.venv"
    ) else (
        where python >nul 2>&1
        if not errorlevel 1 (
            python -m venv "%SCRIPT_DIR%.venv"
        ) else (
            echo Python 3 is required. Install it from https://www.python.org/downloads/
            pause
            exit /b 1
        )
    )

    if not exist "%VENV_PYTHON%" (
        echo SwingSight could not create its Python environment.
        pause
        exit /b 1
    )

    "%VENV_PYTHON%" -m pip install --upgrade pip
    "%VENV_PYTHON%" -m pip install -r "%SCRIPT_DIR%requirements.txt"
    if errorlevel 1 (
        echo SwingSight could not install its required packages.
        pause
        exit /b 1
    )
)

"%VENV_PYTHON%" "%SCRIPT_DIR%src\run.py"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
