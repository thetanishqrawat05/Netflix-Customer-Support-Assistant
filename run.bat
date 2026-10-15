@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment - first run only...
    python -m venv venv || goto :error
    "venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
    echo [setup] Installing dependencies - first run only...
    "venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo [setup] Created .env from .env.example - put your GROQ_API_KEY in it, then run this again.
    exit /b 1
)

"venv\Scripts\python.exe" -m streamlit run app.py %*
exit /b %errorlevel%

:error
echo [setup] Failed. Make sure Python 3.10+ is installed and on PATH.
exit /b 1
