@echo off
cd /d "%~dp0"
if not exist "venv" (
    echo Setting up virtual environment and dependencies (first-time only)...
    python -m venv venv
    call venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)
streamlit run dashboard\app.py
pause
