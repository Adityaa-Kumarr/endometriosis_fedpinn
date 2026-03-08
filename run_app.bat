@echo off
REM Run the Streamlit app using the venv on H: (no C: drive usage for packages).
set PROJECT_ROOT=%~dp0
set VENV_PYTHON=%PROJECT_ROOT%venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo Virtual environment not found. Create it first:
    echo   py -m venv "%PROJECT_ROOT%venv"
    echo   "%PROJECT_ROOT%venv\Scripts\activate.bat"
    echo   pip install -r "%PROJECT_ROOT%requirements.txt"
    exit /b 1
)

cd /d "%PROJECT_ROOT%"
"%VENV_PYTHON%" -m streamlit run app.py %*
