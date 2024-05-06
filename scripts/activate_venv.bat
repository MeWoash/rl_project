@echo off
SET "CURRENT_DIR=%~dp0"
SET "WORKSPACE=%CURRENT_DIR%.."
for /f "delims=" %%i in ("%WORKSPACE%") do set "WORKSPACE=%%~fi"

SET "PYTHON_VENV=%WORKSPACE%/.venv/Scripts/python.exe"
CALL %WORKSPACE%/.venv/Scripts/activate.bat
