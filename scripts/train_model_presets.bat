@echo off
CALL activate_venv.bat
@echo on

python "%WORKSPACE%\src\ModelTools\TrainModelPresets.py"
