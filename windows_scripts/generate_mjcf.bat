@echo off
CALL activate_venv.bat
@echo on
python "%WORKSPACE%\src\MJCFGenerator\Generator.py" && echo Generated
