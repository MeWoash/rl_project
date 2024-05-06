@REM ARG1: requirement path
@REM ARG2: created env path
@REM USAGE: .\create_venv.bat requirements-cuda.txt .venv

SET "VENV_PATH=%2"
CALL python -m venv %VENV_PATH%
CALL "%VENV_PATH%\Scripts\activate.bat"

CALL python -m pip install --upgrade pip

CALL pip install -r %1
