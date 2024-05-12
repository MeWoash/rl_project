@echo off
CALL activate_venv.bat
@echo on
python -m mujoco.viewer --mjcf="%WORKSPACE%\out\mjcf\out.xml"
