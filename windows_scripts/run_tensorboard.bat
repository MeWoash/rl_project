@echo off
CALL activate_venv.bat
@echo on

start chrome http://localhost:6006/
tensorboard --logdir %WORKSPACE%\out\learning --port=6006
