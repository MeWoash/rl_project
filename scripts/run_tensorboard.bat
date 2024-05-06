CALL activate_venv.bat

start chrome http://localhost:6006/
tensorboard --logdir %WORKSPACE%\out\learning --port=6006
