@echo off
REM Wrapper to run Hunyuan3D generate.py using the venv Python.
REM Usage: hy3d.bat --image path/to/image.png [options]
"%~dp0venv\Scripts\python.exe" "%~dp0generate.py" %*
