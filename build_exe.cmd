@echo off
setlocal
cd /d "%~dp0"
if not defined UNTORN_PYTHON set "UNTORN_PYTHON=python"
"%UNTORN_PYTHON%" -m venv .build-venv
if errorlevel 1 exit /b 1
".build-venv\Scripts\python.exe" -m pip install -r requirements-build.txt
if errorlevel 1 exit /b 1
".build-venv\Scripts\python.exe" build_portable.py
exit /b %errorlevel%
