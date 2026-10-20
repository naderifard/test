@echo off
setlocal
set "PY=%USERPROFILE%\anaconda3\envs\agent\python.exe"
if not exist "%PY%" set "PY=python"
cd /d "%~dp0"
"%PY%" main.py %*
