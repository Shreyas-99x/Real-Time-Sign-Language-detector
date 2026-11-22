@echo off
title SignSpeak - Web Application
cd /d "%~dp0"
echo ========================================================
echo   Starting SignSpeak Local Server...
echo ========================================================
python serve.py
if %ERRORLEVEL% NEQ 0 (
    echo Python command failed, trying 'py'...
    py serve.py
)
if %ERRORLEVEL% NEQ 0 (
    echo Trying direct Python 3.12 path...
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" serve.py
)
pause
