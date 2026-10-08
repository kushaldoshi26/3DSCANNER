@echo off
title 3DINVENZA AI Scanner Server

echo ========================================================
echo               3DINVENZA AI SCANNER
echo ========================================================
echo.
echo Server is starting up...
echo You can access the application at: 
echo.
echo    http://localhost:8000
echo.
echo ========================================================
echo.

:: Open the default browser to the localhost link
echo Opening browser...
start http://localhost:8000

:: Start the Python backend server
python api\main.py

pause
