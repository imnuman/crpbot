@echo off
REM HYDRA MT5 Executor - Startup Script
REM Run this after MT5 Terminal is open and logged in

cd /d C:\HYDRA

REM Set environment variables (CHANGE THESE!)
set FTMO_LOGIN=531025383
set FTMO_PASS=c*B@lWp41b784c
set FTMO_SERVER=FTMO-Server3
set API_SECRET=hydra_secret_2024

echo ============================================
echo    HYDRA MT5 Executor Starting...
echo ============================================
echo.

python executor_service.py

echo.
echo Executor stopped. Press any key to exit.
pause
