@echo off
echo ========================================================
echo       AEGIS AUTONOMY: SWARM INTELLIGENCE LAUNCHER
echo ========================================================
echo.
echo Launching Drone Alpha (Primary Node with Telemetry Server)...
start "Aegis Drone Alpha" cmd /k "python src\main_pilot.py --drone_id Alpha"

echo Waiting 5 seconds for Alpha to initialize...
timeout /t 5 /nobreak >nul

echo Launching Drone Bravo (Secondary Node)...
start "Aegis Drone Bravo" cmd /k "python src\main_pilot.py --drone_id Bravo --disable_ws"

echo.
echo [!] Swarm deployed. Both drones are now running in parallel.
echo [!] They are communicating over UDP port 5555.
echo [!] You can track both drones on the React Dashboard (npm run dev).
echo.
pause
