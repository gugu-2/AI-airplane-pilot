@echo off
echo =======================================================
echo Building High-Performance C++ EKF Engine (Windows DLL)
echo =======================================================

cd %~dp0\..\src\cpp_core

echo Compiling fast_ekf.cpp into fast_ekf.dll...
:: Check if g++ is installed
where g++ >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] g++ compiler not found!
    echo Please install MinGW or use WSL to compile the Linux version.
    echo The Python system will safely fallback to Numpy.
    exit /b 1
)

g++ -O3 -shared -fPIC fast_ekf.cpp -o fast_ekf.dll

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] fast_ekf.dll compiled successfully!
    echo The Python EKF will now automatically use the C++ backend.
) else (
    echo [ERROR] Compilation failed.
)
pause
