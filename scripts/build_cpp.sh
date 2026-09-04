#!/bin/bash
echo "======================================================="
echo "Building High-Performance C++ EKF Engine (Linux .so)"
echo "======================================================="

cd "$(dirname "$0")/../src/cpp_core"

echo "Compiling fast_ekf.cpp into fast_ekf.so..."

if ! command -v g++ &> /dev/null; then
    echo "[ERROR] g++ compiler not found!"
    echo "Run: sudo apt install build-essential"
    exit 1
fi

g++ -O3 -shared -fPIC fast_ekf.cpp -o fast_ekf.so

if [ $? -eq 0 ]; then
    echo "[SUCCESS] fast_ekf.so compiled successfully!"
    echo "The Python EKF will now automatically use the C++ backend."
else
    echo "[ERROR] Compilation failed."
fi
