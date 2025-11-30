@echo off
echo ========================================
echo   Skincare Recommendation System
echo   (Manual Installation Method)
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if Python is installed
py --version >nul 2>&1
if errorlevel 1 (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ❌ Python tidak ditemukan!
        echo Silakan install Python terlebih dahulu dari https://python.org
        echo Pastikan centang "Add Python to PATH" saat install
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=python
    )
) else (
    set PYTHON_CMD=py
)

echo ✅ Python ditemukan: %PYTHON_CMD%
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Membuat virtual environment...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo ❌ Gagal membuat virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment berhasil dibuat
) else (
    echo ✅ Virtual environment sudah ada
)

REM Install dependencies using venv python directly
echo 📋 Menginstall dependencies...
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Gagal install dependencies
    echo Coba jalankan manual: venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
echo ✅ Dependencies berhasil diinstall

REM Check database configuration
echo 🗄️ Memeriksa konfigurasi database...
if not exist ".env" (
    echo ❌ File .env tidak ditemukan!
    echo Silakan copy .env.example ke .env dan sesuaikan konfigurasi
    pause
    exit /b 1
)

echo.
echo 🚀 Menjalankan aplikasi...
echo 📱 Aplikasi akan tersedia di: http://localhost:5000
echo 🌐 Atau akses dari perangkat lain: http://192.168.1.42:5000
echo 🛑 Tekan Ctrl+C untuk menghentikan aplikasi
echo.

REM Run the application using venv python directly
venv\Scripts\python.exe app.py

echo.
echo 👋 Aplikasi telah dihentikan
pause