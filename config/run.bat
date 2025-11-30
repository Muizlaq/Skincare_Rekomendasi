@echo off
echo ========================================
echo   Skincare Recommendation System
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

echo ✅ Python ditemukan
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
)

REM Activate virtual environment
echo 🔄 Mengaktifkan virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Gagal mengaktifkan virtual environment
    pause
    exit /b 1
)

REM Check if requirements are installed
echo 📋 Memeriksa dependencies...
%PYTHON_CMD% -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo 📥 Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Gagal install dependencies
        echo Coba jalankan: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo ✅ Dependencies berhasil diinstall
) else (
    echo ✅ Dependencies sudah terinstall
)

REM Check database configuration
echo 🗄️ Memeriksa konfigurasi database...
if not exist ".env" (
    echo ❌ File .env tidak ditemukan!
    echo Silakan copy .env.example ke .env dan sesuaikan konfigurasi
    pause
    exit /b 1
)

REM Initialize database if needed
%PYTHON_CMD% -c "from config import DatabaseConfig; DatabaseConfig.get_connection()" >nul 2>&1
if errorlevel 1 (
    echo 🔧 Inisialisasi database...
    %PYTHON_CMD% -c "from config import DatabaseConfig; DatabaseConfig.init_database()"
)

echo.
echo 🚀 Menjalankan aplikasi...
echo 📱 Aplikasi akan tersedia di: http://localhost:5000
echo 🛑 Tekan Ctrl+C untuk menghentikan aplikasi
echo.

REM Run the application
%PYTHON_CMD% app.py

echo.
echo 👋 Aplikasi telah dihentikan
pause