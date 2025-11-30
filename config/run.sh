#!/bin/bash

echo "========================================"
echo "   Skincare Recommendation System"
echo "========================================"
echo

# Change to script directory
cd "$(dirname "$0")"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 tidak ditemukan!"
    echo "Silakan install Python3 terlebih dahulu"
    echo "Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "macOS: brew install python"
    exit 1
fi

echo "✅ Python3 ditemukan"
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Membuat virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Gagal membuat virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment berhasil dibuat"
fi

# Activate virtual environment
echo "🔄 Mengaktifkan virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "❌ Gagal mengaktifkan virtual environment"
    exit 1
fi

# Check if requirements are installed
echo "📋 Memeriksa dependencies..."
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Gagal install dependencies"
        echo "Coba jalankan: pip install -r requirements.txt"
        exit 1
    fi
    echo "✅ Dependencies berhasil diinstall"
else
    echo "✅ Dependencies sudah terinstall"
fi

# Check database configuration
echo "🗄️ Memeriksa konfigurasi database..."
if [ ! -f ".env" ]; then
    echo "❌ File .env tidak ditemukan!"
    echo "Silakan copy .env.example ke .env dan sesuaikan konfigurasi"
    exit 1
fi

# Initialize database if needed
python -c "from config import DatabaseConfig; DatabaseConfig.get_connection()" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "🔧 Inisialisasi database..."
    python -c "from config import DatabaseConfig; DatabaseConfig.init_database()"
fi

echo
echo "🚀 Menjalankan aplikasi..."
echo "📱 Aplikasi akan tersedia di: http://localhost:5000"
echo "🛑 Tekan Ctrl+C untuk menghentikan aplikasi"
echo

# Run the application
python app.py

echo
echo "👋 Aplikasi telah dihentikan"