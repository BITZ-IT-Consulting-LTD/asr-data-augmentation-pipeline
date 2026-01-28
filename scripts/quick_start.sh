#!/bin/bash
# Quick start script for ASR Data Augmentation Pipeline

set -e

echo "=========================================="
echo "ASR Data Augmentation Pipeline Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "✗ Error: config.yaml not found"
    echo "Please create config.yaml from the template"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit config.yaml with your settings"
echo "2. Run: python pipeline.py"
echo "3. Download audio: bash output/download_audio.sh"
echo ""
echo "For more information, see README.md"
echo "=========================================="
