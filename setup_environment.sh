#!/bin/bash

# LLM Code Search - Environment Setup Script
# This script ensures proper virtual environment setup and dependency installation

set -e  # Exit on any error

echo "🚀 Setting up LLM Code Search Environment..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required system dependencies
echo "🔍 Checking system dependencies..."

if ! command_exists python3; then
    echo "❌ Error: python3 not found. Please install Python 3.8+."
    exit 1
fi

if ! command_exists pip3; then
    echo "❌ Error: pip3 not found. Please install pip."
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ required. Found: $python_version"
    echo "💡 Please install a newer version of Python."
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Check if virtual environment already exists
if [ -d ".venv" ]; then
    echo "⚠️  Virtual environment '.venv' already exists."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🧹 Removing existing virtual environment..."
        rm -rf .venv
    else
        echo "✅ Using existing virtual environment."
        echo "💡 To activate it manually: source .venv/bin/activate"
        
        # Check if already activated
        if [[ "$VIRTUAL_ENV" != "" ]]; then
            echo "✅ Virtual environment already activated."
        else
            echo "⚡ Activating virtual environment..."
            source .venv/bin/activate
        fi
        
        # Check if dependencies are installed
        echo "🔍 Checking existing dependencies..."
        if python3 -c "import flask, transformers, chromadb" >/dev/null 2>&1; then
            echo "✅ Core dependencies already installed."
            echo "🎉 Environment is ready!"
            echo ""
            echo "To activate: source .venv/bin/activate"
            echo "To start: python main.py"
            exit 0
        else
            echo "⚠️  Some dependencies missing. Will install..."
        fi
    fi
fi

# Create fresh virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating fresh virtual environment..."
    if ! python3 -m venv .venv; then
        echo "❌ Error: Failed to create virtual environment."
        echo "💡 Try: python3.9 -m venv .venv or python3.8 -m venv .venv"
        exit 1
    fi
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source .venv/bin/activate

# Verify activation
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Error: Failed to activate virtual environment."
    exit 1
fi

echo "✅ Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip
echo "📈 Upgrading pip..."
if ! python3 -m pip install --upgrade pip; then
    echo "⚠️  Warning: Failed to upgrade pip, continuing with current version..."
fi

# Install dependencies
echo "📚 Installing dependencies from requirements.txt..."
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found in current directory."
    exit 1
fi

if ! pip install -r requirements.txt; then
    echo "❌ Error: Failed to install dependencies."
    echo "💡 Try running: pip install --upgrade pip && pip install -r requirements.txt"
    exit 1
fi

# Verify installation
echo "🔍 Verifying installation..."
if ! python3 -c "
try:
    import flask, transformers, chromadb, sentence_transformers, fuzzywuzzy
    import torch, numpy, yaml, tree_sitter, tree_sitter_cpp, tqdm
    print('✅ All core dependencies installed successfully!')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"; then
    echo "❌ Error: Some dependencies failed to install properly."
    exit 1
fi

# Test core modules
echo "🔧 Testing core modules..."
if ! python3 -c "
try:
    import sys
    import os
    sys.path.append('core')
    from config import load_config, validate_config
    from search import search_keyword
    print('✅ Core modules load successfully!')
except ImportError as e:
    print(f'⚠️  Warning: Core module test failed: {e}')
    print('This may be normal if you haven\\'t run the initial setup yet.')
"; then
    echo "⚠️  Core modules test had issues, but this may be normal for first-time setup."
fi

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate environment: source .venv/bin/activate"
echo "2. Check configuration: python main.py --config"
echo "3. Run setup: python main.py --setup"
echo "4. Start web interface: python main.py"
echo "5. Access at: http://localhost:8081"
echo ""
echo "Alternative options:"
echo "  python main.py --setup    # Run setup + start web"
echo "  python main.py --test     # Test functionality"
echo "  python migrate_config.py  # Migrate old configs"
echo ""
echo "💡 If you encounter issues:"
echo "  - Ensure Python 3.8+ is installed"
echo "  - Check you have 4GB+ RAM available"
echo "  - Verify internet connection for model downloads"
