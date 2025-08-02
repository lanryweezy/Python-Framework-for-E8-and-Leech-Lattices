#!/bin/bash
# E8 Leech Lattice Framework - Installation Script

set -e

echo "E8 Leech Lattice Framework - Installation"
echo "========================================"

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="$(dirname "$SCRIPT_DIR")"

echo "Framework directory: $FRAMEWORK_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "$FRAMEWORK_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$FRAMEWORK_DIR/venv"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$FRAMEWORK_DIR/venv/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the framework in development mode
echo "Installing E8 Leech Framework..."
cd "$FRAMEWORK_DIR"
pip install -e .

# Install optional dependencies based on user choice
echo ""
echo "Optional dependencies:"
echo "1. GPU support (CuPy)"
echo "2. Quantum computing (Qiskit, Cirq)"
echo "3. Visualization (Plotly, Dash)"
echo "4. Development tools (pytest, black, etc.)"
echo "5. All optional dependencies"
echo "6. Skip optional dependencies"

read -p "Choose option (1-6): " choice

case $choice in
    1)
        echo "Installing GPU support..."
        pip install -e ".[gpu]"
        ;;
    2)
        echo "Installing quantum computing support..."
        pip install -e ".[quantum]"
        ;;
    3)
        echo "Installing visualization support..."
        pip install -e ".[visualization]"
        ;;
    4)
        echo "Installing development tools..."
        pip install -e ".[dev]"
        ;;
    5)
        echo "Installing all optional dependencies..."
        pip install -e ".[all,dev]"
        ;;
    6)
        echo "Skipping optional dependencies..."
        ;;
    *)
        echo "Invalid choice. Skipping optional dependencies..."
        ;;
esac

# Create symlink for CLI if not exists
CLI_LINK="/usr/local/bin/e8leech"
if [ ! -L "$CLI_LINK" ] && [ -w "/usr/local/bin" ]; then
    echo "Creating CLI symlink..."
    sudo ln -sf "$FRAMEWORK_DIR/scripts/e8leech" "$CLI_LINK"
elif [ -w "$HOME/.local/bin" ]; then
    echo "Creating CLI symlink in user directory..."
    mkdir -p "$HOME/.local/bin"
    ln -sf "$FRAMEWORK_DIR/scripts/e8leech" "$HOME/.local/bin/e8leech"
    echo "Add $HOME/.local/bin to your PATH if not already present"
fi

# Run basic tests
echo ""
echo "Running basic tests..."
cd "$FRAMEWORK_DIR"
python3 -c "
try:
    from e8leech.lattices.e8_lattice import E8Lattice
    from e8leech.lattices.leech_lattice import LeechLattice
    print('✓ Core lattice modules imported successfully')
except ImportError as e:
    print(f'⚠ Warning: Some modules not available: {e}')

try:
    import numpy as np
    print('✓ NumPy available')
except ImportError:
    print('✗ NumPy not available')

try:
    import numba
    print('✓ Numba available')
except ImportError:
    print('⚠ Numba not available (performance may be reduced)')
"

echo ""
echo "Installation completed!"
echo ""
echo "Usage:"
echo "  Activate environment: source $FRAMEWORK_DIR/venv/bin/activate"
echo "  CLI help: e8leech --help"
echo "  Python import: from e8leech.lattices.e8_lattice import E8Lattice"
echo ""
echo "Examples:"
echo "  e8leech lattice info"
echo "  e8leech lattice quantize --lattice e8 --vector '1,2,3,4,5,6,7,8'"
echo "  e8leech crypto encrypt --algo LWE --message 'Hello World'"
echo ""

