#!/bin/bash
# Build script for QuadTree package

set -e

echo "Building QuadTree package..."

# Change to package directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build the package
echo "Running setup.py build..."
python3 setup.py build

echo "Building wheel..."
python3 setup.py bdist_wheel

echo "Building source distribution..."
python3 setup.py sdist

echo "Build complete!"
echo "Distribution files are in the dist/ directory"
ls -la dist/

echo ""
echo "To install locally:"
echo "  pip install dist/*.whl"
echo ""
echo "To upload to PyPI (after setting up credentials):"
echo "  twine upload dist/*"