# QuadTree - High-Performance Spatial Data Structure

A modern C++17 implementation of a spatial quadtree data structure for Python, optimized for fast 2D point operations.

![QuadTree Security](images/Background%20Color%20-%20no%20preview%20-%20square.jpg)

> **📖 Read the Full Story**: This project is featured in the article [*"AI-Powered Data Structures: Building High-Performance QuadTree with Modern C++17"*](https://nicoliberato.substack.com/p/ai-powered-data-structures-building) on Substack, which details the development process, AI-assisted coding techniques, and architectural decisions behind this implementation.

> **🚀 PyPI Ready**: This package is prepared for publication on PyPI but not yet uploaded. See the [Publishing to PyPI](#publishing-to-pypi) section for instructions.

## Features

- **Fast Operations**: Efficient point insertion, querying, and spatial range searches
- **Collision Detection**: Built-in collision detection with configurable radius
- **Memory Efficient**: Modern C++17 with smart pointers and RAII
- **Python Integration**: Seamless integration with Python data types
- **Production Ready**: Comprehensive test suite and memory management
- **Cross-Platform**: Works on Linux, macOS, and Windows
- **Security First**: Comprehensive vulnerability scanning with Trivy

## Installation

```bash
pip install quadtree
```

Or build from source:

```bash
git clone <repository-url>
cd quadtree_package
pip install .
```

## Quick Start

```python
import quadtree

# Create a quadtree with boundary (x, y, width, height)
qt = quadtree.QuadTree(0, 0, 100, 100)

# Insert points with optional data
qt.insert(10, 20, "point1")
qt.insert(30, 40, {"id": 123, "value": "data"})
qt.insert(50, 60)

# Query rectangular region
points = qt.query(0, 0, 50, 50)
print(f"Found {len(points)} points in region")

# Check if point exists
if qt.contains(10, 20):
    print("Point found!")

# Collision detection
collisions = qt.detect_collisions(radius=15.0)
print(f"Found {len(collisions)} collision pairs")

# Get tree statistics
print(f"Tree contains {qt.size()} points")
print(f"Tree depth: {qt.depth()}")
print(f"Subdivisions: {qt.subdivisions()}")
```

## API Reference

### QuadTree(x, y, width, height)

Create a new quadtree with the specified boundary.

**Parameters:**
- `x`, `y`: Bottom-left corner coordinates
- `width`, `height`: Dimensions of the quadtree boundary

### Methods

- `insert(x, y, data=None)`: Insert a point with optional data
- `query(x, y, width, height)`: Find all points in rectangular region
- `contains(x, y)`: Check if point exists in the tree
- `get_all_points()`: Get all points in the tree
- `detect_collisions(radius)`: Find point pairs within radius distance
- `size()`: Get number of points in the tree
- `empty()`: Check if tree is empty
- `boundary()`: Get tree boundary as (x, y, width, height)
- `depth()`: Get maximum tree depth
- `subdivisions()`: Get number of subdivisions

## Performance

The QuadTree is optimized for:
- **Insertion**: O(log n) average case
- **Query**: O(log n + k) where k is the number of results
- **Memory**: Efficient with automatic cleanup

See the `tests/` directory for comprehensive performance benchmarks.

## Security

### 🔒 Vulnerability Scanning with Trivy

This project implements comprehensive security scanning using [Trivy](https://trivy.dev/), an industry-standard vulnerability scanner. Our security infrastructure includes:

#### Automated Security Pipeline
- **Filesystem Scanning**: Detects vulnerabilities in source code and dependencies
- **Container Scanning**: Scans Docker images for OS and application vulnerabilities
- **Secret Detection**: Prevents accidental exposure of API keys, passwords, and tokens
- **Misconfiguration Detection**: Identifies security configuration issues
- **License Compliance**: Ensures dependency licenses meet requirements

#### Multi-layered Security Testing
```bash
# Run comprehensive security scan locally
./scripts/security_scan.sh

# Run security vulnerability tests
python tests/test_security_vulnerabilities.py

# Run all tests including security
python tests/run_comprehensive_tests.py
```

#### CI/CD Security Integration
Our GitHub Actions pipeline automatically:
- Scans every commit and pull request for vulnerabilities
- Uploads results to GitHub Security tab for centralized monitoring
- Fails builds on critical security issues
- Generates detailed security reports

#### Security Features
- **Memory Safety**: Modern C++17 smart pointers prevent memory leaks
- **Input Validation**: Comprehensive boundary and type checking
- **Error Handling**: Secure error messages without information disclosure
- **SARIF Format**: Industry-standard security report format
- **Daily Scans**: Scheduled vulnerability monitoring

#### Compliance Standards
- ✅ OWASP Top 10 security practices
- ✅ Supply chain security best practices  
- ✅ Secret management security standards
- ✅ Container security guidelines

For detailed security information, see [SECURITY.md](SECURITY.md).

## Publishing to PyPI

This package is ready for publication to the Python Package Index (PyPI). 

### Prerequisites
```bash
# Install publishing tools
pip install build twine

# Ensure you have PyPI credentials configured
# Option 1: Configure in ~/.pypirc
# Option 2: Use environment variables TWINE_USERNAME and TWINE_PASSWORD
# Option 3: Use API tokens (recommended)
```

### Build and Upload Process
```bash
# 1. Clean previous builds
rm -rf build/ dist/ *.egg-info/

# 2. Build the package
python -m build

# 3. Check the distribution
twine check dist/*

# 4. Test upload to PyPI test repository (recommended first)
twine upload --repository testpypi dist/*

# 5. Test installation from test PyPI
pip install --index-url https://test.pypi.org/simple/ quadtree

# 6. If everything works, upload to production PyPI
twine upload dist/*
```

### Post-Publication
```bash
# Verify the package is available
pip install quadtree

# Test the installed package
python -c "import quadtree; print('QuadTree successfully installed from PyPI')"
```

### Version Management
- Update version in `pyproject.toml` before each release
- Follow [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH)
- Tag releases in Git: `git tag v2.0.1 && git push --tags`

### Package Metadata
The package metadata is configured in `pyproject.toml` with:
- Project name: `quadtree`
- Current version: `2.0.0`
- Python compatibility: `>=3.6`
- License: MIT
- Keywords: spatial, data-structure, quadtree, collision-detection

## Development Story

This QuadTree implementation showcases modern AI-assisted software development techniques. The complete development journey, including:

- **AI-Powered Architecture Design**: How AI helped design optimal data structures
- **Modern C++17 Implementation**: Smart pointers, RAII, and exception safety
- **Python-C++ Integration**: Seamless binding with pybind11
- **Security-First Development**: Comprehensive vulnerability scanning with Trivy
- **Performance Optimization**: Memory efficiency and algorithmic improvements

...is documented in the comprehensive article: [**"AI-Powered Data Structures: Building High-Performance QuadTree with Modern C++17"**](https://nicoliberato.substack.com/p/ai-powered-data-structures-building)

## Requirements

- Python 3.6+
- C++17 compatible compiler
- CMake (for building from source)

## License

MIT License - see LICENSE file for details.
