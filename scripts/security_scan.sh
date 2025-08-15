#!/bin/bash

# QuadTree Package Security Vulnerability Scanner
# Uses Trivy to scan for vulnerabilities in code, dependencies, and containers

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TRIVY_CACHE_DIR="${HOME}/.cache/trivy"
REPORT_DIR="./security-reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Trivy is installed
check_trivy_installation() {
    if ! command -v trivy &> /dev/null; then
        print_error "Trivy is not installed. Installing Trivy..."
        
        # Install Trivy based on OS
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux installation
            sudo apt-get update
            sudo apt-get install wget apt-transport-https gnupg lsb-release
            wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
            echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
            sudo apt-get update
            sudo apt-get install trivy
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS installation
            if command -v brew &> /dev/null; then
                brew install trivy
            else
                print_error "Homebrew not found. Please install Trivy manually: https://aquasecurity.github.io/trivy/latest/getting-started/installation/"
                exit 1
            fi
        else
            print_error "Unsupported OS. Please install Trivy manually: https://aquasecurity.github.io/trivy/latest/getting-started/installation/"
            exit 1
        fi
        
        print_success "Trivy installed successfully"
    else
        print_success "Trivy is already installed"
        trivy --version
    fi
}

# Function to create report directory
create_report_directory() {
    mkdir -p "${REPORT_DIR}"
    print_status "Created report directory: ${REPORT_DIR}"
}

# Function to scan filesystem for vulnerabilities
scan_filesystem() {
    print_status "Scanning filesystem for vulnerabilities..."
    
    trivy fs . \
        --format json \
        --output "${REPORT_DIR}/filesystem_scan_${TIMESTAMP}.json" \
        --severity CRITICAL,HIGH,MEDIUM,LOW \
        --scanners vuln,secret,misconfig
    
    # Also generate human-readable report
    trivy fs . \
        --format table \
        --output "${REPORT_DIR}/filesystem_scan_${TIMESTAMP}.txt" \
        --severity CRITICAL,HIGH,MEDIUM,LOW \
        --scanners vuln,secret,misconfig
    
    print_success "Filesystem scan completed"
}

# Function to scan Python dependencies
scan_python_dependencies() {
    print_status "Scanning Python dependencies for vulnerabilities..."
    
    # Create a temporary requirements file if it doesn't exist
    if [ ! -f requirements.txt ]; then
        print_warning "No requirements.txt found. Creating temporary one from pyproject.toml..."
        if [ -f pyproject.toml ]; then
            # Extract dependencies from pyproject.toml (basic extraction)
            python3 -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
    deps = data.get('project', {}).get('dependencies', [])
    if deps:
        with open('temp_requirements.txt', 'w') as req_file:
            for dep in deps:
                req_file.write(dep + '\n')
" 2>/dev/null || echo "setuptools>=45" > temp_requirements.txt
            REQ_FILE="temp_requirements.txt"
        else
            echo "setuptools>=45" > temp_requirements.txt
            REQ_FILE="temp_requirements.txt"
        fi
    else
        REQ_FILE="requirements.txt"
    fi
    
    # Scan using pip-audit if available, otherwise use trivy
    if command -v pip-audit &> /dev/null; then
        print_status "Using pip-audit for dependency scanning..."
        pip-audit --format=json --output="${REPORT_DIR}/python_deps_${TIMESTAMP}.json" || true
        pip-audit --format=cyclonedx --output="${REPORT_DIR}/python_deps_${TIMESTAMP}.xml" || true
    fi
    
    # Always use trivy as well
    if [ -f "${REQ_FILE}" ]; then
        trivy fs . \
            --scanners vuln \
            --format json \
            --output "${REPORT_DIR}/trivy_python_deps_${TIMESTAMP}.json" \
            --severity CRITICAL,HIGH,MEDIUM
        
        trivy fs . \
            --scanners vuln \
            --format table \
            --output "${REPORT_DIR}/trivy_python_deps_${TIMESTAMP}.txt" \
            --severity CRITICAL,HIGH,MEDIUM
    fi
    
    # Clean up temporary file
    [ -f temp_requirements.txt ] && rm temp_requirements.txt
    
    print_success "Python dependency scan completed"
}

# Function to scan for secrets
scan_secrets() {
    print_status "Scanning for exposed secrets and sensitive information..."
    
    trivy fs . \
        --scanners secret \
        --format json \
        --output "${REPORT_DIR}/secrets_scan_${TIMESTAMP}.json"
    
    trivy fs . \
        --scanners secret \
        --format table \
        --output "${REPORT_DIR}/secrets_scan_${TIMESTAMP}.txt"
    
    print_success "Secret scan completed"
}

# Function to scan for misconfigurations
scan_misconfigurations() {
    print_status "Scanning for security misconfigurations..."
    
    trivy fs . \
        --scanners misconfig \
        --format json \
        --output "${REPORT_DIR}/misconfig_scan_${TIMESTAMP}.json"
    
    trivy fs . \
        --scanners misconfig \
        --format table \
        --output "${REPORT_DIR}/misconfig_scan_${TIMESTAMP}.txt"
    
    print_success "Misconfiguration scan completed"
}

# Function to build and scan Docker container (if Dockerfile exists)
scan_container() {
    if [ -f Dockerfile ] || [ -f docker/Dockerfile ]; then
        print_status "Building and scanning Docker container..."
        
        # Create a simple Dockerfile for scanning if none exists
        if [ ! -f Dockerfile ]; then
            cat > Dockerfile.security_scan <<EOF
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir build setuptools wheel

# Build the package
RUN python -m build

CMD ["python", "-c", "import quadtree; print('Package built successfully')"]
EOF
            DOCKERFILE="Dockerfile.security_scan"
        else
            DOCKERFILE="Dockerfile"
        fi
        
        # Build image
        docker build -f "${DOCKERFILE}" -t quadtree-security-scan:latest .
        
        # Scan container image
        trivy image \
            --format json \
            --output "${REPORT_DIR}/container_scan_${TIMESTAMP}.json" \
            --severity CRITICAL,HIGH,MEDIUM \
            quadtree-security-scan:latest
        
        trivy image \
            --format table \
            --output "${REPORT_DIR}/container_scan_${TIMESTAMP}.txt" \
            --severity CRITICAL,HIGH,MEDIUM \
            quadtree-security-scan:latest
        
        # Clean up
        docker rmi quadtree-security-scan:latest || true
        [ -f Dockerfile.security_scan ] && rm Dockerfile.security_scan
        
        print_success "Container scan completed"
    else
        print_warning "No Dockerfile found, skipping container scan"
    fi
}

# Function to generate summary report
generate_summary() {
    print_status "Generating security summary report..."
    
    SUMMARY_FILE="${REPORT_DIR}/security_summary_${TIMESTAMP}.md"
    
    cat > "${SUMMARY_FILE}" <<EOF
# Security Vulnerability Report

**Generated:** $(date)
**Project:** QuadTree Package
**Scan Type:** Comprehensive Security Scan

## Summary

This report contains the results of a comprehensive security vulnerability scan
performed on the QuadTree package using Trivy scanner.

## Scans Performed

### 1. Filesystem Vulnerability Scan
- **File:** filesystem_scan_${TIMESTAMP}.json/txt
- **Scope:** Source code, dependencies, configuration files
- **Severities:** CRITICAL, HIGH, MEDIUM, LOW

### 2. Python Dependencies Scan  
- **File:** python_deps_${TIMESTAMP}.json/txt
- **Scope:** Python package dependencies
- **Severities:** CRITICAL, HIGH, MEDIUM

### 3. Secret Detection Scan
- **File:** secrets_scan_${TIMESTAMP}.json/txt  
- **Scope:** API keys, passwords, tokens, certificates
- **Focus:** Preventing credential exposure

### 4. Misconfiguration Scan
- **File:** misconfig_scan_${TIMESTAMP}.json/txt
- **Scope:** Security configuration issues
- **Focus:** Infrastructure as Code security

$(if [ -f "${REPORT_DIR}/container_scan_${TIMESTAMP}.json" ]; then
echo "### 5. Container Image Scan"
echo "- **File:** container_scan_${TIMESTAMP}.json/txt"
echo "- **Scope:** Docker container vulnerabilities"
echo "- **Severities:** CRITICAL, HIGH, MEDIUM"
fi)

## Report Files

All detailed scan results are available in the following formats:

- **JSON Format:** Machine-readable, suitable for CI/CD integration
- **Table Format:** Human-readable, suitable for manual review

## Next Steps

1. Review all CRITICAL and HIGH severity findings
2. Update vulnerable dependencies to secure versions
3. Remove any exposed secrets or sensitive information
4. Fix security misconfigurations
5. Implement continuous security scanning in CI/CD pipeline

## Compliance

This scan helps ensure compliance with:
- OWASP Top 10 security practices
- Supply chain security best practices  
- Secret management security standards
- Container security guidelines

EOF

    print_success "Summary report generated: ${SUMMARY_FILE}"
}

# Function to check critical vulnerabilities
check_critical_issues() {
    print_status "Checking for critical security issues..."
    
    CRITICAL_FOUND=false
    
    # Check for critical vulnerabilities in JSON reports
    for report in "${REPORT_DIR}"/*_${TIMESTAMP}.json; do
        if [ -f "$report" ]; then
            if jq -e '.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL")' "$report" >/dev/null 2>&1; then
                CRITICAL_FOUND=true
                break
            fi
        fi
    done
    
    if [ "$CRITICAL_FOUND" = true ]; then
        print_error "CRITICAL vulnerabilities found! Please review the reports immediately."
        return 1
    else
        print_success "No CRITICAL vulnerabilities detected."
        return 0
    fi
}

# Main execution function
main() {
    echo "=================================================="
    echo "QuadTree Package Security Vulnerability Scanner"
    echo "=================================================="
    echo
    
    # Check prerequisites
    check_trivy_installation
    
    # Check if jq is available for JSON processing
    if ! command -v jq &> /dev/null; then
        print_warning "jq not found. Installing jq for JSON processing..."
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get update && sudo apt-get install -y jq
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            brew install jq
        fi
    fi
    
    # Create report directory
    create_report_directory
    
    # Update Trivy database
    print_status "Updating Trivy vulnerability database..."
    trivy image --download-db-only
    
    # Perform scans
    scan_filesystem
    scan_python_dependencies  
    scan_secrets
    scan_misconfigurations
    
    # Container scan if Docker is available
    if command -v docker &> /dev/null; then
        scan_container
    else
        print_warning "Docker not found, skipping container scan"
    fi
    
    # Generate summary
    generate_summary
    
    # Check for critical issues
    echo
    echo "=================================================="
    echo "Security Scan Complete"
    echo "=================================================="
    echo
    
    print_status "All reports saved to: ${REPORT_DIR}/"
    ls -la "${REPORT_DIR}/"
    
    echo
    if check_critical_issues; then
        print_success "Security scan completed successfully!"
        exit 0
    else
        print_error "Security scan completed with CRITICAL findings!"
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Usage: $0 [options]"
            echo
            echo "Options:"
            echo "  --help, -h          Show this help message"
            echo "  --report-dir DIR    Set custom report directory (default: ./security-reports)"
            echo
            echo "This script performs comprehensive security vulnerability scanning using Trivy."
            echo "It scans for vulnerabilities, secrets, and misconfigurations in:"
            echo "  - Source code and dependencies"
            echo "  - Python packages" 
            echo "  - Container images (if Docker available)"
            echo "  - Configuration files"
            exit 0
            ;;
        --report-dir)
            REPORT_DIR="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main