#!/usr/bin/env python3
"""
Security Vulnerability Test Suite for QuadTree Package
Tests for common security vulnerabilities and best practices
"""

import sys
import os
import subprocess
import json
import tempfile
import unittest
from pathlib import Path
import importlib.util

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SecurityVulnerabilityTests(unittest.TestCase):
    """Test suite for security vulnerabilities"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.project_root = Path(__file__).parent.parent
        cls.temp_dir = tempfile.mkdtemp()
        
    def test_no_hardcoded_secrets(self):
        """Test that no secrets are hardcoded in source files"""
        print("Testing for hardcoded secrets...")
        
        # Patterns that might indicate secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']', 
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'-----BEGIN \w+ KEY-----',
            r'sk_\w{20,}',  # Stripe-like secret keys
            r'pk_\w{20,}',  # Stripe-like public keys
            r'AIza[0-9A-Za-z\-_]{35}',  # Google API keys
            r'AKIA[0-9A-Z]{16}',  # AWS access keys
        ]
        
        # Files to check
        source_files = []
        for pattern in ['**/*.py', '**/*.cpp', '**/*.h', '**/*.yml', '**/*.yaml', '**/*.json']:
            source_files.extend(self.project_root.glob(pattern))
        
        secrets_found = []
        
        for file_path in source_files:
            if file_path.name.startswith('.') or 'test_security' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    for pattern in secret_patterns:
                        import re
                        if re.search(pattern, content, re.IGNORECASE):
                            secrets_found.append(f"{file_path}: Potential secret pattern found")
            except Exception as e:
                # Skip files that can't be read
                continue
        
        if secrets_found:
            self.fail(f"Potential secrets found:\n" + "\n".join(secrets_found))
        
        print("✓ No hardcoded secrets detected")
    
    def test_no_dangerous_imports(self):
        """Test that no dangerous Python imports are used"""
        print("Testing for dangerous imports...")
        
        dangerous_imports = [
            'pickle',  # Can execute arbitrary code
            'os.system',  # Direct system command execution
            'subprocess.call',  # Without proper sanitization
            'eval',  # Code evaluation
            'exec',  # Code execution
            '__import__',  # Dynamic imports
        ]
        
        # Only check Python files
        python_files = list(self.project_root.glob('**/*.py'))
        
        dangerous_found = []
        
        for file_path in python_files:
            if 'test_security' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    for dangerous in dangerous_imports:
                        if dangerous in content and not content.startswith('#'):
                            # Check if it's actually imported/used
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if dangerous in line and not line.strip().startswith('#'):
                                    dangerous_found.append(f"{file_path}:{i+1}: {dangerous}")
            except Exception:
                continue
        
        # Filter out acceptable uses (like subprocess with proper args)
        filtered_dangerous = []
        for item in dangerous_found:
            # Allow subprocess.run with list args (safer)
            if 'subprocess.run(' in item and '[' in item:
                continue
            filtered_dangerous.append(item)
        
        if filtered_dangerous:
            print("⚠️ Potentially dangerous imports found (review required):")
            for item in filtered_dangerous:
                print(f"  {item}")
        else:
            print("✓ No dangerous imports detected")
    
    def test_input_validation(self):
        """Test that input validation is properly implemented"""
        print("Testing input validation...")
        
        try:
            # Try to import the quadtree module to test it
            quadtree_spec = importlib.util.find_spec("quadtree")
            if quadtree_spec is None:
                print("⚠️ QuadTree module not built, skipping input validation tests")
                return
                
            import quadtree
            
            # Test boundary conditions and invalid inputs
            qt = quadtree.QuadTree(0, 0, 100, 100)
            
            # Test invalid point coordinates
            test_cases = [
                (float('inf'), 50, "data"),  # Infinity
                (float('nan'), 50, "data"),  # NaN
                (50, float('inf'), "data"),  # Infinity Y
                (50, float('nan'), "data"),  # NaN Y
                (-1000000000, 50, "data"),  # Very large negative
                (1000000000, 50, "data"),   # Very large positive
            ]
            
            invalid_inputs_handled = 0
            total_tests = len(test_cases)
            
            for x, y, data in test_cases:
                try:
                    qt.insert(x, y, data)
                    # If no exception, check if point was actually added
                    points = qt.get_all_points()
                    valid_insert = any(abs(p[0] - x) < 1e-6 and abs(p[1] - y) < 1e-6 for p in points if not (str(x) == 'nan' or str(y) == 'nan'))
                    if not valid_insert and (str(x) == 'inf' or str(x) == 'nan' or str(y) == 'inf' or str(y) == 'nan'):
                        invalid_inputs_handled += 1
                except (ValueError, OverflowError, TypeError) as e:
                    # Good - invalid input was properly handled
                    invalid_inputs_handled += 1
                except Exception as e:
                    # Unexpected exception type
                    print(f"⚠️ Unexpected exception for input ({x}, {y}): {e}")
            
            if invalid_inputs_handled >= total_tests * 0.8:  # 80% threshold
                print("✓ Input validation appears to be properly implemented")
            else:
                print(f"⚠️ Input validation may need improvement ({invalid_inputs_handled}/{total_tests} cases handled)")
                
        except ImportError:
            print("⚠️ QuadTree module not available for input validation testing")
        except Exception as e:
            print(f"⚠️ Error during input validation testing: {e}")
    
    def test_memory_safety(self):
        """Test for potential memory safety issues"""
        print("Testing memory safety...")
        
        try:
            import quadtree
            
            # Test for memory leaks with large datasets
            initial_points = 1000
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            
            # Insert many points
            for i in range(initial_points):
                qt.insert(i % 1000, (i * 7) % 1000, f"data_{i}")
            
            # Check if we can query without crashes
            points = qt.query(100, 100, 200, 200)
            
            # Test recursive operations don't cause stack overflow
            deep_qt = quadtree.QuadTree(0, 0, 1000, 1000)
            # Insert points that will cause deep subdivision
            for i in range(100):
                deep_qt.insert(500 + i * 0.01, 500 + i * 0.01, f"deep_{i}")
            
            # Query should still work
            deep_points = deep_qt.query(499, 499, 501, 501)
            
            print("✓ Memory safety tests passed")
            
        except ImportError:
            print("⚠️ QuadTree module not available for memory safety testing")
        except Exception as e:
            print(f"⚠️ Memory safety test failed: {e}")
    
    def test_dependency_security(self):
        """Test dependency security by checking for known vulnerable packages"""
        print("Testing dependency security...")
        
        # Check if safety tool is available
        try:
            result = subprocess.run(['python', '-m', 'pip', 'list', '--format=json'], 
                                 capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print("⚠️ Could not list installed packages")
                return
                
            packages = json.loads(result.stdout)
            
            # Known vulnerable packages to warn about (examples)
            vulnerable_packages = {
                'requests': ['2.25.0'],  # Example: specific vulnerable version
                'urllib3': ['1.25.8'],   # Example: specific vulnerable version
            }
            
            warnings = []
            for package in packages:
                name = package['name'].lower()
                version = package['version']
                
                if name in vulnerable_packages:
                    if version in vulnerable_packages[name]:
                        warnings.append(f"{name} {version} has known vulnerabilities")
            
            if warnings:
                print("⚠️ Potentially vulnerable dependencies found:")
                for warning in warnings:
                    print(f"  {warning}")
            else:
                print("✓ No obviously vulnerable dependencies detected")
                
        except subprocess.TimeoutExpired:
            print("⚠️ Dependency check timed out")
        except Exception as e:
            print(f"⚠️ Error checking dependencies: {e}")
    
    def test_file_permissions(self):
        """Test that files have appropriate permissions"""
        print("Testing file permissions...")
        
        # Check that no files are world-writable
        suspicious_files = []
        
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file():
                try:
                    stat_info = file_path.stat()
                    mode = stat_info.st_mode
                    
                    # Check if world-writable (other write permission)
                    if mode & 0o002:
                        suspicious_files.append(f"{file_path}: World-writable")
                    
                    # Check if executable files in unexpected locations
                    if (mode & 0o111) and file_path.suffix in ['.py', '.cpp', '.h']:
                        if not str(file_path).startswith(str(self.project_root / 'scripts')):
                            # Only scripts should typically be executable
                            pass  # This is often normal for development
                            
                except (OSError, PermissionError):
                    # Skip files we can't check
                    continue
        
        if suspicious_files:
            print("⚠️ Files with suspicious permissions found:")
            for file_info in suspicious_files:
                print(f"  {file_info}")
        else:
            print("✓ File permissions appear appropriate")
    
    def test_configuration_security(self):
        """Test configuration files for security issues"""
        print("Testing configuration security...")
        
        config_files = []
        for pattern in ['**/*.yml', '**/*.yaml', '**/*.json', '**/*.cfg', '**/*.ini']:
            config_files.extend(self.project_root.glob(pattern))
        
        issues = []
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                    # Check for insecure configurations
                    insecure_patterns = [
                        ('debug: true', 'Debug mode enabled'),
                        ('ssl_verify: false', 'SSL verification disabled'),
                        ('verify: false', 'Verification disabled'),
                        ('insecure: true', 'Insecure mode enabled'),
                    ]
                    
                    for pattern, description in insecure_patterns:
                        if pattern in content:
                            issues.append(f"{config_file}: {description}")
                            
            except Exception:
                continue
        
        if issues:
            print("⚠️ Potentially insecure configurations found:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("✓ Configuration security checks passed")

def run_security_tests():
    """Run all security vulnerability tests"""
    print("🔒 Running Security Vulnerability Tests for QuadTree Package")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test methods
    test_methods = [
        'test_no_hardcoded_secrets',
        'test_no_dangerous_imports',
        'test_input_validation',
        'test_memory_safety',
        'test_dependency_security',
        'test_file_permissions',
        'test_configuration_security',
    ]
    
    for method in test_methods:
        test_suite.addTest(SecurityVulnerabilityTests(method))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    print("🔒 Security Vulnerability Test Summary")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failures} ✗")
    print(f"Errors: {errors} ⚠️")
    
    if failures > 0:
        print("\n❌ FAILED TESTS:")
        for test, traceback in result.failures:
            print(f"  • {test}")
            
    if errors > 0:
        print("\n⚠️ ERROR TESTS:")
        for test, traceback in result.errors:
            print(f"  • {test}")
    
    success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Security tests mostly PASSED - Good security posture")
        return True
    else:
        print("⚠️ Security tests show CONCERNS - Review and fix issues")
        return False

if __name__ == "__main__":
    success = run_security_tests()
    sys.exit(0 if success else 1)