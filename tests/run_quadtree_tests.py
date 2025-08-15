#!/usr/bin/env python3
"""
QuadTree Test Suite Runner
Runs all QuadTree tests with progress tracking and summary reporting
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def find_quadtree_tests():
    """Find all QuadTree test files"""
    test_dir = Path("Lib/test")
    if not test_dir.exists():
        print("❌ Test directory not found. Run from CPython root directory.")
        return []
    
    test_files = list(test_dir.glob("test_quadtree*.py"))
    return sorted(test_files)

def run_test(test_file):
    """Run a single test file and capture results"""
    print(f"🧪 Running {test_file.name}...")
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, str(test_file)
        ], capture_output=True, text=True, timeout=300)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_file.name} - PASSED ({duration:.1f}s)")
            return {'name': test_file.name, 'status': 'PASSED', 'duration': duration, 'output': result.stdout}
        else:
            print(f"❌ {test_file.name} - FAILED ({duration:.1f}s)")
            return {'name': test_file.name, 'status': 'FAILED', 'duration': duration, 'error': result.stderr}
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_file.name} - TIMEOUT (>300s)")
        return {'name': test_file.name, 'status': 'TIMEOUT', 'duration': 300, 'error': 'Test timeout'}
    except Exception as e:
        print(f"💥 {test_file.name} - ERROR: {e}")
        return {'name': test_file.name, 'status': 'ERROR', 'duration': 0, 'error': str(e)}

def main():
    """Run all QuadTree tests"""
    print("🔋 QuadTree Test Suite Runner")
    print("=" * 60)
    
    # Find test files
    test_files = find_quadtree_tests()
    if not test_files:
        print("❌ No QuadTree test files found")
        return 1
    
    print(f"Found {len(test_files)} QuadTree test files:")
    for test_file in test_files:
        print(f"  • {test_file.name}")
    print()
    
    # Run tests
    results = []
    total_start_time = time.time()
    
    for i, test_file in enumerate(test_files, 1):
        print(f"[{i}/{len(test_files)}] ", end="")
        result = run_test(test_file)
        results.append(result)
        print()  # Add spacing between tests
    
    total_duration = time.time() - total_start_time
    
    # Generate summary
    print("=" * 60)
    print("📊 TEST SUITE SUMMARY")
    print("=" * 60)
    
    passed = [r for r in results if r['status'] == 'PASSED']
    failed = [r for r in results if r['status'] == 'FAILED'] 
    errors = [r for r in results if r['status'] in ['ERROR', 'TIMEOUT']]
    
    print(f"Total Tests: {len(results)}")
    print(f"✅ Passed: {len(passed)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"💥 Errors: {len(errors)}")
    print(f"⏱️  Total Duration: {total_duration:.1f}s")
    print()
    
    # Show failed tests
    if failed or errors:
        print("❌ Failed/Error Tests:")
        for result in failed + errors:
            print(f"  • {result['name']}: {result['status']}")
            if 'error' in result:
                print(f"    {result['error'][:100]}...")
        print()
    
    # Success rate
    success_rate = len(passed) / len(results) * 100
    print(f"🎯 Success Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("🎉 ALL TESTS PASSED! QuadTree implementation is working perfectly.")
        return 0
    elif success_rate >= 80:
        print("⚠️  Most tests passed, but some issues need attention.")
        return 1
    else:
        print("🚨 Significant test failures detected. Implementation needs review.")
        return 2

if __name__ == "__main__":
    sys.exit(main())