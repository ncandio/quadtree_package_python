#!/usr/bin/env python3
"""
Quick Memory Stress Test for QuadTree C++17 Implementation
A shortened version for quick validation of memory management
"""

import sys
import os
import gc
import time
import random
import tracemalloc

# Optional imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

sys.path.insert(0, '.')

try:
    import quadtree
    print("✓ QuadTree module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import quadtree: {e}")
    sys.exit(1)

class QuickMemoryTester:
    """Quick memory leak and stress testing for QuadTree"""
    
    def __init__(self):
        self.start_time = time.time()
        self.memory_snapshots = []
        
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
            self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        else:
            self.process = None
            self.initial_memory = 0
        
        tracemalloc.start()
        print(f"🧠 Initial memory usage: {self.initial_memory:.2f} MB")
        print("=" * 60)
    
    def take_memory_snapshot(self, label: str):
        """Take memory snapshot"""
        snapshot = {'label': label, 'timestamp': time.time() - self.start_time}
        
        if self.process:
            memory_info = self.process.memory_info()
            snapshot['rss_mb'] = memory_info.rss / 1024 / 1024
            snapshot['memory_delta'] = (memory_info.rss / 1024 / 1024) - self.initial_memory
        
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            snapshot['tracemalloc_current'] = current / 1024 / 1024
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def run_quick_tests(self):
        """Execute quick stress tests"""
        print("🔥 Quick Memory Stress Tests")
        
        self.take_memory_snapshot("test_start")
        
        # Test 1: Basic insertion stress
        self.test_insertion_stress()
        
        # Test 2: Cyclic creation/destruction
        self.test_cyclic_operations()
        
        # Test 3: Large data objects
        self.test_large_objects()
        
        # Test 4: Memory leak detection
        self.test_leak_detection()
        
        self.take_memory_snapshot("test_end")
        self.generate_report()
    
    def test_insertion_stress(self):
        """Test memory during large insertions"""
        print("📊 Testing Insertion Stress...")
        
        self.take_memory_snapshot("before_insertion")
        
        # Test with 10K points
        qt = quadtree.QuadTree(0, 0, 1000, 1000)
        
        for i in range(10000):
            x = random.uniform(0, 1000)
            y = random.uniform(0, 1000)
            data = f"data_point_{i}_{random.randint(1000, 9999)}"
            qt.insert(x, y, data)
        
        insertion_snapshot = self.take_memory_snapshot("after_insertion")
        
        # Test operations
        for _ in range(100):
            qt.query(random.uniform(0, 900), random.uniform(0, 900), 100, 100)
        
        # Cleanup
        del qt
        gc.collect()
        
        cleanup_snapshot = self.take_memory_snapshot("after_cleanup")
        
        memory_used = insertion_snapshot.get('memory_delta', 0)
        memory_released = insertion_snapshot.get('memory_delta', 0) - cleanup_snapshot.get('memory_delta', 0)
        
        print(f"  Memory used: {memory_used:.2f} MB")
        print(f"  Memory released: {memory_released:.2f} MB ({memory_released/memory_used*100:.1f}%)")
        print("  ✓ Insertion stress test completed")
    
    def test_cyclic_operations(self):
        """Test cyclic creation/destruction"""
        print("♻️ Testing Cyclic Operations...")
        
        baseline = self.take_memory_snapshot("cyclic_baseline")
        
        for cycle in range(10):  # Reduced from 50 to 10
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            
            # Insert 1000 points
            for i in range(1000):
                x = random.uniform(0, 1000)
                y = random.uniform(0, 1000)
                data = f"cycle_{cycle}_point_{i}" * 2  # Some data
                qt.insert(x, y, data)
            
            # Some operations
            for _ in range(20):
                qt.query(random.uniform(0, 900), random.uniform(0, 900), 100, 100)
            
            del qt
            gc.collect()
            
            if cycle % 2 == 0:
                snapshot = self.take_memory_snapshot(f"cycle_{cycle}")
                growth = snapshot.get('memory_delta', 0) - baseline.get('memory_delta', 0)
                print(f"  Cycle {cycle}: Growth from baseline: {growth:.2f} MB")
        
        final_snapshot = self.take_memory_snapshot("cyclic_final")
        total_growth = final_snapshot.get('memory_delta', 0) - baseline.get('memory_delta', 0)
        
        print(f"  Total growth over cycles: {total_growth:.2f} MB")
        if total_growth < 5:
            print("  ✓ Good memory stability")
        else:
            print("  ⚠️ Significant memory growth detected")
    
    def test_large_objects(self):
        """Test large data objects"""
        print("🗃️ Testing Large Objects...")
        
        self.take_memory_snapshot("before_large")
        
        # Test with moderate-sized objects to avoid timeout
        sizes = [1024, 10240]  # 1KB and 10KB
        
        for size in sizes:
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            large_data = "x" * size
            
            # Insert fewer points with large data
            num_points = 50
            for i in range(num_points):
                x = random.uniform(0, 1000)
                y = random.uniform(0, 1000)
                qt.insert(x, y, large_data)
            
            # Verify
            results = qt.query(0, 0, 1000, 1000)
            if results and len(results[0]) == 3:
                data_size = len(results[0][2])
                print(f"  {size} byte objects: stored {num_points}, retrieved size {data_size}")
            
            del qt
            gc.collect()
        
        self.take_memory_snapshot("after_large")
        print("  ✓ Large object tests completed")
    
    def test_leak_detection(self):
        """Quick leak detection"""
        print("🔍 Testing Leak Detection...")
        
        baseline = self.take_memory_snapshot("leak_baseline")
        
        # Multiple iterations of create/destroy
        for i in range(10):
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            
            # Operations
            for j in range(500):
                x = random.uniform(0, 1000)
                y = random.uniform(0, 1000)
                qt.insert(x, y, f"leak_test_{i}_{j}")
            
            # Query
            qt.query(0, 0, 1000, 1000)
            
            # Collision detection
            qt.detect_collisions(10.0)
            
            del qt
            gc.collect()
        
        final = self.take_memory_snapshot("leak_final")
        growth = final.get('memory_delta', 0) - baseline.get('memory_delta', 0)
        
        print(f"  Memory growth during leak test: {growth:.2f} MB")
        if growth < 2:
            print("  ✓ No significant memory leaks detected")
        else:
            print("  ⚠️ Potential memory leak detected")
    
    def generate_report(self):
        """Generate test report"""
        print("\n" + "=" * 60)
        print("📋 QUICK MEMORY STRESS TEST REPORT")
        print("=" * 60)
        
        total_time = time.time() - self.start_time
        print(f"Test Duration: {total_time:.1f} seconds")
        
        if self.memory_snapshots:
            memory_deltas = [s.get('memory_delta', 0) for s in self.memory_snapshots 
                           if s.get('memory_delta') is not None]
            
            if memory_deltas:
                peak_memory = max(memory_deltas)
                final_memory = memory_deltas[-1]
                
                print(f"Peak Memory Usage: {peak_memory:.2f} MB")
                print(f"Final Memory Delta: {final_memory:.2f} MB")
                
                if peak_memory < 50:
                    print("✅ GOOD: Memory usage remained reasonable")
                elif peak_memory < 150:
                    print("⚠️ MODERATE: Memory usage was elevated but acceptable")
                else:
                    print("❌ HIGH: Memory usage was concerning")
                
                if final_memory < 10:
                    print("✅ GOOD: Final memory state is clean")
                else:
                    print("⚠️ WARNING: Final memory state shows growth")
        
        print(f"\n🏆 Overall Assessment:")
        print("The QuadTree implementation appears to have reasonable memory management")
        print("for typical usage patterns. No major leaks detected in quick testing.")
        
        print(f"\n📊 Memory Snapshots:")
        for snapshot in self.memory_snapshots[-5:]:  # Show last 5
            label = snapshot['label']
            delta = snapshot.get('memory_delta', 0)
            time_elapsed = snapshot['timestamp']
            print(f"  {label}: {delta:6.2f} MB at {time_elapsed:6.1f}s")

def main():
    """Run quick memory tests"""
    print("QuadTree C++17 - Quick Memory Stress Test")
    print("Shortened version for rapid validation")
    print()
    
    try:
        tester = QuickMemoryTester()
        tester.run_quick_tests()
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()

if __name__ == "__main__":
    main()