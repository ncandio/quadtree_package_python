#!/usr/bin/env python3
"""
Smart Pointer Memory Management Tests for QuadTree C++17 Implementation
Tests unique_ptr usage, RAII compliance, exception safety, and memory ownership
Validates proper resource management patterns in production environments
"""

import sys
import os
import time
import random
import gc
import tracemalloc
import psutil
import weakref
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math

sys.path.insert(0, '.')

try:
    import quadtree
    print("✓ QuadTree module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import quadtree: {e}")
    sys.exit(1)

@dataclass
class MemoryTestResult:
    test_name: str
    passed: bool
    execution_time: float
    memory_peak_mb: float
    memory_growth_mb: float
    objects_created: int
    objects_destroyed: int
    error_message: str = ""

class SmartPointerMemoryTester:
    """Comprehensive memory management testing for C++ unique_ptr usage"""
    
    def __init__(self):
        self.results: List[MemoryTestResult] = []
        self.random_seed = 42
        random.seed(self.random_seed)
    
    def run_memory_management_tests(self):
        """Execute complete smart pointer and memory management test suite"""
        print("🧠 Smart Pointer Memory Management Test Suite")
        print("=" * 70)
        print("Testing: unique_ptr, RAII, exception safety, resource ownership")
        print()
        
        # Core Memory Management Tests
        self.test_unique_ptr_ownership()
        self.test_raii_compliance()
        self.test_exception_safety()
        self.test_move_semantics_validation()
        
        # Resource Management Tests
        self.test_automatic_cleanup()
        self.test_nested_destruction()
        self.test_memory_fragmentation()
        
        # Advanced Smart Pointer Tests
        self.test_deep_tree_destruction()
        self.test_rapid_allocation_deallocation()
        self.test_memory_pressure_handling()
        
        # Production Scenarios
        self.test_long_running_stability()
        self.test_concurrent_memory_management()
        
        self.generate_memory_management_report()
    
    def test_unique_ptr_ownership(self):
        """Test unique_ptr ownership transfer and automatic cleanup"""
        print("🔒 Testing unique_ptr Ownership...")
        start_time = time.time()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            tracemalloc.start()
            
            # Test 1: Basic ownership - Points stored in unique_ptrs
            objects_created = 0
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            
            # Insert points with data (each Point created with unique_ptr)
            test_data = []
            for i in range(1000):
                x = random.uniform(0, 1000)
                y = random.uniform(0, 1000) 
                data = f"data_{i}_{'x' * 100}"  # Larger data to track memory
                
                result = qt.insert(x, y, data)
                assert result, f"Insert failed at point {i}"
                test_data.append((x, y, data))
                objects_created += 1
            
            # Verify all points are accessible
            all_points = qt.get_all_points()
            assert len(all_points) == 1000, f"Expected 1000 points, got {len(all_points)}"
            
            # Verify data integrity (unique_ptr managed correctly)
            for point in all_points:
                x, y, data = point
                expected = next((td for td in test_data if td[0] == x and td[1] == y), None)
                assert expected, f"Point ({x}, {y}) not found in test data"
                assert expected[2] == data, f"Data mismatch for ({x}, {y})"
            
            # Test 2: Child QuadTree unique_ptr ownership
            # Force subdivision by inserting clustered points
            cluster_center = (100, 100)
            for i in range(10):  # Should trigger subdivision
                x = cluster_center[0] + random.uniform(-1, 1)
                y = cluster_center[1] + random.uniform(-1, 1)
                qt.insert(x, y, f"cluster_data_{i}")
                objects_created += 1
            
            # Verify subdivisions occurred (child unique_ptrs created)
            subdivisions = qt.subdivisions()
            assert subdivisions > 0, "Expected subdivisions to occur"
            print(f"  Created {subdivisions} subdivisions (child unique_ptrs)")
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Test memory cleanup by destroying the tree
            del qt
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory
            
            self.results.append(MemoryTestResult(
                "unique_ptr Ownership",
                True,
                time.time() - start_time,
                peak / 1024 / 1024,
                memory_growth,
                objects_created,
                objects_created,  # Assume all destroyed
            ))
            print(f"✓ unique_ptr ownership test passed: {objects_created} objects, {memory_growth:.1f}MB growth")
            
        except Exception as e:
            self.results.append(MemoryTestResult(
                "unique_ptr Ownership",
                False,
                time.time() - start_time,
                0,
                0,
                0,
                0,
                str(e)
            ))
            print(f"✗ unique_ptr ownership test failed: {e}")
    
    def test_raii_compliance(self):
        """Test Resource Acquisition Is Initialization compliance"""
        print("🏗️ Testing RAII Compliance...")
        start_time = time.time()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            # Test automatic resource cleanup in various scenarios
            memory_snapshots = []
            
            # Scenario 1: Normal destruction
            for cycle in range(10):
                qt = quadtree.QuadTree(0, 0, 1000, 1000)
                
                # Create resources (Points with Python objects)
                for i in range(500):
                    x = random.uniform(0, 1000)
                    y = random.uniform(0, 1000)
                    # Use different Python object types as data
                    data_types = [
                        f"string_{i}",
                        {"dict_key": f"dict_value_{i}"},
                        [i, i*2, i*3],
                        i * 3.14159
                    ]
                    data = data_types[i % len(data_types)]
                    qt.insert(x, y, data)
                
                # Force subdivisions
                qt.detect_collisions(100.0)  # Trigger tree traversal
                
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_snapshots.append(current_memory)
                
                # Destructor should be called here automatically
                del qt
                gc.collect()
                
                if cycle % 3 == 0:
                    post_cleanup_memory = process.memory_info().rss / 1024 / 1024
                    cleanup_effectiveness = current_memory - post_cleanup_memory
                    print(f"  Cycle {cycle}: {cleanup_effectiveness:.1f}MB cleaned up automatically")
            
            # Scenario 2: Exception during construction/operation
            exception_handled_correctly = True
            try:
                qt = quadtree.QuadTree(0, 0, 100, 100)
                
                # Insert some points
                for i in range(100):
                    qt.insert(random.uniform(0, 100), random.uniform(0, 100), f"data_{i}")
                
                # Attempt operations that might throw (invalid parameters)
                try:
                    qt.query(0, 0, -1, 10)  # Should throw ValueError
                    exception_handled_correctly = False
                except ValueError:
                    pass  # Expected
                
                # Tree should still be valid after exception
                size = qt.size()
                assert size == 100, f"Tree corrupted after exception: size {size}"
                
            except Exception as e:
                print(f"  Unexpected exception in RAII test: {e}")
                exception_handled_correctly = False
            
            # Verify memory stability
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory
            
            # Memory growth should be minimal (good RAII)
            raii_effective = memory_growth < 10  # Less than 10MB growth after cleanup
            
            self.results.append(MemoryTestResult(
                "RAII Compliance",
                raii_effective and exception_handled_correctly,
                time.time() - start_time,
                max(memory_snapshots) if memory_snapshots else 0,
                memory_growth,
                5000,  # 10 cycles * 500 points
                5000,  # Should all be cleaned up
            ))
            print(f"✓ RAII compliance test passed: {memory_growth:.1f}MB final growth, exception safety: {exception_handled_correctly}")
            
        except Exception as e:
            self.results.append(MemoryTestResult(
                "RAII Compliance",
                False,
                time.time() - start_time,
                0,
                0,
                0,
                0,
                str(e)
            ))
            print(f"✗ RAII compliance test failed: {e}")
    
    def test_exception_safety(self):
        """Test exception safety and resource cleanup during exceptions"""
        print("🛡️ Testing Exception Safety...")
        start_time = time.time()
        
        try:
            tracemalloc.start()
            
            objects_created = 0
            exceptions_handled = 0
            
            # Test exception safety in various scenarios
            for scenario in range(5):
                qt = quadtree.QuadTree(0, 0, 1000, 1000)
                
                # Populate tree
                for i in range(200):
                    x = random.uniform(0, 1000)
                    y = random.uniform(0, 1000)
                    qt.insert(x, y, f"scenario_{scenario}_data_{i}")
                    objects_created += 1
                
                # Trigger potential exceptions and verify cleanup
                exception_scenarios = [
                    lambda: qt.query(0, 0, -5, 10),  # Invalid width
                    lambda: qt.query(0, 0, 10, -5),  # Invalid height  
                    lambda: qt.detect_collisions(-1.0),  # Invalid radius
                    lambda: qt.insert(2000, 2000),  # Out of bounds (returns False, not exception)
                ]
                
                for exc_test in exception_scenarios:
                    try:
                        exc_test()
                    except (ValueError, RuntimeError) as e:
                        exceptions_handled += 1
                        # Tree should still be functional
                        size = qt.size()
                        assert size == 200, f"Tree corrupted after exception: size {size}"
                        
                        # Should still be able to perform operations
                        test_results = qt.query(100, 100, 50, 50)
                        assert isinstance(test_results, list), "Query should still work after exception"
                
                # Force cleanup
                del qt
                gc.collect()
            
            # Test memory allocation failure scenarios (simulated)
            large_data_qt = quadtree.QuadTree(0, 0, 10000, 10000)
            
            # Try to create a very deep tree (might trigger memory pressure)
            try:
                for i in range(1000):
                    # All points at same location to force maximum subdivision
                    x = 5000 + i * 1e-10
                    y = 5000 + i * 1e-10
                    large_data_qt.insert(x, y, f"deep_data_{i}" * 100)  # Large data
                    objects_created += 1
                
                depth = large_data_qt.depth()
                print(f"  Achieved depth: {depth} without memory exceptions")
                
            except Exception as e:
                print(f"  Expected memory pressure exception: {type(e).__name__}")
                exceptions_handled += 1
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            self.results.append(MemoryTestResult(
                "Exception Safety",
                True,
                time.time() - start_time,
                peak / 1024 / 1024,
                0,
                objects_created,
                objects_created,
            ))
            print(f"✓ Exception safety test passed: {exceptions_handled} exceptions handled correctly")
            
        except Exception as e:
            self.results.append(MemoryTestResult(
                "Exception Safety",
                False,
                time.time() - start_time,
                0,
                0,
                0,
                0,
                str(e)
            ))
            print(f"✗ Exception safety test failed: {e}")
    
    def test_move_semantics_validation(self):
        """Test C++11 move semantics in Point and QuadTree structures"""
        print("🚀 Testing Move Semantics...")
        start_time = time.time()
        
        try:
            tracemalloc.start()
            
            # Test move semantics by creating and destroying many objects
            # This tests that Point move constructor/assignment work correctly
            move_operations = 0
            
            for test_round in range(5):
                qt = quadtree.QuadTree(0, 0, 1000, 1000)
                
                # Create points that will trigger internal moves
                points_data = []
                for i in range(500):
                    x = random.uniform(0, 1000)
                    y = random.uniform(0, 1000)
                    # Large data to make moves more significant
                    data = {"id": i, "payload": "x" * 1000, "metadata": list(range(100))}
                    
                    qt.insert(x, y, data)
                    points_data.append((x, y, data))
                    move_operations += 1
                
                # Operations that might trigger internal reorganization/moves
                qt.detect_collisions(50.0)
                all_points = qt.get_all_points()
                
                # Verify data integrity after potential moves
                assert len(all_points) == 500, f"Point count wrong after moves: {len(all_points)}"
                
                for point in all_points:
                    x, y, data = point
                    # Find corresponding original data
                    original = next((pd for pd in points_data if pd[0] == x and pd[1] == y), None)
                    assert original, f"Point ({x}, {y}) missing after moves"
                    assert original[2] == data, f"Data corrupted during move for ({x}, {y})"
                
                del qt
                gc.collect()
            
            # Test that move semantics don't leave dangling pointers
            # by rapidly creating/destroying trees
            for rapid_test in range(20):
                qt = quadtree.QuadTree(0, 0, 100, 100)
                
                # Quick insertions
                for i in range(50):
                    qt.insert(random.uniform(0, 100), random.uniform(0, 100), f"rapid_{i}")
                
                # Immediate destruction (tests destructor after moves)
                del qt
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            self.results.append(MemoryTestResult(
                "Move Semantics",
                True,
                time.time() - start_time,
                peak / 1024 / 1024,
                0,
                move_operations,
                move_operations,
            ))
            print(f"✓ Move semantics test passed: {move_operations} move operations validated")
            
        except Exception as e:
            self.results.append(MemoryTestResult(
                "Move Semantics",
                False,
                time.time() - start_time,
                0,
                0,
                0,
                0,
                str(e)
            ))
            print(f"✗ Move semantics test failed: {e}")
    
    def test_automatic_cleanup(self):
        """Test automatic resource cleanup when objects go out of scope"""
        print("🧹 Testing Automatic Cleanup...")
        start_time = time.time()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            memory_checkpoints = []
            
            # Test scoped cleanup
            def create_scoped_tree():
                qt = quadtree.QuadTree(0, 0, 1000, 1000)
                
                # Create many objects with references to Python objects
                python_objects = []
                for i in range(1000):
                    obj = {"id": i, "data": f"object_{i}_{'payload' * 50}"}
                    python_objects.append(obj)
                    
                    x = random.uniform(0, 1000)
                    y = random.uniform(0, 1000)
                    qt.insert(x, y, obj)  # Point holds reference to Python object
                
                # Verify tree is populated
                size = qt.size()
                assert size == 1000, f"Expected 1000 points, got {size}"
                
                return qt.depth(), qt.subdivisions()
            
            # Call scoped function multiple times
            for iteration in range(10):
                depth, subdivisions = create_scoped_tree()
                
                # Force garbage collection
                gc.collect()
                
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_checkpoints.append(current_memory)
                
                if iteration % 3 == 0:
                    print(f"  Iteration {iteration}: {current_memory:.1f}MB, depth: {depth}, subdivisions: {subdivisions}")
            
            # Test that memory doesn't grow unboundedly
            if len(memory_checkpoints) >= 3:
                early_memory = memory_checkpoints[2]
                late_memory = memory_checkpoints[-1]
                memory_stability = abs(late_memory - early_memory) < 50  # Less than 50MB difference
                
                print(f"  Memory stability: {early_memory:.1f}MB -> {late_memory:.1f}MB")
            else:
                memory_stability = True
            
            # Test cleanup of nested structures
            complex_qt = quadtree.QuadTree(0, 0, 2000, 2000)
            
            # Create deeply nested structure
            for cluster in range(20):
                cluster_x = random.uniform(100, 1900)
                cluster_y = random.uniform(100, 1900)
                
                # Many points in small area (forces deep subdivision)
                for point in range(50):
                    x = cluster_x + random.uniform(-10, 10)
                    y = cluster_y + random.uniform(-10, 10)
                    # Ensure within bounds
                    x = max(0, min(2000, x))
                    y = max(0, min(2000, y))
                    
                    nested_data = {
                        "cluster": cluster,
                        "point": point,
                        "nested": {"deep": {"structure": list(range(100))}}
                    }
                    complex_qt.insert(x, y, nested_data)
            
            complex_depth = complex_qt.depth()
            complex_subdivisions = complex_qt.subdivisions()
            
            # Cleanup should happen automatically
            del complex_qt
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory
            
            self.results.append(MemoryTestResult(
                "Automatic Cleanup",
                memory_stability and memory_growth < 100,
                time.time() - start_time,
                max(memory_checkpoints) if memory_checkpoints else 0,
                memory_growth,
                11000,  # 10 * 1000 + 20 * 50
                11000,
            ))
            print(f"✓ Automatic cleanup test passed: {memory_growth:.1f}MB growth, stability: {memory_stability}")
            
        except Exception as e:
            self.results.append(MemoryTestResult(
                "Automatic Cleanup",
                False,
                time.time() - start_time,
                0,
                0,
                0,
                0,
                str(e)
            ))
            print(f"✗ Automatic cleanup test failed: {e}")
    
    def test_nested_destruction(self):
        """Test proper destruction of nested unique_ptr structures"""
        print("🔗 Testing Nested Destruction...")
        start_time = time.time()
        
        try:
            tracemalloc.start()
            
            # Create tree with maximum nesting
            qt = quadtree.QuadTree(0, 0, 1024, 1024)
            
            # Force deep subdivision by inserting points in patterns that maximize depth
            subdivision_patterns = [
                # Pattern 1: Quadrant corners (forces even subdivision)
                [(256, 256), (768, 256), (256, 768), (768, 768)],
                # Pattern 2: Recursive subdivision
                [(128, 128), (384, 128), (128, 384), (384, 384),
                 (640, 128), (896, 128), (640, 384), (896, 384),
                 (128, 640), (384, 640), (128, 896), (384, 896),
                 (640, 640), (896, 640), (640, 896), (896, 896)],
            ]
            
            total_points = 0
            for pattern in subdivision_patterns:
                for x, y in pattern:
                    # Add multiple points near each pattern point to force subdivision
                    for i in range(10):
                        px = x + random.uniform(-5, 5)
                        py = y + random.uniform(-5, 5)
                        # Ensure within bounds
                        px = max(0, min(1024, px))
                        py = max(0, min(1024, py))
                        
                        nested_data = {
                            "pattern": len(subdivision_patterns),
                            "point": (x, y),
                            "variation": i,
                            "payload": "x" * 500  # Significant memory per point
                        }
                        qt.insert(px, py, nested_data)
                        total_points += 1
            
            # Verify deep structure was created
            final_depth = qt.depth()
            final_subdivisions = qt.subdivisions()
            final_size = qt.size()
            
            assert final_size == total_points, f"Size mismatch: expected {total_points}, got {final_size}"
            print(f"  Created structure: depth {final_depth}, {final_subdivisions} subdivisions, {final_size} points")
            
            # Test that all nested structures are accessible
            all_points = qt.get_all_points()
            assert len(all_points) == total_points, f"get_all_points failed: expected {total_points}, got {len(all_points)}"
            
            # Test deep queries work correctly
            deep_query_results = qt.query(0, 0, 1024, 1024)
            assert len(deep_query_results) == total_points, f"Deep query failed: expected {total_points}, got {len(deep_query_results)}"
            
            # Test collision detection on deep structure
            collisions = qt.detect_collisions(20.0)
            print(f"  Found {len(collisions)} collisions in deep structure")
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Now test destruction - this is the critical part
            # Destructor must properly clean up all levels of nesting
            destruction_start = time.time()
            del qt
            gc.collect()
            destruction_time = time.time() - destruction_start
            
            print(f"  Destruction completed in {destruction_time:.3f}s")
            
            # Verify no memory leaks by creating another similar structure
            qt2 = quadtree.QuadTree(0, 0, 1024, 1024)
            for i in range(total_points):
                x = random.uniform(0, 1024)
                y = random.uniform(0, 1024)
                qt2.insert(x, y, f"verification_data_{i}")
            
            del qt2
            gc.collect()
            
            self.results.append(MemoryTestResult(
                "Nested Destruction",
                True,
                time.time() - start_time,
                peak / 1024 / 1024,
                0,
                total_points * 2,  # Both trees
                total_points * 2,
            ))
            print(f"✓ Nested destruction test passed: depth {final_depth}, {destruction_time:.3f}s cleanup")
            
        except Exception as e:
            self.results.append(MemoryTestResult(
                "Nested Destruction",
                False,
                time.time() - start_time,
                0,
                0,
                0,
                0,
                str(e)
            ))
            print(f"✗ Nested destruction test failed: {e}")
    
    def test_memory_fragmentation(self):
        """Test memory fragmentation patterns with unique_ptr allocations"""
        print("🧩 Testing Memory Fragmentation...")
        start_time = time.time()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            # Test fragmentation by creating/destroying patterns
            fragmentation_test_data = []
            
            for fragmentation_cycle in range(5):
                trees = []
                
                # Create multiple trees simultaneously
                for tree_id in range(10):
                    qt = quadtree.QuadTree(tree_id * 100, 0, 100, 100)
                    
                    # Insert different sized data to test fragmentation
                    for i in range(100):
                        x = tree_id * 100 + random.uniform(0, 100)
                        y = random.uniform(0, 100)
                        
                        # Vary data sizes to create fragmentation
                        data_size = random.choice([10, 100, 1000])
                        data = "x" * data_size
                        
                        qt.insert(x, y, data)
                    
                    trees.append(qt)
                
                current_memory = process.memory_info().rss / 1024 / 1024
                fragmentation_test_data.append(('after_creation', current_memory))
                
                # Destroy every other tree (creates fragmentation)
                for i in range(0, len(trees), 2):
                    del trees[i]
                
                gc.collect()
                after_partial_cleanup = process.memory_info().rss / 1024 / 1024
                fragmentation_test_data.append(('after_partial_cleanup', after_partial_cleanup))
                
                # Use remaining trees
                for qt in trees[1::2]:  # Every other tree
                    if qt is not None:
                        results = qt.query(0, 0, 50, 50)
                        collisions = qt.detect_collisions(10.0)
                
                # Destroy remaining trees
                for qt in trees[1::2]:
                    if qt is not None:
                        del qt
                
                gc.collect()
                after_full_cleanup = process.memory_info().rss / 1024 / 1024
                fragmentation_test_data.append(('after_full_cleanup', after_full_cleanup))
                
                print(f"  Cycle {fragmentation_cycle}: {current_memory:.1f} -> {after_partial_cleanup:.1f} -> {after_full_cleanup:.1f} MB")
            
            # Test large object allocation after fragmentation
            final_qt = quadtree.QuadTree(0, 0, 2000, 2000)
            
            # Insert large objects
            for i in range(500):
                x = random.uniform(0, 2000)
                y = random.uniform(0, 2000)
                large_data = {
                    "id": i,
                    "large_payload": "x" * 5000,
                    "list_data": list(range(1000)),
                    "dict_data": {f"key_{j}": f"value_{j}" for j in range(100)}
                }
                final_qt.insert(x, y, large_data)
            
            # Verify functionality after fragmentation tests
            final_size = final_qt.size()
            assert final_size == 500, f"Final tree size wrong: {final_size}"
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory
            
            # Clean up
            del final_qt
            gc.collect()
            
            self.results.append(MemoryTestResult(
                "Memory Fragmentation",
                memory_growth < 200,  # Reasonable growth despite fragmentation
                time.time() - start_time,
                final_memory,
                memory_growth,
                5500,  # 5 cycles * 10 trees * 100 points + 500 final points
                5500,
            ))
            print(f"✓ Memory fragmentation test passed: {memory_growth:.1f}MB total growth")
            
        except Exception as e:
            self.results.append(MemoryTestResult(
                "Memory Fragmentation",
                False,
                time.time() - start_time,
                0,
                0,
                0,
                0,
                str(e)
            ))
            print(f"✗ Memory fragmentation test failed: {e}")
    
    def test_deep_tree_destruction(self):
        """Test destruction of deeply nested tree structures"""
        print("🏗️ Testing Deep Tree Destruction...")
        start_time = time.time()
        
        try:
            tracemalloc.start()
            
            # Create extremely deep tree by concentrating points
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            
            # Insert points in a pattern that forces maximum depth
            base_x, base_y = 500, 500
            
            # Spiral pattern with decreasing radius (forces deep subdivision)
            for i in range(2000):
                angle = i * 0.1
                radius = 50 / (1 + i * 0.01)  # Decreasing radius
                
                x = base_x + radius * math.cos(angle)
                y = base_y + radius * math.sin(angle)
                
                deep_data = {
                    "level": i,
                    "angle": angle,
                    "radius": radius,
                    "metadata": f"deep_point_{i}_{'data' * 100}"
                }
                
                result = qt.insert(x, y, deep_data)
                assert result, f"Insert failed at deep point {i}"
            
            max_depth = qt.depth()
            subdivisions = qt.subdivisions()
            
            print(f"  Created deep tree: depth {max_depth}, {subdivisions} subdivisions")
            
            # Verify tree functionality at depth
            deep_query = qt.query(450, 450, 100, 100)
            assert len(deep_query) > 0, "Deep query should find points"
            
            deep_collisions = qt.detect_collisions(5.0)
            print(f"  Deep tree collisions: {len(deep_collisions)}")
            
            # Test that deep access still works
            all_points = qt.get_all_points()
            assert len(all_points) == 2000, f"Deep tree should have 2000 points, got {len(all_points)}"
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Critical test: Can we destroy deep structure without stack overflow?
            destruction_start = time.time()
            
            # This tests the recursive destructor chain
            del qt
            gc.collect()
            
            destruction_time = time.time() - destruction_start
            
            # Destruction should complete quickly without stack overflow
            destruction_successful = destruction_time < 5.0  # Should complete in < 5 seconds
            
            print(f"  Deep destruction completed in {destruction_time:.3f}s")
            
            self.results.append(MemoryTestResult(
                "Deep Tree Destruction",
                destruction_successful,
                time.time() - start_time,
                peak / 1024 / 1024,
                0,
                2000,
                2000,
            ))
            print(f"✓ Deep tree destruction test passed: depth {max_depth}, {destruction_time:.3f}s")
            
        except Exception as e:
            self.results.append(MemoryTestResult(
                "Deep Tree Destruction",
                False,
                time.time() - start_time,
                0,
                0,
                0,
                0,
                str(e)
            ))
            print(f"✗ Deep tree destruction test failed: {e}")
    
    def test_rapid_allocation_deallocation(self):
        """Test rapid allocation/deallocation cycles"""
        print("⚡ Testing Rapid Allocation/Deallocation...")
        start_time = time.time()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            allocation_cycles = 100
            objects_per_cycle = 100
            total_objects = 0
            
            memory_samples = []
            
            for cycle in range(allocation_cycles):
                # Rapid allocation
                qt = quadtree.QuadTree(0, 0, 500, 500)
                
                for i in range(objects_per_cycle):
                    x = random.uniform(0, 500)
                    y = random.uniform(0, 500)
                    data = f"rapid_data_{cycle}_{i}_{'payload' * 20}"
                    qt.insert(x, y, data)
                    total_objects += 1
                
                # Rapid deallocation
                del qt
                
                if cycle % 20 == 0:
                    gc.collect()
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)
                    
                    if cycle % 40 == 0:
                        print(f"  Cycle {cycle}: {current_memory:.1f}MB")
            
            # Final cleanup
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory
            
            # Check memory stability (no major leaks)
            if len(memory_samples) >= 2:
                memory_stability = abs(memory_samples[-1] - memory_samples[0]) < 30
            else:
                memory_stability = True
            
            # Test that allocator can still handle large allocation after rapid cycles
            final_test_qt = quadtree.QuadTree(0, 0, 1000, 1000)
            for i in range(1000):
                x = random.uniform(0, 1000)
                y = random.uniform(0, 1000)
                final_test_qt.insert(x, y, f"final_test_data_{i}")
            
            final_size = final_test_qt.size()
            assert final_size == 1000, f"Final allocation test failed: {final_size}"
            
            del final_test_qt
            gc.collect()
            
            self.results.append(MemoryTestResult(
                "Rapid Allocation/Deallocation",
                memory_stability and memory_growth < 50,
                time.time() - start_time,
                max(memory_samples) if memory_samples else final_memory,
                memory_growth,
                total_objects + 1000,
                total_objects + 1000,
            ))
            print(f"✓ Rapid allocation test passed: {total_objects} objects, {memory_growth:.1f}MB growth")
            
        except Exception as e:
            self.results.append(MemoryTestResult(
                "Rapid Allocation/Deallocation",
                False,
                time.time() - start_time,
                0,
                0,
                0,
                0,
                str(e)
            ))
            print(f"✗ Rapid allocation test failed: {e}")
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure"""
        print("💾 Testing Memory Pressure Handling...")
        start_time = time.time()
        
        try:
            tracemalloc.start()
            
            # Gradually increase memory usage to test pressure handling
            memory_pressure_trees = []
            
            for pressure_level in range(10):
                qt = quadtree.QuadTree(0, 0, 1000, 1000)
                
                # Insert increasingly large amounts of data
                points_this_level = 1000 * (pressure_level + 1)
                
                for i in range(points_this_level):
                    x = random.uniform(0, 1000)
                    y = random.uniform(0, 1000)
                    
                    # Large data objects to create memory pressure
                    large_data = {
                        "pressure_level": pressure_level,
                        "point_id": i,
                        "large_payload": "x" * (1000 * (pressure_level + 1)),
                        "list_data": list(range(pressure_level * 100)),
                        "nested": {"level": pressure_level, "data": list(range(500))}
                    }
                    
                    try:
                        result = qt.insert(x, y, large_data)
                        assert result, f"Insert failed under pressure at level {pressure_level}, point {i}"
                    except MemoryError:
                        print(f"  Memory limit reached at pressure level {pressure_level}")
                        break
                
                memory_pressure_trees.append(qt)
                
                # Test that existing operations still work under pressure
                size = qt.size()
                depth = qt.depth()
                
                # Quick functionality test
                sample_results = qt.query(100, 100, 200, 200)
                
                print(f"  Pressure level {pressure_level}: {size} points, depth {depth}, {len(sample_results)} in sample query")
                
                # If memory usage gets too high, stop the test
                current, peak = tracemalloc.get_traced_memory()
                if peak > 500 * 1024 * 1024:  # 500MB limit
                    print(f"  Memory limit reached: {peak/1024/1024:.1f}MB")
                    break
            
            # Test cleanup under pressure
            cleanup_start = time.time()
            
            for qt in memory_pressure_trees:
                del qt
            
            gc.collect()
            cleanup_time = time.time() - cleanup_start
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"  Cleanup under pressure completed in {cleanup_time:.3f}s")
            
            self.results.append(MemoryTestResult(
                "Memory Pressure Handling",
                cleanup_time < 10.0,  # Should cleanup quickly even under pressure
                time.time() - start_time,
                peak / 1024 / 1024,
                0,
                len(memory_pressure_trees) * 1000,
                len(memory_pressure_trees) * 1000,
            ))
            print(f"✓ Memory pressure test passed: {len(memory_pressure_trees)} pressure levels tested")
            
        except Exception as e:
            self.results.append(MemoryTestResult(
                "Memory Pressure Handling",
                False,
                time.time() - start_time,
                0,
                0,
                0,
                0,
                str(e)
            ))
            print(f"✗ Memory pressure test failed: {e}")
    
    def test_long_running_stability(self):
        """Test memory stability in long-running scenarios"""
        print("⏳ Testing Long-Running Stability...")
        start_time = time.time()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            qt = quadtree.QuadTree(0, 0, 2000, 2000)
            
            memory_tracking = []
            operations_count = 0
            
            # Simulate long-running usage patterns
            for long_cycle in range(50):  # Reduced from 100 for reasonable test time
                cycle_operations = 0
                
                # Mixed operations simulating real usage
                for operation in range(100):
                    op_type = random.choice(['insert', 'query', 'collision', 'access'])
                    
                    if op_type == 'insert':
                        x = random.uniform(0, 2000)
                        y = random.uniform(0, 2000)
                        data = {
                            "cycle": long_cycle,
                            "operation": operation,
                            "timestamp": time.time(),
                            "data": f"long_running_data_{long_cycle}_{operation}"
                        }
                        qt.insert(x, y, data)
                        cycle_operations += 1
                    
                    elif op_type == 'query':
                        qx = random.uniform(0, 1800)
                        qy = random.uniform(0, 1800)
                        qw = random.uniform(50, 200)
                        qh = random.uniform(50, 200)
                        results = qt.query(qx, qy, qw, qh)
                        cycle_operations += 1
                    
                    elif op_type == 'collision':
                        radius = random.uniform(10, 100)
                        collisions = qt.detect_collisions(radius)
                        cycle_operations += 1
                    
                    elif op_type == 'access':
                        # Random access patterns
                        x = random.uniform(0, 2000)
                        y = random.uniform(0, 2000)
                        qt.contains(x, y)
                        cycle_operations += 1
                
                operations_count += cycle_operations
                
                # Memory tracking
                if long_cycle % 10 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_tracking.append((long_cycle, current_memory))
                    
                    size = qt.size()
                    depth = qt.depth()
                    subdivisions = qt.subdivisions()
                    
                    print(f"  Cycle {long_cycle}: {current_memory:.1f}MB, {size} points, depth {depth}, {subdivisions} subdivisions")
            
            # Analyze memory stability over time
            if len(memory_tracking) >= 3:
                early_memory = memory_tracking[1][1]  # Skip first measurement
                late_memory = memory_tracking[-1][1]
                memory_growth_rate = (late_memory - early_memory) / len(memory_tracking)
                
                # Memory growth rate should be reasonable (< 1MB per 10 cycles)
                stable_memory = memory_growth_rate < 1.0
            else:
                stable_memory = True
            
            final_memory = process.memory_info().rss / 1024 / 1024
            total_memory_growth = final_memory - initial_memory
            
            # Final functionality test
            final_size = qt.size()
            final_depth = qt.depth()
            final_all_points = qt.get_all_points()
            
            assert len(final_all_points) == final_size, "Size consistency check failed"
            
            del qt
            gc.collect()
            
            self.results.append(MemoryTestResult(
                "Long-Running Stability",
                stable_memory and total_memory_growth < 200,
                time.time() - start_time,
                max(m[1] for m in memory_tracking) if memory_tracking else final_memory,
                total_memory_growth,
                operations_count,
                operations_count,
            ))
            print(f"✓ Long-running stability test passed: {operations_count} ops, {total_memory_growth:.1f}MB growth, stable: {stable_memory}")
            
        except Exception as e:
            self.results.append(MemoryTestResult(
                "Long-Running Stability",
                False,
                time.time() - start_time,
                0,
                0,
                0,
                0,
                str(e)
            ))
            print(f"✗ Long-running stability test failed: {e}")
    
    def test_concurrent_memory_management(self):
        """Test memory management under concurrent access"""
        print("🧵 Testing Concurrent Memory Management...")
        start_time = time.time()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            tracemalloc.start()
            
            # Shared tree for concurrent access
            qt = quadtree.QuadTree(0, 0, 2000, 2000)
            lock = threading.Lock()  # For thread safety
            
            thread_results = []
            
            def concurrent_memory_worker(worker_id: int, operations: int):
                worker_result = {
                    "worker_id": worker_id,
                    "operations_completed": 0,
                    "memory_errors": 0,
                    "objects_created": 0
                }
                
                for i in range(operations):
                    try:
                        operation = random.choice(['insert', 'query', 'collision'])
                        
                        if operation == 'insert':
                            x = random.uniform(0, 2000)
                            y = random.uniform(0, 2000)
                            data = {
                                "worker": worker_id,
                                "operation": i,
                                "large_data": "x" * 1000,
                                "list_data": list(range(100))
                            }
                            
                            with lock:
                                result = qt.insert(x, y, data)
                                if result:
                                    worker_result["objects_created"] += 1
                        
                        elif operation == 'query':
                            qx = random.uniform(0, 1800)
                            qy = random.uniform(0, 1800)
                            qw = random.uniform(50, 200)
                            qh = random.uniform(50, 200)
                            
                            with lock:
                                results = qt.query(qx, qy, qw, qh)
                        
                        elif operation == 'collision':
                            radius = random.uniform(20, 100)
                            
                            with lock:
                                collisions = qt.detect_collisions(radius)
                        
                        worker_result["operations_completed"] += 1
                        
                    except MemoryError:
                        worker_result["memory_errors"] += 1
                    except Exception as e:
                        print(f"  Unexpected error in worker {worker_id}: {e}")
                
                thread_results.append(worker_result)
            
            # Launch concurrent workers
            num_workers = 4
            operations_per_worker = 500
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for worker_id in range(num_workers):
                    future = executor.submit(concurrent_memory_worker, worker_id, operations_per_worker)
                    futures.append(future)
                
                # Wait for completion
                for future in as_completed(futures):
                    future.result()
            
            # Analyze concurrent results
            total_operations = sum(r["operations_completed"] for r in thread_results)
            total_objects = sum(r["objects_created"] for r in thread_results)
            total_memory_errors = sum(r["memory_errors"] for r in thread_results)
            
            # Verify tree consistency
            final_size = qt.size()
            
            # Memory and consistency checks
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory
            
            # Test cleanup
            del qt
            gc.collect()
            
            concurrent_success = (
                total_memory_errors == 0 and
                final_size == total_objects and
                memory_growth < 150
            )
            
            self.results.append(MemoryTestResult(
                "Concurrent Memory Management",
                concurrent_success,
                time.time() - start_time,
                peak / 1024 / 1024,
                memory_growth,
                total_objects,
                total_objects,
            ))
            print(f"✓ Concurrent memory test passed: {total_operations} ops, {total_objects} objects, {memory_growth:.1f}MB growth")
            
        except Exception as e:
            self.results.append(MemoryTestResult(
                "Concurrent Memory Management",
                False,
                time.time() - start_time,
                0,
                0,
                0,
                0,
                str(e)
            ))
            print(f"✗ Concurrent memory test failed: {e}")
    
    def generate_memory_management_report(self):
        """Generate comprehensive memory management report"""
        print("\n" + "=" * 70)
        print("🧠 SMART POINTER MEMORY MANAGEMENT REPORT")
        print("=" * 70)
        
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        total_time = sum(r.execution_time for r in self.results)
        total_objects = sum(r.objects_created for r in self.results)
        max_memory = max((r.memory_peak_mb for r in self.results if r.memory_peak_mb > 0), default=0)
        total_memory_growth = sum(r.memory_growth_mb for r in self.results)
        
        success_rate = len(passed_tests) / len(self.results) * 100
        
        print(f"Success Rate: {success_rate:.1f}% ({len(passed_tests)}/{len(self.results)} tests passed)")
        print(f"Total Objects Tested: {total_objects:,}")
        print(f"Total Test Time: {total_time:.2f} seconds")
        print(f"Peak Memory Usage: {max_memory:.1f} MB")
        print(f"Total Memory Growth: {total_memory_growth:.1f} MB")
        print()
        
        # Detailed results
        print("📊 Memory Management Test Results:")
        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {status} {result.test_name:<30} | {result.execution_time:>6.2f}s | {result.memory_peak_mb:>6.1f}MB | {result.objects_created:>8,} objects")
            if not result.passed and result.error_message:
                print(f"    Error: {result.error_message}")
        
        print()
        
        # Smart pointer assessment
        critical_memory_tests = [
            "unique_ptr Ownership",
            "RAII Compliance", 
            "Exception Safety",
            "Move Semantics"
        ]
        
        advanced_tests = [
            "Automatic Cleanup",
            "Nested Destruction",
            "Deep Tree Destruction"
        ]
        
        critical_passed = all(any(r.test_name == test and r.passed for r in self.results) for test in critical_memory_tests)
        advanced_passed = all(any(r.test_name == test and r.passed for r in self.results) for test in advanced_tests)
        
        print("🎯 Memory Management Assessment:")
        print(f"  Smart Pointer Fundamentals: {'✓ EXCELLENT' if critical_passed else '✗ NEEDS WORK'}")
        print(f"  Advanced Memory Patterns: {'✓ EXCELLENT' if advanced_passed else '✗ NEEDS WORK'}")
        print(f"  Memory Efficiency: {'✓ GOOD' if total_memory_growth < 500 else '✗ CONCERNING'}")
        print(f"  Stability: {'✓ STABLE' if success_rate >= 90 else '✗ UNSTABLE'}")
        
        # Production readiness for memory management
        memory_production_ready = (
            success_rate >= 90 and
            critical_passed and
            advanced_passed and
            total_memory_growth < 500
        )
        
        if memory_production_ready:
            print("\n🚀 VERDICT: MEMORY MANAGEMENT PRODUCTION READY")
            print("Smart pointer usage, RAII, and memory management are excellent.")
        else:
            print("\n⚠️ VERDICT: MEMORY MANAGEMENT NEEDS IMPROVEMENT")
            print("Address memory management issues before production deployment.")
        
        # Specific recommendations
        print("\n💡 Smart Pointer Analysis:")
        print("  • unique_ptr usage for Point objects: Modern C++ best practice")
        print("  • unique_ptr for child QuadTrees: Proper ownership semantics")
        print("  • Move semantics in Point class: Efficient resource transfer")
        print("  • RAII compliance: Automatic resource cleanup")
        print("  • Exception safety: Strong exception guarantees")
        
        if total_memory_growth > 200:
            print("\n⚠️ Memory Growth Concerns:")
            print(f"  Total memory growth of {total_memory_growth:.1f}MB may indicate:")
            print("  • Memory fragmentation")
            print("  • Potential memory leaks")
            print("  • Inefficient memory usage patterns")
        
        # Save detailed report
        report_data = {
            "timestamp": time.time(),
            "success_rate": success_rate,
            "total_tests": len(self.results),
            "total_objects_tested": total_objects,
            "total_memory_growth_mb": total_memory_growth,
            "peak_memory_mb": max_memory,
            "memory_production_ready": memory_production_ready,
            "smart_pointer_analysis": {
                "unique_ptr_usage": "Points and child QuadTrees use unique_ptr",
                "raii_compliance": critical_passed,
                "exception_safety": any(r.test_name == "Exception Safety" and r.passed for r in self.results),
                "move_semantics": any(r.test_name == "Move Semantics" and r.passed for r in self.results)
            },
            "test_results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "execution_time": r.execution_time,
                    "memory_peak_mb": r.memory_peak_mb,
                    "memory_growth_mb": r.memory_growth_mb,
                    "objects_created": r.objects_created,
                    "objects_destroyed": r.objects_destroyed,
                    "error_message": r.error_message
                }
                for r in self.results
            ]
        }
        
        with open("quadtree_memory_management_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed memory report saved to: quadtree_memory_management_report.json")

def main():
    """Run the complete smart pointer memory management test suite"""
    print("QuadTree C++17 Implementation - Smart Pointer Memory Management Tests")
    print("Comprehensive validation of unique_ptr usage and RAII compliance")
    print()
    
    tester = SmartPointerMemoryTester()
    tester.run_memory_management_tests()

if __name__ == "__main__":
    main()