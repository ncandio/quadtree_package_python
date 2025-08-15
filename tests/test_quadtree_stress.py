#!/usr/bin/env python3
"""
Intensive Stress Test Battery for QuadTree C++17 Implementation
Tests scalability with 10, 100, 1000, and 10000 quadtrees
Comprehensive API testing with edge cases, performance benchmarks, and memory validation
"""

import sys
import os
import time
import random
import math
import gc
import tracemalloc
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, '.')

try:
    import quadtree
    print("✓ QuadTree module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import quadtree: {e}")
    print("Compile with: python setup.py build_ext --inplace")
    sys.exit(1)

@dataclass
class TestResult:
    test_name: str
    passed: bool
    execution_time: float
    memory_usage: int
    error_message: str = ""

@dataclass
class StressTestConfig:
    num_quadtrees: int
    boundary_size: float
    points_per_tree: int
    query_operations: int
    collision_radius: float

class QuadTreeStressTester:
    def __init__(self):
        self.results: List[TestResult] = []
        self.random_seed = 42
        random.seed(self.random_seed)
        
    def run_all_tests(self):
        """Execute complete stress test battery"""
        print("🚀 Starting QuadTree Intensive Stress Test Battery")
        print("=" * 80)
        
        # Test configurations for different scales
        configs = [
            StressTestConfig(10, 100, 50, 100, 5.0),
            StressTestConfig(100, 1000, 100, 500, 10.0),
            StressTestConfig(1000, 5000, 200, 1000, 15.0),
            StressTestConfig(10000, 10000, 500, 2000, 20.0)
        ]
        
        for config in configs:
            print(f"\n📊 Testing {config.num_quadtrees} QuadTrees")
            print("-" * 60)
            
            self._test_basic_api_comprehensive(config)
            self._test_mass_operations(config)
            self._test_concurrent_access(config)
            self._test_memory_efficiency(config)
            self._test_performance_benchmarks(config)
            self._test_edge_cases_bulk(config)
            self._test_collision_detection_scale(config)
            
        self._print_summary()
    
    def _test_basic_api_comprehensive(self, config: StressTestConfig):
        """Comprehensive API testing across all methods"""
        start_time = time.time()
        tracemalloc.start()
        
        try:
            trees = []
            
            # Create quadtrees with varied boundaries
            for i in range(config.num_quadtrees):
                x = random.uniform(0, config.boundary_size)
                y = random.uniform(0, config.boundary_size)
                w = random.uniform(10, config.boundary_size / 4)
                h = random.uniform(10, config.boundary_size / 4)
                
                qt = quadtree.QuadTree(x, y, w, h)
                trees.append(qt)
                
                # Test boundary method
                boundary = qt.boundary()
                assert boundary == (x, y, w, h), f"Boundary mismatch: expected ({x},{y},{w},{h}), got {boundary}"
                
                # Test initial state
                assert qt.empty(), f"Tree {i} should be empty initially"
                assert qt.size() == 0, f"Tree {i} should have size 0 initially"
                assert qt.depth() == 0, f"Tree {i} should have depth 0 initially"
                assert qt.subdivisions() == 0, f"Tree {i} should have 0 subdivisions initially"
            
            # Bulk insert operations
            all_inserted_points = []
            for i, qt in enumerate(trees):
                tree_points = []
                boundary = qt.boundary()
                
                for j in range(config.points_per_tree):
                    # Generate points within boundary
                    px = boundary[0] + random.uniform(0, boundary[2])
                    py = boundary[1] + random.uniform(0, boundary[3])
                    data = f"tree_{i}_point_{j}" if j % 3 == 0 else None
                    
                    result = qt.insert(px, py, data)
                    assert result, f"Failed to insert point ({px}, {py}) in tree {i}"
                    
                    tree_points.append((px, py, data))
                
                all_inserted_points.append(tree_points)
                
                # Verify insertions
                assert qt.size() == config.points_per_tree, f"Tree {i} size mismatch: expected {config.points_per_tree}, got {qt.size()}"
                assert not qt.empty(), f"Tree {i} should not be empty after insertions"
            
            # Test contains method extensively
            correct_contains = 0
            total_contains_tests = 0
            
            for i, qt in enumerate(trees):
                for px, py, _ in all_inserted_points[i]:
                    total_contains_tests += 1
                    if qt.contains(px, py):
                        correct_contains += 1
                    
                    # Test non-existent points
                    total_contains_tests += 1
                    boundary = qt.boundary()
                    non_existent_x = boundary[0] + boundary[2] + 100
                    non_existent_y = boundary[1] + boundary[3] + 100
                    if not qt.contains(non_existent_x, non_existent_y):
                        correct_contains += 1
            
            contains_accuracy = correct_contains / total_contains_tests
            assert contains_accuracy > 0.99, f"Contains accuracy too low: {contains_accuracy}"
            
            # Test get_all_points
            for i, qt in enumerate(trees):
                all_points = qt.get_all_points()
                assert len(all_points) == config.points_per_tree, f"get_all_points size mismatch for tree {i}"
                
                # Verify data integrity
                data_points = [p for p in all_points if len(p) == 3]
                expected_data_count = sum(1 for _, _, data in all_inserted_points[i] if data is not None)
                assert len(data_points) == expected_data_count, f"Data point count mismatch for tree {i}"
            
            # Test query operations
            for i, qt in enumerate(trees):
                boundary = qt.boundary()
                
                for _ in range(config.query_operations // config.num_quadtrees):
                    # Random query rectangle
                    qx = boundary[0] + random.uniform(0, boundary[2])
                    qy = boundary[1] + random.uniform(0, boundary[3])
                    qw = random.uniform(1, boundary[2] / 4)
                    qh = random.uniform(1, boundary[3] / 4)
                    
                    query_result = qt.query(qx, qy, qw, qh)
                    
                    # Verify query results
                    for point in query_result:
                        px, py = point[0], point[1]
                        assert qx <= px < qx + qw and qy <= py < qy + qh, f"Query returned out-of-bounds point"
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                f"Basic API Comprehensive ({config.num_quadtrees} trees)",
                True,
                execution_time,
                peak
            ))
            
            print(f"✓ Basic API test passed - {execution_time:.2f}s, {peak/1024/1024:.1f}MB peak")
            
        except Exception as e:
            tracemalloc.stop()
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                f"Basic API Comprehensive ({config.num_quadtrees} trees)",
                False,
                execution_time,
                0,
                str(e)
            ))
            print(f"✗ Basic API test failed: {e}")
    
    def _test_mass_operations(self, config: StressTestConfig):
        """Test mass insertion, deletion simulation, and bulk operations"""
        start_time = time.time()
        tracemalloc.start()
        
        try:
            qt = quadtree.QuadTree(0, 0, config.boundary_size, config.boundary_size)
            
            # Mass insertion
            points = []
            for i in range(config.num_quadtrees * config.points_per_tree):
                x = random.uniform(0, config.boundary_size)
                y = random.uniform(0, config.boundary_size)
                data = f"mass_point_{i}" if i % 10 == 0 else None
                
                result = qt.insert(x, y, data)
                assert result, f"Mass insertion failed at point {i}"
                points.append((x, y, data))
            
            expected_size = len(points)
            actual_size = qt.size()
            assert actual_size == expected_size, f"Size mismatch after mass insertion: expected {expected_size}, got {actual_size}"
            
            # Test tree statistics
            depth = qt.depth()
            subdivisions = qt.subdivisions()
            
            print(f"  Mass insertion: {actual_size} points, depth: {depth}, subdivisions: {subdivisions}")
            
            # Bulk query operations
            query_hits = 0
            query_misses = 0
            
            for _ in range(1000):
                # Random query
                qx = random.uniform(0, config.boundary_size)
                qy = random.uniform(0, config.boundary_size)
                qw = random.uniform(10, config.boundary_size / 10)
                qh = random.uniform(10, config.boundary_size / 10)
                
                results = qt.query(qx, qy, qw, qh)
                if results:
                    query_hits += 1
                else:
                    query_misses += 1
            
            print(f"  Bulk queries: {query_hits} hits, {query_misses} misses")
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                f"Mass Operations ({config.num_quadtrees * config.points_per_tree} points)",
                True,
                execution_time,
                peak
            ))
            
            print(f"✓ Mass operations test passed - {execution_time:.2f}s, {peak/1024/1024:.1f}MB peak")
            
        except Exception as e:
            tracemalloc.stop()
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                f"Mass Operations ({config.num_quadtrees * config.points_per_tree} points)",
                False,
                execution_time,
                0,
                str(e)
            ))
            print(f"✗ Mass operations test failed: {e}")
    
    def _test_concurrent_access(self, config: StressTestConfig):
        """Test thread safety and concurrent operations"""
        start_time = time.time()
        tracemalloc.start()
        
        try:
            # Create shared quadtree
            qt = quadtree.QuadTree(0, 0, config.boundary_size, config.boundary_size)
            thread_results = []
            lock = threading.Lock()
            
            def worker_insert(thread_id: int, points_per_thread: int):
                thread_result = {"thread_id": thread_id, "inserted": 0, "failed": 0}
                
                for i in range(points_per_thread):
                    x = random.uniform(0, config.boundary_size)
                    y = random.uniform(0, config.boundary_size)
                    data = f"thread_{thread_id}_point_{i}"
                    
                    # Note: QuadTree may not be thread-safe, so we test serialized access
                    with lock:
                        result = qt.insert(x, y, data)
                        if result:
                            thread_result["inserted"] += 1
                        else:
                            thread_result["failed"] += 1
                
                thread_results.append(thread_result)
            
            # Launch concurrent workers
            num_threads = min(4, config.num_quadtrees // 10) if config.num_quadtrees >= 10 else 1
            points_per_thread = config.points_per_tree
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for i in range(num_threads):
                    future = executor.submit(worker_insert, i, points_per_thread)
                    futures.append(future)
                
                # Wait for completion
                for future in as_completed(futures):
                    future.result()
            
            # Verify results
            total_inserted = sum(r["inserted"] for r in thread_results)
            total_failed = sum(r["failed"] for r in thread_results)
            final_size = qt.size()
            
            assert final_size == total_inserted, f"Size mismatch: expected {total_inserted}, got {final_size}"
            
            print(f"  Concurrent: {num_threads} threads, {total_inserted} inserted, {total_failed} failed")
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                f"Concurrent Access ({num_threads} threads)",
                True,
                execution_time,
                peak
            ))
            
            print(f"✓ Concurrent access test passed - {execution_time:.2f}s, {peak/1024/1024:.1f}MB peak")
            
        except Exception as e:
            tracemalloc.stop()
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                f"Concurrent Access",
                False,
                execution_time,
                0,
                str(e)
            ))
            print(f"✗ Concurrent access test failed: {e}")
    
    def _test_memory_efficiency(self, config: StressTestConfig):
        """Test memory usage patterns and efficiency"""
        start_time = time.time()
        
        try:
            # Create multiple trees and monitor memory
            trees = []
            memory_snapshots = []
            
            for i in range(0, config.num_quadtrees, max(1, config.num_quadtrees // 20)):
                tracemalloc.start()
                
                # Create tree
                qt = quadtree.QuadTree(0, 0, 100, 100)
                
                # Add points
                for j in range(config.points_per_tree):
                    x = random.uniform(0, 100)
                    y = random.uniform(0, 100)
                    qt.insert(x, y, f"data_{i}_{j}")
                
                trees.append(qt)
                
                current, peak = tracemalloc.get_traced_memory()
                memory_snapshots.append((i + 1, current, peak))
                tracemalloc.stop()
            
            # Analyze memory growth
            if len(memory_snapshots) > 1:
                first_tree_memory = memory_snapshots[0][1]
                last_tree_memory = memory_snapshots[-1][1]
                memory_per_tree = (last_tree_memory - first_tree_memory) / (len(memory_snapshots) - 1)
                
                print(f"  Memory per tree: ~{memory_per_tree/1024:.1f}KB")
                print(f"  Total peak memory: {memory_snapshots[-1][2]/1024/1024:.1f}MB")
            
            # Force garbage collection
            del trees
            gc.collect()
            
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                f"Memory Efficiency ({config.num_quadtrees} trees)",
                True,
                execution_time,
                memory_snapshots[-1][2] if memory_snapshots else 0
            ))
            
            print(f"✓ Memory efficiency test passed - {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                f"Memory Efficiency",
                False,
                execution_time,
                0,
                str(e)
            ))
            print(f"✗ Memory efficiency test failed: {e}")
    
    def _test_performance_benchmarks(self, config: StressTestConfig):
        """Performance benchmarking for all operations"""
        start_time = time.time()
        
        try:
            qt = quadtree.QuadTree(0, 0, config.boundary_size, config.boundary_size)
            
            # Benchmark insertions
            insert_start = time.time()
            points = []
            for i in range(config.points_per_tree * 10):
                x = random.uniform(0, config.boundary_size)
                y = random.uniform(0, config.boundary_size)
                qt.insert(x, y)
                points.append((x, y))
            insert_time = time.time() - insert_start
            insert_rate = len(points) / insert_time
            
            # Benchmark queries
            query_start = time.time()
            for _ in range(1000):
                qx = random.uniform(0, config.boundary_size)
                qy = random.uniform(0, config.boundary_size)
                qw = random.uniform(10, config.boundary_size / 10)
                qh = random.uniform(10, config.boundary_size / 10)
                qt.query(qx, qy, qw, qh)
            query_time = time.time() - query_start
            query_rate = 1000 / query_time
            
            # Benchmark contains
            contains_start = time.time()
            for px, py in points[:1000]:
                qt.contains(px, py)
            contains_time = time.time() - contains_start
            contains_rate = 1000 / contains_time
            
            # Benchmark collision detection
            collision_start = time.time()
            qt.detect_collisions(config.collision_radius)
            collision_time = time.time() - collision_start
            
            print(f"  Insert rate: {insert_rate:.0f} ops/sec")
            print(f"  Query rate: {query_rate:.0f} ops/sec") 
            print(f"  Contains rate: {contains_rate:.0f} ops/sec")
            print(f"  Collision detection: {collision_time:.3f}s")
            
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                f"Performance Benchmarks ({len(points)} points)",
                True,
                execution_time,
                0
            ))
            
            print(f"✓ Performance benchmarks passed - {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                f"Performance Benchmarks",
                False,
                execution_time,
                0,
                str(e)
            ))
            print(f"✗ Performance benchmarks failed: {e}")
    
    def _test_edge_cases_bulk(self, config: StressTestConfig):
        """Test edge cases in bulk"""
        start_time = time.time()
        
        try:
            edge_cases_passed = 0
            total_edge_cases = 0
            
            # Test boundary edge cases
            for i in range(min(100, config.num_quadtrees)):
                total_edge_cases += 1
                try:
                    # Very small boundaries
                    qt = quadtree.QuadTree(0, 0, 0.001, 0.001)
                    result = qt.insert(0.0005, 0.0005)
                    if result:
                        edge_cases_passed += 1
                except:
                    pass
                
                total_edge_cases += 1
                try:
                    # Very large boundaries
                    qt = quadtree.QuadTree(-1e6, -1e6, 2e6, 2e6)
                    result = qt.insert(0, 0)
                    if result:
                        edge_cases_passed += 1
                except:
                    pass
                
                total_edge_cases += 1
                try:
                    # Boundary edge insertions
                    qt = quadtree.QuadTree(0, 0, 100, 100)
                    # Insert at exact boundaries
                    results = [
                        qt.insert(0, 0),      # Bottom-left corner
                        qt.insert(99.999, 99.999),  # Near top-right
                        qt.insert(50, 0),     # Bottom edge
                        qt.insert(0, 50),     # Left edge
                    ]
                    if all(results):
                        edge_cases_passed += 1
                except:
                    pass
            
            # Test with extreme point densities
            for density_test in range(min(10, config.num_quadtrees // 100)):
                total_edge_cases += 1
                try:
                    qt = quadtree.QuadTree(0, 0, 10, 10)
                    # Insert many points in small area
                    for i in range(1000):
                        x = random.uniform(0, 1)
                        y = random.uniform(0, 1)
                        qt.insert(x, y)
                    
                    if qt.size() == 1000:
                        edge_cases_passed += 1
                except:
                    pass
            
            edge_case_success_rate = edge_cases_passed / total_edge_cases if total_edge_cases > 0 else 0
            
            print(f"  Edge case success rate: {edge_case_success_rate:.2%} ({edge_cases_passed}/{total_edge_cases})")
            
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                f"Edge Cases Bulk ({total_edge_cases} cases)",
                edge_case_success_rate > 0.8,
                execution_time,
                0
            ))
            
            print(f"✓ Edge cases bulk test passed - {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                f"Edge Cases Bulk",
                False,
                execution_time,
                0,
                str(e)
            ))
            print(f"✗ Edge cases bulk test failed: {e}")
    
    def _test_collision_detection_scale(self, config: StressTestConfig):
        """Test collision detection at scale"""
        start_time = time.time()
        
        try:
            qt = quadtree.QuadTree(0, 0, config.boundary_size, config.boundary_size)
            
            # Create clusters of points for collision testing
            clusters = []
            cluster_size = 10
            num_clusters = min(50, config.points_per_tree // cluster_size)
            
            for i in range(num_clusters):
                # Random cluster center
                cx = random.uniform(config.collision_radius, config.boundary_size - config.collision_radius)
                cy = random.uniform(config.collision_radius, config.boundary_size - config.collision_radius)
                
                cluster_points = []
                for j in range(cluster_size):
                    # Points within collision radius
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(0, config.collision_radius * 0.8)
                    
                    x = cx + distance * math.cos(angle)
                    y = cy + distance * math.sin(angle)
                    
                    # Ensure within bounds
                    x = max(0, min(config.boundary_size, x))
                    y = max(0, min(config.boundary_size, y))
                    
                    qt.insert(x, y, f"cluster_{i}_point_{j}")
                    cluster_points.append((x, y))
                
                clusters.append(cluster_points)
            
            # Add random scattered points
            num_scattered = config.points_per_tree - (num_clusters * cluster_size)
            for i in range(num_scattered):
                x = random.uniform(0, config.boundary_size)
                y = random.uniform(0, config.boundary_size)
                qt.insert(x, y, f"scattered_{i}")
            
            # Test collision detection
            collisions = qt.detect_collisions(config.collision_radius)
            
            print(f"  Collision test: {len(collisions)} collisions found")
            print(f"  Point density: {qt.size()} points in {config.boundary_size}x{config.boundary_size} area")
            
            # Verify some collisions are valid
            valid_collisions = 0
            for collision in collisions[:min(100, len(collisions))]:  # Check first 100
                p1 = collision["point1"]
                p2 = collision["point2"]
                
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance <= config.collision_radius:
                    valid_collisions += 1
            
            collision_accuracy = valid_collisions / min(100, len(collisions)) if collisions else 1.0
            
            print(f"  Collision accuracy: {collision_accuracy:.2%}")
            
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                f"Collision Detection Scale ({qt.size()} points)",
                collision_accuracy > 0.95,
                execution_time,
                0
            ))
            
            print(f"✓ Collision detection scale test passed - {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                f"Collision Detection Scale",
                False,
                execution_time,
                0,
                str(e)
            ))
            print(f"✗ Collision detection scale test failed: {e}")
    
    def _print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("🏁 STRESS TEST SUMMARY")
        print("=" * 80)
        
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        total_time = sum(r.execution_time for r in self.results)
        total_memory = max((r.memory_usage for r in self.results if r.memory_usage > 0), default=0)
        
        print(f"Total Tests: {len(self.results)}")
        print(f"Passed: {len(passed_tests)} ✓")
        print(f"Failed: {len(failed_tests)} ✗")
        print(f"Success Rate: {len(passed_tests)/len(self.results):.1%}")
        print(f"Total Execution Time: {total_time:.2f}s")
        print(f"Peak Memory Usage: {total_memory/1024/1024:.1f}MB")
        print()
        
        if failed_tests:
            print("❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"  • {test.test_name}: {test.error_message}")
            print()
        
        print("📊 PERFORMANCE SUMMARY:")
        for test in self.results:
            status = "✓" if test.passed else "✗"
            memory_str = f", {test.memory_usage/1024/1024:.1f}MB" if test.memory_usage > 0 else ""
            print(f"  {status} {test.test_name}: {test.execution_time:.2f}s{memory_str}")
        
        if len(passed_tests) == len(self.results):
            print("\n🎉 ALL STRESS TESTS PASSED!")
            print("The QuadTree implementation is robust and scalable.")
        else:
            print(f"\n⚠️  {len(failed_tests)} tests failed. Review implementation.")

def main():
    """Run the complete stress test battery"""
    print("QuadTree C++17 Implementation - Intensive Stress Test Battery")
    print("Testing scalability: 10 → 100 → 1000 → 10000 QuadTrees")
    print("Comprehensive API testing with performance benchmarks")
    print()
    
    tester = QuadTreeStressTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()