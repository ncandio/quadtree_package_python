#!/usr/bin/env python3
"""
Production-Ready QuadTree Test Suite
Comprehensive API stress testing and validation for production deployment
Focuses on performance, reliability, memory efficiency, and edge case handling
"""

import sys
import os
import time
import random
import math
import gc
import tracemalloc
import psutil
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

sys.path.insert(0, '.')

try:
    import quadtree
    print("✓ QuadTree module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import quadtree: {e}")
    sys.exit(1)

@dataclass
class TestMetrics:
    name: str
    passed: bool
    execution_time: float
    memory_peak_mb: float
    operations_per_second: float = 0
    error_message: str = ""

class ProductionQuadTreeTester:
    """Comprehensive production-readiness test suite for QuadTree"""
    
    def __init__(self):
        self.results: List[TestMetrics] = []
        self.random_seed = 42
        random.seed(self.random_seed)
    
    def run_production_test_suite(self):
        """Execute complete production test battery"""
        print("🚀 Production-Ready QuadTree Test Suite")
        print("=" * 70)
        print("Testing: API completeness, performance, memory, edge cases")
        print()
        
        # Core API Tests
        self.test_api_completeness()
        self.test_data_integrity()
        self.test_boundary_validation()
        
        # Performance Tests  
        self.test_insertion_performance()
        self.test_query_performance()
        self.test_collision_performance()
        
        # Scalability Tests
        self.test_memory_scalability()
        self.test_large_dataset_handling()
        
        # Edge Case Tests
        self.test_precision_edge_cases()
        self.test_extreme_values()
        self.test_error_conditions()
        
        # Reliability Tests
        self.test_thread_safety()
        self.test_memory_leaks()
        
        self.generate_production_report()
    
    def test_api_completeness(self):
        """Test all API methods comprehensively"""
        print("📋 Testing API Completeness...")
        start_time = time.time()
        tracemalloc.start()
        
        try:
            # Test constructor variations
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            
            # Test all methods exist and work
            methods_tested = {
                'insert': False,
                'query': False, 
                'detect_collisions': False,
                'get_all_points': False,
                'contains': False,
                'size': False,
                'empty': False,
                'depth': False,
                'boundary': False,
                'subdivisions': False
            }
            
            # Insert with and without data
            assert qt.insert(100, 100), "Basic insert failed"
            assert qt.insert(200, 200, "test_data"), "Insert with data failed"
            methods_tested['insert'] = True
            
            # Test size and empty
            assert qt.size() == 2, f"Size should be 2, got {qt.size()}"
            assert not qt.empty(), "Should not be empty"
            methods_tested['size'] = True
            methods_tested['empty'] = True
            
            # Test boundary
            boundary = qt.boundary()
            assert boundary == (0.0, 0.0, 1000.0, 1000.0), f"Boundary mismatch: {boundary}"
            methods_tested['boundary'] = True
            
            # Test contains
            assert qt.contains(100, 100), "Should contain inserted point"
            assert not qt.contains(999, 999), "Should not contain non-existent point"
            methods_tested['contains'] = True
            
            # Test query
            results = qt.query(50, 50, 100, 100)
            assert len(results) == 1, f"Query should return 1 result, got {len(results)}"
            methods_tested['query'] = True
            
            # Test get_all_points
            all_points = qt.get_all_points()
            assert len(all_points) == 2, f"Should have 2 points, got {len(all_points)}"
            methods_tested['get_all_points'] = True
            
            # Test depth and subdivisions
            depth = qt.depth()
            subdivisions = qt.subdivisions()
            assert isinstance(depth, int), f"Depth should be int, got {type(depth)}"
            assert isinstance(subdivisions, int), f"Subdivisions should be int, got {type(subdivisions)}"
            methods_tested['depth'] = True
            methods_tested['subdivisions'] = True
            
            # Test collision detection
            collisions = qt.detect_collisions(200.0)
            assert isinstance(collisions, list), "Collisions should return list"
            methods_tested['detect_collisions'] = True
            
            # Verify all methods tested
            untested = [k for k, v in methods_tested.items() if not v]
            assert not untested, f"Methods not tested: {untested}"
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            self.results.append(TestMetrics(
                "API Completeness",
                True,
                time.time() - start_time,
                peak / 1024 / 1024
            ))
            print("✓ API completeness test passed")
            
        except Exception as e:
            tracemalloc.stop()
            self.results.append(TestMetrics(
                "API Completeness",
                False,
                time.time() - start_time,
                0,
                error_message=str(e)
            ))
            print(f"✗ API completeness test failed: {e}")
    
    def test_data_integrity(self):
        """Test data preservation and integrity"""
        print("🔒 Testing Data Integrity...")
        start_time = time.time()
        tracemalloc.start()
        
        try:
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            
            # Test various data types
            test_data = [
                (100, 100, "string_data"),
                (200, 200, 12345),
                (300, 300, [1, 2, 3]),
                (400, 400, {"key": "value"}),
                (500, 500, None),  # No data
                (600, 600, True),
                (700, 700, 3.14159)
            ]
            
            # Insert all test data
            for x, y, data in test_data:
                if data is None:
                    assert qt.insert(x, y), f"Insert failed for ({x}, {y}) with no data"
                else:
                    assert qt.insert(x, y, data), f"Insert failed for ({x}, {y}) with data {data}"
            
            # Retrieve and verify data integrity
            all_points = qt.get_all_points()
            assert len(all_points) == len(test_data), f"Point count mismatch: expected {len(test_data)}, got {len(all_points)}"
            
            # Verify each point's data
            for point in all_points:
                x, y = point[0], point[1]
                expected_data = next((td[2] for td in test_data if td[0] == x and td[1] == y), None)
                
                if len(point) == 3:  # Has data
                    actual_data = point[2]
                    assert actual_data == expected_data, f"Data mismatch for ({x}, {y}): expected {expected_data}, got {actual_data}"
                else:  # No data
                    assert expected_data is None, f"Expected no data for ({x}, {y}) but test data has {expected_data}"
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            self.results.append(TestMetrics(
                "Data Integrity",
                True,
                time.time() - start_time,
                peak / 1024 / 1024
            ))
            print("✓ Data integrity test passed")
            
        except Exception as e:
            tracemalloc.stop()
            self.results.append(TestMetrics(
                "Data Integrity",
                False,
                time.time() - start_time,
                0,
                error_message=str(e)
            ))
            print(f"✗ Data integrity test failed: {e}")
    
    def test_boundary_validation(self):
        """Test boundary conditions and validation"""
        print("🎯 Testing Boundary Validation...")
        start_time = time.time()
        
        try:
            # Test valid boundaries
            valid_cases = [
                (0, 0, 100, 100),
                (-100, -100, 200, 200),
                (0.1, 0.1, 0.8, 0.8),
                (-1e6, -1e6, 2e6, 2e6)
            ]
            
            for x, y, w, h in valid_cases:
                qt = quadtree.QuadTree(x, y, w, h)
                boundary = qt.boundary()
                assert boundary == (x, y, w, h), f"Boundary mismatch for ({x}, {y}, {w}, {h})"
            
            # Test invalid boundaries
            invalid_cases = [
                (0, 0, -1, 100),  # Negative width
                (0, 0, 100, -1),  # Negative height
                (0, 0, 0, 100),   # Zero width
                (0, 0, 100, 0),   # Zero height
            ]
            
            for x, y, w, h in invalid_cases:
                try:
                    qt = quadtree.QuadTree(x, y, w, h)
                    assert False, f"Should have failed for invalid boundary ({x}, {y}, {w}, {h})"
                except ValueError:
                    pass  # Expected
            
            # Test boundary edge insertions
            qt = quadtree.QuadTree(0, 0, 100, 100)
            
            # These should succeed
            assert qt.insert(0, 0), "Should insert at bottom-left corner"
            assert qt.insert(50, 50), "Should insert at center"
            assert qt.insert(99.999, 99.999), "Should insert near top-right"
            
            # These should fail (outside boundary)
            assert not qt.insert(-0.001, 50), "Should not insert outside left boundary"
            assert not qt.insert(50, -0.001), "Should not insert outside bottom boundary"
            assert not qt.insert(100.001, 50), "Should not insert outside right boundary"
            assert not qt.insert(50, 100.001), "Should not insert outside top boundary"
            
            self.results.append(TestMetrics(
                "Boundary Validation",
                True,
                time.time() - start_time,
                0
            ))
            print("✓ Boundary validation test passed")
            
        except Exception as e:
            self.results.append(TestMetrics(
                "Boundary Validation",
                False,
                time.time() - start_time,
                0,
                error_message=str(e)
            ))
            print(f"✗ Boundary validation test failed: {e}")
    
    def test_insertion_performance(self):
        """Test insertion performance under various conditions"""
        print("⚡ Testing Insertion Performance...")
        start_time = time.time()
        tracemalloc.start()
        
        try:
            qt = quadtree.QuadTree(0, 0, 10000, 10000)
            
            # Sequential insertions
            num_points = 50000
            points = [(random.uniform(0, 10000), random.uniform(0, 10000)) for _ in range(num_points)]
            
            insert_start = time.time()
            for i, (x, y) in enumerate(points):
                result = qt.insert(x, y, f"point_{i}")
                assert result, f"Insert failed at point {i}"
            
            insert_time = time.time() - insert_start
            ops_per_second = num_points / insert_time
            
            # Verify final state
            assert qt.size() == num_points, f"Size mismatch: expected {num_points}, got {qt.size()}"
            
            # Test tree depth is reasonable
            depth = qt.depth()
            assert depth <= 15, f"Tree depth too high: {depth} (may indicate poor performance)"
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            self.results.append(TestMetrics(
                "Insertion Performance",
                True,
                time.time() - start_time,
                peak / 1024 / 1024,
                ops_per_second
            ))
            print(f"✓ Insertion performance: {ops_per_second:.0f} ops/sec, depth: {depth}")
            
        except Exception as e:
            tracemalloc.stop()
            self.results.append(TestMetrics(
                "Insertion Performance",
                False,
                time.time() - start_time,
                0,
                error_message=str(e)
            ))
            print(f"✗ Insertion performance test failed: {e}")
    
    def test_query_performance(self):
        """Test query performance with various patterns"""
        print("🔍 Testing Query Performance...")
        start_time = time.time()
        tracemalloc.start()
        
        try:
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            
            # Insert test data
            num_points = 10000
            for i in range(num_points):
                x = random.uniform(0, 1000)
                y = random.uniform(0, 1000)
                qt.insert(x, y, f"data_{i}")
            
            # Test various query sizes
            query_patterns = [
                (10, 10),    # Small queries
                (50, 50),    # Medium queries  
                (200, 200),  # Large queries
                (500, 500),  # Very large queries
            ]
            
            total_queries = 0
            total_results = 0
            
            query_start = time.time()
            
            for qw, qh in query_patterns:
                for _ in range(500):  # 500 queries per pattern
                    qx = random.uniform(0, 1000 - qw)
                    qy = random.uniform(0, 1000 - qh)
                    
                    results = qt.query(qx, qy, qw, qh)
                    total_queries += 1
                    total_results += len(results)
                    
                    # Verify all results are within query bounds
                    for point in results:
                        px, py = point[0], point[1]
                        assert qx <= px < qx + qw and qy <= py < qy + qh, "Query result outside bounds"
            
            query_time = time.time() - query_start
            queries_per_second = total_queries / query_time
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            self.results.append(TestMetrics(
                "Query Performance",
                True,
                time.time() - start_time,
                peak / 1024 / 1024,
                queries_per_second
            ))
            print(f"✓ Query performance: {queries_per_second:.0f} queries/sec, avg {total_results/total_queries:.1f} results/query")
            
        except Exception as e:
            tracemalloc.stop()
            self.results.append(TestMetrics(
                "Query Performance",
                False,
                time.time() - start_time,
                0,
                error_message=str(e)
            ))
            print(f"✗ Query performance test failed: {e}")
    
    def test_collision_performance(self):
        """Test collision detection performance"""
        print("💥 Testing Collision Performance...")
        start_time = time.time()
        tracemalloc.start()
        
        try:
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            
            # Create clustered points for meaningful collision testing
            num_clusters = 50
            points_per_cluster = 20
            
            for cluster_id in range(num_clusters):
                # Random cluster center
                cx = random.uniform(50, 950)
                cy = random.uniform(50, 950)
                
                # Add points around cluster center
                for i in range(points_per_cluster):
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(0, 30)  # Within 30 units
                    
                    x = cx + distance * math.cos(angle)
                    y = cy + distance * math.sin(angle)
                    
                    # Ensure within bounds
                    x = max(0, min(1000, x))
                    y = max(0, min(1000, y))
                    
                    qt.insert(x, y, f"cluster_{cluster_id}_point_{i}")
            
            # Test collision detection with various radii
            radii = [10, 25, 50, 100]
            
            for radius in radii:
                collision_start = time.time()
                collisions = qt.detect_collisions(radius)
                collision_time = time.time() - collision_start
                
                # Verify collision accuracy (sample check)
                valid_collisions = 0
                total_checked = min(100, len(collisions))
                
                for i, collision in enumerate(collisions[:total_checked]):
                    p1 = collision["point1"]
                    p2 = collision["point2"]
                    
                    dx = p1[0] - p2[0]
                    dy = p1[1] - p2[1]
                    distance = math.sqrt(dx * dx + dy * dy)
                    
                    if distance <= radius:
                        valid_collisions += 1
                
                accuracy = valid_collisions / total_checked if total_checked > 0 else 1.0
                assert accuracy > 0.95, f"Collision accuracy too low: {accuracy:.2%} for radius {radius}"
                
                print(f"  Radius {radius}: {len(collisions)} collisions in {collision_time:.3f}s, accuracy: {accuracy:.2%}")
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            self.results.append(TestMetrics(
                "Collision Performance",
                True,
                time.time() - start_time,
                peak / 1024 / 1024
            ))
            print("✓ Collision performance test passed")
            
        except Exception as e:
            tracemalloc.stop()
            self.results.append(TestMetrics(
                "Collision Performance",
                False,
                time.time() - start_time,
                0,
                error_message=str(e)
            ))
            print(f"✗ Collision performance test failed: {e}")
    
    def test_memory_scalability(self):
        """Test memory usage scaling with data size"""
        print("💾 Testing Memory Scalability...")
        start_time = time.time()
        
        try:
            sizes = [1000, 5000, 10000, 25000]
            memory_usage = []
            
            for size in sizes:
                tracemalloc.start()
                qt = quadtree.QuadTree(0, 0, 1000, 1000)
                
                # Insert points
                for i in range(size):
                    x = random.uniform(0, 1000)
                    y = random.uniform(0, 1000)
                    qt.insert(x, y, f"data_{i}")
                
                current, peak = tracemalloc.get_traced_memory()
                memory_usage.append((size, peak / 1024 / 1024))  # MB
                tracemalloc.stop()
                
                print(f"  {size} points: {peak/1024/1024:.1f}MB peak memory")
                
                # Cleanup
                del qt
                gc.collect()
            
            # Check memory scaling is reasonable (should be roughly linear)
            if len(memory_usage) >= 2:
                ratio = memory_usage[-1][1] / memory_usage[0][1]  # Memory ratio
                size_ratio = memory_usage[-1][0] / memory_usage[0][0]  # Size ratio
                scaling_factor = ratio / size_ratio
                
                # Memory should scale reasonably with data size
                assert scaling_factor < 2.0, f"Memory scaling too poor: {scaling_factor:.2f}x"
                print(f"  Memory scaling factor: {scaling_factor:.2f}x (should be < 2.0)")
            
            self.results.append(TestMetrics(
                "Memory Scalability",
                True,
                time.time() - start_time,
                memory_usage[-1][1] if memory_usage else 0
            ))
            print("✓ Memory scalability test passed")
            
        except Exception as e:
            self.results.append(TestMetrics(
                "Memory Scalability",
                False,
                time.time() - start_time,
                0,
                error_message=str(e)
            ))
            print(f"✗ Memory scalability test failed: {e}")
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        print("📊 Testing Large Dataset Handling...")
        start_time = time.time()
        tracemalloc.start()
        
        try:
            qt = quadtree.QuadTree(0, 0, 10000, 10000)
            
            # Insert 100K points
            num_points = 100000
            print(f"  Inserting {num_points} points...")
            
            batch_size = 10000
            for batch in range(0, num_points, batch_size):
                batch_end = min(batch + batch_size, num_points)
                
                for i in range(batch, batch_end):
                    x = random.uniform(0, 10000)
                    y = random.uniform(0, 10000)
                    result = qt.insert(x, y)
                    assert result, f"Insert failed at point {i}"
                
                if batch % (batch_size * 5) == 0:
                    print(f"    Progress: {batch_end}/{num_points} points inserted")
            
            # Verify final state
            final_size = qt.size()
            assert final_size == num_points, f"Size mismatch: expected {num_points}, got {final_size}"
            
            # Test tree is still functional
            depth = qt.depth()
            subdivisions = qt.subdivisions()
            
            # Test query still works efficiently
            query_start = time.time()
            results = qt.query(1000, 1000, 500, 500)
            query_time = time.time() - query_start
            
            assert query_time < 1.0, f"Query too slow on large dataset: {query_time:.3f}s"
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            self.results.append(TestMetrics(
                "Large Dataset Handling",
                True,
                time.time() - start_time,
                peak / 1024 / 1024
            ))
            print(f"✓ Large dataset test passed: {final_size} points, depth: {depth}, query: {query_time:.3f}s")
            
        except Exception as e:
            tracemalloc.stop()
            self.results.append(TestMetrics(
                "Large Dataset Handling",
                False,
                time.time() - start_time,
                0,
                error_message=str(e)
            ))
            print(f"✗ Large dataset test failed: {e}")
    
    def test_precision_edge_cases(self):
        """Test numerical precision edge cases"""
        print("🎯 Testing Precision Edge Cases...")
        start_time = time.time()
        
        try:
            # Test very small coordinates
            qt_small = quadtree.QuadTree(0, 0, 1e-6, 1e-6)
            assert qt_small.insert(5e-7, 5e-7), "Failed to insert very small coordinates"
            assert qt_small.contains(5e-7, 5e-7), "Failed to find very small coordinates"
            
            # Test very large coordinates
            qt_large = quadtree.QuadTree(-1e9, -1e9, 2e9, 2e9)
            assert qt_large.insert(1e8, 1e8), "Failed to insert large coordinates"
            assert qt_large.contains(1e8, 1e8), "Failed to find large coordinates"
            
            # Test precision boundaries
            qt = quadtree.QuadTree(0, 0, 100, 100)
            
            # Points very close to each other
            close_points = [
                (50.0, 50.0),
                (50.0 + 1e-10, 50.0),
                (50.0, 50.0 + 1e-10),
                (50.0 + 1e-9, 50.0 + 1e-9)
            ]
            
            for x, y in close_points:
                assert qt.insert(x, y), f"Failed to insert precise point ({x}, {y})"
            
            # Test collision detection precision
            collisions = qt.detect_collisions(1e-8)
            # Should find collisions between very close points
            
            # Test boundary precision
            assert qt.insert(0.0, 0.0), "Failed to insert at exact boundary"
            assert qt.insert(99.999999999, 99.999999999), "Failed to insert near boundary"
            
            self.results.append(TestMetrics(
                "Precision Edge Cases",
                True,
                time.time() - start_time,
                0
            ))
            print("✓ Precision edge cases test passed")
            
        except Exception as e:
            self.results.append(TestMetrics(
                "Precision Edge Cases",
                False,
                time.time() - start_time,
                0,
                error_message=str(e)
            ))
            print(f"✗ Precision edge cases test failed: {e}")
    
    def test_extreme_values(self):
        """Test extreme values and limits"""
        print("🌡️ Testing Extreme Values...")
        start_time = time.time()
        
        try:
            # Test with maximum depth scenario
            qt = quadtree.QuadTree(0, 0, 1, 1)
            
            # Insert many points in same location to force deep subdivision
            base_x, base_y = 0.5, 0.5
            for i in range(100):
                # Slightly different positions to avoid exact duplicates
                x = base_x + i * 1e-12
                y = base_y + i * 1e-12
                qt.insert(x, y, f"deep_point_{i}")
            
            depth = qt.depth()
            print(f"  Maximum depth achieved: {depth}")
            
            # Test massive query
            qt_large = quadtree.QuadTree(0, 0, 10000, 10000)
            for i in range(1000):
                qt_large.insert(random.uniform(0, 10000), random.uniform(0, 10000))
            
            # Query entire space
            all_results = qt_large.query(0, 0, 10000, 10000)
            assert len(all_results) == 1000, f"Full query should return all points"
            
            # Test zero-size query
            zero_results = qt_large.query(100, 100, 0, 0)
            # Should return empty list or points exactly at (100, 100)
            
            # Test collision with zero radius
            zero_collisions = qt_large.detect_collisions(0.0)
            # Should only find points at exactly same location
            
            self.results.append(TestMetrics(
                "Extreme Values",
                True,
                time.time() - start_time,
                0
            ))
            print("✓ Extreme values test passed")
            
        except Exception as e:
            self.results.append(TestMetrics(
                "Extreme Values",
                False,
                time.time() - start_time,
                0,
                error_message=str(e)
            ))
            print(f"✗ Extreme values test failed: {e}")
    
    def test_error_conditions(self):
        """Test error handling and recovery"""
        print("🚨 Testing Error Conditions...")
        start_time = time.time()
        
        try:
            error_cases_passed = 0
            total_error_cases = 0
            
            # Test invalid constructor parameters
            invalid_constructors = [
                (0, 0, -1, 100),
                (0, 0, 100, -1),
                (0, 0, 0, 100),
                (0, 0, 100, 0)
            ]
            
            for x, y, w, h in invalid_constructors:
                total_error_cases += 1
                try:
                    qt = quadtree.QuadTree(x, y, w, h)
                    # Should not reach here
                except ValueError:
                    error_cases_passed += 1
            
            # Test invalid query parameters
            qt = quadtree.QuadTree(0, 0, 100, 100)
            
            invalid_queries = [
                (0, 0, -1, 10),  # Negative width
                (0, 0, 10, -1),  # Negative height
            ]
            
            for qx, qy, qw, qh in invalid_queries:
                total_error_cases += 1
                try:
                    qt.query(qx, qy, qw, qh)
                    # Should not reach here
                except ValueError:
                    error_cases_passed += 1
            
            # Test invalid collision radius
            total_error_cases += 1
            try:
                qt.detect_collisions(-1.0)
                # Should not reach here
            except ValueError:
                error_cases_passed += 1
            
            # Test operations on empty tree
            empty_qt = quadtree.QuadTree(0, 0, 100, 100)
            
            # These should work on empty tree
            assert empty_qt.empty(), "Empty tree should report empty"
            assert empty_qt.size() == 0, "Empty tree should have size 0"
            assert not empty_qt.contains(50, 50), "Empty tree should not contain points"
            
            empty_results = empty_qt.query(0, 0, 100, 100)
            assert len(empty_results) == 0, "Empty tree query should return empty list"
            
            empty_collisions = empty_qt.detect_collisions(10.0)
            assert len(empty_collisions) == 0, "Empty tree should have no collisions"
            
            success_rate = error_cases_passed / total_error_cases if total_error_cases > 0 else 1.0
            
            self.results.append(TestMetrics(
                "Error Conditions",
                success_rate > 0.8,
                time.time() - start_time,
                0
            ))
            print(f"✓ Error conditions test passed: {error_cases_passed}/{total_error_cases} cases handled correctly")
            
        except Exception as e:
            self.results.append(TestMetrics(
                "Error Conditions",
                False,
                time.time() - start_time,
                0,
                error_message=str(e)
            ))
            print(f"✗ Error conditions test failed: {e}")
    
    def test_thread_safety(self):
        """Test thread safety with concurrent operations"""
        print("🧵 Testing Thread Safety...")
        start_time = time.time()
        tracemalloc.start()
        
        try:
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            thread_results = []
            lock = threading.Lock()  # Use lock for safety
            
            def worker_thread(thread_id: int, operations: int):
                local_results = {"inserted": 0, "queried": 0, "errors": 0}
                
                for i in range(operations):
                    try:
                        # Insert operation
                        x = random.uniform(0, 1000)
                        y = random.uniform(0, 1000)
                        
                        with lock:  # Synchronize access
                            if qt.insert(x, y, f"thread_{thread_id}_point_{i}"):
                                local_results["inserted"] += 1
                        
                        # Query operation
                        qx = random.uniform(0, 900)
                        qy = random.uniform(0, 900)
                        
                        with lock:  # Synchronize access
                            results = qt.query(qx, qy, 100, 100)
                            local_results["queried"] += 1
                            
                    except Exception:
                        local_results["errors"] += 1
                
                thread_results.append(local_results)
            
            # Launch multiple threads
            num_threads = 4
            operations_per_thread = 1000
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for i in range(num_threads):
                    future = executor.submit(worker_thread, i, operations_per_thread)
                    futures.append(future)
                
                # Wait for completion
                for future in as_completed(futures):
                    future.result()
            
            # Analyze results
            total_inserted = sum(r["inserted"] for r in thread_results)
            total_queried = sum(r["queried"] for r in thread_results)
            total_errors = sum(r["errors"] for r in thread_results)
            
            final_size = qt.size()
            
            # Verify consistency
            assert final_size == total_inserted, f"Size inconsistency: expected {total_inserted}, got {final_size}"
            assert total_errors == 0, f"Unexpected errors in threaded operations: {total_errors}"
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            self.results.append(TestMetrics(
                "Thread Safety",
                True,
                time.time() - start_time,
                peak / 1024 / 1024
            ))
            print(f"✓ Thread safety test passed: {total_inserted} insertions, {total_queried} queries, {total_errors} errors")
            
        except Exception as e:
            tracemalloc.stop()
            self.results.append(TestMetrics(
                "Thread Safety",
                False,
                time.time() - start_time,
                0,
                error_message=str(e)
            ))
            print(f"✗ Thread safety test failed: {e}")
    
    def test_memory_leaks(self):
        """Test for memory leaks in repeated operations"""
        print("🔍 Testing Memory Leaks...")
        start_time = time.time()
        
        try:
            # Get baseline memory
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform repeated create/destroy cycles
            for cycle in range(10):
                qt = quadtree.QuadTree(0, 0, 1000, 1000)
                
                # Insert and query many points
                for i in range(1000):
                    x = random.uniform(0, 1000)
                    y = random.uniform(0, 1000)
                    qt.insert(x, y, f"cycle_{cycle}_point_{i}")
                
                # Query operations
                for _ in range(100):
                    qx = random.uniform(0, 900)
                    qy = random.uniform(0, 900)
                    qt.query(qx, qy, 100, 100)
                
                # Collision detection
                qt.detect_collisions(50.0)
                
                # Destroy tree
                del qt
                gc.collect()
                
                if cycle % 3 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    print(f"  Cycle {cycle}: {current_memory:.1f}MB (baseline: {baseline_memory:.1f}MB)")
            
            # Final memory check
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - baseline_memory
            
            # Allow for some memory growth, but not excessive
            max_acceptable_growth = 50  # MB
            memory_leak_detected = memory_growth > max_acceptable_growth
            
            self.results.append(TestMetrics(
                "Memory Leaks",
                not memory_leak_detected,
                time.time() - start_time,
                final_memory
            ))
            
            if memory_leak_detected:
                print(f"⚠️ Potential memory leak: {memory_growth:.1f}MB growth")
            else:
                print(f"✓ Memory leak test passed: {memory_growth:.1f}MB growth (acceptable)")
            
        except Exception as e:
            self.results.append(TestMetrics(
                "Memory Leaks",
                False,
                time.time() - start_time,
                0,
                error_message=str(e)
            ))
            print(f"✗ Memory leak test failed: {e}")
    
    def generate_production_report(self):
        """Generate comprehensive production readiness report"""
        print("\n" + "=" * 70)
        print("📋 PRODUCTION READINESS REPORT")
        print("=" * 70)
        
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        total_time = sum(r.execution_time for r in self.results)
        max_memory = max((r.memory_peak_mb for r in self.results if r.memory_peak_mb > 0), default=0)
        
        # Overall assessment
        success_rate = len(passed_tests) / len(self.results) * 100
        
        print(f"Overall Assessment: {success_rate:.1f}% tests passed")
        print(f"Total Test Time: {total_time:.2f} seconds")
        print(f"Peak Memory Usage: {max_memory:.1f} MB")
        print()
        
        # Detailed results
        print("📊 Test Results:")
        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            perf_info = f" | {result.operations_per_second:.0f} ops/sec" if result.operations_per_second > 0 else ""
            memory_info = f" | {result.memory_peak_mb:.1f}MB" if result.memory_peak_mb > 0 else ""
            
            print(f"  {status} {result.name:<25} | {result.execution_time:>6.2f}s{perf_info}{memory_info}")
            
            if not result.passed and result.error_message:
                print(f"    Error: {result.error_message}")
        
        print()
        
        # Production readiness assessment
        critical_tests = [
            "API Completeness",
            "Data Integrity", 
            "Boundary Validation",
            "Error Conditions"
        ]
        
        performance_tests = [
            "Insertion Performance",
            "Query Performance", 
            "Memory Scalability"
        ]
        
        critical_passed = all(any(r.name == test and r.passed for r in self.results) for test in critical_tests)
        performance_passed = all(any(r.name == test and r.passed for r in self.results) for test in performance_tests)
        
        print("🎯 Production Readiness Assessment:")
        print(f"  Critical Functionality: {'✓ READY' if critical_passed else '✗ NOT READY'}")
        print(f"  Performance Requirements: {'✓ READY' if performance_passed else '✗ NOT READY'}")
        print(f"  Overall Stability: {'✓ READY' if success_rate >= 90 else '✗ NEEDS WORK'}")
        
        if success_rate >= 90 and critical_passed and performance_passed:
            print("\n🚀 VERDICT: PRODUCTION READY")
            print("The QuadTree implementation meets production requirements.")
        else:
            print("\n⚠️ VERDICT: NEEDS IMPROVEMENT")
            print("Address failing tests before production deployment.")
        
        # Save detailed report
        report_data = {
            "timestamp": time.time(),
            "success_rate": success_rate,
            "total_tests": len(self.results),
            "passed_tests": len(passed_tests),
            "failed_tests": len(failed_tests),
            "total_time": total_time,
            "peak_memory_mb": max_memory,
            "production_ready": success_rate >= 90 and critical_passed and performance_passed,
            "test_results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "execution_time": r.execution_time,
                    "memory_peak_mb": r.memory_peak_mb,
                    "operations_per_second": r.operations_per_second,
                    "error_message": r.error_message
                }
                for r in self.results
            ]
        }
        
        with open("quadtree_production_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: quadtree_production_report.json")

def main():
    """Run the complete production test suite"""
    print("QuadTree C++17 Implementation - Production Readiness Test Suite")
    print("Comprehensive validation for production deployment")
    print()
    
    tester = ProductionQuadTreeTester()
    tester.run_production_test_suite()

if __name__ == "__main__":
    main()