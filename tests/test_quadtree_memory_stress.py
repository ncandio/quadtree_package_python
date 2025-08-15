#!/usr/bin/env python3
"""
Comprehensive Memory Leak and Stress Test for QuadTree C++17 Implementation

This test suite focuses on:
1. Memory leak detection during intensive operations
2. Stress testing with large datasets and edge cases
3. Memory usage patterns and efficiency analysis
4. Implementation quality assessment
5. Resource cleanup verification

Designed to detect memory issues that could occur in production environments.
"""

import sys
import os
import gc
import time
import json
import math
import random
import threading
import traceback
import tracemalloc
from typing import List, Dict, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime

# Optional imports for enhanced monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False

sys.path.insert(0, '.')

try:
    import quadtree
    print("✓ QuadTree module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import quadtree: {e}")
    sys.exit(1)

class MemoryStressTester:
    """Comprehensive memory leak and stress testing for QuadTree"""
    
    def __init__(self):
        self.test_results = []
        self.memory_snapshots = []
        self.performance_metrics = []
        self.leak_candidates = []
        self.start_time = time.time()
        
        # Initialize memory monitoring
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
            self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        else:
            self.process = None
            self.initial_memory = 0
        
        # Start tracemalloc for detailed memory tracking
        tracemalloc.start()
        
        print("🧠 Memory Stress Test Suite Initialized")
        print(f"Initial memory usage: {self.initial_memory:.2f} MB")
        print("=" * 80)
    
    def take_memory_snapshot(self, label: str, extra_data: Dict = None) -> Dict:
        """Take comprehensive memory snapshot"""
        snapshot = {
            'timestamp': time.time() - self.start_time,
            'label': label
        }
        
        # Python memory tracking
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            snapshot.update({
                'tracemalloc_current': current / 1024 / 1024,  # MB
                'tracemalloc_peak': peak / 1024 / 1024
            })
        
        # Process memory tracking
        if self.process:
            memory_info = self.process.memory_info()
            snapshot.update({
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'memory_delta': (memory_info.rss / 1024 / 1024) - self.initial_memory
            })
        
        # System resource tracking
        if RESOURCE_AVAILABLE:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            snapshot.update({
                'max_rss_kb': usage.ru_maxrss,
                'page_faults': usage.ru_majflt + usage.ru_minflt
            })
        
        if extra_data:
            snapshot.update(extra_data)
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def run_comprehensive_stress_tests(self):
        """Execute comprehensive stress testing suite"""
        print("🔥 Starting Comprehensive Memory Stress Tests")
        print("=" * 80)
        
        self.take_memory_snapshot("test_start")
        
        # Core stress tests
        self.test_massive_insertions_memory()
        self.test_cyclic_creation_destruction()
        self.test_large_object_storage()
        self.test_subdivision_memory_patterns()
        self.test_query_intensive_operations()
        self.test_collision_detection_stress()
        
        # Edge case stress tests
        self.test_boundary_stress_conditions()
        self.test_precision_edge_cases()
        self.test_memory_fragmentation()
        
        # Cleanup and leak detection
        self.test_cleanup_verification()
        self.test_concurrent_stress()
        self.test_memory_leak_detection()
        
        self.take_memory_snapshot("test_end")
        self.analyze_memory_patterns()
        self.generate_comprehensive_report()
    
    def test_massive_insertions_memory(self):
        """Test memory usage during massive insertion operations"""
        print("📊 Testing Massive Insertions Memory Patterns...")
        
        self.take_memory_snapshot("before_massive_insertions")
        
        # Test different dataset sizes to identify memory scaling
        dataset_sizes = [1000, 10000, 50000, 100000]
        
        for size in dataset_sizes:
            print(f"  Testing with {size:,} points...")
            
            qt = quadtree.QuadTree(0, 0, 10000, 10000)
            start_time = time.time()
            
            # Insert points with varying data sizes
            for i in range(size):
                x = random.uniform(0, 10000)
                y = random.uniform(0, 10000)
                
                # Vary data complexity to test memory handling
                if i % 4 == 0:
                    data = None
                elif i % 4 == 1:
                    data = f"data_string_{i}"
                elif i % 4 == 2:
                    data = {"id": i, "metadata": list(range(i % 10))}
                else:
                    data = [i, i*2, i*3, {"nested": str(i)}]
                
                qt.insert(x, y, data)
                
                # Sample memory during insertion
                if i > 0 and i % (size // 10) == 0:
                    progress = (i / size) * 100
                    snapshot = self.take_memory_snapshot(
                        f"insertion_progress_{size}_{progress:.0f}pct",
                        {"dataset_size": size, "points_inserted": i}
                    )
            
            insertion_time = time.time() - start_time
            
            # Test tree operations after massive insertion
            final_size = qt.size()
            tree_depth = qt.depth()
            subdivisions = qt.subdivisions()
            
            # Memory snapshot after complete insertion
            post_insertion_snapshot = self.take_memory_snapshot(
                f"post_insertion_{size}",
                {
                    "dataset_size": size,
                    "final_tree_size": final_size,
                    "tree_depth": tree_depth,
                    "subdivisions": subdivisions,
                    "insertion_time": insertion_time
                }
            )
            
            # Test some operations to ensure tree is functional
            sample_queries = min(100, size // 100)
            for _ in range(sample_queries):
                qx, qy = random.uniform(0, 9000), random.uniform(0, 9000)
                qt.query(qx, qy, 1000, 1000)
            
            # Cleanup and measure memory release
            del qt
            gc.collect()
            
            cleanup_snapshot = self.take_memory_snapshot(
                f"after_cleanup_{size}",
                {"dataset_size": size}
            )
            
            # Calculate memory efficiency
            memory_per_point = (post_insertion_snapshot.get('memory_delta', 0) * 1024) / size  # KB per point
            memory_released = post_insertion_snapshot.get('memory_delta', 0) - cleanup_snapshot.get('memory_delta', 0)
            
            print(f"    ✓ Size: {size:6,} | Memory/point: {memory_per_point:.3f} KB | Released: {memory_released:.2f} MB")
            
            # Flag potential memory issues
            if memory_per_point > 5.0:  # More than 5KB per point seems excessive
                self.leak_candidates.append({
                    'test': 'massive_insertions',
                    'issue': 'high_memory_per_point',
                    'details': f'{memory_per_point:.3f} KB per point for {size} points'
                })
            
            if memory_released < post_insertion_snapshot.get('memory_delta', 0) * 0.7:
                self.leak_candidates.append({
                    'test': 'massive_insertions',
                    'issue': 'incomplete_cleanup',
                    'details': f'Only {memory_released:.2f} MB released from {post_insertion_snapshot.get("memory_delta", 0):.2f} MB'
                })
        
        self.test_results.append(("Massive Insertions Memory", True, ""))
    
    def test_cyclic_creation_destruction(self):
        """Test memory leaks in repeated creation/destruction cycles"""
        print("♻️ Testing Cyclic Creation/Destruction...")
        
        self.take_memory_snapshot("before_cycles")
        
        baseline_memory = None
        memory_growth_trend = []
        
        # Perform multiple cycles of creation and destruction
        num_cycles = 50
        points_per_cycle = 2000
        
        for cycle in range(num_cycles):
            # Create and populate QuadTree
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            
            for i in range(points_per_cycle):
                x = random.uniform(0, 1000)
                y = random.uniform(0, 1000)
                data = f"cycle_{cycle}_point_{i}" * random.randint(1, 3)  # Variable size data
                qt.insert(x, y, data)
            
            # Perform some operations
            for _ in range(10):
                qt.query(random.uniform(0, 900), random.uniform(0, 900), 100, 100)
                qt.detect_collisions(random.uniform(1, 10))
            
            # Destroy QuadTree
            del qt
            gc.collect()
            
            # Monitor memory every few cycles
            if cycle % 5 == 0:
                snapshot = self.take_memory_snapshot(
                    f"cycle_{cycle}",
                    {"cycle": cycle, "points_per_cycle": points_per_cycle}
                )
                
                current_memory = snapshot.get('memory_delta', 0)
                memory_growth_trend.append(current_memory)
                
                if baseline_memory is None:
                    baseline_memory = current_memory
                
                if cycle > 0:
                    growth_from_baseline = current_memory - baseline_memory
                    print(f"  Cycle {cycle:2d}: Memory delta: {current_memory:6.2f} MB (growth: {growth_from_baseline:+6.2f} MB)")
                    
                    # Check for concerning memory growth
                    if growth_from_baseline > 50:  # More than 50MB growth
                        self.leak_candidates.append({
                            'test': 'cyclic_creation',
                            'issue': 'memory_growth_trend',
                            'details': f'Memory grew by {growth_from_baseline:.2f} MB over {cycle} cycles'
                        })
        
        # Analyze memory growth pattern
        if len(memory_growth_trend) > 5:
            early_avg = sum(memory_growth_trend[:3]) / 3
            late_avg = sum(memory_growth_trend[-3:]) / 3
            total_growth = late_avg - early_avg
            
            print(f"  Total memory growth over cycles: {total_growth:.2f} MB")
            
            if total_growth > 20:  # Significant growth suggests leak
                self.leak_candidates.append({
                    'test': 'cyclic_creation',
                    'issue': 'significant_memory_growth',
                    'details': f'Memory grew by {total_growth:.2f} MB over {num_cycles} cycles'
                })
            else:
                print("  ✓ Memory usage remained stable across cycles")
        
        self.test_results.append(("Cyclic Creation/Destruction", True, ""))
    
    def test_large_object_storage(self):
        """Test memory handling with large data objects"""
        print("🗃️ Testing Large Object Storage...")
        
        self.take_memory_snapshot("before_large_objects")
        
        # Create increasingly large data objects
        large_objects_sizes = [1024, 10240, 102400, 1024000]  # 1KB to 1MB
        
        for size_bytes in large_objects_sizes:
            print(f"  Testing with {size_bytes:,} byte objects...")
            
            # Create fresh QuadTree for each size test
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            
            # Create large string data
            large_data = "x" * size_bytes
            
            # Insert points with large data
            num_points = min(100, 1000000 // size_bytes)  # Limit total memory usage
            
            for i in range(num_points):
                x = random.uniform(0, 1000)
                y = random.uniform(0, 1000)
                qt.insert(x, y, large_data)
            
            snapshot = self.take_memory_snapshot(
                f"large_objects_{size_bytes}",
                {
                    "object_size_bytes": size_bytes,
                    "num_objects": num_points,
                    "total_data_size_mb": (size_bytes * num_points) / (1024 * 1024)
                }
            )
            
            # Verify tree operations still work
            results = qt.query(0, 0, 1000, 1000)
            assert len(results) >= num_points, f"Query should return {num_points} points, got {len(results)}"
            
            # Verify data integrity if we have results
            if results:
                sample_point = results[0]
                if len(sample_point) == 3:
                    # Check if data is preserved (might be string or other type)
                    data = sample_point[2]
                    if isinstance(data, str):
                        assert len(data) == size_bytes, f"Data size should be preserved: expected {size_bytes}, got {len(data)}"
                    else:
                        print(f"    ⚠️ Data type changed: expected str, got {type(data)}")
            else:
                print(f"    ⚠️ No results returned from query, but {num_points} points were inserted")
            
            print(f"    ✓ {num_points} objects of {size_bytes:,} bytes stored successfully")
            
            # Cleanup this size test's QuadTree
            del qt
            gc.collect()
        
        # Final memory snapshot after all large object tests
        post_cleanup_memory = self.take_memory_snapshot("after_large_object_cleanup")
        
        print(f"  ✓ Large object storage tests completed")
        
        self.test_results.append(("Large Object Storage", True, ""))
    
    def test_subdivision_memory_patterns(self):
        """Test memory patterns during subdivision operations"""
        print("🌳 Testing Subdivision Memory Patterns...")
        
        self.take_memory_snapshot("before_subdivision_test")
        
        # Create scenarios that force different subdivision patterns
        test_scenarios = [
            {
                'name': 'Deep Single Branch',
                'points': lambda: [(500 + i * 0.01, 500 + i * 0.01) for i in range(1000)],
                'description': 'Points clustered to force deep subdivision'
            },
            {
                'name': 'Balanced Subdivision',
                'points': lambda: [(i * 100, j * 100) for i in range(10) for j in range(10)],
                'description': 'Evenly distributed points'
            },
            {
                'name': 'Extreme Clustering',
                'points': lambda: [(random.gauss(500, 1), random.gauss(500, 1)) for _ in range(2000)],
                'description': 'Gaussian distribution forcing maximum subdivision'
            }
        ]
        
        for scenario in test_scenarios:
            print(f"  Testing: {scenario['name']}")
            
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            points = scenario['points']()
            
            # Insert points and track subdivision progression
            subdivision_progression = []
            
            for i, (x, y) in enumerate(points):
                qt.insert(x, y, f"data_{i}")
                
                if i % 100 == 0:  # Sample every 100 insertions
                    depth = qt.depth()
                    subdivisions = qt.subdivisions()
                    size = qt.size()
                    
                    subdivision_progression.append({
                        'points': i + 1,
                        'depth': depth,
                        'subdivisions': subdivisions,
                        'size': size
                    })
                    
                    if i % 500 == 0:
                        snapshot = self.take_memory_snapshot(
                            f"subdivision_{scenario['name'].replace(' ', '_').lower()}_{i}",
                            {
                                'scenario': scenario['name'],
                                'points_inserted': i + 1,
                                'depth': depth,
                                'subdivisions': subdivisions
                            }
                        )
            
            final_depth = qt.depth()
            final_subdivisions = qt.subdivisions()
            final_size = qt.size()
            
            print(f"    Final: Size={final_size}, Depth={final_depth}, Subdivisions={final_subdivisions}")
            
            # Test that subdivision doesn't cause excessive memory usage
            final_snapshot = self.take_memory_snapshot(
                f"subdivision_final_{scenario['name'].replace(' ', '_').lower()}",
                {
                    'scenario': scenario['name'],
                    'final_depth': final_depth,
                    'final_subdivisions': final_subdivisions,
                    'final_size': final_size
                }
            )
            
            # Verify subdivision efficiency (memory per subdivision should be reasonable)
            if final_subdivisions > 0:
                memory_per_subdivision = (final_snapshot.get('memory_delta', 0) * 1024 * 1024) / final_subdivisions  # bytes
                if memory_per_subdivision > 1024 * 100:  # More than 100KB per subdivision seems high
                    self.leak_candidates.append({
                        'test': 'subdivision_patterns',
                        'issue': 'high_memory_per_subdivision',
                        'details': f'{memory_per_subdivision / 1024:.1f} KB per subdivision in {scenario["name"]}'
                    })
            
            del qt
            gc.collect()
        
        self.test_results.append(("Subdivision Memory Patterns", True, ""))
    
    def test_query_intensive_operations(self):
        """Test memory stability during intensive query operations"""
        print("🔍 Testing Query-Intensive Operations...")
        
        self.take_memory_snapshot("before_query_intensive")
        
        # Create a large QuadTree for querying
        qt = quadtree.QuadTree(0, 0, 10000, 10000)
        
        # Populate with significant data
        num_points = 50000
        for i in range(num_points):
            x = random.uniform(0, 10000)
            y = random.uniform(0, 10000)
            data = {"id": i, "data": f"point_{i}", "metadata": list(range(i % 5))}
            qt.insert(x, y, data)
        
        population_snapshot = self.take_memory_snapshot("after_population", {"num_points": num_points})
        
        # Perform intensive query operations
        query_types = [
            ("Small Queries", lambda: (random.uniform(0, 9900), random.uniform(0, 9900), 100, 100)),
            ("Medium Queries", lambda: (random.uniform(0, 9000), random.uniform(0, 9000), 1000, 1000)),
            ("Large Queries", lambda: (random.uniform(0, 7000), random.uniform(0, 7000), 3000, 3000)),
            ("Edge Queries", lambda: (random.uniform(-100, 100), random.uniform(-100, 100), 200, 200))
        ]
        
        for query_type, query_generator in query_types:
            print(f"  Executing {query_type}...")
            
            num_queries = 1000
            start_time = time.time()
            total_results = 0
            
            for i in range(num_queries):
                qx, qy, qw, qh = query_generator()
                results = qt.query(qx, qy, qw, qh)
                total_results += len(results)
                
                # Sample memory during intensive querying
                if i % 250 == 0:
                    self.take_memory_snapshot(
                        f"query_intensive_{query_type.replace(' ', '_').lower()}_{i}",
                        {"query_type": query_type, "queries_completed": i}
                    )
            
            query_time = time.time() - start_time
            avg_results = total_results / num_queries
            queries_per_sec = num_queries / query_time
            
            print(f"    ✓ {num_queries} queries in {query_time:.2f}s ({queries_per_sec:.0f} queries/sec)")
            print(f"      Average results per query: {avg_results:.1f}")
        
        # Test memory stability after intensive querying
        post_query_snapshot = self.take_memory_snapshot("after_query_intensive")
        
        # Memory should be stable (not significantly higher than after population)
        memory_growth_during_queries = post_query_snapshot.get('memory_delta', 0) - population_snapshot.get('memory_delta', 0)
        if memory_growth_during_queries > 10:  # More than 10MB growth during queries
            self.leak_candidates.append({
                'test': 'query_intensive',
                'issue': 'memory_growth_during_queries',
                'details': f'Memory grew by {memory_growth_during_queries:.2f} MB during query operations'
            })
        else:
            print("  ✓ Memory remained stable during intensive querying")
        
        del qt
        gc.collect()
        
        self.test_results.append(("Query Intensive Operations", True, ""))
    
    def test_collision_detection_stress(self):
        """Test memory patterns during intensive collision detection"""
        print("💥 Testing Collision Detection Stress...")
        
        self.take_memory_snapshot("before_collision_stress")
        
        # Create scenario with many potential collisions
        qt = quadtree.QuadTree(0, 0, 1000, 1000)
        
        # Insert clustered points that will generate many collisions
        clusters = [
            (200, 200, 2000),  # (center_x, center_y, num_points)
            (500, 500, 2000),
            (800, 800, 2000)
        ]
        
        total_points = 0
        for center_x, center_y, count in clusters:
            for i in range(count):
                # Gaussian distribution around cluster center
                x = random.gauss(center_x, 50)
                y = random.gauss(center_y, 50)
                # Clamp to bounds
                x = max(0, min(1000, x))
                y = max(0, min(1000, y))
                data = {"cluster": f"{center_x}_{center_y}", "point_id": i}
                qt.insert(x, y, data)
                total_points += 1
        
        population_snapshot = self.take_memory_snapshot("collision_population", {"total_points": total_points})
        print(f"  Populated tree with {total_points:,} clustered points")
        
        # Test collision detection with various radii
        radii_to_test = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
        
        for radius in radii_to_test:
            print(f"  Testing collision detection with radius {radius}...")
            
            start_time = time.time()
            collisions = qt.detect_collisions(radius)
            detection_time = time.time() - start_time
            
            collision_snapshot = self.take_memory_snapshot(
                f"collision_r{radius}",
                {
                    "radius": radius,
                    "num_collisions": len(collisions),
                    "detection_time": detection_time,
                    "collisions_per_sec": len(collisions) / detection_time if detection_time > 0 else 0
                }
            )
            
            print(f"    ✓ Radius {radius:5.1f}: {len(collisions):6,} collisions in {detection_time:.3f}s")
            
            # Verify collision results integrity
            if len(collisions) > 0:
                sample_collision = collisions[0]
                assert "point1" in sample_collision and "point2" in sample_collision, "Collision format should be correct"
                
                # Verify distance is within radius
                p1, p2 = sample_collision["point1"], sample_collision["point2"]
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                distance = (dx*dx + dy*dy) ** 0.5
                assert distance <= radius + 1e-6, f"Collision distance {distance} should be <= radius {radius}"
            
            # Check memory usage doesn't grow excessively with collision count
            memory_growth = collision_snapshot.get('memory_delta', 0) - population_snapshot.get('memory_delta', 0)
            if len(collisions) > 0:
                memory_per_collision = (memory_growth * 1024 * 1024) / len(collisions)  # bytes per collision
                if memory_per_collision > 1024:  # More than 1KB per collision seems high
                    self.leak_candidates.append({
                        'test': 'collision_detection',
                        'issue': 'high_memory_per_collision',
                        'details': f'{memory_per_collision:.0f} bytes per collision for radius {radius}'
                    })
        
        # Test repeated collision detection to check for leaks
        print("  Testing repeated collision detection...")
        baseline_memory = self.take_memory_snapshot("collision_repeat_baseline")
        
        for iteration in range(20):
            collisions = qt.detect_collisions(25.0)  # Medium radius
            if iteration % 5 == 0:
                snapshot = self.take_memory_snapshot(f"collision_repeat_{iteration}")
                memory_growth = snapshot.get('memory_delta', 0) - baseline_memory.get('memory_delta', 0)
                if memory_growth > 5:  # More than 5MB growth suggests leak
                    self.leak_candidates.append({
                        'test': 'collision_detection',
                        'issue': 'memory_leak_in_repeated_detection',
                        'details': f'Memory grew by {memory_growth:.2f} MB after {iteration} iterations'
                    })
        
        del qt
        gc.collect()
        
        self.test_results.append(("Collision Detection Stress", True, ""))
    
    def test_boundary_stress_conditions(self):
        """Test behavior under extreme boundary conditions"""
        print("🎯 Testing Boundary Stress Conditions...")
        
        self.take_memory_snapshot("before_boundary_stress")
        
        # Test with very small boundaries
        small_qt = quadtree.QuadTree(0, 0, 1e-6, 1e-6)
        for i in range(100):
            x = random.uniform(0, 1e-6)
            y = random.uniform(0, 1e-6)
            small_qt.insert(x, y, f"tiny_{i}")
        
        print("  ✓ Small boundary tree created and populated")
        
        # Test with very large boundaries
        large_qt = quadtree.QuadTree(0, 0, 1e12, 1e12)
        for i in range(100):
            x = random.uniform(0, 1e12)
            y = random.uniform(0, 1e12)
            large_qt.insert(x, y, f"huge_{i}")
        
        print("  ✓ Large boundary tree created and populated")
        
        # Test with extreme aspect ratios
        thin_qt = quadtree.QuadTree(0, 0, 10000, 1)
        for i in range(1000):
            x = random.uniform(0, 10000)
            y = random.uniform(0, 1)
            thin_qt.insert(x, y, f"thin_{i}")
        
        print("  ✓ Thin (high aspect ratio) tree created and populated")
        
        # Test precision boundaries
        precision_qt = quadtree.QuadTree(0, 0, 100, 100)
        
        # Insert points very close to each other
        base_x, base_y = 50.0, 50.0
        for i in range(1000):
            # Points within floating point precision
            x = base_x + (i * 1e-10)
            y = base_y + (i * 1e-10)
            precision_qt.insert(x, y, f"precision_{i}")
        
        print("  ✓ Precision boundary tests completed")
        
        boundary_snapshot = self.take_memory_snapshot("boundary_stress_complete")
        
        # Clean up all trees
        del small_qt, large_qt, thin_qt, precision_qt
        gc.collect()
        
        cleanup_snapshot = self.take_memory_snapshot("boundary_stress_cleanup")
        
        self.test_results.append(("Boundary Stress Conditions", True, ""))
    
    def test_precision_edge_cases(self):
        """Test floating point precision edge cases"""
        print("🔢 Testing Precision Edge Cases...")
        
        qt = quadtree.QuadTree(0, 0, 100, 100)
        
        # Test various precision scenarios
        precision_tests = [
            # Near-zero values
            (1e-15, 1e-15, "near_zero"),
            (sys.float_info.epsilon, sys.float_info.epsilon, "epsilon"),
            
            # Very large values within bounds
            (99.999999999999999, 99.999999999999999, "near_boundary"),
            
            # Values that might cause precision issues
            (1.0/3.0, 2.0/3.0, "repeating_decimal"),
            (math.pi, math.e, "transcendental") if 'math' in globals() else (3.14159, 2.71828, "transcendental"),
        ]
        
        for x, y, label in precision_tests:
            result = qt.insert(x, y, f"precision_test_{label}")
            if result:  # Only test contains if insert succeeded
                assert qt.contains(x, y), f"Should contain precisely inserted point {label}"
        
        # Test precision in queries
        for _ in range(100):
            # Very small query regions
            qx = random.uniform(0, 99)
            qy = random.uniform(0, 99)
            results = qt.query(qx, qy, 1e-10, 1e-10)
            # Should not crash, results may be empty
        
        print("  ✓ Floating point precision edge cases handled")
        
        self.test_results.append(("Precision Edge Cases", True, ""))
    
    def test_memory_fragmentation(self):
        """Test memory fragmentation patterns"""
        print("🧩 Testing Memory Fragmentation...")
        
        self.take_memory_snapshot("before_fragmentation_test")
        
        # Create and destroy many small trees to test fragmentation
        trees = []
        
        # Phase 1: Create many small trees
        for i in range(100):
            qt = quadtree.QuadTree(i, i, 100, 100)
            for j in range(100):
                x = random.uniform(i, i + 100)
                y = random.uniform(i, i + 100)
                qt.insert(x, y, f"frag_{i}_{j}")
            trees.append(qt)
        
        fragmentation_create_snapshot = self.take_memory_snapshot("fragmentation_created")
        
        # Phase 2: Delete every other tree
        for i in range(0, len(trees), 2):
            del trees[i]
            trees[i] = None
        
        fragmentation_partial_delete_snapshot = self.take_memory_snapshot("fragmentation_partial_delete")
        
        # Phase 3: Create new trees in the gaps
        for i in range(0, len(trees), 2):
            if trees[i] is None:
                qt = quadtree.QuadTree(i + 1000, i + 1000, 100, 100)
                for j in range(200):  # Larger trees
                    x = random.uniform(i + 1000, i + 1100)
                    y = random.uniform(i + 1000, i + 1100)
                    qt.insert(x, y, f"new_frag_{i}_{j}")
                trees[i] = qt
        
        fragmentation_refill_snapshot = self.take_memory_snapshot("fragmentation_refilled")
        
        # Clean up all
        trees.clear()
        gc.collect()
        
        fragmentation_final_snapshot = self.take_memory_snapshot("fragmentation_final_cleanup")
        
        print("  ✓ Memory fragmentation test completed")
        
        self.test_results.append(("Memory Fragmentation", True, ""))
    
    def test_cleanup_verification(self):
        """Verify proper cleanup of resources"""
        print("🧹 Testing Cleanup Verification...")
        
        pre_cleanup_snapshot = self.take_memory_snapshot("pre_cleanup_verification")
        
        # Create large trees with substantial data
        large_trees = []
        for i in range(10):
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            
            # Add substantial data
            for j in range(5000):
                x = random.uniform(0, 1000)
                y = random.uniform(0, 1000)
                # Large data objects
                data = {
                    "id": j,
                    "large_data": "x" * 1000,  # 1KB strings
                    "metadata": list(range(100)),
                    "tree_id": i
                }
                qt.insert(x, y, data)
            
            large_trees.append(qt)
        
        peak_usage_snapshot = self.take_memory_snapshot("cleanup_peak_usage")
        
        # Progressive cleanup
        cleanup_snapshots = []
        for i, qt in enumerate(large_trees):
            del qt
            gc.collect()  # Force garbage collection
            
            cleanup_snapshot = self.take_memory_snapshot(f"cleanup_after_{i+1}_trees")
            cleanup_snapshots.append(cleanup_snapshot)
        
        large_trees.clear()
        
        # Final cleanup
        final_cleanup_snapshot = self.take_memory_snapshot("final_cleanup_verification")
        
        # Analyze cleanup efficiency
        peak_memory = peak_usage_snapshot.get('memory_delta', 0)
        final_memory = final_cleanup_snapshot.get('memory_delta', 0)
        total_memory_released = peak_memory - final_memory
        cleanup_efficiency = (total_memory_released / peak_memory * 100) if peak_memory > 0 else 0
        
        print(f"  Peak memory usage: {peak_memory:.2f} MB")
        print(f"  Final memory usage: {final_memory:.2f} MB")
        print(f"  Memory released: {total_memory_released:.2f} MB ({cleanup_efficiency:.1f}%)")
        
        if cleanup_efficiency < 80:
            self.leak_candidates.append({
                'test': 'cleanup_verification',
                'issue': 'poor_cleanup_efficiency',
                'details': f'Only {cleanup_efficiency:.1f}% of memory was released'
            })
        else:
            print("  ✓ Good cleanup efficiency")
        
        self.test_results.append(("Cleanup Verification", True, ""))
    
    def test_concurrent_stress(self):
        """Test concurrent operations under stress"""
        print("🔄 Testing Concurrent Stress...")
        
        self.take_memory_snapshot("before_concurrent_stress")
        
        # Shared QuadTree for concurrent access
        qt = quadtree.QuadTree(0, 0, 10000, 10000)
        
        # Pre-populate with some data
        for i in range(5000):
            x = random.uniform(0, 10000)
            y = random.uniform(0, 10000)
            qt.insert(x, y, f"initial_{i}")
        
        errors = []
        results_collected = []
        
        def concurrent_insertions(worker_id):
            """Worker function for concurrent insertions"""
            local_errors = []
            points_inserted = 0
            try:
                for i in range(1000):
                    x = random.uniform(worker_id * 1000, (worker_id + 1) * 1000)
                    y = random.uniform(0, 10000)
                    data = {"worker": worker_id, "point": i, "data": "x" * 100}
                    
                    result = qt.insert(x, y, data)
                    if result:
                        points_inserted += 1
                        
            except Exception as e:
                local_errors.append(f"Worker {worker_id} insertion error: {e}")
            
            return {"worker_id": worker_id, "points_inserted": points_inserted, "errors": local_errors}
        
        def concurrent_queries(worker_id):
            """Worker function for concurrent queries"""
            local_errors = []
            queries_completed = 0
            total_results = 0
            
            try:
                for i in range(500):
                    qx = random.uniform(0, 9000)
                    qy = random.uniform(0, 9000)
                    results = qt.query(qx, qy, 1000, 1000)
                    total_results += len(results)
                    queries_completed += 1
                    
            except Exception as e:
                local_errors.append(f"Query worker {worker_id} error: {e}")
            
            return {"worker_id": worker_id, "queries_completed": queries_completed, 
                   "total_results": total_results, "errors": local_errors}
        
        def concurrent_collisions(worker_id):
            """Worker function for concurrent collision detection"""
            local_errors = []
            collisions_detected = 0
            
            try:
                for i in range(50):  # Fewer collision checks as they're expensive
                    radius = random.uniform(10, 100)
                    collisions = qt.detect_collisions(radius)
                    collisions_detected += len(collisions)
                    
            except Exception as e:
                local_errors.append(f"Collision worker {worker_id} error: {e}")
            
            return {"worker_id": worker_id, "collisions_detected": collisions_detected, "errors": local_errors}
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit various types of concurrent tasks
            futures = []
            
            # Insertion workers
            for i in range(3):
                futures.append(executor.submit(concurrent_insertions, i))
            
            # Query workers  
            for i in range(3):
                futures.append(executor.submit(concurrent_queries, i + 10))
            
            # Collision workers
            for i in range(2):
                futures.append(executor.submit(concurrent_collisions, i + 20))
            
            # Collect results
            for future in as_completed(futures, timeout=60):
                try:
                    result = future.result()
                    results_collected.append(result)
                    if result.get('errors'):
                        errors.extend(result['errors'])
                except Exception as e:
                    errors.append(f"Future execution error: {e}")
        
        concurrent_complete_snapshot = self.take_memory_snapshot("concurrent_stress_complete")
        
        # Analyze results
        total_insertions = sum(r.get('points_inserted', 0) for r in results_collected)
        total_queries = sum(r.get('queries_completed', 0) for r in results_collected)
        total_collision_results = sum(r.get('collisions_detected', 0) for r in results_collected)
        
        print(f"  Concurrent operations completed:")
        print(f"    Insertions: {total_insertions:,}")
        print(f"    Queries: {total_queries:,}")
        print(f"    Collision results: {total_collision_results:,}")
        print(f"    Errors: {len(errors)}")
        
        if errors:
            print("  ⚠️ Concurrent operation errors detected:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"    - {error}")
            
            self.leak_candidates.append({
                'test': 'concurrent_stress',
                'issue': 'concurrent_operation_errors',
                'details': f'{len(errors)} errors during concurrent operations'
            })
        else:
            print("  ✓ No concurrent operation errors detected")
        
        # Verify data consistency
        try:
            final_size = qt.size()
            all_points = qt.get_all_points()
            consistency_check = len(all_points) == final_size
            
            if consistency_check:
                print("  ✓ Data consistency maintained after concurrent operations")
            else:
                print(f"  ⚠️ Data inconsistency: size()={final_size}, get_all_points()={len(all_points)}")
                self.leak_candidates.append({
                    'test': 'concurrent_stress',
                    'issue': 'data_inconsistency',
                    'details': f'size()={final_size} != get_all_points()={len(all_points)}'
                })
        except Exception as e:
            print(f"  ⚠️ Consistency check failed: {e}")
        
        del qt
        gc.collect()
        
        self.test_results.append(("Concurrent Stress", len(errors) == 0, f"{len(errors)} errors"))
    
    def test_memory_leak_detection(self):
        """Comprehensive memory leak detection"""
        print("🔍 Testing Memory Leak Detection...")
        
        # Get current memory state
        if tracemalloc.is_tracing():
            snapshot_before = tracemalloc.take_snapshot()
        
        baseline_snapshot = self.take_memory_snapshot("leak_detection_baseline")
        
        # Perform operations that should not leak
        for iteration in range(20):
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            
            # Various operations
            for i in range(1000):
                x = random.uniform(0, 1000)
                y = random.uniform(0, 1000)
                data = {"iteration": iteration, "point": i, "data": "x" * 50}
                qt.insert(x, y, data)
            
            # Query operations
            for _ in range(50):
                qt.query(random.uniform(0, 900), random.uniform(0, 900), 100, 100)
            
            # Collision detection
            qt.detect_collisions(10.0)
            
            # Clean up
            del qt
            gc.collect()
            
            # Sample memory periodically
            if iteration % 5 == 0:
                snapshot = self.take_memory_snapshot(
                    f"leak_detection_iter_{iteration}",
                    {"iteration": iteration}
                )
                
                memory_growth = snapshot.get('memory_delta', 0) - baseline_snapshot.get('memory_delta', 0)
                if memory_growth > 20:  # More than 20MB growth
                    self.leak_candidates.append({
                        'test': 'memory_leak_detection',
                        'issue': 'progressive_memory_growth',
                        'details': f'Memory grew by {memory_growth:.2f} MB after {iteration} iterations'
                    })
        
        final_snapshot = self.take_memory_snapshot("leak_detection_final")
        
        # Analyze overall memory growth
        total_growth = final_snapshot.get('memory_delta', 0) - baseline_snapshot.get('memory_delta', 0)
        print(f"  Total memory growth during leak test: {total_growth:.2f} MB")
        
        if total_growth > 10:
            self.leak_candidates.append({
                'test': 'memory_leak_detection',
                'issue': 'significant_total_growth',
                'details': f'Total memory growth of {total_growth:.2f} MB suggests potential leak'
            })
            print(f"  ⚠️ Potential memory leak detected: {total_growth:.2f} MB growth")
        else:
            print("  ✓ No significant memory leaks detected")
        
        # Tracemalloc analysis
        if tracemalloc.is_tracing():
            snapshot_after = tracemalloc.take_snapshot()
            top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            
            print("  Top 5 memory allocation differences:")
            for index, stat in enumerate(top_stats[:5], 1):
                print(f"    {index}. {stat}")
        
        self.test_results.append(("Memory Leak Detection", len(self.leak_candidates) == 0, f"{len(self.leak_candidates)} potential leaks"))
    
    def analyze_memory_patterns(self):
        """Analyze memory usage patterns from snapshots"""
        print("\n📊 Analyzing Memory Usage Patterns...")
        
        if not self.memory_snapshots:
            print("  No memory snapshots available for analysis")
            return
        
        # Calculate memory statistics
        memory_deltas = [s.get('memory_delta', 0) for s in self.memory_snapshots if s.get('memory_delta') is not None]
        
        if memory_deltas:
            min_memory = min(memory_deltas)
            max_memory = max(memory_deltas)
            avg_memory = sum(memory_deltas) / len(memory_deltas)
            
            print(f"  Memory usage statistics:")
            print(f"    Minimum delta: {min_memory:.2f} MB")
            print(f"    Maximum delta: {max_memory:.2f} MB") 
            print(f"    Average delta: {avg_memory:.2f} MB")
            print(f"    Peak increase: {max_memory - min_memory:.2f} MB")
        
        # Look for concerning patterns
        memory_growth_phases = []
        for i in range(1, len(self.memory_snapshots)):
            current = self.memory_snapshots[i].get('memory_delta', 0)
            previous = self.memory_snapshots[i-1].get('memory_delta', 0)
            if current > previous + 5:  # Significant growth
                growth = current - previous
                memory_growth_phases.append({
                    'from': self.memory_snapshots[i-1]['label'],
                    'to': self.memory_snapshots[i]['label'],
                    'growth_mb': growth
                })
        
        if memory_growth_phases:
            print(f"\n  Significant memory growth phases:")
            for phase in memory_growth_phases:
                print(f"    {phase['from']} → {phase['to']}: +{phase['growth_mb']:.2f} MB")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE MEMORY STRESS TEST REPORT")
        print("=" * 80)
        
        # Test results summary
        passed_tests = [r for r in self.test_results if r[1]]
        failed_tests = [r for r in self.test_results if not r[1]]
        
        total_tests = len(self.test_results)
        success_rate = len(passed_tests) / total_tests * 100 if total_tests > 0 else 0
        
        print(f"\n🧪 Test Results Summary:")
        print(f"  Success Rate: {success_rate:.1f}% ({len(passed_tests)}/{total_tests} tests passed)")
        
        # Detailed test results
        for test_name, passed, error_msg in self.test_results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status} {test_name}")
            if not passed and error_msg:
                print(f"      Error: {error_msg}")
        
        # Memory leak analysis
        print(f"\n🔍 Memory Leak Analysis:")
        if self.leak_candidates:
            print(f"  ⚠️ {len(self.leak_candidates)} potential issues detected:")
            
            issue_counts = defaultdict(int)
            for candidate in self.leak_candidates:
                issue_counts[candidate['issue']] += 1
            
            for issue_type, count in issue_counts.items():
                print(f"    • {issue_type}: {count} occurrences")
            
            print(f"\n  Detailed leak candidates:")
            for i, candidate in enumerate(self.leak_candidates[:10], 1):  # Show top 10
                print(f"    {i}. Test: {candidate['test']}")
                print(f"       Issue: {candidate['issue']}")
                print(f"       Details: {candidate['details']}")
                print()
        else:
            print("  ✓ No memory leaks detected")
        
        # Memory usage summary
        if self.memory_snapshots:
            print(f"📊 Memory Usage Summary:")
            memory_deltas = [s.get('memory_delta', 0) for s in self.memory_snapshots if s.get('memory_delta') is not None]
            if memory_deltas:
                print(f"  Peak memory increase: {max(memory_deltas):.2f} MB")
                print(f"  Final memory state: {memory_deltas[-1]:.2f} MB from baseline")
                
                # Memory efficiency assessment
                if max(memory_deltas) < 100:
                    print("  ✓ Good: Memory usage remained under 100MB")
                elif max(memory_deltas) < 500:
                    print("  ⚠️ Moderate: Memory usage between 100-500MB")
                else:
                    print("  ❌ High: Memory usage exceeded 500MB")
        
        # Performance insights
        total_time = time.time() - self.start_time
        print(f"\n⏱️ Performance Summary:")
        print(f"  Total test duration: {total_time:.1f} seconds")
        print(f"  Memory snapshots taken: {len(self.memory_snapshots)}")
        
        # Overall verdict
        print(f"\n🏆 Overall Verdict:")
        
        if success_rate == 100 and len(self.leak_candidates) == 0:
            print("  ✅ EXCELLENT: All tests passed with no memory leaks detected")
            print("     The QuadTree implementation shows excellent memory management")
        elif success_rate >= 90 and len(self.leak_candidates) <= 2:
            print("  ✅ GOOD: High success rate with minimal memory concerns")
            print("     The implementation is solid with minor areas for improvement")
        elif success_rate >= 75:
            print("  ⚠️ FAIR: Most tests passed but some memory issues detected")
            print("     Consider investigating the reported memory concerns")
        else:
            print("  ❌ POOR: Significant issues detected")
            print("     The implementation needs substantial memory management improvements")
        
        # Save detailed report to JSON
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'test_results': [
                {'test': name, 'passed': passed, 'error': error} 
                for name, passed, error in self.test_results
            ],
            'memory_snapshots': self.memory_snapshots,
            'leak_candidates': self.leak_candidates,
            'summary': {
                'total_tests': total_tests,
                'passed_tests': len(passed_tests),
                'success_rate': success_rate,
                'potential_leaks': len(self.leak_candidates),
                'test_duration_seconds': total_time,
                'peak_memory_mb': max(memory_deltas) if memory_deltas else 0
            }
        }
        
        report_filename = f"quadtree_memory_stress_report_{int(time.time())}.json"
        with open(report_filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_filename}")

def main():
    """Run comprehensive memory stress tests"""
    print("QuadTree C++17 Implementation - Comprehensive Memory Stress Test Suite")
    print("Designed to detect memory leaks and assess implementation quality")
    print()
    
    try:
        tester = MemoryStressTester()
        tester.run_comprehensive_stress_tests()
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        traceback.print_exc()
    finally:
        # Clean up tracemalloc
        if tracemalloc.is_tracing():
            tracemalloc.stop()

if __name__ == "__main__":
    main()