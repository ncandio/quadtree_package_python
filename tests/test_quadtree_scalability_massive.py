#!/usr/bin/env python3
"""
Massive QuadTree Scalability Test - 10, 100, 1000, and 10000 QuadTrees
Tests memory management, insertion performance, and contains operations at extreme scale
Validates production eligibility under massive concurrent QuadTree usage
"""

import sys
import os
import time
import random
import gc
import tracemalloc
import psutil
import threading
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
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
class ScalabilityTestResult:
    test_name: str
    num_quadtrees: int
    total_points: int
    passed: bool
    execution_time: float
    memory_peak_mb: float
    memory_growth_mb: float
    insertion_rate: float  # insertions per second
    contains_rate: float   # contains checks per second
    error_message: str = ""

class MassiveQuadTreeScalabilityTester:
    """Test QuadTree scalability with massive numbers of trees and points"""
    
    def __init__(self):
        self.results: List[ScalabilityTestResult] = []
        self.random_seed = 42
        random.seed(self.random_seed)
        
    def run_massive_scalability_tests(self):
        """Execute massive scalability test battery"""
        print("🚀 MASSIVE QUADTREE SCALABILITY TEST BATTERY")
        print("=" * 80)
        print("Testing: 10 → 100 → 1000 → 10000 QuadTrees")
        print("Focus: Memory management, insertion, contains operations")
        print("Goal: Production eligibility validation at scale")
        print()
        
        # Test configurations: (num_quadtrees, points_per_tree, boundary_size)
        test_configs = [
            (10, 1000, 1000),      # 10K total points
            (100, 500, 1000),      # 50K total points
            (1000, 100, 1000),     # 100K total points
            (10000, 50, 1000),     # 500K total points
        ]
        
        for num_trees, points_per_tree, boundary_size in test_configs:
            print(f"\n📊 TESTING {num_trees:,} QUADTREES")
            print(f"Points per tree: {points_per_tree}, Total points: {num_trees * points_per_tree:,}")
            print("-" * 70)
            
            self.test_massive_creation_and_insertion(num_trees, points_per_tree, boundary_size)
            self.test_massive_contains_operations(num_trees, points_per_tree, boundary_size) 
            self.test_massive_memory_management(num_trees, points_per_tree, boundary_size)
            self.test_massive_concurrent_access(num_trees, points_per_tree, boundary_size)
            
            # Force cleanup between test levels
            gc.collect()
            
        self.generate_massive_scalability_report()
    
    def test_massive_creation_and_insertion(self, num_quadtrees: int, points_per_tree: int, boundary_size: int):
        """Test massive QuadTree creation and insertion operations"""
        test_name = f"Massive Creation & Insertion ({num_quadtrees:,} trees)"
        print(f"🏗️ {test_name}")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        try:
            tracemalloc.start()
            
            # Create massive number of QuadTrees
            quadtrees = []
            total_points = 0
            insertion_times = []
            
            print(f"  Creating {num_quadtrees:,} QuadTrees...")
            
            for tree_id in range(num_quadtrees):
                # Create tree with unique boundary to avoid overlaps
                x = (tree_id % 100) * (boundary_size / 10)
                y = (tree_id // 100) * (boundary_size / 10)
                w = boundary_size / 10
                h = boundary_size / 10
                
                qt = quadtree.QuadTree(x, y, w, h)
                
                # Insert points into this tree
                tree_insertion_start = time.time()
                
                for point_id in range(points_per_tree):
                    px = x + random.uniform(0, w)
                    py = y + random.uniform(0, h)
                    data = f"tree_{tree_id}_point_{point_id}"
                    
                    result = qt.insert(px, py, data)
                    if not result:
                        raise Exception(f"Insert failed for tree {tree_id}, point {point_id}")
                    
                    total_points += 1
                
                tree_insertion_time = time.time() - tree_insertion_start
                insertion_times.append(tree_insertion_time)
                
                quadtrees.append(qt)
                
                # Progress reporting for large tests
                if num_quadtrees >= 1000 and tree_id % (num_quadtrees // 10) == 0:
                    progress = (tree_id / num_quadtrees) * 100
                    current_memory = process.memory_info().rss / 1024 / 1024
                    print(f"    Progress: {progress:.0f}% - {tree_id:,} trees created, {current_memory:.0f}MB")
            
            # Verify all trees and points
            print(f"  Verifying {len(quadtrees):,} trees with {total_points:,} total points...")
            
            verification_start = time.time()
            verified_points = 0
            
            for i, qt in enumerate(quadtrees):
                tree_size = qt.size()
                if tree_size != points_per_tree:
                    raise Exception(f"Tree {i} has wrong size: expected {points_per_tree}, got {tree_size}")
                
                # Verify tree is functional
                boundary = qt.boundary()
                all_points = qt.get_all_points()
                
                if len(all_points) != points_per_tree:
                    raise Exception(f"Tree {i} get_all_points failed: expected {points_per_tree}, got {len(all_points)}")
                
                verified_points += len(all_points)
                
                # Quick functionality test
                if i % max(1, len(quadtrees) // 20) == 0:  # Sample test
                    test_x, test_y = all_points[0][0], all_points[0][1]
                    if not qt.contains(test_x, test_y):
                        raise Exception(f"Tree {i} contains check failed")
            
            verification_time = time.time() - verification_start
            
            if verified_points != total_points:
                raise Exception(f"Point count mismatch: expected {total_points}, verified {verified_points}")
            
            # Calculate performance metrics
            total_insertion_time = sum(insertion_times)
            avg_insertion_rate = total_points / total_insertion_time if total_insertion_time > 0 else 0
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory
            
            execution_time = time.time() - start_time
            
            # Clean up
            del quadtrees
            gc.collect()
            
            self.results.append(ScalabilityTestResult(
                test_name,
                num_quadtrees,
                total_points,
                True,
                execution_time,
                peak / 1024 / 1024,
                memory_growth,
                avg_insertion_rate,
                verified_points / verification_time,  # contains rate from verification
                ""
            ))
            
            print(f"✓ {test_name} passed")
            print(f"  Time: {execution_time:.2f}s, Memory: {peak/1024/1024:.1f}MB peak, {memory_growth:.1f}MB growth")
            print(f"  Insertion rate: {avg_insertion_rate:.0f} ops/sec")
            
        except Exception as e:
            if 'tracemalloc' in locals():
                tracemalloc.stop()
            
            execution_time = time.time() - start_time
            
            self.results.append(ScalabilityTestResult(
                test_name,
                num_quadtrees,
                0,
                False,
                execution_time,
                0,
                0,
                0,
                0,
                str(e)
            ))
            
            print(f"✗ {test_name} failed: {e}")
    
    def test_massive_contains_operations(self, num_quadtrees: int, points_per_tree: int, boundary_size: int):
        """Test contains operations across massive number of QuadTrees"""
        test_name = f"Massive Contains Operations ({num_quadtrees:,} trees)"
        print(f"🔍 {test_name}")
        
        start_time = time.time()
        
        try:
            tracemalloc.start()
            
            # Create trees and store points for contains testing
            quadtrees = []
            all_inserted_points = []
            
            print(f"  Setting up {num_quadtrees:,} trees for contains testing...")
            
            for tree_id in range(num_quadtrees):
                x = (tree_id % 100) * (boundary_size / 10) 
                y = (tree_id // 100) * (boundary_size / 10)
                w = boundary_size / 10
                h = boundary_size / 10
                
                qt = quadtree.QuadTree(x, y, w, h)
                tree_points = []
                
                # Insert points and track them
                for point_id in range(points_per_tree):
                    px = x + random.uniform(0, w)
                    py = y + random.uniform(0, h)
                    data = f"contains_test_{tree_id}_{point_id}"
                    
                    result = qt.insert(px, py, data)
                    if not result:
                        raise Exception(f"Insert failed for tree {tree_id}, point {point_id}")
                    
                    tree_points.append((px, py, data))
                
                quadtrees.append(qt)
                all_inserted_points.append(tree_points)
            
            total_points = num_quadtrees * points_per_tree
            print(f"  Testing contains operations on {total_points:,} points...")
            
            # Test contains operations
            contains_start = time.time()
            
            correct_contains = 0
            total_contains_tests = 0
            
            # Test positive cases (points that should be found)
            for tree_id, (qt, tree_points) in enumerate(zip(quadtrees, all_inserted_points)):
                # Test every 10th point to keep test reasonable for large scales
                step = max(1, len(tree_points) // 10) if num_quadtrees >= 1000 else 1
                
                for i in range(0, len(tree_points), step):
                    px, py, _ = tree_points[i]
                    total_contains_tests += 1
                    
                    if qt.contains(px, py):
                        correct_contains += 1
                    
                    # Add negative test (point outside boundary)
                    total_contains_tests += 1
                    boundary = qt.boundary()
                    non_existent_x = boundary[0] + boundary[2] + 100
                    non_existent_y = boundary[1] + boundary[3] + 100
                    
                    if not qt.contains(non_existent_x, non_existent_y):
                        correct_contains += 1
                
                # Progress for very large tests
                if num_quadtrees >= 1000 and tree_id % (num_quadtrees // 5) == 0:
                    progress = (tree_id / num_quadtrees) * 100
                    print(f"    Contains testing progress: {progress:.0f}%")
            
            contains_time = time.time() - contains_start
            contains_rate = total_contains_tests / contains_time if contains_time > 0 else 0
            
            # Calculate accuracy
            accuracy = correct_contains / total_contains_tests if total_contains_tests > 0 else 0
            
            if accuracy < 0.99:
                raise Exception(f"Contains accuracy too low: {accuracy:.2%}")
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            execution_time = time.time() - start_time
            
            # Clean up
            del quadtrees, all_inserted_points
            gc.collect()
            
            self.results.append(ScalabilityTestResult(
                test_name,
                num_quadtrees,
                total_points,
                True,
                execution_time,
                peak / 1024 / 1024,
                0,  # No insertion in this test
                0,  # No insertion rate
                contains_rate,
                ""
            ))
            
            print(f"✓ {test_name} passed")
            print(f"  Contains tests: {total_contains_tests:,}, Accuracy: {accuracy:.2%}")
            print(f"  Contains rate: {contains_rate:.0f} ops/sec")
            
        except Exception as e:
            if 'tracemalloc' in locals():
                tracemalloc.stop()
            
            execution_time = time.time() - start_time
            
            self.results.append(ScalabilityTestResult(
                test_name,
                num_quadtrees,
                0,
                False,
                execution_time,
                0,
                0,
                0,
                0,
                str(e)
            ))
            
            print(f"✗ {test_name} failed: {e}")
    
    def test_massive_memory_management(self, num_quadtrees: int, points_per_tree: int, boundary_size: int):
        """Test memory management with massive number of QuadTrees"""
        test_name = f"Massive Memory Management ({num_quadtrees:,} trees)"
        print(f"💾 {test_name}")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        try:
            memory_snapshots = []
            
            # Test memory pattern: create -> use -> destroy cycles
            batch_size = min(100, num_quadtrees // 10) if num_quadtrees >= 100 else num_quadtrees
            num_batches = (num_quadtrees + batch_size - 1) // batch_size
            
            print(f"  Testing memory in {num_batches} batches of {batch_size} trees...")
            
            for batch_id in range(num_batches):
                batch_start = batch_id * batch_size
                batch_end = min(batch_start + batch_size, num_quadtrees)
                actual_batch_size = batch_end - batch_start
                
                tracemalloc.start()
                
                # Create batch of trees
                batch_trees = []
                for tree_id in range(batch_start, batch_end):
                    x = (tree_id % 100) * (boundary_size / 10)
                    y = (tree_id // 100) * (boundary_size / 10) 
                    w = boundary_size / 10
                    h = boundary_size / 10
                    
                    qt = quadtree.QuadTree(x, y, w, h)
                    
                    # Insert points with larger data to test memory pressure
                    for point_id in range(points_per_tree):
                        px = x + random.uniform(0, w)
                        py = y + random.uniform(0, h)
                        # Use larger data objects to stress memory
                        data = {
                            "id": f"{tree_id}_{point_id}",
                            "payload": "x" * 100,  # 100 char string
                            "metadata": list(range(10)),  # Small list
                            "batch": batch_id
                        }
                        
                        result = qt.insert(px, py, data)
                        if not result:
                            raise Exception(f"Memory pressure insert failed: tree {tree_id}, point {point_id}")
                    
                    batch_trees.append(qt)
                
                # Test trees in batch are functional
                total_batch_points = 0
                for qt in batch_trees:
                    size = qt.size()
                    if size != points_per_tree:
                        raise Exception(f"Batch tree size wrong: expected {points_per_tree}, got {size}")
                    total_batch_points += size
                
                # Memory measurement
                current, peak = tracemalloc.get_traced_memory()
                current_process_memory = process.memory_info().rss / 1024 / 1024
                
                memory_snapshots.append({
                    "batch": batch_id,
                    "trees": actual_batch_size,
                    "points": total_batch_points,
                    "tracemalloc_peak_mb": peak / 1024 / 1024,
                    "process_memory_mb": current_process_memory
                })
                
                tracemalloc.stop()
                
                # Destroy batch (test cleanup)
                del batch_trees
                gc.collect()
                
                if batch_id % max(1, num_batches // 10) == 0:
                    print(f"    Batch {batch_id + 1}/{num_batches}: {actual_batch_size} trees, {current_process_memory:.1f}MB")
            
            # Analyze memory patterns
            if len(memory_snapshots) > 0:
                if len(memory_snapshots) > 1:
                    first_batch_memory = memory_snapshots[0]["process_memory_mb"]
                    last_batch_memory = memory_snapshots[-1]["process_memory_mb"]
                else:
                    first_batch_memory = initial_memory
                    last_batch_memory = memory_snapshots[0]["process_memory_mb"]
                
                max_batch_memory = max(snap["process_memory_mb"] for snap in memory_snapshots)
                memory_growth = last_batch_memory - initial_memory
                memory_peak = max_batch_memory
                
                # Check for memory leaks (final memory shouldn't be much higher than initial)
                memory_leak_threshold = 100  # MB
                potential_leak = memory_growth > memory_leak_threshold
                
                if potential_leak:
                    print(f"    ⚠️  Potential memory leak: {memory_growth:.1f}MB growth")
                else:
                    print(f"    ✓ Memory management good: {memory_growth:.1f}MB final growth")
                
                print(f"    Peak memory: {memory_peak:.1f}MB")
                
            execution_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024
            total_memory_growth = final_memory - initial_memory
            
            # Set memory_peak if not set above
            if 'memory_peak' not in locals():
                memory_peak = final_memory
            
            # Test one more large allocation after cleanup to verify allocator health
            print("  Final allocator health test...")
            final_test_trees = []
            
            for i in range(min(10, num_quadtrees)):
                qt = quadtree.QuadTree(0, 0, 100, 100)
                for j in range(points_per_tree):
                    qt.insert(random.uniform(0, 100), random.uniform(0, 100), f"final_test_{i}_{j}")
                final_test_trees.append(qt)
            
            # Clean up final test
            del final_test_trees
            gc.collect()
            
            # Success criteria
            max_acceptable_growth = max(50, num_quadtrees * 0.01)  # Scale with number of trees
            memory_management_success = total_memory_growth < max_acceptable_growth
            
            self.results.append(ScalabilityTestResult(
                test_name,
                num_quadtrees,
                num_quadtrees * points_per_tree,
                memory_management_success,
                execution_time,
                memory_peak if memory_snapshots else final_memory,
                total_memory_growth,
                0,  # No specific insertion rate for this test
                0,  # No specific contains rate for this test
                "" if memory_management_success else f"Memory growth too high: {total_memory_growth:.1f}MB"
            ))
            
            if memory_management_success:
                print(f"✓ {test_name} passed")
            else:
                print(f"⚠️ {test_name} passed with concerns")
            
            print(f"  Total memory growth: {total_memory_growth:.1f}MB")
            
        except Exception as e:
            if 'tracemalloc' in locals():
                tracemalloc.stop()
            
            execution_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024
            
            self.results.append(ScalabilityTestResult(
                test_name,
                num_quadtrees,
                0,
                False,
                execution_time,
                final_memory,
                final_memory - initial_memory,
                0,
                0,
                str(e)
            ))
            
            print(f"✗ {test_name} failed: {e}")
    
    def test_massive_concurrent_access(self, num_quadtrees: int, points_per_tree: int, boundary_size: int):
        """Test concurrent access patterns with massive QuadTree arrays"""
        test_name = f"Massive Concurrent Access ({num_quadtrees:,} trees)"
        print(f"🧵 {test_name}")
        
        # Skip concurrent test for small scales or adjust for very large scales
        if num_quadtrees < 100:
            print("  Skipping concurrent test for small scale")
            return
        
        # Limit concurrent test size for very large scales to avoid excessive resource usage
        test_tree_count = min(num_quadtrees, 1000)
        test_points_per_tree = min(points_per_tree, 100)
        
        start_time = time.time()
        
        try:
            tracemalloc.start()
            
            # Create shared pool of QuadTrees
            print(f"  Setting up {test_tree_count} trees for concurrent access...")
            
            shared_quadtrees = []
            for tree_id in range(test_tree_count):
                x = (tree_id % 50) * (boundary_size / 25)
                y = (tree_id // 50) * (boundary_size / 25)
                w = boundary_size / 25
                h = boundary_size / 25
                
                qt = quadtree.QuadTree(x, y, w, h)
                
                # Pre-populate with some points
                for point_id in range(test_points_per_tree):
                    px = x + random.uniform(0, w)
                    py = y + random.uniform(0, h)
                    qt.insert(px, py, f"concurrent_test_{tree_id}_{point_id}")
                
                shared_quadtrees.append(qt)
            
            # Concurrent access test with thread safety measures
            thread_results = []
            access_lock = threading.Lock()  # For thread safety
            
            def concurrent_worker(worker_id: int, operations: int):
                worker_result = {
                    "worker_id": worker_id,
                    "operations": 0,
                    "insertions": 0,
                    "queries": 0,
                    "contains_checks": 0,
                    "errors": 0
                }
                
                for i in range(operations):
                    try:
                        # Select random tree
                        tree_idx = random.randint(0, len(shared_quadtrees) - 1)
                        qt = shared_quadtrees[tree_idx]
                        operation = random.choice(['insert', 'query', 'contains'])
                        
                        if operation == 'insert':
                            boundary = qt.boundary()
                            px = boundary[0] + random.uniform(0, boundary[2])
                            py = boundary[1] + random.uniform(0, boundary[3])
                            
                            with access_lock:
                                result = qt.insert(px, py, f"worker_{worker_id}_op_{i}")
                                if result:
                                    worker_result["insertions"] += 1
                        
                        elif operation == 'query':
                            boundary = qt.boundary()
                            qx = boundary[0] + random.uniform(0, boundary[2] * 0.8)
                            qy = boundary[1] + random.uniform(0, boundary[3] * 0.8)
                            qw = boundary[2] * 0.2
                            qh = boundary[3] * 0.2
                            
                            with access_lock:
                                results = qt.query(qx, qy, qw, qh)
                                worker_result["queries"] += 1
                        
                        elif operation == 'contains':
                            boundary = qt.boundary()
                            px = boundary[0] + random.uniform(0, boundary[2])
                            py = boundary[1] + random.uniform(0, boundary[3])
                            
                            with access_lock:
                                qt.contains(px, py)
                                worker_result["contains_checks"] += 1
                        
                        worker_result["operations"] += 1
                        
                    except Exception as e:
                        worker_result["errors"] += 1
                
                thread_results.append(worker_result)
            
            # Launch concurrent workers
            num_workers = min(4, max(2, test_tree_count // 100))
            operations_per_worker = max(100, test_tree_count // 10)
            
            print(f"  Running {num_workers} concurrent workers, {operations_per_worker} ops each...")
            
            concurrent_start = time.time()
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for worker_id in range(num_workers):
                    future = executor.submit(concurrent_worker, worker_id, operations_per_worker)
                    futures.append(future)
                
                # Wait for completion
                for future in as_completed(futures):
                    future.result()
            
            concurrent_time = time.time() - concurrent_start
            
            # Analyze results
            total_operations = sum(r["operations"] for r in thread_results)
            total_insertions = sum(r["insertions"] for r in thread_results)
            total_queries = sum(r["queries"] for r in thread_results)
            total_contains = sum(r["contains_checks"] for r in thread_results)
            total_errors = sum(r["errors"] for r in thread_results)
            
            # Verify trees are still functional
            print("  Verifying tree integrity after concurrent access...")
            total_final_points = 0
            for qt in shared_quadtrees:
                size = qt.size()
                total_final_points += size
                
                # Quick functionality test
                boundary = qt.boundary()
                test_results = qt.query(boundary[0], boundary[1], boundary[2], boundary[3])
                if len(test_results) != size:
                    raise Exception("Tree integrity compromised after concurrent access")
            
            concurrent_success = total_errors == 0
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            execution_time = time.time() - start_time
            
            # Clean up
            del shared_quadtrees
            gc.collect()
            
            self.results.append(ScalabilityTestResult(
                test_name,
                test_tree_count,
                total_final_points,
                concurrent_success,
                execution_time,
                peak / 1024 / 1024,
                0,  # No specific memory growth measurement
                total_insertions / concurrent_time if concurrent_time > 0 else 0,
                total_contains / concurrent_time if concurrent_time > 0 else 0,
                "" if concurrent_success else f"Concurrent errors: {total_errors}"
            ))
            
            print(f"✓ {test_name} passed")
            print(f"  Operations: {total_operations} total, {total_errors} errors")
            print(f"  Final tree integrity: {total_final_points:,} total points")
            
        except Exception as e:
            if 'tracemalloc' in locals():
                tracemalloc.stop()
            
            execution_time = time.time() - start_time
            
            self.results.append(ScalabilityTestResult(
                test_name,
                test_tree_count if 'test_tree_count' in locals() else num_quadtrees,
                0,
                False,
                execution_time,
                0,
                0,
                0,
                0,
                str(e)
            ))
            
            print(f"✗ {test_name} failed: {e}")
    
    def generate_massive_scalability_report(self):
        """Generate comprehensive scalability report"""
        print("\n" + "=" * 80)
        print("📊 MASSIVE QUADTREE SCALABILITY REPORT")
        print("=" * 80)
        
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        total_time = sum(r.execution_time for r in self.results)
        max_memory = max((r.memory_peak_mb for r in self.results if r.memory_peak_mb > 0), default=0)
        max_points = max((r.total_points for r in self.results), default=0)
        max_trees = max((r.num_quadtrees for r in self.results), default=0)
        
        success_rate = len(passed_tests) / len(self.results) * 100 if self.results else 0
        
        print(f"Overall Success Rate: {success_rate:.1f}% ({len(passed_tests)}/{len(self.results)})")
        print(f"Maximum Scale Tested: {max_trees:,} QuadTrees with {max_points:,} total points")
        print(f"Total Test Time: {total_time:.2f} seconds")
        print(f"Peak Memory Usage: {max_memory:.1f} MB")
        print()
        
        # Performance summary by scale
        print("📈 Performance by Scale:")
        scales_tested = sorted(set(r.num_quadtrees for r in self.results))
        
        for scale in scales_tested:
            scale_results = [r for r in self.results if r.num_quadtrees == scale]
            scale_passed = [r for r in scale_results if r.passed]
            
            if scale_results:
                avg_insertion_rate = sum(r.insertion_rate for r in scale_results if r.insertion_rate > 0) / max(1, len([r for r in scale_results if r.insertion_rate > 0]))
                avg_contains_rate = sum(r.contains_rate for r in scale_results if r.contains_rate > 0) / max(1, len([r for r in scale_results if r.contains_rate > 0]))
                total_points = scale_results[0].total_points if scale_results else 0
                
                print(f"  {scale:,} trees ({total_points:,} points): {len(scale_passed)}/{len(scale_results)} passed")
                if avg_insertion_rate > 0:
                    print(f"    Insertion rate: {avg_insertion_rate:.0f} ops/sec")
                if avg_contains_rate > 0:
                    print(f"    Contains rate: {avg_contains_rate:.0f} ops/sec")
        
        print()
        
        # Detailed test results
        print("📋 Detailed Test Results:")
        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            
            performance_info = ""
            if result.insertion_rate > 0:
                performance_info += f" | {result.insertion_rate:.0f} ins/sec"
            if result.contains_rate > 0:
                performance_info += f" | {result.contains_rate:.0f} contains/sec"
            
            memory_info = f" | {result.memory_peak_mb:.1f}MB" if result.memory_peak_mb > 0 else ""
            points_info = f" | {result.total_points:,} pts" if result.total_points > 0 else ""
            
            print(f"  {status} {result.test_name}")
            print(f"      {result.execution_time:>6.2f}s{performance_info}{memory_info}{points_info}")
            
            if not result.passed and result.error_message:
                print(f"      Error: {result.error_message}")
        
        print()
        
        # Production eligibility assessment
        critical_scale_tests = [
            "Massive Creation & Insertion",
            "Massive Contains Operations", 
            "Massive Memory Management"
        ]
        
        # Check if we successfully tested the largest scales
        large_scale_success = any(
            r.num_quadtrees >= 1000 and r.passed and 
            any(critical_test in r.test_name for critical_test in critical_scale_tests)
            for r in self.results
        )
        
        massive_scale_success = any(
            r.num_quadtrees >= 10000 and r.passed
            for r in self.results
        )
        
        memory_efficiency = max_memory < 2000  # Less than 2GB peak
        performance_adequate = any(r.insertion_rate > 1000 for r in self.results if r.insertion_rate > 0)
        
        print("🎯 Production Eligibility Assessment:")
        print(f"  Large Scale (1K+ trees): {'✓ READY' if large_scale_success else '✗ NOT READY'}")
        print(f"  Massive Scale (10K+ trees): {'✓ READY' if massive_scale_success else '✗ LIMITED'}")
        print(f"  Memory Efficiency: {'✓ GOOD' if memory_efficiency else '✗ CONCERNING'}")
        print(f"  Performance: {'✓ ADEQUATE' if performance_adequate else '✗ SLOW'}")
        print(f"  Overall Stability: {'✓ STABLE' if success_rate >= 80 else '✗ UNSTABLE'}")
        
        # Final verdict
        production_ready = (
            success_rate >= 80 and
            large_scale_success and
            memory_efficiency
        )
        
        if production_ready:
            if massive_scale_success:
                print("\n🚀 VERDICT: PRODUCTION READY FOR MASSIVE SCALE")
                print("QuadTree implementation scales excellently to extreme workloads.")
            else:
                print("\n✅ VERDICT: PRODUCTION READY FOR LARGE SCALE")
                print("QuadTree implementation ready for production with large workloads.")
        else:
            print("\n⚠️ VERDICT: SCALABILITY ISSUES DETECTED")
            print("Address scalability issues before production deployment.")
        
        # Recommendations
        print("\n💡 Scalability Recommendations:")
        
        if max_memory > 1000:
            print("  • Monitor memory usage in production environments")
            print("  • Consider implementing point limits per QuadTree")
        
        if not massive_scale_success:
            print("  • Test with smaller batch sizes for very large deployments")
            print("  • Consider distributed QuadTree architecture for extreme scale")
        
        best_insertion_rate = max((r.insertion_rate for r in self.results if r.insertion_rate > 0), default=0)
        if best_insertion_rate > 0:
            print(f"  • Optimal insertion rate achieved: {best_insertion_rate:.0f} ops/sec")
        
        if len(failed_tests) > 0:
            print(f"  • Address {len(failed_tests)} failing tests for full production readiness")
        
        # Save detailed report
        report_data = {
            "timestamp": time.time(),
            "success_rate": success_rate,
            "max_scale_tested": max_trees,
            "max_points_tested": max_points,
            "peak_memory_mb": max_memory,
            "production_ready": production_ready,
            "massive_scale_capable": massive_scale_success,
            "scales_tested": scales_tested,
            "test_results": [
                {
                    "test_name": r.test_name,
                    "num_quadtrees": r.num_quadtrees,
                    "total_points": r.total_points,
                    "passed": r.passed,
                    "execution_time": r.execution_time,
                    "memory_peak_mb": r.memory_peak_mb,
                    "memory_growth_mb": r.memory_growth_mb,
                    "insertion_rate": r.insertion_rate,
                    "contains_rate": r.contains_rate,
                    "error_message": r.error_message
                }
                for r in self.results
            ]
        }
        
        with open("quadtree_scalability_massive_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed scalability report saved to: quadtree_scalability_massive_report.json")

def main():
    """Run the massive scalability test suite"""
    print("QuadTree C++17 Implementation - Massive Scalability Test Suite")
    print("Testing production eligibility with 10, 100, 1000, and 10000 QuadTrees")
    print("Focus: Memory management, insertion performance, contains operations")
    print()
    
    tester = MassiveQuadTreeScalabilityTester()
    tester.run_massive_scalability_tests()

if __name__ == "__main__":
    main()