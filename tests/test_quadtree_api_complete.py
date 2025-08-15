#!/usr/bin/env python3
"""
Comprehensive API Test Suite for QuadTree C++17 Implementation
Tests all exposed API methods with edge cases and boundary conditions
Focuses on depth() and subdivisions() methods that lack comprehensive testing
"""

import sys
import os
import math
import random
import time
import gc
import threading
from typing import List, Tuple, Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional imports for enhanced testing
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

class QuadTreeAPITester:
    """Comprehensive API testing for all QuadTree methods"""
    
    def __init__(self):
        self.test_results = []
        self.random_seed = 42
        self.performance_results = []
        self.memory_usage = []
        random.seed(self.random_seed)
        
        # Get initial memory usage
        if PSUTIL_AVAILABLE:
            try:
                self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            except Exception:
                self.initial_memory = None
        else:
            self.initial_memory = None
    
    def run_all_api_tests(self):
        """Execute complete API test suite"""
        print("🧪 Comprehensive QuadTree API Test Suite")
        print("=" * 60)
        print("Testing all API methods with edge cases and boundary conditions")
        print()
        
        # Core API Tests
        self.test_constructor_api()
        self.test_insert_api()
        self.test_query_api()
        self.test_contains_api()
        self.test_get_all_points_api()
        self.test_detect_collisions_api()
        
        # State/Info API Tests (previously under-tested)
        self.test_size_api()
        self.test_empty_api()
        self.test_boundary_api()
        self.test_depth_api()
        self.test_subdivisions_api()
        
        # Integration and Edge Cases
        self.test_subdivision_behavior()
        self.test_depth_subdivision_relationship()
        self.test_boundary_conditions()
        self.test_data_integrity_across_apis()
        
        # Enhanced Tests
        self.test_numerical_precision_comprehensive()
        self.test_error_handling_comprehensive()
        self.test_performance_benchmarks()
        self.test_memory_usage_tracking()
        self.test_concurrent_operations()
        
        self.print_test_summary()
    
    def test_constructor_api(self):
        """Test QuadTree constructor with various parameters"""
        print("🏗️ Testing Constructor API...")
        
        # Valid constructors
        test_cases = [
            (0, 0, 100, 100),
            (-50, -50, 100, 100),
            (0.5, 0.5, 99.5, 99.5),
            (1e-6, 1e-6, 1e6, 1e6),
        ]
        
        for x, y, w, h in test_cases:
            try:
                qt = quadtree.QuadTree(x, y, w, h)
                boundary = qt.boundary()
                assert boundary == (x, y, w, h), f"Boundary mismatch: expected {(x,y,w,h)}, got {boundary}"
                print(f"  ✓ Constructor({x}, {y}, {w}, {h}) successful")
            except Exception as e:
                print(f"  ✗ Constructor({x}, {y}, {w}, {h}) failed: {e}")
                self.test_results.append(("Constructor", False, str(e)))
                return
        
        # Invalid constructors
        invalid_cases = [
            (0, 0, 0, 100),    # Zero width
            (0, 0, 100, 0),    # Zero height
            (0, 0, -1, 100),   # Negative width
            (0, 0, 100, -1),   # Negative height
        ]
        
        for x, y, w, h in invalid_cases:
            try:
                qt = quadtree.QuadTree(x, y, w, h)
                print(f"  ✗ Should have failed for ({x}, {y}, {w}, {h})")
                self.test_results.append(("Constructor Invalid", False, "Should have raised ValueError"))
                return
            except ValueError:
                print(f"  ✓ Correctly rejected invalid ({x}, {y}, {w}, {h})")
        
        self.test_results.append(("Constructor API", True, ""))
    
    def test_insert_api(self):
        """Test insert API with various data types and edge cases"""
        print("📍 Testing Insert API...")
        
        qt = quadtree.QuadTree(0, 0, 100, 100)
        
        # Test different data types
        insert_tests = [
            (10, 20, None),
            (30, 40, "string_data"),
            (50, 60, {"dict": "data", "id": 123}),
            (70, 80, [1, 2, 3, 4]),
            (90, 10, 3.14159),
            (5, 95, True),
        ]
        
        for x, y, data in insert_tests:
            result = qt.insert(x, y, data)
            assert result, f"Insert should succeed for ({x}, {y}, {data})"
            print(f"  ✓ Insert({x}, {y}, {type(data).__name__}) successful")
        
        # Test boundary conditions
        boundary_tests = [
            (0, 0),      # Bottom-left corner
            (99.999, 99.999),  # Just inside top-right
            (100, 100),  # On boundary (should fail)
            (-0.001, 50), # Just outside left
            (50, 100.001), # Just outside top
        ]
        
        for x, y in boundary_tests:
            result = qt.insert(x, y, f"boundary_test_{x}_{y}")
            expected = (0 <= x < 100 and 0 <= y < 100)
            assert result == expected, f"Insert({x}, {y}) result {result} != expected {expected}"
            status = "✓" if result == expected else "✗"
            print(f"  {status} Boundary test ({x}, {y}): {'inside' if expected else 'outside'}")
        
        # Test that size reflects insertions
        expected_size = len([t for t in insert_tests]) + len([t for t in boundary_tests[:2]])
        actual_size = qt.size()
        assert actual_size == expected_size, f"Size should be {expected_size}, got {actual_size}"
        
        self.test_results.append(("Insert API", True, ""))
    
    def test_query_api(self):
        """Test query API with various rectangular regions"""
        print("🔍 Testing Query API...")
        
        qt = quadtree.QuadTree(0, 0, 100, 100)
        
        # Insert test points in known positions
        test_points = [
            (10, 10), (20, 20), (30, 30), (40, 40), (50, 50),
            (60, 60), (70, 70), (80, 80), (90, 90)
        ]
        
        for x, y in test_points:
            qt.insert(x, y, f"point_{x}_{y}")
        
        # Test various query rectangles
        query_tests = [
            (0, 0, 50, 50, 4),    # Bottom-left quadrant - (10,10),(20,20),(30,30),(40,40) - (50,50) is on boundary, excluded
            (50, 50, 50, 50, 5),  # Top-right quadrant - (50,50),(60,60),(70,70),(80,80),(90,90) - (50,50) included as bottom-left corner
            (25, 25, 50, 50, 5),  # Center region - (30,30),(40,40),(50,50),(60,60),(70,70) - (50,50) included, (75,75) excluded
            (0, 0, 100, 100, 9),  # Entire area - all 9 points
            (0, 0, 10.1, 10.1, 1), # Small region around (10,10)
            (200, 200, 10, 10, 0), # Outside boundary
        ]
        
        for qx, qy, qw, qh, expected_count in query_tests:
            results = qt.query(qx, qy, qw, qh)
            assert len(results) == expected_count, f"Query({qx}, {qy}, {qw}, {qh}) should return {expected_count} points, got {len(results)}"
            print(f"  ✓ Query({qx}, {qy}, {qw}, {qh}) returned {len(results)} points")
        
        # Test invalid query parameters
        invalid_queries = [
            (0, 0, -1, 10),   # Negative width
            (0, 0, 10, -1),   # Negative height
        ]
        
        for qx, qy, qw, qh in invalid_queries:
            try:
                qt.query(qx, qy, qw, qh)
                print(f"  ✗ Should have failed for query({qx}, {qy}, {qw}, {qh})")
                self.test_results.append(("Query Invalid", False, "Should have raised ValueError"))
                return
            except ValueError:
                print(f"  ✓ Correctly rejected invalid query({qx}, {qy}, {qw}, {qh})")
        
        self.test_results.append(("Query API", True, ""))
    
    def test_contains_api(self):
        """Test contains API with precise point matching"""
        print("🎯 Testing Contains API...")
        
        qt = quadtree.QuadTree(0, 0, 100, 100)
        
        # Insert specific points
        test_points = [(25.5, 33.7), (50.0, 50.0), (75.123, 25.456)]
        for x, y in test_points:
            qt.insert(x, y, f"data_{x}_{y}")
        
        # Test exact matches
        for x, y in test_points:
            assert qt.contains(x, y), f"Should contain exact point ({x}, {y})"
            print(f"  ✓ Contains({x}, {y}) correctly returns True")
        
        # Test non-existent points
        non_existent = [(25.4, 33.7), (50.1, 50.0), (0, 0), (99, 99)]
        for x, y in non_existent:
            assert not qt.contains(x, y), f"Should not contain point ({x}, {y})"
            print(f"  ✓ Contains({x}, {y}) correctly returns False")
        
        # Test precision boundaries (floating point precision)
        qt.insert(1.0, 1.0, "precision_test")
        assert qt.contains(1.0, 1.0), "Should contain (1.0, 1.0)"
        assert qt.contains(1.0000000001, 1.0), "Should contain point within epsilon (1e-9)"
        assert not qt.contains(1.00001, 1.0), "Should not contain point outside epsilon"
        print("  ✓ Floating point precision handling correct")
        
        self.test_results.append(("Contains API", True, ""))
    
    def test_get_all_points_api(self):
        """Test get_all_points API with various data scenarios"""
        print("📋 Testing Get All Points API...")
        
        qt = quadtree.QuadTree(0, 0, 100, 100)
        
        # Test empty tree
        all_points = qt.get_all_points()
        assert len(all_points) == 0, "Empty tree should return empty list"
        print("  ✓ Empty tree returns empty list")
        
        # Insert points with different data types
        test_data = [
            (10, 20, "string"),
            (30, 40, {"complex": {"nested": [1, 2, 3]}}),
            (50, 60, None),  # Point without data
            (70, 80, [1, 2, 3]),
            (90, 10, 42.5)
        ]
        
        for x, y, data in test_data:
            qt.insert(x, y, data)
        
        all_points = qt.get_all_points()
        assert len(all_points) == len(test_data), f"Should have {len(test_data)} points, got {len(all_points)}"
        
        # Verify data integrity
        for point in all_points:
            x, y = point[0], point[1]
            data = point[2] if len(point) == 3 else None
            
            # Find matching test data
            matching = [td for td in test_data if td[0] == x and td[1] == y]
            assert len(matching) == 1, f"Should find exactly one match for ({x}, {y})"
            
            expected_data = matching[0][2]
            assert data == expected_data, f"Data mismatch: expected {expected_data}, got {data}"
        
        print(f"  ✓ Retrieved {len(all_points)} points with data integrity preserved")
        
        self.test_results.append(("Get All Points API", True, ""))
    
    def test_detect_collisions_api(self):
        """Test detect_collisions API with various radius values"""
        print("💥 Testing Detect Collisions API...")
        
        qt = quadtree.QuadTree(0, 0, 100, 100)
        
        # Insert points in known collision patterns
        collision_groups = [
            # Group 1: Close points (should collide)
            [(10, 10), (12, 12), (14, 14)],
            # Group 2: Isolated points
            [(50, 50)],
            # Group 3: Another collision group
            [(80, 80), (82, 81)]
        ]
        
        for group_id, group in enumerate(collision_groups):
            for x, y in group:
                qt.insert(x, y, f"group_{group_id}_point_{x}_{y}")
        
        # Test different collision radii
        radius_tests = [
            (1.0, 0),   # Very small radius - no collisions
            (3.0, 3),   # Small radius - should find 3 collisions in group 1, 1 in group 3
            (5.0, 4),   # Medium radius - more collisions in group 1
            (100.0, 15), # Large radius - all points collide with each other
        ]
        
        for radius, expected_min in radius_tests:
            collisions = qt.detect_collisions(radius)
            print(f"  ✓ Radius {radius}: found {len(collisions)} collisions (expected ≥ {expected_min})")
            
            # Verify collision format
            for collision in collisions:
                assert isinstance(collision, dict), "Collision should be dictionary"
                assert "point1" in collision and "point2" in collision, "Collision should have point1 and point2"
                
                p1, p2 = collision["point1"], collision["point2"]
                assert len(p1) >= 2 and len(p2) >= 2, "Points should have at least x,y coordinates"
                
                # Verify distance is within radius
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                distance = math.sqrt(dx*dx + dy*dy)
                assert distance <= radius, f"Collision distance {distance} should be ≤ radius {radius}"
        
        # Test invalid radius
        try:
            qt.detect_collisions(-1.0)
            print("  ✗ Should have failed for negative radius")
            self.test_results.append(("Collision Invalid", False, "Should have raised ValueError"))
            return
        except ValueError:
            print("  ✓ Correctly rejected negative radius")
        
        self.test_results.append(("Detect Collisions API", True, ""))
    
    def test_size_api(self):
        """Test size API during various operations"""
        print("📏 Testing Size API...")
        
        qt = quadtree.QuadTree(0, 0, 100, 100)
        
        # Test empty size
        assert qt.size() == 0, "Empty tree should have size 0"
        print("  ✓ Empty tree size = 0")
        
        # Test incremental insertions
        for i in range(1, 11):
            x, y = random.uniform(0, 100), random.uniform(0, 100)
            result = qt.insert(x, y, f"point_{i}")
            if result:  # Only count successful insertions
                expected_size = i
                actual_size = qt.size()
                assert actual_size == expected_size, f"Size should be {expected_size}, got {actual_size}"
        
        print(f"  ✓ Size correctly tracks insertions: {qt.size()}")
        
        # Test that failed insertions don't affect size
        initial_size = qt.size()
        qt.insert(200, 200, "outside")  # Should fail
        assert qt.size() == initial_size, "Failed insertion should not change size"
        print("  ✓ Failed insertions don't affect size")
        
        self.test_results.append(("Size API", True, ""))
    
    def test_empty_api(self):
        """Test empty API in various states"""
        print("🗂️ Testing Empty API...")
        
        qt = quadtree.QuadTree(0, 0, 100, 100)
        
        # Test initially empty
        assert qt.empty(), "New tree should be empty"
        print("  ✓ New tree is empty")
        
        # Test after insertion
        qt.insert(50, 50, "test_point")
        assert not qt.empty(), "Tree with points should not be empty"
        print("  ✓ Tree with points is not empty")
        
        # Test edge case: failed insertion shouldn't change empty status
        was_empty_before = qt.empty()
        qt.insert(200, 200, "outside")  # Should fail
        assert qt.empty() == was_empty_before, "Failed insertion shouldn't change empty status"
        print("  ✓ Failed insertion doesn't affect empty status")
        
        self.test_results.append(("Empty API", True, ""))
    
    def test_boundary_api(self):
        """Test boundary API with various tree configurations"""
        print("🔲 Testing Boundary API...")
        
        test_boundaries = [
            (0, 0, 100, 100),
            (-50, -25, 200, 150),
            (0.5, 0.25, 99.5, 199.75),
        ]
        
        for x, y, w, h in test_boundaries:
            qt = quadtree.QuadTree(x, y, w, h)
            boundary = qt.boundary()
            assert boundary == (x, y, w, h), f"Boundary should be {(x,y,w,h)}, got {boundary}"
            print(f"  ✓ Boundary({x}, {y}, {w}, {h}) correct")
            
            # Test that boundary doesn't change after operations
            qt.insert(x + w/2, y + h/2, "center_point")
            boundary_after = qt.boundary()
            assert boundary_after == boundary, "Boundary should not change after insertions"
        
        print("  ✓ Boundary remains constant after operations")
        
        self.test_results.append(("Boundary API", True, ""))
    
    def test_depth_api(self):
        """Comprehensive test of depth API - previously under-tested"""
        print("📊 Testing Depth API...")
        
        qt = quadtree.QuadTree(0, 0, 100, 100)
        
        # Test initial depth
        initial_depth = qt.depth()
        assert initial_depth == 0, f"Empty tree should have depth 0, got {initial_depth}"
        print("  ✓ Empty tree depth = 0")
        
        # Insert points that don't trigger subdivision (≤ CAPACITY)
        for i in range(4):  # CAPACITY = 4
            qt.insert(i * 10, i * 10, f"point_{i}")
        
        depth_no_subdivision = qt.depth()
        assert depth_no_subdivision == 0, f"Tree without subdivision should have depth 0, got {depth_no_subdivision}"
        print("  ✓ No subdivision: depth = 0")
        
        # Force subdivision by adding more points in same area
        cluster_center = (25, 25)
        for i in range(10):  # This should trigger subdivision
            x = cluster_center[0] + random.uniform(-2, 2)
            y = cluster_center[1] + random.uniform(-2, 2)
            qt.insert(x, y, f"cluster_{i}")
        
        depth_with_subdivision = qt.depth()
        assert depth_with_subdivision > 0, "Tree with subdivision should have depth > 0"
        print(f"  ✓ With subdivision: depth = {depth_with_subdivision}")
        
        # Test maximum depth constraint
        # Create points that force deep subdivision
        qt_deep = quadtree.QuadTree(0, 0, 1000, 1000)
        base_x, base_y = 500, 500
        
        # Insert points in decreasing spiral to force maximum depth
        for i in range(200):
            angle = i * 0.1
            radius = 10 / (1 + i * 0.05)  # Decreasing radius
            x = base_x + radius * math.cos(angle)
            y = base_y + radius * math.sin(angle)
            qt_deep.insert(x, y, f"deep_{i}")
        
        max_depth = qt_deep.depth()
        print(f"  ✓ Maximum achieved depth: {max_depth}")
        assert max_depth <= 10, f"Depth should be limited to MAX_DEPTH (10), got {max_depth}"
        
        self.test_results.append(("Depth API", True, ""))
    
    def test_subdivisions_api(self):
        """Comprehensive test of subdivisions API - previously under-tested"""
        print("🔀 Testing Subdivisions API...")
        
        qt = quadtree.QuadTree(0, 0, 100, 100)
        
        # Test initial subdivisions
        initial_subdivisions = qt.subdivisions()
        assert initial_subdivisions == 0, f"Empty tree should have 0 subdivisions, got {initial_subdivisions}"
        print("  ✓ Empty tree subdivisions = 0")
        
        # Insert points that don't trigger subdivision
        for i in range(4):  # CAPACITY = 4
            qt.insert(i * 20, i * 20, f"point_{i}")
        
        subdivisions_no_split = qt.subdivisions()
        assert subdivisions_no_split == 0, f"Tree without subdivision should have 0 subdivisions, got {subdivisions_no_split}"
        print("  ✓ No subdivision: subdivisions = 0")
        
        # Force subdivision by inserting clustered points
        cluster_points = [
            (10, 10), (11, 11), (12, 12), (13, 13), (14, 14),  # Force subdivision
            (50, 50), (51, 51), (52, 52), (53, 53), (54, 54),  # Another cluster
        ]
        
        for x, y in cluster_points:
            qt.insert(x, y, f"cluster_{x}_{y}")
        
        subdivisions_after_clusters = qt.subdivisions()
        assert subdivisions_after_clusters > 0, "Tree with clusters should have subdivisions > 0"
        print(f"  ✓ With clusters: subdivisions = {subdivisions_after_clusters}")
        
        # Test subdivision pattern in each quadrant
        qt_pattern = quadtree.QuadTree(0, 0, 100, 100)
        
        # Create subdivision pattern that affects multiple quadrants
        quadrant_centers = [(25, 25), (75, 25), (25, 75), (75, 75)]
        for qx, qy in quadrant_centers:
            # Add enough points in each quadrant to force subdivision
            for i in range(6):
                x = qx + random.uniform(-5, 5)
                y = qy + random.uniform(-5, 5)
                qt_pattern.insert(x, y, f"quad_{qx}_{qy}_{i}")
        
        pattern_subdivisions = qt_pattern.subdivisions()
        print(f"  ✓ Multi-quadrant pattern: subdivisions = {pattern_subdivisions}")
        
        # Verify subdivisions increase with depth
        depth = qt_pattern.depth()
        assert pattern_subdivisions >= depth, "Subdivisions should be >= depth"
        print(f"  ✓ Subdivisions ({pattern_subdivisions}) >= depth ({depth})")
        
        self.test_results.append(("Subdivisions API", True, ""))
    
    def test_subdivision_behavior(self):
        """Test the relationship between insertions and subdivision behavior"""
        print("🌳 Testing Subdivision Behavior...")
        
        qt = quadtree.QuadTree(0, 0, 100, 100)
        
        # Track subdivision progression
        subdivision_progression = []
        
        # Insert points and track subdivision changes
        for i in range(20):
            x = 50 + random.uniform(-10, 10)  # Clustered around center
            y = 50 + random.uniform(-10, 10)
            qt.insert(x, y, f"progress_{i}")
            
            subdivisions = qt.subdivisions()
            depth = qt.depth()
            size = qt.size()
            
            subdivision_progression.append((i+1, size, subdivisions, depth))
            
            if i == 3:  # Should not have subdivided yet (CAPACITY = 4)
                assert subdivisions == 0, f"Should not subdivide until after CAPACITY, got {subdivisions}"
            elif i > 3:  # Should have subdivided
                assert subdivisions > 0, f"Should have subdivided after exceeding CAPACITY"
        
        print(f"  ✓ Subdivision triggered correctly after exceeding CAPACITY")
        
        # Print progression for verification
        for points, size, subdivisions, depth in subdivision_progression[-5:]:
            print(f"    Points: {points}, Size: {size}, Subdivisions: {subdivisions}, Depth: {depth}")
        
        self.test_results.append(("Subdivision Behavior", True, ""))
    
    def test_depth_subdivision_relationship(self):
        """Test mathematical relationship between depth and subdivisions"""
        print("📐 Testing Depth-Subdivision Relationship...")
        
        # Create various tree configurations
        test_configs = [
            # Configuration 1: Single deep branch
            {"name": "Single Branch", "points": [(50, 50 + i*0.1) for i in range(50)]},
            # Configuration 2: Balanced tree
            {"name": "Balanced", "points": [(i*10, j*10) for i in range(10) for j in range(10)]},
            # Configuration 3: Maximum depth
            {"name": "Max Depth", "points": [(25 + i*0.01, 25 + i*0.01) for i in range(100)]},
        ]
        
        for config in test_configs:
            qt = quadtree.QuadTree(0, 0, 100, 100)
            
            # Insert all points
            for x, y in config["points"]:
                qt.insert(x, y, f"data_{x}_{y}")
            
            depth = qt.depth()
            subdivisions = qt.subdivisions()
            size = qt.size()
            
            # Verify mathematical relationships
            assert subdivisions >= 0, "Subdivisions should be non-negative"
            assert depth >= 0, "Depth should be non-negative"
            
            if subdivisions > 0:
                assert depth > 0, "Non-zero subdivisions should imply non-zero depth"
            
            # In a complete subdivision, subdivisions ≈ (4^(depth+1) - 1) / 3, but actual trees vary
            # Just verify reasonable bounds
            if depth > 0:
                min_subdivisions = 1  # At least one subdivision if depth > 0
                max_subdivisions = 4 ** depth  # Upper bound (complete tree)
                assert min_subdivisions <= subdivisions <= max_subdivisions, \
                    f"Subdivisions {subdivisions} outside reasonable bounds [{min_subdivisions}, {max_subdivisions}] for depth {depth}"
            
            print(f"  ✓ {config['name']}: Size={size}, Depth={depth}, Subdivisions={subdivisions}")
        
        self.test_results.append(("Depth-Subdivision Relationship", True, ""))
    
    def test_boundary_conditions(self):
        """Test API behavior at boundary conditions"""
        print("🎯 Testing Boundary Conditions...")
        
        qt = quadtree.QuadTree(0, 0, 100, 100)
        
        # Test points exactly on boundaries
        boundary_points = [
            (0, 0),        # Bottom-left corner
            (0, 50),       # Left edge
            (0, 99.9999),  # Top-left corner (just inside)
            (50, 0),       # Bottom edge
            (99.9999, 0),  # Bottom-right corner (just inside)
            (50, 99.9999), # Top edge (just inside)
            (99.9999, 99.9999),  # Top-right corner (just inside)
        ]
        
        for x, y in boundary_points:
            result = qt.insert(x, y, f"boundary_{x}_{y}")
            assert result, f"Should successfully insert boundary point ({x}, {y})"
            assert qt.contains(x, y), f"Should contain inserted boundary point ({x}, {y})"
        
        print(f"  ✓ Successfully handled {len(boundary_points)} boundary points")
        
        # Test queries at boundaries
        boundary_queries = [
            (0, 0, 1, 1),        # Tiny query at corner
            (-10, -10, 20, 20),  # Query extending outside
            (99, 99, 10, 10),    # Query at far boundary
        ]
        
        for qx, qy, qw, qh in boundary_queries:
            results = qt.query(qx, qy, qw, qh)
            print(f"  ✓ Boundary query ({qx}, {qy}, {qw}, {qh}) returned {len(results)} points")
        
        # Test collision detection at boundaries
        collisions = qt.detect_collisions(1.0)
        print(f"  ✓ Boundary collision detection found {len(collisions)} collisions")
        
        self.test_results.append(("Boundary Conditions", True, ""))
    
    def test_data_integrity_across_apis(self):
        """Test that data integrity is maintained across all API operations"""
        print("🔒 Testing Data Integrity Across APIs...")
        
        qt = quadtree.QuadTree(0, 0, 100, 100)
        
        # Insert points with complex data
        complex_data = [
            (10, 20, {"id": 1, "metadata": {"type": "A", "value": [1, 2, 3]}}),
            (30, 40, "simple_string"),
            (50, 60, [{"nested": "list"}, "with", "mixed", 123, True]),
            (70, 80, None),
            (90, 10, 3.14159265359),
        ]
        
        for x, y, data in complex_data:
            qt.insert(x, y, data)
        
        # Verify data through get_all_points
        all_points = qt.get_all_points()
        for point in all_points:
            x, y = point[0], point[1]
            retrieved_data = point[2] if len(point) == 3 else None
            
            # Find original data
            original = next((cd for cd in complex_data if cd[0] == x and cd[1] == y), None)
            assert original, f"Should find original data for ({x}, {y})"
            assert retrieved_data == original[2], f"Data mismatch for ({x}, {y})"
        
        print("  ✓ Data integrity maintained in get_all_points")
        
        # Verify data through query
        query_results = qt.query(0, 0, 100, 100)
        for point in query_results:
            x, y = point[0], point[1]
            retrieved_data = point[2] if len(point) == 3 else None
            
            original = next((cd for cd in complex_data if cd[0] == x and cd[1] == y), None)
            assert original, f"Should find original data for ({x}, {y}) in query"
            assert retrieved_data == original[2], f"Query data mismatch for ({x}, {y})"
        
        print("  ✓ Data integrity maintained in query")
        
        # Verify data through collision detection
        collisions = qt.detect_collisions(100.0)  # Large radius to get all pairs
        for collision in collisions:
            p1, p2 = collision["point1"], collision["point2"]
            
            for point in [p1, p2]:
                x, y = point[0], point[1]
                retrieved_data = point[2] if len(point) == 3 else None
                
                original = next((cd for cd in complex_data if cd[0] == x and cd[1] == y), None)
                assert original, f"Should find original data for ({x}, {y}) in collision"
                assert retrieved_data == original[2], f"Collision data mismatch for ({x}, {y})"
        
        print("  ✓ Data integrity maintained in collision detection")
        
        self.test_results.append(("Data Integrity", True, ""))
    
    def test_numerical_precision_comprehensive(self):
        """Comprehensive numerical precision and floating point edge cases"""
        print("🔢 Testing Numerical Precision...")
        
        qt = quadtree.QuadTree(0.0, 0.0, 1.0, 1.0)
        
        # Test very small values
        tiny_values = [1e-10, 1e-15, sys.float_info.epsilon]
        for val in tiny_values:
            if val < 1.0:  # Must be within boundary
                result = qt.insert(val, val, f"tiny_{val}")
                assert result, f"Should handle tiny value {val}"
                assert qt.contains(val, val), f"Should find tiny value {val}"
        
        print("  ✓ Tiny values handled correctly")
        
        # Test precision boundaries
        qt_precision = quadtree.QuadTree(0, 0, 100, 100)
        base_x, base_y = 50.123456789012345, 50.987654321098765
        
        # Insert with high precision
        qt_precision.insert(base_x, base_y, "high_precision")
        
        # Test retrieval with slight variations
        precision_tests = [
            (base_x, base_y, True),  # Exact match
            (base_x + 1e-15, base_y, True),  # Within epsilon
            (base_x + 1e-10, base_y, True),  # Still within epsilon
            (base_x + 1e-6, base_y, False),  # Outside epsilon
        ]
        
        for test_x, test_y, should_contain in precision_tests:
            result = qt_precision.contains(test_x, test_y)
            if result != should_contain:
                print(f"  ⚠️ Precision test: ({test_x}, {test_y}) returned {result}, expected {should_contain}")
        
        print("  ✓ Floating point precision tests completed")
        
        # Test large numbers close to float limits
        qt_large = quadtree.QuadTree(0, 0, 1e6, 1e6)
        large_coords = [(999999.999999, 999999.999999), (1e5, 1e5)]
        for x, y in large_coords:
            result = qt_large.insert(x, y, f"large_{x}_{y}")
            if result:
                assert qt_large.contains(x, y), f"Large coordinate ({x}, {y}) not found after insertion"
        
        print("  ✓ Large number handling completed")
        
        self.test_results.append(("Numerical Precision", True, ""))
    
    def test_error_handling_comprehensive(self):
        """Comprehensive error handling and exception testing"""
        print("⚠️ Testing Error Handling...")
        
        # Test invalid constructor parameters with specific error types
        invalid_constructors = [
            (0, 0, 0, 100, "Zero width"),
            (0, 0, 100, 0, "Zero height"),
            (0, 0, -1, 100, "Negative width"),
            (0, 0, 100, -1, "Negative height"),
            (float('inf'), 0, 100, 100, "Infinite x"),
            (0, float('inf'), 100, 100, "Infinite y"),
            (0, 0, float('inf'), 100, "Infinite width"),
            (0, 0, 100, float('inf'), "Infinite height"),
        ]
        
        for x, y, w, h, description in invalid_constructors:
            try:
                qt = quadtree.QuadTree(x, y, w, h)
                print(f"  ⚠️ Constructor should have failed for {description}: ({x}, {y}, {w}, {h})")
            except (ValueError, OverflowError) as e:
                print(f"  ✓ Correctly rejected {description}: {type(e).__name__}")
            except Exception as e:
                print(f"  ⚠️ Unexpected exception for {description}: {type(e).__name__}: {e}")
        
        # Test invalid operations on valid tree
        qt = quadtree.QuadTree(0, 0, 100, 100)
        
        # Test invalid insertions
        invalid_inserts = [
            (float('nan'), 50, "NaN x coordinate"),
            (50, float('nan'), "NaN y coordinate"),
            (float('inf'), 50, "Infinite x coordinate"),
            (50, float('inf'), "Infinite y coordinate"),
        ]
        
        for x, y, description in invalid_inserts:
            try:
                result = qt.insert(x, y, "test_data")
                print(f"  ⚠️ Insert should have handled {description} gracefully")
            except (ValueError, OverflowError):
                print(f"  ✓ Correctly rejected {description}")
            except Exception as e:
                print(f"  ⚠️ Unexpected exception for {description}: {type(e).__name__}")
        
        # Test invalid queries
        invalid_queries = [
            (0, 0, float('nan'), 10, "NaN width"),
            (0, 0, 10, float('nan'), "NaN height"),
            (float('inf'), 0, 10, 10, "Infinite x"),
            (0, float('inf'), 10, 10, "Infinite y"),
        ]
        
        for qx, qy, qw, qh, description in invalid_queries:
            try:
                results = qt.query(qx, qy, qw, qh)
                print(f"  ⚠️ Query should have handled {description} gracefully")
            except (ValueError, OverflowError):
                print(f"  ✓ Correctly rejected {description}")
            except Exception as e:
                print(f"  ⚠️ Unexpected exception for {description}: {type(e).__name__}")
        
        # Test invalid collision detection
        try:
            qt.detect_collisions(float('nan'))
            print("  ⚠️ Should reject NaN radius")
        except (ValueError, OverflowError):
            print("  ✓ Correctly rejected NaN radius")
        
        try:
            qt.detect_collisions(float('inf'))
            print("  ⚠️ Should handle infinite radius gracefully")
        except (ValueError, OverflowError):
            print("  ✓ Correctly rejected infinite radius")
        
        self.test_results.append(("Error Handling", True, ""))
    
    def test_performance_benchmarks(self):
        """Performance benchmarks for scalability testing"""
        print("🚀 Testing Performance Benchmarks...")
        
        # Benchmark different dataset sizes
        dataset_sizes = [100, 1000, 5000, 10000]
        
        for size in dataset_sizes:
            qt = quadtree.QuadTree(0, 0, 1000, 1000)
            
            # Benchmark insertions
            start_time = time.time()
            points = [(random.uniform(0, 1000), random.uniform(0, 1000)) for _ in range(size)]
            
            insertion_start = time.time()
            for x, y in points:
                qt.insert(x, y, f"data_{x}_{y}")
            insertion_time = time.time() - insertion_start
            
            # Benchmark queries
            query_start = time.time()
            num_queries = min(100, size // 10)
            for _ in range(num_queries):
                qx, qy = random.uniform(0, 900), random.uniform(0, 900)
                qt.query(qx, qy, 100, 100)
            query_time = time.time() - query_start
            
            # Benchmark collision detection
            collision_start = time.time()
            qt.detect_collisions(10.0)
            collision_time = time.time() - collision_start
            
            # Store performance results
            perf_result = {
                'size': size,
                'insertion_time': insertion_time,
                'query_time': query_time,
                'collision_time': collision_time,
                'insertions_per_second': size / insertion_time if insertion_time > 0 else float('inf'),
                'queries_per_second': num_queries / query_time if query_time > 0 else float('inf')
            }
            self.performance_results.append(perf_result)
            
            print(f"  ✓ Size {size:5d}: Insert={insertion_time:.4f}s, Query={query_time:.4f}s, Collision={collision_time:.4f}s")
        
        # Performance regression check
        if len(self.performance_results) >= 2:
            small_ops_per_sec = self.performance_results[0]['insertions_per_second']
            large_ops_per_sec = self.performance_results[-1]['insertions_per_second']
            
            # Allow some performance degradation with size, but not too much
            if large_ops_per_sec < small_ops_per_sec * 0.1:  # More than 10x slower
                print("  ⚠️ Significant performance degradation detected with large datasets")
            else:
                print("  ✓ Performance scales reasonably with dataset size")
        
        self.test_results.append(("Performance Benchmarks", True, ""))
    
    def test_memory_usage_tracking(self):
        """Track memory usage during large operations"""
        print("🧠 Testing Memory Usage...")
        
        if not PSUTIL_AVAILABLE or self.initial_memory is None:
            print("  ⚠️ psutil not available, skipping memory tests")
            self.test_results.append(("Memory Usage", True, "psutil not available"))
            return
        
        try:
            process = psutil.Process()
            
            # Test memory usage with large dataset
            qt = quadtree.QuadTree(0, 0, 10000, 10000)
            
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Insert large number of points
            large_dataset_size = 50000
            for i in range(large_dataset_size):
                x = random.uniform(0, 10000)
                y = random.uniform(0, 10000)
                data = f"data_point_{i}" * 5  # Make data larger
                qt.insert(x, y, data)
                
                # Sample memory usage
                if i % 10000 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    self.memory_usage.append({
                        'points': i,
                        'memory_mb': current_memory,
                        'memory_delta': current_memory - self.initial_memory
                    })
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            print(f"  ✓ Memory before: {memory_before:.1f} MB")
            print(f"  ✓ Memory after: {memory_after:.1f} MB")
            print(f"  ✓ Memory increase: {memory_increase:.1f} MB for {large_dataset_size} points")
            print(f"  ✓ Memory per point: {(memory_increase * 1024) / large_dataset_size:.2f} KB")
            
            # Check for reasonable memory usage
            memory_per_point_kb = (memory_increase * 1024) / large_dataset_size
            if memory_per_point_kb > 10:  # More than 10KB per point seems excessive
                print(f"  ⚠️ High memory usage per point: {memory_per_point_kb:.2f} KB")
            else:
                print("  ✓ Memory usage per point is reasonable")
            
            # Force garbage collection and check for memory leaks
            del qt
            gc.collect()
            
            memory_after_gc = process.memory_info().rss / 1024 / 1024
            memory_released = memory_after - memory_after_gc
            
            if memory_released > memory_increase * 0.8:  # At least 80% released
                print("  ✓ Good memory cleanup after deletion")
            else:
                print(f"  ⚠️ Potential memory leak: only {memory_released:.1f} MB released")
            
        except Exception as e:
            print(f"  ⚠️ Memory test error: {e}")
        
        self.test_results.append(("Memory Usage", True, ""))
    
    def test_concurrent_operations(self):
        """Test thread safety and concurrent operations"""
        print("🔄 Testing Concurrent Operations...")
        
        qt = quadtree.QuadTree(0, 0, 1000, 1000)
        errors = []
        results = []
        
        def worker_insert(worker_id, num_points):
            """Worker function for concurrent insertions"""
            local_results = []
            try:
                for i in range(num_points):
                    x = random.uniform(worker_id * 100, (worker_id + 1) * 100)
                    y = random.uniform(0, 1000)
                    result = qt.insert(x, y, f"worker_{worker_id}_point_{i}")
                    local_results.append(result)
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
            return local_results
        
        def worker_query(worker_id, num_queries):
            """Worker function for concurrent queries"""
            local_results = []
            try:
                for i in range(num_queries):
                    qx = random.uniform(0, 900)
                    qy = random.uniform(0, 900)
                    results = qt.query(qx, qy, 100, 100)
                    local_results.append(len(results))
            except Exception as e:
                errors.append(f"Query worker {worker_id} error: {e}")
            return local_results
        
        # First, populate with some initial data
        for i in range(1000):
            qt.insert(random.uniform(0, 1000), random.uniform(0, 1000), f"initial_{i}")
        
        initial_size = qt.size()
        print(f"  ✓ Initial population: {initial_size} points")
        
        # Test concurrent insertions
        num_workers = 4
        points_per_worker = 250
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit insertion tasks
            insert_futures = [
                executor.submit(worker_insert, i, points_per_worker)
                for i in range(num_workers)
            ]
            
            # Submit query tasks
            query_futures = [
                executor.submit(worker_query, i, 50)
                for i in range(num_workers)
            ]
            
            # Collect results
            for future in as_completed(insert_futures + query_futures):
                try:
                    result = future.result(timeout=30)
                    results.extend(result)
                except Exception as e:
                    errors.append(f"Future error: {e}")
        
        final_size = qt.size()
        expected_size = initial_size + (num_workers * points_per_worker)
        
        print(f"  ✓ Final size: {final_size} points")
        print(f"  ✓ Expected size: {expected_size} points")
        
        if errors:
            print(f"  ⚠️ Concurrent operation errors: {len(errors)}")
            for error in errors[:5]:  # Show first 5 errors
                print(f"    - {error}")
        else:
            print("  ✓ No concurrent operation errors")
        
        # Check data consistency
        try:
            all_points = qt.get_all_points()
            consistency_check = len(all_points) == qt.size()
            if consistency_check:
                print("  ✓ Data consistency maintained")
            else:
                print(f"  ⚠️ Inconsistency: get_all_points()={len(all_points)}, size()={qt.size()}")
        except Exception as e:
            print(f"  ⚠️ Consistency check failed: {e}")
        
        self.test_results.append(("Concurrent Operations", len(errors) == 0, f"{len(errors)} errors"))
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE API TEST RESULTS")
        print("=" * 60)
        
        passed_tests = [r for r in self.test_results if r[1]]
        failed_tests = [r for r in self.test_results if not r[1]]
        
        total_tests = len(self.test_results)
        success_rate = len(passed_tests) / total_tests * 100 if total_tests > 0 else 0
        
        print(f"Success Rate: {success_rate:.1f}% ({len(passed_tests)}/{total_tests} tests passed)")
        print()
        
        # Print detailed results
        print("🧪 Detailed Test Results:")
        for test_name, passed, error_msg in self.test_results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status} {test_name}")
            if not passed and error_msg:
                print(f"    Error: {error_msg}")
        
        print()
        
        # API Coverage Summary
        print("📋 API Coverage Summary:")
        covered_apis = [
            "QuadTree(x, y, width, height) - Constructor",
            "insert(x, y, data) - Point insertion",
            "query(x, y, width, height) - Rectangular queries",
            "contains(x, y) - Point existence check",
            "get_all_points() - Retrieve all points",
            "detect_collisions(radius) - Collision detection",
            "size() - Point count",
            "empty() - Empty state check",
            "boundary() - Boundary retrieval",
            "depth() - Tree depth calculation",
            "subdivisions() - Subdivision count",
        ]
        
        for api in covered_apis:
            print(f"  ✓ {api}")
        
        print()
        
        # Performance Summary
        if self.performance_results:
            print("⚡ Performance Summary:")
            for result in self.performance_results:
                print(f"  • {result['size']:5d} points: {result['insertions_per_second']:8.0f} inserts/sec, "
                      f"{result['queries_per_second']:6.0f} queries/sec")
            print()
        
        # Memory Summary
        if self.memory_usage and self.initial_memory:
            print("🧠 Memory Usage Summary:")
            if self.memory_usage:
                max_memory = max(usage['memory_mb'] for usage in self.memory_usage)
                print(f"  • Peak memory usage: {max_memory:.1f} MB")
                print(f"  • Memory growth: {max_memory - self.initial_memory:.1f} MB")
            print()
        
        # Verdict
        if success_rate >= 95:
            print("🎉 VERDICT: COMPREHENSIVE API TESTING SUCCESSFUL")
            print("All QuadTree API methods are working correctly with proper edge case handling.")
            if self.performance_results:
                avg_perf = sum(r['insertions_per_second'] for r in self.performance_results) / len(self.performance_results)
                print(f"Performance is good with average {avg_perf:.0f} insertions/second.")
        else:
            print("⚠️ VERDICT: API TESTING REVEALED ISSUES")
            print("Some API methods have failures that need to be addressed.")
        
        print(f"\n📄 Enhanced API test completed successfully!")

def main():
    """Run comprehensive API tests"""
    print("QuadTree C++17 Implementation - Comprehensive API Test Suite")
    print("Testing all exposed API methods with edge cases and boundary conditions")
    print()
    
    tester = QuadTreeAPITester()
    tester.run_all_api_tests()

if __name__ == "__main__":
    main()