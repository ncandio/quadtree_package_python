#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

try:
    import quadtree
    print("✓ Successfully imported quadtree module")
except ImportError as e:
    print(f"✗ Failed to import quadtree: {e}")
    print("Make sure to compile the module first with:")
    print("python setup.py build_ext --inplace")
    sys.exit(1)

def test_basic_functionality():
    """Test basic quadtree operations"""
    print("\n=== Testing Basic Functionality ===")
    
    # Create quadtree
    qt = quadtree.QuadTree(0, 0, 100, 100)
    print("✓ Created QuadTree(0, 0, 100, 100)")
    
    # Test empty
    assert qt.empty(), "QuadTree should be empty initially"
    print("✓ empty() works correctly")
    
    # Test size
    assert qt.size() == 0, "Size should be 0 initially"
    print("✓ size() works correctly")
    
    # Test boundary
    boundary = qt.boundary()
    assert boundary == (0.0, 0.0, 100.0, 100.0), f"Boundary should be (0, 0, 100, 100), got {boundary}"
    print("✓ boundary() works correctly")
    
    # Insert points
    points = [(10, 20), (30, 40), (50, 60), (70, 80), (90, 10)]
    for x, y in points:
        result = qt.insert(x, y)
        assert result, f"Insert failed for point ({x}, {y})"
    print(f"✓ Inserted {len(points)} points successfully")
    
    # Test size after insertion
    assert qt.size() == len(points), f"Size should be {len(points)}, got {qt.size()}"
    print("✓ size() correct after insertions")
    
    # Test not empty
    assert not qt.empty(), "QuadTree should not be empty after insertions"
    print("✓ empty() correct after insertions")
    
    # Test contains
    for x, y in points:
        assert qt.contains(x, y), f"Should contain point ({x}, {y})"
    print("✓ contains() works for inserted points")
    
    # Test doesn't contain non-existent point
    assert not qt.contains(999, 999), "Should not contain point (999, 999)"
    print("✓ contains() works for non-existent points")
    
    # Test get_all_points
    all_points = qt.get_all_points()
    assert len(all_points) == len(points), f"Should have {len(points)} points, got {len(all_points)}"
    print("✓ get_all_points() returns correct number of points")
    
    # Test query
    query_result = qt.query(0, 0, 50, 50)
    expected_in_region = [(x, y) for x, y in points if x < 50 and y < 50]
    assert len(query_result) == len(expected_in_region), f"Query should return {len(expected_in_region)} points, got {len(query_result)}"
    print("✓ query() works correctly")

def test_with_data():
    """Test quadtree with associated data"""
    print("\n=== Testing With Data ===")
    
    qt = quadtree.QuadTree(0, 0, 100, 100)
    
    # Insert points with data
    test_data = [
        (10, 20, "point1"),
        (30, 40, "point2"),
        (50, 60, {"id": 3, "name": "point3"}),
        (70, 80, [1, 2, 3])
    ]
    
    for x, y, data in test_data:
        result = qt.insert(x, y, data)
        assert result, f"Insert failed for point ({x}, {y}) with data {data}"
    print(f"✓ Inserted {len(test_data)} points with data")
    
    # Test retrieving points with data
    all_points = qt.get_all_points()
    assert len(all_points) == len(test_data), f"Should have {len(test_data)} points"
    
    # Check that data is preserved
    for point in all_points:
        if len(point) == 3:  # Point with data
            x, y, data = point
            # Find matching test data
            matching = [td for td in test_data if td[0] == x and td[1] == y]
            assert len(matching) == 1, f"Should find exactly one match for ({x}, {y})"
            assert matching[0][2] == data, f"Data mismatch for ({x}, {y})"
    print("✓ Data preservation works correctly")

def test_collision_detection():
    """Test collision detection"""
    print("\n=== Testing Collision Detection ===")
    
    qt = quadtree.QuadTree(0, 0, 100, 100)
    
    # Insert points that should collide
    points = [(10, 10), (12, 12), (50, 50), (90, 90)]
    for x, y in points:
        qt.insert(x, y, f"point_{x}_{y}")
    
    # Test collision detection with radius 5
    collisions = qt.detect_collisions(5.0)
    
    # Points (10,10) and (12,12) should collide (distance ≈ 2.83 < 5)
    expected_collisions = 1
    assert len(collisions) == expected_collisions, f"Should have {expected_collisions} collision, got {len(collisions)}"
    print("✓ Collision detection works correctly")

def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\n=== Testing Edge Cases ===")
    
    # Test invalid dimensions
    try:
        qt = quadtree.QuadTree(0, 0, -1, 10)
        assert False, "Should raise ValueError for negative width"
    except ValueError:
        print("✓ Correctly rejects negative width")
    
    try:
        qt = quadtree.QuadTree(0, 0, 10, 0)
        assert False, "Should raise ValueError for zero height"
    except ValueError:
        print("✓ Correctly rejects zero height")
    
    # Test out of bounds insertion
    qt = quadtree.QuadTree(0, 0, 10, 10)
    result = qt.insert(15, 5)  # Outside boundary
    assert not result, "Should return False for out of bounds insertion"
    print("✓ Correctly handles out of bounds insertion")
    
    # Test invalid query parameters
    try:
        qt.query(0, 0, -1, 5)
        assert False, "Should raise ValueError for negative width in query"
    except ValueError:
        print("✓ Correctly rejects negative width in query")
    
    try:
        qt.detect_collisions(-1.0)
        assert False, "Should raise ValueError for negative radius"
    except ValueError:
        print("✓ Correctly rejects negative radius")

def main():
    """Run all tests"""
    print("Testing Modern C++17 QuadTree Implementation")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_with_data()
        test_collision_detection()
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("The C++17 QuadTree implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()