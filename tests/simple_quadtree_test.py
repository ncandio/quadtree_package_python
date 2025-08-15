#!/usr/bin/env python3

import quadtree

def simple_test():
    """A simple test demonstrating QuadTree usage"""
    print("Simple QuadTree Test")
    print("=" * 20)
    
    # Create a QuadTree covering area from (0,0) to (100,100)
    qt = quadtree.QuadTree(0, 0, 100, 100)
    print(f"Created QuadTree: {qt.boundary()}")
    
    # Insert some points
    points = [(25, 25), (75, 75), (25, 75), (75, 25)]
    for x, y in points:
        qt.insert(x, y, f"data_{x}_{y}")
    
    print(f"Inserted {len(points)} points")
    print(f"QuadTree size: {qt.size()}")
    print(f"Is empty: {qt.empty()}")
    
    # Query for points in left half (0, 0, 50, 100)
    left_points = qt.query(0, 0, 50, 100)
    print(f"Points in left half: {left_points}")
    
    # Query for points in top half (0, 0, 100, 50)
    top_points = qt.query(0, 0, 100, 50)
    print(f"Points in top half: {top_points}")
    
    # Check if specific points exist
    print(f"Contains (25, 25): {qt.contains(25, 25)}")
    print(f"Contains (50, 50): {qt.contains(50, 50)}")
    
    # Get all points
    all_points = qt.get_all_points()
    print(f"All points: {all_points}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    simple_test()