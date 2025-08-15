#!/usr/bin/env python3

from setuptools import setup, Extension
import sys

# C++17 compilation flags
cpp_args = ['-std=c++17']
link_args = []

# Platform-specific optimizations
if sys.platform == 'darwin':  # macOS
    cpp_args += ['-stdlib=libc++', '-mmacosx-version-min=10.9']
    link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.9']
elif sys.platform.startswith('linux'):  # Linux
    cpp_args += ['-O3', '-march=native']

quadtree_module = Extension(
    'quadtree',
    sources=['quadtree.cpp'],
    language='c++',
    extra_compile_args=cpp_args,
    extra_link_args=link_args,
)

setup(
    name='quadtree',
    version='2.0.0',
    description='Modern C++17 QuadTree spatial data structure',
    long_description='''
A high-performance spatial quadtree data structure implemented in modern C++17
for efficient 2D point operations including:

- Fast point insertion and querying
- Spatial range queries
- Collision detection
- Support for arbitrary Python data attachment to points
- Memory-efficient design with smart pointers
- Exception-safe implementation
    '''.strip(),
    author='CPython Contributor',
    ext_modules=[quadtree_module],
    zip_safe=False,
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)