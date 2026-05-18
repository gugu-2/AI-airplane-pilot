"""
R17 FIX: Central pytest configuration file.
Adds the project src/ directory to sys.path once for all tests,
so individual test files don't need fragile manual sys.path.insert() calls.
Run all tests from the project root: pytest tests/ -v
"""
import sys
import os

# Add src to path for all test files automatically
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
