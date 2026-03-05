#!/usr/bin/env python3
"""
Test Runner for FLIM Pipeline

Runs all tests and generates reports.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def run_tests(
    test_file=None,
    verbose=False,
    coverage=False,
    markers=None,
    fast=False
):
    """
    Run pytest with specified options.
    
    Args:
        test_file: Specific test file to run (None = all)
        verbose: Verbose output
        coverage: Run with coverage report
        markers: Run tests with specific markers
        fast: Skip slow tests
    
    Returns:
        Exit code
    """
    # Build pytest command
    cmd = ["pytest"]
    
    # Add test file or directory
    if test_file:
        cmd.append(str(test_file))
    else:
        cmd.append("tests/")
    
    # Add options
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if coverage:
        cmd.extend(["--cov=flimkit", "--cov-report=term",
                    "--cov-report=term-missing:skip-covered"])
    
    if markers:
        cmd.extend(["-m", markers])
    
    if fast:
        cmd.extend(["-m", "not slow"])
    
    # Add useful options
    cmd.extend([
        "--tb=short",           # Short traceback format
        "--strict-markers",     # Strict marker checking
        "-ra",                  # Show summary of all test outcomes
    ])
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    # 👇 Ensure Python can find flimkit by setting PYTHONPATH
    env = os.environ.copy()
    project_root = str(Path(__file__).parent.parent)  # goes up one level from flim_tests/
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
    
    # Run tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent, env=env)
    
    return result.returncode


def run_unit_tests(verbose=False):
    """Run only unit tests."""
    print("\n=== Running Unit Tests ===\n")
    return run_tests(markers="unit", verbose=verbose)


def run_integration_tests(verbose=False):
    """Run only integration tests."""
    print("\n=== Running Integration Tests ===\n")
    return run_tests(test_file="tests/test_integration.py", verbose=verbose)


def run_all_tests(verbose=False, coverage=False):
    """Run all tests."""
    print("\n=== Running All Tests ===\n")
    return run_tests(verbose=verbose, coverage=coverage)


def check_dependencies():
    """Check that required dependencies are installed."""
    required = ['pytest', 'numpy']
    optional = ['pytest-cov', 'pytest-xdist']
    
    missing_required = []
    missing_optional = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_required.append(package)
    
    for package in optional:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        print("ERROR: Missing required packages:")
        for pkg in missing_required:
            print(f"  - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print("WARNING: Missing optional packages (some features disabled):")
        for pkg in missing_optional:
            print(f"  - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing_optional))
        print()
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run tests for FLIM pipeline"
    )
    
    parser.add_argument(
        "suite",
        nargs="?",
        choices=["all", "unit", "integration"],
        default="all",
        help="Which test suite to run"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "-f", "--fast",
        action="store_true",
        help="Skip slow tests"
    )
    
    parser.add_argument(
        "--file",
        type=Path,
        help="Run specific test file"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Run tests
    if args.file:
        exit_code = run_tests(
            test_file=args.file,
            verbose=args.verbose,
            coverage=args.coverage,
            fast=args.fast
        )
    elif args.suite == "unit":
        exit_code = run_unit_tests(verbose=args.verbose)
    elif args.suite == "integration":
        exit_code = run_integration_tests(verbose=args.verbose)
    else:  # all
        exit_code = run_all_tests(
            verbose=args.verbose,
            coverage=args.coverage
        )
    
    # Print summary
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)
    
    if args.coverage:
        log_path = Path(__file__).parent / "coverage_report.txt"
        print(f"\nCoverage summary written to terminal above.")
        print(f"Re-run with: pytest --cov=flimkit --cov-report=term-missing > {log_path}")
        print(f"to save a text report to {log_path}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())