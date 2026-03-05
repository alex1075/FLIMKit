# Quick Start - Testing the FLIM Pipeline

## 5-Minute Test Setup

### 1. Install Dependencies (30 seconds)

```bash
pip install pytest numpy
```

Optional (for coverage reports):
```bash
pip install pytest-cov
```


### 3. Run Tests (30 seconds)

```bash
cd your_project
python run_tests.py
```

That's it! You should see output like:

```
Running: pytest tests/ -q --tb=short --strict-markers -ra
============================================================
..........................................................  [100%]
60 passed in 2.34s
============================================================
✓ All tests passed!
============================================================
```

## What Gets Tested?

✓ **XML/XLIF parsing** - Tile positions, metadata extraction
✓ **PTU decoding** - Histogram extraction, time axis creation
✓ **Tile stitching** - Canvas computation, overlap handling
✓ **Integration** - Complete workflows, error handling
✓ **Memory efficiency** - Memmap usage, large datasets

## Running Specific Tests

### Test Individual Modules

```bash
# Test XLIF parsing only
pytest tests/test_xml_utils.py -v

# Test PTU decoding only
pytest tests/test_decode.py -v

# Test integration only
pytest tests/test_integration.py -v
```

### Test Specific Functions

```bash
# Test one specific function
pytest tests/test_xml_utils.py::test_parse_xlif_tile_positions -v

# Test one class
pytest tests/test_decode.py::TestDecode -v
```

### Run with Different Options

```bash
# Verbose output
python run_tests.py -v

# With coverage report
python run_tests.py --coverage

# Skip slow tests
python run_tests.py --fast

# Run specific file
python run_tests.py --file tests/test_xml_utils.py
```

## Understanding Test Output

### Passed Test
```
tests/test_xml_utils.py::test_parse_xlif_tile_positions PASSED  [10%]
```

### Failed Test
```
tests/test_decode.py::test_summed_decay FAILED  [50%]
```

Shows:
- File location
- Test name
- Status (PASSED/FAILED)
- Progress percentage

### Summary
```
60 passed in 2.34s
```

Shows total tests passed and time taken.

## Mock Data Examples

### Generate Test Project

```python
from mock_data import generate_test_project

# Create complete test project
project = generate_test_project(
    base_dir=Path("test_project/"),
    roi_name="R 2",
    n_tiles=4,
    layout="2x2"
)

print(f"XLIF: {project['xlif_path']}")
print(f"PTUs: {project['ptu_files']}")
```

### Generate Synthetic Decay

```python
from mock_data import generate_synthetic_decay

# Create decay with known lifetime
decay = generate_synthetic_decay(
    tau_ns=2.0,     # 2 ns lifetime
    bg=10.0,        # 10 counts background
    peak_counts=1000.0,
    noise=True
)

# Use for testing fitting algorithms
```

## Troubleshooting

### "No module named pytest"

```bash
pip install pytest
```

### "No module named code.utils.xml_utils"

Make sure you're running from project root:
```bash
cd your_project
python run_tests.py
```

Or set PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### Tests fail with import errors

Check your project structure matches:
```
your_project/
├── code/
│   ├── PTU/
│   │   ├── decode.py
│   │   └── stitch.py
│   └── utils/
│       └── xml_utils.py
└── tests/
    ├── test_xml_utils.py
    ├── test_decode.py
    └── test_integration.py
```

## Coverage Report

Generate coverage report:
```bash
python run_tests.py --coverage
```

View HTML report:
```bash
open htmlcov/index.html
```

Shows:
- Which lines are tested
- Which lines are NOT tested
- Overall coverage percentage

## Writing Your Own Tests

### Simple Test Template

```python
def test_my_function():
    """Test my function."""
    result = my_function(input_data)
    assert result == expected_output
```

### Test with Fixtures

```python
import pytest

@pytest.fixture
def test_data():
    """Create test data."""
    return create_data()

def test_with_fixture(test_data):
    """Test using fixture."""
    result = process(test_data)
    assert result is not None
```

### Test Error Handling

```python
def test_error():
    """Test error handling."""
    with pytest.raises(ValueError):
        my_function(invalid_input)
```

## Next Steps

1. Run tests: `python run_tests.py`
2. Check coverage: `python run_tests.py --coverage`
3. Read full docs: `README.md` <= To be written

## Common Commands Cheatsheet

```bash
# Run all tests
python run_tests.py

# Verbose output
python run_tests.py -v

# With coverage
python run_tests.py --coverage

# Specific suite
python run_tests.py unit
python run_tests.py integration

# Specific file
python run_tests.py --file tests/test_xml_utils.py

# Fast (skip slow tests)
python run_tests.py --fast

# Direct pytest (advanced)
pytest tests/ -v
pytest tests/test_xml_utils.py::test_parse_xlif_tile_positions -v
pytest -m unit  # Run only unit tests
pytest -m "not slow"  # Skip slow tests
pytest -k "xml"  # Run tests matching "xml"
```

## Success!

If you see:
```
✓ All tests passed!
```

The FLIM pipeline integration is working! 

Now you can confidently:
- Stitch tiles
- Fit FLIM data
- Process complete projects
- Know the code works as expected
