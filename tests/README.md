# Tests

This folder contains lightweight tests for checking the basic structure of the artifact.

Run all tests with:

```bash
python -m pytest tests/


### `tests/test_artifact_structure.py`

```python
from pathlib import Path


def test_required_directories_exist():
    root = Path(__file__).resolve().parents[1]

    required_dirs = [
        "configs",
        "echo",
        "examples",
        "scripts",
        "tests",
    ]

    for dirname in required_dirs:
        assert (root / dirname).exists(), f"Missing directory: {dirname}"


def test_example_files_exist():
    root = Path(__file__).resolve().parents[1]

    assert (root / "examples" / "mini_queries.json").exists()
    assert (root / "examples" / "mini_sales.csv").exists()