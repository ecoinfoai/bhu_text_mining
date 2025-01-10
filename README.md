# bhu_text_mining

## Setup
This project uses Poetry for dependency management and packaging.

### Installation
1. Install dependencies:
   ```bash
   poetry install
   ```

2. Run tests:
   ```bash
   poetry run pytest
   ```

3. Run pylint:
   ```bash
   poetry run pylint src/
   ```

## Project Structure
```
.
├── src/
│   └── bhu_text_mining/
│       ├── __init__.py
│       └── sample.py
├── tests/
│   ├── __init__.py
│   └── test_sample.py
├── conftest.py
├── .pylintrc
├── pyproject.toml
└── README.md
```
