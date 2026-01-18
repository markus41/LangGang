# {{ cookiecutter.project_name }}

{{ cookiecutter.description }}

## Setup

```bash
python{{ cookiecutter.python_version }} -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```python
from {{ cookiecutter.project_slug }} import main

main()
```

## Author

{{ cookiecutter.author }} ({{ cookiecutter.email }})

## Version

{{ cookiecutter.version }}
