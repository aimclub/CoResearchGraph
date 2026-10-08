# Contributing to CoScientist

Thank you for your interest in contributing to CoScientist! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. We expect all contributors to:

- Be respectful and considerate in communication
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before submitting a bug report:

1. **Check the issue tracker** to avoid duplicates
2. **Verify the bug** by testing with the latest version
3. **Collect relevant information**:
   - Python version
   - Package versions
   - Error messages
   - Steps to reproduce

Submit a bug report using this template:

```markdown
**Description**
A clear description of the bug.

**Steps to Reproduce**
1. Go to '...'
2. Run '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., macOS, Ubuntu]
- Python version: [e.g., 3.12.5]
- Package versions: [list relevant packages]

**Additional Context**
Any other relevant information.
```

### Suggesting Enhancements

We welcome enhancement suggestions! Please:

1. **Search existing suggestions** before creating a new one
2. **Describe the enhancement** with clear use cases
3. **Explain why** this would benefit the project
4. **Provide examples** if possible

### Pull Requests

#### Pull Request Process

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CoScientist.git
   cd CoScientist
   ```

3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

4. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

5. **Make your changes**:
   - Follow the coding standards
   - Write tests for your changes
   - Ensure all tests pass

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

   Follow these commit message conventions:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Docs:` for documentation changes
   - `Refactor:` for code refactoring
   - `Test:` for adding/updating tests
   - `Chore:` for maintenance tasks

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Open a Pull Request** on GitHub

#### Pull Request Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Test update

## Checklist
- [ ] My code follows the project's coding standards
- [ ] I have performed a self-review of my code
- [ ] I have added tests that prove my fix/feature works
- [ ] All new and existing tests pass
- [ ] I have updated documentation if necessary
- [ ] My changes generate no new warnings

## Related Issues
Fixes #issue_number

## Additional Context
Any other information about the changes.
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Use type hints for function signatures
- Docstrings for all public functions and classes

### Code Formatting

We use Black for code formatting:

```bash
# Format code
black CoScientist/

# Check formatting
black --check CoScientist/
```

### Import Sorting

We use isort for import sorting:

```bash
# Sort imports
isort CoScientist/

# Check imports
isort --check-only CoScientist/
```

### Type Checking

We use mypy for type checking:

```bash
# Run type checker
mypy CoScientist/

# With specific configuration
mypy CoScientist/ --ignore-missing-imports
```

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Short description of the function.

    Longer description if needed, explaining the purpose
    and behavior of the function.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param2 is negative.

    Example:
        >>> result = function_name("test", 5)
        >>> print(result)
        True
    """
    pass
```

## Testing

### Writing Tests

All new features must include tests. We use pytest and pytest-asyncio:

```python
import pytest
import asyncio

# Synchronous test
def test_synchronous_function():
    result = my_function("input")
    assert result == "expected"

# Asynchronous test
@pytest.mark.asyncio
async def test_async_function():
    result = await my_async_function("input")
    assert result == "expected"

# Parametrized test
@pytest.mark.parametrize("input_value,expected", [
    ("value1", "result1"),
    ("value2", "result2"),
])
def test_parametrized(input_value, expected):
    assert my_function(input_value) == expected
```

### Test File Structure

Place tests in the `tests/` directory:

```
tests/
├── conftest.py              # Shared fixtures
├── agents/
│   ├── test_agents.py
│   └── test_prompts.py
├── chemical_utils/
│   ├── test_chemical_functions.py
│   ├── test_retrosynthesis.py
│   └── test_ocr_pipeline.py
├── config/
│   └── test_settings.py
├── logging/
│   └── test_logger.py
├── paper_parser/
│   ├── test_parse_and_split.py
│   ├── test_utils.py
│   └── test_s3_connection.py
├── storage/
│   └── test_models.py
└── tools/
    ├── test_fedotmas_tools.py
    ├── test_retrieval_tools.py
    └── test_web_tools.py
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/agents/test_agents.py -v

# Run with coverage
pytest tests/ --cov=CoScientist --cov-report=html

# Run async tests
pytest tests/ --asyncio-mode=auto

# Run with verbose output
pytest tests/ -v --tb=short

# Run tests matching a pattern
pytest tests/ -k "test_function_name"
```

### Test Fixtures

Define shared fixtures in `conftest.py`:

```python
import pytest
from CoScientist.config import Settings

@pytest.fixture
def test_settings():
    """Provide test settings."""
    return Settings(
        llm__main_model="test-model",
        # ... other test configurations
    )

@pytest.fixture
def mock_api_response():
    """Provide mock API response."""
    return {
        "status": "success",
        "data": {"result": "test_result"}
    }
```

## Documentation

### Updating Documentation

When you make changes that affect the API or behavior:

1. **Update docstrings** for affected functions
2. **Update README.md** if adding new features
3. **Update API.md** with new/changed endpoints
4. **Add examples** for new functionality


## Branching Strategy

We use a simplified Git Flow:

- `main` - Stable, production-ready code
- `develop` - Integration branch for features
- `feature/*` - New feature development
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

### Branch Naming

```
feature/add-new-agent
fix/agent-timeout-issue
docs/update-api-documentation
refactor/improve-error-handling
```

## Review Process

### What reviewers look for:

1. **Correctness** - Does the code do what it's supposed to?
2. **Style** - Does it follow our coding standards?
3. **Tests** - Are there adequate tests?
4. **Documentation** - Is the code properly documented?
5. **Performance** - Are there any performance concerns?
6. **Security** - Are there any security vulnerabilities?

### Review Timeline

- We aim to review PRs within 48 hours
- Complex changes may take longer
- Please be patient; we appreciate your contribution

## Development Workflow

### Daily Development

1. Pull latest changes:
   ```bash
   git checkout develop
   git pull origin develop
   ```

2. Create/update your feature branch:
   ```bash
   git checkout feature/your-feature
   git rebase develop  # Keep up to date
   ```

3. Make changes and commit:
   ```bash
   git add .
   git commit -m "Your commit message"
   ```

4. Run tests locally:
   ```bash
   pytest tests/ -v
   black --check .
   mypy .
   ```

5. Push and create PR:
   ```bash
   git push origin feature/your-feature
   ```

### Before Releasing

1. Ensure all tests pass
2. Update version in relevant files
3. Update CHANGELOG.md
4. Create release PR to main

## Questions?

If you have questions:

- Open an issue for bugs/enhancements
- Check existing documentation
- Ask in discussions

## Recognition

Contributors will be recognized in:

- The project's README.md contributors section
- GitHub's contributor graph
- Release notes for their contributions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Your contributions make CoScientist better for everyone. We appreciate your time and effort!
