import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "incremental: mark tests as part of an incremental test class"
    )
    
def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords and call.excinfo is not None:
        parent = item.parent
        parent._previousfailed = item

def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        prevfailed = getattr(item.parent, "_previousfailed", None)
        if prevfailed is not None:
            pytest.skip(f"previous test failed ({prevfailed.name})")