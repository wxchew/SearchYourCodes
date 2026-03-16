"""
Shared pytest fixtures for SearchYourCodes tests.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def sample_cpp_code():
    return """
#include <iostream>

namespace sim {

class Motor {
public:
    Motor(double speed);
    void step(double dt);
    double getSpeed() const { return speed_; }
private:
    double speed_;
};

Motor::Motor(double speed) : speed_(speed) {}

void Motor::step(double dt) {
    // Update motor position
    position_ += speed_ * dt;
}

} // namespace sim
"""


@pytest.fixture
def sample_python_code():
    return '''
class Calculator:
    """A simple calculator class."""

    def __init__(self, initial_value=0):
        self.value = initial_value

    def add(self, x):
        """Add x to the current value."""
        self.value += x
        return self

    def multiply(self, x):
        """Multiply the current value by x."""
        self.value *= x
        return self


def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
'''


@pytest.fixture
def flask_app():
    """Create a test Flask app."""
    from app.main import app
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(flask_app):
    """Create a test client."""
    return flask_app.test_client()
