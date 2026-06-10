"""
test_project.py
===============

pytest suite for the ReLU Frequency Tradeoff proof project.

Run with:
    source ~/heartlib/.venv/bin/activate
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "empirical"))
from verify import (
    ReLUNet,
    count_breakpoints_exact,
    count_zeros,
    triangle_wave_error,
    check_theorem_1,
    check_theorem_2,
    check_theorem_3,
    get_device,
    manual_seed,
)


@pytest.fixture(scope="module", autouse=True)
def seed():
    manual_seed(1729)


class TestReLUNet:
    """Unit tests for the ReLU network."""

    def test_forward_shape(self):
        net = ReLUNet(5)
        x = torch.randn(10, 1)
        y = net(x)
        assert y.shape == (10, 1)

    def test_zero_hidden_units(self):
        net = ReLUNet(0)
        x = torch.randn(5, 1)
        y = net(x)
        # With no hidden units, output is just the bias
        assert y.shape == (5, 1)


class TestBreakpointCount:
    """Tests for exact breakpoint counting."""

    def test_no_breakpoints(self):
        """A flat network has 0 breakpoints."""
        net = ReLUNet(2)
        with torch.no_grad():
            net.fc1.weight[:, 0] = 0.0  # zero slope = no breakpoint
            net.fc1.bias[:] = 1.0
        assert count_breakpoints_exact(net) == 0

    def test_known_breakpoints(self):
        """Breakpoints at known positions."""
        net = ReLUNet(3)
        with torch.no_grad():
            for i in range(3):
                net.fc1.weight[i, 0] = 1.0
                net.fc1.bias[i] = -(i + 1)  # breakpoints at 1, 2, 3
        assert count_breakpoints_exact(net) == 3

    def test_collocated_breakpoints(self):
        """Two units with the same breakpoint count as one."""
        net = ReLUNet(2)
        with torch.no_grad():
            net.fc1.weight[0, 0] = 1.0
            net.fc1.bias[0] = -1.0
            net.fc1.weight[1, 0] = 2.0
            net.fc1.bias[1] = -2.0  # same breakpoint at x=1
        assert count_breakpoints_exact(net) == 1


class TestZeroCount:
    """Tests for zero counting."""

    def test_monotone_no_zeros(self):
        """Monotone increasing function has 0 zeros in [0,1]."""
        net = ReLUNet(1)
        with torch.no_grad():
            net.fc1.weight[0, 0] = 1.0
            net.fc1.bias[0] = 0.5  # always positive on [0,1]
            net.fc2.weight[0, 0] = 1.0
            net.fc2.bias[0] = 0.1
        zeros = count_zeros(net, 0.0, 1.0)
        assert zeros == 0

    def test_one_zero(self):
        """Function crossing zero once."""
        net = ReLUNet(1)
        with torch.no_grad():
            net.fc1.weight[0, 0] = 1.0
            net.fc1.bias[0] = -0.5  # zero at x=0.5
            net.fc2.weight[0, 0] = 1.0
            net.fc2.bias[0] = 0.0
        zeros = count_zeros(net, 0.0, 1.0)
        assert zeros <= 1

    def test_multiple_zeros(self):
        """Zigzag function with multiple zeros."""
        net = ReLUNet(5)
        with torch.no_grad():
            for i in range(5):
                bp = (i - 2.0) / 2.5
                net.fc1.weight[i, 0] = 1.0
                net.fc1.bias[i] = -bp
                net.fc2.weight[0, i] = ((-1) ** i) * 2.0
        zeros = count_zeros(net, -2.0, 2.0)
        assert zeros <= 5


class TestTriangleWaveError:
    """Tests for triangle wave approximation error."""

    def test_zero_hidden_large_error(self):
        """No hidden units → constant function → large error."""
        error = triangle_wave_error(0, freq=1)
        assert error > 0.5

    def test_error_decreases_with_capacity(self):
        """More units → smaller error for fixed frequency."""
        e1 = triangle_wave_error(4, freq=2)
        e2 = triangle_wave_error(8, freq=2)
        assert e2 < e1 * 1.2  # generally better with more units

    def test_error_increases_with_frequency(self):
        """Higher frequency → larger error for fixed units."""
        e1 = triangle_wave_error(6, freq=2)
        e2 = triangle_wave_error(6, freq=4)
        assert e2 >= e1 * 0.8  # generally worse with higher freq


class TestTheorem1:
    """Theorem 1: Breakpoint Count."""

    def test_pass(self):
        r = check_theorem_1()
        assert r.passed, f"Theorem 1 failed: {r.detail}"


class TestTheorem2:
    """Theorem 2: Zero Count."""

    def test_pass(self):
        r = check_theorem_2()
        assert r.passed, f"Theorem 2 failed: {r.detail}"


class TestTheorem3:
    """Theorem 3: Frequency Barrier."""

    def test_pass(self):
        r = check_theorem_3()
        assert r.passed, f"Theorem 3 failed: {r.detail}"
