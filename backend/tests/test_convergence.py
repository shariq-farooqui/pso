import numpy as np
import pytest

from pso.fitness import ConvergenceCalculator


def test_precision_sphere():
    sphere_calculator = ConvergenceCalculator("sphere", 3)
    best_score = 0.05
    best_position = np.array([0.01, 0.02, 0.03])

    assert sphere_calculator.precision_score(best_score) == 0.05
    expected_position_precision = np.linalg.norm(best_position - sphere_calculator.optimal_position)
    assert sphere_calculator.precision_position(best_position) == expected_position_precision


def test_invalid_function_name():
    with pytest.raises(ValueError):
        ConvergenceCalculator("invalid", 3)
