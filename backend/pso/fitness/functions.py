import numpy as np


def quadratic_1d(position: np.ndarray) -> float:
    """Calculates the value of the quadratic function for a 1-dimensional position.

    Args:
        position (np.ndarray): The position for which the function value is to be calculated.

    Returns:
        float: The value of the quadratic function for the given position.

    Raises:
        TypeError: If the position is not an array.
    """
    if np.isscalar(position):
        raise TypeError("Position must be an array.")
    return position[0]**2


def quadratic_2d(position: np.ndarray) -> float:
    """Calculates the value of the quadratic function for a 2-dimensional position.

    Args:
        position (np.ndarray): The position for which the function value is to be calculated.

    Returns:
        float: The value of the quadratic function for the given position.

    Raises:
        TypeError: If the position is not an array.
    """
    if np.isscalar(position):
        raise TypeError("Position must be an array.")
    return position[0]**2 + position[1]**2


def sphere(position: np.ndarray) -> float:
    """Calculates the value of the sphere function for a given position.

    Args:
        position (np.ndarray): The position for which the function value is to be calculated.

    Returns:
        float: The value of the sphere function for the given position.

    Raises:
        TypeError: If the position is not an array.
    """
    if np.isscalar(position):
        raise TypeError("Position must be an array.")
    return np.sum(np.square(position))


def rastrigin(position: np.ndarray) -> float:
    """Calculates the value of the Rastrigin function for a given position in n dimensions.

    The Rastrigin function is defined as:
        f(x) = A * n + sum(x_i^2 - A * cos(2 * pi * x_i)) for i = 1 to n
    where A is a constant (usually 10) and n is the number of dimensions.

    Args:
        position (np.ndarray): The position for which the function value is to be calculated.

    Returns:
        float: The value of the Rastrigin function for the given position.

    Raises:
        TypeError: If the position is not an array.
    """
    if np.isscalar(position):
        raise TypeError("Position must be an array.")

    A = 10
    n = len(position)
    sum_term = np.sum(position**2 - A * np.cos(2 * np.pi * position))
    return A * n + sum_term
