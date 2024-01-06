import numpy as np


class ConvergenceCalculator:
    """A class used to calculate the convergence of a particle swarm optimization algorithm.

    Attributes
    ----------
    function_name : str
        The name of the function being optimized.
    dimensions : int
        The number of dimensions of the function being optimized.
    optimal_position : numpy.ndarray
        The optimal position of the function being optimized.
    optimal_score : float
        The optimal score of the function being optimized.

    Methods
    -------
    precision_score(best_score: float) -> float:
        Calculates the precision score of the particle swarm optimization algorithm.
    precision_position(best_position: numpy.ndarray) -> numpy.ndarray:
        Calculates the precision position of the particle swarm optimization algorithm.
    """

    OPTIMAL_VALUES = {
        "sphere": {
            "optimal_position": lambda dim: np.zeros(dim),
            "optimal_score": 0,
        },
        "quadratic_1d": {
            "optimal_position": lambda dim: np.zeros(1),
            "optimal_score": 0,
        },
        "quadratic_2d": {
            "optimal_position": lambda dim: np.zeros(2),
            "optimal_score": 0,
        },
        "rastrigin": {
            "optimal_position": lambda dim: np.zeros(dim),
            "optimal_score": 0,
        },
    }

    def __init__(self, function_name: str, dimensions: int, tolerance: float = 1e-6):
        """ Constructs a new ConvergenceCalculator instance.

        Args:
            function_name (str): The name of the function being optimized.
            dimensions (int): The number of dimensions of the function being optimized.
            tolerance (float, optional): The tolerance of the convergence check. Defaults to 1e-6.
        """
        if function_name not in self.OPTIMAL_VALUES:
            raise ValueError(f"Unknown function: {function_name}")

        self.function_name = function_name
        self.dimensions = dimensions
        self.tolerance = tolerance
        self.optimal_position = self.OPTIMAL_VALUES[function_name]["optimal_position"](dimensions)
        self.optimal_score = self.OPTIMAL_VALUES[function_name]["optimal_score"]

    def precision_score(self, best_score: float) -> float:
        """Calculates the precision score of the particle swarm optimization algorithm.

        Args:
            best_score (float): The best score achieved by the particle swarm optimization algorithm.

        Returns:
            float: The precision score of the particle swarm optimization algorithm.
        """

        return float(abs(best_score - self.optimal_score))

    def precision_position(self, best_position: np.ndarray) -> float:
        """Calculates the precision position of the particle swarm optimization algorithm.

        Args:
            best_position (numpy.ndarray): The best position achieved by the particle swarm optimization algorithm.

        Returns:
            float: The precision position of the particle swarm optimization algorithm,
                represented as the Euclidean distance from the optimal position.
        """
        return float(np.linalg.norm(best_position - self.optimal_position))

    def check_convergence(self, best_score: float):
        """Checks whether the particle swarm optimization algorithm has converged.

        Args:
            best_score (float): The best score achieved by the particle swarm optimization algorithm.

        Returns:
            bool: Whether the particle swarm optimization algorithm has converged.
        """
        return bool(abs(best_score - self.optimal_score) <= self.tolerance)

    def convergence_rate(self, convergence_iteration: int, max_iterations: int):
        """Calculates proportion of iterations used to converge.

        Args:
            convergence_iteration (int): The iteration at which the particle swarm optimization algorithm converged.
            max_iterations (int): The maximum number of iterations allowed for the particle swarm optimization
                algorithm.

        Returns:
            float: The proportion of iterations used to converge.
        """
        return (convergence_iteration / max_iterations) * 100
