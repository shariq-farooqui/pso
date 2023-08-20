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
    }

    def __init__(self, function_name: str, dimensions: int):
        """
        Parameters
        ----------
        function_name : str
            The name of the function being optimized.
        dimensions : int
            The number of dimensions of the function being optimized.

        Raises
        ------
        ValueError
            If the function name is not recognized.
        """
        if function_name not in self.OPTIMAL_VALUES:
            raise ValueError(f"Unknown function: {function_name}")

        self.function_name = function_name
        self.dimensions = dimensions
        self.optimal_position = self.OPTIMAL_VALUES[function_name]["optimal_position"](dimensions)
        self.optimal_score = self.OPTIMAL_VALUES[function_name]["optimal_score"]

    def precision_score(self, best_score: float) -> float:
        """
        Calculates the precision score of the particle swarm optimization algorithm.

        Parameters
        ----------
        best_score : float
            The best score achieved by the particle swarm optimization algorithm.

        Returns
        -------
        float
            The precision score of the particle swarm optimization algorithm.
        """
        return abs(best_score - self.optimal_score)

    def precision_position(self, best_position: np.ndarray) -> np.ndarray:
        """
        Calculates the precision position of the particle swarm optimization algorithm.

        Parameters
        ----------
        best_position : numpy.ndarray
            The best position achieved by the particle swarm optimization algorithm.

        Returns
        -------
        numpy.ndarray
            The precision position of the particle swarm optimization algorithm.
        """
        return np.linalg.norm(best_position - self.optimal_position)
