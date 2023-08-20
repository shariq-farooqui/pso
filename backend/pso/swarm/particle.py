from typing import Callable

import numpy as np


class Particle:
    """A particle in the swarm.

    Attributes:
        problem_type (str): The type of optimization problem, either "min" or "max".
        position (numpy.ndarray): The current position of the particle in the search space.
        velocity (numpy.ndarray): The current velocity of the particle.
        best_position (numpy.ndarray): The best position found by the particle so far.
        best_score (float): The best score found by the particle so far.
        score (float): The score of the particle's current position.
        neighbours (list): The list of neighbouring particles.
    """

    def __init__(self, problem_type: str) -> None:
        """Initializes a new Particle object.

        Args:
            problem_type (str): The type of optimization problem, either "min" or "max".
        """
        self.problem_type = problem_type
        self.position = None
        self.velocity = None
        self.best_position = None
        self.best_score = np.inf if self.problem_type == "min" else -np.inf
        self.score = None
        self.neighbours = []

    def evaluate(self, objective_function: Callable) -> None:
        """Evaluates the particle's current position and updates its best position and score if necessary.

        Args:
            objective_function (Callable): The objective function to evaluate the particle's position with.
        """
        self.score = objective_function(self.position)
        if self.problem_type == "max" and self.score > self.best_score:
            self.best_score = self.score
            self.best_position = self.position.copy()
        elif self.problem_type == "min" and self.score < self.best_score:
            self.best_score = self.score
            self.best_position = self.position.copy()
