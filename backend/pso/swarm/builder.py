import numpy as np

from pso import fitness

from .particle import Particle
from .swarm import Swarm


class SwarmBuilder:
    """A builder class for creating Swarm objects with specified parameters.

    Attributes:
        swarm (Swarm): The Swarm object being built.
    """

    def __init__(self) -> None:
        """Initializes a new SwarmBuilder object with an empty Swarm."""
        self.swarm = Swarm()

    def with_problem_type(self, problem_type: str):
        """Sets the problem type for the Swarm.

        Args:
            problem_type (str): The problem type, either 'min' or 'max'.

        Returns:
            SwarmBuilder: The SwarmBuilder object with the problem type set.

        Raises:
            ValueError: If problem_type is not 'min' or 'max'.
        """
        if problem_type not in ["min", "max"]:
            raise ValueError("Problem type must be either 'min' or 'max'.")
        self.swarm.problem_type = problem_type
        self.swarm.global_best_score = np.inf if problem_type == "min" else -np.inf
        return self

    def with_max_iterations(self, max_iterations: int):
        """Sets the maximum number of iterations for the Swarm.

        Args:
            max_iterations (int): The maximum number of iterations.

        Returns:
            SwarmBuilder: The SwarmBuilder object with the maximum iterations set.

        Raises:
            ValueError: If max_iterations is less than 1.
        """
        if max_iterations < 1:
            raise ValueError("Maximum iterations must be at least 1.")
        self.swarm.max_iterations = max_iterations
        return self

    def with_objective_function(self, objective_function: str):
        """Sets the objective function for the Swarm.

        Args:
            objective_function (str): The name of the objective function.

        Returns:
            SwarmBuilder: The SwarmBuilder object with the objective function set.

        Raises:
            AttributeError: If the objective function does not exist.
        """
        try:
            self.swarm.objective_function = getattr(fitness, objective_function)
        except AttributeError:
            raise AttributeError(f"Objective function {objective_function} does not exist.")
        return self

    def with_num_particles(self, num_particles: int):
        """Sets the number of particles for the Swarm.

        Args:
            num_particles (int): The number of particles.

        Returns:
            SwarmBuilder: The SwarmBuilder object with the number of particles set.

        Raises:
            ValueError: If num_particles is less than 2.
        """
        if num_particles < 2:
            raise ValueError("Number of particles must be at least 2.")
        self.swarm.num_particles = num_particles
        self.swarm.particles = [Particle(self.swarm.problem_type) for _ in range(num_particles)]
        return self

    def with_bounds(self, bounds: list[tuple[float, float]]):
        """Sets the bounds for the Swarm.

        Args:
            bounds (list[tuple[float, float]]): A list of tuples representing the lower and upper bounds for each
                dimension.

        Returns:
            SwarmBuilder: The SwarmBuilder object with the bounds set.

        Raises:
            ValueError: If bounds is not a list of tuples of length 2, or if a lower bound is greater
                than its corresponding upper bound.
        """
        for bound in bounds:
            if len(bound) != 2:
                raise ValueError("Bounds must be a list of tuples of length 2.")
            if bound[0] > bound[1]:
                raise ValueError("Lower bound must be less than upper bound.")
        self.swarm.bounds = bounds
        self.swarm.lower_bounds, self.swarm.upper_bounds = zip(*bounds)
        self.swarm.dimensions = len(bounds)
        return self

    def with_cognitive_weight(self, cognitive_weight: float):
        """Sets the cognitive weight for the Swarm.

        Args:
            cognitive_weight (float): The cognitive weight.

        Returns:
            SwarmBuilder: The SwarmBuilder object with the cognitive weight set.

        Raises:
            ValueError: If cognitive_weight is negative.
        """
        if cognitive_weight < 0:
            raise ValueError("Cognitive weight must be non-negative.")
        self.swarm.cognitive_weight = cognitive_weight
        return self

    def with_social_weight(self, social_weight: float):
        """Sets the social weight for the Swarm.

        Args:
            social_weight (float): The social weight.

        Returns:
            SwarmBuilder: The SwarmBuilder object with the social weight set.

        Raises:
            ValueError: If social_weight is negative.
        """
        if social_weight < 0:
            raise ValueError("Social weight must be non-negative.")
        self.swarm.social_weight = social_weight
        return self

    def with_inertia_weight(self, inertia_weight: float):
        """Sets the inertia weight for the Swarm.

        Args:
            inertia_weight (float): The inertia weight.

        Returns:
            SwarmBuilder: The SwarmBuilder object with the inertia weight set.

        Raises:
            ValueError: If inertia_weight is negative.
        """
        if inertia_weight < 0:
            raise ValueError("Inertia weight must be non-negative.")
        self.swarm.inertia_weight = inertia_weight
        return self

    def build(self):
        """Builds the Swarm object with the specified parameters.

        Returns:
            Swarm: The built Swarm object.

        Raises:
            ValueError: If any required properties are missing.
        """
        for particle in self.swarm.particles:
            particle.problem_type = self.swarm.problem_type

        required_attributes = [
            "problem_type",
            "num_particles",
            "bounds",
            "cognitive_weight",
            "social_weight",
            "inertia_weight",
            "objective_function",
            "max_iterations",
        ]
        for attribute in required_attributes:
            if getattr(self.swarm, attribute) is None:
                raise ValueError(f"Swarm is missing required property: {attribute}")
        return self.swarm

    def build_from_dict(self, swarm_dict: dict):
        """Builds the Swarm object from a dictionary of parameters.

        Args:
            swarm_dict (dict): A dictionary of Swarm parameters.

        Returns:
            Swarm: The built Swarm object.
        """
        builder_methods = {
            "problem_type": self.with_problem_type,
            "num_particles": self.with_num_particles,
            "bounds": self.with_bounds,
            "cognitive_weight": self.with_cognitive_weight,
            "social_weight": self.with_social_weight,
            "inertia_weight": self.with_inertia_weight,
            "objective_function": self.with_objective_function,
            "max_iterations": self.with_max_iterations,
        }
        for key, value in swarm_dict.items():
            if key in builder_methods:
                builder_methods[key](value)
        return self.build()

    def demo_swarm(self):
        """Returns a demo Swarm object with default parameters.

        Returns:
            Swarm: The demo Swarm object.
        """
        return (self.with_problem_type("min").with_max_iterations(10).with_objective_function(
            "sphere").with_num_particles(10).with_bounds([
                (-5, 5),
                (-5, 5),
            ]).with_cognitive_weight(2.0).with_social_weight(2.0).with_inertia_weight(0.5).build())
