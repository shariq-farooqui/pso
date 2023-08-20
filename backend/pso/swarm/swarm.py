import uuid
from datetime import datetime
from typing import Callable

from .particle import Particle


class Swarm:
    """A class representing a swarm of particles for Particle Swarm Optimization (PSO).

    Attributes:
        run_id (str): A unique identifier for the current run of the swarm.
        current_iteration (int): The current iteration of the swarm.
        created_at (float): The timestamp when the swarm was created.
        finished_at (float): The timestamp when the swarm finished running.
        max_iterations (int): The maximum number of iterations for the swarm.
        problem_type (str): The type of problem being solved by the swarm.
        num_particles (int): The number of particles in the swarm.
        dimensions (int): The number of dimensions in the problem space.
        bounds (list[tuple[float, float]]): The bounds of the problem space.
        cognitive_weight (float): The cognitive weight for the swarm.
        social_weight (float): The social weight for the swarm.
        inertia_weight (float): The inertia weight for the swarm.
        objective_function (Callable): The objective function to be optimized by the swarm.
        particles (list[Particle]): The particles in the swarm.
        global_best_score (float): The global best score found by the swarm.
        global_best_position (list[float]): The global best position found by the swarm.
        position_initialised (bool): Whether the particle positions have been initialised.
        velocity_initialised (bool): Whether the particle velocities have been initialised.
        topology_initialised (bool): Whether the swarm topology has been initialised.
        score_precision (list[float]): The score based precision of the swarm at each iteration.
        position_precision (list[float]): The position based precision of the swarm at each iteration.
        converged (bool): Whether the swarm has converged.
        convergence_iteration (int): The iteration at which the swarm converged.
        convergence_rate (float): The convergence rate of the swarm.
    """

    def __init__(self,
                 problem_type: str | None = None,
                 max_iterations: int | None = None,
                 num_particles: int | None = None,
                 dimensions: int | None = None,
                 bounds: list[tuple[float, float]] | None = None,
                 cognitive_weight: float | None = None,
                 social_weight: float | None = None,
                 inertia_weight: float | None = None,
                 objective_function: Callable | None = None) -> None:
        """Initializes a Swarm object.

        Args:
            problem_type (str, optional): The type of problem being solved by the swarm. Defaults to None.
            max_iterations (int, optional): The maximum number of iterations for the swarm. Defaults to None.
            num_particles (int, optional): The number of particles in the swarm. Defaults to None.
            dimensions (int, optional): The number of dimensions in the problem space. Defaults to None.
            bounds (list[tuple[float, float]], optional): The bounds of the problem space. Defaults to None.
            cognitive_weight (float, optional): The cognitive weight for the swarm. Defaults to None.
            social_weight (float, optional): The social weight for the swarm. Defaults to None.
            inertia_weight (float, optional): The inertia weight for the swarm. Defaults to None.
            objective_function (Callable, optional): The objective function to be optimized by the swarm.
                Defaults to None.
        """
        self.run_id = str(uuid.uuid4())
        self.current_iteration = 0
        self.created_at = datetime.now().timestamp()
        self.finished_at = None
        self.max_iterations = max_iterations
        self.problem_type = problem_type
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.inertia_weight = inertia_weight
        self.objective_function = objective_function
        self.particles: list[Particle] = []
        self.global_best_score = None
        self.global_best_position = None
        self.position_initialised = False
        self.velocity_initialised = False
        self.topology_initialised = False
        self.score_precision = []
        self.position_precision = []
        self.converged = False
        self.convergence_iteration = None
        self.convergence_rate = None

    def evaluate(self) -> None:
        """Evaluates the objective function for each particle in the swarm."""
        for particle in self.particles:
            particle.evaluate(self.objective_function)
