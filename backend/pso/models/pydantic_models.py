import os

from pydantic import BaseModel


class SwarmConfig(BaseModel):
    """A Pydantic model representing the configuration for a Particle Swarm Optimization (PSO) algorithm.

    Attributes:
        topology (str): The topology of the swarm.
        problem_type (str): The type of problem being solved by the PSO algorithm.
        num_particles (int): The number of particles in the swarm.
        max_iterations (int): The maximum number of iterations for the algorithm.
        bounds (list[tuple[float, float]]): A list of tuples representing the lower and upper bounds for
            each dimension of the problem.
        cognitive_weight (float): The cognitive weight parameter for the PSO algorithm.
        social_weight (float): The social weight parameter for the PSO algorithm.
        inertia_weight (float): The inertia weight parameter for the PSO algorithm.
        objective_function (str): The name of the objective function being optimized by the PSO algorithm.
    """
    topology: str
    problem_type: str
    num_particles: int
    max_iterations: int
    bounds: list[tuple[float, float]]
    cognitive_weight: float
    social_weight: float
    inertia_weight: float
    objective_function: str


class AnimationRequest(BaseModel):
    """A Pydantic model representing a request for an animation of a PSO algorithm.

    Attributes:
        run_id (str): The ID of the PSO algorithm run.
        animation_type (str): The type of animation to generate.
        design_index (int): The index of the design to animate.
        fixed_values (list[float] | None): A list of fixed values for the design, or None if no values are fixed.
    """
    run_id: str
    animation_type: str
    design_index: int
    fixed_values: list[float] | None = None


class ObjectiveEvaluationRequest(BaseModel):
    """A Pydantic model representing a request to evaluate an objective function.

    Attributes:
        function_name (str): The name of the objective function to evaluate.
        position (list[float]): The position at which to evaluate the objective function.
    """
    function_name: str
    position: list[float]


class AppSettings(BaseModel):
    """A Pydantic model representing the application settings.

    Attributes:
        DATABASE_URL (str): The URL of the database.
        PSO_VERSION (str): The version of the PSO algorithm.
        BUILD_ENVIRONMENT (str): The environment in which the application is running.
    """

    DATABASE_URL: str = os.environ.get("DATABASE_URL")
    PSO_VERSION: str = os.environ.get("PSO_VERSION")
    BUILD_ENVIRONMENT: str = os.environ.get("BUILD_ENVIRONMENT")
