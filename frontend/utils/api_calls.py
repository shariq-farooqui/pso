import json
from tenacity import retry, wait_random_exponential, stop_after_attempt

import requests

BASE_URL = "http://backend:8000"
headers = {"Content-Type": "application/json"}


def ping() -> dict:
    """Ping the backend server to check if it's running.

    Returns:
        dict: A dictionary containing the server status.
    """
    url = f"{BASE_URL}/ping"
    response = requests.get(url)
    return response.json()


def ping_database() -> dict:
    """Ping the database to check if it's running.

    Returns:
        dict: A dictionary containing the database status.
    """
    url = f"{BASE_URL}/ping_database"
    response = requests.get(url)
    return response.json()


def get_objective_functions() -> list[str]:
    """Get a list of available objective functions.

    Returns:
        list[str]: A list of available objective function names.
    """
    url = f"{BASE_URL}/available_objective_functions"
    response = requests.get(url)
    return response.json()


@retry(wait=wait_random_exponential(multiplier=1, max=10),
       stop=stop_after_attempt(5))
def evaluate_position(function_name: str,
                      position: list[float]) -> list[float]:
    """Evaluate an objective function at a given position.

    Args:
        function_name (str): The name of the objective function to evaluate.
        position (list[float]): The position to evaluate the objective function at.

    Returns:
        list[float]: The result of evaluating the objective function at the given position.
    """
    url = f"{BASE_URL}/evaluate_objective_function"
    payload = json.dumps({
        "function_name": function_name,
        "position": position,
    })
    response = requests.post(url, data=payload, headers=headers)
    if response.status_code != 200:
        raise ValueError("Error evaluating position")
    return response.json()["result"]


def run_pso(
    topology: str,
    problem_type: str,
    num_particles: int,
    max_iterations: int,
    bounds: list[list[float]],
    cognitive_weight: float,
    social_weight: float,
    inertia_weight: float,
    objective_function: str,
) -> dict:
    """Run a particle swarm optimization (PSO) algorithm.

    Args:
        topology (str): The topology of the PSO algorithm.
        problem_type (str): The type of problem being optimized.
        num_particles (int): The number of particles in the PSO algorithm.
        max_iterations (int): The maximum number of iterations to run the PSO algorithm for.
        bounds (list[list[float]]): The bounds of the problem being optimized.
        cognitive_weight (float): The cognitive weight parameter of the PSO algorithm.
        social_weight (float): The social weight parameter of the PSO algorithm.
        inertia_weight (float): The inertia weight parameter of the PSO algorithm.
        objective_function (str): The name of the objective function to optimize.

    Returns:
        dict: A dictionary containing the results of the PSO algorithm.
    """
    url = f"{BASE_URL}/run"
    payload = json.dumps({
        "topology": topology,
        "objective_function": objective_function,
        "problem_type": problem_type,
        "num_particles": num_particles,
        "max_iterations": max_iterations,
        "bounds": bounds,
        "cognitive_weight": cognitive_weight,
        "social_weight": social_weight,
        "inertia_weight": inertia_weight,
    })
    response = requests.post(url, headers=headers, data=payload)
    return response.json()


def get_run_details(run_id: str) -> list[dict]:
    """Get details about a PSO run.

    Args:
        run_id (str): The ID of the PSO run to get details for.

    Returns:
        list[dict]: A list of dictionaries containing details about the PSO run.
    """
    url = f"{BASE_URL}/run/{run_id}"
    response = requests.get(url)
    return response.json()


def animate_pso(run_id: str,
                animation_type: str,
                design_index: int,
                fixed_values: list[float] | None = None) -> str:
    """Animate a PSO run.

    Args:
        run_id (str): The ID of the PSO run to animate.
        animation_type (str): The type of animation to generate.
        design_index (int): The index of the design to animate.
        fixed_values (list[float], optional): A list of fixed values to use in the animation. Defaults to None.

    Returns:
        str: The filepath of the generated animation.
    """
    url = f"{BASE_URL}/animate"
    data = {
        "run_id": run_id,
        "design_index": design_index,
        "animation_type": animation_type,
    }
    if fixed_values:
        data["fixed_values"] = fixed_values
    payload = json.dumps(data)
    response = requests.post(url, headers=headers, data=payload)
    return response.json()["filepath"]
