import uuid

import numpy as np
import pytest

from pso.fitness import sphere
from pso.swarm import Swarm, SwarmBuilder


@pytest.fixture
def swarm_config():
    return {
        "problem_type": "min",
        "max_iterations": 10,
        "num_particles": 10,
        "dimensions": 2,
        "bounds": [(-5, 5), (-5, 5)],
        "cognitive_weight": 0.5,
        "social_weight": 0.5,
        "inertia_weight": 0.5,
        "objective_function": "sphere",
    }


@pytest.fixture
def swarm(swarm_config):
    swarm_builder = SwarmBuilder()
    return swarm_builder.build_from_dict(swarm_config)


def test_swarm_defaults():
    swarm = Swarm()

    assert swarm.problem_type is None
    assert swarm.max_iterations is None
    assert swarm.num_particles is None
    assert swarm.dimensions is None
    assert swarm.bounds is None
    assert swarm.cognitive_weight is None
    assert swarm.social_weight is None
    assert swarm.inertia_weight is None
    assert swarm.objective_function is None
    assert isinstance(uuid.UUID(swarm.run_id), uuid.UUID)
    assert swarm.current_iteration == 0
    assert isinstance(swarm.created_at, float)
    assert swarm.finished_at is None
    assert swarm.particles == []
    assert swarm.global_best_score is None
    assert swarm.global_best_position is None
    assert swarm.position_initialised is False
    assert swarm.velocity_initialised is False
    assert swarm.topology_initialised is False


def test_swarm_builder(swarm):
    assert swarm.problem_type == "min"
    assert swarm.max_iterations == 10
    assert swarm.num_particles == 10
    assert swarm.dimensions == 2
    assert swarm.bounds == [(-5, 5), (-5, 5)]
    assert swarm.cognitive_weight == 0.5
    assert swarm.social_weight == 0.5
    assert swarm.inertia_weight == 0.5
    assert swarm.objective_function == sphere


def test_swarm_evaluate(swarm):
    for particle in swarm.particles:
        particle.position = np.random.uniform(swarm.lower_bounds, swarm.upper_bounds, swarm.dimensions)
    expected_scores = [sphere(particle.position) for particle in swarm.particles]
    swarm.evaluate()
    actual_scores = [particle.score for particle in swarm.particles]
    assert actual_scores == expected_scores
