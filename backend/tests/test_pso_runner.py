import asyncio

import pytest

from pso.pipelines import PSORunner, StandardGlobalPSOPipeline, StandardRingPSOPipeline
from pso.swarm import SwarmBuilder


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


def test_standard_global_pso_pipeline(swarm):
    pipeline = StandardGlobalPSOPipeline()

    # Remove MongoDBExporter processor
    pipeline.processors.pop()

    runner = PSORunner(pipeline, swarm)
    swarm = asyncio.run(runner.run())

    assert swarm.position_initialised is True
    assert swarm.velocity_initialised is True
    assert swarm.topology_initialised is True
    assert swarm.current_iteration == swarm.max_iterations
    assert swarm.global_best_position is not None
    assert swarm.global_best_score is not None


def test_standard_ring_pso_pipeline(swarm):
    pipeline = StandardRingPSOPipeline()

    # Remove MongoDBExporter processor
    pipeline.processors.pop()

    runner = PSORunner(pipeline, swarm)
    swarm = asyncio.run(runner.run())

    assert swarm.position_initialised is True
    assert swarm.velocity_initialised is True
    assert swarm.topology_initialised is True
    assert swarm.current_iteration == swarm.max_iterations
    assert swarm.global_best_position is not None
    assert swarm.global_best_score is not None
