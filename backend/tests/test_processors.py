import numpy as np
import pytest

from pso.processors import (
    InitialiseGlobalTopology,
    InitialisePosition,
    InitialiseRingTopology,
    InitialiseVelocity,
    ParticleEvaluator,
    SwarmEvaluator,
    UpdatePosition,
    UpdateVelocity,
)
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


def test_initialise_position(swarm):
    processor = InitialisePosition()
    assert swarm.position_initialised is False
    swarm = processor.process(swarm)
    assert swarm.position_initialised is True
    for particle in swarm.particles:
        assert np.all(particle.position >= swarm.lower_bounds)
        assert np.all(particle.position <= swarm.upper_bounds)


def test_initialise_velocity(swarm):
    processor = InitialiseVelocity()
    assert swarm.velocity_initialised is False
    swarm = processor.process(swarm)
    assert swarm.velocity_initialised is True
    for particle in swarm.particles:
        assert np.all(particle.velocity >= -1)
        assert np.all(particle.velocity <= 1)


def test_global_topology(swarm):
    init_position = InitialisePosition()
    init_velocity = InitialiseVelocity()
    processor = InitialiseGlobalTopology()

    swarm = init_position.process(swarm)
    swarm = init_velocity.process(swarm)
    assert swarm.topology_initialised is False
    swarm = processor.process(swarm)
    assert swarm.topology_initialised is True

    for particle in swarm.particles:
        assert len(particle.neighbours) == swarm.num_particles - 1
        assert particle not in particle.neighbours


def test_ring_topology(swarm):
    init_position = InitialisePosition()
    init_velocity = InitialiseVelocity()
    processor = InitialiseRingTopology()

    swarm = init_position.process(swarm)
    swarm = init_velocity.process(swarm)
    assert swarm.topology_initialised is False
    swarm = processor.process(swarm)
    assert swarm.topology_initialised is True

    for index, particle in enumerate(swarm.particles):
        assert len(particle.neighbours) == 2
        assert particle not in particle.neighbours
        left_particle = swarm.particles[index - 1]
        right_particle = swarm.particles[(index + 1) % swarm.num_particles]
        assert particle.neighbours == [left_particle, right_particle]


def test_particle_evaluator(swarm):
    init_position = InitialisePosition()
    init_velocity = InitialiseVelocity()
    global_topology = InitialiseGlobalTopology()
    processor = ParticleEvaluator()

    swarm = init_position.process(swarm)
    swarm = init_velocity.process(swarm)
    swarm = global_topology.process(swarm)

    assert swarm.particles[0].score is None
    swarm = processor.process(swarm)
    assert swarm.particles[0].score is not None


def test_swarm_evaluator(swarm):
    init_position = InitialisePosition()
    init_velocity = InitialiseVelocity()
    global_topology = InitialiseGlobalTopology()
    particle_evaluator = ParticleEvaluator()
    processor = SwarmEvaluator()

    swarm = init_position.process(swarm)
    swarm = init_velocity.process(swarm)
    swarm = global_topology.process(swarm)
    swarm = particle_evaluator.process(swarm)

    assert np.isinf(swarm.global_best_score)
    assert swarm.global_best_position is None
    assert swarm.score_precision == []
    assert swarm.position_precision == []
    assert swarm.converged is False
    assert swarm.convergence_iteration is None
    swarm.global_best_score = 0
    swarm.global_best_position = np.zeros(swarm.dimensions)
    swarm = processor.process(swarm)

    assert swarm.global_best_score is not None
    assert swarm.global_best_position is not None
    assert len(swarm.score_precision) == 1
    assert len(swarm.position_precision) == 1
    assert isinstance(swarm.score_precision[0], float)
    assert isinstance(swarm.position_precision[0], float)
    assert swarm.converged is True
    assert swarm.convergence_iteration == 0
    assert swarm.convergence_rate == 0


def test_velocity_update(swarm):
    init_position = InitialisePosition()
    init_velocity = InitialiseVelocity()
    global_topology = InitialiseGlobalTopology()
    particle_evaluator = ParticleEvaluator()
    swarm_evaluator = SwarmEvaluator()
    processor = UpdateVelocity()

    swarm = init_position.process(swarm)
    swarm = init_velocity.process(swarm)
    swarm = global_topology.process(swarm)
    swarm = particle_evaluator.process(swarm)
    swarm = swarm_evaluator.process(swarm)

    old_velocities = [particle.velocity.copy() for particle in swarm.particles]
    swarm = processor.process(swarm)

    for index, particle in enumerate(swarm.particles):
        assert np.all(particle.velocity >= swarm.lower_bounds)
        assert np.all(particle.velocity <= swarm.upper_bounds)
        assert np.any(particle.velocity != old_velocities[index])


def test_position_update(swarm):
    init_position = InitialisePosition()
    init_velocity = InitialiseVelocity()
    global_topology = InitialiseGlobalTopology()
    particle_evaluator = ParticleEvaluator()
    swarm_evaluator = SwarmEvaluator()
    update_velocity = UpdateVelocity()
    processor = UpdatePosition()

    swarm = init_position.process(swarm)
    swarm = init_velocity.process(swarm)
    swarm = global_topology.process(swarm)
    swarm = particle_evaluator.process(swarm)
    swarm = swarm_evaluator.process(swarm)
    swarm = update_velocity.process(swarm)

    old_positions = [particle.position.copy() for particle in swarm.particles]
    new_velocities = [particle.velocity.copy() for particle in swarm.particles]
    swarm = processor.process(swarm)

    for index, particle in enumerate(swarm.particles):
        expected_position = old_positions[index] + new_velocities[index]
        expected_position = np.clip(expected_position, swarm.lower_bounds, swarm.upper_bounds)
        assert np.all(particle.position == expected_position)
