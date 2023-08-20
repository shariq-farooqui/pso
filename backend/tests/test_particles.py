import numpy as np
import pytest

from pso.fitness import sphere
from pso.swarm import Particle
from pso.topology import GlobalTopology


@pytest.fixture
def objective_function():
    return sphere


def test_evaluate_min(objective_function):
    particle = Particle(problem_type="min")
    assert np.isinf(particle.best_score)

    particle.position = np.array([1, 2, 3])
    particle.evaluate(objective_function)

    assert particle.score == 14
    assert np.array_equal(particle.best_position, np.array([1, 2, 3]))
    assert particle.best_score == 14

    particle.position = np.array([1, 1, 1])
    particle.evaluate(objective_function)

    assert particle.score == 3
    assert np.array_equal(particle.best_position, np.array([1, 1, 1]))
    assert particle.best_score == 3


def test_evaluate_max(objective_function):
    particle = Particle(problem_type="max")
    assert np.isneginf(particle.best_score)

    particle.position = np.array([1, 1, 1, 1])
    particle.evaluate(objective_function)

    assert particle.score == 4
    assert np.array_equal(particle.best_position, np.array([1, 1, 1, 1]))
    assert particle.best_score == 4

    particle.position = np.array([1, 2, 3, 4])
    particle.evaluate(objective_function)

    assert particle.score == 30
    assert np.array_equal(particle.best_position, np.array([1, 2, 3, 4]))
    assert particle.best_score == 30


def test_particle_references():
    problem_type = "min"
    particles = [Particle(problem_type) for _ in range(2)]

    topology = GlobalTopology(particles)

    particle_one = particles[0]
    particle_two = particles[1]
    assert particle_one.score is None
    assert particle_two.score is None

    particle_one.neighbours = topology.assign_neighbours(0)
    assert particle_one.neighbours == [particle_two]
    assert particle_one.neighbours[0].score is None

    particle_two.score = 0
    assert particle_two.score == 0
    assert particle_one.neighbours[0].score == 0
