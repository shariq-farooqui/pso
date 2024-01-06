import pytest

from pso.swarm import Particle
from pso.topology import GlobalTopology, RingTopology


@pytest.fixture
def problem_type():
    return "min"


def test_neighbours_allocation(problem_type):
    particles = [Particle(problem_type) for _ in range(2)]

    topology = GlobalTopology(particles)
    particle_one = particles[0]
    particle_two = particles[1]
    assert particle_one.neighbours == []
    assert particle_two.neighbours == []

    particle_one.neighbours = topology.assign_neighbours(0)
    assert particle_one.neighbours == [particle_two]
    assert particle_two.neighbours == []

    particle_two.neighbours = topology.assign_neighbours(1)
    assert particle_one.neighbours == [particle_two]
    assert particle_two.neighbours == [particle_one]


def test_ring_topology_assignments(problem_type):
    particles = [Particle(problem_type) for _ in range(5)]
    topology = RingTopology(particles)

    particle_two = particles[2]
    particle_two.neighbours = topology.assign_neighbours(2)
    assert particle_two.neighbours == [particles[1], particles[3]]

    particle_zero = particles[0]
    particle_zero.neighbours = topology.assign_neighbours(0)
    assert particle_zero.neighbours == [particles[4], particles[1]]

    particle_four = particles[4]
    particle_four.neighbours = topology.assign_neighbours(4)
    assert particle_four.neighbours == [particles[3], particles[0]]

    particles = [Particle(problem_type) for _ in range(2)]
    topology = RingTopology(particles)
    particle_zero = particles[0]
    particle_zero.neighbours = topology.assign_neighbours(0)
    assert particle_zero.neighbours == [particles[1], particles[1]]


def test_global_topology_assignments(problem_type):
    particles = [Particle(problem_type) for _ in range(5)]
    topology = GlobalTopology(particles)

    particle_zero = particles[0]
    particle_zero.neighbours = topology.assign_neighbours(0)

    assert len(particle_zero.neighbours) == 4
    assert particle_zero.neighbours == [particles[1], particles[2], particles[3], particles[4]]
