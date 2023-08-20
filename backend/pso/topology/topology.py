from abc import ABC, abstractmethod

from pso.swarm import Particle


class Topology(ABC):
    """Abstract base class for defining a topology for the PSO algorithm.

    Attributes:
        particles (list[Particle]): A list of particles in the swarm.
    """

    def __init__(self, particles: list[Particle]) -> None:
        """Initializes a new instance of the Topology class.

        Args:
            particles (list[Particle]): A list of particles in the swarm.
        """
        self.particles = particles

    @abstractmethod
    def assign_neighbours(self, current_particle_index: int) -> list[int]:
        """Abstract method for assigning neighbours to a particle.

        Args:
            current_particle_index (int): The index of the current particle.

        Returns:
            list[int]: A list of indices of neighbouring particles.
        """
        pass


class GlobalTopology(Topology):
    """Class for defining a global topology for the PSO algorithm.

    Attributes:
        particles (list[Particle]): A list of particles in the swarm.
    """

    def __init__(self, particles: list[Particle]) -> None:
        """Initializes a new instance of the GlobalTopology class.

        Args:
            particles (list[Particle]): A list of particles in the swarm.
        """
        super().__init__(particles)

    def assign_neighbours(self, current_particle_index: int) -> list[int]:
        """Assigns all particles in the swarm as neighbours to the current particle.

        Args:
            current_particle_index (int): The index of the current particle.

        Returns:
            list[int]: A list of indices of neighbouring particles.
        """
        return [particle for i, particle in enumerate(self.particles) if i != current_particle_index]


class RingTopology(Topology):
    """Class for defining a ring topology for the PSO algorithm.

    Attributes:
        particles (list[Particle]): A list of particles in the swarm.
    """

    def __init__(self, particles: list[Particle]) -> None:
        """Initializes a new instance of the RingTopology class.

        Args:
            particles (list[Particle]): A list of particles in the swarm.
        """
        super().__init__(particles)

    def assign_neighbours(self, current_particle_index: int) -> list[int]:
        """Assigns the two adjacent particles in the swarm as neighbours to the current particle.

        Args:
            current_particle_index (int): The index of the current particle.

        Returns:
            list[int]: A list of indices of neighbouring particles.
        """
        return [
            self.particles[current_particle_index - 1],
            self.particles[(current_particle_index + 1) % len(self.particles)],
        ]
