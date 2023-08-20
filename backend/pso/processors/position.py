import numpy as np

from pso.swarm import Swarm

from .processor import Processor


class InitialisePosition(Processor[Swarm]):
    """Initialises the position of particles in the swarm.

    This processor is only used once, at the start of the optimisation process.

    """

    def process(self, swarm: Swarm) -> Swarm:
        """Initialises the position of particles in the swarm.

        Args:
            swarm (Swarm): The Swarm object to initialise positions for.

        Returns:
            Swarm: The Swarm object with initialised positions.
        """
        if not swarm.position_initialised:
            for particle in swarm.particles:
                particle.position = np.random.uniform(swarm.lower_bounds, swarm.upper_bounds, swarm.dimensions)
            swarm.position_initialised = True
        return swarm


class UpdatePosition(Processor[Swarm]):
    """Updates the position of particles in the swarm.

    """

    def process(self, swarm: Swarm) -> Swarm:
        """Updates the position of particles in the swarm.

        Args:
            swarm (Swarm): The Swarm object to update positions for.

        Returns:
            Swarm: The Swarm object with updated positions.
        """
        for particle in swarm.particles:
            particle.position += particle.velocity
            particle.position = np.clip(particle.position, swarm.lower_bounds, swarm.upper_bounds)
        return swarm
