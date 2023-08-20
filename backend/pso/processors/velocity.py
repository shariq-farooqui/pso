import numpy as np

from pso.swarm import Swarm

from .processor import Processor


class InitialiseVelocity(Processor[Swarm]):
    """Initialises the velocity of particles in a swarm.

    This processor is only used once, at the start of the optimisation process.

    Args:
        Processor: A generic Processor class that takes in a Swarm type.

    Returns:
        Swarm: The swarm with initialised velocities for its particles.
    """

    def process(self, swarm: Swarm) -> Swarm:
        """Initialises the velocity of particles in a swarm.

        Args:
            swarm (Swarm): The swarm whose particles' velocities are to be initialised.

        Returns:
            Swarm: The swarm with initialised velocities for its particles.
        """
        if not swarm.velocity_initialised:
            for particle in swarm.particles:
                particle.velocity = np.random.uniform(-1, 1, swarm.dimensions)
            swarm.velocity_initialised = True
        return swarm


class UpdateVelocity(Processor[Swarm]):
    """Updates the velocity of particles in a swarm.

    Args:
        Processor: A generic Processor class that takes in a Swarm type.

    Returns:
        Swarm: The swarm with updated velocities for its particles.
    """

    def process(self, swarm: Swarm) -> Swarm:
        """Updates the velocity of particles in a swarm.

        Args:
            swarm (Swarm): The swarm whose particles' velocities are to be updated.

        Returns:
            Swarm: The swarm with updated velocities for its particles.
        """
        optimisation_type = swarm.problem_type
        for particle in swarm.particles:
            if optimisation_type == "max":
                neighbour_best_position = max(particle.neighbours, key=lambda p: p.best_score).best_position
            else:
                neighbour_best_position = min(particle.neighbours, key=lambda p: p.best_score).best_position

            cognitive = swarm.cognitive_weight * np.random.random(swarm.dimensions) * \
                (particle.best_position - particle.position)
            social = swarm.social_weight * np.random.random(swarm.dimensions) * \
                (neighbour_best_position - particle.position)

            particle.velocity = swarm.inertia_weight * particle.velocity + cognitive + social
            particle.velocity = np.clip(particle.velocity, swarm.lower_bounds, swarm.upper_bounds)
        return swarm
