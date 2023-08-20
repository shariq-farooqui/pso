from pso.swarm import Swarm
from pso.topology import GlobalTopology, RingTopology

from .processor import Processor


class InitialiseGlobalTopology(Processor[Swarm]):
    """Initializes the global topology for the swarm.

    This processor initializes the global topology for the swarm by assigning
    every particle in the swarm as a neighbour to every other particle in the
    swarm. This process is only used once, at the start of the optimisation.

    """

    def process(self, swarm: Swarm) -> Swarm:
        """Initializes the global topology for the swarm.

        Args:
            swarm (Swarm): The swarm object to initialize the global topology for.

        Returns:
            Swarm: The updated swarm object with the global topology initialized.
        """
        if not swarm.topology_initialised:
            topology = GlobalTopology(swarm.particles)
            for i, particle in enumerate(swarm.particles):
                particle.neighbours = topology.assign_neighbours(i)
            swarm.topology_initialised = True
        return swarm


class InitialiseRingTopology(Processor[Swarm]):
    """Initializes the ring topology for the swarm.

    This processor initializes the ring topology for the swarm by assigning
    adjacent particles in the swarm as neighbours. This process is only used
    once, at the start of the optimisation.

    """

    def process(self, swarm: Swarm) -> Swarm:
        """Initializes the ring topology for the swarm.

        Args:
            swarm (Swarm): The swarm to initialize the ring topology for.

        Returns:
            Swarm: The swarm with the ring topology initialized.

        """
        if not swarm.topology_initialised:
            topology = RingTopology(swarm.particles)
            for i, particle in enumerate(swarm.particles):
                particle.neighbours = topology.assign_neighbours(i)
            swarm.topology_initialised = True
        return swarm
