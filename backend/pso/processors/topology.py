from pso.swarm import Swarm
from pso.topology import GlobalTopology, RingTopology

from .processor import Processor


class InitialiseGlobalTopology(Processor[Swarm]):
    """Initializes the global topology for the swarm.

    This processor initializes the global topology for the swarm by assigning
    every particle in the swarm as a neighbour to every other particle in the
    swarm. This process is only used once, at the start of the optimisation.

    Attributes:
        swarm (Swarm): The swarm to initialize the topology for.

    Returns:
        Swarm: The swarm with the initialized topology.
    """

    def process(self, swarm: Swarm) -> Swarm:
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

    Attributes:
        swarm (Swarm): The swarm to initialize the topology for.

    Returns:
        Swarm: The swarm with the initialized topology.
    """

    def process(self, swarm: Swarm) -> Swarm:
        if not swarm.topology_initialised:
            topology = RingTopology(swarm.particles)
            for i, particle in enumerate(swarm.particles):
                particle.neighbours = topology.assign_neighbours(i)
            swarm.topology_initialised = True
        return swarm
