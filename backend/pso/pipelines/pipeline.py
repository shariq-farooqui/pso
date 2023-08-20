from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pso.processors import (
    InitialiseGlobalTopology,
    InitialisePosition,
    InitialiseRingTopology,
    InitialiseVelocity,
    MongoDBExporter,
    ParticleEvaluator,
    Processor,
    SwarmEvaluator,
    UpdatePosition,
    UpdateVelocity,
)
from pso.swarm import Swarm

T = TypeVar("T")


class Pipeline(ABC, Generic[T]):
    """Abstract base class for defining a pipeline.

    A pipeline is a sequence of processors that are applied to a given input data.
    """

    @abstractmethod
    def run(self, data: T) -> T:
        """Runs the pipeline on the given input data.

        Args:
            data: The input data to be processed.

        Returns:
            The processed output data.
        """
        pass


class PSOPipeline(Pipeline[Swarm]):
    """A pipeline for running Particle Swarm Optimization (PSO) algorithm.

    This pipeline applies a sequence of processors to a swarm of particles to optimize a given objective function.
    """

    def __init__(self, processors: list[Processor[Swarm]]) -> None:
        """Initializes the PSO pipeline with a list of processors.

        Args:
            processors: A list of processors to be applied to the swarm of particles.
        """
        self.processors = processors

    def run(self, swarm: Swarm) -> Swarm:
        """Runs the PSO pipeline on the given swarm of particles.

        Args:
            swarm: The swarm of particles to be optimized.

        Returns:
            The optimized swarm of particles.
        """
        for processor in self.processors:
            swarm = processor.process(swarm)
        return swarm


class StandardGlobalPSOPipeline(PSOPipeline):
    """A pipeline for running Standard Global PSO algorithm.

    This pipeline applies a sequence of processors to a swarm of particles to optimize a given objective function.
    The topology used in this algorithm is global.
    """

    def __init__(self) -> None:
        """Initializes the Standard Global PSO pipeline with a sequence of processors."""

        super().__init__([
            InitialisePosition(),
            InitialiseVelocity(),
            InitialiseGlobalTopology(),
            ParticleEvaluator(),
            SwarmEvaluator(),
            UpdateVelocity(),
            UpdatePosition(),
            MongoDBExporter(),
        ])


class StandardRingPSOPipeline(PSOPipeline):
    """A pipeline for running Standard Ring PSO algorithm.

    This pipeline applies a sequence of processors to a swarm of particles to optimize a given objective function.
    The topology used in this algorithm is ring.
    """

    def __init__(self) -> None:
        """Initializes the Standard Ring PSO pipeline with a sequence of processors."""

        super().__init__([
            InitialisePosition(),
            InitialiseVelocity(),
            InitialiseRingTopology(),
            ParticleEvaluator(),
            SwarmEvaluator(),
            UpdateVelocity(),
            UpdatePosition(),
            MongoDBExporter(),
        ])
