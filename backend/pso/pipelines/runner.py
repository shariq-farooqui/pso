from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pso.pipelines import PSOPipeline
from pso.swarm import Swarm
from pso.utils.logger import get_logger

T = TypeVar("T")


class PipelineRunner(ABC, Generic[T]):
    """Abstract base class for pipeline runners.

    Attributes:
        T: Generic type variable for the return type of the run method.
    """

    @abstractmethod
    def run(self, iterations: int) -> T:
        """Runs the pipeline for the specified number of iterations.

        Args:
            iterations: The number of iterations to run the pipeline for.

        Returns:
            The result of running the pipeline.
        """
        pass


class PSORunner(PipelineRunner[Swarm]):
    """A pipeline runner for particle swarm optimization (PSO) pipelines.

    Attributes:
        pipeline: The PSO pipeline to run.
        swarm: The swarm to run the pipeline on.
        logger: The logger to use for logging.
    """

    def __init__(self, pipeline: PSOPipeline, swarm: Swarm):
        """Initializes a new instance of the PSORunner class.

        Args:
            pipeline: The PSO pipeline to run.
            swarm: The swarm to run the pipeline on.
        """
        self.pipeline = pipeline
        self.swarm = swarm
        self.logger = get_logger(__name__)

    def run(self) -> Swarm:
        """Runs the PSO pipeline on the swarm.

        Returns:
            The swarm after running the pipeline.
        """
        for i in range(self.swarm.max_iterations):
            self.swarm = self.pipeline.run(self.swarm)
            self.swarm.current_iteration += 1
        self.logger.info(f"Best position: {self.swarm.global_best_position}")
        self.logger.info(f"Best score: {self.swarm.global_best_score}")
        return self.swarm
