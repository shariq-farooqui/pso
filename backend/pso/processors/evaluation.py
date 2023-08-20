import numpy as np

from pso.fitness import ConvergenceCalculator
from pso.swarm import Swarm

from .processor import Processor


class ParticleEvaluator(Processor[Swarm]):
    """A class to evaluate the fitness of each particle in a swarm.

    """

    def process(self, swarm: Swarm) -> Swarm:
        """Evaluates the fitness of each particle in the swarm.

        Args:
            swarm (Swarm): The swarm to evaluate.

        Returns:
            Swarm: The swarm with updated particle scores.
        """
        swarm.evaluate()
        return swarm


class SwarmEvaluator(Processor[Swarm]):
    """A class to evaluate the fitness of a swarm.

    """

    def process(self, swarm: Swarm) -> Swarm:
        """Evaluates the fitness of a swarm and updates the global best score and position.

        Args:
            swarm (Swarm): The swarm to evaluate.

        Returns:
            Swarm: The swarm with updated global best score and position.
        """
        best_score = swarm.global_best_score
        best_position = swarm.global_best_position
        for particle in swarm.particles:
            score = particle.score
            if swarm.problem_type == "max" and (best_score == -np.inf or score > best_score):
                best_score = score
                best_position = particle.position.copy()
            elif swarm.problem_type == "min" and (best_score == np.inf or score < best_score):
                best_score = score
                best_position = particle.position.copy()
        swarm.global_best_score = best_score
        swarm.global_best_position = best_position

        convergence = ConvergenceCalculator(swarm.objective_function.__name__, swarm.dimensions)
        swarm.score_precision.append(convergence.precision_score(best_score))
        swarm.position_precision.append(convergence.precision_position(best_position))
        return swarm
