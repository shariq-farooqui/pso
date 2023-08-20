import os
from datetime import datetime

from pymongo import MongoClient

from pso.swarm import Swarm

from .processor import Processor


class MongoDBExporter(Processor[Swarm]):
    """
    A class used to export PSO swarm data to MongoDB.

    Attributes
    ----------
    database_url : str
        The URL of the MongoDB database.
    client : MongoClient
        The MongoDB client instance.
    database : pymongo.database.Database
        The MongoDB database instance.
    collection : pymongo.collection.Collection
        The MongoDB collection instance.

    """

    def __init__(self, collection: str = "runs"):
        """
        Constructs a new MongoDBExporter instance.

        Parameters
        ----------
        collection : str, optional
            The name of the MongoDB collection to use (default is "runs").
        """
        self.database_url = os.environ.get("DATABASE_URL")
        self.client = MongoClient(self.database_url)
        self.database = self.client.pso
        self.collection = self.database[collection]

    def process(self, swarm: Swarm) -> Swarm:
        """
        Processes the given swarm and exports the data to MongoDB.

        Parameters
        ----------
        swarm : Swarm
            The swarm to process.

        Returns
        -------
        Swarm
            The processed swarm.
        """
        if swarm.current_iteration == swarm.max_iterations:
            swarm.finished_at = datetime.now().timestamp()

        document = {
            "run_id":
            swarm.run_id,
            "iteration":
            swarm.current_iteration,
            "created_at":
            swarm.created_at,
            "finished_at":
            swarm.finished_at,
            "settings": {
                "problem_type": swarm.problem_type,
                "num_particles": swarm.num_particles,
                "dimensions": swarm.dimensions,
                "bounds": swarm.bounds,
                "cognitive_weight": swarm.cognitive_weight,
                "social_weight": swarm.social_weight,
                "inertia_weight": swarm.inertia_weight,
                "objective_function": swarm.objective_function.__name__,
            },
            "global_best": {
                "score": swarm.global_best_score,
                "position": swarm.global_best_position.tolist(),
            },
            "particles": [{
                "particle_id": idx,
                "position": particle.position.tolist(),
                "velocity": particle.velocity.tolist(),
                "best_position": particle.best_position.tolist(),
                "best_score": particle.best_score,
            } for idx, particle in enumerate(swarm.particles)],
        }

        self.collection.insert_one(document)
        return swarm
