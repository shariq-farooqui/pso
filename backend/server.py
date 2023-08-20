import inspect
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase

from pso import fitness
from pso.models import AnimationRequest, AppSettings, ObjectiveEvaluationRequest, SwarmConfig
from pso.pipelines import PSORunner, StandardGlobalPSOPipeline, StandardRingPSOPipeline
from pso.swarm import SwarmBuilder
from pso.utils import Animator

app = FastAPI()
objective_functions = {name: func for name, func in inspect.getmembers(fitness, inspect.isfunction)}


async def get_settings() -> AppSettings:
    """Returns an AppSettings instance.

    Returns:
        AppSettings: The application settings.
    """
    return AppSettings()


async def get_database(settings: AppSettings = Depends(get_settings)) -> AsyncIOMotorDatabase:
    """Returns an AsyncIOMotorDatabase instance based on the provided AppSettings.

    Args:
        settings (AppSettings): The application settings.

    Returns:
        AsyncIOMotorDatabase: The database instance.
    """
    client = AsyncIOMotorClient(settings.DATABASE_URL)
    database = client.get_database()
    return database


async def get_collection(database: AsyncIOMotorDatabase = Depends(get_database)) -> AsyncIOMotorCollection:
    """Returns an AsyncIOMotorCollection instance based on the provided AsyncIOMotorDatabase.

    Args:
        database (AsyncIOMotorDatabase): The database instance.

    Returns:
        AsyncIOMotorCollection: The collection instance.
    """
    collection = database.get_collection("runs")
    return collection


@app.get("/ping")
async def ping(settings: AppSettings = Depends(get_settings)):
    """Returns a dictionary containing the build environment and PSO version.

    Args:
        settings (AppSettings): The application settings.

    Returns:
        dict: A dictionary containing the build environment and PSO version.
    """
    return {"build_environment": settings.BUILD_ENVIRONMENT, "version": settings.PSO_VERSION}


@app.get("/ping_database")
async def ping_database(database: AsyncIOMotorDatabase = Depends(get_database)):
    """Returns a dictionary containing the status of the MongoDB connection.

    Args:
        database (AsyncIOMotorDatabase): The database instance.

    Returns:
        dict: A dictionary containing the status of the MongoDB connection.
    """
    try:
        await database.command("ismaster")
        return {"status": "MongoDB connection is working properly."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred with MongoDB: {str(e)}")


@app.get("/available_objective_functions")
async def available_objective_functions() -> list[str]:
    """Returns a list of available objective functions.

    Returns:
        list[str]: A list of available objective functions.
    """
    return list(objective_functions.keys())


@app.post("/run")
async def run(config: SwarmConfig) -> dict:
    """Runs the PSO algorithm based on the provided SwarmConfig.

    Args:
        config (SwarmConfig): The swarm configuration.

    Returns:
        dict: A dictionary containing the run ID, global best score, global best position, and time taken in seconds.
    """
    swarm_builder = SwarmBuilder()
    swarm_config = config.model_dump()
    topology = swarm_config.pop("topology")
    swarm = swarm_builder.build_from_dict(swarm_config)

    if topology == "global":
        pipeline = StandardGlobalPSOPipeline()
    elif topology == "ring":
        pipeline = StandardRingPSOPipeline()
    else:
        raise HTTPException(status_code=400, detail="Invalid topology")

    pso = PSORunner(pipeline, swarm)
    final_swarm = await pso.run()
    best_score = final_swarm.global_best_score
    best_position = final_swarm.global_best_position
    created_at = datetime.fromtimestamp(final_swarm.created_at)
    finished_at = datetime.fromtimestamp(final_swarm.finished_at)
    time_taken = finished_at - created_at
    response = {
        "run_id": final_swarm.run_id,
        "global_best_score": best_score,
        "global_best_position": best_position.tolist(),
        "time_taken_seconds": time_taken.total_seconds(),
    }
    return response


@app.get("/run/{run_id}")
async def get_run(run_id: str, collection: AsyncIOMotorCollection = Depends(get_collection)):
    """Returns a list of documents based on the provided run ID.

    Args:
        run_id (str): The run ID.
        collection (AsyncIOMotorCollection): The collection instance.

    Returns:
        list: A list of documents based on the provided run ID.
    """
    cursor = collection.find({"run_id": run_id}, {"_id": 0}).sort("iteration", 1)
    documents = await cursor.to_list(length=None)
    if not documents:
        raise HTTPException(status_code=404, detail="Run ID not found")
    return documents


@app.post("/animate")
async def animate_run(request: AnimationRequest, collection: AsyncIOMotorCollection = Depends(get_collection)) -> dict:
    """Animates the PSO algorithm based on the provided AnimationRequest.

    Args:
        request (AnimationRequest): The animation request.
        collection (AsyncIOMotorCollection): The collection instance.

    Returns:
        dict: A dictionary containing the filepath of the animation.
    """
    run_id = request.run_id
    animation_type = request.animation_type
    design_index = request.design_index
    fixed_values = request.fixed_values
    cursor = collection.find({"run_id": run_id}).sort("iteration", 1)
    documents = await cursor.to_list(length=None)
    if not documents:
        raise HTTPException(status_code=404, detail="Run ID not found")
    animator = Animator(documents=documents,
                        run_id=run_id,
                        animation_type=animation_type,
                        design_index=design_index,
                        fixed_values=fixed_values)
    filepath = animator.animate()

    return {"filepath": filepath}


@app.post("/evaluate_objective_function")
async def evaluate_objective_function(request: ObjectiveEvaluationRequest):
    """Evaluates the objective function based on the provided ObjectiveEvaluationRequest.

    Args:
        request (ObjectiveEvaluationRequest): The objective evaluation request.

    Returns:
        dict: A dictionary containing the result of the objective function evaluation.
    """
    try:
        chosen_function = objective_functions[request.function_name]
    except KeyError:
        raise HTTPException(status_code=404, detail="Objective function not found")
    values = request.position
    result = chosen_function(values)
    return {"result": result}
