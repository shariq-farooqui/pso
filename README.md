# Particle Swarm Optimisation Visualisation Tool

An interactive platform for visualising and experimenting with Particle Swarm Optimisation (PSO) algorithms with real-time visualisation, configurable parameters, and performance metrics.

## Overview

This application provides an interface for exploring Particle Swarm Optimisation, a population-based optimisation technique inspired by social behaviour of birds and fish. The tool enables:

- Configuration of PSO algorithms with different topologies and parameters
- Visualisation of the optimisation process through dynamic animations
- Analysis of performance metrics like convergence rate and precision
- Comparison of different PSO configurations and their effectiveness

## Features

- **Multi-dimensional PSO Support**: 1D, 2D, and n-dimensional problems with dimension reduction for visualisation
- **Visualisation Options**: Multiple animation types (1D/2D plots, contour plots, PCA)
- **Topology Options**: Global and ring topologies
- **Configurable Parameters**: Cognitive weight, social weight, inertia
- **Objective Functions**: Sphere, quadratic, rastrigin with configurable bounds
- **Performance Metrics**: Precision tracking, convergence detection, execution time

## Architecture

- **Backend**: FastAPI-based Python service implementing PSO algorithms
- **Frontend**: Streamlit interface for configuration and visualisation
- **Database**: MongoDB for storing run history and results
- **Containers**: Docker Compose for service management

## Getting Started

### Prerequisites
- Docker and Docker Compose
- Make (optional, for convenience commands)

### Installation

1. Clone the repository:
```
git clone https://github.com/shariq-farooqui/pso.git
cd pso
```

2. Create a media volume for storing animations:
```
docker volume create media
```

> **Note**: The media volume is created externally to persist animation files between container restarts. This allows animations to be reused across sessions and prevents the need to regenerate them each time the containers are restarted, significantly improving performance.

3. Build and start the application:
```
make build
make start
```

4. Access the application:
```
http://localhost:8501
```

## Usage

### Running a PSO Optimisation

1. Navigate to the "Playground" page
2. Configure parameters (topology, problem type, objective function, weights, bounds)
3. Click "RUN PSO" to start the optimisation
4. View results and animation

### Viewing Past Runs

1. Navigate to the "All Runs" page
2. Select a run ID to view detailed results

## Development

### Running Tests
```
cd backend
pip install -e .
pytest
```

### Project Structure
```
├── backend/              # FastAPI service
│   ├── pso/              # PSO implementation
│   └── tests/            # Test suite
├── frontend/             # Streamlit UI
└── docker-compose.yml    # Service orchestration
```

