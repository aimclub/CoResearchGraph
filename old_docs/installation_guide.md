# Installation guide

ChemCoScientist depends on several services: ChromaDB, embedding and reranker services, MinIO (S3), AutoML, and generative models. Each runs in its own separate Docker container.

## Installation

### Project Components Setup

1. ChromaDB, reranker service, embedding service:
    1. Clone/update the repository (use `/home/chem-paper-assistant/` location on the server)
    2. Run `cd infrastructure/chroma`
    3. Run `docker compose up`
2. AutoML
       
    1. cd infrastructure/automl

    2. path/to/python3.10.exe -m venv env

    3. pip install -r requirements.txt

    4. source env/Scripts/activate

    5. python automl_api.py

    6. In automl_api.py script you should set port where you want to run server.

4. Generative models
    1. Instructions for build and run container with generative models
        
        The easiest way to work with this part of the project is to build a container on a server with an available video card.
        
        ```
        git clone https://github.com/ITMO-NSS-team/MADD.git
        ```
        
        You need to specify the required parameters in the DockerFile, such as:
        ```
        GEN_APP_PORT (the port on which you plan to deploy the container with generative models),
        ML_MODEL_URL (The address (IP and port) where you plan to host the server with predictive models), 
        HF_TOK (for downloading trained models), 
        GITHUB_TOKEN (for the ability to make commits to the code).
        ```
        ```
        cd infrastructure/generative_models
        
        docker build -t generative_model_backend .
        ```

    2. Running a container

        The container may take quite a long time to build, since the environment for its operation requires a long installation and time. However, this is done quite simply.
        
        Next, after you have created an image on your server (or locally), you need to run the container with the command:
        ```
        docker run --name molecule_generator --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=<your device ID> -it --init generative_model_backend:latest bash
        
        OR 
        
        docker run --name molecule_generator --runtime=nvidia -e --gpus all -it --init generative_model_backend:latest bash
        ```
        The container should automatically launch a server with the FastAPI and generative models. However, if this doesn't happen, you should manually run the code
        ```
        bash /projects/MADD/infrastructure/generative_models/api.sh
        ```
6. ChemCoScientist app:
    1. Clone/update the repository (use `/home/chem-paper-assistant/` location on the server)
    2. Create a `config.env` file in the root of the project based on [example_config.env](../example_config.env)
    3. Adjust the path to the volume if necessary in [docker-compose.yml](../docker/docker-compose.yml)
    4. Run `cd docker`
    5. Run `docker compose up`
