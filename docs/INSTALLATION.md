# Installation Guide

This guide provides detailed instructions for installing and configuring CoScientist.

## Prerequisites

Before installing CoScientist, ensure you have the following:

- **Python 3.12.5** or higher
- **pip** (latest version recommended)
- **Git** (for cloning repositories)
- Access to required API services

### Required API Keys

You will need to obtain API keys for the following services:

| Service | Purpose | Required |
|---------|---------|----------|
| OpenAI API | LLM inference | Yes |
| Tavily API | Web search | Recommended |
| OpenAlex API | Academic literature | Optional |
| S3 Storage | Paper storage | Optional |

## Step 1: Clone the Repository

```bash
git clone https://github.com/ITMO-NSS-team/CoScientist.git
cd CoScientist
```

## Step 2: Create Virtual Environment (Recommended)

Creating a virtual environment is recommended to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

## Step 3: Install Git Dependencies

CoScientist requires two Git-based packages that cannot be installed from PyPI:

### 3.1 Install rag_tools

This package provides RAG-based MCP server retrieval functionality:

```bash
pip install git+https://github.com/fiestaxxl/rag_tools.git
```

**What this installs:**
- RAG retrieval implementation
- PostgreSQL storage backend
- Embedding and reranking utilities
- Tool metadata management

### 3.2 Install FEDOT.MAS

This package provides multi-agent system capabilities for experiment execution:

```bash
pip install git+ssh://git@github.com/ITMO-NSS-team/FEDOT.MAS.git#subdirectory=packages/fedotmas
```

**Note:** This requires SSH access to GitHub. If you don't have SSH configured:

```bash
# Option 1: Use HTTPS with token
pip install git+https://<YOUR_TOKEN>@github.com/ITMO-NSS-team/FEDOT.MAS.git#subdirectory=packages/fedotmas

# Option 2: Clone and install locally
git clone git@github.com:ITMO-NSS-team/FEDOT.MAS.git
cd FEDOT.MAS
pip install packages/fedotmas/
cd ..
```

**What this installs:**
- Multi-agent system orchestration
- HTTP MCP server integration
- Experiment pipeline execution
- Task management utilities

### 3.3 Verify Git Dependencies

After installation, verify both packages are installed:

```python
import rag_tools
import fedotmas

print(f"rag_tools version: {rag_tools.__version__}")
print(f"fedotmas version: {fedotmas.__version__}")
```

## Step 4: Install CoScientist

### From Source

```bash
# Using requirements.txt file
pip install -r requirements.txt

# Or install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev]"
```

## Step 5: Configure Environment Variables

### 5.1 Create Environment File

```bash
cp CoScientist/examples/example_config.env .env
```

### 5.2 Edit Configuration

Open `.env` and configure the following sections:

#### LLM Configuration

```env
# Service Configuration
LLM__SERVICE_KEY=your-service-key
LLM__OPENAI_API_KEY=sk-your-openai-key

# Model Endpoints
LLM__MAIN_URL=https://openrouter.ai/api/v1
LLM__MAIN_MODEL=qwen/qwen3-235b-a22b-2507
LLM__SCENARIO_MODEL=qwen/qwen3-235b-a22b-2507

# Service URLs
LLM__SERVICE_URL=https://openrouter.ai/api/v1
LLM__SERVICE_CC_URL=https://openrouter.ai/api/v1/chat/completions
LLM__VISION_URL=https://openrouter.ai/api/v1;google/gemini-2.5-flash
LLM__MARKER_MODEL=google/gemini-2.0-flash-lite-001

# Allowed Providers
LLM__ALLOWED_PROVIDERS=["google-vertex", "azure"]
```

#### Services APIs

```env
SERVICES__TAVILY_API_KEY=your-tavily-api-key
SERVICES__OPENALEX_API_KEY=your-openalex-api-key
```

#### Storage Paths

```env
STORAGE__PARSE_RESULTS=PaperAnalysis/parse_results
STORAGE__CHROMA_STORAGE=PaperAnalysis/chromadb
STORAGE__PAPERS_STORAGE=PaperAnalysis/papers
STORAGE__DS_STORAGE=ChemCoScientist/data_store/datasets
STORAGE__IMG_STORAGE=ChemCoScientist/data_store/imgs
STORAGE__LOGGING_PATH=logs/
```

#### RAG Tools Configuration

```env
# Database
QDRANT__URL=http://localhost:6333
QDRANT__API_KEY=your_qdrant_api_key #Optional, can be empty

POSTGRES__HOST=localhost
POSTGRES__PORT=5432
POSTGRES__USER=rag_tools
POSTGRES__PASSWORD=rag_tools_password
POSTGRES__DATABASE=rag_tools

# Embedding if planning to use local embedder
EMBEDDING__MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING__DEVICE=cpu
EMBEDDING__BATCH_SIZE=32

API_EMBEDDING__URL=http://localhost:5002/embed
API_EMBEDDING__API_KEY= #optional if api requires api key

# Reranker
RERANKER__MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER__DEVICE=cpu
RERANKER__BATCH_SIZE=32

API_RERANKER__URL=http://localhost:5001/rerank
API_RERANKER__TOP_K=20

# RAG Settings
RAG__DEFAULT_TOP_K=10
RAG__RERANK_TOP_K=5
RAG__MIN_RELEVANCE_SCORE=0.3
RAG__CHUNK_SIZE=512
RAG__CHUNK_OVERLAP=50

WORK_DIR=/tmp/rag_tools
```

## Step 6: Set Up External Services

### 6.1 PostgreSQL and Qdrant Databases (Required for RAG)

The RAG tools require a PostgreSQL and Qdrant databases, see installation guide in the official [repo](https://github.com/fiestaxxl/rag_tools/tree/main)

### 6.2 ChromaDB (Optional)

For vector storage of paper content:

1. ChromaDB, reranker service, embedding service:
    1. Clone/update the repository (use `/home/chem-paper-assistant/` location on the server)
    2. Run `cd infrastructure/chroma`
    3. Run `docker compose up`

### 6.3 Chemical Services (Optional)

For chemical computing features:

```bash
# These services should be running separately
# Refer to OpenChemIE and ASKCOS documentation
```

Configure hosts and ports:

```env
HOSTS_PORTS__CHEM_SERVICES_HOST=localhost
HOSTS_PORTS__CHEM_SERVICES_PORT=8000
HOSTS_PORTS__RETROSYNTHESIS_SERVICES_HOST=localhost
HOSTS_PORTS__RETROSYNTHESIS_SERVICES_PORT=8001
```

### 6.4 Generative models
    1. Instructions for build and run container with generative models
        
        The easiest way to work with this part of the project is to build a container on a server with an available gpu resources.
        
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

## Step 7: Verify Installation

### 7.1 Basic Import Test

```python
from CoScientist import (
    CoScientistManager,
    orchestrator_agent,
    hypotheses_agent,
    research_agent,
    fedot_agent
)

print("All imports successful!")
```

### 7.2 Configuration Test

```python
from CoScientist.config import get_settings

settings = get_settings()
print(f"LLM Model: {settings.llm.main_model}")
print(f"Storage Path: {settings.storage.root_dir}")
```

### 7.3 Run Basic Test

```bash
python -m CoScientist.main
```

You should see:

```
CoScientist (ADK) initialized

Enter query (or 'exit'):
```

## Next Steps

After successful installation:

1. Read the [README.md](./README.md) for an overview
2. Review the [API Documentation](./API.md)
3. Check the [Contributing Guide](./CONTRIBUTING.md)
4. Explore the examples in `CoScientist/examples/`
5. Run the test suite to verify your setup
