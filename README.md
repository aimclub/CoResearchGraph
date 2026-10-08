# CoScientist

A multi-agent system for solving scientific tasks, built on Google Agent Development Kit (ADK), FEDOT.MAS, and RAG-based tool retrieval. If your are interested in the old version of CoScientist (including ChemCoScientist) see branch *CoScientist_old*.

## Overview

CoScientist is an intelligent multi-agent system designed to automate scientific discovery and research tasks. It orchestrates multiple specialized agents that work together to:

- Generate scientific hypotheses
- Conduct literature and web research
- Execute computational experiments using FEDOT.MAS
- Retrieve relevant tools from a database of MCP servers via RAG

---

## Table of Contents

- [Architecture](#architecture)
- [Agents](#agents)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Development](#development)
- [Dependencies](#dependencies)
- [Documentation](#documentation)
- [License](#license)

---

## Architecture

The system is built on the Google Agent Development Kit (ADK) and consists of the following key components:

```
CoScientist/
├── agents/              # Multi-agent definitions
│   ├── agents.py        # Agent implementations
│   └── prompts.py       # Agent instructions
├── chemical_utils/      # Chemistry-related utilities
│   ├── chemical_functions.py    # Molecule extraction, docking
│   ├── retrosynthesis.py        # Retrosynthesis services
│   └── ocr_pipeline.py         # OCR for molecules/reactions
├── config/              # Configuration management
│   └── settings.py     # Pydantic settings
├── logging/             # Logging utilities
│   └── logger.py       # Application logging
├── paper_parser/        # Scientific paper parsing
│   ├── parse_and_split.py      # PDF parsing with marker
│   ├── s3_connection.py        # S3 storage utilities
│   ├── utils.py                # Parser utilities
│   └── parser_prompts.py       # LLM prompts
├── storage/             # Data models
│   └── models.py       # Retrieval result models
├── tools/               # Tool definitions
│   ├── fedotmas_tools.py    # FEDOT.MAS integration
│   ├── retrieval_tools.py    # RAG retrieval tools
│   ├── web_tools.py         # Web search tools
│   └── utils.py             # Tool utilities
├── main.py              # Main entry point
└── examples/            # Example configurations
```

## Agents

### OrchestratorAgent
The main coordinator agent that orchestrates the entire workflow. It:
- Understands user queries
- Plans minimal steps to solve tasks
- Delegates to specialized agents
- Combines results from multiple agents

### HypothesesAgent
Generates scientifically grounded hypotheses that can be validated. It:
- Proposes 2-5 distinct hypotheses per task
- Keeps them concise and actionable
- Prefers testable and experimentally verifiable ideas

### ResearchAgent
Retrieves scientific knowledge from literature and web sources. It:
- Gathers reliable information
- Produces clear, accurate answers
- Extracts key points and uncertainties

### ToolRetrieverAgent
Retrieves relevant MCP servers from a RAG database. It:
- Breaks tasks into capabilities
- Uses RAG to find appropriate tools
- Returns server IDs for task execution

### ExperimentAgent (FEDOT)
Executes computational experiments via FEDOT.MAS. It:
- Builds multi-agent pipelines from text descriptions
- Runs ML/data processing experiments
- Validates hypotheses computationally

### TaskExecutorAgent
The execution router — the single entry point for work that must actually run.
It executes nothing itself; it picks a path and delegates:
1. **ToolPipelineAgent** — the ready-made-tools path: retrieves relevant MCP
   servers, deploys them, and runs the experiment with them.
2. **CoderAgent** — the engineering path: writes and runs code in the sandbox
   when no existing tool covers the task.

When the pipeline reports `NO_MATCHING_TOOL`, the router re-issues the same task
to the coder itself, so the orchestrator never has to guess up front whether a
ready tool exists.

## Key Features

### Multi-Agent Coordination
- Hierarchical agent orchestration
- Tool-based agent invocation
- Sequential and parallel task execution

### Chemical Computing
- Molecule extraction from images and PDFs
- Reaction extraction and analysis
- Retrosynthesis planning
- Molecular docking calculations
- OCR pipeline for chemical structures

### Scientific Paper Processing
- PDF parsing with Marker
- Semantic HTML chunking
- Table and image extraction
- S3-based storage

### Tool Retrieval
- RAG-based MCP server discovery
- Hybrid retrieval (dense + sparse)
- Semantic reranking

### Experiment Execution
- FEDOT.MAS integration
- Automated pipeline generation
- Multi-server orchestration

## Installation

### Prerequisites

- Python 3.12.11
- pip package manager
- Git
- Access to required APIs (OpenAI, Tavily, etc.)

### Required Git Dependencies

CoScientist requires two additional Git packages that must be installed separately:

```bash
# Install rag_tools for RAG-based tool retrieval
pip install git+https://github.com/fiestaxxl/rag_tools.git

# Install FEDOT.MAS for multi-agent experiment execution
pip install git+ssh://git@github.com/ITMO-NSS-team/FEDOT.MAS.git#subdirectory=packages/fedotmas
```

### Full Installation

```bash
# Clone the repository
git clone https://github.com/ITMO-NSS-team/CoScientist.git
cd CoScientist

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required Git dependencies
pip install git+https://github.com/fiestaxxl/rag_tools.git
pip install git+ssh://git@github.com/ITMO-NSS-team/FEDOT.MAS.git#subdirectory=packages/fedotmas

# Install the package and dependencies
pip install -e .
```

### Environment Configuration

Copy the example configuration and update with your credentials:

```bash
cp CoScientist/examples/example_config.env .env
```

Required environment variables:

```env
# LLM Configuration
LLM__SERVICE_KEY=your-service-key
LLM__OPENAI_API_KEY=your-openai-key
LLM__MAIN_URL=https://openrouter.ai/api/v1
LLM__MAIN_MODEL=qwen/qwen3-235b-a22b-2507

# Service APIs
SERVICES__TAVILY_API_KEY=your-tavily-key
SERVICES__OPENALEX_API_KEY=your-openalex-key

# RAG Tools Configuration
POSTGRES__HOST=localhost
POSTGRES__PORT=5432
POSTGRES__USER=rag_tools
POSTGRES__PASSWORD=rag_tools_password
POSTGRES__DATABASE=rag_tools
```

For detailed installation instructions, see [INSTALLATION.md](./INSTALLATION.md).

## Usage

### Basic Usage

```python
import asyncio
from CoScientist import create_manager

async def main():
    # Create manager
    manager = await create_manager()

    # Run a query
    result = await manager.run("What are potential drug candidates for COVID-19?")
    print(result)

    await manager.close()

asyncio.run(main())
```

### CLI Interface

```bash
python -m CoScientist.main

OR

uv run python -m CoScientist.main
```

### Chemical Computing Examples

```python
from CoScientist.chemical_utils import (
    calculate_docking_score,
    retrosynthesis_result,
    molecules_ocr
)

# Calculate docking score
score = calculate_docking_score(smiles="CCO", pdb_id="6lu7")

# Plan retrosynthesis
routes = retrosynthesis_result(smiles="C1CCCCC1", mode="fast")

# Extract molecules from images
results = molecules_ocr(["image1.jpg", "image2.png"])
```

### Paper Parsing

```python
from CoScientist.paper_parser import parse_with_marker, html_chunking

# Parse a paper
paper_name, output_dir = parse_with_marker("paper.pdf", use_llm=True)

# Read and chunk the parsed HTML
with open(f"{output_dir}/{paper_name}.html") as f:
    html_content = f.read()

chunks = html_chunking(html_content, paper_name, metadata)
```

## Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest CoScientist/tests/

# Run with coverage
pytest CoScientist/tests/ --cov=CoScientist

# Run specific test file
pytest CoScientist/tests/agents/test_agents.py -v

# Run async tests
pytest CoScientist/tests/ -v --asyncio-mode=auto
```
Please note that the integration tests depend on auxiliary services hosted on the ITMO servers and therefore require VPN access.

## Development

### Project Structure

```
CoScientist/
├── CoScientist/          # Main package
│   ├── __init__.py      # Package initialization
│   ├── main.py          # Entry point
│   ├── agents/          # Agent definitions
│   ├── chemical_utils/   # Chemistry utilities
│   ├── config/          # Configuration
│   ├── logging/         # Logging
│   ├── paper_parser/    # Paper processing
│   ├── storage/         # Data models
│   └── tools/           # Tool definitions
├── tests/               # Test suite
├── docs/                # Documentation
└── examples/            # Example configurations
```

### Adding New Agents

1. Define the agent in `CoScientist/agents/agents.py`
2. Create agent instructions in `CoScientist/agents/prompts.py`
3. Export the agent in `CoScientist/agents/__init__.py`

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm

my_agent = LlmAgent(
    name="MyAgent",
    model=LiteLlm(model="your-model"),
    instruction="Your agent instructions here",
    description="Agent description"
)
```

### Adding New Tools

1. Implement the tool in the appropriate module
2. Create a toolset class if needed
3. Export in `CoScientist/tools/__init__.py`

## Documentation

- [Installation Guide](./docs/INSTALLATION.md) - Detailed installation instructions
- [API Documentation](./docs/API.md) - Complete API reference
- [Contributing Guide](./docs/CONTRIBUTING.md) - How to contribute
- [Architecture](./docs/ARCHITECTURE.md) - System architecture details
- [Microfluidics Profile](./docs/MICROFLUIDICS.md) - Reduced deployment for the microfluidics case (ТЗ agent + planner + literature analysis); run it with `COSCIENTIST_CONFIG=microfluidics` or `python scripts/run_microfluidics_web.py`

## Dependencies

### Core Dependencies

- `google-adk` - Agent Development Kit
- `google-genai` - Google generative AI
- `litellm` - Unified LLM interface
- `pydantic` - Data validation
- `pydantic-settings` - Settings management

### External Services

- `rag_tools` - RAG-based tool retrieval
- `fedotmas` - Multi-agent experiment execution

### Optional Dependencies

- `chromadb` - Vector database
- `marker` - PDF parsing
- `openchemie` - Chemical structure recognition
- `boto3` - AWS S3 integration

## TODO

- Add SQLite-backed persistence for registered web users and their sessions.
  Persist ADK session state, chat history, roadmaps, and the last active session
  so work can be restored after a service restart. During local testing,
  CoScientist intentionally keeps users and sessions in memory; they survive a
  browser refresh while the service is running, but are cleared on restart.
  The same persistence layer should eventually provide transactional storage
  for global knowledge and scoped execution/research graphs when multi-worker
  or distributed A2A deployment is required; the current JSON graph backend is
  intentionally single-process/single-writer.

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

## Authors

ITMO-NSS-team

## Acknowledgments

- Google Agent Development Kit team
- FEDOT.MAS development team
- RAG Tools development team

## Citation

If you use CoScientist in your research, please cite:

### APA format:

    ITMO-NSS-team (2025). CoScientist repository [Computer software]. https://github.com/ITMO-NSS-team/CoScientist

### BibTeX format:
```bibtex
    @misc{CoScientist,

        author = {ITMO-NSS-team},

        title = {CoScientist repository},

        year = {2025},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/ITMO-NSS-team/CoScientist.git}},

        url = {https://github.com/ITMO-NSS-team/CoScientist.git}

    }
```

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Contact the maintainers


## Changelog

### v1.0.0 (2025-01)
- Initial release
- Multi-agent orchestration
- FEDOT.MAS integration
- RAG-based tool retrieval
- Chemical computing utilities
- Paper parsing pipeline
