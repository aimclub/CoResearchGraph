# API Documentation

Complete API reference for CoScientist.

## Table of Contents

- [Main Module](#main-module)
- [Agents](#agents)
- [Chemical Utilities](#chemical-utilities)
- [Configuration](#configuration)
- [Logging](#logging)
- [Paper Parser](#paper-parser)
- [Storage](#storage)
- [Tools](#tools)

---

## Main Module

### CoScientistManager

Main manager class for running the CoScientist multi-agent system.

```python
from CoScientist import CoScientistManager

manager = CoScientistManager(
    app_name: str = "coscientist_app",
    user_id: str = "user_1",
    session_id: str = "session_001"
)
```

#### Methods

##### `__init__`

```python
def __init__(
    self,
    app_name: str = "coscientist_app",
    user_id: str = "user_1",
    session_id: str = "session_001"
) -> None
```

Initialize the CoScientistManager.

**Parameters:**
- `app_name` (str): Name of the application session.
- `user_id` (str): Identifier for the user.
- `session_id` (str): Unique session identifier.

##### `initialize`

```python
async def initialize() -> None
```

Initialize the session service and runner. Must be called before `run()`.

##### `run`

```python
async def run(query: str, verbose: bool = True) -> str
```

Execute a query through the orchestrator agent.

**Parameters:**
- `query` (str): User query to process.
- `verbose` (bool): Whether to print events. Default: True.

**Returns:**
- `str`: Final agent response.

**Example:**
```python
manager = await create_manager()
result = await manager.run("What are potential drug candidates for cancer?")
print(result)
```

##### `close`

```python
async def close() -> None
```

Cleanup resources. Call when done with the manager.

---

### create_manager

```python
async def create_manager() -> CoScientistManager
```

Create and initialize a CoScientistManager instance.

**Returns:**
- `CoScientistManager`: Initialized manager instance.

**Example:**
```python
manager = await create_manager()
```

---

## Agents

### orchestrator_agent

Main orchestration agent that coordinates all other agents.

```python
from CoScientist.agents import orchestrator_agent
```

**Type:** `LlmAgent`

**Tools Available:**
- HypothesesAgent
- ResearchAgent
- TaskExecutorAgent (routes execution to the MCP tool pipeline or the CoderAgent)
- MedicalAgent

**Purpose:** Coordinates the entire workflow by delegating to specialized agents.

---

### hypotheses_agent

Agent for generating scientific hypotheses.

```python
from CoScientist.agents import hypotheses_agent
```

**Type:** `LlmAgent`

**Output Key:** `hypotheses`

**Purpose:** Generate testable, scientifically grounded hypotheses for given tasks.

---

### research_agent

Agent for retrieving and mining scientific knowledge.

```python
from CoScientist.agents import research_agent
```

**Type:** `LlmAgent`

**Output Key:** `search_results`

**Tools:** `websearch_toolset_instance`

**Purpose:** Gather information from literature and web sources.

---

### fedot_agent

Agent for running computational experiments via FEDOT.MAS.

```python
from CoScientist.agents import fedot_agent
```

**Type:** `LlmAgent`

**Output Key:** `fedot_results`

**Tools:** `fedot_toolset_instance`

**Purpose:** Execute ML/experiments using FEDOT.MAS multi-agent pipelines.

---

### tool_retriever_agent

Agent for retrieving relevant MCP servers from RAG database.

```python
from CoScientist.agents import tool_retriever_agent
```

**Type:** `LlmAgent`

**Output Key:** `retrieved_tools`

**Output Schema:** `RetrievalFinalResult`

**Tools:** `retrieval_toolset_instance`

**Purpose:** Find appropriate tools for task execution.

---

### task_execution_agent

Execution router over the two ways a task can actually be run.

```python
from CoScientist.agents import task_execution_agent
```

**Type:** `LlmAgent`

**Subordinates (AgentTools):**
1. `ToolPipelineAgent` — `SequentialAgent`: tool preparation (retrieve → rerank →
   deploy) followed by `ExecutorSwitchAgent`, which runs the deployed MCP tools
   via `ExperimentAgent` — or, when the reranker returned no usable ranking,
   hands the whole unfiltered candidate set to `FedotAgent` instead
2. `coder_agent` — writes and runs code in the sandbox

**Purpose:** Deliver a task's result, choosing between ready-made MCP tools and
engineering work; on a `NO_MATCHING_TOOL` abstention it re-issues the task to the
coder itself.

---

### Agent Prompts

Prompts are no longer importable strings. They are templates registered in
`CoScientist/agents/prompts/templates.py` and rendered by the assembler
(`CoScientist.assembly`) from `CoScientist/agents/system.yaml` — tool lists,
agent rosters and HITL sections are filled from the same config that wires the
agents. To inspect a rendered prompt:

```python
from CoScientist.assembly import build_system

system = build_system()
print(system.agent("ResearchAgent").instruction)
```

---

## Chemical Utilities

### chemical_functions

#### calculate_docking_score

```python
from CoScientist.chemical_utils import calculate_docking_score

result = calculate_docking_score(
    smiles: str,
    pdb_id: str
) -> Dict
```

Calculate docking score for a molecule.

**Parameters:**
- `smiles` (str): SMILES string of the molecule.
- `pdb_id` (str): PDB ID of the receptor structure.

**Returns:**
- `Dict`: Docking score and metadata.

**Example:**
```python
result = calculate_docking_score(smiles="CCO", pdb_id="6lu7")
```

---

#### extract_reactions_from_pdf

```python
from CoScientist.chemical_utils import extract_reactions_from_pdf

reactions = extract_reactions_from_pdf(file: bytes) -> List[Dict]
```

Extract reactions from a PDF file.

**Parameters:**
- `file` (bytes): PDF file content.

**Returns:**
- `List[Dict]`: List of reactions for each page.

---

#### extract_reactions_from_figure

```python
from CoScientist.chemical_utils import extract_reactions_from_figure

reactions = extract_reactions_from_figure(image: bytes) -> List[Dict]
```

Extract reactions from an image.

**Parameters:**
- `image` (bytes): Image file content.

**Returns:**
- `List[Dict]`: List of reactions detected.

---

#### extract_molecules_from_pdf

```python
from CoScientist.chemical_utils import extract_molecules_from_pdf

molecules = extract_molecules_from_pdf(file: bytes) -> List[Dict]
```

Extract molecules from a PDF file.

**Parameters:**
- `file` (bytes): PDF file content.

**Returns:**
- `List[Dict]`: List of molecules with SMILES and bbox.

---

#### extract_molecules_from_figure

```python
from CoScientist.chemical_utils import extract_molecules_from_figure

molecules = extract_molecules_from_figure(image: bytes) -> List[Dict]
```

Extract molecules from an image.

**Parameters:**
- `image` (bytes): Image file content.

**Returns:**
- `List[Dict]`: List of molecules with SMILES and bbox.

---

#### convert_image_to_smiles

```python
from CoScientist.chemical_utils import convert_image_to_smiles

smiles = convert_image_to_smiles(image: bytes) -> str
```

Convert a chemical structure image to SMILES.

**Parameters:**
- `image` (bytes): Image file content.

**Returns:**
- `str`: SMILES string.

---

#### remove_keys

```python
from CoScientist.chemical_utils import remove_keys

cleaned = remove_keys(
    obj: Any,
    keys_to_remove: set[str] = {"bbox", "score"}
) -> Any
```

Remove specified keys from a nested dictionary or list.

**Parameters:**
- `obj` (Any): Object to process.
- `keys_to_remove` (set[str]): Keys to remove. Default: `{"bbox", "score"}`.

**Returns:**
- `Any`: Cleaned object.

---

### retrosynthesis

#### retrosynthesis_result

```python
from CoScientist.chemical_utils import retrosynthesis_result

result = retrosynthesis_result(
    smiles: str,
    mode: str = "fast",
    max_routes: int = 5
) -> Dict[str, Any]
```

Get retrosynthesis routes for a target molecule.

**Parameters:**
- `smiles` (str): Target molecule SMILES.
- `mode` (str): Search mode. Options: "fast", "balanced", "deep". Default: "fast".
- `max_routes` (int): Maximum routes to return. Default: 5.

**Returns:**
- `Dict[str, Any]`: Retrosynthesis result with routes, steps, and metadata.

**Example:**
```python
result = retrosynthesis_result(smiles="C1CCCCC1", mode="fast", max_routes=3)
routes = result["routes"]
```

---

#### classify_reaction_smiles

```python
from CoScientist.chemical_utils import classify_reaction_smiles

result = classify_reaction_smiles(
    smiles: List[str],
    num_results: int = 10
) -> Dict[str, Any]
```

Classify chemical reactions.

**Parameters:**
- `smiles` (List[str]): List of reaction SMILES.
- `num_results` (int): Number of classification results. Default: 10.

**Returns:**
- `Dict[str, Any]`: Classification results with reaction classes.

---

#### forward_predict_products

```python
from CoScientist.chemical_utils import forward_predict_products

result = forward_predict_products(
    smiles: List[str],
    backend: str = "wldn5",
    model_name: str = "pistachio",
    reagents: str = "",
    solvent: str = ""
) -> Dict[str, Any]
```

Predict products from reaction inputs.

**Parameters:**
- `smiles` (List[str]): Reactant SMILES.
- `backend` (str): Prediction backend. Default: "wldn5".
- `model_name` (str): Model name. Default: "pistachio".
- `reagents` (str): Reagents string.
- `solvent` (str): Solvent string.

**Returns:**
- `Dict[str, Any]`: Predicted products with scores.

---

### ocr_pipeline

#### molecules_ocr

```python
from CoScientist.chemical_utils import molecules_ocr

result = molecules_ocr(images: List[str]) -> Dict
```

Extract molecules from image paths using OCR.

**Parameters:**
- `images` (List[str]): List of image file paths.

**Returns:**
- `Dict`: Dictionary mapping filenames to extracted SMILES.

**Example:**
```python
result = molecules_ocr(["mol1.jpg", "mol2.png"])
```

---

#### reactions_ocr

```python
from CoScientist.chemical_utils import reactions_ocr

result = reactions_ocr(images: List[str]) -> Dict
```

Extract reactions from image paths using OCR.

**Parameters:**
- `images` (List[str]): List of image file paths.

**Returns:**
- `Dict`: Dictionary with reaction details.

---

#### draw_bboxes_on_image

```python
from CoScientist.chemical_utils import draw_bboxes_on_image

annotated = draw_bboxes_on_image(
    image: bytes,
    bboxes: Dict
) -> bytes
```

Draw bounding boxes on an image.

**Parameters:**
- `image` (bytes): Original image.
- `bboxes` (Dict): Bounding box coordinates.

**Returns:**
- `bytes`: Annotated image as JPEG.

---

#### render_molecule_detections

```python
from CoScientist.chemical_utils import render_molecule_detections

rendered = render_molecule_detections(
    images: List,
    bboxes_list: List,
    res_path: Optional[str] = None
) -> List[Tuple[str, bytes]]
```

Render molecule detections with bounding boxes.

**Parameters:**
- `images` (List): List of images.
- `bboxes_list` (List): Bounding box coordinates.
- `res_path` (Optional[str]): Optional save path.

**Returns:**
- `List[Tuple[str, bytes]]`: List of (filename, image_bytes).

---

## Configuration

### Settings

```python
from CoScientist.config import Settings, get_settings

settings = get_settings()
```

#### Settings Fields

##### llm

```python
settings.llm: LLMSettings
```

LLM configuration.

**Fields:**
- `service_key` (Optional[str]): Service API key.
- `openai_api_key` (Optional[str]): OpenAI API key.
- `tavily_api_key` (Optional[str]): Tavily API key.
- `main_url` (Optional[str]): Main LLM service URL.
- `main_model` (Optional[str]): Main model identifier.
- `vision_url` (Optional[str]): Vision model URL.
- `marker_model` (Optional[str]): Marker model for PDF parsing.

##### services

```python
settings.services: ServicesSettings
```

External services configuration.

**Fields:**
- `tavily_api_key` (Optional[str]): Tavily API key.
- `openalex_api_key` (Optional[str]): OpenAlex API key.

##### storage

```python
settings.storage: StorageSettings
```

Storage paths configuration.

**Fields:**
- `root_dir` (Path): Root directory.
- `parse_results` (Optional[str]): Parse results path.
- `chroma_storage` (Optional[str]): ChromaDB storage path.
- `papers_storage` (Optional[str]): Papers storage path.
- `logging_path` (Optional[str]): Logging path. Default: "logs/".

##### hosts_ports

```python
settings.hosts_ports: HostsPortsSettings
```

Service hosts and ports.

**Fields:**
- `chroma_host`, `embedding_host`, `reranker_host`, etc.
- Corresponding port settings.

##### s3

```python
settings.s3: S3Settings
```

S3 storage configuration.

**Fields:**
- `use_s3` (bool): Enable S3 usage. Default: False.
- `endpoint_url` (Optional[str]): S3 endpoint.
- `access_key` (Optional[str]): S3 access key.
- `secret_key` (Optional[str]): S3 secret key.
- `bucket_name` (Optional[str]): Bucket name.

##### tool_rag

```python
settings.tool_rag: ToolRAGSettings
```

RAG tools configuration from rag_tools package.

---

## Logging

### get_logger

```python
from CoScientist.logging import get_logger

logger = get_logger()
```

Get the application logger.

**Returns:**
- `logging.Logger`: Configured logger instance.

**Example:**
```python
logger = get_logger()
logger.info("Processing started")
logger.error("An error occurred")
```

---

## Paper Parser

### parse_and_split

#### parse_with_marker

```python
from CoScientist.paper_parser import parse_with_marker

paper_name, output_dir = parse_with_marker(
    paper_name: str,
    use_llm: bool = False
) -> Tuple[str, Path]
```

Parse a PDF paper using Marker.

**Parameters:**
- `paper_name` (str): Path to PDF file.
- `use_llm` (bool): Use LLM enhancement. Default: False.

**Returns:**
- `Tuple[str, Path]`: Paper stem name and output directory.

**Example:**
```python
stem, out_dir = parse_with_marker("paper.pdf", use_llm=True)
```

---

#### clean_up_html

```python
from CoScientist.paper_parser import clean_up_html

cleaned_html, mapping = clean_up_html(
    doc_dir: Path,
    file_name: str,
    html: str,
    s3_service: Optional[S3BucketService] = None,
    paper_s3_prefix: Optional[str] = None
) -> Tuple[str, Dict]
```

Clean HTML by removing irrelevant sections.

**Parameters:**
- `doc_dir` (Path): Document directory.
- `file_name` (str): HTML filename.
- `html` (str): HTML content.
- `s3_service` (Optional[S3BucketService]): S3 service for storage.
- `paper_s3_prefix` (Optional[str]): S3 prefix.

**Returns:**
- `Tuple[str, Dict]`: Cleaned HTML and image mapping.

---

#### html_chunking

```python
from CoScientist.paper_parser import html_chunking

chunks = html_chunking(
    html_string: str,
    paper_name: str,
    paper_summary: Any
) -> List[Document]
```

Chunk HTML into semantic passages.

**Parameters:**
- `html_string` (str): HTML content.
- `paper_name` (str): Paper name.
- `paper_summary` (Any): Metadata summary.

**Returns:**
- `List[Document]`: List of chunked documents.

---

### utils

#### convert_to_base64

```python
from CoScientist.paper_parser import convert_to_base64

base64_str = convert_to_base64(file_path: str) -> str
```

Convert image to base64 string.

**Parameters:**
- `file_path` (str): Image file path.

**Returns:**
- `str`: Base64 encoded string.

---

#### load_image_as_binary

```python
from CoScientist.paper_parser import load_image_as_binary

image_bytes = load_image_as_binary(file_path: str) -> bytes
```

Load image as binary.

**Parameters:**
- `file_path` (str): Image file path.

**Returns:**
- `bytes`: Image content as bytes.

---

#### prompt_func

```python
from CoScientist.paper_parser import prompt_func

message = prompt_func(data: Dict) -> HumanMessage
```

Create structured message for LLM.

**Parameters:**
- `data` (Dict): Data with "text" and "image" keys.

**Returns:**
- `HumanMessage`: LangChain message.

---

### s3_connection

#### S3BucketService

```python
from CoScientist.paper_parser import S3BucketService

s3 = S3BucketService(
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket_name: str = "default"
)
```

S3 storage service class.

##### Methods

###### `create_s3_client`

```python
client = s3.create_s3_client() -> boto3.client
```

Create S3 client.

---

###### `upload_file_object`

```python
s3.upload_file_object(
    prefix: str,
    source_file_name: str,
    file_path: str
) -> None
```

Upload file to S3.

---

###### `list_objects`

```python
objects = s3.list_objects(prefix: str) -> List[str]
```

List objects in bucket.

---

###### `delete_file_object`

```python
s3.delete_file_object(prefix: str, source_file_name: str) -> None
```

Delete file from S3.

---

###### `generate_presigned_url`

```python
url = s3.generate_presigned_url(
    s3_key: str,
    method: str = 'get_object',
    expiration: int = 360
) -> str
```

Generate presigned URL.

---

## Storage

### RetrievalFinalResult

```python
from CoScientist.storage import RetrievalFinalResult

result = RetrievalFinalResult(
    servers_id: List[str],
    queries: List[str],
    task: str
)
```

Result model for tool retrieval.

**Fields:**
- `servers_id` (List[str]): List of selected MCP server IDs.
- `queries` (List[str]): Queries used for retrieval.
- `task` (str): Original task description.

---

### RetrievalToolResult

```python
from CoScientist.storage import RetrievalToolResult

tool_result = RetrievalToolResult(
    tool: str,
    server_id: str,
    description: str,
    score: float
)
```

Individual tool retrieval result.

**Fields:**
- `tool` (str): Tool name.
- `server_id` (str): Server identifier.
- `description` (str): Tool description.
- `score` (float): Relevance score.

---

## Tools

### FedotMASToolset

```python
from CoScientist.tools import FedotMASToolset, fedot_toolset_instance

fedot_toolset = FedotMASToolset(prefix: str = "fedot_")
```

Toolset for FEDOT.MAS integration.

#### Methods

##### `fedot_tool`

```python
async def fedot_tool(
    self,
    task_description: str,
    server_ids: List[str]
) -> Dict[str, Any]
```

Execute FEDOT.MAS multi-agent pipeline.

**Parameters:**
- `task_description` (str): Task to execute.
- `server_ids` (List[str]): MCP server IDs for execution.

**Returns:**
- `Dict[str, Any]`: Execution result with status and data.

---

### RetrievalToolSet

```python
from CoScientist.tools import RetrievalToolSet, retrieval_toolset_instance

retrieval_toolset = RetrievalToolSet(prefix: str = "rag_")
```

Toolset for RAG-based tool retrieval.

#### Methods

##### `retrieve_tools`

```python
async def retrieve_tools(self, query: str) -> Dict[str, Any]
```

Retrieve relevant MCP tools.

**Parameters:**
- `query` (str): Query for tool retrieval.

**Returns:**
- `Dict[str, Any]`: Retrieved tools with scores.

---

##### `get_server_info`

```python
async def get_server_info(self, server_id: str) -> Dict[str, Any]
```

Get MCP server metadata.

**Parameters:**
- `server_id` (str): Server ID.

**Returns:**
- `Dict[str, Any]`: Server information.

---

### websearch_toolset_instance

```python
from CoScientist.tools import websearch_toolset_instance
```

**Type:** `McpToolset`

Web search toolset using Tavily MCP.

---

## Best Practices

### Async Usage

```python
# Always use await with async functions
manager = await create_manager()
result = await manager.run("Your query")
await manager.close()

# Or run in event loop
asyncio.run(main())
```

### Error Handling

```python
try:
    result = await manager.run(query)
except Exception as e:
    logger.error(f"Error processing query: {e}")
    result = "An error occurred"
```

### Resource Management

```python
# Always close manager when done
manager = await create_manager()
try:
    result = await manager.run(query)
finally:
    await manager.close()
```
