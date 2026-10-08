# CoScientist Architecture

This document provides a detailed technical overview of the CoScientist multi-agent system architecture.

## Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Agent System](#agent-system)
4. [Data Flow](#data-flow)
5. [Storage Layer](#storage-layer)
6. [Tool Integration](#tool-integration)
7. [Configuration Management](#configuration-management)
8. [External Dependencies](#external-dependencies)
9. [Sequence Diagrams](#sequence-diagrams)

---

## System Overview

CoScientist is a multi-agent system designed for scientific discovery and research automation. It combines:

- **Google Agent Development Kit (ADK)** for agent orchestration
- **Large Language Models** for reasoning and decision-making
- **FEDOT.MAS** for automated experiment pipeline execution
- **RAG-based tool retrieval** for dynamic MCP server discovery
- **Chemical computing services** for molecular analysis

### Design Principles

1. **Modularity**: Each component is self-contained and loosely coupled
2. **Extensibility**: Easy to add new agents, tools, and capabilities
3. **Scalability**: Stateless design allows horizontal scaling
4. **Observability**: Comprehensive logging for debugging and monitoring

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CoScientist                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐      ┌──────────────────────┐    │
│  │   Client     │────▶│  Orchestrator│ ────▶│   HypothesesAgent    │    │
│  │   (User)     │     │    Agent     │      └──────────────────────┘    │
│  └──────────────┘     │              │                                  │
│                       │              │     ┌──────────────────────┐     │
│                       │              │────▶│   ResearchAgent      │     │
│                       │              │     └──────────────────────┘     │
│                       │              │                                  │
│                       │              │     ┌──────────────────────┐     │
│                       │              │────▶│ TaskExecutorAgent    │     │
│                       │              │     │      (router)        │     │
│                       │              │     │  ┌──────────────┐    │     │
│                       │              │     │  │ ToolPipeline │    │     │
│                       │              │     │  │  Agent       │    │     │
│                       │              │     │  │ (prep → run) │    │     │
│                       │              │     │  └──────────────┘    │     │
│                       │              │     │  ┌──────────────┐    │     │
│                       │              │     │  │ CoderAgent   │    │     │
│                       │              │     │  └──────────────┘    │     │
│                       └──────────────┘     └──────────────────────┘     │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                         External Services                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐    │
│  │  FEDOT.MAS   │  │ RAG Tools    │  │   ChromaDB   │  │  Tavily   │    │
│  │   (MCP)      │  │   (MCP)      │  │  (Vector DB) │  │   API     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └───────────┘    │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │  Chemical    │  │Retrosynthesis│  │   S3         │                   │
│  │  Services    │  │   Service    │ │  Storage      │                   │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Agent System

### Agent Hierarchy

```
CoScientist Agents
│
├── OrchestratorAgent (Root)
│   ├── HypothesesAgent
│   ├── ResearchAgent
│   ├── MedicalAgent
│   └── TaskExecutorAgent (LLM router)
│       ├── ToolPipelineAgent (Sequential)
│       │   ├── ToolPreparerAgent (retrieve → rerank → deploy MCP servers)
│       │   └── ExecutorSwitchAgent (runs exactly one of:)
│       │       ├── ExperimentAgent (runs the deployed MCP tools)
│       │       └── FedotAgent (fallback: reranker produced no usable ranking)
│       └── CoderAgent
│           └── DatasetCollectorAgent
```

### Agent Specifications

#### 1. OrchestratorAgent

**Purpose**: Central coordinator for all agent activities

**Architecture**:
- Type: `LlmAgent` (Google ADK)
- Model: Configurable LLM via LiteLLM
- Tools: AgentTool instances for sub-agents

**Responsibilities**:
- Parse user queries
- Plan execution strategy
- Delegate to specialized agents
- Aggregate results
- Manage conversation state

**Decision Logic**:
```python
if task_needs_ideas:
    delegate(hypotheses_agent)
if task_needs_knowledge:
    delegate(research_agent)
if task_needs_computation:
    delegate(task_executor_agent)
```

#### 2. HypothesesAgent

**Purpose**: Generate testable scientific hypotheses

**Architecture**:
- Type: `LlmAgent`
- Output Key: `hypotheses`
- No external tools


#### 3. ResearchAgent

**Purpose**: Gather scientific knowledge from various sources

**Architecture**:
- Type: `LlmAgent`
- Output Key: `search_results`
- Tools: `websearch_toolset_instance` (Tavily MCP)

**Capabilities**:
- Web search via Tavily API
- Literature search
- Knowledge synthesis

#### 4. ToolRetrieverAgent

**Purpose**: Discover relevant MCP servers using RAG

**Architecture**:
- Type: `LlmAgent`
- Output Key: `retrieved_tools`
- Output Schema: `RetrievalFinalResult`
- Tools: `retrieval_toolset_instance`

**RAG Pipeline**:
1. Query embedding via APIEmbedder
2. Vector search in Qdrant
3. Hybrid reranking (BM25 + API Reranker)
4. Top-k selection

#### 5. ExecutorSwitchAgent (ExperimentAgent / FedotAgent)

**Purpose**: Execute computational experiments with the prepared MCP tools

**Architecture**:
- Type: `custom:executor_switch` — runs exactly ONE of its two children
- Children: `ExperimentAgent` (default), `FedotAgent` (fallback)

`ExperimentAgent` is a ReAct `LlmAgent` (`output_key: fedot_results`) that calls
the task's selected MCP servers directly through `dynamic_tools`, resolved from
`filtered_tools` + `deployed_mcps` on every turn.

`FedotAgent` (`tools: [fedot, ...]`, same `output_key`) runs only when the tool
reranker never actually JUDGED the retrieved candidates — its reply was
unreadable, or carried no usable `{index, score}` pair, and the local
cross-encoder could not stand in for it. `filtered_tools` is then empty for a
reason that says nothing about relevance, so abstaining to `CoderAgent` would
discard tools retrieval found correctly. Instead `fedot_tool` receives the whole
unfiltered `accumulated_tools` pool and FEDOT.MAS's meta-agent does the
selection itself — with a bounded time budget, since a malformed model reply
rather than a deliberate choice triggered the run. Gated by
`EXECUTOR__FEDOT_FALLBACK`.

A switch rather than two siblings in the sequence because `AgentTool` returns
the LAST content-bearing event of the agent it wraps, and a stood-down sibling
still emits one — it would overwrite the executor's answer.

**Execution Flow (FedotAgent)**:
1. Receive task description
2. Build FEDOT.MAS pipeline
3. Connect to MCP servers
4. Execute pipeline
5. Return results

#### 6. TaskExecutorAgent

**Purpose**: Route a task to the execution path that can deliver it

**Architecture**:
- Type: `LlmAgent` (router — no tools of its own)
- Output Key: `executor_results`
- Subordinates (AgentTools):
  - `ToolPipelineAgent` (`SequentialAgent`: [ToolPreparerAgent,
    ExecutorSwitchAgent]) — runs the task with EXISTING MCP tools
  - `CoderAgent` — writes and runs code in the sandbox when no tool matches

**Routing**: engineering work goes straight to the coder; tool-shaped work goes
to the pipeline. If the pipeline abstains with `NO_MATCHING_TOOL` (see
`redirect_when_no_tools`), the router re-issues the same task to the coder
instead of bubbling the abstention up to the orchestrator.

---

## Data Flow

### User Query Flow

```
User Query
    │
    ▼
┌─────────────────┐
│ Parse & Validate│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Orchestrator    │
│ Decision Making │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌──────────┐
│Hypoth.│ │ Research │
│Agent  │ │  Agent   │
└───┬───┘ └────┬─────┘
    │          │
    └────┬─────┘
         │
         ▼
┌─────────────────┐
│  Tool Retriever │
│     Agent       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Experiment     │
│    Agent        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Aggregate &    │
│   Respond       │
└─────────────────┘
```


---

## Storage Layer

### Data Models

#### RetrievalFinalResult
```python
class RetrievalFinalResult(BaseModel):
    servers_id: List[str]  # Selected MCP server IDs
    queries: List[str]      # Queries used for retrieval
    task: str              # Original task description
```

#### RetrievalToolResult
```python
class RetrievalToolResult(BaseModel):
    tool: str              # Tool name
    server_id: str         # Server identifier
    description: str       # Tool description
    score: float          # Relevance score
```

### Storage Backends

| Component | Technology | Purpose |
|----------|------------|---------|
| Vector Store | ChromaDB | Paper embeddings |
| Tool Store | PostgreSQL | MCP server metadata |
| Document Store | S3 | PDF/image storage |
| Cache | In-Memory | Session data |

---

## Tool Integration

### Toolset Architecture

```
┌─────────────────────────────────────────┐
│           Toolset Registry              │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────────┐  ┌──────────────-─┐  │
│  │FedotMASToolset│  |WebSearchToolset│  │
│  └───────┬───────┘  └───────┬─────-──┘  │
│          │                  │           │
│          ▼                  ▼           │
│  ┌───────────────┐  ┌───────────────┐   │
│  │  FEDOT.MAS    │  │  Tavily MCP   │   │
│  │    Client     │  │    Client     │   │
│  └───────────────┘  └───────────────┘   │
│                                         │
│               ┌───────────────┐         │
│               │RetrievalToolset│        │
│               └───────┬───────┘         │
│                       │                 │
│                       ▼                 │
│               ┌───────────────┐         │
│               │  RAG Client   │         │
│               │   (HTTP)      │         │
│               └───────────────┘         │
│                                         │
└─────────────────────────────────────────┘
```

### MCP Server Communication

```python
# Tool Retrieval Flow
class RetrievalToolSet(BaseToolset):
    async def retrieve_tools(self, query: str) -> Dict:
        # 1. Embed query
        embedder = APIEmbedder(settings.api_embedding)
        query_embedding = embedder.embed(query)

        # 2. Search in vector store
        results = await vector_store.search(
            query_embedding,
            top_k=settings.rag.default_top_k
        )

        # 3. Rerank results
        reranker = HybridReranker(...)
        reranked = reranker.rerank(results)

        return {"tools": reranked}
```

---

## Configuration Management

### Settings Hierarchy

```
Settings
├── llm (LLMSettings)
│   ├── main_model
│   ├── service_key
│   ├── openai_api_key
│   └── ...
├── services (ServicesSettings)
│   ├── tavily_api_key
│   └── openalex_api_key
├── storage (StorageSettings)
│   ├── root_dir
│   ├── parse_results
│   └── ...
├── hosts_ports (HostsPortsSettings)
│   ├── chroma_host
│   ├── chem_services_host
│   └── ...
├── s3 (S3Settings)
│   ├── endpoint_url
│   ├── access_key
│   └── ...
└── tool_rag (ToolRAGSettings)
    ├── postgres
    ├── embedding
    └── ...
```

### Environment Variable Mapping

```
LLM__MAIN_MODEL          → settings.llm.main_model
SERVICES__TAVILY_API_KEY → settings.services.tavily_api_key
STORAGE__LOGGING_PATH    → settings.storage.logging_path
HOSTS_PORTS__CHROMA_HOST → settings.hosts_ports.chroma_host
S3__ENDPOINT_URL         → settings.s3.endpoint_url
```

---


## Monitoring and Observability

### Logging Architecture

```
Application Logs ──▶ Logger ──┬──▶ Console Handler
                             │
                             └──▶ File Handler ──▶ logs/app.log
```


## Appendix: Component Dependencies

```
CoScientist
├── agents/
│   └── agents.py
│       ├── Requires: google-adk
│       ├── Requires: litellm
│       └── Requires: config
├── chemical_utils/
│   └── Requires: Chemical Services API
├── config/
│   └── Requires: pydantic-settings
├── logging/
│   └── Built-in Python logging
├── paper_parser/
│   ├── Requires: marker
│   ├── Requires: langchain
│   └── Requires: S3/boto3
├── storage/
│   ├── Requires: rag_tools
│   └── Requires: pydantic
└── tools/
    ├── Requires: fedotmas
    ├── Requires: rag_tools
    └── Requires: google-adk tools
```

---

## Glossary

| Term | Definition |
|------|------------|
| MCP | Model Context Protocol - Standard for tool integration |
| RAG | Retrieval-Augmented Generation |
| FEDOT | Framework for Evolutionary Data Operations |
| ADK | Agent Development Kit |
| Toolset | Collection of related tools |
| Agent | Autonomous entity that can make decisions |

---

## References

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [FEDOT.MAS GitHub](https://github.com/ITMO-NSS-team/FEDOT.MAS)
- [RAG Tools GitHub](https://github.com/fiestaxxl/rag_tools)
- [LiteLLM Documentation](https://docs.litellm.ai/)
