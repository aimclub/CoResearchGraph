# Project structure

[ChemCoScientist](https://github.com/ITMO-NSS-team/CoScientist/tree/main/ChemCoScientist) is an AI assistant designed 
specifically for chemists. It operates as a multi-agent system built on the LangGraph framework, based on a graph 
structure composed of nodes/agents and edges. Each node performs a single function, while agents encompass multiple 
tools and dynamically select the most appropriate ones for a given task.

The project leverages the graph infrastructure implemented in [ProtoLLM](https://github.com/aimclub/ProtoLLM/tree/main/protollm/agents) 
and includes core components such as the graph builder and a set of universal agents: Web Search Node, Supervisor Node, 
Plan Node, Replan Node, Summary Node, and Chat Node.

ChemCoScientist extends this foundation by introducing several new agents, detailed below. Additionally, 
it incorporates select agents from Smolagents, including CodeAgent, LiteLLMModel, and OpenAIServerModel.
It also utilizes modules and agents from the [CoScientist](https://github.com/ITMO-NSS-team/CoScientist/tree/main/CoScientist) 
library.

The workflow is depicted in the image below. Agents highlighted in green are implemented in ProtoLLM, those in pink 
belong to ChemCoScientist, and agents shown in yellow are from CoScientist.

![Multi-Agent System](../diagram.png)

## Agents/nodes

- **Dataset Builder Agent**
    - **Description**: Generates a dataset for a specified chemical task using external data sources and a code agent, with support for ChEMBL and BindingDB.

    - **Types of requests**:

- **ML/DL Agent**
    - **Description**: An agent equipped with tools for training machine learning models, monitoring their training status, predicting molecular properties using trained models, and generating novel molecules.

    - **Tools**:
        - `get_state_from_server` - Facilitates monitoring the readiness and state of different model deployments.
        - `get_case_state_from_server` - Allows checking the progress or result of a model training or prediction process.
        - `predict_prop_by_smiles` - Predicts molecular properties using pre-trained machine learning models.
        - `just_ml_training` - Trains a machine learning model for predicting molecular properties.
        - `generate_mol_by_case` - Generates molecules based on a specified model and quantity.
        - `run_ml_dl_training_by_daemon` - Trains models for predicting chemical properties or classifying molecules based on input features.

    - **Types of requests**:

- **Chemist Node**
    - **Description**: An agent with tools for retrieving biochemical data, performing molecular property predictions, SMILES and name conversions, executing code, and visualizing molecular structures.

    - **Tools**:
        - `fetch_BindingDB_data` - Retrieves protein-ligand binding affinity data from BindingDB.
        - `fetch_chembl_data` - Retrieves compound activity data for a specified protein target from the ChEMBL database.
        - `python_repl_tool` - Used to perform calculations or execute Python code.
        - `calc_prop_tool` - Predicts a molecular property based on its SMILES representation.
        - `name2smiles` - Convert a molecule name to its SMILES representation.
        - `smiles2name` - Converts a SMILES string representing a molecule into its IUPAC name.
        - `smiles2prop` - Calculate molecular properties from a SMILES string or IUPAC name.
        - `visualize_molecule` - Visualizes a molecule from its SMILES representation and saves the 3D structure as an HTML file.

    - **Types of requests**:

- **Nanoparticle Node**
    - **Description**: An agent specialized in nanoparticle synthesis, property prediction, shape analysis, and image generation/interpretation.

    - **Tools**:
        - `synthesis_generation` - Generates a detailed synthesis procedure based on a given description of the nanoparticles.
        - `predict_nanoparticle_entrapment_eff` - Predicts the entrapment efficiency of a nanomaterial based on its descriptive text.
        - `predict_nanoparticle_shape` - Predicts the shape of a nanomaterial from its descriptive text.
        - `generate_nanoparticle_images` - Generates an image representation of a nanoparticle based on its specified shape.
        - `analyse_nanoparticle_images` - Analyzes images to determine the shape of nanoparticles present.

    - **Types of requests**:

- **Paper Analysis Agent**
    - **Description**: This agent analyzes scientific papers to respond to user queries. It leverages a vector database (ChromaDB) containing chemical papers alongside user-uploaded documents to deliver well-informed answers by extracting relevant information.

    - **Tools**:
        - `explore_chemistry_database` — Provides answers based on information retrieved from a database of chemical papers (based on RAG).
        - `explore_my_papers` — Generates responses using information extracted from user-uploaded papers.
        - `select_papers` — Identifies and selects relevant papers from the database corresponding to the query.

    - **Types of requests**:
        - Load papers and inquire about their content (e.g., `What reaction is depicted in Figure 3?`).
        - Pose general chemistry questions (e.g., `What components are involved in the synthesis of BASHY dyes, and what are their applications?`).
        - Request the assistant to locate papers on a specific topic (e.g., `Find papers on the synthesis of Glionitrin A/B.`).

## Add new agents/nodes
- Add the agent/node implementation to [agents.py](../ChemCoScientist/agents/agents.py)
- Add tools for the new agent in a new file in [tools](../ChemCoScientist/tools)
- Edit config in [create_conf.py](../ChemCoScientist/conf/create_conf.py):
    - Add the agent/node description to `additional_agents_description`
    - Add the agent/node to `conf['configurable']['scenario_agents']` and 
      `conf['configurable']['scenario_agent_funcs']`
    - Add tools for agent in `conf['configurable']['tools_for_agents']`
    - If necessary pass additional arguments to agent/node in `conf['configurable']['additional_agents_info']`
    - If necessary pass additional prompts to universal agents from ProtoLLM in `conf['configurable']['additional_agents_info']` (they will override the default prompts)

Agents communicate via a shared state, whose structure is defined within ProtoLLM. 
Modifications to this library can be made as needed.

## Project Structure

- **ChemCoScientist** - provides the core functionality for building an AI-powered scientific assistant focused on chemistry
    - `agents` - implementation of LLM agents tailored for diverse scientific tasks in chemistry
    - `conf` - configuration files
    - `dataset_handler` - module responsible for downloading data from ChEMBL
    - `frontend` - Streamlit-based frontend application providing the project’s interactive user interface
    - `paper_analysis` - module implementing RAG to answer queries based on scientific publications in chemistry
    - `tools` - collection of tools utilized by the agents
- **CoScientist** - a library for developing AI-driven scientific computing applications (soon it will be moved to separate repo)
    - `paper_parser` - a module for parsing scientific papers using marker-pdf, performing post-processing, and storing processed data in S3
    - `scientific_agents` - implementation of LLM agents tailored for diverse scientific tasks in different fields
- **docker** - Dockerfiles to build and run ChemCoScientist
- **docs** - project documentation detailing architecture, usage, and development guidelines
- **infrastructure** - scripts for building and deploying supplementary services supporting the project
    - `automl`
    - `chroma` - scripts to run ChromaDB, embedding and reranking services
    - `generative_models`
- **metrics** - scripts to measure the effectiveness of the paper_analysis module, assessing context extraction quality and final answer accuracy from the LLM
- **tests**
    - `integration` - integration tests for the paper_analysis module