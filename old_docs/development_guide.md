# Welcome to Contributing

## Table of Contents
- [Development Setup](#development-setup)
- [Run the Project](#run-the-project)
- [Branching Rules](#branching-rules)
- [Code Review](#code-review-and-pull-request-guidelines)
- [Code Style](#code-style-and-standards)
- [Tests](#how-to-run-tests)
- [Reporting Issues](#reporting-issues)

## Development Setup
Run ChemCoScientist locally:

**Prerequisites:** requires Python >=3.12,<3.13, Poetry >=2

1. Clone the repository to a local directory of your choice.
2. Create a new environment and install dependencies:
```commandline
poetry install
poetry run pip install --no-deps git+https://github.com/aimclub/ProtoLLM.git@main
```
3. Create a `config.env` file in the root of the project based on [example_config.env](../example_config.env).
4. Turn on ITMO VPN to get access to all necessary services (ChromaDB, embedding and reranker services, MinIO (S3), AutoML, and generative models).

## Run the Project 
**Run in CLI:**
1. Add a new query in [main_cli.py](../ChemCoScientist/main_cli.py), e.g.:
```
inputs = {"input": "Generate an image of spherical nanoparticles."}
```
2. Run [main_cli.py](../ChemCoScientist/main_cli.py) (it will execute the new query)

**Run the GUI (Streamlit app):**
1. Run `streamlit run ChemCoScientist/streamlit_app.py`
2. The app will be available at http://localhost:8501

## Branching Rules
This project uses a simplified version of Git Flow. The development branch is `main`.
Direct pushes to main are not allowed. To contribute, always create a new branch following the prefix rules:
- feature/ — for new features or enhancements
- bug/ — for bug fixes
- hotfix/ — for urgent fixes to production issues

Optionally include ticket or issue numbers for traceability, e.g., bug/123-fix-login-error.

Workflow:
- Always rebase your branch onto main before opening a PR.
- Write meaningful commit messages tied to your branch purpose.
- Delete branches after they are merged to keep the repo clean.

## Code Review and Pull Request Guidelines
- Changes must be merged to main via Pull Requests after code review.
- PR requires at least 1 approval to merge.
- All comments must be answered.
- Comment authors resolve their comments.
- Squash all commits into one before merging.
- PR author merges the PR.

## Code Style and Standards
Code formatting is maintained using Black with its standard settings. See [black.yml](../.github/workflows/black.yml).

## How to Run Tests
The complete suite of tests can be found in the [tests folder](../tests). Please note that the integration tests depend on auxiliary services hosted on the ITMO servers and therefore require VPN access.
Execute the tests either directly within the IDE or via the command line interface using: `pytest tests`.

## Reporting Issues
Please create a new task on the [board](https://github.com/orgs/ITMO-NSS-team/projects/24).
