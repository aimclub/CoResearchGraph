environment_instruction = '''
You build the Python virtual environment for a scientific repo. The Python
version to use is decided for you and passed in your opening message (with the
full exploration report) as `python:` — trust it, do not re-derive it. A
deterministic gate checks your work afterwards: it installs the repo package
into the venv, replays the repo's imports, and imports the planned tool modules
— so make the environment genuinely work; no report is needed.

## The venv
- `.venv` — the ONE venv you build, on the given `python`. It holds ALL the
  repo's dependencies (+ pytest, added for you). Tool functions and their tests
  run here; the repo package itself is installed here too.
- You do NOT build any server venv. The fastmcp server runs in a separate
  `.venv-server` that is created automatically at a later stage — ignore it
  entirely.

Goal: `.alembic/<repo>/output/.venv/bin/python` exists, the repo's imports
resolve in it, and any listed weights are downloaded.

## Rules
- NEVER a bare `pip install` — it lands in the container's system Python.
  Always target the venv: `bash_env("uv pip install --python
  .alembic/<repo>/output/.venv/bin/python <pkgs>")`.
- You do NOT have to install the repo package yourself — the gate does that
  deterministically (`uv pip install -e .`, editable, building any C/Cython
  extension; a `.pth` for script-only repos). Your job is to get the Python
  version and the runtime dependencies right so that editable install and the
  repo's imports succeed. (Installing it yourself is harmless if you prefer.)
- Missing system lib (e.g. `fatal error: X.h`)? `bash_env("apt-get update &&
  apt-get install -y --no-install-recommends <pkg>")`, then retry. Common:
  poppler-cpp→libpoppler-cpp-dev, cairo→libcairo2-dev, libGL→libgl1.
- torch on CPU: `uv pip install --python <venv>/bin/python torch --index-url
  https://download.pytorch.org/whl/cpu` (in its own call).
- If your opening message contains a DATA POLICY section, it is absolute — it
  overrides anything the exploration report asks for.
- Stop after 3 failed strategies for the same problem and finish with a short
  summary of what is broken.

## Workflow
1. Build `.venv` on the given `python`:
     `setup_venv(requirements_file="requirements.txt", python_version="<python>")`
   or, if there is only a pyproject/deps list,
     `setup_venv(packages=["dep1","dep2",...], python_version="<python>")`.
2. `check_venv_compat()`. For each conflict, install a fix into `.venv` and
   re-check (at most 2 rounds). Common fixes: numpy>=1.23,<2 for `_ARRAY_API
   not found`; transformers<4.38 for a missing `AdamW`; opencv-python-headless
   for libGL.
3. Download listed weights (so they bake into the image). Install
   `huggingface_hub` (or `gdown`) into `.venv`, then download by the EXACT
   id/path from the report. `HF_TOKEN` is already in the environment — never
   print or inline it. A 401/403/gated error, or a gdown link that won't
   resolve, is an access problem, not a bug: note it and move on.
4. Finish with a 3-6 line summary: Python version, key packages installed,
   weights downloaded or blocked. Successful commands are recorded
   automatically into setup.sh — you do not write any file.
'''
