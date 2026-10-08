FROM python:3.12-slim

WORKDIR /app

ENV POETRY_VIRTUALENVS_CREATE=false

# Install system packages
RUN apt-get update && \
    apt-get install -y build-essential vim git && \
    apt-get clean

# Install poetry
RUN pip install poetry && \
    export PATH="/root/.local/bin:$PATH"

# Install project dependencies
COPY pyproject.toml .
RUN poetry --no-interaction --no-ansi --no-root install

# Install ProtoLLM in a separate call so updating the repo will not require full dependency reinstall
RUN poetry run pip install --no-cache --no-deps git+https://github.com/aimclub/ProtoLLM.git@main

# Copy project files afterwards so changing project file will not require full image rebuild
COPY ChemCoScientist/ ChemCoScientist/
COPY CoScientist/ CoScientist/
COPY definitions.py .
COPY config.env .

EXPOSE 8501

CMD ["streamlit", "run", "ChemCoScientist/streamlit_app.py"]