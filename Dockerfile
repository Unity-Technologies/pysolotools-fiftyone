# syntax=docker/dockerfile:1.2
FROM condaforge/miniforge3

# Copy python project
ADD . /workspace/
WORKDIR /workspace

# Create template conda environment
RUN conda env create -f conda.yml
RUN echo "source activate template" >> ~/.bashrc

# Switch shell to bash and pre-activate venv
SHELL ["conda", "run", "-n", "template", "/bin/bash", "-c"]

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python - -p
ENV PATH /root/.local/bin:$PATH

# Install poetry project using GCP credentials
RUN --mount=type=secret,id=gcp_credentials \
    export POETRY_HTTP_BASIC_GCP_USERNAME="_json_key_base64" \
    && export POETRY_HTTP_BASIC_GCP_PASSWORD="$(</run/secrets/gcp_credentials)" \
    && poetry install
