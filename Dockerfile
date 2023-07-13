FROM --platform=linux/amd64 condaforge/mambaforge:latest

WORKDIR /app
COPY . /app

RUN mamba env create -f environment.yml

SHELL ["conda", "run", "-n", "chromactivity_env", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "-n", "chromactivity_env", "/bin/bash", "-i"]
