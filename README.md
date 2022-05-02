# Welcome to the `python-template` project

+ Project managed by: [OTO team](https://github.com/orgs/Unity-Technologies/teams/oto)
+ Public slack channel: [#devs-oto](https://unity.slack.com/messages/C02G0QXQK42/)
+ Github action statuses:
  + ![latest release](https://github.com/Unity-Technologies/python-template/workflows/release/badge.svg)
  + ![latest status](https://github.com/Unity-Technologies/python-template/workflows/push/badge.svg)

## Table of Contents

+ [📚 Summary](#-Summary)
+ [📦 Package management](#-Package-management)
+ [🏝 Virtual environment](#-Virtual-environment)
+ [🪛 Installation](#-Installation)
+ [📍 Dependencies update](#-Dependencies-update)
+ [🧪 Tests](#-Tests)
+ [🐳 Docker](#-Docker)
+ [🪝 pre-commit hooks](#-pre-commit-hooks)
+ [🚚 Release](#-Release)

## 📚 Summary

This python project template includes the following features:

+ 📦 `poetry` python package configuration
+ 🧪 `pytest` testing framework
+ 🐍 `conda` configuration to manage the python version, the virtual environment & system dependencies
+ 🐳 `Dockerfile` to build a container image
+ 🎬 github action workflows:
  + ⬆️ push workflow:
    + 🚔 linting tests:
      + 🔍 code linting with `flake8`, `pylint` & `mypy`
      + ⚫️ code formatting with `black`
      + 📤 imports order with `isort`
      + 📃 docstring style with `pydocstyle`
      + 🔐 security linting with `bandit`
    + 1️⃣ unit tests
    + ⛓ functional tests
  + 🚚 release workflow:
    + ✅ verifies the release tag and the python project version
    + 🐳 publishes docker image to our [docker registry](https://console.cloud.google.com/artifacts/python/unity-oto-ml-stg/us-central1/docker-registry/python-template)
    + 🐍 publishes python package to our [python registry](https://console.cloud.google.com/artifacts/python/unity-oto-ml-stg/us-central1/python-registry/python-template)
+ 🪝 `pre-commit` hooks configuration matching the github action test workflow

## 📦 Package management

This template uses `poetry` for packaging and dependency management.
Check its [documentation](https://python-poetry.org/docs/master/) for more information.

To install `poetry`, use the following command:

```shell
curl -sSL https://install.python-poetry.org | python3 - -p
```

Because we want to leverage the latest plugin functionalities of `poetry`, we are currently installing the `1.2.0` preview version. Hence, make sure that:

```shell
poetry --version
```

returns something above or equal to `1.2.0a2`.

`poetry>=1.2.0a2` can use keyring to look up GCP credentials and log in to [Google Artifact Registry](https://cloud.google.com/artifact-registry), therefore make sure to install the following plugin to enable this feature:

```shell
poetry plugin add keyrings.google-artifactregistry-auth
```

This plugin will enable the resolution of private dependencies as well as publishing `python-template`  both from/to the Artifact Registry.

*Notes:*

- On certain Linux distributions, the Google Python Auth backend might not have the highest priority which would prevent poetry from access the Artifact Registry. A solution is to set it as the default keyring backend using::

```shell
mkdir $HOME/.config/python_keyring
echo $'[backend]\ndefault-keyring=keyrings.gauth.GooglePythonAuth' > $HOME/.config/python_keyring/keyringrc.cfg
```

- `poetry` version `1.1.x` can also be used, but it will not be able to utilize the keyring library (see [dependencies update](#-Dependencies-update)).

## 🪛 Installation

To install the `python-template` project, we first need to clone the repository:

```shell
git clone https://github.com/Unity-Technologies/python-template.git
```

or alternatively using the `gh` [CLI](https://cli.github.com/):

```shell
gh repo clone Unity-Technologies/python-template
```

To complete the installation, we recommend using a [🏝 Virtual environment](#-Virtual-environment) in conjunction with poetry. Nevertheless, `template` can also be installed with:

```shell
pip install python-template/
```

although keep in mind that `pip` will not provide an editable install and will not use the locked dependencies stored in `poetry.lock`.

## 🏝 Virtual environment

To install the `python-template`, a dedicated virtual environment should be created.

We recommend installing `conda` using the [`miniforge`](https://github.com/conda-forge/miniforge) installer.
Create a `conda` virtual environment with the following command:

```shell
cd python-template/
conda env create -f conda.yml
conda activate template
```

The `python-template` python package can then be installed using the following command:

```shell
poetry install
```

## 📍 Dependencies update

This `python-template` project uses our GCP python Artifact Registry as the main package source and falls back to pypi.org when the dependency is not found.
To update the project dependencies run the command:

```shell
poetry update
```

If the commands fails to use keyring to access GCP credentials, try to authenticate using the `gcloud` CLI:

```shell
gcloud auth application-default login
```

Alternatively, authentication to GCP Artifact Registry can also be achieved using the following environment variables:

```shell
export POETRY_HTTP_BASIC_GCP_USERNAME="_json_key" \
export POETRY_HTTP_BASIC_GCP_PASSWORD="$(</path/to/gcp_key.json)"
```

## 🧪 Tests

The following command line allows to test code & docstring formatting as well as code & security linting:

```shell
poetry run pytest --bandit --black --flake8 --pylint --isort --pydocstyle --mypy
```

The unit tests can be run using:

```shell
poetry run pytest --cov-fail-under 100 --cov-config pyproject.toml --cov=template -m unit_test
```

Additionally, functional tests can be run with:

```shell
poetry run pytest --cov-fail-under 100 --cov-config pyproject.toml --cov=template -m functional_test
```

## 🐳 Docker

### 🏗 Building the image

First, make sure to clean any non versioned files as the content of the root directory will be copied over to the image:

```shell
git clean -dfx
```

To be able to install the dependencies hosted in our GCP artifact registry, we will need a JSON key  encoded in base64 to allow the Google Cloud SDK to log in a service account.

The `docker` image can then be built & run using:

```shell
base64 path/to/gcloud-json-key -w 0 > /path/to/gcloud-json-base64-key
DOCKER_BUILDKIT=1 docker build --secret id=gcp_credentials,src=/path/to/gcloud-json-base64-key -t python-template:latest .
docker run -it --rm python-template:latest
```

### ⬇️ Pulling the image

The `python-template` docker image can be directly pulled from our Google Artifact Registry using a service account JSON key file to login:

```shell
gcloud auth activate-service-account --key-file=/path/to/gcloud-key.json
gcloud auth configure-docker us-central1-docker.pkg.dev
docker run -it --rm us-central1-docker.pkg.dev/unity-oto-ml-stg/docker-registry/python-template:latest
```

## 🪝 pre-commit hooks

Hooks can be installed with the command:

```shell
pre-commit install -t pre-commit -t prepare-commit-msg
```

The `pre-commit` hooks specified in `.pre-commit-config.yaml` will be run on every single git commit commands.
They can additionally be run on demand with:

```shell
pre-commit run --all-files
```

## 🚚 Release

Each github release will trigger a github action `release` workflow that will publish the following artifacts to our Google Artifact Registry:

+ [🐍 python package](https://console.cloud.google.com/artifacts/python/unity-oto-ml-stg/us-central1/python-registry/python-template)
+ [🐳 docker image](https://console.cloud.google.com/artifacts/python/unity-oto-ml-stg/us-central1/docker-registry/python-template)

Both of these artifacts are automatically published to our `docker` & `python` Google Artifact Registries.
The github action `release` workflow will make sure that the github release tag matches the python package version before attempting any uploading.

# Converting to public repository
Any and all Unity software of any description (including components) (1) whose source is to be made available other than under a Unity source code license or (2) in respect of which a public announcement is to be made concerning its inner workings, may be licensed and released only upon the prior approval of Legal.
The process for that is to access, complete, and submit this [FORM](https://docs.google.com/forms/d/e/1FAIpQLSe3H6PARLPIkWVjdB_zMvuIuIVtrqNiGlEt1yshkMCmCMirvA/viewform).
