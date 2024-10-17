#!/bin/sh

set -e

cd /app/gigmate

echo 'api { credentials {"access_key": "<KEY>", "secret_key": "<KEY>"} }' > ~/clearml.conf

# Install Poetry
pip install poetry

poetry config virtualenvs.create false

apt-get update
apt-get install -y build-essential
poetry install --with dataset
# pip install audiomentations

# Install dependencies
poetry install

# Install PyTorch from pre-releases
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Run the training module
poetry run python -m gigmate.data_preprocessing.pipeline