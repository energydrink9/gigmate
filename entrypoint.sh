#!/bin/sh

set -e

cd /app/gigmate

# Install Poetry
pip install poetry

# Install dependencies
poetry install

# Duplicate clearml.conf setup (this might be redundant; consider removing if already set during build)
echo 'api { credentials {"access_key": "<KEY>", "secret_key": "<KEY>"} }' > ~/clearml.conf

# Install PyTorch from pre-releases
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Install the package
python -m pip install .

# Run the training module
python -m gigmate.training.train