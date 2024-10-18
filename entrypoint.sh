#!/bin/sh

set -e

cd /app/gigmate

echo 'api { credentials {"access_key": "<KEY>", "secret_key": "<KEY>"} }' > ~/clearml.conf

# Install Poetry
pip3 install poetry

poetry config virtualenvs.create false

apt-get update
apt-get install -y build-essential
poetry install --with dataset

# Install dependencies
poetry install

# Install PyTorch from pre-releases and run the training module
pip3 install --upgrade --force-reinstall --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
python -m gigmate.domain.predict