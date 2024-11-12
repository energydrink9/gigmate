#!/bin/sh

cd /app/gigmate

echo 'api { credentials {"access_key": "<KEY>", "secret_key": "<KEY>"} }' > ~/clearml.conf

# Install Poetry
pip3 install poetry || true

#poetry config virtualenvs.create false

apt-get update || true
apt-get install -y build-essential || true

echo "Installing dependencies"
poetry install --no-interaction --no-ansi || true

# Install PyTorch from pre-releases and run the training module
echo "Installing latest PyTorch"
poetry run pip install --upgrade --force-reinstall --pre torch torchaudio torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 || true

poetry run pip install "numpy==1.26.4"

echo "Running script"
poetry run --no-interaction --no-ansi python -m gigmate.domain.predict