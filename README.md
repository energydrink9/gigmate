![example workflow](https://github.com/energydrink9/gigmate/actions/workflows/python-app.yml/badge.svg)


### Initial setup

In order to start using the application, make sure to run the following commands that will install the package and the necessary dependencies.

```
pip install poetry
poetry install
poetry install -E dataset
```

#### Environment Variables

If you want to run training or dataset generation (for inference it's not needed), make sure the following environment variables are set before running the application (replace the <> tokens with the actual secrets):

```sh
export CLEARML_API_ACCESS_KEY=<YOUR_CLEARML_API_ACCESS_KEY_HERE>
export CLEARML_API_SECRET_KEY=<YOUR_CLEARML_API_SECRET_KEY_HERE>
```

Also this one in case you are running on Apple Silicon:
```sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Inference

```sh
poetry run python -m gigmate.api.complete_audio # To start the API
poetry run python -m gigmate.play # To start the client
```

### Dataset generation
Please refer to the following repository for dataset generation information: https://github.com/energydrink9/stem_continuation_dataset_generator

### Training

In order to start a training run, make sure to have a dataset available and then install the [latest nightly version](https://pytorch.org/get-started/locally/) of PyTorch. On Mac you can use the following command:
```
pip3 install --pre torch torchvision torchaudio torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
```

After that, run the following command to start training:

```
python -m gigmate.train
```
