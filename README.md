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

Source directory should contain a folder for each artist. The artist folders should contain one compressed file for each song. The compressed file should contain one .wav file for each stem. In order to identify the different stems, the stem files should have names containing the values present at `src/gigmate/data_preprocessing/steps/merge.py` in the `STEM_NAMES` variable.

#### Dataset preparation

To prepare the dataset, use the following command, replacing the string <SOURCE-DIRECTORY> with the path to the directory containing the source files:

```sh
poetry run python -m gigmate.data_preprocessing.prepare <SOURCE-DIRECTORY>
```

The pipeline will uncompress the song archives and convert all the files to OGG format. The original files will be deleted and the prepared dataset will be uploaded to the ClearML repository.

#### Dataset creation

To process the files obtained at the preparation step, use the following command, replacing the string <PREPARED-DATASET-ID> with the id of the pre-processed dataset obtained from the previous step:

```sh
poetry run python -m gigmate.data_preprocessing.process <PREPARED-DATASET-ID>
```

The pipeline will augment, distort, encode and split the samples into chunks, generating three different folders for the train, validation and test sets. The result will be uploaded to ClearML into 3 different datasets.

### Training

In order to start a training run, make sure to have a dataset available and then install the [latest nightly version](https://pytorch.org/get-started/locally/) of PyTorch. On Mac you can use the following command:
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

After that, run the following command to start training:

```
python -m gigmate.train
```
