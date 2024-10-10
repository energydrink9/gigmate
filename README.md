### Development

Setup:
```
python -m pip install -e .
```

Train
```
python -m gigmate.train
```

Test
```
python -m gigmate.test
```

Create dataset
```
python -m gigmate.data_preprocessing.pipeline
```


### Run

```sh
python -m pip install .
python -m gigmate.train # To start training
python -m gigmate.test # To start evaluation
python -m gigmate.api.complete_audio # To start the API
python -m gigmate.play # To start the client
```

### Environment Variables

CLI:

Make sure the following environment variables are set before running the application (replace the <> tokens with the actual secrets):

```sh
export CLEARML_API_ACCESS_KEY=<YOUR_CLEARML_API_ACCESS_KEY_HERE>
export CLEARML_API_SECRET_KEY=<YOUR_CLEARML_API_SECRET_KEY_HERE>
```

Also this one in case you are running on Apple Silicon:
```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```