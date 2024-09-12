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


Jupyter

```
!pip install clearml
!pip install torch
!pip install torchinfo
!pip install torchmetrics
!pip install git+https://github.com/Yikai-Liao/symusic
!pip install git+https://github.com/Natooz/MidiTok
!pip install scikit-learn
!pip install lightning
```

### Auth

CLI:

Run the following command to authenticate.

```sh
source clearml-auth.sh
```