from train import train_model
from test import test_model

def init():
    print('Starting model training')
    model = train_model()
    print('Model training complete')

    print('Starting model evaluation')
    test_model(model)
    print('Model evaluation complete')

if __name__ == '__main__':
    init()