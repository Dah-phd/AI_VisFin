from tensorflow import keras
import numpy as np
from sys import argv
from modules.data_feed import DataSet
MODEL_NAME = argv[1]
MODEL = keras.models.load_model(MODEL_NAME)
TICKS = int(argv[2])


def main():
    print(f'Training {MODEL_NAME}!')
    data = DataSet('filtered_data_e.csv')
    MODEL.fit(
        data.data_set,
        data.data_keys,
        epochs=10,
        verbose=1,
        shuffle=True)
    MODEL.save(MODEL_NAME)
    print(f'{MODEL_NAME} SAVED!')


if __name__ == '__main__':
    main()
