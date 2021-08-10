from tensorflow import keras, expand_dims
import numpy as np
from sys import argv
from modules.data_feed import *
DATABASE = PrepareData('daily__us__nasdaq_stocks.csv', '<DATE>', skip=5)
DATABASE.clean_db()
MODEL_NAME = argv[1]
MODEL = keras.models.load_model(MODEL_NAME)
TICKS = int(argv[2])


def main():
    print(f'Training {MODEL_NAME}!')
    for tick in range(TICKS):
        data = DATABASE.mass_hacking(tick=tick, horizon=84)
        tr_data = expand_dims(np.array(data[0]), axis=-1)
        tr_key = np.array(data[1])
        if tr_data.shape[0] > 0:
            MODEL.fit(
                tr_data,
                tr_key,
                epochs=10,
                verbose=1,
                shuffle=True)
    MODEL.save(MODEL_NAME)
    print(f'{MODEL_NAME} SAVED!')


if __name__ == '__main__':
    main()
