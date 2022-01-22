import tensorflow as tf
import numpy as np
from sys import argv
import os
from modules.data_feed import DataSet
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
MODEL_NAME = argv[1]
FEED_SPEED = int(argv[2])
VALIDATION_SPLIT = float(argv[3])
MODEL = tf.keras.models.load_model(MODEL_NAME)


def main_by_generator(validate=None):
    print(f'Training {MODEL_NAME}!')
    data_gen = DataSet('filtered_data_e.csv', False)
    n = 0
    csv_logger = \
        tf.keras.callbacks.CSVLogger('log.csv', separator=',', append=True)
    while True:
        data_gen.generate(n*FEED_SPEED, n*FEED_SPEED+FEED_SPEED)
        if data_gen.shape[0] == 0:
            print('data gen in empty ... finishing script!')
            break
        print(
            f'Training Session {n+1}, with data set of shape: {data_gen.shape}'
        )
        tf.keras.backend.clear_session()
        if validate:
            print(f'Using validation split: {validate}')
            MODEL.fit(
                data_gen.data_set,
                data_gen.data_keys,
                batch_size=1000,
                epochs=10,
                verbose=1,
                validation_split=validate,
                shuffle=True,
                callbacks=[csv_logger])
        else:
            MODEL.fit(
                data_gen.data_set,
                data_gen.data_keys,
                batch_size=1000,
                epochs=10,
                verbose=1,
                shuffle=True,
                callbacks=[csv_logger])
        MODEL.save(MODEL_NAME)
        n += 1
        print(f'{MODEL_NAME} SAVED!')
    log_name = f'zlog_{MODEL_NAME.split(".")[0]}.csv'
    os.rename('log.csv', log_name)
    print(f'Training stats saved in file {log_name}')


if __name__ == '__main__':
    main_by_generator(VALIDATION_SPLIT)
