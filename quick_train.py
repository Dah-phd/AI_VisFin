import tensorflow as tf
import numpy as np
from sys import argv
from modules.data_feed import DataSet
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
MODEL_NAME = argv[1]
FEED_SPEED = int(argv[2])
MODEL = tf.keras.models.load_model(MODEL_NAME)


def main_by_generator():
    print(f'Training {MODEL_NAME}!')
    data_gen = DataSet('filtered_data_e.csv', False)
    n = 0
    while True:
        if data_gen.generate(n*FEED_SPEED, n*FEED_SPEED+FEED_SPEED):
            print(data_gen.shape)
            print(f'Training Session {n+1}')
            tf.keras.backend.clear_session()
            MODEL.fit(
                data_gen.data_set,
                data_gen.data_keys,
                batch_size=1000,
                epochs=1,
                verbose=1,
                shuffle=True)
            MODEL.save(MODEL_NAME)
            n += 1
            print(f'{MODEL_NAME} SAVED!')
        else:
            break
    print('===== ending training session =====')


if __name__ == '__main__':
    main_by_generator()
