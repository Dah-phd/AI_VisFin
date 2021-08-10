from operator import index
from tensorflow.keras import models
import modules.test as test


def main(model: str, data: str, index_col: str or None = None, skip: int = 0):
    tester = test.Tester(model, data, index_col=index_col, skip=skip)
    tester.test()
    print(f'Model: ')
    for res in tester.overview():
        print(res)


if __name__ == '__main__':
    main(
        model='VisFin_v5.h5',
        data='daily__us__nasdaq_stocks.csv',
        index_col='<DATE>',
        skip=5)
