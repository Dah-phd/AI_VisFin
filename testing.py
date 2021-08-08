from tensorflow.keras import models
import modules.test as test


def main(model: str, data: str):
    tester = test.Tester(model, data)
    tester.test()
    print(f'Model: ')
    for res in tester.overview():
        print(res)


if __name__ == '__main__':
    main(model='VisFin_v5.h5', data='modules/first_test_data.csv')
