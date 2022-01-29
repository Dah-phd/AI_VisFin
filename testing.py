from tensorflow.keras import models
import modules.test as test


def main(model: str):
    tester = test.Tester(model)
    tester.test()
    print(f'Model full test: {model}')
    print(tester.overview())

    tester = test.Tester(model, pure_test=True)
    tester.test()
    print(f'Model pure test: {model}')
    print(tester.overview())


if __name__ == '__main__':
    main(
        model='VisFin_v6_20drop0003lr.h5',
    )
