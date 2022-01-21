from tensorflow.keras import models
import modules.test as test


def main(model: str):
    tester = test.Tester(model)
    tester.test()
    print(f'Model full test: {model}')
    for res in tester.overview():
        print(res)

    tester = test.Tester(model, pure_test=True)
    tester.test()
    print(f'Model pure test: {model}')
    for res in tester.overview():
        print(res)


if __name__ == '__main__':
    main(
        model='VisFin_v6_nodrop00007lr.h5',
    )
