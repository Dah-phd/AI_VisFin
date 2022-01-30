from tensorflow.python.keras.backend import exp
from .data_feed import DataSet
from tensorflow import keras, expand_dims
import numpy as np
import pandas as pd
import json


class Tester:
    def __init__(self,
                 vis_fin_model: str,
                 data_csv: str = 'filtered_data_e.csv',
                 skip: int = 0,
                 test_days: int = 5,
                 pure_test: bool = False) -> None:
        self.model = keras.models.load_model(vis_fin_model)
        self.model_name = vis_fin_model
        self.pure_test = '_pure' if pure_test else ''
        self.file_for_predictions = \
            f'{vis_fin_model[:-3]}{self.pure_test}_test_results.csv'
        # for five days
        if pure_test:
            skip = test_days
        self.data = DataSet(data_csv, load_pure_test=pure_test, skip=skip)
        # results
        self.basic_stats = False
        self.predictions = []
        self.graphs = []
        self.expected = []

    def test(self):
        if self.data.generate(0, 3000):
            self.predictions = self.model.predict(self.data.data_set)
            print(self.predictions.shape)
            self.expected = self.data.data_keys
        pd.DataFrame({'Expected': list(self.expected), 'Results': list(self.predictions)}).to_csv(
            self.file_for_predictions, index=False)

    def _test_vals(self) -> dict:
        match = 0
        snd_match = 0
        direction_match = 0
        for i in range(self.predictions.shape[0]):
            to_list = list(self.predictions[i])
            predict_i = to_list.index(max(to_list))
            snd_predict_i = self._second_largest(to_list, predict_i)
            expect_i = list(self.expected[i]).index(
                max(self.expected[i]))
            if predict_i == expect_i:
                match += 1
            elif (predict_i < 3 and expect_i < 3) or \
                    (predict_i > 3 and expect_i > 3):
                direction_match += 1
            if predict_i == 3 and expect_i == 3:
                direction_match += 1
            if snd_predict_i == expect_i:
                snd_match += 1
        return {
            'Model': self.model_name + self.pure_test,
            'Total tests': self.predictions.shape[0],
            'Exact match': match,
            'Matched by second highest': snd_match,
            'Match gain or loss': direction_match,
            'Exact match rate': match/self.predictions.shape[0],
            'Gain/Loss match rate': (match+direction_match)/self.predictions.shape[0],
            'Random match rate': self.coin_test[2],
            'Random direction match rate': self.coin_test[3]/self.coin_test[1],
            'Random match test params': f'Matches ({self.coin_test[0]})/N({self.coin_test[1]})',
        }

    def store_results(self):
        file_name = f'{self.model_name}{self.pure_test}_results.json'
        with open(file_name, 'a+') as data_saver:
            data_saver.write(json.dumps(self.basic_stats))

    def overview(self) -> dict:
        if self.basic_stats:
            return self.basic_stats
        else:
            self.coin_test = self._coin_flip_equivalent()
            self.basic_stats = self._test_vals()
            self.store_results()
            return self.basic_stats

    @ staticmethod
    def _second_largest(categorical_ls: list, ls_max: int):
        ls = categorical_ls.copy()
        ls[ls_max] = 0
        return ls.index(max(ls))

    @ staticmethod
    def _coin_flip_equivalent(n: int = 10000):
        rand_ls_1 = np.random.randint(0, 8, n)
        rand_ls_2 = np.random.randint(0, 8, n)
        match = 0
        direction = 0
        for r1, r2 in zip(rand_ls_1, rand_ls_2):
            if r1 == r2:
                match += 1
            elif r1 == 3 and r2 == 4:
                match += 1
            elif r2 == 3 and r1 == 4:
                match += 1
            if r1 < 3 and r2 < 3:
                direction += 1
            elif r1 == 3 and r2 == 3:
                direction += 1
            elif r1 > 4 and r2 > 4:
                direction += 1
        return (match, n, match/n, direction, direction/n)
