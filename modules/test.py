from tensorflow.python.keras.backend import exp
from .data_feed import DataSet
from tensorflow import keras, expand_dims
import numpy as np
import pandas as pd
import json


class Tester:
    def __init__(self, vis_fin_model: str,
                 data_csv: str,
                 skip: int = 0) -> None:
        self.model = keras.models.load_model(vis_fin_model)
        self.file_for_predictions = 'results_for_' + \
            vis_fin_model[:-3]+'.csv'
        # for five days
        self.data = DataSet('filtered_data_e.csv', False, skip=skip)
        # results
        self.basic_stats = False
        self.predictions = []
        self.graphs = []
        self.expected = []

    def test(self):
        if self.data.generate(0, 3000):
            self.predictions.append(self.model.predict(self.data.data_set))
            self.expected.append(self.data.data_keys)
        pd.DataFrame({'Expected': self.expected, 'Results': self.predictions}).to_csv(
            self.file_for_predictions, index=False)

    def _test_vals(self):
        match = 0
        snd_match = 0
        direction_match = 0
        for i in range(self.predictions.shape[0]):
            to_list = list(self.predictions[i])
            predict_i = to_list.index(max(to_list))
            snd_predict_i = self._second_largest(to_list, predict_i)
            expect_i = self.expected[i].index(
                max(self.expected[i]))
            if predict_i == expect_i:
                match += 1
            elif (predict_i < 3 and expect_i < 3) or \
                    (predict_i > 3 and expect_i > 3):
                direction_match += 1
            if snd_predict_i == expect_i:
                snd_match += 1
        return {'Total tests': self.predictions.shape[0],
                'Exact match': match,
                'Matched by second highest': snd_match,
                'Match gain or loss': direction_match,
                'Exact match rate': match/self.predictions.shape[0],
                'Gain/Loss match rate': (match+direction_match)/self.predictions.shape[0],
                'Random match rate': self.coin_test[2],
                'Random direction match rate': self.coin_test[3]/self.coin_test[1],
                'Random match test params': f'Matches ({self.coin_test[0]})/N({self.coin_test[1]})', }

    def store_results(self):
        file_name = 'results.json'
        with open(file_name, 'a+') as data_saver:
            data_saver.write('\n')
            data_saver.write(self.file_for_predictions[:-4])
            data_saver.write(json.dumps(self.basic_stats, indent=2))

    def overview(self):
        if self.basic_stats:
            return self.basic_stats
        else:
            self.coin_test = self._coin_flip_equivalent()
            self.basic_stats = [self._test_vals()]
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
            elif r1 > 4 and r2 > 4:
                direction += 1
        return (match, n, match/n, direction, direction/n)
