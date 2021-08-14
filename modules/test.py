from tensorflow.python.keras.backend import exp
from .data_feed import PrepareData
from tensorflow import keras, expand_dims
import numpy as np
import pandas as pd
import json


class Tester:
    def __init__(self, vis_fin_model: str,
                 data_csv: str,
                 days_to_test: int = 5,
                 skip: int = 0,
                 index_col: str or None = None) -> None:
        self.model = keras.models.load_model(vis_fin_model)
        self.test_data = pd.read_csv(data_csv, index_col=index_col)
        self.days_to_test = days_to_test
        self.file_for_predictions = 'results_for_' + \
            data_csv[:-4]+f'_{5-skip}_new_days.csv'
        # for five days
        self.graph_maker = PrepareData(
            database=data_csv, date_col=index_col, skip=skip)
        self.graph_maker.clean_db()
        # results
        self.basic_stats = False
        self.predictions = []
        self.graphs = []
        self.expected = []
        self.st_devs = []

    def test(self):
        for i in range(self.days_to_test):
            graph = []
            expected = []
            st_devs = []
            for key in self.graph_maker.keys:
                grapth_tup = self.graph_maker.make_graph(key, start=i)
                if grapth_tup[0].shape == (201, 43):
                    graph.append(grapth_tup[0])
                    expected.append(grapth_tup[1])
                    st_devs.append(grapth_tup[2])
            self.predictions.append(self.model.predict(
                expand_dims(graph, axis=-1)))
            self.expected.append(expected)
            self.st_devs.append(st_devs)
        pd.DataFrame({'Expected': self.expected, 'Results': self.predictions}).to_csv(
            self.file_for_predictions, index=False)

    def _test_vals(self, i_gm):
        match = 0
        snd_match = 0
        direction_match = 0
        for i in range(self.predictions[i_gm].shape[0]):
            to_list = list(self.predictions[i_gm][i])
            predict_i = to_list.index(max(to_list))
            snd_predict_i = self._second_largest(to_list, predict_i)
            expect_i = self.expected[i_gm][i].index(
                max(self.expected[i_gm][i]))
            if predict_i == expect_i:
                match += 1
            elif (predict_i < 3 and expect_i < 3) or \
                    (predict_i > 3 and expect_i > 3):
                direction_match += 1
            if snd_predict_i == expect_i:
                snd_match += 1
        return {'Total tests': self.predictions[i_gm].shape[0],
                'Exact match': match,
                'Matched by second highest': snd_match,
                'Match gain or loss': direction_match,
                'Exact match rate': match/self.predictions[i_gm].shape[0],
                'Gain/Loss match rate': (match+direction_match)/self.predictions[i_gm].shape[0],
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
            self.basic_stats = [self._test_vals(
                i) for i in range(self.days_to_test)]
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
