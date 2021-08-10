from numpy.lib.arraysetops import isin
from tensorflow.python.keras.backend import exp
from .data_feed import PrepareData
from tensorflow import keras, expand_dims
import numpy as np
import pandas as pd


class _SubPrepareData(PrepareData):
    def __init__(self, full_data: pd.DataFrame):
        self.full_data = full_data
        self.keys = list(self.full_data.columns)[1:]

    def make_graph(self, key, start=0, step=5, horizon=42, width=200):
        data = self.full_data[key][start:start+step+horizon+1]
        data = np.log(np.array(data[:-1])/np.array(data[1:]))[::-1]
        data_categorical = sum(data[0:step])
        data = data[step:]
        base_categorical = np.std(data)
        category = self._categorize(data_categorical, base_categorical)
        data = self._restructure(data, width)
        if not data:
            return False
        return (data, category, data_categorical, base_categorical)


class Tester:
    def __init__(self, vis_fin_model: str,
                 data_csv: str,
                 days_to_test: int = 5,
                 skip: int = 0,
                 index_col: str or None = None) -> None:
        self.model = keras.models.load_model(vis_fin_model)
        self.test_data = pd.read_csv(data_csv, index_col=index_col)
        self.skip = skip
        # for five days
        self.graph_makers = [_SubPrepareData(
            self.test_data.iloc[i:]) for i in range(days_to_test)]
        for gm in self.graph_makers:
            gm.clean_db()
        self.basic_stats = False
        self.predictions = []
        self.data = []
        self.graphs = []
        self.expected = []
        self.st_devs = []

    def test(self):
        for graph_maker in self.graph_makers:
            data = []
            graph = []
            expected = []
            st_devs = []
            for key in graph_maker.keys:
                grapth_tup = graph_maker.make_graph(key, start=self.skip)
                if grapth_tup:
                    graph.append(grapth_tup[0])
                    expected.append(grapth_tup[1])
                    data.append(grapth_tup[2])
                    st_devs.append(grapth_tup[3])
            self.predictions.append(self.model.predict(
                expand_dims(graph, axis=-1)))
            self.data.append(data)
            self.expected.append(expected)
            self.st_devs.append(st_devs)

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

    def overview(self):
        if self.basic_stats:
            return self.basic_stats
        else:
            self.coin_test = self._coin_flip_equivalent()
            self.basic_stats = [self._test_vals(i)
                                for i in range(len(self.graph_makers))]

            return self.basic_stats

    @ staticmethod
    def _second_largest(categorical_ls: list, ls_max: int):
        ls = categorical_ls.copy()
        ls[ls_max] = 0
        return ls.index(max(ls))

    @staticmethod
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
