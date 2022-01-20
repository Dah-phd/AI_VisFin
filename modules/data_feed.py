from re import T
import pandas as pd
import numpy as np
from tensorflow import expand_dims


class DataSet():
    def __init__(
        self,
        db: str,
        auto_fill_set: bool = True,
        date_col: str = 'date',
        forecast_len: int = 5,
        horizon: int = 42,
        graph_width: int = 200,
        skip: int or None = 5,  # rows kept for pure testing
        load_pure_test: bool = False,
    ) -> None:
        self.db_loc = db
        self._date_col = date_col
        self._skip = skip
        self.load_test = load_pure_test
        self._forecast_len = forecast_len
        self._horizon = horizon
        self._graph_width = graph_width
        self.data_set = []
        self.data_keys = []
        self.data_base = pd.DataFrame()
        self.shape = (0)
        if auto_fill_set:
            self.build_data()

    @property
    def horizon(self):
        return self._horizon

    @property
    def forecast_len(self):
        return self._forecast_len

    @property
    def graph_width(self):
        return self._graph_width

    def set_test_period(self, days: int):
        self._skip = days

    def load_from_db(self, from_, to):
        if not self.load_test:
            self.data_base: pd.DataFrame = pd.read_csv(self.db_loc)\
                .iloc[self._skip:]
        else:
            self.data_base: pd.DataFrame = pd.read_csv(self.db_loc)\
                .iloc[:(self.forecast_len+self.horizon+1)]
        self.data_base[self._date_col] = pd.to_datetime(
            self.data_base[self._date_col])
        self.data_base.sort_values(self._date_col)
        self.data_base.drop(columns=['date'], inplace=True)
        if not self.load_test:
            try:
                self.data_base = self.data_base[
                    self.data_base.columns[from_:to]
                ]
            except Exception:
                return False
        return True

    def generate(self, from_: int, to: int):
        del self.data_set
        del self.data_keys
        del self.data_base
        if not self.load_from_db(from_, to):
            return False
        self.build_data()
        return True

    def build_data(self):
        data_set = []
        data_keys = []
        removed = 0
        for col_i in range(len(self.data_base.columns)):
            serries_to_array = np.array(
                self.data_base[self.data_base.columns[col_i]]
            )
            for i in range(len(serries_to_array)):
                start_position = i*self.forecast_len
                end_position = start_position+self.forecast_len+self.horizon+1
                if end_position >= len(serries_to_array):
                    break
                graph_category_mix = self.make_graph(
                    serries_to_array[start_position:end_position]
                )
                if graph_category_mix[0].shape != (201, 43):
                    removed += 1
                    continue
                data_set.append(graph_category_mix[0])
                data_keys.append(graph_category_mix[1])
        data_set = np.array(data_set)
        self.data_set = expand_dims(data_set, axis=-1)
        self.shape = self.data_set.shape
        self.data_keys = np.array(data_keys)

    def make_graph(self, data: np.array):
        """
        key, str, is a key from self.keys;
        start, int, where to start in the key value;
        step, int, is the offset from the initial period used for category;
        horizon is the data made into graph;
        width, int, the width of the generated grid graph;
        """
        data: np.array = np.log(data[:-1]/data[1:])[::-1]
        base_categorical = np.std(data[:-self.forecast_len])
        category = self._categorize(
            sum(data[-self.forecast_len:]), base_categorical)
        return (np.array(self._restructure(data[:-self.forecast_len], self.graph_width)), category, base_categorical)

    def _categorize(self, data, base):
        cateogry = [0, 0, 0, 0, 0, 0, 0]
        step = base/2
        position = 3
        if data > step and data <= step*2:
            position -= 1
        elif data < -step and data >= step*-2:
            position += 1
        elif data > step*2 and data <= step*3:
            position -= 2
        elif data < step*-2 and data >= step*-3:
            position += 2
        elif data > step*3:
            position -= 3
        elif data < step*-3:
            position += 3
        cateogry[position] = 1
        return cateogry

    def _restructure(self, data, width):
        last = int(width/2)
        board = [[0 for _ in range(len(data)+1)] for _ in range(width+1)]
        board[last][0] = 1
        for position, value in enumerate(data, start=1):
            value = last+round(value*width)
            if value < 0 or value > width:
                return False
            board[value][position] = 1
            last = value
        return board
