import pandas as pd
import numpy as np
from tensorflow import expand_dims


class DataSet():
    def __init__(
        self,
        db: str,
        date_col: str = 'date',
        forecast_len: int = 5,
        horizon: int = 42,
        graph_width: int = 200,
        skip: int or None = 5,  # rows kept for pure testing
        **kwargs
    ) -> None:
        self.data_base: pd.DataFrame = pd.read_csv(db, **kwargs)
        self.data_base: pd.DataFrame = self.data_base.iloc[skip:]
        self.data_base[date_col] = pd.to_datetime(self.data_base[date_col])
        self.data_base.sort_values(date_col)
        self.data_base.drop(columns=['date'], inplace=True)
        self._forecast_len = forecast_len
        self._horizon = horizon
        self._graph_width = graph_width
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

    def build_data(self):
        data_set = []
        data_keys = []
        total_len = len(self.data_base.columns)
        for n_eq, col in enumerate(self.data_base):
            serries_to_array = np.array(self.data_base[col])
            print(f'Processing equity {n_eq} for {total_len} in total!')
            for i in range(len(serries_to_array)):
                start_position = i*self.forecast_len
                end_position = start_position+self.forecast_len+self.horizon+1
                if end_position >= len(serries_to_array):
                    break
                graph_category_mix = self.make_graph(
                    serries_to_array[
                        start_position:end_position
                    ]
                )
                print(graph_category_mix[0].shape)
                data_set.append(graph_category_mix[0])
                data_keys.append(graph_category_mix[1])
            if n_eq == 2:
                break
            print('Finished!')
        self.data_set = expand_dims(np.array(data_set), axis=-1)
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


# class PrepareData():
#     def __init__(self, database: str, date_col: str, skip: int or None = None):
#         self.full_data = pd.read_csv(database, index_col=date_col)
#         self.keys = self.full_data.columns
#         if not isinstance(skip, type(None)):
#             self.full_data = self.full_data[skip:]

#     def make_graph(self, key, start=0, step=5, horizon=42, width=200):
#         """
#         key, str, is a key from self.keys;
#         start, int, where to start in the key value;
#         step, int, is the offset from the initial period used for category;
#         horizon is the data made into graph;
#         width, int, the width of the generated grid graph;
#         """
#         try:
#             data = self.full_data[key][start:start+step+horizon+1]
#         except IndexError:
#             return 'Finished'
#         data = np.log(np.array(data[:-1])/np.array(data[1:]))[::-1]
#         base_categorical = np.std(data[:-step])
#         category = self._categorize(sum(data[-step:]), base_categorical)
#         if data.shape == (0,):
#             return False
#         print(data[:-step].shape)
#         graph = np.array(self._restructure(data[:-step], width))
#         print(graph.shape)
#         return (graph, category, base_categorical)

#     def clean_db(self):
#         for col in self.keys:
#             if self.full_data[col].isna().any():
#                 self.full_data.drop(columns=col, inplace=True)
#         self.keys = self.full_data.columns

#     def test_data_graph(self, key, start=0, step=5, horizon=42, width=200):
#         """
#         key, str, is a key from self.keys;
#         start, int, where to start in the key value;
#         step, int, is the offset from the initial period used for category;
#         horizon is the data made into graph;
#         width, int, the width of the generated grid graph;
#         """
#         try:
#             data = self.full_data[key][start:start+step+horizon+1]
#         except IndexError:
#             return 'Finished'
#         data = np.log(np.array(data[:-1])/np.array(data[1:]))[::-1]
#         base_categorical = np.std(data[:-step])
#         category = self._categorize(sum(data[-step:]), base_categorical)
#         if not data:
#             return False
#         return (self._restructure(data[:-step], width), category, base_categorical)

#     def _categorize(self, data, base):
#         cateogry = [0, 0, 0, 0, 0, 0, 0]
#         step = base/2
#         position = 3
#         if data > step and data <= step*2:
#             position -= 1
#         elif data < -step and data >= step*-2:
#             position += 1
#         elif data > step*2 and data <= step*3:
#             position -= 2
#         elif data < step*-2 and data >= step*-3:
#             position += 2
#         elif data > step*3:
#             position -= 3
#         elif data < step*-3:
#             position += 3
#         cateogry[position] = 1
#         return cateogry

#     def _restructure(self, data, width):
#         last = int(width/2)
#         board = [[0 for _ in range(len(data)+1)] for _ in range(width+1)]
#         board[last][0] = 1
#         for position, value in enumerate(data, start=1):
#             value = last+round(value*width)
#             if value < 0 or value > width:
#                 return False
#             board[value][position] = 1
#             last = value
#         return board

#     def use_hacks(self, key, step=5, horizon=42, width=200):
#         '''
#         returns all the values related to info available for the asset
#         '''
#         test_data = []
#         categorical = []
#         stds = []
#         epoch = 0
#         while True:
#             start = epoch*(step+horizon+1)
#             epoch += 1
#             recieved = self.make_graph(key, start=start, step=step,
#                                        horizon=horizon, width=width)
#             if recieved == 'Finished':
#                 break
#             elif not recieved:
#                 continue
#             test_data.append(recieved[0])
#             categorical.append(recieved[1])
#             stds.append(recieved[2])
#         return (test_data, categorical, stds)

#     def mass_hacking(self, tick, _from=0, _to=None, step=5, horizon=42, width=200):
#         '''
#         We use different assets simultaniously, to make the hacking ethical!!!
#         tick, int, should change in every use to fetch new data;

#         exemple:
#             for tick in range(100):
#                 data = self.mass_hacking(tick)
#                 if not data:
#                     break
#                 model.train(data=data[0],evaluate=data[1])

#         _from and _to can be used to spec range from the list if the set is extremely big;
#         else is make_graph;
#         '''
#         test_data = []
#         categorical = []
#         stds = []
#         for key in self.keys[_from:_to]:
#             start = tick*(step+horizon+1)
#             result = self.make_graph(key, start=start, step=step,
#                                      horizon=horizon, width=width)
#             if not result or result[0].shape != (width + 1, horizon+1):
#                 continue
#             test_data.append(result[0])
#             categorical.append(result[1])
#             stds.append(result[2])
#         return (test_data, categorical, stds)
