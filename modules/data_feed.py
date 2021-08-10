from numpy import array, std, log
import pandas as pd


class PrepareData():
    def __init__(self, database: str, date_col: str, skip: int or None = None):
        self.full_data = pd.read_csv(database, index_col=date_col)
        self.keys = self.full_data.columns
        if not isinstance(skip, type(None)):
            self.full_data = self.full_data[skip:]

    def make_graph(self, key, start=0, step=5, horizon=42, width=200):
        """
        key, str, is a key from self.keys;
        start, int, where to start in the key value;
        step, int, is the offset from the initial period used for category;
        horizon is the data made into graph;
        width, int, the width of the generated grid graph;
        """
        try:
            data = self.full_data[key][start:start+step+horizon+1]
        except IndexError:
            return 'Finished'
        data = data[step:]
        data = log(array(data[:-1])/array(data[1:]))[::-1]
        data_categorical = sum(data[0:step])
        base_categorical = std(data)
        category = self._categorize(data_categorical, base_categorical)
        data = self._restructure(data, width)
        if not data:
            return False
        return (array(data), category, base_categorical)

    def clean_db(self):
        for col in self.keys:
            if self.full_data[col].isna().any():
                self.full_data.drop(columns=col, inplace=True)
        self.keys = self.full_data.columns

    def test_data_graph(self, key, start=0, step=5, horizon=42, width=200):
        """
        key, str, is a key from self.keys;
        start, int, where to start in the key value;
        step, int, is the offset from the initial period used for category;
        horizon is the data made into graph;
        width, int, the width of the generated grid graph;
        """
        try:
            data = self.full_data[key][start:start+step+horizon+1]
        except IndexError:
            return 'Finished'
        data = log(array(data[:-1])/array(data[1:]))[::-1]
        data_categorical = sum(data[0:step])
        data = data[step:]
        base_categorical = std(data)
        category = self._categorize(data_categorical, base_categorical)
        data = self._restructure(data, width)
        if not data:
            return False
        return (data, category, base_categorical)

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

    def use_hacks(self, key, step=5, horizon=42, width=200):
        '''
        returns all the values related to info available for the asset
        '''
        test_data = []
        categorical = []
        stds = []
        epoch = 0
        while True:
            start = epoch*(step+horizon+1)
            epoch += 1
            recieved = self.make_graph(key, start=start, step=step,
                                       horizon=horizon, width=width)
            if recieved == 'Finished':
                break
            elif not recieved:
                continue
            test_data.append(recieved[0])
            categorical.append(recieved[1])
            stds.append(recieved[2])
        return (test_data, categorical, stds)

    def mass_hacking(self, tick, _from=0, _to=None, step=5, horizon=42, width=200):
        '''
        We use different assets simultaniously, to make the hacking ethical!!!
        tick, int, should change in every use to fetch new data;

        exemple:
            for tick in range(100):
                data = self.mass_hacking(tick)
                if not data:
                    break
                model.train(data=data[0],evaluate=data[1])

        _from and _to can be used to spec range from the list if the set is extremely big;
        else is make_graph;
        '''
        test_data = []
        categorical = []
        stds = []
        for key in self.keys[_from:_to]:
            start = tick*(step+horizon+1)
            result = self.make_graph(key, start=start, step=step,
                                     horizon=horizon, width=width)
            if not result or result[0].shape != (201, 85):
                continue
            test_data.append(result[0])
            categorical.append(result[1])
            stds.append(result[2])
        return (test_data, categorical, stds)
