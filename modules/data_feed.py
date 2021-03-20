if __name__ == '__main__':
    from ctrl_sql3 import DATABASE
    from output import export
else:
    from .ctrl_sql3 import DATABASE
    from .output import export

from numpy import array, std, log


class prepare_data():
    def __init__(self, database, table):
        self.db = DATABASE(database, table)
        self.full_data = export(self.db)
        self.keys = [key for key in self.full_data.keys()][1:]

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
            return False
        data_categorical = sum(data[0:step])
        data = data[step:]
        data = log(array(data[:-1])/array(data[1:]))[::-1]
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
        elif data > step*2 and <=step*3:
            position - =2
        elif data < step*-2 and >=step*-3:
            position + =2
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

    def use_hacks(self, key, start=0, step=5, horizon=42, width=200):
        pass

    def mass_hacking(self, key, start=0, step=5, horizon=42, width=200):
        pass
