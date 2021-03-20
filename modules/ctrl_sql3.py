import sqlite3


class DATABASE:
    def __init__(self, database, table=None):
        self.database = database
        self.db = sqlite3.connect(self.database)
        self.cursor = self.db.cursor()
        if table:
            self.set_table(table)

    def set_table(self, table):
        self.table = table
        print(f'Connected to {self.table} table!')

    def build_table(self, table):
        self.table = table
        self.cursor.execute(
            f'CREATE TABLE {self.table} (date_ date primary key)'
        )
        print(f'{self.table} was created!')

    def drop_table(self):
        self.cursor.execute(
            f'DROP TABLE {self.table}'
        )
        print('Table destroyed!')

    def freeze(self):
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return [tab[0] for tab in self.cursor]

    def freeze_tab(self):
        self.cursor.execute(f'PRAGMA table_info ({self.table})')
        return [t[1] for t in self.cursor]

    def q_all(self, print_=0):
        self.cursor.execute(
            f'SELECT * FROM {self.table}'
        )
        if print_ != 0:
            for t in self.cursor:
                print(t)
        else:
            return [t for t in self.cursor]

    def fix_dates(self, date, table):
        for t in table:
            if str(t[0]) == date:
                return
        self.cursor.execute(
            "INSERT INTO " + self.table + " (date_) VALUES (?)", (date,))

    def import_data(self, DF):
        ls = [item for item in DF.items()]
        dates = ls[0][1]
        self.cursor.execute(f'SELECT date_ FROM {self.table}')
        temp = [t for t in self.cursor]
        for t in dates:
            self.fix_dates(t, temp)
        for asset in ls[1:]:
            name = asset[0]
            try:
                self.cursor.execute(
                    f'ALTER TABLE {self.table} ADD {name} float'
                )
                print('ADDING column')
            except:
                print('CORRECTING/ADDING to existing entry')
            val = asset[1]
            self.cursor.executemany(
                "UPDATE " + self.table +
                " SET " + name + " = (?) WHERE date_ = (?)", zip(val, dates)
            )
        self.db.commit()

    def check_data(self):
        tabs = self.freeze_tab()
        tabs.pop(0)
        for tab in tabs:
            print('Checking column: ', tab)
            self.cursor.execute(
                "SELECT date_, " + tab +
                " FROM " + self.table +
                " ORDER BY date_"
            )
            n_val = 0
            rows = [row for row in self.cursor]
            for row in rows:
                if not row[1]:
                    print('For date: ', row[0], '\n The value is: None')
                    self.cursor.execute(
                        "UPDATE " + self.table +
                        " SET " + tab + " = (?) WHERE date_ = (?)",
                        (n_val, row[0])
                    )
                    print('REPLACED WITH PREVIOUS: ', n_val)
                    self.db.commit()
                else:
                    n_val = row[1]
            print(tab, ' is checked!')

    def drop_column(self, data_string):
        meta = [t for t in self.cursor.execute(
            f'PRAGMA table_info ({self.table})')]
        meta_structure = ''
        meta_val = ''
        holder = ''
        for t in meta:
            if t[1] == data_string:
                meta_index = meta.index(t)
                print(meta_index)
            else:
                meta_structure += t[1] + ' ' + t[2] + \
                    (' primary key' if t[5] == 1 else '') + ', '
                meta_val += t[1] + ', '
                holder += '?, '
        if not meta_index:
            print('Column not found!')
            return
        elif meta_index == 0:
            print('Can not remove PK!')
            return
        data = self.q_all()
        data = [tuple(t1 for n, t1 in enumerate(t) if meta_index != n)
                for t in data]
        self.drop_table()
        self.cursor.execute(
            f'CREATE TABLE {self.table} ({meta_structure[:-2]})'
        )
        self.cursor.executemany(
            f'INSERT INTO {self.table} ({meta_val[:-2]}) VALUES ({holder[:-2]})', data)
        self.db.commit()
        print('Table rebuild!')
