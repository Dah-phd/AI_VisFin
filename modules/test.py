if __name__ == '__main__':
    from ctrl_sql3 import DATABASE
    from output import export
else:
    from .ctrl_sql3 import DATABASE
    from .output import export


db = DATABASE('historic_data.db', 'returns')
full_data = export(db)
print(list(full_data.keys()))
