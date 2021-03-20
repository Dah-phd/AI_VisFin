from modules.data_feed import *


inst = prepare_data('historic_data.db', 'returns')
for key in inst.keys:
    a = inst.make_graph(key)
    if a:
        b = a

for row in b:
    print(row)
