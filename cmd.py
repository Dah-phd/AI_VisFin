from modules.data_feed import *


modeling = prepare_data('historic_data.db', 'returns')
print('TOTAL ASSETS IN DB: ', len(modeling.keys))
print('Commands to use cmd.modeling.make_graph to make single graph from key.')
print('Commands to use cmd.modeling.use_hacks to make all graph from key.')
print('Commands to use cmd.modeling.mass_hacks to make graph from all keys.')
print('exemple:')
print('\tfor tick in range(100):')
print('\t\tdata = cmd.modeling.mass_hacking(tick)')
print('\t\tif not data:')
print('\t\t\tbreak')
print('\t\tmodel.train(data=data[0], evaluate=data[1])')
