"""
Sample script for loading the pickle object obtained from hyperband
"""

import pickle

results = pickle.load(open('logs/res/res_file', 'rb'))
# d_loss = pickle.load(open('logs/res/d_loss', 'rb'))

b=[]
c=[]

for i in results.data:
    c.append(results.data[i]['config'])
    b.append(results.data[i]['results'])

print(b)
print(c)