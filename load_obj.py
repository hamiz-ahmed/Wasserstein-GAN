import pickle

a = pickle.load(open('logs/res/res_file', 'rb'))

b=[]
c=[]
for i in a.data:
    c.append(a.data[i]['config'])
    b.append(a.data[i]['results'])

print(b)
print(c)