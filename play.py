import numpy as np
ar = np.array([0,20,10,30])
pos = np.array([0,2,1,3])
y = np.argsort(pos)
print(y)
print(ar[y])