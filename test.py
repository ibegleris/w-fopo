import numpy as np
from joblib import Parallel, delayed

#D = {'b':1, 'a':2, 'c':3}
a = np.arange(1,10)
b = np.arange(10,20)
c = np.arange(20,30)
A = {'b':1, 'a':2, 'c':3}
B = {'b':4, 'a':5, 'c':6}
C = {'a':1, 'b':2, 'c':3}

E = {'l':1}
DD = (A,B,C)




def f(a,b,c,l):
	print(a,b,c,l)
	return None


print('passing dictionary:')
#f(**A)

print('pas')
num_cores =2
A = Parallel(n_jobs=num_cores)(delayed(f)(**{**DD[i],**E}) for i in range(len(DD)))
