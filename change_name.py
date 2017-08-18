import os

a = 16 
b = 124

v = [i for i in range(a,b+1)]

for i in v:
	os.system('mv output0/output'+str(i)+ ' output0/output'+str(i-3))
	os.system('sleep 5')