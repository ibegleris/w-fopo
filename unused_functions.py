import numpy as numpy
"""
This file is where good ideas deamed a failiure end up. 
1. The Convergence predictor
	Was the idea to predict if we have reached converenge But a sensible global stop/method could not be established. 

"""
def converge_checker(w,P_out,tol):
	
	if len(P_out) -1 == w.size_vec[0]:
		w.found(P_out)
		w.size_update(w.size)
		return False

	w.found(P_out)

	print('sizes', w.size, ' with error', w.error[-1])

	if w.error[-1] <= tol:
		w.converged += 1
		w.size_update(w.size)
		if w.converged == 3:
			return True
		else:
			return False
	elif w.error[-1] > w.error[-2]:
		w.size =w.size +int(w.size*0.1)
	else:
		w.size = w.size - int(w.size*0.1)
	if w.size <= 0:
		w.size = 10
	w.size_update(w.size)
	w.converged = 0
	return False


class Window(object):
	def __init__(self, size):
		self.size = size
		self.size_vec = [size]
		self.averages = []
		self.error = []
		self.converged = 0

	def size_update(self,size):
		self.size = int(size)
		if self.size < 0:
			print('warning!!!')
			self.size = 2
		self.size_vec.append(self.size)
		return None

	def straight_line(self):
		try:
			alpha = (self.error[-1] - self.error[-2])/(self.size_vec[-1] - self.size_vec[-2])
		except RuntimeWarning:
			alpha = (self.error[-1] - self.error[-2])
		beta = self.error[-1] - alpha * self.size_vec[-2]
		self.size_update(-beta/alpha)
		return None

	def found(self,A):
		print(self.size)
		mean = np.mean(A[-self.size:])
		var = np.std(A[-self.size:])
		self.averages.append(mean)
		self.error.append(100*np.std(self.averages[-3:])/np.mean(self.averages[-3:]))
		#self.error.append(100*var/mean)
		return None