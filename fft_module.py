import numpy as np
import scipy.fftpack as scifft
from time import time

try:
	import accelerate.mkl.fftpack as mklfft
	def timing(N, nm,times):
		a = np.random.rand(2**N, nm)*100 + 1j*np.random.rand(2**N, nm)*100
		dt = []
		method_names = ('scipy','mklfft','numpy')
		methods = (scifft.ifft,mklfft.ifft,np.fft.ifft)
		dt_average = []
		for method in methods:
			dt = []
			for i in range(times):
				t = time()
				test = method(a.T).T
				t = time() - t

				dt.append(t)
			dt_average.append(np.average(dt))
		

		top = np.argsort(dt_average)
		method_sorted = [method_names[i] for i in top]
		dt_average_sorted = [dt_average[i] for i in top]
		return method_sorted ,dt_average_sorted


	def pick(N,nm,times):
		a = timing(N, nm, times)[0][0]
		if a == 'scipy':
			def mfft(x):
				return scifft.fft(x.T).T
			def imfft(x):
				return scifft.ifft(x.T).T

		elif a == 'mklfft':
			def mfft(x):
				return mklfft.fft(x.T).T
			def imfft(x):
				return mklfft.ifft(x.T).T
		else: 
			def mfft(x):
				return np.fft.fft(x.T).T
			def imfft(x):
				return np.fft.ifft(x.T).T
		return mfft,imfft,a


except ImportError:
	print("You dont have accelerate on this system, defaulting to scipy")
	def mfft(x):
		return scifft.fft(x.T).T
	def imfft(x):
		return scifft.ifft(x.T).T
	def timing(N, nm,times):
		return mfft,imfft,('scipy',)









if __name__ == '__main__':
	import matplotlib.pyplot as plt
	print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
	points = []
	start = 100
	time1 = []
	time2 = []
	time3 = []
	for i in range(1,start,1):
		points.append(i)
		a ,b = timing(12,1,i)
		time1.append(b[0])
		time2.append(b[1])

		time3.append(b[2])
		names = a
	plt.plot(points, time1, label= names[0])
	plt.plot(points, time2, label= names[1])

	plt.plot(points, time3, label= names[2])
	plt.legend()
	plt.show()
