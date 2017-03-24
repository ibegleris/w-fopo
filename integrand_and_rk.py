from __future__ import division,print_function
import numpy as np
import scipy.fftpack
from scipy.constants import pi, c
from numpy.fft import fftshift
from scipy.fftpack import fft, ifft



try:
	import accelerate
	jit = accelerate.numba.jit
	autojit = accelerate.numba.autojit
	complex128 = accelerate.numba.complex128
	float64 = accelerate.numba.float64
	vectorize = accelerate.numba.vectorize
	import mkl
	max_threads = mkl.get_max_threads()
	#mkl.set_num_threads(1)
except ImportError:
	import numba
	vectorize = numba.vectorize
	autojit, jit = numba.autojit, numba.jit
	pass

def RK5mm(dAdzmm,u1,dz,M,n2,lamda,tsh,dt,hf, w_tiled):
	"""
	Propagates the nonlinear operator for 1 step using a 5th order Runge
	Kutta method
	use: [A delta] = RK5mm(u1, dz)
	where u1 is the initial time vector
	hf is the Fourier transform of the Raman nonlinear response time
	dz is the step over which to propagate
    
    in output: A is new time vector
    delta is the norm of the maximum estimated error between a 5th
    order and a 4th order integration
	"""
	A1 = dz*dAdzmm(u1,
					M,n2,lamda,tsh,dt,hf,w_tiled)
	A2 = dz*dAdzmm(u1 + (1./5)*A1,
					M,n2,lamda,tsh,dt,hf,w_tiled)
	
	A3 = dz*dAdzmm(u1 + (3./40)*A1 + (9./40)*A2,
					M,n2,lamda,tsh,dt,hf,w_tiled)
	A4 = dz*dAdzmm(u1 + (3./10)*A1 - (9./10)*A2 + (6./5)*A3,
					M,n2,lamda,tsh,dt,hf,w_tiled)
	A5 = dz*dAdzmm(u1 - (11./54)*A1 + (5./2)*A2 - (70./27)*A3 + (35./27)*A4,
					M,n2,lamda,tsh,dt,hf,w_tiled)
	A6 = dz*dAdzmm(u1 + (1631./55296)*A1 + (175./512)*A2 + (575./13824)*A3  + (44275./110592)*A4 + (253./4096)*A5,
					M,n2,lamda,tsh,dt,hf,w_tiled)
	A  = u1 + (37./378)*A1 + (250./621)*A3 + (125./594)*A4 + (512./1771)*A6 #Fifth order accuracy
	Afourth = u1 + (2825./27648)*A1 + (18575./48384)*A3 + (13525./55296)*A4 + (277./14336)*A5 + (1./4)*A6#Fourth order accuracy

	delta = np.linalg.norm(A - Afourth,2)
	return A, delta


def dAdzmm_roff_s0(u0,M,n2,lamda,tsh,dt,hf, w_tiled):
	"""
	calculates the nonlinear operator for a given field u0
	use: dA = dAdzmm(u0)
	"""
	M3 =  np.abs(u0)**2
	N = M*u0*M3
	N *= -1j*n2*2*pi/lamda
	return N


def dAdzmm_roff_s1(u0,M,n2,lamda,tsh,dt,hf, w_tiled):
	"""
	calculates the nonlinear operator for a given field u0
	use: dA = dAdzmm(u0)
	"""
	M3 = np.abs(u0)**2
	N = M*u0*M3
	N = -1j*n2*2*pi/lamda*(N + tsh*ifft((w_tiled)*fft(N)))
	return N


def dAdzmm_ron_s0(u0,M,n2,lamda,tsh,dt,hf, w_tiled):
	"""
	calculates the nonlinear operator for a given field u0
	use: dA = dAdzmm(u0)
		"""
	M3 =  np.abs(u0)**2

	N = 0.82*M *u0*M3 + 0.18*M*u0*dt*fftshift(ifft(fft(M3)*hf))	
	N *= -1j*n2*2*pi/lamda
	return N
"""
def dAdzmm_ron_s1(u0,M,n2,lamda,tsh,dt,hf, w_tiled):
	M3 =  np.abs(u0)**2
	N = (0.82*M3 + 0.18*dt*fftshift(ifft(fft(M3)*hf)))*M *u0
	N = -1j*n2*2*pi/lamda*(N + tsh*ifft((w_tiled)*fft(N)))
	return N
"""
#from time import time
#import sys
#@vectorize('complex128(complex128,float64,float64,float64,float64,float64,complex128,float64)')

@profile
def dAdzmm_ron_s1(u0,M,n2,lamda,tsh,dt,hf, w_tiled):
	#calculates the nonlinear operator for a given field u0
	#use: dA = dAdzmm(u0)
	#t1 = time()
	#M3 =  np.abs(u0)**2
	M3 =  uabs(np.ascontiguousarray(u0.real),np.ascontiguousarray(u0.imag))

	#print(np.isfortran(u0))
	#print(np.isfortran(M3))
	#print(np.isfortran(fft(M3)*hf))

	temp = fftshift(ifft(fft(M3)*hf))
	#for i in (M, u0,M3, dt, temp):
	#	print(i.dtype)
	N = nonlin(M, u0,M3, dt, temp)
	#print(np.isfortran(N))
	#print(np.isfortran(w_tiled * fft(N)))
	#sys.exit()
	#N = M*u0*(0.82*M3 + 0.18*dt*temp)
	#temp = multi(w_tiled,fft(N))
	
	N = -1j*n2*2*pi/lamda* (N + tsh*ifft(w_tiled * fft(N)))
	#temp = ifft(w_tiled*fft(N))
	#N = self_step(n2, lamda,N, tsh, temp,np.pi )
	#t2 = time() - t1
	#print(t2)
	#sys.exit()
	return N
trgt = 'cpu'
#trgt = 'parallel'
#trgt = 'cuda'

@vectorize(['complex128(complex128,complex128)'],target = trgt)
def multi(x,y):
	return x*y

@vectorize(['complex128(complex128,complex128)'],target = trgt)
def add(x,y):
	return x + y

@vectorize(['float64(float64,float64)'],target = 'cuda') # default to 'cpu'
def uabs(u0r,u0i):
    return u0r*u0r + u0i*u0i

@vectorize(['complex128(float64,complex128,float64,float64,complex128)'],target = trgt)
def nonlin(M, u0,M3, dt,temp):
    return M*u0*(0.82*M3 + 0.18*dt*temp)

@vectorize(['complex128(float64,float64,complex128,float64,complex128,float64)'],target = trgt)
def self_step(n2, lamda,N, tsh, temp,rp ):
    return -1j*n2*2*rp/lamda*(N + tsh*temp)


"""
#Dormant-Prince-Not found to be faster than cash-karp
#@autojit
#
def RK5mm(dAdzmm,u1,dz,M,n2,lamda,tsh,dt,hf, w_tiled, fft,ifft):
	A1 = dz*dAdzmm(u1,
					M,n2,lamda,tsh,dt,hf, w_tiled, fft,ifft)
	A2 = dz*dAdzmm(u1 + (1./5)*A1,
				  	M,n2,lamda,tsh,dt,hf, w_tiled, fft,ifft)
	A3 = dz*dAdzmm(u1 + (3./40)*A1	   + (9./40)*A2,
					M,n2,lamda,tsh,dt,hf, w_tiled, fft,ifft)
	A4 = dz*dAdzmm(u1 + (44./45)*A1	  - (56./15)*A2		+ (32./9)*A3,
					M,n2,lamda,tsh,dt,hf, w_tiled, fft,ifft)
	A5 = dz*dAdzmm(u1 + (19372./6561)*A1 - (25360./2187)*A2   + (64448./6561)*A3	  - (212./729)*A4,
					M,n2,lamda,tsh,dt,hf, w_tiled, fft,ifft)
	A6 = dz*dAdzmm(u1 + (9017./3168)*A1  - (355./33)*A2	   + (46732./5247)*A3	  + (49./176)*A4   - (5103./18656)*A5,
					M,n2,lamda,tsh,dt,hf, w_tiled, fft,ifft)
	temp =(35./384)*A1						  + (500./1113)*A3		+ (125./192)*A4  - (2187./6784)*A5 + (11./84)*A6
	A7 = dz*dAdzmm(u1 +temp,
						M,n2,lamda,tsh,dt,hf, w_tiled, fft,ifft)
	
	A	   = u1 + temp
	Afourth = u1 + (5179/57600)*A1 + (7571/16695)*A3 + (393/640)*A4 - (92097/339200)*A5 + (187/2100)*A6+ (1/40)*A7#Fourth order accuracy

	delta = np.linalg.norm(A - Afourth,2)
	return A, delta
"""
