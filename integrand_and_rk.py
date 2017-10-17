from __future__ import division, print_function
import numpy as np
from scipy.constants import pi
from numpy.fft import fftshift
from scipy.fftpack import fft, ifft

from six.moves import builtins

try:
    import accelerate
    jit = accelerate.numba.jit
    autojit = accelerate.numba.autojit
    complex128 = accelerate.numba.complex128
    float64 = accelerate.numba.float64
    vectorize = accelerate.numba.vectorize
    import mkl
    max_threads = mkl.get_max_threads()
except ImportError:
    import numba
    vectorize = numba.vectorize
    autojit, jit = numba.autojit, numba.jit
    cfunc = numba.cfunc
    generated_jit = numba.generated_jit
    pass
# Pass through the @profile decorator if line profiler (kernprof) is not in use
# Thanks Paul!
try:
    builtins.profile
except AttributeError:
    def profile(func):
        return func

@profile
def RK45CK(dAdzmm, u1, dz, M1,M2,Q, n2, lamda, tsh, dt, hf, w_tiled):
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
    A1 = dz*dAdzmm(u1, M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled)
    u2 = A2_temp(u1, A1)

    A2 = dz*dAdzmm(u2, M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled)
    
    u3 = A3_temp(u1, A1,A2)
    A3 = dz*dAdzmm(u3, M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled)
    
    u4 = A4_temp(u1, A1, A2, A3)
    A4 = dz*dAdzmm(u4, M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled)
    
    u5 = A5_temp(u1, A1, A2, A3, A4)
    A5 = dz*dAdzmm(u5, M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled)
    
    u6 = A6_temp(u1, A1, A2, A3, A4, A5)
    A6 = dz*dAdzmm(u6, M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled)
    

    A =  A_temp(u1, A1, A3, A4, A6) # Fifth order accuracy
    Afourth =  Afourth_temp(u1, A1, A3, A4,A5, A6) # Fourth order accuracy
    #sys.exit()
    #delta = np.linalg.norm(A - Afourth, 2)
    delta = np.zeros(len(A[:,0]))
    for ii in range(len(A[:,0])):
        delta[ii] = np.linalg.norm(A[ii,:] - Afourth[ii,:],2)
    delta = np.max(delta)
    sys.exit()
    return A, delta

trgt = 'cpu'
#trgt = 'parallel'
#trgt = 'cuda'

@jit
def Afourth_temp(u1, A1, A3, A4,A5, A6):
    return u1 + (2825./27648)*A1 + (18575./48384)*A3 + (13525./55296) * \
        A4 + (277./14336)*A5 + (1./4)*A6

@jit
def A_temp(u1, A1, A3, A4, A6):
    return u1 + (37./378)*A1 + (250./621)*A3 + (125./594) * \
        A4 + (512./1771)*A6

@jit
def A2_temp(u1, A1):
    return u1 + (1./5)*A1

@jit
def A3_temp(u1, A1, A2):
    return u1 + (3./40)*A1 + (9./40)*A2

@jit
def A4_temp(u1, A1, A2, A3):
    return u1 + (3./10)*A1 - (9./10)*A2 + (6./5)*A3

@jit
def A5_temp(u1, A1, A2, A3, A4):
    return u1 - (11./54)*A1 + (5./2)*A2 - (70./27)*A3 + (35./27)*A4


@jit
def A6_temp(u1, A1, A2, A3, A4, A5):
    return u1 + (1631./55296)*A1 + (175./512)*A2 + (575./13824)*A3 +\
                   (44275./110592)*A4 + (253./4096)*A5

@profile
def RK34(dAdzmm, u1, dz, M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled):
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
    #third order:
    A1 = dz*dAdzmm(u1,
                    M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled)
    A2 = dz*dAdzmm(u1 + 0.5*A1,
                    M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled)

    A3 = dz*dAdzmm(u1 - A1 + 2*A2,
                    M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled)
    Athird = u1 + 1/6 * (A1 + 4 * A2 + A3)


    A3 = dz*dAdzmm(u1 + 0.5*A2,
                    M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled)
    
    A4 = dz*dAdzmm(u1 + A3,
                    M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled)
    
    A = u1 + 1/6 * (A1 + 2 * A2 + 2* A3 +  A4) 
    delta = np.linalg.norm(A - Athird, 2)
    return A, delta

@profile
def dAdzmm_roff_s0(u0, M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled):
    """
    calculates the nonlinear operator for a given field u0
    use: dA = dAdzmm(u0)
    """
    M3 = uabs(np.ascontiguousarray(u0.real), np.ascontiguousarray(u0.imag))
    N = nonlin_ker(M1, M2, Q, u0, M3)
    N *= -1j*n2*2*pi/lamda
    return N


@profile
def dAdzmm_roff_s1(u0, M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled):
    """
    calculates the nonlinear operator for a given field u0
    use: dA = dAdzmm(u0)
    """
    #print('no')
    M3 = uabs(np.ascontiguousarray(u0.real), np.ascontiguousarray(u0.imag))
    N = nonlin_ker(M1, M2, Q, u0, M3)
    N = -1j*n2*2*pi/lamda*(N + tsh*ifft((w_tiled)*fft(N)))
    return N

@profile
def dAdzmm_ron_s0(u0, M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled):
    """
    calculates the nonlinear operator for a given field u0
    use: dA = dAdzmm(u0)
            """
    print(u0.real.flags)
    print(u0.imag.flags)
    M3 = uabs(np.ascontiguousarray(u0.real), np.ascontiguousarray(u0.imag))

    temp = fftshift(ifft(fft(M3)*hf))
    N = nonlin_ram(M1, M2, Q, u0, M3, dt, temp)
    N *= -1j*n2*2*pi/lamda
    return N

@profile
def dAdzmm_ron_s1(u0, M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled):
    #M3 = uabs(u0.real, u0.imag)

    M3 = np.zeros([len(M2[0,:]),u0.shape[1]])   
    for ii in range(M2.shape[1]):
        M3[ii,:] = u0[M2[0,ii],:]*np.conj(u0[M2[1,ii],:])
    M4 = fftshift(ifft(fft(M3)*hf)) # creates matrix M4
    fr = 0.18
    N = np.zeros(np.shape(u0),dtype='complex')
    for ii in range(M1.shape[1]):
        N[M1[0,ii],:] += (1-fr)*(2*Q[0,ii] + Q[1,ii]) \
                                *u0[M1[1,ii],:]*M3[M1[4,ii],:] + \
                               3*fr*Q[0,ii]*u0[M1[1,ii],:]*M4[M1[4,ii],:]
    
    #N = nonlin_ram(M1, M2, Q, u0, M3, dt, temp)
    N = -1j*n2*2*pi/lamda * (N + tsh*ifft(w_tiled * fft(N)))
    """
    for ii in range(M1.shape[1]):
        N[:,int(M1[0,ii])-1] += (1-fr)*(2*(M1[4,ii]) + M1[5,ii]) \
                                *u0[:,int(M1[1,ii])-1]*M3[:,int(M1[6,ii])-1] + \
                               3*fr*M1[4,ii]*u0[:,int(M1[1,ii])-1]*M4[:,int(M1[6,ii])-1]
    N = -1j*n2*2*np.pi/lamda*(N + tsh*imfft((np.tile(w,(len(u0[0,:]),1)).T + woffset)*mfft(N)))
    """
    print(N)
    return N
trgt = 'cpu'
#trgt = 'parallel'
#trgt = 'cuda'


@vectorize(['complex128(complex128,complex128)'], target=trgt)
def multi(x, y):
    return x*y


@vectorize(['complex128(complex128,complex128)'], target=trgt)
def add(x, y):
    return x + y


@vectorize(['float64(float64,float64)'], target=trgt)
def uabs(u0r, u0i):
    return u0r*u0r + u0i*u0i

@vectorize(['complex128(int8,int8,float64,complex128,\
                float64)'], target=trgt)
def nonlin_ker(M1, M2, Q, u0, M3):
    return 0.82*M*u0*M3


@vectorize(['complex128(int8,int8,float64,complex128,\
				float64,float64,complex128)'], target=trgt)
def nonlin_ram(M1, M2, Q,u0, M3, dt, temp):
    return M*u0*(0.82*M3 + 0.18*dt*temp)


@vectorize(['complex128(float64,float64,complex128,\
			float64,complex128,float64)'], target=trgt)
def self_step(n2, lamda, N, tsh, temp, rp):
    return -1j*n2*2*rp/lamda*(N + tsh*temp)



#Dormant-Prince, Not found to be faster than cash-karp
@profile
def RK45DP(dAdzmm, u1, dz, M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled):
	A1 = dz*dAdzmm(u1,
                   M1, M2, Q, n2, lamda, tsh, dt, hf, w_tiled)
	A2 = dz*dAdzmm(u1 + (1./5)*A1,
				  	M1, M2, Q,n2,lamda,tsh,dt,hf, w_tiled)
	A3 = dz*dAdzmm(u1 + (3./40)*A1	   + (9./40)*A2,
					M1, M2, Q,n2,lamda,tsh,dt,hf, w_tiled)
	A4 = dz*dAdzmm(u1 + (44./45)*A1	  - (56./15)*A2		+ (32./9)*A3,
					M1, M2, Q,n2,lamda,tsh,dt,hf, w_tiled)
	A5 = dz*dAdzmm(u1 + (19372./6561)*A1 - (25360./2187)*A2   + (64448./6561)*A3	  - (212./729)*A4,
					M1, M2, Q,n2,lamda,tsh,dt,hf, w_tiled)
	A6 = dz*dAdzmm(u1 + (9017./3168)*A1  - (355./33)*A2	   + (46732./5247)*A3	  + (49./176)*A4   - (5103./18656)*A5,
					M1, M2, Q,n2,lamda,tsh,dt,hf, w_tiled)
	A = u1+ (35./384)*A1						  + (500./1113)*A3		+ (125./192)*A4  - (2187./6784)*A5 + (11./84)*A6
	A7 = dz*dAdzmm(A,
						M1, M2, Q,n2,lamda,tsh,dt,hf, w_tiled)
	
	Afourth = u1 + (5179/57600)*A1 + (7571/16695)*A3 + (393/640)*A4 - (92097/339200)*A5 + (187/2100)*A6+ (1/40)*A7#Fourth order accuracy

	delta = np.linalg.norm(A - Afourth,2)
	return A, delta

