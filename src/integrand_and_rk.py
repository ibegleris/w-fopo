import numpy as np
from scipy.constants import pi
from numpy.fft import fftshift
from scipy.fftpack import fft, ifft
#fft, ifft = np.fft.fft, np.fft.ifft
from six.moves import builtins


import numba
complex128 = numba.complex128

vectorize = numba.vectorize
autojit, jit = numba.autojit, numba.jit
cfunc = numba.cfunc
generated_jit = numba.generated_jit
guvectorize = numba.guvectorize

# Pass through the @profile decorator if line profiler (kernprof) is not in use
# Thanks Paul!
try:
    builtins.profile
except AttributeError:
    def profile(func):
        return func


@profile
def RK45CK(dAdzmm, u1, dz, M1,M2,Q, tsh, dt, hf, w_tiled, gam_no_aeff):
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
    A1 = dz*dAdzmm(u1, M1, M2, Q, tsh, dt, hf, w_tiled, gam_no_aeff)
    u2 = A2_temp(u1, A1)

    A2 = dz*dAdzmm(u2, M1, M2, Q, tsh, dt, hf, w_tiled, gam_no_aeff)
    
    u3 = A3_temp(u1, A1,A2)
    A3 = dz*dAdzmm(u3, M1, M2, Q, tsh, dt, hf, w_tiled, gam_no_aeff)
    
    u4 = A4_temp(u1, A1, A2, A3)
    A4 = dz*dAdzmm(u4, M1, M2, Q, tsh, dt, hf, w_tiled, gam_no_aeff)
    
    u5 = A5_temp(u1, A1, A2, A3, A4)
    A5 = dz*dAdzmm(u5, M1, M2, Q, tsh, dt, hf, w_tiled,gam_no_aeff)
    
    u6 = A6_temp(u1, A1, A2, A3, A4, A5)
    A6 = dz*dAdzmm(u6, M1, M2, Q, tsh, dt, hf, w_tiled,gam_no_aeff)
    

    A =  A_temp(u1, A1, A3, A4, A6) # Fifth order accuracy
    Afourth =  Afourth_temp(u1, A1, A3, A4,A5, A6) # Fourth order accuracy

    delta = np.linalg.norm(A - Afourth,2, axis = 1).max()

    return A, delta

trgt = 'cpu'
#trgt = 'parallel'
#trgt = 'cuda'

@jit(nopython=True,nogil = True)
def Afourth_temp(u1, A1, A3, A4, A5, A6):
    return u1 + (2825./27648)*A1 + (18575./48384)*A3 + (13525./55296) * \
        A4 + (277./14336)*A5 + (1./4)*A6

@jit(nopython=True,nogil = True)
def A_temp(u1, A1, A3, A4, A6):
    return u1 + (37./378)*A1 + (250./621)*A3 + (125./594) * \
        A4 + (512./1771)*A6

@jit(nopython=True,nogil = True)
def A2_temp(u1, A1):
    return u1 + (1./5)*A1

@jit(nopython=True,nogil = True)
def A3_temp(u1, A1, A2):
    return u1 + (3./40)*A1 + (9./40)*A2

@jit(nopython=True,nogil = True)
def A4_temp(u1, A1, A2, A3):
    return u1 + (3./10)*A1 - (9./10)*A2 + (6./5)*A3

@jit(nopython=True,nogil = True)
def A5_temp(u1, A1, A2, A3, A4):
    return u1 - (11./54)*A1 + (5./2)*A2 - (70./27)*A3 + (35./27)*A4


@jit(nopython=True,nogil = True)
def A6_temp(u1, A1, A2, A3, A4, A5):
    return u1 + (1631./55296)*A1 + (175./512)*A2 + (575./13824)*A3 +\
                   (44275./110592)*A4 + (253./4096)*A5




@jit(nogil = True)
def dAdzmm_roff_s0(u0, M1, M2, Q, tsh, dt, hf, w_tiled,gam_no_aeff):
    """
    calculates the nonlinear operator for a given field u0
    use: dA = dAdzmm(u0)
    """
    M3 = uabs(u0,M2)
    N = nonlin_kerr(M1, Q, u0, M3)
    N *= gam_no_aeff

    return N



@jit(nogil = True)
def dAdzmm_roff_s1(u0, M1, M2, Q, tsh, dt, hf, w_tiled,gam_no_aeff):
    """
    calculates the nonlinear operator for a given field u0
    use: dA = dAdzmm(u0)
    """
    M3 = uabs(u0,M2)
    N = nonlin_kerr(M1, Q, u0, M3)
    N = gam_no_aeff * (N + tsh*ifft(w_tiled * fft(N)))
    return N


@jit(nogil = True)
def dAdzmm_ron_s0(u0, M1, M2, Q, tsh, dt, hf, w_tiled,gam_no_aeff):
    """
    calculates the nonlinear operator for a given field u0
    use: dA = dAdzmm(u0)
    """
    M3 = uabs(u0,M2)
    M4 = dt*fftshift(ifft(fft(M3)*hf)) # creates matrix M4
    N = nonlin_ram(M1, Q, u0, M3, M4)
    N *= gam_no_aeff
    return N


#@jit(nogil = True)
@profile
def dAdzmm_ron_s1(u0, M1, M2, Q, tsh, dt, hf, w_tiled,gam_no_aeff):
    """
    calculates the nonlinear operator for a given field u0
    use: dA = dAdzmm(u0)
    """
    M3 = uabs(u0,M2)
    M4 = dt*fftshift(ifft(multi(fft(M3),hf))) # creates matrix M4
    N = nonlin_ram(M1, Q, u0, M3, M4)
    N = gam_no_aeff * (N + tsh*ifft(multi(w_tiled,fft(N))))
    return N


@jit(nopython=True,nogil = True)
def multi(a,b):
    return a * b


@guvectorize(['void(complex128[:,:], int64[:,:], float64[:,:])'],\
                 '(n,m),(o,l)->(l,m)',target = trgt)
def uabs(u0,M2,M3):
    for ii in range(M2.shape[1]):
        M3[ii,:] = (u0[M2[0,ii],:]*u0[M2[1,ii],:].conj()).real


@guvectorize(['void(int64[:,:], complex128[:,:], complex128[:,:],\
            float64[:,:], complex128[:,:], complex128[:,:])'],\
            '(w,a),(i,a),(m,n),(l,n),(l,n)->(m,n)',target = trgt)
def nonlin_ram(M1, Q, u0, M3, M4, N):
    N[:,:] = 0
    for ii in range(M1.shape[1]):
        N[M1[0,ii],:] += u0[M1[1,ii],:]*(0.82*(2*Q[0,ii] + Q[1,ii]) \
                                *M3[M1[4,ii],:] + \
                               0.54*Q[0,ii]*M4[M1[4,ii],:])


@guvectorize(['void(int64[:,:], complex128[:,:], complex128[:,:],\
            float64[:,:], complex128[:,:])'],\
            '(w,a),(i,a),(m,n),(l,n)->(m,n)',target = trgt)
def nonlin_kerr(M1, Q, u0, M3, N):
    N[:,:] = 0
    for ii in range(M1.shape[1]):
        N[M1[0,ii],:] += 0.82*(2*Q[0,ii] + Q[1,ii]) \
                                *u0[M1[1,ii],:]*M3[M1[4,ii],:]
                

@vectorize(['complex128(float64,float64,complex128,\
			float64,complex128,float64)'], target=trgt)
def self_step(n2, lamda, N, tsh, temp, rp):
    return -1j*n2*2*rp/lamda*(N + tsh*temp)
