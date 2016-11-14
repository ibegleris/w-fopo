from __future__ import division,print_function
import numpy as np
import scipy.fftpack
from scipy.constants import pi, c
from scipy.fftpack import fftshift



def RK5mm(dAdzmm,u1,dz,M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft):
    #
    #Propagates the nonlinear operator for 1 step using a 5th order Runge
    #Kutta method
    #use: [A delta] = RK5mm(u1, dz)
    #where u1 is the initial time vector
    #hf is the Fourier transform of the Raman nonlinear response time
    #dz is the step over which to propagate
   # 
   # in output: A is new time vector
   # delta is the norm of the maximum estimated error between a 5th
   # order and a 4th order integration
   #
    #from accelerate import profiler
    #p = profiler.Profile(signatures=False)
    #p.enable()
    A1 = dz*dAdzmm(u1,                                                                                                  M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft)
    A2 = dz*dAdzmm(u1 + (1./5)*A1,                                                                                      M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft)
    A3 = dz*dAdzmm(u1 + (3./40)*A1       + (9./40)*A2,                                                                  M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft)
    A4 = dz*dAdzmm(u1 + (3./10)*A1       - (9./10)*A2       + (6./5)*A3,                                                M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft)
    A5 = dz*dAdzmm(u1 - (11./54)*A1      + (5./2)*A2        - (70./27)*A3      + (35./27)*A4,                           M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft)
    A6 = dz*dAdzmm(u1 + (1631./55296)*A1 + (175./512)*A2    + (575./13824)*A3  + (44275./110592)*A4 + (253./4096)*A5,   M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft)
    
    A       = u1 + (37./378)*A1 + (250./621)*A3 + (125./594)*A4 + (512./1771)*A6 #Fifth order accuracy
    Afourth = u1 + (2825./27648)*A1 + (18575./48384)*A3 + (13525./55296)*A4 + (277./14336)*A5 + (1./4)*A6#Fourth order accuracy
    delta = np.zeros(len(A[0,:]))
    for ii in range(len(A[0,:])):
        delta[ii] = np.linalg.norm(A[:,ii] - Afourth[:,ii],2)
    return A, np.max(delta)


"""
#Dormant-Prince-Not found to be faster than cash-karp
#@autojit
#
def RK5mm(dAdzmm,u1,dz,M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf):
    A1 = dz*dAdzmm(u1,                                                                                                                      M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft)
    A2 = dz*dAdzmm(u1 + (1./5)*A1,                                                                                                          M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft)
    A3 = dz*dAdzmm(u1 + (3./40)*A1       + (9./40)*A2,                                                                                      M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft)
    A4 = dz*dAdzmm(u1 + (44./45)*A1      - (56./15)*A2        + (32./9)*A3,                                                                 M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft)
    A5 = dz*dAdzmm(u1 + (19372./6561)*A1 - (25360./2187)*A2   + (64448./6561)*A3      - (212./729)*A4,                                      M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft)
    A6 = dz*dAdzmm(u1 + (9017./3168)*A1  - (355./33)*A2       + (46732./5247)*A3      + (49./176)*A4   - (5103./18656)*A5,                  M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft)
    temp =(35./384)*A1                          + (500./1113)*A3        + (125./192)*A4  - (2187./6784)*A5 + (11./84)*A6
    A7 = dz*dAdzmm(u1 +temp,     M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf)
    
    A       = u1 + temp
    Afourth = u1 + (5179/57600)*A1 + (7571/16695)*A3 + (393/640)*A4 - (92097/339200)*A5 + (187/2100)*A6+ (1/40)*A7#Fourth order accuracy
    delta = np.zeros(len(A[0,:]))
    for ii in range(len(A[0,:])):
        delta[ii] = np.linalg.norm(A[:,ii] - Afourth[:,ii],2)
    return A, np.max(delta)
"""


def dAdzmm_roff_s0(u0,M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft):
    """
    calculates the nonlinear operator for a given field u0
    use: dA = dAdzmm(u0)
    """
    M3 = np.zeros([len(t),len(M2[0,:])],dtype=np.complex128)  
    for ii in range(M2.shape[1]):
        M3[:,ii] = u0[:,M2[0,ii]]*np.conj(u0[:,M2[1,ii]])  
    N = np.zeros(np.shape(u0),dtype=np.complex128) 
    for ii in range(M1.shape[1]):
        N[:,int(M1[0,ii])] += (2*(M1[4,ii]) + M1[5,ii])\
                                *u0[:,int(M1[1,ii])]*M3[:,int(M1[6,ii])]
    N *= -1j*n2*2*pi/lamda
    return N


def dAdzmm_roff_s1(u0,M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft):
    """
    calculates the nonlinear operator for a given field u0
    use: dA = dAdzmm(u0)
    """
    M3 = np.zeros([len(t),len(M2[0,:])],dtype=np.complex128)  
    for ii in range(M2.shape[1]):
        M3[:,ii] = u0[:,M2[0,ii]]*np.conj(u0[:,M2[1,ii]])  
    N = np.zeros(np.shape(u0),dtype=np.complex128) 
    for ii in range(M1.shape[1]):
        N[:,int(M1[0,ii])] += (2*(M1[4,ii]) + M1[5,ii])\
                                *u0[:,int(M1[1,ii])]*M3[:,int(M1[6,ii])]
    
    N = -1j*n2*2*pi/lamda*(N + tsh*ifft((np.tile(w,(len(u0[0,:]),1)).T + woffset)*fft(N)))

    return N


def dAdzmm_ron_s0(u0,M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft):

    """
    calculates the nonlinear operator for a given field u0
    use: dA = dAdzmm(u0)
        """
    fr = 0.18
    M3 = np.zeros([len(t),len(M2[0,:])],dtype=np.complex128)  
    for ii in range(M2.shape[1]):
        M3[:,ii] = u0[:,M2[0,ii]]*np.conj(u0[:,M2[1,ii]])  
    N = np.zeros(np.shape(u0),dtype=np.complex128) 
    temp1 = fft(M3)
    temp2 = np.tile(hf,(len(M2[1,:]),1)).T
    temp3 = temp1*temp2
    temp4 = ifft(temp3)
    M4 = dt*fftshift(temp4,1)*fr
    for ii in range(M1.shape[1]):
        N[:,int(M1[0,ii])] += (1-fr)*3*M1[4,ii] \
                                *u0[:,int(M1[1,ii])]*M3[:,int(M1[6,ii])] + \
                               3*fr*M1[4,ii]*u0[:,int(M1[1,ii])]*M4[:,int(M1[6,ii])]
    N *= -1j*n2*2*pi/lamda
    return N


def dAdzmm_ron_s1(u0,M1,M2,t,n2,lamda,tsh,w,woffset,dt,hf,fft,ifft):
    fr = 0.18

    M3 = np.zeros([len(t),len(M2[0,:])],dtype=np.complex)  
    for ii in range(M2.shape[1]):
        M3[:,ii] = u0[:,M2[0,ii]]*np.conj(u0[:,M2[1,ii]])

    N = np.zeros(np.shape(u0),dtype=np.complex)
    temp1 = fft(M3)
    temp2 = np.tile(hf,(len(M2[1,:]),1)).T
    temp3 = temp1*temp2
    temp4 = ifft(temp3)

    M4 = dt*fftshift(temp4)*fr
    for ii in range(M1.shape[1]):
        N[:,int(M1[0,ii])] += (1-fr)*3*M1[4,ii] \
                                *u0[:,int(M1[1,ii])]*M3[:,int(M1[6,ii])] + \
                               3*fr*M1[4,ii]*u0[:,int(M1[1,ii])]*M4[:,int(M1[6,ii])]

    temp1 = fft(N)
    temp2 = np.tile(w,(len(u0[0,:]),1)).T + woffset
    temp3 = temp1*temp2
    temp4 = ifft(temp3)
    temp5 = tsh*temp4
    temp6 = N + temp5
    N = -1j*n2*2*pi/lamda*temp6


    return N