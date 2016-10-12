# -*- coding: utf-8 -*-
from __future__ import division, print_function
"""
Created on Mon Oct 26 15:56:07 2015

@author: john
"""
import sys
import numpy as np
import scipy.fftpack
scfft,fftshift,scifft = scipy.fftpack.fft, scipy.fftpack.fftshift, scipy.fftpack.ifft
from scipy.linalg import norm
from scipy.constants import pi, c
from scipy.io import loadmat
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.io import savemat
import pandas as pan

from math import isinf
import pickle

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from integrand_and_rk import *


try:
    import accelerate
    jit = accelerate.numba.jit
    autojit = accelerate.numba.autojit
    from accelerate import numba
    vectorize, float64,complex128 = numba.vectorize,numba.float64,numba.complex128 
except ImportError:
    print("install the accelerate packadge from anaconda or change the source code ie remove references to @jit and accelerate imports")
    pass

try:
    import mklfft
    mkfft = mklfft.fftpack.fft
    imkfft = mklfft.fftpack.ifft
except ImportError:
    print("install the mklfft packadge. Sometimes it works well others not.")
    imkfft = scipy.fftpack.ifft
    mkfft = scipy.fftpack.fft
    pass

try:
    import mkl
except ImportError:
    print("MKL libaries help when you are not running in paralel. There is a free academic lisence by continuum analytics")
    pass

def dbm2w(dBm):
    """This function converts a power given in dBm to a power given in W.
       Inputs::
           dBm(float): power in units of dBm
       Returns::
           Power in units of W (float)
    """
    return 1e-3*10**((dBm)/10.)


def w2dbm(W):
    """This function converts a power given in W to a power given in dBm.
       Inputs::
           W(float): power in units of W
       Returns::
           Power in units of dBm(float)
    """
    a = 10.*np.log10(W)+30
    try:   
        a[a == -np.inf] = -100
        a[a == np.inf]  = -100
        a[a <= -100] = -100
    except:
        pass

    return a


def mfft(a):
    return scfft(a.T).T


def imfft(a):
    return scifft(a.T).T


def mmfft(a):
    return mkfft(a.T).T


def immfft(a):
    return imkfft(a.T).T


class raman_object(object):
    def __init__(self,a,b = None):
        self.on = a
        self.how = b
        self.hf = None 
    
    def raman_load(self,t,dt):
        if self.on == 'on':
            print('Raman on')
            if self.how == 'analytic':
                print(self.how)
                t11 = 12.2e-3      # [ps]
                t2 = 32e-3         # [ps]
                htan = (t11**2 + t2**2)/(t11*t2**2)*np.exp(-t/t2*(t>=0))*np.sin(t/t11)*(t>=0)   # analytical response
                hf = mfft(htan)   # Fourier transform of the analytic nonlinear response
            elif self.how == 'load':
                print('loading for silica')
                # loads the measured response (Stolen et al. JOSAB 1989)
                mat = loadmat('loading_data/silicaRaman.mat')
                ht = mat['ht']
                t1 = mat['t1']        
                htmeas_f = InterpolatedUnivariateSpline(t1*1e-3,ht)
                htmeas = htmeas_f(t)     
                htmeas *=(t>0)*(t<1)    # only measured between +/- 1 ps)
                htmeas /= (dt*np.sum(htmeas))    # normalised
                hf = mfft(htmeas)   # Fourier transform of the measured nonlinear response
            else:
                hf = None
            self.hf = hf
        return self.hf   


def dispersion_operator(lamda_c,int_fwm,sim_wind):
    """
    Calculates the dispersion operator in rad/m units
    """
    alpha = int_fwm.alphadB/4.343
    c_norm = c*1e-12                                                                        #Speed of light [m/ps]                                                                         #Central wavelength [nm]
    beta0 = 0
    beta1 = 0
    beta2 = 0
    beta3 = 6.75e-5
    beta4 = -1e-7
    

    
    wc = 2*pi * c_norm /sim_wind.lamda
    w0 = 2*pi * c_norm / lamda_c

    beta2 += beta3 * (wc - w0) + 0.5*beta4 * (wc - w0)**2
    beta3 = beta3
    beta4 = beta4


    w = sim_wind.w # + sim_wind.woffset

    betap = np.zeros([int_fwm.nm,5])

    betap[0,0] = beta0
    betap[0,1] = beta1
    betap[0,2] = beta2
    betap[0,3] = beta3
    betap[0,4] = beta4
    
    Dop = np.zeros([int_fwm.nt,int_fwm.nm],dtype=np.complex)
    Dop[:,:] = -alpha/2
    Dop[:,0] -= 1j*(betap[0,0] +  betap[0,1]*(w) + (betap[0,2]*(w)**2)/2. + (betap[0,3]*(w)**3)/6.+ (betap[0,4]*(w)**3)/6.)
    return Dop

"""
def dispersion_operator_polyval(nm,fmed,D01,D11,S01,S11,dbeta0,dbeta1,lamda_c,alphadB,w0,w,lv,lamda):
    alpha = alphadB/4.343
    c_norm = c                                                                        #Speed of light [m/ps]                                                                         #Central wavelength [nm]
    D = 1e-12*1e6*np.array([D01,D11])                                                             #[ps/m**2]
    S = 1e-12*1e15*np.array([S01,S11])                                                            #[ps/m**3]
    beta2 = -D[:]*(lamda_c**2/(2*pi*c_norm))                                                #[ps**2/m]
    beta3 = lamda_c**4*S[:]/(4*(pi*c_norm)**2)+lamda_c**3*D[:]/(2*(pi*c_norm)**2) 

    b0_01 = 0
    b0_11 = b0_01 + dbeta0
    b1_01 = 0
    b1_11 = b1_01 + dbeta1
    b2_01,b2_11 = beta2[0], beta2[1]
    b3_01,b3_11 = beta3[0], beta3[1]
    p01 = [b3_01/6,b2_01/2, b1_01, b0_01];
    p11 = [b3_11/6,b2_11/2, b1_11, b0_11];

    k1 = np.polyval(p01,2*pi*c/(lv*1e-9)-w0);
    k2 = np.polyval(p11,2*pi*c/(lv*1e-9)-w0);

    beta = [k1, k2]
    lambda2 = lv*1e-9

    NN = 21
    bn = np.zeros([nm,len(w)],dtype=np.complex128)
    p = np.zeros([nm,22],dtype=np.complex128)
    for ii in range(nm):
        bn[ii,:] = InterpolatedUnivariateSpline((2*pi*c/lambda2 - 2*pi*fmed)*1e-12, beta[ii][:])(fftshift(w))
        p[ii,:] = np.polyfit((2*pi*c/lambda2 - 2*pi*c/lamda )*1e-12,beta[ii][:],NN); # obtains the expansion coefficients in w - wmed in ps^n/m
        
        bn[ii,:] = bn[ii,:] - p[0,NN] - p[0,NN-1]*fftshift(w)
        bn[ii,:] = fftshift(bn[ii,:])
    Dop = -alpha/2 -1j*bn
    return Dop.T
"""
def plotter_dbm(nm,lv,power_watts,xl,t,u,xtlim,which):
        
    fig = plt.figure(figsize=(20.0, 10.0))
    for ii in range(nm):
        plt.plot(lv,np.real(power_watts[:,ii]) - np.max(np.real(np.real(power_watts[:,ii]))),'-*',label='mode'+str(ii))
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.xlabel(r'$\lambda (nm)$')
    plt.ylabel(r'$Spectrum time space (dBm)$')
    plt.title("wavelength space")
    plt.grid()
    plt.xlim(xl)
    #plt.legend()
    
    plt.savefig("figures/wavelength_space"+str(which))
    #plt.show()
    plt.close(fig)
    fig = plt.figure(figsize=(20.0, 10.0))
    for ii in range(nm):
        plt.plot(t,np.abs(u[:,ii,which])**2,'*-',label='mode'+str(ii))
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.title("time space")
    plt.grid()
    plt.xlabel(r'$t(ps)$')
    plt.ylabel(r'$Spectrum$')
    plt.xlim(xtlim)
    #plt.legend()
    plt.savefig("figures/time_space"+str(which))

    #plt.close(fig)
    return 0 

def plotter(nm,lv,U,xl,t,u,xtlim,which):
    fig = plt.figure(figsize=(20.0, 10.0))
    for ii in range(nm):
        plt.plot(lv,np.real(U[:,ii,which]),'*-',label='mode'+str(ii))
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.grid()
    plt.xlabel(r'$\lambda (nm)$',fontsize=18)
    plt.ylabel(r'$Spectrum_time_space_(dBm)$',fontsize=18)
    plt.title("wavelength_space")
    plt.xlim(xl)
    plt.legend()
    plt.close(fig)

    fig = plt.figure(figsize=(20.0, 10.0))
    for ii in range(nm):
        plt.plot(t,np.abs(u[:,ii,which])**2,'*-',label='mode'+str(ii))
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.grid()
    plt.title("time_space")
    plt.xlabel(r'$t(ps)$',fontsize=18)
    plt.ylabel(r'$Spectrum$',fontsize=18)
    plt.xlim(xtlim)
    plt.legend()
    plt.close(fig)
    return 0 


def plotter_dbm_lams_large(modes,lv,U,xl,t,xtlim,which,lams_vec):
    fig = plt.figure(figsize=(20.0, 10.0))
    for mode in modes:     
        for lamm,lamda in enumerate(lams_vec):    
            plt.plot(lv,np.real(U[lamm,:,mode]),'-*',label=str(mode))
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.xlabel(r'$\lambda (nm)$',fontsize=18)
    plt.ylabel(r'$Spectrum time space (dBm)$',fontsize=18)
    plt.xlim(xl)
    plt.grid()
    plt.ylim(-60,0)
    plt.legend()
    plt.savefig("figures/wavelength_space_large.png",fontsize=18)
    plt.close('all')
    return 0


def animator(lv,t,P0_f_sparce,U,zv,xl):    
    def animate(i):
        line.set_ydata(w2dbm(np.real(U[:,0,i])/P0_f_sparce)- np.max(w2dbm(np.real(U[:,0,0])/P0_f_sparce)))  # update the data
        line2.set_ydata(w2dbm(np.real(U[:,1,i])/P0_f_sparce)- np.max(w2dbm(np.real(U[:,0,0])/P0_f_sparce)))
        return line,line2


    def init():
        line.set_ydata(np.ma.array(t, mask=True))
        return line,


    fig, ax = plt.subplots()
    i = range(len(zv))
    line,  = ax.plot(lv,w2dbm(np.real(U[:,0,0])/P0_f_sparce) - np.max(w2dbm(np.real(U[:,0,0])/P0_f_sparce)),label= 'mode0')
    line2, = ax.plot(lv,w2dbm(np.real(U[:,1,0])/P0_f_sparce) - np.max(w2dbm(np.real(U[:,0,0])/P0_f_sparce)),label= 'mode1')
    ax.set_xlabel(r'$\lambda (nm)$')
    ax.set_ylabel('Spectrum (dBm)')
    ax.set_xlim(xl[0],xl[1])
    ax.set_ylim(-60,0)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(zv)), init_func=init,
                                  interval=1000, blit=True)
    ani.save('figures/basic_animation.mp4', fps=1, extra_args=['-vcodec', 'libx264'])
    plt.show()
    return 0


def energy_conservation(entot):
    if not(np.allclose(entot, entot[0])):
        fig = plt.figure()
        plt.plot(entot)
        plt.grid()
        plt.xlabel("nplots(snapshots)",fontsize=18)
        plt.ylabel("Total energy",fontsize=18)
        plt.show()
        sys.exit("energy is not conserved")
    return 0


################################Grid check for fft optimisation########################
def check_ft_grid(fv):
    if np.log2(np.shape(fv)[0]) == int(np.log2(np.shape(fv)[0])):
        print("------------------------------------------------------------------------------------")
        print("All is good with the grid for fft's:", np.shape(fv)[0])
        nt = np.shape(fv)[0]
    else:
        print("fix the grid for optimization of the fft's, grid:", np.shape(fv)[0])
        sys.exit()
    return 0
########################################################################################

"""---------- Q Matrices ----------------------------------"""
def Q_matrixes(nm,n2,lamda,gama=None):    
    if nm==1:
        mat  = loadmat('loading_data/M1_M2_1m_new.mat') #loads M1 and M2 matrices
        M1 = np.real(mat['M1'])
        M2 = mat['M2']
        M2[:,:] -=1 
        M1[0:4] -=1
        M1[-1] -=1
        if gama != None:
            M1[4] = gama /(3*n2*(2*pi/lamda)) 
            M1[5] = gama /(3*n2*(2*pi/lamda)) 
            
    if nm==2:
        #mat  = loadmat('M1_M2_'+str(nm)+'m')  # loads M1 and M2 matrices
        mat = loadmat("loading_data/M1_M2_new_2m.mat")
        M1 = np.real(mat['M1'])
        M2 = mat['M2']
        M2[:] -=1 
        M1[:4,:] -=1 
        M1[6,:] -=1

        
    return M1,M2

class mode1_mod(object):
    def __init__(self,power_watts,where,lv,pos,nt):
        self.pos = 2*where[0] - pos
        self.pow = power_watts[self.pos,0]
        self.lam = lv[self.pos]


class mode1_mod2:
    def __init__(self,power_watts,where,lv,pos,nt):
        self.pos = where[0] - 2*(where[0] - pos)
        self.pow = power_watts[self.pos,0]
        self.lam = lv[self.pos]


class mode2_pc:
    def __init__(self,power_watts,where,lv,pos,nt):
        self.pos = nt - where[1] + (where[0] - pos)
        self.pow = power_watts[self.pos,1]
        self.lam = lv[self.pos]


class mode2_bs:
    def __init__(self,power_watts,where,lv,pos,nt):
        self.pos = nt - where[1] - (where[0] - pos)
        self.pow = power_watts[self.pos,1]
        self.lam = lv[self.pos]


class sim_parameters(object):
    def __init__(self,n2,nm,alphadB):
        self.n2 = n2
        self.nm = nm
        self.alphadB = alphadB


    def general_options(self,maxerr,ss='1',ram='on',how='load'):
        self.maxerr = maxerr
        self.ss = ss
        self.raman = raman_object(ram,how)
        return None


    def propagation_parameters(self,N,z,nplot,dz_less,wavelength_space):
        self.N = N
        self.nt = 2**self.N
        self.z = z
        self.nplot = nplot
        self.dzstep = self.z/self.nplot
        self.dz = self.dzstep/dz_less
        self.wavelength_space = wavelength_space
        return None


class sim_window(object):
    def __init__(self,fv,lamda,lamda_c,int_fwm):
        self.lamda = lamda
        self.lmin = 1e-3*c/np.max(fv)                                                #[nm]
        self.lmax = 1e-3*c/np.min(fv)                                                #[nm]
        self.lv = 1e-3*c/fv                                                          #[nm]
        self.fmed = c/(lamda)                                                  #[Hz]
        self.deltaf = 1e-3*(c/self.lmin - c/self.lmax)                                         #[THz]
        self.df = self.deltaf/int_fwm.nt                                                 #[THz]
        self.T = 1/self.df                                                                #Time window (period)[ps]
        self.woffset = 2*pi*(self.fmed - c/lamda)*1e-12                                   #[rad/ps]
        self.woffset2 = 2*pi*(self.fmed - c/lamda_c)*1e-12                                #[rad/ps] Offset of central freequency and that of the experiment  
        self.xl = np.array([self.lmin, self.lmax])                                             # wavelength limits (for plots) (nm)
        self.w0 = 2*pi*self.fmed                                                          # central angular frequency [rad/s]
        self.tsh = 1/self.w0*1e12                                                         # shock time [ps]
        self.dt = self.T/int_fwm.nt                                                               #timestep (dt)     [ps]
        self.t = (range(int_fwm.nt)-np.ones(int_fwm.nt)*int_fwm.nt/2)*self.dt                                     #time vector       [ps]
        self.w = 2*pi*np.append(range(0,int(int_fwm.nt/2)),range(int(-int_fwm.nt/2),0,1))/self.T          #angular frequency vector [rad/ps]          
        self.vs = fftshift(self.w/(2*pi))                     # frequency vector[THz] (shifted for plotting)
        self.lv = c/(self.fmed+self.vs*1e12)*1e9                   # wavelength vector [nm]
        self.zv = int_fwm.dzstep*np.asarray(range(0,int_fwm.nplot+1))    # space vector [m]
        self.xtlim =np.array([-self.T/2, self.T/2])  # time limits (for plots)


#class WDM1(object):
#    def port1(U)

#class WDM(object):
#    def __init__(self,func1,funt2):
#        self.port1 = func1
#        self.port2 = func2
#        return None


def lams_s_vary(wave,s_pos,from_pump,int_fwm,sim_wind,where,P0_p1,P0_s,Dop,M1,M2):   
        if from_pump:
            s_pos = where[0] - wave
        else:
            s_pos -= wave
        u = np.zeros([len(sim_wind.t),int_fwm.nm,len(sim_wind.zv)],dtype='complex128')    # initialisation (for fixed steps)
        U = np.zeros([len(sim_wind.t),int_fwm.nm,len(sim_wind.zv)],dtype='complex128')    #

        pquant = np.sum(1.054e-34*(sim_wind.w*1e12 + sim_wind.w0)/(sim_wind.T*1e-12))  # Quantum noise (Peter's version)
        noise = (pquant/2)**0.5*(np.random.randn(int_fwm.nm,int_fwm.nt) + 1j*np.random.randn(int_fwm.nm,int_fwm.nt))

        u[:,:,0] = noise.T
        u[:,0,0] += (P0_p1)**0.5
        
        woff1 = -(s_pos - where[0])*2*pi*sim_wind.df
        u[:,0,0] += (P0_s)**0.5 * np.exp(-1j*(woff1)*sim_wind.t)

        U[:,:,0] = fftshift(np.abs(sim_wind.dt*mfft(u[:,:,0]))**2,(0,1,1))
        print(U[s_pos,0,0])

        "----------------------Plot the inputs------------------------------------"
        #plotter_dbm(int_fwm.nm,sim_wind.lv,w2dbm(U),sim_wind.xl,sim_wind.t,u,sim_wind.xtlim,0)
        #sys.exit()
        "-------------------------------------------------------------------------"

        int_fwm.raman.raman_load(sim_wind.t,sim_wind.dt) # bring the raman if needed
        string = "dAdzmm_r"+str(int_fwm.raman.on)+"_s"+str(int_fwm.ss)
        #dAdzmm = eval(string)
        func_dict = {'dAdzmm_ron_s1':dAdzmm_ron_s1,
             'dAdzmm_ron_s0':dAdzmm_ron_s0,
             'dAdzmm_roff_s0':dAdzmm_roff_s0,
             'dAdzmm_roff_s1':dAdzmm_roff_s1}
        dAdzmm = func_dict[string]
        hf = int_fwm.raman.hf
        "--------------------------Pulse propagation--------------------------------"
        badz = 0        #counter for bad steps
        goodz = 0       #counter for good steps
        dztot = 0       #total distance traveled
        dzv = np.zeros(1)   
        dzv[0] = int_fwm.dz
        u1 = np.copy(u[:,:,0])
        energy = np.NaN*np.ones([int_fwm.nm,int_fwm.nplot+1])
        entot = np.NaN*np.ones(int_fwm.nplot+1)
        for ii in range(int_fwm.nm):
            energy[ii,0] = np.linalg.norm(u1[:,ii],2)**2  # energy per mode
        dz = int_fwm.dz * 1
        entot[0,]= np.sum(energy[:,0]**1)         # total energy (must be conserved)  
        for jj in range(int_fwm.nplot):
            exitt = False
            while not(exitt):
                delta = 2*int_fwm.maxerr                       # trick to do the first iteration
                while delta > int_fwm.maxerr:    
                    u1new = imfft(np.exp(Dop*dz/2)*mfft(u1))
                    u1new = np.ascontiguousarray(u1)
                    A, delta = RK5mm(dAdzmm,u1new,dz,M1,M2,sim_wind.t,int_fwm.n2,sim_wind.lamda,sim_wind.tsh,sim_wind.w,sim_wind.woffset,sim_wind.dt,hf) # calls a 5th order Runge Kutta routine
                    if (delta > int_fwm.maxerr):
                        dz *= (int_fwm.maxerr/delta)**0.25   # calculate the step (shorter) to redo
                        badz += 1
                #####################################Successful step##############################################
                u1 = imfft(np.exp(Dop*dz/2)*mfft(A))                           # propagate the remaining half step             
                dztot += dz                                                    # update the propagated distance
                goodz += 1                                                     # update the number of steps taken
                dzv = np.append(dzv,dz)                                        # store the dz just taken
                dz2 = np.min([0.95*dz*(int_fwm.maxerr/delta)**0.2,0.95*int_fwm.dzstep])        # calculate the next step (longer)                                                      # without exceeding max dzstep
                ###################################################################################################
                
                if dztot == (int_fwm.dzstep*(jj+1)):
                    exitt = True

                elif ((dztot + dz2 )>= int_fwm.dzstep*(jj+1)):
                    dz2 = int_fwm.dzstep*(jj+1) - dztot   
                dz = np.copy(dz2) 
                ###################################################################################################
            
            u[:,:,jj+1] = u1
            U[:,:,jj+1] = fftshift(np.abs(sim_wind.dt*mfft(u[:,:,jj+1]))**2,(0,1,1))
            for ii in range(int_fwm.nm):
                energy[ii,jj+1] = norm(u1[:,ii],2)**2 # energy per mode
            entot[jj+1] = np.sum(energy[:,jj+1])             # total energy
        print(delta,badz)
        "-------------------------------------------------------------------------------------------------------------------"
        power_dbm = w2dbm(np.abs(U[:,:,-1]))
        max_norm = np.max(power_dbm[:,0])
        power_dbm[:,0] -= max_norm

        "Looking for the peaks themselves for all peaks of"
        modulation = mode1_mod(power_dbm, where,sim_wind.lv,s_pos,int_fwm.nt)
        print('---------------------------------------------')
        print('Pump1     :  ', sim_wind.lv[where[0]], power_dbm[where[0],0])
        print('Signal     : ', sim_wind.lv[s_pos], power_dbm[s_pos,0])
        print('idler: ',modulation.lam, modulation.pow)
        print('----------------------------------------------')
        return power_dbm[:,:],s_pos, modulation,max_norm
