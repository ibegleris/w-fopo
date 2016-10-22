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
ifftshift = scipy.fftpack.ifftshift
from math import isinf
import pickle
from scipy.integrate import simps
import matplotlib as mpl
mpl.use('Agg')
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


def plotter_dbm(nm,lv,power_watts,xl,t,u,xtlim,which,filename=None,title=None,im=None):
        
    fig = plt.figure(figsize=(20.0, 10.0))
    for ii in range(nm):
        plt.plot(lv,np.real(power_watts[:,ii,which]),'-*',label='mode'+str(ii))
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.xlabel(r'$\lambda (nm)$')
    plt.ylabel(r'$Spectrum time space (dBm)$')
    plt.title(title)
    plt.grid()
    plt.xlim(900,1200)
    plt.ylim(-100,80)
    try:
        if im != None:
            #im = plt.imread(D)
            newax = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor='NE')
            newax.imshow(im)
            newax.axis('off')
    except FutureWarning:
        pass
    plt.legend()
    if filename == None:
        plt.savefig("figures/wavelength_space"+str(which),bbox_inched='tight')
    else:
        plt.savefig("figures/wavelength_space"+filename,bbox_inched='tight')
    #plt.show()
    plt.close(fig)
    #fig = plt.figure(figsize=(20.0, 10.0))
    #for ii in range(nm):
    #    plt.plot(t,np.abs(u[:,ii,which])**2,'*-',label='mode'+str(ii))
    #plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    #plt.title("time space")
    #plt.grid()
    #plt.xlabel(r'$t(ps)$')
    #plt.ylabel(r'$Spectrum$')
    #plt.xlim(xtlim)
    #plt.legend()
    #plt.savefig("figures/time_space"+str(which))

    plt.close(fig)
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
    #plt.ylim(-60,0)
    plt.legend()
    plt.savefig("figures/wavelength_space_large.png",fontsize=18,bbox_inched='tight')
    plt.close('all')
    return 0


def energy_conservation(entot):
    if not(np.allclose(entot, entot[0])):
        fig = plt.figure()
        plt.plot(entot)
        plt.grid()
        plt.xlabel("nplots(snapshots)",fontsize=18)
        plt.ylabel("Total energy",fontsize=18)
        #plt.show()
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
        self.lv = 1e-3*c/fv                                                         #[nm]
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



class WDM(object):
    def __init__(self,x1,x2):
        self.x1 = x1 # High part of port 1 
        self.x2 = x2 # Low wavelength of port 1
        self.omega = 0.5*pi/np.abs(x1 - x2)
        self.phi = pi - self.omega*self.x2
        return None


    def il_port1(self,lamda):
        return (np.sin(self.omega*lamda+self.phi))**2


    def il_port2(self,lamda):
        return (np.cos(self.omega*lamda+self.phi))**2

    def wdm_port1_pass(self,U,sim_wind):
        U[:,0] *= (self.il_port1(sim_wind.lv))**0.5
        u = imfft(ifftshift(U[:,:],axes=(0,))/sim_wind.dt)
        U_true = fftshift(np.abs(sim_wind.dt*mfft(u[:,:]))**2,axes=(0,))
        return u,U,U_true

    def wdm_port2_pass(self,U,sim_wind):
        U[:,0] *= (self.il_port2(sim_wind.lv))**0.5
        u = imfft(ifftshift(U[:,:],axes=(0,))/sim_wind.dt)
        U_true = fftshift(np.abs(sim_wind.dt*mfft(u[:,:]))**2,axes=(0,))
        return u,U,U_true

    def plot(self,lamda):
        fig = plt.figure()
        plt.plot(lamda,self.il_port1(lamda), label = "%0.2f" % (self.x1*1e9) +' nm port') 
        plt.plot(lamda,self.il_port2(lamda), label =  "%0.2f" % (self.x2*1e9) +' nm port')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),ncol=2)
        plt.xlabel(r'$\lambda (\mu m)$')
        plt.ylabel(r'$Insertion loss (dB)$')
        plt.savefig('figures/WDM_high_'+str(self.x1)+'_low_'+str(self.x2)+'.png')
        #plt.show()
        return None


    def plot_dB(self,lamda):
        fig = plt.figure()
        plt.plot(lamda,10*np.log10(self.il_port1(lamda)), label =  "%0.2f" % (self.x1*1e9) +' nm port') 
        plt.plot(lamda,10*np.log10(self.il_port2(lamda)), label =  "%0.2f" % (self.x2*1e9) +' nm port')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=2)
        plt.xlabel(r'$\lambda (\mu m)$')
        plt.ylabel(r'$Insertion loss (dB)$')
        plt.ylim(-60,0)
        plt.savefig('figures/WDM_dB_high_'+str(self.x1)+'_low_'+str(self.x2)+'.png')
        #plt.show()
        return None


def lams_s_vary(wave,s_pos,from_pump,int_fwm,sim_wind,where,P0_p1,P0_s,Dop,M1,M2):   
        if from_pump:
            s_pos = where[0] - wave
        else:
            s_pos -= wave
        u = np.zeros([len(sim_wind.t),int_fwm.nm,len(sim_wind.zv)],dtype='complex128')    # initialisation (for fixed steps)
        U = np.zeros([len(sim_wind.t),int_fwm.nm,len(sim_wind.zv)],dtype='complex128')    #
        U_true = np.copy(U)
        pquant = np.sum(1.054e-34*(sim_wind.w*1e12 + sim_wind.w0)/(sim_wind.T*1e-12))  # Quantum noise (Peter's version)
        noise = (pquant/2)**0.5*(np.random.randn(int_fwm.nm,int_fwm.nt) + 1j*np.random.randn(int_fwm.nm,int_fwm.nt))

        u[:,:,0] = noise.T
        u[:,0,0] += (P0_p1)**0.5

        U[:,:,0] = fftshift(sim_wind.dt*mfft(u[:,:,0]),axes=(0,))
        U_true[:,:,0] = fftshift(np.abs(sim_wind.dt*mfft(u[:,:,0]))**2,axes=(0,))
        "----------------------Plot the inputs------------------------------------"
        #plotter_dbm(int_fwm.nm,sim_wind.lv,w2dbm(U),sim_wind.xl,sim_wind.t,u,sim_wind.xtlim,0)
        #sys.exit()
        "-------------------------------------------------------------------------"

        int_fwm.raman.raman_load(sim_wind.t,sim_wind.dt) # bring the raman if needed
        string = "dAdzmm_r"+str(int_fwm.raman.on)+"_s"+str(int_fwm.ss)
        func_dict = {'dAdzmm_ron_s1':dAdzmm_ron_s1,
             'dAdzmm_ron_s0':dAdzmm_ron_s0,
             'dAdzmm_roff_s0':dAdzmm_roff_s0,
             'dAdzmm_roff_s1':dAdzmm_roff_s1}
        pulse_pos_dict_or = ( 'after propagation', "pass WDM2", "pass WDM1 on port2 (remove pump)", 'add more pump','out')
        

        keys = ['loading_data/green_dot_fopo/pngs/'+str(i)+str('.png') for i in range(7)]
        D_pic = [plt.imread(i) for i in keys]
        

        dAdzmm = func_dict[string]
        hf = int_fwm.raman.hf

        #Define te WDMs
        WDM1 = WDM(1050, 1200)
        WDM2 = WDM(1200, 930)
        WDM1.plot_dB(sim_wind.lv)
        WDM2.plot_dB(sim_wind.lv)
        WDM1.plot(sim_wind.lv)
        WDM2.plot(sim_wind.lv)
        #sys.exit()

        
        plotter_dbm(int_fwm.nm,sim_wind.lv,w2dbm(U_true),sim_wind.xl,sim_wind.t,u,sim_wind.xtlim,0,'0','original pump',D_pic[0])
        
        #Pass the original pump through the WDM1 port1
        u[:,:,0],U[:,:,0], U_true[:,:,0] = WDM1.wdm_port1_pass(U[:,:,0],sim_wind)
        

        at_WDM_in = U[:,:,0]
        
        rounds = 200

        for ro in range(rounds):
            pulse_pos_dict = ['round '+ str(ro)+', ' + i for i in pulse_pos_dict_or]
            plotter_dbm(int_fwm.nm,sim_wind.lv,w2dbm(U_true),sim_wind.xl,sim_wind.t,u,sim_wind.xtlim,0,str(ro)+'1',pulse_pos_dict[3],D_pic[5])
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
                U[:,:,jj+1] = fftshift(sim_wind.dt*mfft(u[:,:,jj+1]),axes=(0,))
                U_true[:,:,jj+1] = fftshift(np.abs(sim_wind.dt*mfft(u[:,:,jj+1]))**2,axes=(0,))
                for ii in range(int_fwm.nm):
                    energy[ii,jj+1] = norm(u1[:,ii],2)**2 # energy per mode
                entot[jj+1] = np.sum(energy[:,jj+1])             # total energy
            print(delta,badz)
            "-------------------------------------------------------------------------------------------------------------------"
           
        

            plotter_dbm(int_fwm.nm,sim_wind.lv,w2dbm(U_true),sim_wind.xl,sim_wind.t,u,sim_wind.xtlim,-1,str(ro)+'2',pulse_pos_dict[0],D_pic[2])
            print('round', ro)


            # pass through WDM2 to get the signal only
            u[:,:,-1],U[:,:,-1], U_true[:,:,-1] = WDM2.wdm_port1_pass(U[:,:,-1],sim_wind)
            plotter_dbm(int_fwm.nm,sim_wind.lv,w2dbm(U_true),sim_wind.xl,sim_wind.t,u,sim_wind.xtlim,-1,str(ro)+'3',pulse_pos_dict[1],D_pic[3])



            # To see what is out
            b,a, U_true[:,:,-1] = WDM2.wdm_port2_pass(U[:,:,-1],sim_wind)
            plotter_dbm(int_fwm.nm,sim_wind.lv,w2dbm(U_true),sim_wind.xl,sim_wind.t,u,sim_wind.xtlim,-1,str(ro)+'5',pulse_pos_dict[3],D_pic[6])

            # Pass through the WDM1 port2 to get rid of its pump
            u[:,:,-1],U[:,:,-1], U_true[:,:,-1] = WDM1.wdm_port2_pass(U[:,:,-1],sim_wind)
            plotter_dbm(int_fwm.nm,sim_wind.lv,w2dbm(U_true),sim_wind.xl,sim_wind.t,u,sim_wind.xtlim,-1,str(ro)+'4',pulse_pos_dict[2],D_pic[4])


            # Add the original pump 
            U[:,:,-1] += at_WDM_in


            u[:,:,-1] = imfft(ifftshift(U[:,:,-1],axes=(0,))/sim_wind.dt)
            U_true[:,:,-1] = fftshift(np.abs(sim_wind.dt*mfft(u[:,:,-1]))**2,axes=(0,))

            

            u[:,:,0] = imfft(ifftshift(U[:,:,-1],axes=(0,))/sim_wind.dt)
            U_true[:,:,0] = U_true[:,:,-1]
            
 
       

        U[:,:,-1] = fftshift(np.abs(sim_wind.dt*mfft(u[:,:,0]))**2,axes=(0,))
        power_dbm = w2dbm(np.abs(U[:,:,-1]))
        max_norm = np.max(power_dbm[:,0])
        power_dbm -= max_norm

        #"Looking for the peaks themselves for all peaks of"
        modulation = mode1_mod(power_dbm, where,sim_wind.lv,s_pos,int_fwm.nt)
        #print('---------------------------------------------')
        #print('Pump1     :  ', sim_wind.lv[where[0]], power_dbm[where[0],0])
        #print('Signal     : ', sim_wind.lv[s_pos], power_dbm[s_pos,0])
        #print('idler: ',modulation.lam, modulation.pow)
        #print('----------------------------------------------')
        return power_dbm[:,:],s_pos, modulation,max_norm,rounds
