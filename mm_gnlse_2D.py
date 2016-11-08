from __future__ import division,print_function
import numpy as np
from time import time
from scipy.constants import c, pi
from functions import *
from sys import exit
from scipy.io import savemat
import scipy.fftpack
from joblib import Parallel, delayed
import multiprocessing
import sys
try:
    import mkl
    max_threads = mkl.get_max_threads()
    mkl.set_num_threads(max_threads)
except ImportError:
    print("MKL libaries help when you are not running in paralel. There is a free academic lisence by continuum analytics")
    pass



def lams_s_vary(wave,s_pos,from_pump,int_fwm,sim_wind,where,P0_p1,P0_s,Dop,M1,M2):   
	if from_pump:
	    s_pos = where[0] - wave
	else:
	    s_pos -= wave
	u = np.zeros([len(sim_wind.t),int_fwm.nm,len(sim_wind.zv)],dtype='complex128')    # initialisation (for fixed steps)
	U = np.zeros([len(sim_wind.t),int_fwm.nm,len(sim_wind.zv)],dtype='complex128')    #
	Uabs = np.copy(U)
	pquant = np.sum(1.054e-34*(sim_wind.w*1e12 + sim_wind.w0)/(sim_wind.T*1e-12))  # Quantum noise (Peter's version)
	noise = (pquant/2)**0.5*(np.random.randn(int_fwm.nm,int_fwm.nt) + 1j*np.random.randn(int_fwm.nm,int_fwm.nt))
	
	u[:,:,0] = noise.T
	u[:,0,0] += (P0_p1)**0.5

	U[:,:,0] = fftshift(sim_wind.dt*mfft(u[:,:,0]),axes=(0,))
	Uabs[:,:,0] = fftshift(np.abs(sim_wind.dt*mfft(u[:,:,0]))**2,axes=(0,))
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
	#WDM1 = WDM(1050, 1200)
	#WDM2 = WDM(1200, 930)
	#WDM1.plot_dB(sim_wind.lv)
	#WDM2.plot_dB(sim_wind.lv)
	#WDM1.plot(sim_wind.lv)
	#WDM2.plot(sim_wind.lv)
	#sys.exit()
	print(np.max(w2dbm(Uabs)))

	plotter_dbm(int_fwm.nm,sim_wind,w2dbm(Uabs),u,0,'0','original pump',D_pic[0])

	#Pass the original pump through the WDM1 port1
	#u[:,:,0],U[:,:,0], Uabs[:,:,0] = WDM1.wdm_port1_pass(U[:,:,0],sim_wind)


	#at_WDM_in = U[:,:,0]

	rounds = 1

	for ro in range(rounds):
		print('round', ro)

		pulse_pos_dict = ['round '+ str(ro)+', ' + i for i in pulse_pos_dict_or]
		plotter_dbm(int_fwm.nm,sim_wind,w2dbm(Uabs),u,0,str(ro)+'1',pulse_pos_dict[3],D_pic[5])
		u,U,Uabs = pulse_propagation(u,U,Uabs,int_fwm,M1,M2,sim_wind,hf,Dop,dAdzmm)




		#plotter_dbm(int_fwm.nm,sim_wind,w2dbm(Uabs),u,-1,str(ro)+'2',pulse_pos_dict[0],D_pic[2])
		

		# pass through WDM2 to get the signal only
		#u[:,:,-1],U[:,:,-1], Uabs[:,:,-1] = WDM2.wdm_port1_pass(U[:,:,-1],sim_wind)
		#plotter_dbm(int_fwm.nm,sim_wind,w2dbm(Uabs),u,-1,str(ro)+'3',pulse_pos_dict[1],D_pic[3])



		# To see what is out
		#b,a, Uabs[:,:,-1] = WDM2.wdm_port2_pass(U[:,:,-1],sim_wind)
		#plotter_dbm(int_fwm.nm,sim_wind,w2dbm(Uabs),-1,str(ro)+'5',pulse_pos_dict[3],D_pic[6])

		# Pass through the WDM1 port2 to get rid of its pump
		#u[:,:,-1],U[:,:,-1], Uabs[:,:,-1] = WDM1.wdm_port2_pass(U[:,:,-1],sim_wind)
		#plotter_dbm(int_fwm.nm,sim_wind,w2dbm(Uabs),-1,str(ro)+'4',pulse_pos_dict[2],D_pic[4])


		# Add the original pump 
		#U[:,:,-1] += at_WDM_in


		#u[:,:,-1] = imfft(ifftshift(U[:,:,-1],axes=(0,))/sim_wind.dt)
		#Uabs[:,:,-1] = fftshift(np.abs(sim_wind.dt*mfft(u[:,:,-1]))**2,axes=(0,))



		u[:,:,0] = imfft(ifftshift(U[:,:,-1],axes=(0,))/sim_wind.dt)
		Uabs[:,:,0] = Uabs[:,:,-1]




	#U[:,:,-1] = fftshift(np.abs(sim_wind.dt*mfft(u[:,:,0]))**2,axes=(0,))
	

	power_dbm = w2dbm(np.abs(Uabs[:,:,-1]))
	max_norm = np.max(power_dbm[:,0])
	return power_dbm,s_pos,max_norm,rounds



def lam_p2_vary(lam_s_max,lam_p1,Power_input,int_fwm,plot_conv,gama,par = False,grid_only = False,timing= False):      
	
	P0_p1 = Power_input 		  #[w]
	P0_s  = 0		              #[w]

	lamda = lam_p1*1e-9           #central wavelength of the grid[m]
	lamda_c = 1051.85e-9#1052.95e-9          #central freequency of the dispersion
	"----------------------Obtain the Q matrixes------------------------------"
	M1,M2 = Q_matrixes(int_fwm.nm,int_fwm.n2,lamda,gama)
	"-------------------------------------------------------------------------"
	fv,where = fv_creator(700,lam_p1,int_fwm)
	sim_wind = sim_window(fv,lamda,lamda_c,int_fwm)
	if grid_only:
		return sim_wind

	"------------------------------Dispersion operator--------------------------------------"
	betas = np.array([[0,0,0,6.755e-2,-1.001e-4]])*1e-3 # betas at ps/m (given in ps/km)
	Dop = dispersion_operator(betas,lamda_c,int_fwm,sim_wind)
	
	"---------------------------------------------------------------------------------------"
	lam_s_max -= 2**(int_fwm.N-1)
	lam_s_max +=2
	waves = range(1,lam_s_max)
	waves = [1]

	UU = np.zeros([len(waves),len(sim_wind.t),int_fwm.nm],dtype='complex128')
	s_pos_vec = np.zeros(len(waves),dtype=int)
	mod_lam = np.zeros(len(waves))
	mod_pow = np.zeros(len(waves))
	P0_s_out = np.zeros(len(waves))
	par = 0
	if par:
		num_cores = 4
		print("you have picked to run the signal wavelength in parallel. Make sure Mkl is dissabled.")
		res = Parallel(n_jobs=num_cores)(delayed(lams_s_vary)(wave,where[0],True,int_fwm,sim_wind,where,P0_p1,P0_s,Dop,M1,M2) for wave in waves)
		
		for i in range(len(waves)):
			UU[i,:,:] = res[i][0]
			s_pos_vec[i] = res[i][1]
			mod_pow[i] = res[i][2].pow
			mod_lam[i] = res[i][2].lam
			P0_s_out[i] = np.real(UU[i,s_pos_vec[i],0])
	else:
		for ii,wave in enumerate(waves):
		    UU[ii,:,:], s_pos_vec[ii], mod,rounds = lams_s_vary(wave,where[0],True,int_fwm,sim_wind,where,P0_p1,P0_s,Dop,M1,M2)
		    P0_s_out[ii] = np.real(UU[ii,s_pos_vec[ii],0])
		    break
	lams_vec = sim_wind.lv[s_pos_vec.astype(int)]
	lam_p2_nm = sim_wind.lv[-where[-1]]

	plotter_dbm_lams_large([0],sim_wind,UU,-1,lams_vec) 

	return mod_lam,lams_vec,P0_s_out,mod_pow,rounds


def main():
	"-----------------------------Stable parameters----------------------------"
	n2 = 2.5e-20                				# n2 for silica [m/W]
	nm = 1                      				# number of modes
	alphadB = 0#0.0011666666666666668             # loss [dB/m]
	gama = 10e-3 								# w/m
	Power_input = 13                      		#[W]
	"-----------------------------General options------------------------------"

	maxerr = 1e-6            	# maximum tolerable error per step
	ss = 1                      # includes self steepening term
	ram = 'off'                  # Raman contribution 'on' if yes and 'off' if no
	
	"----------------------------Simulation parameters-------------------------"
	N = 13
	z = 18				 	# total distance [m]
	nplot = 100                  # number of plots
	nt = 2**N 					# number of grid points
	dzstep = z/nplot            # distance per step
	dz_less = 1e3
	dz = dzstep/dz_less         # starting guess value of the step
	wavelength_space = True    	# Set wavelength space for grid calculation


	int_fwm = sim_parameters(n2,nm,alphadB)
	int_fwm.general_options(maxerr,ss,ram)
	int_fwm.propagation_parameters(N, z, nplot, dz_less, wavelength_space)

	"---------------------FWM wavelengths----------------------------------------"
	lam_p1 = 1051.85              #[nm]
	lams_max_asked = 1051	   	  #[nm]
	lv = lam_p2_vary(2,lam_p1,Power_input,int_fwm,0,gama,False,True).lv[::-1]
	lv_lams = np.abs(np.asanyarray(lv) - lams_max_asked)
	lams_index = np.where(lv_lams == np.min(lv_lams))[0][0]+1
	print('S_max wavelength asked for: '+str(lams_max_asked),
			'With this grid the best I can do is: ',lv[lams_index])
	lensig = np.shape(range(1,lams_index))[0]
	"----------------------------------------------------------------------------"


	mod_lam,lams_vec,P0_s_out,mod_pow,rounds = \
						lam_p2_vary(lensig,lam_p1,Power_input,int_fwm,1,
							gama,par = False,grid_only = False,timing= False)
	print('\a')


	return None


if __name__ =='__main__':
	main()


