from __future__ import division,print_function
import numpy as np
from time import time
from scipy.constants import c, pi
from functions import *
import os
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


def lam_p2_vary(lam_s_max,lam_p1,Power_input,int_fwm,plot_conv,par = False,grid_only = False,timing= False):      
	P0_p1 = Power_input 		                           				#[w]
	P0_s  = 0		                                                    #[w]

	lamda = lam_p1*1e-9                           #central wavelength of the grid[m]
	lamda_c = 1052.95e-9#1.5508e-6                           #central freequency of the dispersion
	"----------------------------dispersion_fwm_parameters---------------------------"

	gama = 10e-3#None					 # w/m
	"-------------------------------------------------------------------------------"

	"----------------------Obtain the Q matrixes------------------------------"
	M1,M2 = Q_matrixes(int_fwm.nm,int_fwm.n2,lamda,gama)
	"-------------------------------------------------------------------------"



	lam_start = 800

	print(int_fwm.N)

	f_p1 = 1e-3*c/lam_p1
	f_start = 1e-3*c/lam_start

	fv1 = np.linspace(f_start,f_p1,2**(int_fwm.N - 1))
		
	fv = np.ndarray.tolist(fv1)

	diff = fv[1] - fv[0]



	for i in range(2**(int_fwm.N -1)):
		fv.append(fv[-1]+diff)
	fv = np.asanyarray(fv)




	######################################################################################


	################################Grid check for fft optimisation########################
	check_ft_grid(fv)
	#######################################################################################


	##############################Where are the pumps on the grid and is the grid uniform?#########################
	where = [2**(int_fwm.N-1)] 
	lvio = []
	for i in range(len(fv)-1):
	    lvio.append(fv[i+1] - fv[i])
	    
	grid_error = np.asanyarray(lvio)[:] + diff
	if not(np.allclose(grid_error,0,rtol=0,atol=1e-13)):
	    print(grid_error)

	#####################################################################################################################

	sim_wind = sim_window(fv,lamda,lamda_c,int_fwm)
	if grid_only:
		return sim_wind

	"------------------------------Dispersion operator--------------------------------------"
	Dop = dispersion_operator(lamda_c,int_fwm,sim_wind)
	"---------------------------------------------------------------------------------------"


	lam_s_max -= 2**(int_fwm.N-1)
	lam_s_max +=2
	waves = range(1,lam_s_max)



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
			max_norm = res[i][3]
	else:
		for ii,wave in enumerate(waves):
		    UU[ii,:,:], s_pos_vec[ii], mod,max_norm,rounds = lams_s_vary(wave,where[0],True,int_fwm,sim_wind,where,P0_p1,P0_s,Dop,M1,M2)
		    mod_pow[ii] = mod.pow
		    mod_lam[ii] = mod.lam
		    P0_s_out[ii] = np.real(UU[ii,s_pos_vec[ii],0])
		    break
	lams_vec = sim_wind.lv[s_pos_vec.astype(int)]
	lam_p2_nm = sim_wind.lv[-where[-1]]

	plotter_dbm_lams_large([0],sim_wind.lv,UU,sim_wind.xl,sim_wind.t,sim_wind.xtlim,-1,lams_vec) 

	return mod_lam,lams_vec,max_norm,P0_s_out,mod_pow,rounds


def main():
	"------Stable parameters-----"
	n2 = 2.5e-20                # n2 for silica [m/W]
	nm = 1                      # number of modes
	alphadB = 0.0011666666666666668                 # loss [dB/m]
	"---------------------- General options -----------------------------------"

	maxerr = 1e-3            # maximum tolerable error per step (variable step size version)
	ss = 1                      # includes self steepening term
	ram = 'on'                  # Raman contribution 'on' if yes and 'off' if no
	
	"---------- Simulation parameters ----------------------------------"
	N = 11
	nt = 2**N 					# number of grid points, There is a check that makes sure that it is a power of 2 for the FFT
	z = 18					# total distance [m]
	nplot = 50                  # number of plots
	dzstep = z/nplot            # distance per step
	dz_less = 1e3
	dz = dzstep/dz_less         # starting guess value of the step
	wavelength_space = True    	# Set wavelength space for grid calculation


	int_fwm = sim_parameters(n2,nm,alphadB)
	int_fwm.general_options(maxerr,ss,ram)
	int_fwm.propagation_parameters(N, z, nplot, dz_less, wavelength_space)

	"""---------------------FWM wavelengths----------------------------------------"""
	Power_input = 13                      #[W] 


	lam_p1 = 1050                             #[nm]
	lams_max_asked = 1051

	lv = lam_p2_vary(2,lam_p1,Power_input,int_fwm,0,False,True).lv[::-1]
	lv_lams = np.abs(np.asanyarray(lv) - lams_max_asked)
	lams_index = np.where(lv_lams == np.min(lv_lams))[0][0]+1
	print("S_max wavelength asked for: "+str(lams_max_asked), "With this grid the best I can do is: ",lv[lams_index])
 
	lensig = np.shape(range(1,lams_index))[0]



	mod_lam,lams_vec,max_norm,P0_s_out,mod_pow,rounds = lam_p2_vary(lensig,lam_p1,Power_input,int_fwm,1,False)
	print('\a')

	os.system("rm figures/*.pdf")
	
	strings_large = ["convert figures/wavelength_space0.png "]
	#sys.exit()

	for i in range(6):
		strings_large.append("convert ")

	for ro in range(rounds):
		for i in range(5):
			strings_large[i+1] += "figures/wavelength_space"+str(ro)+str(i+1)+".png "
		for w in range(1,4):
			if i ==5:
				break
			strings_large[0] += "figures/wavelength_space"+str(ro)+str(w)+".png "
	
	for i in range(6):
		strings_large[0] += "figures/wavelength_space_large.png "
		os.system(strings_large[i]+"figures/wavelength_space"+str(i)+".pdf")
		#print("-------------------------------------------------------------")
		#print(i)
		#print(strings_large[i])
	
	#os.system("open figures/wavelength_space.pdf")
	os.system("rm figures/*.png")
	for i in range(6):
		os.system('convert -delay 30 figures/wavelength_space'+str(i)+'.pdf figures/wavelength_space'+str(i)+'.mp4')
	return None


if __name__ =='__main__':
	main()

