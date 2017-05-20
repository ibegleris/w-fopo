from __future__ import division, print_function
import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
from scipy.constants import c, pi
from joblib import Parallel, delayed
from scipy.fftpack import fftshift, fft
import os
import time as timeit
os.system('export FONTCONFIG_PATH=/etc/fonts')
from functions import *
from fft_module import *
import sys
from time import time,sleep
def oscilate(sim_wind,int_fwm,noise_obj,TFWHM_p, TFWHM_s,index,master_index,P0_p1,P0_s, f_p, f_s,p_pos,s_pos,splicers_vec,
			WDM_vec,M, hf, Dop, dAdzmm,D_pic,pulse_pos_dict_or,plots):
	u = np.zeros(
		[len(sim_wind.t), len(sim_wind.zv)], dtype='complex128')
	U = np.zeros([len(sim_wind.t),
				  len(sim_wind.zv)], dtype='complex128')	#
	
	T0_p = TFWHM_p/2/(np.log(2))**0.5
	T0_s = TFWHM_s/2/(np.log(2))**0.5
	noise_new = noise_obj.noise_func(int_fwm)
	u[:, 0] = noise_new

	woff1 = (p_pos+(int_fwm.nt)//2)*2*pi*sim_wind.df
	u[:, 0] += (P0_p1)**0.5  * np.exp(1j*(woff1)*sim_wind.t)


	woff2 = -(s_pos - (int_fwm.nt-1)//2)*2*pi*sim_wind.df
	u[:, 0] += (P0_s)**0.5 * np.exp(-1j*(woff2)*sim_wind.t)#*np.exp(-sim_wind.t**2/T0_s)


	U[:, 0] = fftshift(fft(u[:,0]))
	
	sim_wind.w_tiled = sim_wind.w + sim_wind.woffset
	master_index = str(master_index)


	

	plotter_dbm(index,int_fwm.nm, sim_wind, u, U, P0_p1,
				P0_s, f_p, f_s, 0,0,0,0,master_index, '00', 'original pump', D_pic[0],plots)
	# Splice1
	#(u[:, :, 0], U[:, :, 0]) = splicers_vec[1].pass_through(
	#	(U[:, :, 0],noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind)[0]

	# Splice2
	#(u[:, :, 0], U[:, :, 0])  = splicers_vec[1].pass_through(
	#	(U[:, :, 0], noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind)[0]

	# Splice3
	#(u[:, :, 0], U[:, :, 0]) = splicers_vec[1].pass_through(
	#	(U[:, :, 0], noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind)[0]
	

	U_original_pump = np.copy(U[:, 0])

	# Pass the original pump through the WDM1, port1 is in to the loop, port2 junk
	noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
	u[:, 0], U[:, 0] = WDM_vec[0].pass_through(
		(U[:, 0], noise_new), sim_wind)[0]
	

	max_rounds = int(sys.argv[1])
	ro = -1
	min_circ_error = 0.01 # relative percentage error in power
	P_portb,P_portb_prev = 3*min_circ_error ,min_circ_error
	rel_error = 100
	while ro < max_rounds:

		P_portb_prev = P_portb
		ro +=1
		print('round', ro)
		pulse_pos_dict = [
			'round ' + str(ro)+', ' + i for i in pulse_pos_dict_or]

		plotter_dbm(index,int_fwm.nm, sim_wind, u, U, P0_p1,
					P0_s, f_p, f_s, 0, ro,P_portb,rel_error,master_index, str(ro)+'1', pulse_pos_dict[3], D_pic[5],plots)

		# Splice3
		#noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
		#(u[:, 0], U[:, 0]) = splicers_vec[1].pass_through(
		#	(U[:, 0], noise_new), sim_wind)[0]

		u, U = pulse_propagation(
			u, U, int_fwm, M, sim_wind, hf, Dop, dAdzmm)
	
		plotter_dbm(index,int_fwm.nm, sim_wind, u, U, P0_p1,
					P0_s, f_p, f_s, -1,ro,P_portb,rel_error,master_index, str(ro)+'2', pulse_pos_dict[0], D_pic[2],plots)
		"""
		# Splice4
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
		(u[:, -1], U[:, -1]) = splicers_vec[1].pass_through(
			(U[:, -1], noise_new), sim_wind)[0]

		# Splice5
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
		(u[:, -1], U[:, -1]) = splicers_vec[1].pass_through(
			(U[:, -1], noise_new), sim_wind)[0]

		# Splice6
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
		(u[:, -1], U[:, -1]) = splicers_vec[1].pass_through(
			(U[:, -1], noise_new), sim_wind)[0]
		"""
		# pass through WDM2 port 2 continues and port 1 is out of the loop
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
		(out1, out2),(u[:, -1], U[:, -1])  = WDM_vec[1].pass_through(
			(U[:, -1], noise_new), sim_wind)
		
	
		
		
		plotter_dbm(index,int_fwm.nm, sim_wind, u, U, P0_p1,
					P0_s, f_p, f_s, -1, ro,P_portb,rel_error,master_index,str(ro)+'3', pulse_pos_dict[1], D_pic[3],plots)


		"""
		# Splice7 after WDM2 for the signal
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
		(u[:, -1], U[:, -1]) = splicers_vec[1].pass_through(
			(U[:, -1], noise_new), sim_wind)[0]
		"""


		# Splice7 after WDM2 for the signal
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
		(u[:, -1], U[:, -1]) = splicers_vec[2].pass_through(
			(U[:, -1], noise_new), sim_wind)[0]


		# Pass again through WDM1 with the signal now
		(u[:, 0], U[:, 0]) = WDM_vec[0].pass_through(
			(U_original_pump, U[:, -1]), sim_wind)[0]
		
		
		################################The outbound stuff#####################
		U_out = np.reshape(out2, (len(sim_wind.t), 1))
		u_out = np.reshape(out1, (len(sim_wind.t), 1))
		plotter_dbm(index,int_fwm.nm, sim_wind, u_out, U_out, P0_p1,
					P0_s, f_p, f_s, -1, ro,P_portb,rel_error,master_index,str(ro)+'4', pulse_pos_dict[4], D_pic[6],plots)
		"""
		# Splice8 before WDM3
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
		(out1, out2) = splicers_vec[1].pass_through(
			(out2, noise_new), sim_wind)[0]
		"""

		# WDM3 port 1 continues and port 2 is portA in experiment
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
		(utemp, Utemp),(u_portA, U_portA)  = WDM_vec[2].pass_through(
			(out2, noise_new), sim_wind)
		
		
		U_portA = np.reshape(U_portA, (len(sim_wind.t), 1))
		u_portA = np.reshape(u_portA, (len(sim_wind.t), 1))
		
		
		plotter_dbm(index,int_fwm.nm, sim_wind , u_portA,
			U_portA, P0_p1, P0_s, f_p, f_s, -1, ro,P_portb,rel_error,master_index,'portA/'+str(ro),
			'round '+str(ro)+', portA',plots=plots)
		"""
		# Splice9 before WDM4
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
		(out1, out2)= splicers_vec[1].pass_through(
			(out2, noise_new), sim_wind)[0]
		"""
		# WDM4 port 1 goes to port B and port 2 to junk
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
		(u_portB, U_portB)  = WDM_vec[3].pass_through(
			(out2, noise_new), sim_wind)[0]
		

		
		U_portB = np.reshape(U_portB, (len(sim_wind.t), 1))
		u_portB = np.reshape(u_portB, (len(sim_wind.t), 1))
  		
		plotter_dbm(index,int_fwm.nm, sim_wind, u_portB,
					U_portB, P0_p1, P0_s, f_p, f_s, -1, ro,P_portb,rel_error,master_index,'portB/'+str(ro),
					'round '+str(ro)+', portB',plots=plots)

		fv_id = idler_limits(sim_wind, U_portB)
		P_portb = power_idler(U_portB,sim_wind.fv,sim_wind.T,fv_id)
		rel_error = 100*np.abs(P_portb - P_portb_prev)/P_portb_prev
		
	return None

#'num_cores':num_cores, 'maxerr':maxerr, 'ss':ss, 'ram':ram, 'plots': plots
#					'N':N, 'nt':nt,'nplot':nplot}


def formulate(index,n2,gama, alphadB, z, P_p, P_s, TFWHM_p,TFWHM_s,spl_losses,betas,
				lamda_c, WDMS_pars, lamp, lams,num_cores, maxerr, ss, ram, plots,
				 N, nt, nplot,master_index):
	"------------------propagation paramaters------------------"
	dzstep = z/nplot						# distance per step
	dz_less = 1e10
	#dz = dzstep/dz_less		 # starting guess value of the step
	int_fwm = sim_parameters(n2, 1, alphadB)
	int_fwm.general_options(maxerr, raman_object, ss, ram)
	int_fwm.propagation_parameters(N, z, nplot, dz_less)
	lamda = lamp*1e-9  # central wavelength of the grid[m]
	fv_idler_int = 10 #safety for the idler to be spotted used only for idler power
	"-----------------------------f-----------------------------"
	

	"---------------------Aeff-Qmatrixes-----------------------"
	M = Q_matrixes(int_fwm.nm, int_fwm.n2, lamda, gama)
	"----------------------------------------------------------"


	"---------------------Grid&window-----------------------"
	fv, where = fv_creator(lamp,lams,int_fwm,prot_casc =0)
	p_pos,s_pos = where
	sim_wind = sim_window(fv, lamda, lamda_c, int_fwm,fv_idler_int)
	"----------------------------------------------------------"


	"---------------------Loss-in-fibres-----------------------"
	slice_from_edge = (sim_wind.fv[-1] - sim_wind.fv[0])/100
	loss = Loss(int_fwm, sim_wind, amax=5)

	int_fwm.alpha = loss.atten_func_full(fv)

	"----------------------------------------------------------"


	"--------------------Dispersion----------------------------"
	Dop = dispersion_operator(betas, lamda_c, int_fwm, sim_wind)
	"----------------------------------------------------------"
	

	"--------------------Noise---------------------------------"
	pquant = np.sum(1.054e-34*(sim_wind.w*1e12 + sim_wind.w0)/
				(sim_wind.T*1e-12))
	noise_obj = Noise(sim_wind)
	"----------------------------------------------------------"


	"---------------Formulate the functions to use-------------"
	string = "dAdzmm_r"+str(int_fwm.ram)+"_s"+str(int_fwm.ss)
	func_dict = {'dAdzmm_ron_s1': dAdzmm_ron_s1,
				 'dAdzmm_ron_s0': dAdzmm_ron_s0,
				 'dAdzmm_roff_s0': dAdzmm_roff_s0,
			 	'dAdzmm_roff_s1': dAdzmm_roff_s1}
	pulse_pos_dict_or = ('after propagation', "pass WDM2",
						 "pass WDM1 on port2 (remove pump)",
						 'add more pump', 'out')

	keys = ['loading_data/green_dot_fopo/pngs/' +
		str(i)+str('.png') for i in range(7)]
	D_pic = [plt.imread(i) for i in keys]

	dAdzmm = func_dict[string]
	raman = raman_object(int_fwm.ram, int_fwm.how)
	raman.raman_load(sim_wind.t, sim_wind.dt)
	hf = raman.hf
	"--------------------------------------------------------"

	"----------------------Formulate WDMS--------------------"
	if WDMS_pars == 'signal_locked':

		Omega = 2*pi*c/(lamp*1e-9) - 2*pi*c/(lams*1e-9) 
		omegai = 2*pi*c/(lamp*1e-9) +Omega
		lami = 1e9*2*pi*c/(omegai)
		WDMS_pars = ([lamp, lams+20], 	# WDM up downs in wavelengths [m]
					[lami, lams ],
					[lami,lamp],
					[lami, lams])
	#print(WDMS_pars)
	#sys.exit()
	#print(lamp)
	WDMS_pars = ([lamp, 1200], 	# WDM up downs in wavelengths [m]
				[930,  1200],
				[930,lamp],
				[930, 1200])
	WDM_vec = [WDM(i[0], i[1],sim_wind.fv,c) for i in WDMS_pars]# WDM up downs in wavelengths [m]


	"--------------------------------------------------------"
	#for ei,i in enumerate(WDM_vec):
	#	i.plot(filename = str(ei))
	"----------------------Formulate splicers--------------------"
	splicers_vec = [Splicer(loss = i) for i in spl_losses]
	"------------------------------------------------------------"

	f_p,f_s = 1e-3*c/lamp, 1e-3*c/lams

	oscilate(sim_wind,int_fwm,noise_obj,TFWHM_p, TFWHM_s,index,master_index,P_p,P_s, f_p, f_s,p_pos,s_pos,splicers_vec,
			WDM_vec,M, hf, Dop, dAdzmm,D_pic,pulse_pos_dict_or,plots)

	
	#while cop !=0:
	#	sleep(0.5)
	return None





def main():
	"-----------------------------Stable parameters----------------------------"
	num_cores = 6							# Number of computing cores for sweep
	maxerr = 1e-13							# maximum tolerable error per step in integration
	ss = 1					  				# includes self steepening term
	ram = 'on'				  				# Raman contribution 'on' if yes and 'off' if no
	plots = False 							# Do you want plots, be carefull it makes the code very slow!
	N = 14									# 2**N grid points
	nt = 2**N 								# number of grid points
	nplot = 2								# number of plots within fibre min is 2
	"--------------------------------------------------------------------------"
	stable_dic = {'num_cores':num_cores, 'maxerr':maxerr, 'ss':ss, 'ram':ram, 'plots': plots,
					'N':N, 'nt':nt,'nplot':nplot}
	"------------------------Can be variable parameters------------------------"
	n2 = 2.5e-20							# Nonlinear index [m/W]
	gama = 10e-3 							# Overwirtes n2 and Aeff w/m
	alphadB = 0#0.0011667#666666666668		# loss within fibre[dB/m]
	z = 18									# Length of the fibre
	P_p = np.arange(3,5.2,0.2)				# Pump power [W]
	#P_p = [4.3]
	#P_p = [5.5,]
	P_s = 0*100e-3#[10e-3,100e-3,1]							# Signal power [W]
	TFWHM_p = 0								# full with half max of pump
	TFWHM_s = 0								# full with half max of signal
	spl_losses = [[0,0,1.],[0,0,1.2],[0,0,1.3],[0,0,1.4]]					# loss of each type of splices [dB] 
	spl_losses = [0,0,1.4]
	betas = np.array([0, 0, 0, 6.756e-2,	# propagation constants [ps^n/m]
			-1.002e-4, 3.671e-7])*1e-3								
	lamda_c = 1051.85e-9		
				# Zero dispersion wavelength [nm]
	#max at ls,li = 1095, 1010
	variation = [-30,-20,-10,0,10,20,30]
	#WDMS_pars = ([1051.5, 1095], 	# WDM up downs in wavelengths [m]
	#			[1011.4,  1095],
	#			[1011.4,1051.5],
	#			[1011.4, 1095])
	WDMS_pars = ([1048.17107345, 1200.39], 	# WDM up downs in wavelengths [m]
				[930,  1200.39],
				[930,1048.17107345],
				[930, 1200.39])
	

	#WDMS_pars = []
	#for i in variation:
	#	WDMS_pars.append(([1051.5, 1095+i], 	# WDM up downs in wavelengths [m]
	#					[1011.4,  1095],
	#					[1011.4,1051.5],
	#					[1011.4, 1095])) 
	#print(WDMS_pars)
	#sys.exit()
	#WDMS_pars = 'signal_locked' # lockes the WDMS to keep the max amount of signal in the cavity (seeded)

		

	lamp = [1048.17107345, 1047.1]							# Pump wavelengths [nm]
	#lamp = [1047.5,1047.9,1048.3,1048.6,1049,1049.5,1049.8,1050.2,1050.6,1051,1051.4]
	#lamp = [1050,1050.5,1051,1051.5]
	lams = 1200.39#np.arange(1093, 1097, 0.5)#[:-1]
	#lams = [1094,1095,1096]
	#lams = [1085,1115]
	#lams = [lams[1]]									# Signal wavelength [nm]
	var_dic = {'n2':n2, 'gama':gama, 'alphadB':alphadB, 'z':z, 'P_p':P_p,
				 'P_s':P_s,'TFWHM_p':TFWHM_p, 'TFWHM_s':TFWHM_s,
				 'spl_losses':spl_losses, 'betas':betas,
				  'lamda_c':lamda_c, 'WDMS_pars':WDMS_pars,
				   'lamp':lamp, 'lams':lams}
	"--------------------------------------------------------------------------"
	outside_var_key = 'lamp'
	inside_var_key = 'P_p'
	#outside_var_key, inside_var_key = inside_var_key, outside_var_key
	inside_var = var_dic[inside_var_key]
	outside_var = var_dic[outside_var_key]
	del var_dic[outside_var_key]
	del var_dic[inside_var_key]
	"----------------------------Simulation------------------------------------"
	D_ins = [{'index':i, inside_var_key: insvar} for i,insvar in enumerate(inside_var)]


	large_dic = {**stable_dic, **var_dic}

	if len(inside_var) < num_cores:
		num_cores = len(inside_var)
	
	#division_of_cores = len()//max_num_cores
	#while division_of_cores !=
	#num_cores = max_num_cores
	profiler_bool = float(sys.argv[2])
	for kk,variable in enumerate(outside_var):
		create_file_structure(kk)

		_temps = create_destroy(inside_var,str(kk))
		_temps.prepare_folder()
		large_dic ['master_index'] = kk
		large_dic [outside_var_key] = variable
		#formulate(**{**D_ins[0],** large_dic})
		if profiler_bool:
			#print('profiling')
			for i in range(len(D_ins)):
				formulate(**{**D_ins[i],** large_dic})
		else:
			#print('not profiling')
			A = Parallel(n_jobs = num_cores)(delayed(formulate)(**{**D_ins[i],** large_dic}) for i in range(len(D_ins)))
		_temps.cleanup_folder()
	

	print('\a')
	return None

if __name__ == '__main__':
	start = time()
	main()
	dt = time() - start
	print(dt, 'sec', dt/60, 'min', dt/60/60, 'hours')
