from __future__ import division, print_function
import numpy as np
import matplotlib as mpl
from scipy.fftpack import fft, ifft
mpl.use('Agg')
from scipy.constants import c, pi
from scipy.io import savemat
from joblib import Parallel, delayed
from scipy.fftpack import fftshift, ifftshift,ifft,fft
import multiprocessing
import sys
import os
import time as timeit
os.system('export FONTCONFIG_PATH=/etc/fonts')
from functions import *
from fft_module import *

def lams_s_vary(wave, s_pos, from_pump, int_fwm, sim_wind,
				where, P0_p1, P0_s, Dop, M,index,plots,var_loss,pump_wave = ''):
	if from_pump:
		s_pos = where[0] - wave
	else:
		s_pos -= wave
   
	# initialisation (for fixed steps)
	u = np.zeros(
		[len(sim_wind.t), len(sim_wind.zv)], dtype='complex128')
	U = np.zeros([len(sim_wind.t),
				  len(sim_wind.zv)], dtype='complex128')	#
	# Quantum noise (Peter's version)
	pquant = np.sum(
		1.054e-34*(sim_wind.w*1e12 + sim_wind.w0)/(sim_wind.T*1e-12))
	noise_obj = Noise(sim_wind)

	noise = noise_obj.noise_func(int_fwm)

	TFWHM = 0.04
	T0 = TFWHM/2/(np.log(2))**0.5
	u[:, 0] = noise

	u[:, 0] += (P0_p1)**0.5  # *np.exp(-sim_wind.t**2/T0)

	woff1 = -(s_pos - int_fwm.nt//2)*2*pi*sim_wind.df
	u[:, 0] += (P0_s)**0.5 * np.exp(-1j*(woff1)*sim_wind.t)

	U[:, 0] = fftshift(sim_wind.dt*fft(u[:,0]))
	sim_wind.w_tiled = sim_wind.w
	#Uabs[:, :, 0] = fftshift(np.abs(sim_wind.dt*fft(u[:, :, 0]))**2, axes=(0,))

	"----------------------Plot the inputs------------------------------------"
	#plotter_dbm(int_fwm.nm,sim_wind.lv,w2dbm(U),sim_wind.xl,sim_wind.t,u,sim_wind.xtlim,0)
	f_p,f_s = sim_wind.fv[int_fwm.nt//2], sim_wind.fv[s_pos]
	#plotter_dbm(index,int_fwm.nm, sim_wind, w2dbm(Uabs), u, U, P0_p1,
	#			P0_s, f_p, f_s, 0)
	#plotter_dbm(int_fwm.nm,sim_wind,w2dbm(Uabs),u,U, P0_p1, P0_s, f_p, f_s,0,filename=None,title=None,im = 0)

	"-------------------------------------------------------------------------"

	# bring the raman if needed
	
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
	#sys.exit()
	plotter_dbm(index,int_fwm.nm, sim_wind, u, U, P0_p1,
				P0_s, f_p, f_s, 0,0,0,0,pump_wave, '00', 'original pump', D_pic[0],plots)


	# Define te WDM objects
	#WDM1 = WDM(1050, 1200, sim_wind.fv,c)
	#WDM2 = WDM(930, 1200, sim_wind.fv, c)
	#WDM3 = WDM(930, 1050, sim_wind.fv, c)
	#WDM4 = WDM(930, 1200, sim_wind.fv, c)
	l_p, l_s,l_i  =1048.17107345, 1200.39, 930
	WDM1 = WDM(l_p, l_s,sim_wind.fv,c)
	WDM2 = WDM(l_i, l_s,sim_wind.fv,c)
	WDM3 = WDM(l_i, l_p,sim_wind.fv,c)
	WDM4 = WDM(l_i, l_s,sim_wind.fv,c)
	#WDM1.plot(sim_wind.lv, True)
	#WDM2.plot(sim_wind.lv, True)
	#WDM3.plot(sim_wind.lv, True)
	#WDM4.plot(sim_wind.lv, True)
	#WDM1.plot_dB(sim_wind.lv)
	#WDM2.plot_dB(sim_wind.lv)
	#WDM3.plot_dB(sim_wind.lv)
	#WDM4.plot_dB(sim_wind.lv)

	
	# Define the splicer object
	splicer1 = Splicer(loss=0.4895)
	splicer2 = Splicer(loss=0.142225011896)
	splicer1 = Splicer(loss=0.)
	splicer2 = Splicer(loss=0.)
	print(var_loss)
	splicer3 = Splicer(loss = var_loss)
	# Pass the original pump through its 2 splice losses.



	# Splice1
	#(u[:, :, 0], U[:, :, 0]) = splicer2.pass_through(
	#	(U[:, :, 0],noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind)[0]

	# Splice2
	#(u[:, :, 0], U[:, :, 0])  = splicer2.pass_through(
	#	(U[:, :, 0], noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind)[0]

	# Splice3
	#(u[:, :, 0], U[:, :, 0]) = splicer2.pass_through(
	#	(U[:, :, 0], noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind)[0]
	

	U_original_pump = np.copy(U[:, 0])

	# Pass the original pump through the WDM1, port1 is in to the loop, port2 junk
	noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind, fft)
	u[:, 0], U[:, 0] = WDM1.pass_through(
		(U[:, 0], noise_new), sim_wind)[0]
	
	#from scipy.integrate import simps
	#power = simps(np.abs(U_original_pump[:,0])**2, sim_wind.fv)/(2*np.max(sim_wind.t))
	#print(power)
	max_rounds = 20
	ro = -1
	min_circ_error = 0.01 # relative percentage error in power
	P_portb,P_portb_prev = 3*min_circ_error ,min_circ_error

	rel_error = 100*np.abs(P_portb - P_portb_prev)/P_portb_prev
	#t3 = time()
	while ro < max_rounds:# and rel_error  > min_circ_error:
		#print(P_portb, 100*np.abs(P_portb - P_portb_prev)/P_portb_prev)
		P_portb_prev = P_portb
		ro +=1
		print('round', ro)
		pulse_pos_dict = [
			'round ' + str(ro)+', ' + i for i in pulse_pos_dict_or]
		#t1 = time()
		plotter_dbm(index,int_fwm.nm, sim_wind, u, U, P0_p1,
					P0_s, f_p, f_s, 0, ro,P_portb,rel_error,pump_wave, str(ro)+'1', pulse_pos_dict[3], D_pic[5],plots)
		#t2 = time() - t1
		#print('Time of writting:', t2)
		# Splice3
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind, fft)
		(u[:, 0], U[:, 0]) = splicer1.pass_through(
			(U[:, 0], noise_new), sim_wind)[0]
		#t1 = time()
		u, U = pulse_propagation(
			u, U, int_fwm, M, sim_wind, hf, Dop, dAdzmm)
		#t2 = time() - t1
		#print('Time of pulse_prop:', t2)
		#sys.exit()
		plotter_dbm(index,int_fwm.nm, sim_wind, u, U, P0_p1,
					P0_s, f_p, f_s, -1,ro,P_portb,rel_error,pump_wave, str(ro)+'2', pulse_pos_dict[0], D_pic[2],plots)
		
		# Splice4
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind, fft)
		(u[:, -1], U[:, -1]) = splicer1.pass_through(
			(U[:, -1], noise_new), sim_wind)[0]

		# Splice5
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind, fft)
		(u[:, -1], U[:, -1]) = splicer2.pass_through(
			(U[:, -1], noise_new), sim_wind)[0]

		# Splice6
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind, fft)
		(u[:, -1], U[:, -1]) = splicer2.pass_through(
			(U[:, -1], noise_new), sim_wind)[0]
		
		# pass through WDM2 port 2 continues and port 1 is out of the loop
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind, fft)
		(out1, out2),(u[:, -1], U[:, -1])  = WDM2.pass_through(
			(U[:, -1], noise_new), sim_wind)
		
		
		
		plotter_dbm(index,int_fwm.nm, sim_wind, u, U, P0_p1,
					P0_s, f_p, f_s, -1, ro,P_portb,rel_error,pump_wave,str(ro)+'3', pulse_pos_dict[1], D_pic[3],plots)



		# Splice7 after WDM2 for the signal
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind, fft)
		(u[:, -1], U[:, -1]) = splicer2.pass_through(
			(U[:, -1], noise_new), sim_wind)[0]



		# Splice7 after WDM2 for the signal
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind, fft)
		(u[:, -1], U[:, -1]) = splicer3.pass_through(
			(U[:, -1], noise_new), sim_wind)[0]


		# Pass again through WDM1 with the signal now
		(u[:, 0], U[:, 0]) = WDM1.pass_through(
			(U_original_pump, U[:, -1]), sim_wind)[0]
		
		
		################################The outbound stuff#####################
		U_out = np.reshape(out2, (len(sim_wind.t), 1))
		u_out = np.reshape(out1, (len(sim_wind.t), 1))
		plotter_dbm(index,int_fwm.nm, sim_wind, u_out, U_out, P0_p1,
					P0_s, f_p, f_s, -1, ro,P_portb,rel_error,pump_wave,str(ro)+'4', pulse_pos_dict[4], D_pic[6],plots)
		
		# Splice8 before WDM3
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind, fft)
		(out1, out2) = splicer2.pass_through(
			(out2, noise_new), sim_wind)[0]
		

		# WDM3 port 1 continues and port 2 is portA in experiment
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind, fft)
		(utemp, Utemp),(u_portA, U_portA)  = WDM3.pass_through(
			(out2, noise_new), sim_wind)
		
		
		U_portA = np.reshape(U_portA, (len(sim_wind.t), 1))
		u_portA = np.reshape(u_portA, (len(sim_wind.t), 1))
		
		
		plotter_dbm(index,int_fwm.nm, sim_wind , u_portA,
			U_portA, P0_p1, P0_s, f_p, f_s, -1, ro,P_portb,rel_error,pump_wave,'portA/'+str(ro),
			'round '+str(ro)+', portA',plots=plots)
		
		# Splice9 before WDM4
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind, fft)
		(out1, out2)= splicer2.pass_through(
			(out2, noise_new), sim_wind)[0]
		
		# WDM4 port 1 goes to port B and port 2 to junk
		noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind, fft)
		(u_portB, U_portB)  = WDM4.pass_through(
			(out2, noise_new), sim_wind)[0]
		

		
		U_portB = np.reshape(U_portB, (len(sim_wind.t), 1))
		u_portB = np.reshape(u_portB, (len(sim_wind.t), 1))
  		
		plotter_dbm(index,int_fwm.nm, sim_wind, u_portB,
					U_portB, P0_p1, P0_s, f_p, f_s, -1, ro,P_portb,rel_error,pump_wave,'portB/'+str(ro),
					'round '+str(ro)+', portB',plots=plots)

		fv_id = idler_limits(sim_wind, U_portB)
		P_portb = power_idler(U_portB,sim_wind.fv,sim_wind.T,fv_id)
		rel_error = 100*np.abs(P_portb - P_portb_prev)/P_portb_prev
	#tend = time() - t3
	#print('time of round trip:', tend)
	#sys.exit()
	return None



def lam_p2_vary(lam_s_max,pump_index, lam_p1, power_pump_input,power_signal_input, int_fwm, plot_conv, gama, where_save,fv_idler_int,plots,var_loss,par=False, grid_only=False, timing=False,pump_wave=''):

	P0_p1 = power_pump_input  # [w]
	P0_s = power_signal_input  # [w]

	lamda = lam_p1*1e-9  # central wavelength of the grid[m]
	# 1052.95e-9		  #central freequency of the dispersion
	#lamda_c = 1052.95e-9	#fast axis
	#lamda_c = 1052.44e-9	#average axis
	lamda_c = 1051.85e-9	#slow axis
	"----------------------Obtain the Q matrixes------------------------------"
	M = Q_matrixes(int_fwm.nm, int_fwm.n2, lamda, gama)
	"-------------------------------------------------------------------------"
	#print(lam_p1)
	fv, where = fv_creator(700, lam_p1, int_fwm)
	sim_wind = sim_window(fv, lamda, lamda_c, int_fwm,fv_idler_int)

	if grid_only:
		return sim_wind


	#int_fwm.alphadB = 0.0011666666666666668
	slice_from_edge = (sim_wind.fv[-1] - sim_wind.fv[0])/8
	loss = Loss(int_fwm, sim_wind, amax=5)
	#loss.plot(fv)
	int_fwm.alpha = loss.atten_func_full(fv)
	"------------------------------Dispersion operator--------------------------------------"
	# betas at ps/m (given in ps^n/km)
	#betas = np.array([[0, 0, 0, 6.753e-2, -1.001e-4, 2.753e-7]])*1e-3
	#betas = np.array([[0, 0, 0, 6.755e-2, -1.001e-4, 2.673e-7]])*1e-3
	betas = np.array([0, 0, 0, 6.756e-2, -1.002e-4, 3.671e-7])*1e-3
	
	Dop = dispersion_operator(betas, lamda_c, int_fwm, sim_wind)

	"---------------------------------------------------------------------------------------"
	lam_s_max -= 2**(int_fwm.N-1)
	lam_s_max += 2
	#waves = range(100,lam_s_max,10)
	#waves = [1]
	waves = [lam_s_max]

	UU = np.zeros(
		[len(waves), len(sim_wind.t), int_fwm.nm], dtype='complex128')
	s_pos_vec = np.zeros(len(waves), dtype=int)
	mod_lam = np.zeros(len(waves))
	mod_pow = np.zeros(len(waves))
	P0_s_out = np.zeros(len(waves))
	par = 0
	if par:
		num_cores = 4
		print(
			"you have picked to run the signal wavelength in parallel. Make sure Mkl is dissabled.")
		res = Parallel(n_jobs=num_cores)(delayed(lams_s_vary)(wave, where[
			0], True, int_fwm, sim_wind, where, P0_p1, P0_s, Dop, M) for wave in waves)

		for i in range(len(waves)):
			UU[i, :, :] = res[i][0]
			s_pos_vec[i] = res[i][1]
			mod_pow[i] = res[i][2].pow
			mod_lam[i] = res[i][2].lam
			P0_s_out[i] = np.real(UU[i, s_pos_vec[i], 0])
	else:
		for ii, wave in enumerate(waves):
			#os.system('cp -r output output_dump/output'+str(ii))
			lams_s_vary(
				wave, where[0], True, int_fwm, sim_wind, where, P0_p1, P0_s, Dop, M,pump_index,plots,var_loss,pump_wave)
			P0_s_out[ii] = np.real(UU[ii, s_pos_vec[ii], 0])
			# break
	lams_vec = sim_wind.lv[s_pos_vec.astype(int)]
	lam_p2_nm = sim_wind.lv[-where[-1]]

	#plotter_dbm_lams_large([0], sim_wind, UU, -1, lams_vec)
	#if plots:
	#	animator_pdf_maker(max_rounds,pump_index)
	os.system('mv output'+pump_wave+'/output'+ str(pump_index)+' output_dump_'+where_save+'/output'+str(pump_index))
	return 0


def main():
	"-----------------------------Stable parameters----------------------------"
	n2 = 2.5e-20							# n2 for silica [m/W]
	nm = 1					  				# number of modes
	alphadB = 0*0.0011666666666666668			# loss within fibre[dB/m]
	gama = 10e-3 							# w/m
	num_cores = 6
	"-----------------------------General options------------------------------"
	maxerr = 1e-13							# maximum tolerable error per step
	ss = 1					  				# includes self steepening term
	ram = 'on'				  				# Raman contribution 'on' if yes and 'off' if no
	plots = False 							# Do you want plots, be carefull it makes the code very slow!
	"----------------------------Simulation parameters-------------------------"
	N = 13									# 2**N grid points 
	z = 18									# total distance [m]
	nplot = 2								# number of plots within fibre
	nt = 2**N 								# number of grid points
	dzstep = z/nplot						# distance per step
	dz_less = 1e2
	dz = dzstep/dz_less		 # starting guess value of the step
	wavelength_space = True		# Set wavelength space for grid calculation
	fv_idler_int = 10 		# [THz]
	int_fwm = sim_parameters(n2, nm, alphadB)
	int_fwm.general_options(maxerr, raman_object, ss, ram)
	int_fwm.propagation_parameters(N, z, nplot, dz_less, wavelength_space)

	#fft, ifft, fft_method = pick(N, nm, 100,num_cores)
	#print(fft)
	fft_method = 'intel_scipy'
	print('picked: ', fft_method, 'as a method')
	"---------------------FWM wavelengths----------------------------------------"
	lam_p1 = 1051.4  # [nm]
	lams_max_asked = 1250  # [nm]

	lv = lam_p2_vary(2, 0,lam_p1, 1,1,int_fwm, 0, gama,
			 'pump_wavelengths',fv_idler_int,False,var_loss = None,par = False,grid_only = True,timing= False).lv[::-1]

	lv_lams = np.abs(np.asanyarray(lv) - lams_max_asked)
	lams_index = np.where(lv_lams == np.min(lv_lams))[0][0]+1
	#print('S_max wavelength asked for: '+str(lams_max_asked),
	#	  'With this grid the best I can do is: ', lv[lams_index])
	lensig = np.shape(range(1, lams_index))[0]
	"----------------------------------------------------------------------------"
	
	#print(
	#	"The fft method that was found to be faster for your system is:", fft_method)
	os.system('rm output_dump_pump_powers'	)

	lamdaP_vec = [1047.4580276554484,1047.8542826528856,
				  1048.1710734507749,1048.236639751706,
				  1048.2626870941695,1048.7982221124762,
				  1048.8279348320191, 1048.9161470177778]
	lamdaP_vec = [1048.17107345,]
	#lamdaP_vec = [1048.92,]
	#lamdaP_vec = [1051.16397037,]
	var_loss = 0
	for kk,pp in enumerate(lamdaP_vec):
		#kk = sys.argv[1]
		create_file_structure(kk)
		#pump_wavelengths = (1.0488816316376193e-06*1e9,)
		pump_wavelengths = (pp,)
		#print(pump_wavelengths)
		#Power_inputs = np.arange(3.2,5,0.15)#np.arange(3.4,5,0.2)#np.arange(3.4,6,0.4)#np.arange(4.4,8.4,0.5)
		#Power_inputs = np.arange(4.8,5,1)
		#var_losses = np.linspace(0, 5, 30)
		#var_losses = np.arange(0.5,5.5,1)#1,2,3,4,5)
		var_losses = np.array([1.3])
		#var_losses = np.arange(1.46,1.48,0.005)
		#var_losses = np.arange(1.48,1.49,0.005)
		#var_losses = np.arange(1.495,1.605,0.005)
		#var_losses = np.array([1.3,1.4,1.5])
		#Power_inputs = np.copy(var_losses)
		#var_losses = [0]
		#Power_inputs = (3.5,4,4.5,5)
		Power_inputs = (5,)
		Power_signal = 0
		lam_p1 = pump_wavelengths[0]
		_power = create_destroy(Power_inputs,str(kk))
		_power.prepare_folder()

		Power_input = 4.3
		if len(var_losses) < num_cores:
			num_cores = len(var_losses)
		if num_cores > 1:
			#os.system("taskset -p 0xff %d" % os.getpid()) # fixes the numpy affinity problem
			A = Parallel(n_jobs=num_cores)(delayed(lam_p2_vary)(lensig,i,lam_p1,Power_input,Power_signal,int_fwm,1
								,gama,'pump_powers',fv_idler_int,plots,var_loss,par = False,grid_only = False,timing= False,pump_wave = str(kk))\
								 for i, var_loss in enumerate(var_losses))
		else:

			for i, var_loss in enumerate(var_losses):
				A = lam_p2_vary(lensig,i,lam_p1,Power_input,Power_signal,int_fwm,1
								,gama,'pump_powers',fv_idler_int,plots,var_loss,par = False,grid_only = False,timing= False,pump_wave = str(kk))
		

		"""
		####Power
		if num_cores > 1:
			#os.system("taskset -p 0xff %d" % os.getpid()) # fixes the numpy affinity problem
			A = Parallel(n_jobs=num_cores)(delayed(lam_p2_vary)(lensig,i,lam_p1,Power_input,Power_signal,int_fwm,1
								,gama,'pump_powers',fv_idler_int,plots,var_loss,par = False,grid_only = False,timing= False,pump_wave = str(kk))\
								 for i, Power_input in enumerate(Power_inputs))
		else:

			for i, Power_input in enumerate(Power_inputs):
				A = lam_p2_vary(lensig, i,lam_p1, Power_input,Power_signal, int_fwm, 1, gama,
				 'pump_powers',fv_idler_int,plots,var_loss,par=False, grid_only=False, timing=False,pump_wave=str(kk))
		"""

		_power.cleanup_folder()
		moved = 256
		os.system('mkdir output_dump_pump_powers/'+str(kk))
		moved = os.system('mv output_dump_pump_powers/output* output_dump_pump_powers/'+str(kk) )
		while moved != 0: 
			timeit.sleep(2)
		"""
		if kk !=0:
			#kk = sys.argv[1]
			create_file_structure(kk)
			#pump_wavelengths = (1.0488816316376193e-06*1e9,)
			pump_wavelengths = (pp*1e9,)
			print(pump_wavelengths)
			Power_inputs = (4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5)
			#Power_inputs = (6.5,)
			Power_signal = 0
			#Power_inputs = (4.5,5)
			#Power_inputs = (13,)
			#Power_inputs = tuple(np.arange(4,7,0.1))
			#print(np.shape(Power_inputs))
		
			#Power_inputs = (2.5,3.5,4.5,5.5)
			#Power_inputs = (10.5,)
			#Power_inputs = np.arange(4.4,8.6,0.15)
			#Power_inputs = (6,)
			#Power_inputs = tuple(np.arange(4,7.1,0.1))
			#Power_inputs = (0,)
			lam_p1 = pump_wavelengths[0]

			#Power_input = (13,)
			_power = create_destroy(Power_inputs,str(kk))
			_power.prepare_folder()
			#Power_inputs = (6,6.5,7)
			#iters = (6,7,8)
			#iters, Power_inputs = ()

			if num_cores > 1:
				os.system("taskset -p 0xff %d" % os.getpid()) # fixes the numpy affinity problem
				A = Parallel(n_jobs=num_cores)(delayed(lam_p2_vary)(lensig,i,lam_p1,Power_input,Power_signal,int_fwm,1
									,gama,'pump_powers',fv_idler_int,plots,par = False,grid_only = False,timing= False,pump_wave = str(kk))\
									 for i, Power_input in enumerate(Power_inputs))
			else:

				for i, Power_input in enumerate(Power_inputs):
					A = lam_p2_vary(lensig, i,lam_p1, Power_input,Power_signal, int_fwm, 1, gama,
					 'pump_powers',fv_idler_int,plots,par=False, grid_only=False, timing=False,pump_wave=str(kk))
			

			_power.cleanup_folder()
			moved = 256
			os.system('mkdir output_dump_pump_powers/'+str(kk))
			moved = os.system('mv output_dump_pump_powers/output* output_dump_pump_powers/'+str(kk) )
			while moved != 0: 
				timeit.sleep(2)
 			"""
	print('\a')
   	
	return None

	
	Power_input = 7
	Power_signal = 0
	#pump_wavelengths = (1047, 1047.5, 1047.9, 1048.3, 1048.6,
	#					1049.0, 1049.5, 1049.8, 1050.2, 1050.6, 1051.0, 1051.4)
	pump_wavelengths = (1047,)##1047.6,1047.8,1048.5,1048.9,1049.6)
	#pump_wavelengths = (1049.6,)
	#pump_wavelengths = ( 1050.6, 1051.0, 1051.4)
	#pump_wavelengths = (1050.6,)
	#pump_wavelengths = tuple(np.arange(1047,1052.6,0.2))
	#lamp_centr = 1.0488816316376193e-06*1e9
	#pump_wavelengths = np.linspace(lamp_centr - lamp_centr*0.001,lamp_centr + lamp_centr*0.001, 10 )

	#pump_wavelengths = (1051.8,1052)
	#iters = (24,25)
	_wavelength = create_destroy(pump_wavelengths)
	#zip(iters, pump_wavelengths)
	create_file_structure()
	var_loss = 0
	_wavelength.prepare_folder()
	#print(pump_wavelengths)
	if num_cores > 1:
		A = Parallel(n_jobs=num_cores)(delayed(lam_p2_vary)(lensig,i,lam_p1,Power_input,Power_signal,int_fwm,1
							,gama,'pump_wavelengths',fv_idler_int,plots,var_loss,par = False,grid_only = False,timing= False) for i, lam_p1 in enumerate(pump_wavelengths))
	else:
		print('going parrallel')
		for i,lam_p1 in enumerate(pump_wavelengths):
			A = lam_p2_vary(lensig,i,lam_p1,Power_input,Power_signal,int_fwm,1
								,gama,'pump_wavelengths',fv_idler_int,plots,var_loss,par = False,grid_only = False,timing= False)

	_wavelength.cleanup_folder()

		
	return None

from time import time
if __name__ == '__main__':
	start = time()
	main()
	dt = time() - start
	print(dt, 'sec', dt/60, 'min', dt/60/60, 'hours')





