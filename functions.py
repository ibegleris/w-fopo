# -*- coding: utf-8 -*-
from __future__ import division, print_function
import sys
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from scipy.linalg import norm
from scipy.constants import pi, c
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.integrate import simps
from scipy.fftpack import fftshift, ifftshift
from math import isinf, factorial
from integrand_and_rk import *
from data_plotters_animators import *
import cmath
import pandas as pd
phasor = np.vectorize(cmath.polar)
try:
	import accelerate
	jit = accelerate.numba.jit
	autojit = accelerate.numba.autojit
	from accelerate import numba
	vectorize, float64, complex128 = numba.vectorize, numba.float64, numba.complex128
except AttributeError:
	print(
		"install the accelerate packadge from anaconda or change \
		the source code ie remove references to @jit and accelerate imports")
	pass


def dbm2w(dBm):
	"""This function converts a power given in dBm to a power given in W.
	   Inputs::
		   dBm(float): power in units of dBm
	   Returns::
		   Power in units of W (float)
	"""
	return 1e-3*10**((dBm)/10.)


def w2dbm(W, floor=-100):
	"""This function converts a power given in W to a power given in dBm.
	   Inputs::
		   W(float): power in units of W
	   Returns::
		   Power in units of dBm(float)
	"""
	if type(W) != np.ndarray:
		if W > 0:
			return 10. * np.log10(W) + 30
		elif W == 0:
			return floor
		else:
			print(W)
			raise(ZeroDivisionError)
	a = 10. * (np.ma.log10(W)).filled(floor/10-3) + 30
	return a


class raman_object(object):

	def __init__(self, a, b=None):
		self.on = a
		self.how = b
		self.hf = None

	def raman_load(self, t, dt, fft, ifft):
		if self.on == 'on':
			print('Raman on')
			if self.how == 'analytic':
				print(self.how)
				t11 = 12.2e-3	  # [ps]
				t2 = 32e-3		 # [ps]
				# analytical response
				htan = (t11**2 + t2**2)/(t11*t2**2) * \
					np.exp(-t/t2*(t >= 0))*np.sin(t/t11)*(t >= 0)
				# Fourier transform of the analytic nonlinear response
				self.hf = fft(htan)
			elif self.how == 'load':
				# loads the measured response (Stolen et al. JOSAB 1989)
				mat = loadmat('loading_data/silicaRaman.mat')
				ht = mat['ht']
				t1 = mat['t1']
				htmeas_f = InterpolatedUnivariateSpline(t1*1e-3, ht)
				htmeas = htmeas_f(t)
				htmeas *= (t > 0)*(t < 1)	# only measured between +/- 1 ps)
				htmeas /= (dt*np.sum(htmeas))	# normalised
				# Fourier transform of the measured nonlinear response
				self.hf = fft(htmeas)
			else:
				self.hf = None
		
			return self.hf


def dispersion_operator(betas, lamda_c, int_fwm, sim_wind):
	"""
	Calculates the dispersion operator in rad/m units
	INputed are the dispersion operators at the omega0
	Local include the taylor expansion to get these opeators at omegac 
	Returns Dispersion operator
	"""
	#print(betas)
	c_norm = c*1e-12  # Speed of light [m/ps] #Central wavelength [nm]
	wc = 2*pi * c_norm / sim_wind.lamda
	w0 = 2*pi * c_norm / lamda_c
	betap = np.zeros_like(betas)
	for i in range(int_fwm.nm):
		for j in range(len(betas.T)):
			if j ==0:
				betap[i,j] = betas[i,j]
			fac = 0
			for k in range(j, len(betas.T)):
				betap[i, j] += (1/factorial(fac)) * \
					betas[i, k] * (wc - w0)**(fac)
				fac += 1

	#betap = betas
	#sys.exit()
	w = sim_wind.w # - wc + w0
	Dop = np.zeros([int_fwm.nt, int_fwm.nm], dtype=np.complex)
	alpha = np.reshape(int_fwm.alpha, np.shape(Dop))
	
	#print(betap)
	#sys.exit()
	Dop[:, :] = -fftshift(alpha/2)

	# set the fundemental betas as the one of the first mode
	beta0, beta1 = betap[0, 0], betap[0, 1]
	betap[:, 0] -= beta0
	betap[:, 1] -= beta1
	for i in range(int_fwm.nm):
		for j, bb in enumerate(betap[i, :]):
			Dop[:, i] -= 1j*(w**j * bb / factorial(j))
	return Dop


def Q_matrixes(nm, n2, lamda, gama=None):
	"Calculates the Q matrices from importing them from a file. CHnages the gama if given"
	if nm == 1:
		# loads M1 and M2 matrices
		mat = loadmat('loading_data/M1_M2_1m_new.mat')
		M1 = np.real(mat['M1'])
		M2 = mat['M2']
		M2[:, :] -= 1
		M1[0:4] -= 1
		M1[-1] -= 1
		gamma_or = 3*n2*(2*pi/lamda)*M1[4]
		if gama is not None:
			M1[4] = gama / (3*n2*(2*pi/lamda))
			M1[5] = gama / (3*n2*(2*pi/lamda))

	if nm == 2:
		mat = loadmat("loading_data/M1_M2_new_2m.mat")
		M1 = np.real(mat['M1'])
		M2 = mat['M2']
		M2[:] -= 1
		M1[:4, :] -= 1
		M1[6, :] -= 1
	return M1, M2


class sim_parameters(object):

	def __init__(self, n2, nm, alphadB):
		self.n2 = n2
		self.nm = nm
		self.alphadB = alphadB

	def general_options(self, maxerr, raman_object,ss='1', ram='on', how='load'):
		self.maxerr = maxerr
		self.ss = ss
		self.ram = ram
		self.how = how
		return None

	def propagation_parameters(self, N, z, nplot, dz_less, wavelength_space):
		self.N = N
		self.nt = 2**self.N
		self.z = z
		self.nplot = nplot
		self.dzstep = self.z/self.nplot
		self.dz = self.dzstep/dz_less
		self.wavelength_space = wavelength_space
		return None




class sim_window(object):

    def __init__(self, fv, lamda, lamda_c, int_fwm,fv_idler_int):
        self.lamda = lamda
        self.lmin = 1e-3*c/np.max(fv)  # [nm]
        self.lmax = 1e-3*c/np.min(fv)  # [nm]
        self.lv = 1e-3*c/fv  # [nm]
        self.fmed = c/(lamda)  # [Hz]
        self.deltaf = 1e-3*(c/self.lmin - c/self.lmax)  # [THz]
        self.df = self.deltaf/int_fwm.nt  # [THz]
        self.T = 1/self.df  # Time window (period)[ps]
        self.woffset = 2*pi*(self.fmed - c/lamda)*1e-12  # [rad/ps]
        # [rad/ps] Offset of central freequency and that of the experiment
        self.woffset2 = 2*pi*(self.fmed - c/lamda_c)*1e-12
        # wavelength limits (for plots) (nm)
        self.xl = np.array([self.lmin, self.lmax])
        self.fv = fv
        # central angular frequency [rad/s]
        self.w0 = 2*pi*self.fmed
        # shock time [ps]
        self.tsh = 1/self.w0*1e12
        self.dt = self.T/int_fwm.nt  # timestep (dt)	 [ps]
        # time vector	   [ps]
        self.t = (range(int_fwm.nt)-np.ones(int_fwm.nt)*int_fwm.nt/2)*self.dt
        # angular frequency vector [rad/ps]
        self.w = 2*pi * \
        	np.append(
        		range(0, int(int_fwm.nt/2)), range(int(-int_fwm.nt/2), 0, 1))/self.T
        # frequency vector[THz] (shifted for plotting)
        self.vs = fftshift(self.w/(2*pi))
        # wavelength vector [nm]
        self.lv = c/(self.fmed+self.vs*1e12)*1e9
        # space vector [m]
        self.zv = int_fwm.dzstep*np.asarray(range(0, int_fwm.nplot+1))
        self.xtlim = np.array([-self.T/2, self.T/2])  # time limits (for plots)
        self.fv_idler_int = fv_idler_int
        self.fv_idler_tuple = (self.fmed*1e-12 - fv_idler_int, self.fmed*1e-12 + fv_idler_int)


def idler_limits(sim_wind, U):
    
    size = len(U[:,0,0])
    max_int = np.argsort(U[(size//2 + 1):,0,0])[0]
    max_int += size//2
    print(sim_wind.fv[max_int])
    
    sim_wind.fv_idler_tuple = (sim_wind.fv[max_int] - sim_wind.fv_idler_int, sim_wind.fv[max_int] + sim_wind.fv_idler_int)
    fv_id = (np.where(np.abs(sim_wind.fv - sim_wind.fv_idler_tuple[0]) == \
     np.min(np.abs(sim_wind.fv - sim_wind.fv_idler_tuple[0])))[0][0],\
    np.where(np.abs(sim_wind.fv - sim_wind.fv_idler_tuple[1])
    == np.min(np.abs(sim_wind.fv - sim_wind.fv_idler_tuple[1])))[0][0])
    return fv_id

class Loss(object):

	def __init__(self, int_fwm, sim_wind, amax=None, apart_div=8):
		"""
		Initialise the calss Loss, takes in the general parameters and 
		the freequenbcy window. From that it determines where the loss will become
		freequency dependent. With the default value being an 8th of the difference
		of max and min. 

		"""
		self.alpha = int_fwm.alphadB/4.343
		if amax == None:
			self.amax = self.alpha
		else:
			self.amax = amax/4.343

		self.flims_large = (np.min(sim_wind.fv), np.max(sim_wind.fv))
		try:
			temp = len(apart_div)
			self.begin = apart_div[0]
			self.end = apart_div[1]
		except TypeError:

			self.apart = np.abs(self.flims_large[1] - self.flims_large[0])
			self.apart /= apart_div
			self.begin = self.flims_large[0] + self.apart
			self.end = self.flims_large[1] - self.apart

	def atten_func_full(self, fv):
		aten = []

		a_s = ((self.amax - self.alpha) / (self.flims_large[0] - self.begin),

			   (self.amax - self.alpha) / (self.flims_large[1] - self.end))
		b_s = (-a_s[0] * self.begin, -a_s[1] * self.end)

		for f in fv:
			if f <= self.begin:
				aten.append(a_s[0] * f + b_s[0])
			elif f >= self.end:
				aten.append(a_s[1] * f + b_s[1])
			else:
				aten.append(0)
		return np.asanyarray(aten) + self.alpha

	def plot(self, fv):
		fig = plt.figure()
		y = self.atten_func_full(fv)
		plt.plot(fv, y)
		plt.xlabel("Frequency (Thz)")
		plt.ylabel("Attenuation (cm -1 )")
		plt.savefig(
			"output/output0/figures/WDMs&loss/loss_function_fibre.png", bbox_inches='tight')
		plt.close(fig)

class WDM(object):

	def __init__(self, x1, x2, fv,c, modes=1):
		"""
			This class represents a 2x2 WDM coupler. The minimum and maximums are
			given and then the object represents the class with WDM_pass the calculation
			done.
		"""
		self.l1 =  x1   # High part of port 1
		self.l2 =  x2  # Low wavelength of port 1
		self.f1 = 1e-3 * c / self.l1   # High part of port 1
		self.f2 = 1e-3 * c / self.l2  # Low wavelength of port 1
		self.omega = 0.5*pi/np.abs(self.f1 - self.f2)
		self.phi = pi - self.omega*self.f2
		#self.fv = fv
		self.fv_wdm = self.omega*fv+self.phi

		#self.A = np.array([[np.reshape(np.cos(self.fv), (len(self.fv), modes)),
		#						np.reshape(np.sin(self.fv), (len(self.fv), modes))],
		#					   [-np.reshape(np.sin(self.fv), (len(self.fv), modes)),
		#						np.reshape(np.cos(self.fv), (len(self.fv), modes))]])
		
		eps = np.reshape(np.sin(self.fv_wdm), (len(self.fv_wdm), modes))
		eps2 = 1j*np.reshape(np.cos(self.fv_wdm), (len(self.fv_wdm), modes))
		self.A = np.array([[eps ,eps2],
						   [eps2,eps]])
		return None

	def U_calc(self, U_in):
		"""
		Uses the array defined in __init__ to calculate 
		the outputed amplitude in arbitary units

		"""
		Uout = (self.A[0,0] * U_in[0] + self.A[0,1] * U_in[1],)
		Uout += (self.A[1,0] * U_in[0] + self.A[1,1] * U_in[1],)

		return Uout

	def pass_through(self, U_in, sim_wind, fft, ifft):
		"""
		Passes the amplitudes through the object. returns the u, U and Uabs
		in a form of a tuple of (port1,port2)
		"""
		U_out = self.U_calc(U_in)

		u_out, U_true = (), ()
		for i, UU in enumerate(U_out):
			u_out += (ifft(ifftshift(UU, axes=(0,))/sim_wind.dt),)
			#U_true += (fftshift(np.abs(sim_wind.dt *
			#						   fft(u_out[i]))**2, axes = (0,)),)
		return ((u_out[0], U_out[0]), (u_out[1], U_out[1]))

		
	def il_port1(self, lamda = None):
		freq = 1e-3 * c / lamda
		return (np.sin(self.omega*freq+self.phi))**2

			

	def il_port2(self, lamda):
		freq = 1e-3 * c / lamda
		return (np.cos(self.omega*freq+self.phi))**2


	def plot(self, lamda,filename= False,xlim = (900, 1250)):
		fig = plt.figure()
		plt.plot(lamda, self.il_port1(lamda), label="%0.2f" %
				 (self.l1) + ' nm port')
		plt.plot(lamda, self.il_port2(lamda), label="%0.2f" %
				 (self.l2) + ' nm port')
		plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=2)
		plt.xlabel(r'$\lambda (\mu m)$')
		plt.xlim()
		plt.ylabel('Power Ratio')
		plt.xlim(xlim)
		if filename:
			os.system('mkdir output/WDMs_loss')
			plt.savefig(
				'output/WDMs_loss/WDM_high_'+str(self.l1)+'_low_'+str(self.l2)+'.png')
		else: 
			plt.show()
		plt.close(fig)
		return None

	def plot_dB(self, lamda,filename = False):
		fig = plt.figure()
		plt.plot(lamda, 10*np.log10(self.il_port1(lamda)),
				 label="%0.2f" % (self.l1*1e9) + ' nm port')
		plt.plot(lamda, 10*np.log10(self.il_port2(lamda)),
				 label="%0.2f" % (self.l2*1e9) + ' nm port')
		plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=2)
		plt.xlabel(r'$\lambda (\mu m)$')
		plt.ylabel(r'$Insertion loss (dB)$')
		plt.ylim(-60, 0)
		#plt.xlim((900, 1250))
		if filename:
		
			plt.savefig('output/WDMs&loss/WDM_dB_high_' +
					str(self.l1)+'_low_'+str(self.l2)+'.png')
		else:
			plt.show()
		plt.close(fig)
		return None

def create_file_structure():
	"""
	Is set to create and destroy the filestructure needed 
	to run the program so that the files are not needed in the repo
	"""
	folders_large = ('output_dump_pump_powers', 'output_dump_pump_wavelengths', 'output_final', 'output')
	folders_large += (folders_large[-1] + '/output',)
	folders_large += (folders_large[-1] + '/data',)
	folders_large += (folders_large[-2] + '/figures',)
	
	outs = folders_large[-1]
	folders_figures = ('/freequency', '/time', '/wavelength')
	for i in folders_figures:
		folders_figures +=(i+'/portA',i+'/portB')
	for i in folders_figures:
		folders_large += (outs + i,)
	folders_large += (outs+'/WDMs',)
	for i in folders_large:
		if not os.path.isdir(i):
			os.system('mkdir ' + i)
	return None


class Splicer(WDM):

	def __init__(self, loss=1):
		self.loss = loss

	def U_calc(self, U_in):
		"""
		Operates like a beam splitter that reduces the optical power by the loss given (in dB).

		"""

		c1 = 10**(-0.1*self.loss/2)
		c2 = (1 - 10**(-0.1*self.loss))**0.5

		U_out1 = U_in[0] * c1 + 1j* U_in[1] * c2
		U_out2 = 1j* U_in[0] * c2 + U_in[1] * c1 
		return U_out1, U_out2


class Noise(object):

	def __init__(self, sim_wind):
		self.pquant = np.sum(
			1.054e-34*(sim_wind.w*1e12 + sim_wind.w0)/(sim_wind.T*1e-12))
		self.pquant = (self.pquant/2)**0.5
		return None

	def noise_func(self, int_fwm):
		noise = self.pquant * (np.random.randn(int_fwm.nt, int_fwm.nm)
							   + 1j*np.random.randn(int_fwm.nt, int_fwm.nm))

		#noise = self.pquant *np.ones([int_fwm.nt,int_fwm.nm])
		return noise

	def noise_func_freq(self, int_fwm, sim_wind, fft):
		noise = self.noise_func(int_fwm)
		noise_freq = fftshift(sim_wind.dt * fft(noise), axes=(0,))
		return noise_freq


def pulse_propagation(u, U, int_fwm, M1, M2, sim_wind, hf, Dop, dAdzmm, fft, ifft):
	"--------------------------Pulse propagation--------------------------------"
	badz = 0  # counter for bad steps
	#goodz = 0  # counter for good steps
	dztot = 0  # total distance traveled
	dzv = np.zeros(1)
	dzv[0] = int_fwm.dz
	u1 = np.copy(u[:, :, 0])

	#energy = np.NaN*np.ones([int_fwm.nm, int_fwm.nplot+1])
	#entot = np.NaN*np.ones(int_fwm.nplot+1)
	#for ii in range(int_fwm.nm):
	#	energy[ii, 0] = np.linalg.norm(u1[:, ii], 2)**2  # energy per mode
	dz = int_fwm.dz * 1
	# total energy (must be conserved)
	for jj in range(int_fwm.nplot):
		exitt = False
		while not(exitt):
			# trick to do the first iteration
			delta = 2*int_fwm.maxerr
			while delta > int_fwm.maxerr:
				u1new = ifft(np.exp(Dop*dz/2)*fft(u1))
				A, delta = RK5mm(dAdzmm, u1new, dz, M1, M2, sim_wind.t, int_fwm.n2, sim_wind.lamda, sim_wind.tsh,
								 sim_wind.w, sim_wind.woffset, sim_wind.dt, hf,sim_wind.w_tiled, fft, ifft)  # calls a 5th order Runge Kutta routine
				if (delta > int_fwm.maxerr):
					# calculate the step (shorter) to redo
					dz *= (int_fwm.maxerr/delta)**0.25
					badz += 1
			#####################################Successful step###############
			# propagate the remaining half step
			u1 = ifft(np.exp(Dop*dz/2)*fft(A))
			# update the propagated distance
			dztot += dz
			# update the number of steps taken
			#goodz += 1
			# store the dz just taken
			dzv = np.append(dzv, dz)
			# calculate the next step (longer)
			# # without exceeding max dzstep
			dz2 = np.min(
				[0.95*dz*(int_fwm.maxerr/delta)**0.2, 0.95*int_fwm.dzstep])
			###################################################################

			if dztot == (int_fwm.dzstep*(jj+1)):
				exitt = True

			elif ((dztot + dz2) >= int_fwm.dzstep*(jj+1)):
				dz2 = int_fwm.dzstep*(jj+1) - dztot
			dz = np.copy(dz2)
			###################################################################

		u[:, :, jj+1] = u1
		U[:, :, jj+1] = fftshift(sim_wind.dt*fft(u[:, :, jj+1]), axes=(0,))
		#Uabs[
		#	:, :, jj+1] = fftshift(np.abs(sim_wind.dt*fft(u[:, :, jj+1]))**2, axes=(0,))
		#for ii in range(int_fwm.nm):
		#	energy[ii, jj+1] = norm(u1[:, ii], 2)**2  # energy per mode
		#entot[jj+1] = np.sum(energy[:, jj+1])			 # total energy
	int_fwm.dz  = dz*1
	print(badz)
	return u, U


def dbm_nm(U, sim_wind, int_fwm):
	"""
	Converts The units of freequency to units of dBm/nm
	"""
	U_out = U / sim_wind.T**2
	U_out = -1*w2dbm(U_out)
	dlv = [sim_wind.lv[i+1] - sim_wind.lv[i]
		   for i in range(len(sim_wind.lv) - 1)]
	dlv = np.asanyarray(dlv)
	for i in range(int_fwm.nm):
		U_out[:, i] /= dlv[i]
	return U_out


def fv_creator(lam_start, lam_p1, int_fwm):
	"""
	Creates the freequency grid of the simmualtion and returns it. The middle 
	freequency gris is the pump freequency and where on the grid the pump lies.
	"""
	#lam_start = 800
	f_p1 = 1e-3*c/lam_p1
	f_start = 1e-3*c/lam_start

	fv1 = np.linspace(f_p1, f_start, 2**(int_fwm.N - 1))
	fv = np.ndarray.tolist(fv1)
	diff = fv[1] - fv[0]

	for i in range(2**(int_fwm.N - 1)):
		fv.insert(0,fv[0]-diff)
	fv = np.asanyarray(fv)
	check_ft_grid(fv, diff)

	where = [2**(int_fwm.N-1)]
	return fv, where


def energy_conservation(entot):
	if not(np.allclose(entot, entot[0])):
		fig = plt.figure()
		plt.plot(entot)
		plt.grid()
		plt.xlabel("nplots(snapshots)", fontsize=18)
		plt.ylabel("Total energy", fontsize=18)
		# plt.show()
		plt.close()
		sys.exit("energy is not conserved")
	return 0


def check_ft_grid(fv, diff):
	"""Grid check for fft optimisation"""
	if fv.any() < 0:
		sys.exit("some of your grid is negative")

	if np.log2(np.shape(fv)[0]) == int(np.log2(np.shape(fv)[0])):
		print(
			"------------------------------------------------------------------------------------")
		print("All is good with the grid for fft's:", np.shape(fv)[0])
		nt = np.shape(fv)[0]
	else:
		print(" ")
		sys.exit(
			"fix the grid for optimization of the fft's, grid:", np.shape(fv)[0])

	lvio = []
	for i in range(len(fv)-1):
		lvio.append(fv[i+1] - fv[i])

	grid_error = np.abs(np.asanyarray(lvio)[:]) - np.abs(diff)
	if not(np.allclose(grid_error, 0, rtol=0, atol=1e-12)):
		print(np.max(grid_error))
		sys.exit("your grid is not uniform")
	return 0

class create_destroy(object):
	"""
	creates and destroys temp folder that is used for computation. Both methods needs to be run
	before you initiate a new variable
	""" 
	def __init__(self, variable):
		self.variable = variable
		return None


	def cleanup_folder(self):
		for i in range(len(self.variable)):
			os.system('rm -r output/output'+str(i))
		return None


	def prepare_folder(self):
		for i in range(len(self.variable)):
			os.system('cp -r output/output/ output/output'+str(i))
		return None


def power_idler(spec,fv,T,fv_id):
    """
    Set to calculate the power of the idler. The possitions
    at what you call an idler are given in fv_id
    spec: the spectrum in freequency domain
    fv: the freequency vector
    T: time window
    fv_id: tuple of the starting and
    ending index at which the idler is calculated

    """
    P_bef = simps(np.abs(spec[fv_id[0]:fv_id[1],0,0])**2,fv[fv_id[0]:fv_id[1]])
    P_bef /= 2*T
    return P_bef 