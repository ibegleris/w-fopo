from __future__ import division
from math import factorial
from functions import *
import pytest
from scipy.fftpack import fft,ifft,fftshift
scfft,iscfft = fft,ifft
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.interpolate import InterpolatedUnivariateSpline
"---------------------------------W and dbm conversion tests--------------"
def test_dbm2w():
	assert dbm2w(30) == 1


def test1_w2dbm():
	assert w2dbm(1) == 30


def test2_w2dbm():
	a = np.zeros(100)
	floor = np.random.rand(1)[0]
	assert_array_almost_equal(w2dbm(a,-floor), -floor*np.ones(len(a)))


def test3_w2dbm():
	with pytest.raises(ZeroDivisionError):
		w2dbm(-1)

"------------------------------------------------------fft test--------------"
try: 
	from accelerate.fftpack import fft, ifft
	def test_fft():
		x = np.random.rand(11,10)
		assert_array_almost_equal(fft(x.T).T, scfft(x))


	def test_ifft():
		x = np.random.rand(10,10)
		assert_array_almost_equal(ifft(x.T).T, iscfft(x))
except:
	from scipy.fftpack import fft, ifft
	pass

"--------------------------------------------Raman response--------------"
def test_raman_off():
	ram = raman_object('off')
	ram.raman_load(np.random.rand(10),np.random.rand(1)[0],fft,ifft)
	assert ram.hf == None


def test_raman_load():
	ram = raman_object('on','load')
	ram.raman_load(np.random.rand(10),np.random.rand(1)[0],fft,ifft)
	assert 0 == 0


def test_raman_analytic():
	ram = raman_object('on','analytic')
	ram.raman_load(np.random.rand(10),np.random.rand(1)[0],fft,ifft)
	assert 0 == 0



"----------------------------Dispersion operator--------------"
class int_fwms(object):
		def __init__(self,nm,alpha,nt):
			self.nm = nm
			self.alphadB = alpha
			self.alpha = self.alphadB/4.343
			self.nt = nt
class sim_windows(object):
	def __init__(self,lamda,lv,lmax,lmin):
		self.lamda = lamda
		self.lmax,self.lmin = lmax,lmin
		self.w = 2*pi*c/self.lamda*1e-12
		self.lv = lv
		self.nt = 512
		self.fv = c/self.lv
		self.fmed = c/(self.lamda)
		self.deltaf = 1e-3*(c/self.lmin - c/self.lmax)
		self.df = self.deltaf/len(lv)
		self.T = 1/self.df 
		self.dt = self.T/len(lv)         
def test_dispersion():
	nt = 512
	lmin,lmax = 1000e-9,2000e-9
	lamda = np.linspace(lmin,lmax,nt)

	lamda0 = 1500e-9
	lamdac = 1550e-9

	sim_wind = sim_windows(lamda0,lamda,lmin,lmax)
	int_fwm = int_fwms(1, 0.1,nt)

	betas = np.array([[0,0,0,6.755e-2,-1.001e-4]])*1e-3

	betas_disp = dispersion_operator(betas,lamdac,int_fwm,sim_wind)

	betas_exact = np.loadtxt('testing_data/exact_dispersion.py').view(complex)
	assert_array_almost_equal(betas_disp,betas_exact)




"-----------------------Full soliton--------------------------------------------"	
def test_pulse_propagation():
	"SOLITON TEST. IF THIS FAILS GOD HELP YOU!"
	


	n2 = 2.5e-20                				# n2 for silica [m/W]
	nm = 1                      				# number of modes
	alphadB = 0#0.0011666666666666668             # loss [dB/m]
	gama = 10e-3 								# w/m
	Power_input = 13                      		#[W]
	"-----------------------------General options------------------------------"

	maxerr = 1e-13            	# maximum tolerable error per step
	ss = 1                      # includes self steepening term
	ram = 'on'                  # Raman contribution 'on' if yes and 'off' if no
	
	"----------------------------Simulation parameters-------------------------"
	N = 10
	z = 10				 	# total distance [m]
	nplot = 100                  # number of plots
	nt = 2**N 					# number of grid points
	dzstep = z/nplot            # distance per step
	dz_less = 1e4
	dz = dzstep/dz_less         # starting guess value of the step
	

	lam_p1 = 500
	lamda_c = 500e-9
	lamda = lam_p1*1e-9
	
	N_sol = 1 
	TFWHM = 0.03
	
	beta2 = -11.83e-3
	gama = 1
	T0 = TFWHM/2/(np.log(2)); 
	P0_p1 =  np.abs(beta2) / (gama * T0)



	int_fwm = sim_parameters(n2,nm,alphadB)
	int_fwm.general_options(maxerr,ss,ram)
	int_fwm.propagation_parameters(N, z, nplot, dz_less, True)


	#print(lam_p1)
	fv,where = fv_creator(lam_p1 - 100,lam_p1,int_fwm)
	sim_wind = sim_window(fv,lamda,lamda_c,int_fwm)

	
	betas = np.array([[0,0,beta2,0,0]])*1e-3 # betas at ps/m (given in ps/km)
	Dop = dispersion_operator(betas,lamda_c,int_fwm,sim_wind)

	dAdzmm = dAdzmm_ron_s1
	M1,M2 = Q_matrixes(1,n2,lamda,gama=gama)
	int_fwm.raman.raman_load(sim_wind.t,sim_wind.dt,fft,ifft)
	hf = int_fwm.raman.hf
	

	u = np.zeros([len(sim_wind.t),int_fwm.nm,len(sim_wind.zv)],dtype='complex128')
	U = np.zeros([len(sim_wind.t),int_fwm.nm,len(sim_wind.zv)],dtype='complex128')
	Uabs = np.copy(U)
	



	u[:,0,0] = (P0_p1)**0.5/ np.cosh(sim_wind.t/T0)*np.exp(-1j*(sim_wind.woffset)*sim_wind.t);

	U[:,:,0] = fftshift(sim_wind.dt*fft(u[:,:,0]),axes=(0,))
	Uabs[:,:,0] = fftshift(np.abs(sim_wind.dt*fft(u[:,:,0]))**2,axes=(0,))
	u,U,Uabs  = pulse_propagation(u,U,Uabs,int_fwm,M1,M2,sim_wind,hf,Dop,dAdzmm,fft,ifft)

	#plotter_dbm(1, sim_wind, Uabs, u, 0)
	#plotter_dbm(1, sim_wind, Uabs, u, -1)

	assert_array_almost_equal(np.abs(U[:,0,0])**2,np.abs(U[:,0,-1])**2)




"-------------------------------WDM------------------------------------"
class Test_WDM(object):
	"""WDM test. first it makes sure that the port multiplyers are equal to one 
	over the grid and after looks in to a random pumped WDM and asks if
	U_port1 + U_port2 = U ( not much to ask a?)"""
	def test1_WDM(self):

		self.x1 = 1550
		self.x2 = 1555
		self.nt = 3
		self.lamda = np.linspace(1000, 2000,self.nt)
		WDMS = WDM(self.x1, self.x2)
		assert_array_almost_equal(WDMS.il_port1(self.lamda) +WDMS.il_port2(self.lamda),np.ones(self.nt))
	def test2_WDM(self):
		self.x1 = 1550
		self.x2 = 1555
		self.nt = 10
		self.lmax, self.lmin = 1450, 1600
		self.lamda = np.linspace(self.lmax, self.lmin,self.nt)

		sim_wind = sim_windows(self.lamda,self.lamda,self.lmax, self.lmin)

		WDMS = WDM(self.x1, self.x2)
		
		U =  np.random.randn(self.nt, 1)+1j* np.random.randn(self.nt, 1)

		port1 = WDMS.wdm_port1_pass(U,sim_wind,fft,ifft)
		port2 = WDMS.wdm_port2_pass(U,sim_wind,fft,ifft)
		assert_array_almost_equal(port1[1]+port2[1], U)