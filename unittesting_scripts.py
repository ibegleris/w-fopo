from __future__ import division
from math import factorial
from functions import *
import pytest
from scipy.fftpack import fft,ifft,fftshift
scfft,iscfft = fft,ifft
import numpy as np
from scipy.io import loadmat
from numpy.testing import assert_allclose,assert_approx_equal,assert_almost_equal,assert_raises
from scipy.interpolate import InterpolatedUnivariateSpline
from data_plotters_animators import *
import matplotlib.pyplot as plt
"---------------------------------W and dbm conversion tests--------------"
def test_dbm2w():
	assert dbm2w(30) == 1


def test1_w2dbm():
	assert w2dbm(1) == 30


def test2_w2dbm():
	a = np.zeros(100)
	floor = np.random.rand(1)[0]
	assert_allclose(w2dbm(a,-floor), -floor*np.ones(len(a)))


def test3_w2dbm():
	with pytest.raises(ZeroDivisionError):
		w2dbm(-1)

"------------------------------------------------------fft test--------------"
try: 
	from accelerate.fftpack import fft, ifft
	def test_fft():
		x = np.random.rand(11,10)
		assert_allclose(fft(x.T).T, scfft(x))


	def test_ifft():
		x = np.random.rand(10,10)
		assert_allclose(ifft(x.T).T, iscfft(x))
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
	D = loadmat('testing_data/Raman_measured.mat')
	t = D['t']
	t = np.asanyarray([t[i][0] for i in range(t.shape[0])])
	dt = D['dt'][0][0]
	hf_exact = D['hf']
	hf_exact = np.asanyarray([hf_exact[i][0] for i in range(hf_exact.shape[0])])
	hf = ram.raman_load(t,dt,fft,ifft)

	assert_allclose(hf, hf_exact)


def test_raman_analytic():
	ram = raman_object('on','analytic')
	D = loadmat('testing_data/Raman_analytic.mat')
	t = D['t']
	t = np.asanyarray([t[i][0] for i in range(t.shape[0])])
	dt = D['dt'][0][0]
	hf_exact = D['hf']
	hf_exact = np.asanyarray([hf_exact[i][0] for i in range(hf_exact.shape[0])])
	hf = ram.raman_load(t,dt,fft,ifft)

	assert_allclose(hf, hf_exact)


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

	loss = Loss(int_fwm, sim_wind, amax = 0.1)
	alpha_func = loss.atten_func_full(sim_wind.fv)
	int_fwm.alphadB = alpha_func
	int_fwm.alpha = int_fwm.alphadB

	betas = np.array([0,0,0,6.755e-2,-1.001e-4])*1e-3

	betas_disp = dispersion_operator(betas,lamdac,int_fwm,sim_wind)

	betas_exact = np.loadtxt('testing_data/exact_dispersion.py').view(complex)
	assert_allclose(betas_disp,betas_exact[:,0])


"-----------------------Full soliton--------------------------------------------"	
def pulse_propagations(ram,ss,N_sol = 1):
	"SOLITON TEST. IF THIS FAILS GOD HELP YOU!"
	


	n2 = 2.5e-20								# n2 for silica [m/W]
	nm = 1					  				# number of modes
	alphadB = 0#0.0011666666666666668			 # loss [dB/m]
	gama = 3e-3 								# w/m
	"-----------------------------General options------------------------------"
	maxerr = 1e-15				# maximum tolerable error per step
	"----------------------------Simulation parameters-------------------------"
	N = 13
	z = 18				 	# total distance [m]
	nplot = 1				  # number of plots
	nt = 2**N 					# number of grid points
	dzstep = z/nplot			# distance per step
	dz_less = 1e10
	dz = dzstep/dz_less		 # starting guess value of the step

	lam_p1 = 900
	lamda_c = 900e-9
	lamda = lam_p1*1e-9
	
	beta2 = 11.83e-5
	P0_p1 = 1

	T0 =  (N_sol**2 * np.abs(beta2) / (gama * P0_p1))**0.5
	TFWHM = (2*np.log(1+2**0.5)) * T0

	int_fwm = sim_parameters(n2,nm,alphadB)
	int_fwm.general_options(maxerr,raman_object,ss,ram)
	int_fwm.propagation_parameters(N, z, nplot, dz_less, True)

	fv,where = fv_creator(lam_p1 - 10,lam_p1,int_fwm)
	sim_wind = sim_window(fv,lamda,lamda_c,int_fwm,fv_idler_int = 1)
	
	loss = Loss(int_fwm, sim_wind, amax =	int_fwm.alphadB)
	alpha_func = loss.atten_func_full(sim_wind.fv)
	int_fwm.alphadB = alpha_func
	int_fwm.alpha = int_fwm.alphadB
	betas = np.array([0,0,beta2]) # betas at ps/m
	Dop = dispersion_operator(betas,lamda_c,int_fwm,sim_wind)

	string = "dAdzmm_r"+str(ram)+"_s"+str(ss)
	func_dict = {'dAdzmm_ron_s1': dAdzmm_ron_s1,
				'dAdzmm_ron_s0': dAdzmm_ron_s0,
				'dAdzmm_roff_s0': dAdzmm_roff_s0,
				'dAdzmm_roff_s1': dAdzmm_roff_s1}
	pulse_pos_dict_or = ('after propagation', "pass WDM2",
						"pass WDM1 on port2 (remove pump)",
						'add more pump', 'out')

	dAdzmm = func_dict[string]

	M = Q_matrixes(1,n2,lamda,gama=gama)
	raman = raman_object(int_fwm.ram, int_fwm.how)
	raman.raman_load(sim_wind.t, sim_wind.dt, fft, ifft)
	
	if raman.on == 'on':	
		hf = raman.hf
	else:
		hf = None


	u = np.zeros([len(sim_wind.t),len(sim_wind.zv)],dtype='complex128')
	U = np.zeros([len(sim_wind.t),len(sim_wind.zv)],dtype='complex128')

	sim_wind.w_tiled = sim_wind.w



	u[:,0] = (P0_p1)**0.5 / np.cosh(sim_wind.t/T0)*np.exp(-1j*(sim_wind.woffset)*sim_wind.t)
	U[:,0] = fftshift(sim_wind.dt*fft(u[:,0]))
	
	u,U  = pulse_propagation(u,U,int_fwm,M,sim_wind,hf,Dop,dAdzmm,fft,ifft)

	U_start = np.abs(U[:,0])**2
	

	fig1 = plt.figure()
	plt.plot(sim_wind.fv,U_start)
	plt.savefig('1.png')
	fig2 = plt.figure()
	plt.plot(sim_wind.fv,np.abs(U[:,-1])**2)
	plt.savefig('2.png')	
	fig3 = plt.figure()
	plt.plot(sim_wind.t,np.abs(u[:,0])**2)
	plt.savefig('3.png')
	fig4 = plt.figure()
	plt.plot(sim_wind.t,np.abs(u[:,-1])**2)
	plt.savefig('4.png')	
	fig5 = plt.figure()
	plt.plot(fftshift(sim_wind.w),(np.abs(U[:,-1])**2 - U_start))
	plt.savefig('error.png')
	fig6 = plt.figure()
	plt.plot(sim_wind.t,np.abs(u[:,-1])**2 - np.abs(u[:,0])**2)
	plt.savefig('error2.png')
	return u,U


def test_energy_r0_ss0():
	u,U = pulse_propagations('off', 0,N_sol=np.abs(np.random.randn()))
	E = []
	for i in range(np.shape(u)[1]):
		E.append(np.linalg.norm(u[:,i], 2)**2)
	assert np.all(x == E[0] for x in E)


def test_energy_r0_ss1():
	u,U = pulse_propagations('off', 1,N_sol=np.abs(np.random.randn()))
	E = []
	for i in range(np.shape(u)[1]):
		E.append(np.linalg.norm(u[:,i], 2)**2)
	assert np.all(x == E[0] for x in E)


def test_energy_r1_ss0():
	u,U = pulse_propagations('on', 0,N_sol=np.abs(np.random.randn()))
	E = []
	for i in range(np.shape(u)[1]):
		E.append(np.linalg.norm(u[:,i], 2)**2)
	assert np.all(x == E[0] for x in E)


def test_energy_r1_ss1():
	u,U = pulse_propagations('on', 1,N_sol=np.abs(np.random.randn()))
	E = []
	for i in range(np.shape(u)[1]):
		E.append(np.linalg.norm(u[:,i], 2)**2)
	assert np.all(x == E[0] for x in E)


def test_solit_r0_ss0():
	u,U = pulse_propagations('off', 0)
	assert_allclose(np.abs(u[:,0])**2 , np.abs(u[:,-1])**2)


"-------------------------------WDM------------------------------------"
class Test_WDM(object):
	"""
	Tests conservation of energy in freequency and time space as well as the 
	absolute square value I cary around in the code.
	"""
	def test1_WDM_freq(self):
		self.x1 = 930
		self.x2 = 1050
		self.nt = 3

		self.lv = np.linspace(900, 1250,2**self.nt)
		self.fv = 1e3 * c/ self.lv

		WDMS = WDM(self.x1, self.x2,self.fv, c)
		sim_wind = sim_windows(self.lv,self.lv,900, 1250)
		
		U1 = 10*(np.random.randn(2**self.nt,1) + 1j * np.random.randn(2**self.nt,1))
		U2 = 10 *(np.random.randn(2**self.nt,1) + 1j * np.random.randn(2**self.nt,1))
		

		
		U_in = (U1, U2)
		a,b = WDMS.pass_through(U_in,sim_wind,fft,ifft)
		U_out1,U_out2 = a[1], b[1]

		U_in_tot = np.abs(U1)**2 + np.abs(U2)**2
		U_out_tot = np.abs(U_out1)**2 + np.abs(U_out2)**2

		assert_allclose(U_in_tot,U_out_tot)

	def test2_WDM_time(self):
		self.x1 = 930
		self.x2 = 1050
		self.nt = 3

		self.lv = np.linspace(900, 1250,2**self.nt)
		self.fv = 1e3 * c/ self.lv

		WDMS = WDM(self.x1, self.x2,self.fv, c)
		sim_wind = sim_windows(self.lv,self.lv,900, 1250)
		
		U1 = 10*(np.random.randn(2**self.nt,1) + 1j * np.random.randn(2**self.nt,1))
		U2 = 10 *(np.random.randn(2**self.nt,1) + 1j * np.random.randn(2**self.nt,1))
		

		u_in1 = ifftshift(ifft(U1)/sim_wind.dt, axes=(0,))
		u_in2 = ifftshift(ifft(U2)/sim_wind.dt, axes=(0,))

		U_in = (U1, U2)
		u_in_tot = np.abs(u_in1)**2 + np.abs(u_in2)**2

		a,b = WDMS.pass_through(U_in,sim_wind,fft,ifft)
		u_out1,u_out2 = a[0], b[0]

		
		u_out_tot = np.abs(u_out1)**2 + np.abs(u_out2)**2

		assert_allclose( u_in_tot, u_out_tot)


		

class int_fwmss(object):
	def __init__(self, alphadB):
		self.alphadB = alphadB
class sim_windowss(object):
	def __init__(self, fv):
		self.fv  = fv
class Test_loss:
	def test_loss1(a):
		fv = np.linspace(200, 600,1024)
		alphadB = 1
		sim_wind = sim_windowss(fv)
		int_fwm =  int_fwmss(alphadB)
		loss = Loss(int_fwm, sim_wind, amax = alphadB)
		alpha_func = loss.atten_func_full(sim_wind.fv)
		assert_allclose(alpha_func, np.ones_like(alpha_func)*alphadB/4.343)
	def test_loss2(a):
		fv = np.linspace(200, 600,1024)
		alphadB = 1
		sim_wind = sim_windowss(fv)
		int_fwm =  int_fwmss(alphadB)
		loss = Loss(int_fwm, sim_wind, amax = 2*alphadB)
		alpha_func = loss.atten_func_full(sim_wind.fv)
		maxim = np.max(alpha_func)
		assert maxim == 2*alphadB/4.343

	def test_loss3(a):
		fv = np.linspace(200, 600,1024)
		alphadB = 1
		sim_wind = sim_windowss(fv)
		int_fwm =  int_fwmss(alphadB)
		loss = Loss(int_fwm, sim_wind, amax = 2*alphadB)
		alpha_func = loss.atten_func_full(sim_wind.fv)
		minim = np.min(alpha_func)
		assert minim == alphadB/4.343


class Test_splicer():

	def test1_splicer_freq(self):
		self.x1 = 930
		self.x2 = 1050
		self.nt = 3

		self.lv = np.linspace(900, 1250,2**self.nt)
		splicer = Splicer(loss = np.random.rand()*10)
		sim_wind = sim_windows(self.lv,self.lv,900, 1250)
		
		U1 = 10*(np.random.randn(2**self.nt,1) + 1j * np.random.randn(2**self.nt,1))
		U2 = 10 *(np.random.randn(2**self.nt,1) + 1j * np.random.randn(2**self.nt,1))
		
		
		
		U_in = (U1, U2)
		a,b = splicer.pass_through(U_in,sim_wind,fft,ifft)
		U_out1,U_out2 = a[1], b[1]

		U_in_tot = np.abs(U1)**2 + np.abs(U2)**2
		U_out_tot = np.abs(U_out1)**2 + np.abs(U_out2)**2

		assert_allclose(U_in_tot,U_out_tot)

	def test2_WDM_time(self):
		self.x1 = 930
		self.x2 = 1050
		self.nt = 3

		self.lv = np.linspace(900, 1250,2**self.nt)


		splicer = Splicer(loss = np.random.rand()*10)
		sim_wind = sim_windows(self.lv,self.lv,900, 1250)
		
		U1 = 10*(np.random.randn(2**self.nt,1) + 1j * np.random.randn(2**self.nt,1))
		U2 = 10 *(np.random.randn(2**self.nt,1) + 1j * np.random.randn(2**self.nt,1))
		

		u_in1 = ifftshift(ifft(U1)/sim_wind.dt, axes=(0,))
		u_in2 = ifftshift(ifft(U2)/sim_wind.dt, axes=(0,))

		U_in = (U1, U2)
		u_in_tot = np.abs(u_in1)**2 + np.abs(u_in2)**2

		a,b = splicer.pass_through(U_in,sim_wind,fft,ifft)
		u_out1,u_out2 = a[0], b[0]

		
		u_out_tot = np.abs(u_out1)**2 + np.abs(u_out2)**2

		assert_allclose( u_in_tot, u_out_tot)
	



def test_read_write1():
	os.system('rm testing_data/hh51_test.hdf5')
	A = np.random.rand(10,3,5) + 1j* np.random.rand(10,3,5)
	B  = np.random.rand(10)
	C = 1
	save_variables('hh51_test','0',filepath = 'testing_data/',
					A = A, B = B, C=C)
	A_copy, B_copy, C_copy = np.copy(A), np.copy(B), np.copy(C)
	del A,B,C
	D = read_variables('hh51_test', '0', filepath='testing_data/')

	A,B,C = D['A'], D['B'], D['C']
	#locals().update(D)
	assert_allclose(A,A_copy)


def test_read_write2():

	os.system('rm testing_data/hh52_test.hdf5')
	A = np.random.rand(10,3,5) + 1j* np.random.rand(10,3,5)
	B  = np.random.rand(10)
	C = 1
	save_variables('hh52_test','0',filepath = 'testing_data/',
					A = A, B = B, C=C)
	A_copy, B_copy, C_copy = np.copy(A), np.copy(B), np.copy(C)
	del A,B,C
	D = read_variables('hh52_test', '0', filepath='testing_data/')
	A,B,C = D['A'], D['B'], D['C']
	#locals().update(D)
	assert_allclose(B,B_copy)
	return None


def test_read_write3():

	os.system('rm testing_data/hh53_test.hdf5')
	A = np.random.rand(10,3,5) + 1j* np.random.rand(10,3,5)
	B  = np.random.rand(10)
	C = 1
	save_variables('hh53_test','0',filepath = 'testing_data/',
					A = A, B = B, C=C)
	A_copy, B_copy, C_copy = np.copy(A), np.copy(B), np.copy(C)
	del A,B,C
	D = read_variables('hh53_test', '0', filepath='testing_data/')
	A,B,C = D['A'], D['B'], D['C']
	#locals().update(D)
	assert C == C_copy
	return None


def test_fv_creator():
	class int_fwm1(object):
		def __init__(self):
			self.N =  10

	int_fwm = int_fwm1()
	lam_start = 1000
	lam_p1 = 1200
	fv, where = fv_creator(lam_start, lam_p1, int_fwm)
	mins = np.min(1e-3*c/fv)
	assert_almost_equal(lam_start,mins)


def test_noise():
	class sim_windows(object):
		def __init__(self):
			self.w = 10 
			self.T = 0.1
			self.w0 = 9
	class int_fwms(object):
		def __init__(self):
			self.nt = 1024
			self.nm = 1	
	int_fwm = int_fwms()
	sim_wind = sim_windows()	
	noise = Noise(sim_wind)
	n1 = noise.noise_func(int_fwm)
	n2 = noise.noise_func(int_fwm)
	assert_raises(AssertionError, assert_almost_equal, n1, n2)