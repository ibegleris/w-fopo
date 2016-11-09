from __future__ import division

from functions import *
import pytest
from scipy.fftpack import fft,ifft
scfft,iscfft = fft,ifft
import numpy as np
from numpy.testing import assert_array_almost_equal

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

def test_dispersion():
	betas = [0,0,-5e-3,0,0]

	assert 0==0
