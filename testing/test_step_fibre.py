import sys
sys.path.append('src')

'-------------------------Testing-step-index-fibre--------------------'
from step_index import *
from functions import birfeg_variation
import numpy as np
from numpy.testing import assert_allclose
def eigenvalues_test_case(l_vec, a_vec,err, margin):

    fibre = Fibre()
    per = ['ge', 'sio2']
    #err = 0.002
    ncore, nclad = fibre.indexes(l_vec, a_vec, per, err)
    # fibre.plot_fibre_n(l_vec,a_vec,per,err)
    E = Eigenvalues(l_vec, a_vec, ncore, nclad)

    u_vec = np.zeros([len(a_vec), len(l_vec)])
    w_vec = np.zeros(u_vec.shape)
    for i, a in enumerate(a_vec):
        print('New a = ', a)
        for j, l in enumerate(l_vec):
            u_vec[i, j], w_vec[i, j] = E.eigen_solver(margin, i, j)
    return u_vec, w_vec, E.V, ncore, nclad


class Test_eigenvalues:

    def test_V(self):
        margin = 5e-15
        a_med = 10e-6
        a_err_p = 0.01
        l_span = 300e-9
        l_p = 1550e-9
        low_a = a_med - a_err_p * a_med
        high_a = a_med + a_err_p * a_med

        l_vec = np.linspace(l_p - l_span, l_p + l_span, 20)
        a_vec = np.linspace(low_a, high_a, 5)
        err = np.linspace(-0.001, 0.001,5)
        u_vec, w_vec, V_vec, ncore, nclad = eigenvalues_test_case(
            l_vec, a_vec,err, margin)

        assert_allclose((u_vec**2 + w_vec**2)**0.5, V_vec)


class Test_betas:

    margin = 5e-15
    a_med = 7e-6
    a_err_p = 0.01
    l_span = 300e-9
    l_p = 1550e-9
    low_a = a_med - a_err_p * a_med
    high_a = a_med + a_err_p * a_med

    l_vec = np.linspace(l_p - l_span, l_p + l_span, 2**4)
    a_vec = np.linspace(low_a, high_a, 5)
    err = np.linspace(-0.001, 0.001,5)
    o_vec = 1e-12*2*pi * c / l_vec
    o = (o_vec[0]+o_vec[-1])/2
    u_vec, w_vec, V_vec, ncore, nclad =\
        eigenvalues_test_case(l_vec, a_vec,err, margin)

    betas = np.zeros([len(a_vec), len(o_vec)])
    beta_interpo = np.zeros(betas.shape)
    b = Betas(u_vec, w_vec, l_vec, o_vec, o, ncore, nclad)

    def test_neffs(self):
        for i, a in enumerate(self.a_vec):
            self.betas[i, :] = self.b.beta_func(self.o_vec,i)
        neffs = self.betas/(1e12*self.o_vec/c)
        assert (neffs < self.b.core).all() and (neffs > self.b.clad).all()

    def test_poly(self):
        for i, a in enumerate(self.a_vec):
            self.betas[i, :] = self.b.beta_func(self.o_vec,i)
            beta_coef = self.b.beta_extrapo(self.o_vec,i)
            p = np.poly1d(beta_coef)
            self.beta_interpo[i, :] = p(self.b.o_norm)
        assert_allclose(self.betas, self.beta_interpo, rtol=1e-07)

    def test_taylor(self):
        taylor_dispersion = np.zeros([len(self.a_vec), len(self.b.o_vec)])
        for i, a in enumerate(self.a_vec):
            self.betas[i, :] = self.b.beta_func(self.o_vec,i)
            beta_coef = self.b.beta_extrapo(self.o_vec,i)
            p = np.poly1d(1e-12*self.b.o_norm)
            betass = self.b.beta_dispersions(self.o_vec,i)
            for j, bb in enumerate(betass):
                taylor_dispersion[
                    i, :] += (bb/factorial(j))*((self.b.o_vec - self.b.o))**j
        assert_allclose(self.betas, taylor_dispersion, rtol=1e-7)

class Test_birefring():
    """
    Tests a conservation of energy for the birefringence angle change 
    that occurs at the wobbly sections. 
    """
    Num_a = 10
    Dtheta = birfeg_variation(Num_a, 2)

    def test_energy_1m(self):
        u1 = 100*(np.random.randn(2,1, 1024) +
              1j * np.random.randn(2,1, 1024))
        u2 = self.Dtheta.bire_pass(u1,0)
        assert_allclose(np.abs(u1)**2, np.abs(u2)**2)

    def test_energy_2m(self):
        u1 = 100*(np.random.randn(2,2, 1024) +
              1j * np.random.randn(2,2, 1024))
        u2 = self.Dtheta.bire_pass(u1,0)
        assert_allclose(np.abs(u1[:,0,:])**2 + np.abs(u1[:,1,:])**2,
                        np.abs(u2[:,0,:])**2 + np.abs(u2[:,1,:])**2)
