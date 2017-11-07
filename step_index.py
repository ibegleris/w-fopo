import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brenth, root
from scipy.constants import c, pi
from scipy.special import jv, kv
from scipy.interpolate import interp1d
from scipy.integrate import simps
import sys
import warnings
from pprint import pprint
from math import factorial
from itertools import combinations_with_replacement, permutations
import h5py
import os
from scipy.interpolate import UnivariateSpline


def jv_(n, z):
    return 0.5 * (jv(n-1, z) - jv(n+1, z))


def kv_(n, z):
    return -0.5 * (kv(n-1, z) + kv(n+1, z))


def save_variables_step(filename, variables, filepath=''):

    file = os.path.join(filepath, filename+'.hdf5')
    os.system('rm '+file)
    with h5py.File(file, 'a') as f:
        for i in (variables):
            f.create_dataset(str(i), data=variables[i])
    return None


class Fibre(object):
    """
    Fibre class. Set to initiate the functions needed
    for a fibre ( Seilmier equations etc).
    """

    def indexes(self, l):
        self.core, self.clad = self.selm_core(l), self.selm_clad(l)

    def selm_core(self, l):
        return 2.145 * np.ones_like(l)

    def selm_clad(self, l):
        return 2.144 * np.ones_like(l)

    def V_func(self, l_vec, a_vec):
        V_vec = np.empty([len(a_vec), len(l_vec)])
        temp = (2 * pi / l_vec)*(self.core**2 - self.clad**2)**0.5
        for i, a in enumerate(a_vec):
            V_vec[i, :] = temp * a
        if (V_vec > 2.405).any():
            print(V_vec[V_vec > 2.405])
            sys.exit('nm > 1!!!')
        else:
            self.V = V_vec

        return None


class Eigenvalues(Fibre):
    """
    Sets up to solve and solves the eigenvalue equation
    to find the eigenvalues of HE11. Only works on a single mode 
    fibre. Inherits V number function from Fibre class.
    """

    def __init__(self, l, r):
        self.indexes(l)
        self.V_func(l, r)
        self.ratio = self.clad/self.core
        return None

    def w_f(self, u, i, j):
        """
        Relasionship between the eigenvalues and V. 
        """
        return (self.V[i, j]**2 - u**2)**0.5

    def eq(self, u_vec, i, j, n=1):
        """
        The eigenvalue equation of a single mdoe fibre,
        set by default to find the HE11 mode. 
        """
        u = u_vec
        w = self.w_f(u, i, j)

        a =  (jv_(n, u)/(u*jv(n, w)) + kv_(n, w)/(w*kv(n, w))) * \
            (jv_(n, u)/(u*jv(n, u)) + kv_(n, w)/(w*kv(n, w))*self.ratio[j]**2) \
            - n**2 * (1/u**2 + 1/w**2) * (1/u**2 + self.ratio[j]**2 / w**2)

        return a

    def eigen_solver(self, margin, i, j):
        """
        Finds the eigenvalues of the fibre using breth. 
        Inputs:
            margin: A safety margin to save from the Zero division errors 
                    that arrise from the Eigenvalue equation if u,w = 0
        Returns:
        u, w: eigenvalues of the equation, system exists if non are 
              found and the equation is plotted. 
        """
        converged = False
        while not(converged) and (margin < self.V[i, j] - margin):
            try:
                Rr = brenth(self.eq, margin, self.V[
                            i, j] - margin, args=(i, j), full_output=True)
                converged = Rr[1].converged
            except ValueError:
                pass
                margin *= 10

        if converged:
            return Rr[0], self.w_f(Rr[0], i, j)
        else:
            print('----------------------------No solutions found--------------------')
            print(' V = ', self.V)
            print('------------------------------------------------------------------')
            #u = np.linspace(1e-6, self.V[0,-1] - 1e-6, 2048)
            # print(self.V)
            #e = self.eq(u,0,-1)
            #plt.plot(np.abs(u), e)
            #plt.plot(np.abs(u), np.abs(e))
            #plt.xlim(u.min(), u.max())
            #plt.ylim(-10, 10)
            # plt.show()
            sys.exit(1)


class Betas(Fibre):
    """Calculates the betas of the fibre mode. """

    def __init__(self, u_vec, w_vec, l_vec, o_vec, o):
        self.k = 2*pi/l_vec
        self.u = u_vec
        self.w = w_vec
        self.indexes(l_vec)
        self.o_vec = o_vec
        self.o = o
        self.o_norm = self.o_vec - self.o
        return None

    def beta_func(self, i):
        return (self.k**2*((self.core/self.u[i, :])**2 +
                           (self.clad/self.w[i, :])**2)/(1/self.u[i, :]**2
                                                         + 1/self.w[i, :]**2))**0.5

    def beta_extrapo(self, i):
        """
        Gets the polyonomial coefficiencts of beta(omega) with the 
        highest order possible. 
        """
        betas = self.beta_func(i)
        deg = 30
        fitted = False
        # warnings.warn(Warning())
        with warnings.catch_warnings():
            while not(fitted):
                warnings.filterwarnings('error')
                try:
                    coef = np.polyfit(self.o_norm, betas, deg=deg)
                    fitted = True
                except Warning:
                    deg -= 1
        return coef

    def beta_dispersions(self, i):
        coefs = self.beta_extrapo(i)
        betas = np.empty_like(coefs)
        for i, c in enumerate(coefs[::-1]):
            betas[i] = c * factorial(i)
        return betas

from numba import jit


class Modes(Fibre):
    """docstring for Modes"""

    def __init__(self, o_vec, o_c, beta_c, u_vec, w_vec, a_vec, x, y):
        self.n = 1
        o_norm = o_vec - o_c
        self.coordintes(x, y)
        self.beta_c = beta_c
        self.core = self.selm_core(2*pi*c/o_c)
        self.neff = self.beta_c / (o_c / c)
        self.u_vec, self.w_vec = np.zeros(u_vec.shape[0]),\
            np.zeros(u_vec.shape[0])
        for i in range(u_vec.shape[0]):
            self.u_vec[i] = interp1d(o_norm, u_vec[i, :], kind='cubic')(0)
            self.w_vec[i] = interp1d(o_norm, w_vec[i, :], kind='cubic')(0)
        self.a_vec = a_vec
        return None

    def coordintes(self, x, y):
        self.x, self.y = x, y
        self.X, self.Y = np.meshgrid(x, y)
        self.R = (self.X**2 + self.Y**2)**0.5
        self.T = np.arctan(self.Y/self.X)
        return None

    def pick_eigens(self, i):
        self.u = self.u_vec[i]
        self.w = self.w_vec[i]
        self.beta = self.beta_c[i]
        self.a = self.a_vec[i]
        self.s = self.n * (1/self.u**2 + 1/self.w**2) /\
            (jv_(self.n, self.u)/(self.n*jv(self.n, self.u))
             + kv_(self.n, self.w)/(self.w*kv(self.n, self.w)))
        return None

    @jit
    def E_r(self, r, theta):
        r0_ind = np.where(r <= self.a)
        r1_ind = np.where(r > self.a)
        temp = np.zeros(r.shape, dtype=np.complex128)
        r0, r1 = r[r0_ind], r[r1_ind]
        temp[r0_ind] = -1j * self.beta*self.a / \
            self.u*(0.5*(1 - self.s) * jv(self.n - 1, self.u * r0 / self.a)
                    - 0.5*(1 + self.s)*jv(self.n + 1, self.u * r0 / self.a))

        temp[r1_ind] = -1j * self.beta*self.a*jv(self.n, self.u)/(self.w*kv(self.n, self.w)) \
            * (0.5*(1 - self.s) * kv(self.n - 1, self.w * r1 / self.a)
               + 0.5*(1 + self.s)*kv(self.n+1, self.w * r1 / self.a))

        return temp*np.cos(self.n*theta), temp*np.cos(self.n*theta+pi/2)

    @jit
    def E_theta(self, r, theta):
        r0_ind = np.where(r <= self.a)
        r1_ind = np.where(r > self.a)
        temp = np.zeros(r.shape, dtype=np.complex128)
        r0, r1 = r[r0_ind], r[r1_ind]
        temp[r0_ind] = 1j * self.beta*self.a / \
            self.u*(0.5*(1 - self.s) * jv(self.n - 1, self.u * r0 / self.a)
                    + 0.5*(1 + self.s)*jv(self.n+1, self.u * r0 / self.a))

        temp[r1_ind] = 1j * self.beta*self.a * \
            jv(self.n, self.u)/(self.w*kv(self.n, self.w)) \
            * (0.5*(1 - self.s) * kv(self.n - 1, self.w * r1 / self.a)
               - 0.5*(1 + self.s)*kv(self.n+1, self.w * r1 / self.a))
        return temp*np.sin(self.n*theta), temp*np.sin(self.n*theta+pi/2)

    @jit
    def E_zeta(self, r, theta):
        r0_ind = np.where(r <= self.a)
        r1_ind = np.where(r > self.a)
        temp = np.zeros(r.shape, dtype=np.complex128)
        r0, r1 = r[r0_ind], r[r1_ind]
        temp[r0_ind] = jv(self.n, self.u*r0/self.a)
        temp[r1_ind] = jv(self.n, self.u) * \
            kv(self.n, self.w*r1/self.a)/kv(self.n, self.w)
        return temp*np.cos(self.n*theta), temp*np.cos(self.n*theta+pi/2)

    def E_carte(self):
        Er = self.E_r(self.R, self.T)
        Et = self.E_theta(self.R, self.T)
        Ex, Ey, Ez = [], [], []
        for i in range(len(Er)):
            Ex.append(Er[i] * np.cos(self.T) - Et[i] * np.sin(self.T))
            Ey.append(Er[i] * np.sin(self.T) + Et[i] * np.cos(self.T))
        Ez = self.E_zeta(self.R, self.T)

        return (Ex[0], Ey[0], Ez[0]), (Ex[1], Ey[1], Ez[1])
    #@profile

    def Q_matrixes(self):
        self.combination_matrix_assembler()
        Q_large = []
        for i in range(len(self.a_vec)):
            self.pick_eigens(i)
            E = np.asanyarray(self.E_carte())
            N = self.Normalised_N(E, i)
            Q = np.zeros([2, len(self.M1_top)], dtype=np.complex)
            for j, com in enumerate(self.M1_top):
                d = self.product_4(com, E)
                Q[0, j] = self.core**2 / (N[com[0]]
                                          * N[com[1]]*N[com[2]]*N[com[3]])*self.integrate(d[0])
                Q[1, j] = self.core**2 / (N[com[0]]
                                          * N[com[1]]*N[com[2]]*N[com[3]])*self.integrate(d[1])
            if i is 0:
                list_to_keep = []
                for i in range(len(self.M1_top)):
                    if Q[0, i] > 1 or Q[1, i] > 1:
                        list_to_keep.append(i)
            Q = Q[:, list_to_keep]
            Q_large.append(Q)
        M1 = np.asanyarray(self.M1_top)
        M1 = M1[list_to_keep].T
        M2 = np.unique(M1[-2:, :].T, axis=0).T
        M1_end = []
        for i, m1 in enumerate(M1[-2:, :].T):
            for j, m2 in enumerate(M2.T):
                if (m1 == m2).all():
                    M1_end.append(j)
                    break
        M1 = np.vstack((M1, np.asanyarray(M1_end)))
        for qq in Q_large:
            assert len(qq[0, :]) == M1.shape[1]
            assert len(qq[1, :]) == M1.shape[1]
        return M1, M2, np.asanyarray(Q_large)

    def product_2(self, E):
        d1 = np.einsum('ijk,ijk->jk', np.conj(E[0, :, :, :]), E[0, :, :, :])
        d2 = np.einsum('ijk,ijk->jk', np.conj(E[1, :, :, :]), E[1, :, :, :])
        return np.abs(d1), np.abs(d2)

    def product_4(self, com, E):
        d1 = np.einsum('ijk,ijk->jk', np.conj(E[com[0], :, :, :]), E[com[1], :, :, :]) *\
            np.einsum('ijk,ijk->jk', E[com[2], :, :,
                                       :], np.conj(E[com[3], :, :, :]))
        d2 = np.einsum('ijk,ijk->jk', np.conj(E[com[0], :, :, :]), np.conj(E[com[3], :, :, :])) *\
            np.einsum('ijk,ijk->jk', E[com[2], :, :, :], E[com[1], :, :, :])
        return d1, d2

    def denom_integral(self):

        return None

    def Normalised_N(self, E, i):
        """
        Calculated the normalised N integrals for the Q's for two modes.
        """
        d = self.product_2(E)
        return (self.neff[i] * self.integrate(d[0]))**0.5,\
            (self.neff[i] * self.integrate(d[1]))**0.5

    def combination_matrix_assembler(self):
        tot = []
        for i in combinations_with_replacement(range(2), 4):
            l = permutations(i)
            for j in l:
                tot.append(j)
        a = list(set(tot))
        a.sort()
        self.M1_top = a
        return None

    def integrate(self, z):
        """
        Integrates twice using Simpsons rule from scipy
        to allow 2D  integration.
        """
        return simps(simps(z, self.y), self.x)


def fibre_creator(a_vec, l_vec, filename='step_index_2m', N_points=512):
    margin = 1e-8
    o_vec = 2*pi * c/l_vec
    o = (o_vec[0]+o_vec[-1])/2

    fibre = Fibre()
    u_vec = np.zeros([len(a_vec), len(l_vec)])
    w_vec = np.zeros(u_vec.shape)
    E = Eigenvalues(l_vec, a_vec)
    for j, l in enumerate(l_vec):
        fibre.indexes(l)
        for i, a in enumerate(a_vec):
            u_vec[i, j], w_vec[i, j] = E.eigen_solver(margin, i, j)

    taylor_dispersion = np.zeros([len(a_vec), len(o_vec)])
    betas_large = []
    betas_central = np.zeros_like(a_vec)

    b = Betas(u_vec, w_vec, l_vec, o_vec, o)
    beta2_large = []
    for i, a in enumerate(a_vec):
        betas = b.beta_func(i)
        beta_coef = b.beta_extrapo(i)
        beta2_large.append(UnivariateSpline(
            o_vec, betas).derivative(n=2)(o_vec))
        p = np.poly1d(beta_coef)
        betas_central[i] = p(0)
        betass = b.beta_dispersions(i)
        betas_large.append(betas)
        for j, bb in enumerate(betass):
            taylor_dispersion[i, :] += (bb/factorial(j))*(o_vec - o)**j
    #min_beta = np.min([len(i) for i in betas_large])
    #betas = np.zeros([len(a_vec), min_beta])
    # for i in range(len(betas_large)):
    #    betas[i, :] = betas_large[i][:min_beta]

    r_max = np.max(a_vec)
    x = np.linspace(-2*r_max, 2*r_max, N_points)
    y = np.linspace(-2*r_max, 2*r_max, N_points)

    M = Modes(o_vec, o, betas_central, u_vec, w_vec, a_vec, x, y)
    M1, M2, Q_large = M.Q_matrixes()
    Export_dict = {'M1': M1, 'M2': M2,
                   'Q_large': Q_large, 'betas': taylor_dispersion}

    save_variables_step(filename,  variables=Export_dict,
                        filepath='loading_data/')

    return betas_large, Q_large, M, beta2_large


def main(a_med,a_err_p,l_p,l_span):



    low_a = a_med - a_err_p * a_med
    high_a = a_med + a_err_p * a_med

    l_vec = np.linspace(l_p - l_span, l_p + l_span, 20)
    a_vec = np.linspace(low_a, high_a, 10)

    betas, Q_large, M, beta2 = fibre_creator(a_vec, l_vec, N_points=512)
    fig = plt.figure(figsize=(15, 7.5))
    for i, a in enumerate(a_vec):
        plt.plot(l_vec*1e9, 1e24*beta2[i][:], label=r'$\alpha = $'+'{0:.2f}'.format(a*1e6)+r'$\mu m$')
        plt.xlabel(r'$\lambda(nm)$')
        plt.ylabel(r'$\beta_{2} (ps^{2}/m)$')
    plt.legend()
    plt.show()
   
    fig = plt.figure(figsize=(15, 7.5))
    plt.ticklabel_format(useOffset=False)
    for i, a in enumerate(a_vec):
        plt.plot(l_vec*1e9, betas[i][:] / (2*pi/l_vec), label=r'$\alpha = $'+'{0:.2f}'.format(a*1e6)+r'$\mu m$')
        plt.xlabel(r'$\lambda(nm)$')
        plt.ylabel(r'$n_{eff}$')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(15, 7.5))
    plt.plot(a_vec*1e6, Q_large[:,0,0]*1e-12)
    plt.xlabel(r'$\a(\mu m)$')
    plt.ylabel(r'$Q (\mu m)$')
    plt.legend()
    plt.show()
    HE11x, HE11y = [], []
    for i in range(len(a_vec)):
        M.pick_eigens(i)
        res = M.E_carte()
        HE11x.append(res[0])
        HE11y.append(res[1])

    E = (np.abs(HE11x[0][0])**2 + np.abs(HE11x[0][1])
         ** 2 + np.abs(HE11x[0][2])**2)**0.5

    X, Y = np.meshgrid(M.x*2e6, M.y*2e6)
    #X *= 1e6
    #Y *= 1e6

    Enorm = E/np.max(E)
    sp = 100
    fig = plt.figure(figsize=(7.5, 7.5))
    plt.contourf(X, Y, Enorm, 10, cmap=plt.cm.jet)
    plt.quiver(X[::sp, ::sp], Y[::sp, ::sp], np.abs(
        HE11x[0][0][::sp, ::sp]), np.abs(HE11x[0][1][::sp, ::sp]), headlength=80)
    plt.quiver(X[::sp, ::sp], Y[::sp, ::sp], np.abs(
        HE11y[0][0][::sp, ::sp]), np.abs(HE11y[0][1][::sp, ::sp]), headlength=80)
    plt.xlabel(r'$x(\mu m)$')
    plt.ylabel(r'$y(\mu m)$')
    plt.show()

    return None

if __name__ == '__main__':
    a_med = 6e-6
    a_err_p = 0.01
    l_span = 300e-9
    l_p = 1550e-9
    main(a_med,a_err_p,l_p,l_span)
