import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brenth, root
from scipy.constants import c, pi
from scipy.special import jv, kv
import sys
import warnings
from pprint import pprint
from math import factorial


def jv_(n, z):
    return 0.5 * (jv(n-1, z) - jv(n+1, z))


def kv_(n, z):
    return -0.5 * (kv(n-1, z) + kv(n+1, z))


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
        return 2.1 * np.ones_like(l)

    def V_func(self, l, r):
        self.V = (2 * pi / l)*r*(self.core**2 - self.clad**2)**0.5
        if self.V > 2.405:
            print(self.V)
            sys.exit('nm > 1!!!')
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

    def w_f(self, u):
        """
        Relasionship between the eigenvalues and V. 
        """
        return (self.V**2 - u**2)**0.5

    def eq(self, u, n=1):
        """
        The eigenvalue equation of a single mdoe fibre,
        set by default to find the HE11 mode. 
        """
        w = self.w_f(u)
        return (jv_(n, u)/(u*jv(n, w)) + kv_(n, w)/(w*kv(n, w))) * \
            (jv_(n, u)/(u*jv(n, u)) + kv_(n, w)/(w*kv(n, w))*self.ratio**2) \
            - n**2 * (1/u**2 + 1/w**2) * (1/u**2 + self.ratio**2 / w**2)

    def eigen_solver(self, margin):
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
        while not(converged) and (margin < self.V - margin):
            try:
                Rr = brenth(self.eq, margin, self.V - margin, full_output=True)
                converged = Rr[1].converged
            except ValueError:
                pass
                margin *= 10

        if converged:
            return Rr[0], self.w_f(Rr[0])
        else:
            print('--------No solutions found--------')
            print(' V = ', self.V)
            print('----------------------------------')
            u = np.linspace(1e-6, self.V - 1e-6, 2048)
            e = self.eq(u)
            plt.plot(np.abs(u), e)
            plt.plot(np.abs(u), np.abs(e))
            plt.xlim(u.min(), u.max())
            plt.ylim(-10, 10)
            plt.show()
            sys.exit(1)


class Betas(Fibre):
    """Calculates the betas of the fibre mode. """

    def __init__(self, u_vec, w_vec, l_vec, o_vec, o):
        self.k = 2*pi/l_vec
        self.u = np.asanyarray(u_vec)
        self.w = np.asanyarray(w_vec)
        self.core = self.selm_core(l_vec)
        self.clad = self.selm_clad(l_vec)
        self.o_vec = o_vec
        self.o = o
        self.o_norm = self.o_vec - self.o
        return None

    def beta_func(self):
        return self.k**2*((self.core/self.u)**2 +
                          (self.clad/self.u)**2)/(1/self.u**2
                                                  + self.w**2)

    def beta_extrapo(self):
        betas = self.beta_func()
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

    def beta_dispersions(self):
        coefs = self.beta_extrapo()
        betas = np.empty_like(coefs)
        for i, c in enumerate(coefs[::-1]):
            betas[i] = c * factorial(i)
        return betas


class Modes(object):
    """docstring for Modes"""

    def __init__(self, beta_m, u_vec, w_vec, a):
        self.n = 1
        self.beta = beta_m
        self.u = u_vec
        self.w = w_vec
        self.a = a
        self.s = self.n * (1/self.u**2 + 1/self.w**2) /\
            (jv_(self.n, self.u)/(self.n*jv(self.n, self.u))
             + kv_(self.n, self.w)/(self.w*kv(self.n, self.w)))
        return None

    def E_r(self, r, theta):
        r0 = r[r <= self.a]
        r1 = r[r > self.a]

        temp0 = -1j * self.beta*self.a / \
            self.u*(0.5*(1 - self.s) * jv(self.n - 1, self.u * r0 / self.a)
                    - 0.5*(1 + self.s)*jv(self.n + 1, self.u * r0 / self.a))

        temp1 = -1j * self.beta*self.a*jv(self.n, self.u)/(self.w*kv(self.n, self.w)) \
            * (0.5*(1 - self.s) * kv(self.n - 1, self.w * r1 / self.a)
               + 0.5*(1 + self.s)*kv(self.n+1, self.w * r1 / self.a))
        temp = np.concatenate(temp0, temp1)
        return temp*np.cos(self.n*theta), temp*np.cos(self.n*theta+pi/2)

    def E_theta(self, r, theta):
        r0 = r[r <= self.a]
        r1 = r[r > self.a]

        temp0 = 1j * self.beta*self.a / \
            self.u*(0.5*(1 - self.s) * jv(self.n - 1, self.u * r0 / self.a)
                    + 0.5*(1 + self.s)*jv(self.n+1, self.u * r0 / self.a))
        temp1 = 1j * self.beta*self.a * \
            jv(self.n, self.u)/(self.w*self.kv(self.n, self.w)) \
            * (0.5*(1 - self.s) * kv(self.n - 1, self.w * r1 / self.a)
               - 0.5*(1 + self.s)*kv(self.n+1, self.e * r1 / self.a))

        temp = np.concatenate(temp0, temp1)
        return temp*np.sin(self.n*theta), temp*np.sin(self.n*theta+pi/2)

    def E_zeta(self, r, theta):
        r0 = r[r <= self.a]
        r1 = r[r > self.a]
        temp0 = jv(self.n, self.u*r/self.a)
        Ez_temp = jv(self.n, self.u) * \
            kv(self.n, self.w*r1/a)/kv(self.n, self.w)
        temp = np.concatenate(temp0, temp1)
        return temp*np.cos(self.n*theta), temp*np.cos(self.n*theta+pi/2)

    def E_carte(self, r, theta):
        Er = self.E_r(r, theta)
        Et = self.E_theta(r, theta)
        Ex, Ey, Ez = [], [], []
        for i in range(len(Er)):
            Ex.append(Er[i] * np.cos(theta) - Et[i] * np.sin(theta))
            Ey.append(Er[i] * np.sin(theta) + Et[i] * np.cos(theta))

        Ez = self.E_zeta(r, theta)
        return Ex, Ey, Ez


def main():
    margin = 1e-8
    a_med = 1e-6
    a_err_p = 0.01
    l_span = 419e-9
    l_p = 1550e-9
    low_a = a_med - a_err_p * a_med
    high_a = a_med + a_err_p * a_med

    l_vec = np.linspace(l_p - l_span, l_p + l_span, 2*512)
    a_vec = np.linspace(low_a, high_a, 2*512)
    o_vec = 2*pi * c/l_vec
    o = (o_vec[0]+o_vec[-1])/2

    fibre = Fibre()
    u_vec, w_vec = [], []
    for l, a in zip(l_vec, a_vec):
        fibre.indexes(l)
        E = Eigenvalues(l, a)
        u, w = E.eigen_solver(margin)
        u_vec.append(u)
        w_vec.append(w)
        #print(u, w, E.V, (u**2+w**2)**0.5)
    u_vec, w_vec = np.asanyarray(u_vec), np.asanyarray(w_vec)
    b = Betas(u_vec, w_vec, l_vec, o_vec, o)
    betas = b.beta_func()
    beta_coef = b.beta_extrapo()
    p = np.poly1d(beta_coef)
    betass = b.beta_dispersions()
    taylor_dispersion = np.zeros_like(o_vec)
    for i, bb in enumerate(betass):
        taylor_dispersion += (bb/factorial(i))*(o_vec - o)**i
    beta_m = p(0)

    M = Modes(beta_m, u_vec, w_vec, a_med)
    r = np.linspace(0, 2e-6, 512)
    theta = np.linspace(0, 2*pi, 512)
    E_vecs = M.E_carte(r, theta)

    """
    plt.plot(o_vec - o, betas, '-', label='real')
    plt.plot(o_vec - o, p(b.o_vec - b.o), '--', label='fitted')
    plt.plot(o_vec - o, taylor_dispersion, '*', label='Taylor')
    plt.legend()
    plt.show()
    print((betas - p(b.o_vec - b.o))/betas)
    print((taylor_dispersion - p(b.o_vec - b.o))/betas)
    """
    return None

if __name__ == '__main__':
    main()
