import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brenth, root
from scipy.constants import c, pi
from scipy.special import jv, kv
import sys


def jv_(n, z):
    return 0.5 * (jv(n-1, z) - jv(n+1, z))


def kv_(n, z):
    return -0.5 * (kv(n-1, z) + kv(n+1, z))


class Fibre(object):
    """
    Fibre class. Set to initiate the functions needed
    for a fibre ( Seilmier equations etc).
    """
    def indexes(self, n1, n2, l):
        self.n1, self.n2 = n1,n2
        self.core, self.clad = self.selm_core(l), self.selm_clad(l)
    
    def selm_core(self,l):
        return self.n1
    

    def selm_clad(self,l):
        return self.n2

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
    def __init__(self, n1, n2, l, r):
        self.indexes(n1, n2,l)
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
                margin *=10

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


def main():
    n1, n2 = 2.145, 2.1
    margin = 1e-8
    r_med = 1e-6
    r_err_p = 0.01
    l_span = 419e-9
    l_p = 1550e-9
    low_r = r_med - r_err_p * r_med
    high_r = r_med + r_err_p * r_med

    L = np.linspace(l_p - l_span, l_p + l_span,5)
    R = np.linspace(low_r, high_r,5)#np.random.uniform(low_r, high_r, len(L))
    fibre = Fibre()
    for l, r in zip(L, R):
        fibre.indexes(n1, n2,l)
        E = Eigenvalues(n1, n2, l, r)
        u, w = E.eigen_solver( margin)
        print(u, w, E.V, (u**2+w**2)**0.5)
    return None


if __name__ == '__main__':
    main()
