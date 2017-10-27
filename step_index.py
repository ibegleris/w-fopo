import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root,brenth
from scipy.constants import c, pi
from scipy.special import jv, kv
import sys
import math
from numba import jit, vectorize


def jv_(n, z):
    return 0.5 * (jv(n-1, z) - jv(n+1, z))


def kv_(n, z):
    return -0.5 * (kv(n-1, z) + kv(n+1, z))


class Fibre(object):

    def indexes(self, n1, n2):
        self.core, self.clad = n1, n2

    def V_func(self, l, r):
        self.V = (2 * pi / l)*r*(self.core**2 - self.clad**2)**0.5
        return None


class Eigenvalues(Fibre):

    def __init__(self, n1, n2, l, r):
        self.indexes(n1, n2)
        self.V_func(l, r)
        self.ratio = self.clad/self.core
        return None

    def w_f(self, u):
        return (self.V**2 - u**2)**0.5

    def eq(self, u, n=1):
        w = self.w_f(u)
        return (jv_(n, u)/(u*jv(n, w)) + kv_(n, w)/(w*kv(n, w))) * \
            (jv_(n, u)/(u*jv(n, u)) + kv_(n, w)/(w*kv(n, w))*self.ratio**2) \
            - n**2 * (1/u**2 + 1/w**2) * (1/u**2 + self.ratio**2 / w**2)


def eigen_solver(E, guess, tol=1e-8):

    #Rr = root(E.eq, guess, tol=tol,factor = E.V - 0.1)
    Rr = brenth(E.eq, 1e-6,E.V- 1e-6, full_output = True)
    if Rr[1].converged:
        return Rr[0], E.w_f(Rr[0])
    else:
        print('--------No solutions found--------')
        print('guess:', guess, ' V = ', E.V)
        print('----------------------------------')
        u = np.linspace(0.1, E.V, 2048)
        e = E.eq(u)
        plt.plot(np.abs(u), np.abs(e))
        plt.xlim(u.min(), u.max())
        plt.ylim(-1, 1)
        plt.show()
        sys.exit(1)


def main():
    n1, n2 = 1.445, 1.444
    r_med = 10.5e-6
    r_err_p = 0.01
    N = 2048*100
    fibre = Fibre()
    fibre.indexes(n1, n2)

    l, r = 1.55e-6, r_med
    E = Eigenvalues(n1, n2, l, r)
    u, w = eigen_solver(E, 1, tol=1e-8)
    print(u,w)
    return None


if __name__ == '__main__':
    main()
