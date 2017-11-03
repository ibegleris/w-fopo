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
from itertools import combinations_with_replacement,permutations

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
        self.u = u_vec
        self.w = w_vec
        self.core = self.selm_core(l_vec)
        self.clad = self.selm_clad(l_vec)
        self.o_vec = o_vec
        self.o = o
        self.o_norm = self.o_vec - self.o
        return None

    def beta_func(self, i):
        dr =  self.k**2*((self.core/self.u[i, :])**2 +
                          (self.clad/self.u[i, :])**2)/(1/self.u[i, :]**2
                                                        + self.w[i, :]**2)
        print(dr/self.k)
        return dr
    def beta_extrapo(self, i):
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


class Modes(object):
    """docstring for Modes"""

    def __init__(self,o_vec,o_c, beta_c, u_vec, w_vec, a_vec,x,y):
        self.n = 1
        o_norm = o_vec - o_c
        self.coordintes(x,y)
        self.beta_c = beta_c
        print(self.beta_c)
        print(o_c)
        print(o_c/c)
        self.neff = self.beta_c/ (o_c / c)
        print(self.neff)
        sys.exit()
        self.u_vec,self.w_vec = np.zeros(u_vec.shape[0]),\
                            np.zeros(u_vec.shape[0])
        for i in range(u_vec.shape[0]):
            self.u_vec[i] = interp1d(o_norm, u_vec[i,:],kind = 'cubic')(0)
            self.w_vec[i] = interp1d(o_norm, w_vec[i,:],kind = 'cubic')(0)
        self.a = a_vec[i]
        return None
    
    def coordintes(self,x,y):
        x,y = np.meshgrid(x,y)
        self.R = (x**2 + y**2)**0.5
        self.T = np.arctan(y/x)
        return None

    def pick_eigens(self,i):
        self.u = self.u_vec[i]
        self.w = self.w_vec[i]
        self.beta = self.beta_c[i] 
        self.s = self.n * (1/self.u**2 + 1/self.w**2) /\
            (jv_(self.n, self.u)/(self.n*jv(self.n, self.u))
             + kv_(self.n, self.w)/(self.w*kv(self.n, self.w)))
        return None

    def E_r(self, r, theta):
        r0_ind =np.where(r <= self.a)
        r1_ind = np.where(r > self.a)
        temp = np.zeros(r.shape,dtype = np.complex128)
        r0,r1 = r[r0_ind], r[r1_ind]
        temp[r0_ind] = -1j * self.beta*self.a / \
            self.u*(0.5*(1 - self.s) * jv(self.n - 1, self.u * r0 / self.a)
                    - 0.5*(1 + self.s)*jv(self.n + 1, self.u * r0 / self.a))
       
        temp[r1_ind] = -1j * self.beta*self.a*jv(self.n, self.u)/(self.w*kv(self.n, self.w)) \
            * (0.5*(1 - self.s) * kv(self.n - 1, self.w * r1 / self.a)
               + 0.5*(1 + self.s)*kv(self.n+1, self.w * r1 / self.a))

   
        return temp*np.cos(self.n*theta), temp*np.cos(self.n*theta+pi/2)

    def E_theta(self, r, theta):
        r0_ind =np.where(r <= self.a)
        r1_ind = np.where(r > self.a)
        temp = np.zeros(r.shape,dtype = np.complex128)
        r0,r1 = r[r0_ind], r[r1_ind]
        temp[r0_ind] =1j * self.beta*self.a / \
            self.u*(0.5*(1 - self.s) * jv(self.n - 1, self.u * r0 / self.a)
                    + 0.5*(1 + self.s)*jv(self.n+1, self.u * r0 / self.a))
      
        temp[r1_ind] = 1j * self.beta*self.a * \
            jv(self.n, self.u)/(self.w*kv(self.n, self.w)) \
            * (0.5*(1 - self.s) * kv(self.n - 1, self.w * r1 / self.a)
               - 0.5*(1 + self.s)*kv(self.n+1, self.w * r1 / self.a))
        return temp*np.sin(self.n*theta), temp*np.sin(self.n*theta+pi/2)

    def E_zeta(self, r, theta):
        r0_ind =np.where(r <= self.a)
        r1_ind = np.where(r > self.a)
        temp = np.zeros(r.shape,dtype = np.complex128)
        r0,r1 = r[r0_ind], r[r1_ind]
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

        return (Ex[0], Ey[0], Ez[0]),(Ex[1], Ey[1], Ez[1])
    
    def Q_matrixes(self):
        self.E = self.E_carte()
        self.combination_matrix_assembler()
        print(self.M1_top)
        return  
    
    def combination_matrix_assembler(self):
        tot = []
        for i in combinations_with_replacement(range(len(self.E)),4):
             l = permutations(i)
             for j in l:
                 tot.append(j)
        a = list(set(tot))
        a.sort()
        self.M1_top = a
        return None
    def integrate(self,x,y,z):
        """
        Integrates twice using Simpsons rule from scipy
        to allow 2D  integration.
        """
        return simps(simps(z, x), y)

def main():
    margin = 1e-8
    a_med = 1e-6
    a_err_p = 0.01
    l_span = 300e-9
    l_p = 1550e-9
    low_a = a_med - a_err_p * a_med
    high_a = a_med + a_err_p * a_med

    l_vec = np.linspace(l_p - l_span, l_p + l_span, 20)
    a_vec = np.linspace(low_a, high_a, 10)
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
    for i, a in enumerate(a_vec):
        betas = b.beta_func(i)
        beta_coef = b.beta_extrapo(i)
        p = np.poly1d(beta_coef)
        betas_central[i] = p(0)
        betass = b.beta_dispersions(i)
        betas_large.append(betass)
        for j, bb in enumerate(betass):
            taylor_dispersion[i, :] += (bb/factorial(j))*(o_vec - o)**j
    min_beta = np.min([len(i) for i in betas_large])
    betas = np.zeros([len(a_vec), min_beta])
    for i in range(len(betas_large)):
        betas[i, :] = betas_large[i][:min_beta]

    r_max = np.max(a_vec)
    N_points = 128
    x = np.linspace(-2*r_max,2*r_max,N_points)
    y = np.linspace(-2*r_max,2*r_max,N_points)

    M = Modes(o_vec,o, betas_central, u_vec, w_vec, a_vec,x,y)
    HE11x,HE11y = [],[]
    for i in range(len(a_vec)):
        M.pick_eigens(i)
        res = M.E_carte()
        HE11x.append(res[0])
        HE11y.append(res[1])
        M.Q_matrixes()
    E = (np.abs(HE11x[0][0])**2 + np.abs(HE11x[0][1])**2 + np.abs(HE11x[0][2])**2)**0.5
    

    X,Y = np.meshgrid(x,y)
   
    Enorm = E/np.max(E)
    sp = 50
    fig = plt.figure()
    plt.contourf(X, Y, Enorm,10,cmap = plt.cm.jet)
    plt.quiver(X[::sp,::sp], Y[::sp,::sp], np.abs(HE11x[0][0][::sp,::sp]), np.abs(HE11x[0][1][::sp,::sp]),headlength=10)
    #plt.colorbar()
    plt.show()

    E = (np.abs(HE11y[0][0])**2 + np.abs(HE11y[0][1])**2 + np.abs(HE11y[0][2])**2)**0.5
    Enorm = E/np.max(E)

    fig = plt.figure()
    plt.contourf(X, Y, Enorm,10,cmap = plt.cm.jet)
    plt.quiver(X[::sp,::sp], Y[::sp,::sp], np.abs(HE11y[0][0][::sp,::sp]), np.abs(HE11y[0][1][::sp,::sp]),headlength=10)

    plt.show()
    return None

if __name__ == '__main__':
    main()
