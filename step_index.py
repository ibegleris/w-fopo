from step_index_functions import *
from data_plotters_animators import save_variables
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

@profile
def fibre_creator(a_vec, f_vec, dnerr, per=['ge', 'sio2'], filename='step_index_2m', filepath='loading_data/step_data/', N_points=512):
    """
    Creates a step index fibre for a given radius vector a_vec over a f_vec freequency window
    given. It then calculates the overlaps (Q matrixes from P Horaks paper) and exports both
    dispersion and Q_matrxes to an HDF5 file. The combinations that regard Q matrixes less than 
    unity are dissregarded. 
    """
    l_vec = c / (1e12*f_vec)

    margin = 5e-15
    o_vec = 1e-12*2*pi * c/l_vec
    o = (o_vec[0]+o_vec[-1])/2

    fibre = Fibre()

    ncore, nclad = fibre.indexes(l_vec, a_vec, per, dnerr)
    #fibre.plot_fibre_n(l_vec, a_vec, per, dnerr)

    E = Eigenvalues(l_vec, a_vec, ncore, nclad)

    u_vec = np.zeros([len(a_vec), len(l_vec)])
    w_vec = np.zeros(u_vec.shape)
    print('Creating fibre...')
    for i, a in enumerate(a_vec):
        print('New a = ', a)
        for j, l in enumerate(l_vec):
            u_vec[i, j], w_vec[i, j] = E.eigen_solver(margin, i, j)

    taylor_dispersion = np.zeros([len(a_vec), len(o_vec)])
    betas_large = []
    beta_large = []
    betas_central = np.zeros_like(a_vec)

    b = Betas(u_vec, w_vec, l_vec, o_vec, o, ncore, nclad)
    beta2_large = []
    for i, a in enumerate(a_vec):
        beta = b.beta_func(o_vec, i)
        beta_large.append(beta)
        beta_coef = b.beta_extrapo(o_vec, i)
        temp = UnivariateSpline(o_vec, beta)
        der = temp.derivative(n=2)
        der2 = der(o_vec)
        beta2_large.append(der2)
        p = np.poly1d(beta_coef)
        betas_central[i] = p(0)
        betass = b.beta_dispersions(o_vec, i)

        betas_large.append(betass)
        for j, bb in enumerate(betass):
            taylor_dispersion[i, :] += (bb/factorial(j))*(o_vec - o)**j
    beta_large = np.asanyarray(beta_large)
    M = Modes(o_vec, o, betas_central,
              u_vec, w_vec, a_vec, N_points, per, dnerr)

    M1, M2, Q_large = M.Q_matrixes()
    Export_dict = {'M1': M1, 'M2': M2,
                   'Q_large': Q_large, 'betas': betas_large,
                   'a_vec': a_vec, 'fv': f_vec, 'dnerr': dnerr}
    # print(filepath)
    return beta_large, Q_large, M, beta2_large, ncore, nclad, Export_dict


class Sidebands(object):

    def __init__(self, Q_large, a_vec, o_vec,beta_large, P=6, n2=2.5e-20):
        self.o_vec = o_vec
        print(o_vec)
        omega_m = (o_vec[0] + o_vec[-1])/2
        gama = np.real(n2 * (1e12*omega_m) / (c * (3 / Q_large[:,0,0])))

        self.dbnon = 2 * gama * P
        self.interp_betas = [
            interp1d(self.o_vec, b, kind='cubic') for b in beta_large]
        self.a_vec = a_vec

    def dbeta(self, Omega_side, omega_p):
        """
        db = np.zeros(len(self.a_vec))
        for i in range(len(self.a_vec)):
            db[i] = self.interp_betas[i](
                omega_p - Omega_side) + self.interp_betas[i](omega_p + Omega_side) \
                - 2*self.interp_betas[i](omega_p)
        print(Omega_side,db)
        """
        i = 0
        db = self.interp_betas[i](omega_p - Omega_side) + self.interp_betas[i](omega_p + Omega_side) \
                - 2*self.interp_betas[i](omega_p)# +10*self.dbnon[i]
        return db

    def solve_omega_side(self,lamdap):
        omegap = 1e-12*c*(2*pi/(lamdap*1e-9))
        Omega_side = 1#fsolve(self.dbeta, [ 1.49011612e-08], args=(omegap, ))
        omega_sides = np.linspace(0, 200)
        plt.plot(omega_sides, self.dbeta(omega_sides, omegap))
        plt.ylim([-1,1])
        plt.show()
        return Omega_side


def main(a_med, a_err_p, l_p, l_span, N_points):

    low_a = a_med - a_err_p * a_med
    high_a = a_med + a_err_p * a_med
    #l_span = 50e-9
    l_vec = np.linspace(l_p + l_span, l_p - l_span, 2**6)
    #l_vec = np.linspace(1600e-9, 1500e-9, 32)
    f_vec = 1e-12*c/l_vec
    print('Frequency step: ', np.max(
        [f_vec[i+1] - f_vec[i] for i in range(len(f_vec)-1)]), 'Thz')
    #a_vec = np.linspace(2.2e-6, 2.2e-6, 1)
    a_vec = np.linspace(low_a, high_a, 1)
    per = ['ge', 'sio2']
    err_med = 0.02*0.01
    err = err_med*np.random.randn(len(a_vec))
    betas, Q_large, M, beta2, ncore, nclad =\
        fibre_creator(a_vec, f_vec, err, per=per, N_points=N_points)[:-1]

    side = Sidebands(Q_large, a_vec, 2*pi*f_vec,betas)
    sd = side.solve_omega_side(1750)
    print(sd)
    #sys.exit()
    fig1 = plt.figure(figsize=(15, 7.5))
    for i, a in enumerate(a_vec):
        plt.plot(l_vec*1e9,
                 (-2*pi*c/l_vec**2)*beta2[i][:]*1e-24/1e-6, label=r'$\alpha = $'+'{0:.2f}'.format(a*1e6)+r'$\mu m$')
        plt.xlabel(r'$\lambda(nm)$')
        plt.ylabel(r'$D (ps^{2}/nm km)$')
    plt.axhline(0, color='black')
    plt.legend()

    fig2 = plt.figure(figsize=(15, 7.5))
    for i, a in enumerate(a_vec):
        plt.plot(l_vec*1e9, beta2[i][:], label=r'$\alpha = $' +
                 '{0:.2f}'.format(a*1e6)+r'$\mu m$')
        plt.xlabel(r'$\lambda(nm)$')
        plt.ylabel(r'$\beta_{2} (ps^{2}/m)$')
    plt.axhline(0, color='black')
    plt.legend()

    print(np.asanyarray(betas).shape)

    fig3 = plt.figure(figsize=(15, 7.5))
    plt.ticklabel_format(useOffset=False)
    for i, a in enumerate(a_vec):
        plt.plot(l_vec*1e9, betas[i][:] / (2*pi/l_vec),
                 label=r'$\alpha = $'+'{0:.6f}'.format(a*1e6)+r'$\mu m$')
        plt.xlabel(r'$\lambda(nm)$')
        plt.ylabel(r'$n_{eff}$')
    plt.plot(l_vec*1e9, ncore[0][:], '--',
             label=r'core $\alpha = $'+'{0:.2f}'.format(a_vec[0]*1e6)+r'$\mu m$')
    plt.plot(l_vec*1e9, nclad[0][:], '--',
             label=r'core $\alpha = $'+'{0:.2f}'.format(a_vec[0]*1e6)+r'$\mu m$')
    plt.legend()
    plt.ylim([1.44, 1.47])

    fig4 = plt.figure(figsize=(15, 7.5))
    plt.plot(a_vec*1e6, Q_large[:, 0, 0].real*1e-12)

    plt.xlabel(r'$\alpha(\mu m)$')
    plt.ylabel(r'$Q (\mu m)$')
    plt.legend()

    nm = len(a_vec)

    sp = N_points//4
    fig = plt.figure(figsize=(150.0, 15.0))
    plt.subplots_adjust(hspace=0.1)
    for i, v in enumerate(range(nm)):
        rc = a_vec[i]
        xc = np.linspace(-rc, rc, 1024)
        yc = np.sqrt(-xc**2+rc**2)
        xc *= 1e6
        yc *= 1e6
        v = v+1
        ax1 = plt.subplot(nm//4 + (1 if nm % 4 else 0), 4, v)

        M.set_coordinates(2*np.max(a_vec))
        M.pick_eigens(i)
        res = M.E_carte()
        M.X *= 1e6
        M.Y *= 1e6
        HE11x, HE11y = res
        E = (np.abs(HE11y[0])**2 + np.abs(HE11y[1])
             ** 2 + np.abs(HE11y[2])**2)**0.5
        Enorm = E/np.max(E)

        plt.contourf(M.X, M.Y, Enorm, 10, cmap=plt.cm.jet)
        ax1.plot(xc, yc, 'black', linewidth=2.0)
        ax1.plot(xc, -yc, 'black', linewidth=2.0)
        plt.quiver(M.X[::sp, ::sp], M.Y[::sp, ::sp], np.abs(
            HE11x[0][::sp, ::sp]), np.abs(HE11x[1][::sp, ::sp]), headlength=80)
        plt.quiver(M.X[::sp, ::sp], M.Y[::sp, ::sp], np.abs(
            HE11y[0][::sp, ::sp]), np.abs(HE11y[1][::sp, ::sp]), headlength=80)
        plt.axis('equal')
        ax1.set_xlim([-2*np.max(a_vec)*1e6, 2*np.max(a_vec)*1e6])
        ax1.set_ylim([-2*np.max(a_vec)*1e6, 2*np.max(a_vec)*1e6])
        plt.axis('off')

    plt.show()
    return None

if __name__ == '__main__':
    import matplotlib as mpl
    font = {'size': 18}
    mpl.rc('font', **font)
    a_med = 2.20e-6
    a_err_p = 0.01
    l_span = 1000e-9
    l_p = 1550e-9
    N_points = 128
    main(a_med, a_err_p, l_p, l_span, N_points)
