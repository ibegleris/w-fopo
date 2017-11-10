from step_index_functions import *


def fibre_creator(a_vec, l_vec, err=0.000, per=[60, 20], filename='step_index_2m', N_points=512):
    margin = 5e-15
    o_vec = 2*pi * c/l_vec
    o = (o_vec[0]+o_vec[-1])/2

    fibre = Fibre()

    ncore, nclad = fibre.indexes(l_vec, a_vec, per, err)
    #fibre.plot_fibre_n(l_vec,a_vec,per,err)
    
    E = Eigenvalues(l_vec, a_vec, ncore, nclad)

    u_vec = np.zeros([len(a_vec), len(l_vec)])
    w_vec = np.zeros(u_vec.shape)
    for i, a in enumerate(a_vec):
        print('New a = ', a)
        for j, l in enumerate(l_vec):

            u_vec[i, j], w_vec[i, j] = E.eigen_solver(margin, i, j)

    
    taylor_dispersion = np.zeros([len(a_vec), len(o_vec)])
    betas_large = []
    betas_central = np.zeros_like(a_vec)

    b = Betas(u_vec, w_vec, l_vec, o_vec, o, ncore,nclad)
    beta2_large = []
    for i, a in enumerate(a_vec):
        betas = b.beta_func(i)
        beta_coef = b.beta_extrapo(i)
        temp = UnivariateSpline(o_vec, betas)
        der = temp.derivative(n=2)
        der2 = der(o_vec)
        beta2_large.append(der2)
        p = np.poly1d(beta_coef)
        betas_central[i] = p(0)
        betass = b.beta_dispersions(i)
        betas_large.append(betas)
        for j, bb in enumerate(betass):
            taylor_dispersion[i, :] += (bb/factorial(j))*(o_vec - o)**j
    #min_beta = np.min([len(i) for i in betas_large])
    #betas = np.zeros([len(a_vec), min_beta])
    #for i in range(len(betas_large)):
    #    betas[i, :] = betas_large[i][:min_beta]

    r_max = np.max(a_vec)
    x = np.linspace(-2*r_max, 2*r_max, N_points)
    y = np.linspace(-2*r_max, 2*r_max, N_points)

    M = Modes(o_vec, o, betas_central,\
         u_vec, w_vec, a_vec, x, y, per, err)
    M1, M2, Q_large = M.Q_matrixes()
    Export_dict = {'M1': M1, 'M2': M2,
                   'Q_large': Q_large, 'betas': taylor_dispersion}
    
    save_variables_step(filename,  variables=Export_dict,
                        filepath='loading_data/')
   
    return betas_large, Q_large, M, beta2_large


def main(a_med, a_err_p, l_p, l_span, N_points):

    low_a = a_med - a_err_p * a_med
    high_a = a_med + a_err_p * a_med

    l_vec = np.linspace(l_p + l_span, l_p - l_span, 1024)

    a_vec = np.linspace(0.2e-6, 1e-6, 10)

    betas, Q_large, M, beta2 = fibre_creator(a_vec, l_vec, N_points = N_points)

    fig = plt.figure(figsize=(15, 7.5))
    for i, a in enumerate(a_vec):
        plt.plot(l_vec*1e9, 1e24 *
                 beta2[i][:], label=r'$\alpha = $'+'{0:.2f}'.format(a*1e6)+r'$\mu m$')
        plt.xlabel(r'$\lambda(nm)$')
        plt.ylabel(r'$\beta_{2} (ps^{2}/m)$')
    plt.legend()
    plt.show()
    sys.exit()
    fig = plt.figure(figsize=(15, 7.5))
    plt.ticklabel_format(useOffset=False)
    for i, a in enumerate(a_vec):
        plt.plot(l_vec*1e9, betas[i][:] / (2*pi/l_vec),
                 label=r'$\alpha = $'+'{0:.2f}'.format(a*1e6)+r'$\mu m$')
        plt.xlabel(r'$\lambda(nm)$')
        plt.ylabel(r'$n_{eff}$')
    plt.legend()

    fig = plt.figure(figsize=(15, 7.5))
    plt.plot(a_vec*1e6, Q_large[:, 0, 0].real*1e-12)
    plt.xlabel(r'$\alpha(\mu m)$')
    plt.ylabel(r'$Q (\mu m)$')
    plt.legend()
    plt.show()
    max_a = 2*np.max(a_vec)
    #HE11x, HE11y, E = [], [],[]
    x, y = np.linspace(-max_a, max_a, N_points),\
            np.linspace(-max_a, max_a, N_points)
    #X, Y = np.meshgrid(x, y)
    nm = len(a_vec)
    
    sp = N_points//4
    fig = plt.figure(figsize=(150.0, 15.0))
    plt.subplots_adjust(hspace=0.1)
    for i,v in enumerate(range(nm)):
        rc = a_vec[i]
        xc = np.linspace(-rc,rc,1024)
        yc = np.sqrt(-xc**2+rc**2)
        xc *= 1e6
        yc *= 1e6
        v = v+1
        ax1 = plt.subplot(np.ceil(nm//4)+1, 4, v)
        

        M.pick_eigens(i)
        M.coordinates(x,y)
        res = M.E_carte()
        M.X *= 1e6
        M.Y *= 1e6 
        HE11x, HE11y = res
        E = (np.abs(HE11y[0])**2+ np.abs(HE11y[1])**2 + np.abs(HE11y[2])**2)**0.5
        Enorm = E/np.max(E)
        
        plt.contourf(M.X, M.Y, Enorm, 10, cmap=plt.cm.jet)
        ax1.plot(xc, yc,'black',linewidth=2.0)
        ax1.plot(xc,-yc,'black',linewidth=2.0)
        plt.quiver(M.X[::sp, ::sp], M.Y[::sp, ::sp], np.abs(
            HE11x[0][::sp, ::sp]), np.abs(HE11x[1][::sp, ::sp]), headlength=80)
        plt.quiver(M.X[::sp, ::sp], M.Y[::sp, ::sp], np.abs(
            HE11y[0][::sp, ::sp]), np.abs(HE11y[1][::sp, ::sp]), headlength=80)
        plt.axis('equal')
        ax1.set_xlim([-max_a*1e6,max_a*1e6])
        ax1.set_ylim([-max_a*1e6,max_a*1e6])
    #X *= 1e6
    #Y *= 1e6
    #Enorm = E/np.max(E)
    plt.show()
    #sys.exit()

    return None
    
if __name__ == '__main__':
    a_med = 3e-6
    a_err_p = 0.1
    l_span = 50e-9
    l_p = 1550e-9
    N_points= 128
    main(a_med, a_err_p, l_p, l_span,N_points)
