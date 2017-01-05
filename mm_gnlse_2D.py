from __future__ import division, print_function
import numpy as np
from scipy.constants import c, pi
from scipy.io import savemat
from joblib import Parallel, delayed
from scipy.fftpack import fftshift, ifftshift
import multiprocessing
import sys
#import dill
import os
from functions import *
from fft_module import *
try:
    import mkl
    max_threads = mkl.get_max_threads()
    mkl.set_num_threads(max_threads)
except ImportError:
    print(
        "MKL libaries help when you are not running in paralel.\
         There is a free academic lisence by continuum analytics")

    pass


def lams_s_vary(wave, s_pos, from_pump, int_fwm, sim_wind,
                where, P0_p1, P0_s, Dop, M1, M2, fft, ifft,index,plots):
    if from_pump:
        s_pos = where[0] - wave
    else:
        s_pos -= wave
   
    # initialisation (for fixed steps)
    u = np.zeros(
        [len(sim_wind.t), int_fwm.nm, len(sim_wind.zv)], dtype='complex128')
    U = np.zeros([len(sim_wind.t), int_fwm.nm,
                  len(sim_wind.zv)], dtype='complex128')    #
    Uabs = np.copy(U)
    # Quantum noise (Peter's version)
    pquant = np.sum(
        1.054e-34*(sim_wind.w*1e12 + sim_wind.w0)/(sim_wind.T*1e-12))
    noise_obj = Noise(sim_wind)

    noise = noise_obj.noise_func(int_fwm)

    TFWHM = 0.04
    T0 = TFWHM/2/(np.log(2))**0.5
    u[:, :, 0] = noise

    u[:, 0, 0] += (P0_p1)**0.5  # *np.exp(-sim_wind.t**2/T0)

    woff1 = -(s_pos - int_fwm.nt//2)*2*pi*sim_wind.df
    u[:, 0, 0] += (P0_s)**0.5 * np.exp(-1j*(woff1)*sim_wind.t)

    U[:, :, 0] = fftshift(sim_wind.dt*fft(u[:, :, 0]), axes=(0,))
    Uabs[:, :, 0] = fftshift(np.abs(sim_wind.dt*fft(u[:, :, 0]))**2, axes=(0,))

    "----------------------Plot the inputs------------------------------------"
    #plotter_dbm(int_fwm.nm,sim_wind.lv,w2dbm(U),sim_wind.xl,sim_wind.t,u,sim_wind.xtlim,0)
    f_p,f_s = sim_wind.fv[int_fwm.nt//2], sim_wind.fv[s_pos]
    #plotter_dbm(index,int_fwm.nm, sim_wind, w2dbm(Uabs), u, U, P0_p1,
    #            P0_s, f_p, f_s, 0)
    #plotter_dbm(int_fwm.nm,sim_wind,w2dbm(Uabs),u,U, P0_p1, P0_s, f_p, f_s,0,filename=None,title=None,im = 0)

    "-------------------------------------------------------------------------"

    # bring the raman if needed
    int_fwm.raman.raman_load(sim_wind.t, sim_wind.dt, fft, ifft)
    string = "dAdzmm_r"+str(int_fwm.raman.on)+"_s"+str(int_fwm.ss)
    func_dict = {'dAdzmm_ron_s1': dAdzmm_ron_s1,
                 'dAdzmm_ron_s0': dAdzmm_ron_s0,
                 'dAdzmm_roff_s0': dAdzmm_roff_s0,
                 'dAdzmm_roff_s1': dAdzmm_roff_s1}
    pulse_pos_dict_or = ('after propagation', "pass WDM2",
                         "pass WDM1 on port2 (remove pump)",
                         'add more pump', 'out')

    keys = ['loading_data/green_dot_fopo/pngs/' +
            str(i)+str('.png') for i in range(7)]
    D_pic = [plt.imread(i) for i in keys]

    dAdzmm = func_dict[string]
    hf = int_fwm.raman.hf

    # Define te WDM objects
    WDM1 = WDM(1200, 1050, sim_wind.lv)
    WDM2 = WDM(930, 1200, sim_wind.lv)
    WDM3 = WDM(930, 1050, sim_wind.lv)
    WDM4 = WDM(930, 1200, sim_wind.lv)
    """
    WDM1.plot(sim_wind.lv)
    WDM2.plot(sim_wind.lv)
    WDM3.plot(sim_wind.lv)
    WDM4.plot(sim_wind.lv)
    WDM1.plot_dB(sim_wind.lv)
    WDM2.plot_dB(sim_wind.lv)
    WDM3.plot_dB(sim_wind.lv)
    WDM4.plot_dB(sim_wind.lv)
    """
    
    # Define the splicer object
    splicer1 = Splicer(loss=0.2)
    splicer2 = Splicer(loss=0.1)
    # Pass the original pump through its 3 splice losses.

    # Splice1
    utemp, Utemp, Uabstemp = splicer2.pass_through((U[:, :, 0],noise_obj.noise_func_freq(int_fwm, sim_wind, fft)),
                                                   sim_wind, fft, ifft)
    u[:, :, 0], U[:, :, 0], Uabs[:, :, 0] = utemp[0], Utemp[0], Uabstemp[0]

    # Splice2
    utemp, Utemp, Uabstemp = splicer2.pass_through(
        (U[:, :, 0], noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind, fft, ifft)
    u[:, :, 0], U[:, :, 0], Uabs[:, :, 0] = utemp[0], Utemp[0], Uabstemp[0]

    # Splice3
    utemp, Utemp, Uabstemp = splicer2.pass_through(
        (U[:, :, 0], noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind, fft, ifft)
    u[:, :, 0], U[:, :, 0], Uabs[:, :, 0] = utemp[0], Utemp[0], Uabstemp[0]

    plotter_dbm(index,int_fwm.nm, sim_wind, w2dbm(Uabs), u, U, P0_p1,
                P0_s, f_p, f_s, 0, '00', 'original pump', D_pic[0],plots)

    U_original_pump = np.copy(U[:, :, 0])

    # Pass the original pump through the WDM1 port1
    utemp, Utemp, Uabstemp = WDM1.pass_through(
        (U[:, :, 0], noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind, fft, ifft)
    u[:, :, 0], U[:, :, 0], Uabs[:, :, 0] = utemp[1], Utemp[1], Uabstemp[1]

    max_rounds = 200

    for ro in range(max_rounds):
        print('round', ro)

        pulse_pos_dict = [
            'round ' + str(ro)+', ' + i for i in pulse_pos_dict_or]
        plotter_dbm(index,int_fwm.nm, sim_wind, w2dbm(Uabs), u, U, P0_p1,
                    P0_s, f_p, f_s, 0, str(ro)+'1', pulse_pos_dict[3], D_pic[5],plots)


        #print('power_init:', P0_p1,'power_after',np.average((np.abs(u[:,:,0])**2)))
       
        # Splice4
        utemp, Utemp, Uabstemp = splicer1.pass_through(
            (U[:, :, 0], noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind, fft, ifft)
        u[:, :, 0], U[:, :, 0], Uabs[:, :, 0] = utemp[0], Utemp[0], Uabstemp[0]
    
        u, U, Uabs = pulse_propagation(
            u, U, Uabs, int_fwm, M1, M2, sim_wind, hf, Dop, dAdzmm, fft, ifft)
        plotter_dbm(index,int_fwm.nm, sim_wind, w2dbm(Uabs), u, U, P0_p1,
                    P0_s, f_p, f_s, -1, str(ro)+'2', pulse_pos_dict[0], D_pic[2],plots)

        # Twice through splices after propagation
        # Splice5
        utemp, Utemp, Uabstemp = splicer1.pass_through(
            (U[:, :, -1], noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind, fft, ifft)
        u[:, :, -1], U[:, :, -1], Uabs[:,:, -1] = utemp[0], Utemp[0], Uabstemp[0]



        # Splice6
        utemp, Utemp, Uabstemp = splicer2.pass_through(
            (U[:, :, -1], noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind, fft, ifft)
        u[:, :, -1], U[:, :, -1], Uabs[:,
                                       :, -1] = utemp[0], Utemp[0], Uabstemp[0]

        # pass through WDM2 port 2 continues and port 1 is out of the loop
        utemp, Utemp, Uabstemp = WDM2.pass_through(
            (U[:, :, -1], noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind, fft, ifft)
        
        u[:, :, -1], U[:, :, -1], Uabs[:,:, -1] = utemp[1], Utemp[1], Uabstemp[1]
        out1, out2, out3 = utemp[0], Utemp[0], Uabstemp[0]
        
        plotter_dbm(index,int_fwm.nm, sim_wind, w2dbm(Uabs), u, U, P0_p1,
                    P0_s, f_p, f_s, -1, str(ro)+'3', pulse_pos_dict[1], D_pic[3],plots)

        # Splice7 after WDM2 for the signal
        utemp, Utemp, Uabstemp = splicer2.pass_through(
            (U[:, :, -1], noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind, fft, ifft)
        u[:, :, -1], U[:, :, -1], Uabs[:,:, -1] = utemp[0], Utemp[0], Uabstemp[0]

        # Pass again through WDM1 with the signal now
        utemp, Utemp, Uabstemp = WDM1.pass_through(
            (U_original_pump, U[:, :, -1]), sim_wind, fft, ifft)
        u[:, :, 0], U[:, :, 0], Uabs[:, :, 0] = utemp[1], Utemp[1], Uabstemp[1]
        

        ################################The outbound stuff#####################
        # Splice8 before WDM3
        utemp, Utemp, Uabstemp = splicer2.pass_through(
            (out2, noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind, fft, ifft)
        out1, out2, out3 = utemp[0], Utemp[0], Uabstemp[0]

        utemp, Utemp, Uabstemp = WDM3.pass_through(
            (out2, noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind, fft, ifft)
        out1, out2, out3 = utemp[0], Utemp[0], Uabstemp[0]
        u_portA, U_portA, Uabs_portA = utemp[1], Utemp[1], Uabstemp[1]
        
        Uabs_portA = w2dbm(np.reshape(Uabs_portA, (len(sim_wind.t), int_fwm.nm, 1)))
        U_portA = np.reshape(U_portA, (len(sim_wind.t), int_fwm.nm, 1))
        u_portA = np.reshape(u_portA, (len(sim_wind.t), int_fwm.nm, 1))
        plotter_dbm(index,int_fwm.nm, sim_wind, Uabs_portA , u_portA,
                    U_portA, P0_p1, P0_s, f_p, f_s,
                    -1, 'portA/'+str(ro), 
                    'round '+str(ro)+', portA',plots=plots)
        # Splice9 before WDM4
        utemp, Utemp, Uabstemp = splicer2.pass_through(
            (out2, noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind, fft, ifft)
        out1, out2, out3 = utemp[0], Utemp[0], Uabstemp[0]

        utemp, Utemp, Uabstemp = WDM4.pass_through(
            (out2, noise_obj.noise_func_freq(int_fwm, sim_wind, fft)), sim_wind, fft, ifft)
        u_portB, U_portB, Uabs_portB = utemp[0], Utemp[0], Uabstemp[0]
        

        Uabs_portB = w2dbm(np.reshape(Uabs_portB, (len(sim_wind.t), int_fwm.nm, 1)))
        U_portB = np.reshape(U_portB, (len(sim_wind.t), int_fwm.nm, 1))
        u_portB = np.reshape(u_portB, (len(sim_wind.t), int_fwm.nm, 1))
  

        plotter_dbm(index,int_fwm.nm, sim_wind, Uabs_portB , u_portB,
                    U_portB, P0_p1, P0_s, f_p, f_s, -1, 'portB/'+str(ro),
                    'round '+str(ro)+', portB',plots=plots)
    

    u[:, :, -1], U[:, :, -1], Uabs[:, :, -1] = utemp[0], Utemp[0], Uabstemp[0]

    #power_dbm = w2dbm(np.abs(Uabs[:,:,-1]))
    power_dbm = np.reshape(w2dbm(np.abs(Uabs_portB[:,0,0])), (len(sim_wind.t), 1))
    max_norm = np.max(power_dbm)

    return power_dbm, s_pos, max_norm, max_rounds


def sfft(x):
    return scifft.fft(x.T).T
def isfft(x):
    return scifft.ifft(x.T).T
def lam_p2_vary(lam_s_max,pump_index, lam_p1, power_pump_input,power_signal_input, int_fwm, plot_conv, gama, fft, ifft, where_save,plots,par=False, grid_only=False, timing=False):

    P0_p1 = power_pump_input  # [w]
    P0_s = power_signal_input  # [w]

    lamda = lam_p1*1e-9  # central wavelength of the grid[m]
    # 1052.95e-9          #central freequency of the dispersion
    lamda_c = 1051.85e-9
    "----------------------Obtain the Q matrixes------------------------------"
    M1, M2 = Q_matrixes(int_fwm.nm, int_fwm.n2, lamda, gama)
    "-------------------------------------------------------------------------"
    print(lam_p1)
    fv, where = fv_creator(700, lam_p1, int_fwm)
    sim_wind = sim_window(fv, lamda, lamda_c, int_fwm)

    if grid_only:
        return sim_wind


    #int_fwm.alphadB = 0.0011666666666666668
    slice_from_edge = (sim_wind.fv[-1] - sim_wind.fv[0])/8
    loss = Loss(int_fwm, sim_wind, amax=int_fwm.alphadB , apart_div=(sim_wind.fv[0] + slice_from_edge, sim_wind.fv[-1] - slice_from_edge))
    #loss.plot(fv)
    int_fwm.alpha = loss.atten_func_full(fv)
    "------------------------------Dispersion operator--------------------------------------"
    # betas at ps/m (given in ps/km)
    betas = np.array([[0, 0, 0, 6.755e-2, -1.001e-4, 3.671e-7]])*1e-3
    Dop = dispersion_operator(betas, lamda_c, int_fwm, sim_wind)

    "---------------------------------------------------------------------------------------"
    lam_s_max -= 2**(int_fwm.N-1)
    lam_s_max += 2
    #waves = range(100,lam_s_max,10)
    #waves = [1]
    waves = [lam_s_max]

    UU = np.zeros(
        [len(waves), len(sim_wind.t), int_fwm.nm], dtype='complex128')
    s_pos_vec = np.zeros(len(waves), dtype=int)
    mod_lam = np.zeros(len(waves))
    mod_pow = np.zeros(len(waves))
    P0_s_out = np.zeros(len(waves))
    par = 0
    if par:
        num_cores = 4
        print(
            "you have picked to run the signal wavelength in parallel. Make sure Mkl is dissabled.")
        res = Parallel(n_jobs=num_cores)(delayed(lams_s_vary)(wave, where[
            0], True, int_fwm, sim_wind, where, P0_p1, P0_s, Dop, M1, M2, fft, ifft) for wave in waves)

        for i in range(len(waves)):
            UU[i, :, :] = res[i][0]
            s_pos_vec[i] = res[i][1]
            mod_pow[i] = res[i][2].pow
            mod_lam[i] = res[i][2].lam
            P0_s_out[i] = np.real(UU[i, s_pos_vec[i], 0])
    else:
        for ii, wave in enumerate(waves):
            os.system('cp -r output output_dump/output'+str(ii))
            UU[ii, :, :], s_pos_vec[ii], mod, max_rounds = lams_s_vary(
                wave, where[0], True, int_fwm, sim_wind, where, P0_p1, P0_s, Dop, M1, M2, fft, ifft,pump_index,plots)
            P0_s_out[ii] = np.real(UU[ii, s_pos_vec[ii], 0])
            # break
            
    lams_vec = sim_wind.lv[s_pos_vec.astype(int)]
    lam_p2_nm = sim_wind.lv[-where[-1]]

    #plotter_dbm_lams_large([0], sim_wind, UU, -1, lams_vec)
    if plots:
    	animator_pdf_maker(max_rounds,pump_index)
    print('mv -r output/output'+ str(pump_index)+' output_dump_'+where_save+'/output'+str(pump_index))
    os.system('mv output/output'+ str(pump_index)+' output_dump_'+where_save+'/output'+str(pump_index))
    return 0


def main():
    "-----------------------------Stable parameters----------------------------"
    n2 = 2.5e-20                			# n2 for silica [m/W]
    nm = 1                      			# number of modes
    alphadB = 0.0011666666666666668         # loss within fibre[dB/m]
    gama = 10e-3 							# w/m
    Power_input = 13  # [W]
    Power_signal = 0  # [W]
    num_cores = 4
    "-----------------------------General options------------------------------"

    maxerr = 1e-13            				# maximum tolerable error per step
    ss = 1                      			# includes self steepening term
    ram = 'on'                  			# Raman contribution 'on' if yes
                                            # and 'off' if no
    plots = False  							# Do you want plots, be carefull it makes the code very slow!
    "----------------------------Simulation parameters-------------------------"
    N = 13
    z = 18						# total distance [m]
    nplot = 100				# number of plots
    nt = 2**N 					# number of grid points
    dzstep = z/nplot            # distance per step
    dz_less = 1e4
    dz = dzstep/dz_less         # starting guess value of the step
    wavelength_space = True    	# Set wavelength space for grid calculation

    int_fwm = sim_parameters(n2, nm, alphadB)
    int_fwm.general_options(maxerr, ss, ram)
    int_fwm.propagation_parameters(N, z, nplot, dz_less, wavelength_space)
    if num_cores > 1:
        fft = sfft
        ifft = isfft
        fft_method = 0
    else:
        fft, ifft, fft_method = pick(N, nm, 100)
    "---------------------FWM wavelengths----------------------------------------"
    lam_p1 = 1051.4  # [nm]
    lams_max_asked = 1250  # [nm]

    lv = lam_p2_vary(
        2, 0,lam_p1, Power_input,Power_signal,int_fwm, 0, gama, fft, ifft,
             'pump_wavelengths',False,par = False,grid_only = True,timing= False).lv[::-1]

    lv_lams = np.abs(np.asanyarray(lv) - lams_max_asked)
    lams_index = np.where(lv_lams == np.min(lv_lams))[0][0]+1
    print('S_max wavelength asked for: '+str(lams_max_asked),
          'With this grid the best I can do is: ', lv[lams_index])
    lensig = np.shape(range(1, lams_index))[0]
    "----------------------------------------------------------------------------"

    print(
        "The fft method that was found to be faster for your system is:", fft_method)

    
    pump_wavelengths = (1.0488816316376193e-06*1e9,)
    #Power_inputs = (3,3.5,4,4.5,5,5.5,6,6.5,7)
    Power_inputs = (3,4,4.5,5,5.5,6,6.5,7)
    #Power_inputs = (6,)
    #Power_inputs = tuple(np.arange(4,7.1,0.1))
    #Power_inputs = (0,)
    lam_p1 = pump_wavelengths[0]
    #Power_input = (13,)
    _power = create_destroy(Power_inputs)
    _power.prepare_folder()
    #Power_inputs = (6,6.5,7)
    #iters = (6,7,8)
    #iters, Power_inputs = ()
    if num_cores > 1:
        A = Parallel(n_jobs=num_cores)(delayed(lam_p2_vary)(lensig,i,lam_p1,Power_input,Power_signal,int_fwm,1
                            ,gama,fft,ifft,'pump_powers',plots,par = False,grid_only = False,timing= False) for i, Power_input in enumerate(Power_inputs))
    else:

        for i, Power_input in enumerate(Power_inputs):
            A = lam_p2_vary(lensig, i,lam_p1, Power_input,Power_signal, int_fwm, 1, gama,
                fft, ifft, 'pump_powers',plots,par=False, grid_only=False, timing=False)
            


    _power.cleanup_folder()
 
    print('\a')
   
    sys.exit()

    
    
    Power_input = 5
    pump_wavelengths = (1047.5, 1047.9, 1048.3, 1048.6,
                        1049.0, 1049.5, 1049.8, 1050.2, 1050.6, 1051.0, 1051.4)
    pump_wavelengths = tuple(np.arange(1045,1052.1,0.2))
    #pump_wavelengths = (1051.8,1052)
    #iters = (24,25)
    _wavelength = create_destroy(pump_wavelengths)
    #zip(iters, pump_wavelengths)


    _wavelength.prepare_folder()
    #print(pump_wavelengths)
    if num_cores > 1:
        A = Parallel(n_jobs=num_cores)(delayed(lam_p2_vary)(lensig,i,lam_p1,Power_input,Power_signal,int_fwm,1
                            ,gama,fft,ifft,'pump_wavelengths',par = False,grid_only = False,timing= False) for i, lam_p1 in enumerate(pump_wavelengths))
    else:
        print('going parrallel')
        for i,lam_p1 in enumerate(pump_wavelengths):
        	A = lam_p2_vary(lensig,i,lam_p1,Power_input,Power_signal,int_fwm,1
        						,gama,fft,ifft,'pump_wavelengths',par = False,grid_only = False,timing= False)

    _wavelength.cleanup_folder()

        
    return None


if __name__ == '__main__':
    main()



