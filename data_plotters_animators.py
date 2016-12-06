import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os
import h5py

def plotter_dbm(nm,sim_wind,power_watts,u,U,P0_p,P0_s,which,filename=None,title=None,im = 0):
	fig = plt.figure(figsize=(20.0, 10.0))
	for ii in range(nm):
		plt.plot(sim_wind.lv,np.real(power_watts[:,ii,which]),'-*',label='mode'+str(ii))
	plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
	plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
	plt.xlabel(r'$\lambda (nm)$',fontsize=18)
	plt.ylabel(r'$Spectrum (a.u.)$',fontsize=18)
	plt.ylim([-80,80])
	plt.xlim([np.min(sim_wind.lv),np.max(sim_wind.lv)])
	plt.xlim([900,1250])
	plt.title(title)
	plt.grid()
	if type(im) != int:
		newax = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor='NE')
		newax.imshow(im)
		newax.axis('off')
	if filename == None:
		plt.show()
	else:
		plt.savefig("output/figures/wavelength/"+filename,bbox_inched='tight')
	
	plt.close(fig)

	fig = plt.figure(figsize=(20.0, 10.0))
	for ii in range(nm):
		plt.plot(sim_wind.fv,np.real(power_watts[:,ii,which]),'-*',label='mode'+str(ii))
	plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
	plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
	plt.xlabel(r'$f (THz)$',fontsize=18)
	plt.ylabel(r'$Spectrum (a.u.)$',fontsize=18)
	plt.xlim([np.min(sim_wind.fv),np.max(sim_wind.fv)])
	plt.ylim([-80,80])
	plt.title(title)
	plt.grid()
	if type(im) != int:
		newax = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor='NE')
		newax.imshow(im)
		newax.axis('off')
	if filename == None:
		plt.show()
	else:
		plt.savefig("output/figures/freequency/"+filename,bbox_inched='tight')
	plt.close(fig)


	fig = plt.figure(figsize=(20.0, 10.0))
	for ii in range(nm):
	    plt.plot(sim_wind.t,np.abs(u[:,ii,which])**2,'*-',label='mode'+str(ii))
	plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
	plt.title("time space")
	plt.ylim([0,160])
	plt.grid()
	plt.xlabel(r'$t(ps)$')
	plt.ylabel(r'$Spectrum$')
	if type(im) != int:
		newax = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor='NE')
		newax.imshow(im)
		newax.axis('off')

	
	if filename == None:
		plt.show()
	else:

		plt.savefig("output/figures/time/"+filename)

		try:
			save_variables('data_large', filename[:-1]+'/'+filename[-1],filepath = 'output/data/',U=U, t = sim_wind.t, u = u,
		 					fv = sim_wind.fv, lv = sim_wind.lv, power_watts=power_watts,
		 					 which = which,nm=nm,P0_p = P0_p, P0_s = P0_s)
		except RuntimeError:
			os.system('rm output/data/data_large.hdf5')
			save_variables('data_large', filename[:-1]+'/'+filename[-1],filepath = 'output/data/',U=U, t = sim_wind.t, u = u,
		 					fv = sim_wind.fv, lv = sim_wind.lv, power_watts=power_watts,
		 					 which = which,nm=nm,P0_p = P0_p, P0_s = P0_s)
			pass
	plt.close(fig)

	return 0 

def plotter_dbm_load():
	#class sim_window(object):
	plotter_dbm(nm, sim_wind, power_watts, u, which)
	return None


def plotter_dbm_lams_large(modes,sim_wind,U,which,lams_vec):
    fig = plt.figure(figsize=(20.0, 10.0))
    for mode in modes:     
        for lamm,lamda in enumerate(lams_vec):    
            plt.plot(sim_wind.lv,np.real(U[lamm,:,mode]),'-*',label=str(mode))
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.xlabel(r'$\lambda (nm)$',fontsize=18)
    plt.ylabel(r'$Spectrum (a.u.)$',fontsize=18)
    plt.grid()
    #plt.ylim([-70,0])
    plt.xlim([900,1250])
    plt.xlim([np.min(sim_wind.lv),np.max(sim_wind.lv)])
    plt.savefig("output/figures/wavelength/wavelength_space_final.png",bbox_inched='tight')
    
    #plt.close('all')
   
    fig = plt.figure(figsize=(20.0, 10.0))
    for mode in modes:     
        for lamm,lamda in enumerate(lams_vec):    
            plt.plot(sim_wind.fv,np.real(U[lamm,:,mode]),'-*',label=str(mode))
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.xlabel(r'$f (THz)$',fontsize=18)
    plt.ylabel(r'$Spectrum (a.u.)$',fontsize=18)
    plt.xlim([np.min(sim_wind.fv),np.max(sim_wind.fv)])
    plt.grid()
    plt.savefig("output/figures/freequency/freequency_space_final.png",bbox_inched='tight')
    
    plt.close('all')
    return 0




def animator_pdf_maker(rounds):
	"""
	Creates the animation and pdf of the FOPO at different parts of the FOPO 
	using convert from imagemagic. Also removes the pngs so be carefull

	"""
	print("making pdf's and animations.")
	space = ('wavelength','freequency','time')
	for sp in space:    
		file_loc = 'output/'+'figures/'+sp+'/'
		strings_large = ['convert '+file_loc+'00.png ']
		for i in range(4):
		    strings_large.append("convert ")
		for ro in range(rounds):
			for i in range(4):
			    strings_large[i+1] += file_loc+str(ro)+str(i+1)+'.png '
			for w in range(1,4):
				if i ==5:
					break
				strings_large[0] += file_loc+str(ro)+str(w)+'.png '
		for i in range(4):
			os.system(strings_large[i]+file_loc+str(i)+'.pdf')
		
		

		file_loca = file_loc+'portA/'
		file_locb = file_loc+'portB/' 
		string_porta = 'convert '
		string_portb = 'convert '
		for i in range(rounds):
			string_porta += file_loca + str(i) + '.png '
			string_portb += file_loca + str(i) + '.png '

		string_porta += file_loca+'porta.pdf '
		string_portb += file_locb+'portb.pdf '
		os.system(string_porta)
		os.system(string_portb)
		
		for i in range(4):
			os.system('convert -delay 30 '+file_loc+str(i)+'.pdf '+file_loc+str(i)+'.mp4')
		os.system('convert -delay 30 '+ file_loca +'porta.pdf ' +file_loca+'porta.mp4 ')
		os.system('convert -delay 30 '+ file_locb +'portb.pdf ' +file_locb+'portb.mp4 ')
		
	
		for i in (file_loc,file_loca,file_locb):
			print('rm '+ i +'*.png')
			os.system('rm '+ i +'*.png')
		os.system('sleep 5')
	return None





def read_variables(filename,layer,filepath=''):
	with h5py.File(filepath+str(filename)+'.hdf5','r') as f:
		D = {}

		for i in f.get(layer).keys():
			print(layer + '/' + str(i))
			D[str(i)] = f.get(layer + '/' + str(i)).value
	return D

def save_variables(filename, layers,filepath = '',**variables):
    with h5py.File(filepath + filename +'.hdf5','a') as f:
        for i in (variables):
            f.create_dataset(layers+'/'+str(i), data=variables[i])
    return None