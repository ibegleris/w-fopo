import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def plotter_dbm(nm,sim_wind,power_watts,u,which,filename=None,title=None,im = 0):
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
		plt.savefig("figures/wavelength/wavelength_space"+filename,bbox_inched='tight')
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
		plt.savefig("figures/freequency/freequency_space"+filename,bbox_inched='tight')
	plt.close(fig)


	fig = plt.figure(figsize=(20.0, 10.0))
	for ii in range(nm):
	    plt.plot(sim_wind.t,np.abs(u[:,ii,which])**2,'*-',label='mode'+str(ii))
	plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
	plt.title("time space")
	plt.grid()
	plt.xlabel(r'$t(ps)$')
	plt.ylabel(r'$Spectrum$')
	#plt.xlim(xtlim)
	plt.legend()
	plt.savefig("figures/time_space"+str(which))
	#plt.show()
	plt.close(fig)
	return 0 


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
    plt.savefig("figures/wavelength/wavelength_space_final.png",bbox_inched='tight')
    
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
    plt.savefig("figures/freequency/freequency_space_final.png",bbox_inched='tight')
    
    plt.close('all')
    
    
    
    #plt.show()

    return 0




def animator_pdf_maker(rounds):
	"""
	Creates the animation and pdf of the FOPO at different parts of the FOPO 
	using convert from imagemagic. Also removes the pngs so be carefull

	"""
	print("making pdf's and animations.")
	space = ('wavelength','freequency')
	for sp in space:    
		os.system('rm figures/'+sp+'/*.pdf')
		strings_large = ['convert figures/'+sp+'/'+sp+'_space0.png ']
		for i in range(4):
		    strings_large.append("convert ")
		for ro in range(rounds):
			for i in range(4):
			    strings_large[i+1] += 'figures/'+sp+'/'+sp+'_space'+str(ro)+str(i+1)+'.png '
			for w in range(1,4):
				if i ==5:
					break
				strings_large[0] += 'figures/'+sp+'/'+sp+'_space'+str(ro)+str(w)+'.png '
		for i in range(4):
			os.system(strings_large[i]+'figures/'+sp+'/'+sp+'_space'+str(i)+'.pdf')
		
		string_porta = 'convert '
		string_portb = 'convert '
		for i in range(rounds):
			string_porta += 'figures/'+sp+'/'+sp+'_space'+str(i)+'portA.png '

			string_portb += 'figures/'+sp+'/'+sp+'_space'+str(i)+'portB.png '
		string_porta += 'figures/'+sp+'/'+sp+'portA.pdf '
		string_portb += 'figures/'+sp+'/'+sp+'portB.pdf '
		os.system(string_porta)
		os.system(string_portb)
		
		for i in range(4):
			os.system('convert -delay 30 figures/'+sp+'/'+sp+'_space'+str(i)+'.pdf figures/'+sp+'/'+sp+'_space'+str(i)+'.mp4')
		os.system('convert -delay 30 ' 'figures/'+sp+'/'+sp+'portA.pdf ' 'figures/'+sp+'/portA.mp4')
		os.system('convert -delay 30 ' 'figures/'+sp+'/'+sp+'portB.pdf ' 'figures/'+sp+'/portB.mp4')
		os.system('rm figures/'+sp+'/*.png')
	return None