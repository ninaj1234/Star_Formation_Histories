import os
import sys
import operator
#from tqdm import tqdm
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import rc

from astropy.io import fits

import astropy.cosmology as ac
lcdm = ac.FlatLambdaCDM(H0=70,Om0=0.3)
 
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip
import bagpipes as pipes
from bagpipes import utils

z = 5

sphinx_filt_list = np.loadtxt("sphinx_filt_list_nircam_primer.txt", dtype="str")
my_catalogue = "../corrected_mock_photometry_nircam_sphinx_z_"+str(z)+".dat"

fit_info = {}                            # The fit instructions dictionary
fit_info["redshift"] = z     # Vary observed redshift from 0 to 10


def load_data(ID):
    # load data in mJy
    halo_ids = np.loadtxt(my_catalogue,usecols=[0])
    loss = np.loadtxt(my_catalogue,usecols=[1])

    row = int(ID)

    # add data from MY OWN psf-matched JWST catalogue
    data_fluxes = np.loadtxt(my_catalogue,usecols=[2,3,4,5,6,7,8,9])

    jwst_fluxes = data_fluxes[row]
    jwst_fluxerrs = jwst_fluxes*0.1 # SNR of 10 for all bands for now - same as desika
    
    halo_id = halo_ids[row]
    los = loss[row]
    
    print ("halo id", halo_id, "los", los)

    print ("jwst_fluxes",jwst_fluxes)
    print ("jwst_fluxerrs",jwst_fluxerrs)

    photometry = np.c_[jwst_fluxes, jwst_fluxerrs]

    return photometry
    
    
def ascii_to_df(file,**kwargs):
    """ Opens .dat/.ascii files that to pandas that may have '#' as first column
        any **kwargs passed to read_csv()
    """
    
    df = pd.read_csv(file,sep="\\s+",**kwargs)
    
    if df.columns[0]=='#':
        df_cols = df.columns
        df.drop(labels=df_cols[-1],axis=1,inplace=True)
        df.columns = df_cols[1:]
 
    return df

 
def plot_pipes_SED(lamb_m,spectrum_atten,fit,save=True):
 
    cm_subsection = np.linspace(0., 1., 6)
    #colors = [cm.rainbow(x) for x in cm_subsection]
    import cmasher as cmr
    colors = [cmr.guppy(x) for x in cm_subsection]
    mycolor1 = colors[3]
    mycolor2 = colors[5]
    #old color was 'crimson'
    
    fit.posterior.get_advanced_quantities()
 
    fig,axs = plt.subplots(figsize=(4,3))
 
    obs_photom = fit.galaxy.photometry #flam in erg/s/cm2/A
    obs_photom[:,1:] *= 3.3354e10 * obs_photom[:,0][:,None]**2 #erg/s/cm2/A -> uJy
 
    axs.errorbar(obs_photom[:,0],obs_photom[:,1],yerr=obs_photom[:,2],
                 marker='o',ls='None',markersize=6,ecolor='grey',markeredgewidth=1.5,
                 markerfacecolor='white',markeredgecolor='grey',zorder=1000,alpha=1.0)
 
    model_photom = np.concatenate([fit.galaxy.filter_set.eff_wavs[:,None],
                                   np.percentile(fit.posterior.samples["photometry"], q=(50,16, 84), axis=0).T],axis=1)
    model_photom[:,1:] *= 3.3354e10 * model_photom[:,0][:,None]**2 #erg/s/cm2/A -> uJy
 
    model_photom_yerr = np.concatenate([(model_photom[:,1]-model_photom[:,2])[None],
                                        (model_photom[:,3]-model_photom[:,1])[None]],axis=0)
    axs.errorbar(model_photom[:,0],model_photom[:,1],yerr=model_photom_yerr,
                 marker='None',ls='None',
                 elinewidth=6,ecolor=mycolor1,capsize=0)
 
    # if redshift is fixed in fit, no longer has posterior.samples
    try:
        redshift = np.median(fit.posterior.samples["redshift"])
    except KeyError:
        redshift = fit.fitted_model.model_components["redshift"]
 
    spec = np.concatenate([(1.+redshift)*fit.posterior.model_galaxy.wavelengths[:,None],
                           np.percentile(fit.posterior.samples["spectrum_full"],q=(50, 16, 84), axis=0).T],axis=1)
    spec[:,1:] *= 3.3354e10 * spec[:,0][:,None]**2 #erg/s/cm2/A -> uJy
 
    axs.plot(spec[:,0],spec[:,1],color=mycolor1,lw=1,alpha=0.9,zorder=0)
    axs.fill_between(spec[:,0],spec[:,2],spec[:,3],color=mycolor1,lw=0,alpha=0.2,zorder=0)
    
    axs.plot(lamb_m*1e10,spectrum_atten,color='grey',lw=1,alpha=0.3)
    
    print ("spec[:,0]",spec[:,0])
    print ("spectrum_atten",spectrum_atten)
    
    axs.set_xlim(xmin=600,xmax=60000)
    axs.set_ylim(ymin=np.min(obs_photom[:,1])-0.2*np.abs(np.min(obs_photom[:,1])),ymax=np.max(obs_photom[:,1])+0.2*np.abs(np.max(obs_photom[:,1])))
 
    #axs.set_ylim(ymin=np.min(obs_photom[:,1])*0.1,ymax=np.max(obs_photom[:,1])*10)


    #axs.set_yscale("log")

    axs.set_xlabel(r'$\lambda_{\rm obs}/\mu m$')
    axs.set_ylabel(r'F$_{\nu}/\mu {\rm Jy}$')
 
    axs.tick_params(axis='both',direction='in',which='both',top=True,right=True)
    axs.tick_params(axis='both',which='major',length=4,width=1,labelsize=10)
    axs.tick_params(axis='both',which='minor',length=2,width=1)
    axs.xaxis.set_major_locator(ticker.AutoLocator())
    axs.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    axs.yaxis.set_major_locator(ticker.AutoLocator())
    axs.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
 
    axs.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(x/10000)))
 
    if save:
        path = "pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_SED_new_corrected.pdf"
        plt.savefig(path,bbox_inches='tight')
        print(f'Saved SED: {path}')
 
    return 0
    
    
catalog_df = ascii_to_df('../corrected_mock_photometry_nircam_sphinx_z_'+str(z)+'.dat')
#catalog_df.rename(mapper={'id':'ID'},axis=1,inplace=True)
 
#catalog_df['ID_numeric'] = catalog_df.ID.values
#catalog_df['origin'] = 'RJM'
#catalog_df['ID'] = 'RJM_'+catalog_df.cat+'_'+catalog_df.ID.astype(int).astype(str)
 
 
def get_fit_object(idx, run_info, n_posterior=1000):
    # retrieve fit,
    # assumes [catalog_df, load_photometry, primer_filter_set, fit_instructions] in global namespace
    print ("idx", idx)
    galaxy = pipes.galaxy(ID=idx,
                          load_data=load_data,
                          spectrum_exists=False,
                          filt_list=sphinx_filt_list)
    print ("galaxy loaded", galaxy)
    fit = pipes.fit(galaxy=galaxy,
                    fit_instructions=fit_info,
                    run=run_info,
                    n_posterior=n_posterior)
    print ("fit loaded", fit)

    return fit
    
    
def extract_fit_results(fit,los,z,lamb_m,spectrum_atten,create_SED_plot=True):
    """ Pull emission line EW for the posteriors & save to .npz file
        [Optionally create & save an SED plot]
    """
 
    if create_SED_plot:
        plot_pipes_SED(lamb_m,spectrum_atten,fit=fit,save=True)
 
    fit.posterior.get_advanced_quantities()
    power_law_func = lambda x,a,b: (10**a)*x**b
 
    ### Initialise model
    fit.posterior.fitted_model._update_model_components(fit.posterior.samples2d[0, :])
    model_comp = fit.posterior.fitted_model.model_components
    model = pipes.model_galaxy(model_comp, spec_wavs=np.arange(30000., 55000., 1))
 
    #define index list of dict in format used by Bagpipes for clarity, not all info used
    Hbeta_ew_dict = dict(name='Hb_EW',type='EW',feature=[4851,4871],continuum=[[4800,4851],[4861,4900]])
    OIII5007_ew_dict = dict(name='OIII5007_EW',type='EW',feature=[4997,5017],continuum=[[4980,4997],[5017,5040]])
    OIII4960_ew_dict = dict(name='OIII4960_EW',type='EW',feature=[4950,4970],continuum=[[4930,4950],[4970,4980]])
    Halpha_ew_dict = dict(name='Ha_EW',type='EW',feature=[6552,6572],continuum=[[6450,6552],[6572,6650]]) # H  1  6562.81A
    
    # index_list = [OIII5007_ew_dict,OIII4960_ew_dict,Hbeta_ew_dict]
    index_list = [Halpha_ew_dict]
 
    line_ews = np.zeros((fit.n_posterior,len(index_list)))
    # continuum_region = (4540,5225)
    continuum_region = (6400,6700)
    line_wavelengths = np.array([np.median(idx['feature']) for idx in index_list])
 
    for j in range(fit.n_posterior):
        
        #Update pipes.fit.posterior with parameters from sample
        fit.posterior.fitted_model._update_model_components(fit.posterior.samples2d[j, :])
        
        #Update pipes.model_galaxy with parameters
        model.update(fit.posterior.fitted_model.model_components)
        
        #isolate redshift
        z = model.model_comp['redshift']
       
        #Extract updated model line_fluxes
        # f_line = np.array([model.line_fluxes["O  3  5006.84A"],
        #                    model.line_fluxes["O  3  4958.91A"],
        #                    model.line_fluxes["H  1  4861.33A"] ])
        f_line = np.array([model.line_fluxes["H  1  6562.81A"]])
 
        # Get fit spectrum posterior realisation
        spec_rest_wavelengths = fit.posterior.model_galaxy.wavelengths
        spec_flux_posterior = fit.posterior.samples["spectrum_full"][j]
        
        # Isolate galaxy continuum region
        continuun_region_bool = ( (spec_rest_wavelengths>=continuum_region[0])
                                & (spec_rest_wavelengths<=continuum_region[1]) )
        spec_rest_wavelengths,spec_flux_posterior = ( spec_rest_wavelengths[continuun_region_bool],
                                                      spec_flux_posterior[continuun_region_bool] )
 
        # Perform a sigma_clip
        spec_flux_posterior_masked = sigma_clip(spec_flux_posterior,
                                                cenfunc='median',stdfunc='mad_std',
                                                masked=True,sigma=2)
 
        # fit powerlaw through continuum region
        func_param,_ = curve_fit(power_law_func,
                                 spec_rest_wavelengths[~spec_flux_posterior_masked.mask],
                                 spec_flux_posterior_masked.data[~spec_flux_posterior_masked.mask],
                                 maxfev=int(1e4))
         
        # sample fitted continuum at wavelengths of each line
        f_continuum = power_law_func(line_wavelengths,*func_param)
            
        line_ews[j] = (f_line / f_continuum) / (1+z) #calculate EW0= (1+z)^-1  * fline/fcontinuum
 
    ### Save data to pipes/{run}/extracted/ [& create if doesn't exist]
    working_dir = os.getcwd()
 
    if not os.path.exists(working_dir + "/pipes/extracted"):
        os.mkdir(working_dir + "/pipes/extracted")
    if fit.run != ".":
        if not os.path.exists("pipes/extracted/" + fit.run):
            os.mkdir("pipes/extracted/" + fit.run)
 
    # file is 2d np.ndarray (n_posterior, n_index)
    # path = "pipes/extracted/" + fit.run + "/" + fit.galaxy.ID + "_EW0.npy"
    path = "pipes/extracted/" + fit.run + "/" + fit.galaxy.ID + "_EW0_Halpha_corrected.npy"
 
    np.save(path, line_ews)
    print(f'Saved: {path}')
    
    # need to fix this
    true_line = output_fits['HI_6562.8_dir_'+str(my_los)][i]
    true_continuum = output_fits['cont_6583_dir_'+str(my_los)][i]
    true_ew = true_line*1./true_continuum*1./(1+z)
    
    # IS THE (1+z) above correct? results look more reasonable without
    
    path = "pipes/extracted/" + fit.run + "/" + fit.galaxy.ID + "true_EW0_Halpha.npy"
 
    np.save(path, true_ew)
    print(f'Saved: {path}')
    
    return 0
    
    
def add_z_axis(ax, z_on_y=False, zvals=[5,6,7,8,9,10,11,12]):

    if z_on_y:
        ax2 = ax.twinx()
        ax2.set_yticks(np.interp(zvals, utils.z_array, utils.age_at_z))
        ax2.set_yticklabels(["$" + str(z) + "$" for z in zvals])
        ax2.set_ylim(ax.get_ylim())

    else:
        ax2 = ax.twiny()
        ax2.set_xticks(np.interp(zvals, utils.z_array, utils.age_at_z))
        ax2.set_xticklabels(["$" + str(z) + "$" for z in zvals])
        ax2.set_xlim(ax.get_xlim())
        
    ax2.set_xlabel("$\\mathrm{Redshift}$")


    if z_on_y:
        ax2.set_ylabel("$\\mathrm{Redshift}$")
        ax2.set_xlabel("$\\mathrm{Redshift}$")

    return ax2

def add_sfh_posterior(my_annotation, fit, ax, z_axis=True, zorder=4, zvals=[5,6,7,8,9,10,12,15,20]):
    
    # Calculate median redshift and median age of Universe
    if "redshift" in fit.fitted_model.params:
        redshift = np.median(fit.posterior.samples["redshift"])

    else:
        redshift = fit.fitted_model.model_components["redshift"]

    age_of_universe = np.interp(redshift, utils.z_array, utils.age_at_z)

    # Calculate median and confidence interval for SFH posterior
    post = np.percentile(fit.posterior.samples["sfh"], (16, 50, 84), axis=0).T

    # Plot the SFH
    x = age_of_universe - fit.posterior.sfh.ages*10**-9

    my_post = post[:, 1]

    #print ("post[:, 1]",my_post)
    #print ("is post >0 anywhere?",my_post[my_post>0])
    
    ax.plot(x, post[:, 1], color='crimson', zorder=zorder+1,label=my_annotation)
    ax.fill_between(x, post[:, 0], post[:, 2], color='crimson',
                    alpha=0.2, zorder=zorder, lw=0)
                 
    print ("x",x)
    print("np.max(x)",np.max(x))
    
    # just for example 135 - FIX FOR THE REST
    
    #ax.vlines(np.max(x)-0.0736233073569571,ymin=2e-3,ymax=20,color='crimson')
    #ax.axvspan(np.max(x)-0.04883316984297192, np.max(x)-0.1021537977205526, alpha=0.2, color='crimson')

    #ax.set_ylim(0., np.max([ax.get_ylim()[1], 1.1*np.max(post[:, 2])]))
    ax.set_xlim(age_of_universe, 0)

    # Add redshift axis along the top
    if z_axis:
        ax2 = add_z_axis(ax, zvals=zvals)
        ax2.tick_params(axis='both',direction='in',which='both',top=True,bottom=False,right=True)


    # Set axis labels
    ax.set_ylabel("$\\mathrm{SFR\\ /\\ M_\\odot\\ \\mathrm{yr}^{-1}}$")
    ax.set_xlabel("$\\mathrm{Age\\ of\\ Universe\\ /\\ Gyr}$")
    ax.legend()
    if z_axis:
        return ax2
    
    
def extract_sfh_results(halo_id,fit,los,z,lamb_m,spectrum_atten,my_annotation,create_SED_plot=True):
    """ Pull sfh """
 
    ### Initialise model
    fit.posterior.fitted_model._update_model_components(fit.posterior.samples2d[0, :])
    model_comp = fit.posterior.fitted_model.model_components

    '''
    for j in range(fit.n_posterior):
        
        #Update pipes.fit.posterior with parameters from sample
        fit.posterior.fitted_model._update_model_components(fit.posterior.samples2d[j, :])
        
        #Update pipes.model_galaxy with parameters
        model.update(fit.posterior.fitted_model.model_components)
    '''
    
    fig = plt.figure(figsize=(4,2.5))
    ax = plt.subplot()

    add_sfh_posterior(my_annotation, fit, ax)
    
    my_sfh_file = '/Users/rachelcochrane/Documents/allz_sphinx/sphinx_data/sfhs_z'+str(z)+'.json'
    with open(my_sfh_file) as project_file:
        dat_sfh = json.load(project_file)

    #plt.plot(-1e-3*np.array(dat_sfh['age_bins'][1:])+0.748,dat_sfh['sfhs'][str(halo_id)],color='grey',alpha=0.7)
    # correct this each time
    if z==5:
        my_addition = 1.152
    elif z==6:
        my_addition = 0.914
    elif z==7:
        my_addition = 0.748
    elif z==8:
        my_addition = 0.627
    elif z==9:
        my_addition = 0.535
    elif z==10:
        my_addition = 0.464
        
    plt.plot(-1e-3*np.array(dat_sfh['age_bins'][1:])+my_addition,dat_sfh['sfhs'][str(halo_id)],color='grey',alpha=0.7)

    ax.set_ylim(2e-3,20)
    ax.set_yscale("log")
    
    ax.tick_params(axis='both',direction='in',which='both',top=False,right=True)
    ax.tick_params(axis='both',which='major',length=4,width=1,labelsize=10)
    ax.tick_params(axis='both',which='minor',length=2,width=1)
    yaxvals = [0.01,0.1,1,10]
    ax.set_yticks(yaxvals)
    ax.set_yticklabels(["$" + str(myy) + "$" for myy in yaxvals])
    
    path = "pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_SFH_new_corrected.pdf"
    plt.savefig(path,bbox_inches='tight')
    print(f'Saved SED: {path}')



#my_path = 'burstycont'
#my_path = 'cont'
my_path = 'dbpl'
#my_path='singleburst'

my_cat_path = '/Users/rachelcochrane/Documents/allz_sphinx/sed_fitting/pipes/cats/matched_cats/'

f = open(f"../sphinx_data/all_spec_z{z}.json")
dat = json.load(f)

#df_sphinx = pd.read_csv("sphinx_data/all_basic_data.csv")
#df_z_sphinx = df_sphinx[df_sphinx['redshift']==redshift]

output_fits = fits.open(my_cat_path+'corrected_z_'+str(z)+'_'+my_path+'_Salim_matched_with_basic_data.fits')
output_fits = output_fits[1].data

#for i in range(0,len(output_fits['halo_id'])):
for i in range(316,317):
#for i in range(126,127):

    fit = get_fit_object(i,'corrected_z_'+str(z)+'_'+my_path+'_Salim')

    if my_path == 'exponential':
        my_annotation = 'Exponential'
    if my_path == 'delayed':
        my_annotation = 'Delayed exponential'
    elif my_path == 'burstycont':
        my_annotation = 'Bursty Continuity'
    elif my_path == 'cont':
        my_annotation = 'Continuity'
    elif my_path == 'singleburst':
        my_annotation = 'Single Burst'
    elif my_path == 'dbpl':
        my_annotation = 'Double Power Law'


    my_los = output_fits['lines_of_sight'][i]
    halo_id = output_fits['halo_id'][i]
    print ("halo_id HERE ",halo_id)

    lamb_ang_orig = np.array(dat[str(halo_id)]["wavelengths"])*1e4
    lamb_m = np.array(dat[str(halo_id)]["wavelengths"])*1e-6

    #spectrum_atten = 10.**np.array(dat[str(halo_id)][f"dir_{my_los}"]["total"])* 1e23 * 1e6 * lamb_ang_orig * lamb_m*1./(3*1e8)*1e10
    spectrum_atten = 10.**np.array(dat[str(halo_id)][f"dir_{my_los}"]["total"])* 1e23 * 1e6 #*1./(3*1e8)*1e10
    
    #try:
    extract_fit_results(fit, my_los,z,lamb_m,spectrum_atten,create_SED_plotcreate_SED_plot=True)
    #extract_sfh_results(halo_id,fit,my_los,z,lamb_m,spectrum_atten,my_annotation,create_SED_plot=True)
    
    #except:
    #print ("can't plot")
