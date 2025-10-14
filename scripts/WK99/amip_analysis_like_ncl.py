# %%
import numpy as np
import xarray as xr
# our local module:
import wavenumber_frequency_functions as wf
import matplotlib as mpl
import glob as glob

import matplotlib.pyplot as plt

# %%
def wf_analysis(x, **kwargs):
    """Return normalized spectra of x using standard processing parameters."""
    # Get the "raw" spectral power
    # OPTIONAL kwargs: 
    # segsize, noverlap, spd, latitude_bounds (tuple: (south, north)), dosymmetries, rmvLowFrq

    z2 = wf.spacetime_power(x, **kwargs)
    print(z2)
    z2avg = z2.mean(dim='component')
    z2.loc[{'frequency':0}] = np.nan # get rid of spurious power at \nu = 0
    # the background is supposed to be derived from both symmetric & antisymmetric
    background = wf.smooth_wavefreq(z2avg, kern=wf.simple_smooth_kernel(), nsmooth=50, freq_name='frequency')
    # separate components
    z2_sym = z2[0,...]
    z2_asy = z2[1,...]
    # normalize
    nspec_sym = z2_sym / background 
    nspec_asy = z2_asy / background
    return nspec_sym, nspec_asy, background


def plot_normalized_symmetric_spectrum(s, ofil=None):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""
    fb = [0, .8]  # frequency bounds for plot
    # get data for dispersion curves:
    swfreq,swwn = wf.genDispersionCurves()
    # swfreq.shape # -->(6, 3, 50)
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    fig, ax = plt.subplots()
    c = 'darkgray' # COLOR FOR DISPERSION LINES/LABELS
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20,20))
    z.loc[{'frequency':0}] = np.nan
    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z, levels=np.arange(1.1, 1.71, 0.1), cmap='YlOrRd',  extend='max')
    for ii in range(3,6):
        ax.plot(swk[ii, 0,:], swf[ii,0,:], color=c)
        ax.plot(swk[ii, 1,:], swf[ii,1,:], color=c)
        ax.plot(swk[ii, 2,:], swf[ii,2,:], color=c)
    ax.axvline(0, linestyle='dashed', color='lightgray')
    ax.set_xlim([-15,15])
    ax.set_ylim(fb)    
    ax.set_title("Normalized Symmetric Component")
    fig.colorbar(img)
    if ofil is not None:
        fig.savefig(ofil, bbox_inches='tight', dpi=144)

def plot_normalized_symmetric_spectrum_new(s, ofil=None):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""
    # get data for dispersion curves:
    swfreq,swwn = wf.genDispersionCurves()
    # swfreq.shape # -->(6, 3, 50)
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    fig, ax = plt.subplots()
    c = 'darkgray' # COLOR FOR DISPERSION LINES/LABELS
    z = s.transpose().sel(frequency=slice(0,0.5), wavenumber=slice(-20,20))
    z.loc[{'frequency':0}] = np.nan
    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z)
    for ii in range(3,6):
        ax.plot(swk[ii, 0,:], swf[ii,0,:], color=c)
        ax.plot(swk[ii, 1,:], swf[ii,1,:], color=c)
        ax.plot(swk[ii, 2,:], swf[ii,2,:], color=c)
    ax.axvline(0, linestyle='dashed', color='lightgray')
    ax.set_xlim([-15,15])
    ax.set_ylim([0,0.5])
    fig.colorbar(img)
    if ofil is not None:
        fig.savefig(ofil, bbox_inches='tight', dpi=144)

def plot_normalized_asymmetric_spectrum(s, ofil=None):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""

    fb = [0, .8]  # frequency bounds for plot
    # get data for dispersion curves:
    swfreq,swwn = wf.genDispersionCurves()
    # swfreq.shape # -->(6, 3, 50)
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    fig, ax = plt.subplots()
    c = 'darkgray' # COLOR FOR DISPERSION LINES/LABELS
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-15,15))
    z.loc[{'frequency':0}] = np.nan
    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z, levels=np.arange(1.1, 1.71, 0.1), cmap='YlOrRd', extend='max')
    for ii in range(0,3):
        ax.plot(swk[ii, 0,:], swf[ii,0,:], color=c)
        ax.plot(swk[ii, 1,:], swf[ii,1,:], color=c)
        ax.plot(swk[ii, 2,:], swf[ii,2,:], color=c)
    ax.axvline(0, linestyle='dashed', color='lightgray')
    ax.set_xlim([-15,15])
    ax.set_ylim(fb)
    ax.set_title("Normalized Anti-symmetric Component")
    fig.colorbar(img)
    if ofil is not None:
        fig.savefig(ofil, bbox_inches='tight', dpi=144)

#
# LOAD DATA, x = DataArray(time, lat, lon), e.g., daily mean precipitation
#
def get_data(filename, variablename):
    try: 
        ds = xr.open_mfdataset(filename,combine='nested',concat_dim='time').sel(time=slice('1981-01-01 00:00:00','2014-12-31 18:00:00'))
        print(ds)
    except ValueError:
        ds = xr.open_dataset(filename, decode_times=False)
    
    return ds[variablename]

# %%

if __name__ == "__main__":
    #
    # input file -- could make this a CLI argument
    #
    #fili = "/project2/tas1/itbaxter/NeuralGCM_Decadal_Simulations/data/processed/MJO/ace2_era5_pminuse_rate_mjo_full_1.000000.nc" 
    model = 'CESM2-WACCM'
    fili = sorted(glob.glob(f"/project/tas1/itbaxter/for-tiffany/amip/pr/pr_day_{model}_amip_r1i1p1f1*nc"))
    vari = "pr"
    #
    # Loading data ... example is very simple
    #
    data = get_data(fili, vari)  # returns OLR
    print(data)

    #
    # Options ... right now these only go into wk.spacetime_power()
    #
    latBound = (-15,15)  # latitude bounds for analysis
    spd      = 1    # SAMPLES PER DAY
    nDayWin  = 96   # Wheeler-Kiladis [WK] temporal window length (days)
    nDaySkip = -65  # time (days) between temporal windows [segments]
                    # negative means there will be overlapping temporal segments
    twoMonthOverlap = 65
    opt      = {'segsize': nDayWin, 
                'noverlap': twoMonthOverlap, 
                'spd': spd, 
                'latitude_bounds': latBound, 
                'dosymmetries': True, 
                'rmvLowFrq':False}

    
    #years = np.unique(data['member_id'].values)
    #sym_list = []
    #asym_list = []
    #bg_list = []

    #for yr in years[:]:
    #    data_yr = data.sel(member_id=yr)
    #    if data_yr.time.size > 0:
    #        sym_yr, asym_yr, background_yr = wf_analysis(data_yr, **opt)
    #        sym_list.append(sym_yr)
    #        asym_list.append(asym_yr)
    #        bg_list.append(background_yr)

    sym_yr, asym_yr, background_yr = wf_analysis(data.compute(), **opt)
    print(sym_yr,asym_yr,background_yr)
    # Average over years
    #symComponent = xr.concat(sym_list, dim='member_id') #.mean(dim='member_id')
    #asymComponent = xr.concat(asym_list, dim='member_id') #.mean(dim='member_id')
    #background = xr.concat(bg_list, dim='member_id')

    sym_yr.to_netcdf(f'./amip/{model}_symComponents.nc',mode='w')
    asym_yr.to_netcdf(f'./amip/{model}_asymComponents.nc',mode='w')
    background_yr.to_netcdf(f'./amip/{model}_backgroundComponents.nc',mode='w')
    #ngcm_Component = xr.Dataset(dict(symmetric=sym_yr,asymmetric=asym_yr,background=background_yr))     
    #ngcm_Component.to_netcdf('./amip/amip_Components.nc',mode='w')

    #
    # Plot averaged results
    #
    outPlotName = "amip_pminuse_symmetric_plot.png"
    plot_normalized_symmetric_spectrum(symComponent.mean('member_id'), outPlotName)

    outPlotName = "amip_pminuse_asymmetric_plot.png"
    plot_normalized_asymmetric_spectrum(asymComponent.mean('member_id'), outPlotName)

