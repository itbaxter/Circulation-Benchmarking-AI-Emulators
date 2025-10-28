# %%
import numpy as np
import xarray as xr
# our local module:
import wavenumber_frequency_functions as wf
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

# %%
def parse_args():
    parser = argparse.ArgumentParser(description='Eddy co-spectra (RH91) analysis')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Base data directory (expects era5/, ace2/, ngcm/, amip/ subfolders). If None, uses script-relative ../../..-data')
    return parser.parse_args()

# %%
def wf_analysis(x, **kwargs):
    """Return normalized spectra of x using standard processing parameters."""
    # Get the "raw" spectral power
    # OPTIONAL kwargs: 
    # segsize, noverlap, spd, latitude_bounds (tuple: (south, north)), dosymmetries, rmvLowFrq

    z2 = wf.spacetime_power(x, **kwargs)
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
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-15,15))
    z.loc[{'frequency':0}] = np.nan
    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z, levels=np.linspace(0.2, 3.0, 16), cmap='Spectral_r',  extend='both')
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
    img = ax.contourf(kmesh0, vmesh0, z, levels=np.linspace(0.2, 1.8, 16), cmap='Spectral_r', extend='both')
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
        ds = xr.open_dataset(filename)
        print(ds)
    except ValueError:
        ds = xr.open_dataset(filename, decode_times=False)
    
    return ds[variablename]

if __name__ == "__main__":
    #
    # Input file and variable
    #
    args = parse_args()
    for i in range(37):
        fili = f"{args.data_dir}/ace2/pr/ace2_era5_37full_pminuse_rate_mjo_{i:03.0f}.nc" 
        vari = "PRATEsfc"

        # Load data
        data = get_data(fili, vari)  # returns DataArray(time, lat, lon)
        #
        # Options for wf_analysis
        #
        latBound = (-15, 15)   # latitude bounds
        spd = 1               # samples per day
        nDayWin = 96          # temporal window length (days)
        twoMonthOverlap = -65  # overlap size

        opt = {
            'segsize': nDayWin,
            'noverlap': twoMonthOverlap,
            'spd': spd,
            'latitude_bounds': latBound,
            'dosymmetries': True,
            'rmvLowFrq': False,
        }

        #
        # Perform wf_analysis per year
        #
        years = np.unique(data['time.year'].values)

        data_yr = data.sel(time=slice('2001-01-01','2010-12-31'))
        sym_list, asym_list, bg_list = wf_analysis(data_yr, **opt)

        # Average over years
        symComponent = sym_list 
        asymComponent = asym_list 
        background = bg_list 

        symComponent.to_dataset(name='symComponent').to_netcdf(f'./data/ace2_37full_{i:03.0f}_symComponent.nc')
        asymComponent.to_dataset(name='asymComponent').to_netcdf(f'./data/ace2_37full_{i:03.0f}_asymComponent.nc')
        background.to_dataset(name='background').to_netcdf(f'./data/ace2_37full_{i:03.0f}_background.nc')

        #
        # Plot averaged results
        #
        #outPlotName = f"ace2_pratesfc_{i:03.0f}_symmetric_plot.png"
        #plot_normalized_symmetric_spectrum(symComponent, outPlotName)

        #outPlotName = f"ace2_pratesfc_{i:03.0f}_asymmetric_plot.png"
        #plot_normalized_asymmetric_spectrum(asymComponent, outPlotName)

# %%
