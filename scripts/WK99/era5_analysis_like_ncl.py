# %%
import numpy as np
import xarray as xr
# our local module:
import wavenumber_frequency_functions as wf
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob as glob
import xesmf as xe
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

    #low_freq_mask = np.abs(z2.frequency) < 0.05
    #z2.loc[{'frequency': low_freq_mask}] = np.nan
    z2.loc[{'frequency':0}] = np.nan # get rid of spurious power at \nu = 0
    # Also consider removing very low frequencies (e.g., < 1/60 cpd)
    
    # the background is supposed to be derived from both symmetric & antisymmetric
    background = wf.smooth_wavefreq(z2avg, kern=wf.simple_smooth_kernel(), nsmooth=100, freq_name='frequency')
    background.coords['component'] ='background'
    # separate components
    z2_sym = z2[0,...]
    z2_asy = z2[1,...]
    # normalize
    nspec_sym = z2_sym / background 
    nspec_asy = z2_asy / background
    return nspec_sym, nspec_asy, background


def plot_normalized_symmetric_spectrum(s, ofil=None):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""
    fb = [0, .5]  # frequency bounds for plot
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

    fb = [0, .5]  # frequency bounds for plot
    # get data for dispersion curves:
    swfreq,swwn = wf.genDispersionCurves()
    # swfreq.shape # -->(6, 3, 50)]
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    fig, ax = plt.subplots()
    c = 'darkgray' # COLOR FOR DISPERSION LINES/LABELS
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20,20))
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

class XESMFRegridder:
    def __init__(self, ds_out, method='bilinear', periodic=False, **kwargs):
        self.ds_out = ds_out
        self.method = method
        print('Starting')

    def regrid(self,ds):
        regridder = xe.Regridder(ds, self.ds_out, self.method)
        dr_out = regridder(ds, keep_attrs=True)

        return dr_out

    def add_lat_lon_bounds(self, ds, lat_name='lat', lon_name='lon'):
        """
        Function to add latitude and longitude bounds to a dataset
        for conservative regridding with xESMF.

        Parameters:
        - ds: xarray Dataset or DataArray containing latitude and longitude coordinates
        - lat_name: Name of the latitude coordinate in the dataset (default is 'lat')
        - lon_name: Name of the longitude coordinate in the dataset (default is 'lon')

        Returns:
        - ds: Dataset with added 'lat_bnds' and 'lon_bnds' variables
        """

        # Get latitude and longitude coordinates
        lat = ds[lat_name]
        lon = ds[lon_name]

        # Calculate latitude bounds
        lat_diff = np.diff(lat) / 2.0
        lat_bnds = np.empty((lat.size, 2), dtype=np.float64)
        lat_bnds[:, 0] = lat - np.concatenate(([lat_diff[0]], lat_diff))
        lat_bnds[:, 1] = lat + np.concatenate((lat_diff, [lat_diff[-1]]))

        # Calculate longitude bounds
        lon_diff = np.diff(lon) / 2.0
        lon_bnds = np.empty((lon.size, 2), dtype=np.float64)
        lon_bnds[:, 0] = lon - np.concatenate(([lon_diff[0]], lon_diff))
        lon_bnds[:, 1] = lon + np.concatenate((lon_diff, [lon_diff[-1]]))

        # Add latitude and longitude bounds to dataset
        ds.coords['lat_bnds'] = (('lat', 'bnds'), lat_bnds)
        ds.coords['lon_bnds'] = (('lon', 'bnds'), lon_bnds)

        return ds

#
# LOAD DATA, x = DataArray(time, lat, lon), e.g., daily mean precipitation
#
def get_data(filename, variablename):
    try: 
        ds = xr.open_dataset(filename).drop_vars('expver').drop_vars('number')
    except ValueError:
        ds = xr.open_mfdataset(filename,combine='nested',concat_dim='time',decode_times=False)
    
    return ds[variablename].compute()


# %%
if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    fili = f"{data_dir}/era5/pr/era5_pminuse_rate_full.nc" 
    vari = "avg_tprate"
    #
    # Loading data ... 
    #
    data = get_data(fili, vari)  # returns OLR
    # %%
    #
    # Options ... right now these only go into wk.spacetime_power()
    #
    latBound = (-10,10)  # latitude bounds for analysis
    spd      = 4    # SAMPLES PER DAY
    nDayWin  = 96   # Wheeler-Kiladis [WK] temporal window length (days)
    nDaySkip = 15  # time (days) between temporal windows [segments]
                    # negative means there will be overlapping temporal segments
    twoMonthOverlap = 50
    opt      = {'segsize': nDayWin, 
                'noverlap': twoMonthOverlap, 
                'spd': spd, 
                'latitude_bounds': latBound, 
                'dosymmetries': True, 
                'rmvLowFrq':True}


    seasonal_cycle = data.groupby('time.dayofyear').mean()
    data_detrend = data.groupby('time.dayofyear') - seasonal_cycle

    symComponent, asymComponent, background  = wf_analysis(data_detrend.sel(time=slice('2018-01-01','2023-12-31')),**opt)
    symComponent, asymComponent

    background.to_netcdf('./data/era5_background.nc')

    # %%
    era5_out = xr.concat([symComponent,asymComponent],dim='component')
    era5_out.to_netcdf('./data/era5_wk_Components.nc')

# %%
