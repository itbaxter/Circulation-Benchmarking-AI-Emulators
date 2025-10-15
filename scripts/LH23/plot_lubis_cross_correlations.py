# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from eofs.xarray import Eof
from scipy.signal import periodogram, welch
import psutil  # For memory profiling
#from eofs.xarray import Eof
import glob as glob
from scipy.signal import periodogram
from scipy.interpolate import interp1d
from scipy.signal import correlate
from matplotlib.gridspec import GridSpec
from eofs.xarray import Eof
from scipy.signal import detrend
import scipy.signal as sig
from scipy.stats import chi2
import pandas as pd

import gc

# %%
def print_memory_usage(stage):
    """Print memory usage at different stages."""
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"[{stage}] Memory usage: {memory_info.rss / (1024 ** 2) / 1000} GB")  # Memory usage in MB

# %%
from scipy import stats
from scipy.linalg import eigh 

def lead_lag_correlation(data1, data2, max_lag=None):
    """
    Compute lead-lag correlation between two xarray DataArrays or numpy arrays along time axis.

    Parameters:
    data1, data2 : xarray.DataArray or numpy.ndarray
        Time series data with the time dimension on axis 0.
    max_lag : int, optional
        Maximum number of lags to compute. If None, will compute all possible lags.

    Returns:
    lags : numpy.ndarray
        Array of lag values.
    correlations : numpy.ndarray
        Cross-correlation values normalized by the product of the standard deviations.
    """
    # Ensure data are numpy arrays
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    # Compute length of time axis
    n_time = data1.shape[0]

    # Compute the cross-correlation
    corr = correlate(data1 - np.mean(data1), data2 - np.mean(data2), mode='full')

    # Normalize by the standard deviations and the number of points
    corr /= (np.std(data1) * np.std(data2) * n_time)

    # Create an array of lag values
    lags = np.arange(-n_time + 1, n_time)

    # Optionally limit the lags to max_lag
    if max_lag is not None:
        lag_mask = np.abs(lags) <= max_lag
        lags = lags[lag_mask]
        corr = corr[lag_mask]

    return corr

# %%
def compute_modes_v2(z1):
    pcs = z1     # Leading principal component (z1)

    max_lag = 120

    z2z1 = xr.apply_ufunc(lead_lag_correlation,
                            pcs.sel(mode=1),
                            pcs.sel(mode=0),
                            input_core_dims=[['time'],['time']],
                            output_core_dims=[['lag']],
                            kwargs=dict(max_lag=max_lag),
                            vectorize= True)
    z2z1.coords['lag'] = np.arange(-max_lag,max_lag+1)

    z1z1 = xr.apply_ufunc(lead_lag_correlation,
                            pcs.sel(mode=0),
                            pcs.sel(mode=0),
                            input_core_dims=[['time'],['time']],
                            output_core_dims=[['lag']],
                            kwargs=dict(max_lag=max_lag),
                            vectorize= True)
    z1z1.coords['lag'] = np.arange(-max_lag,max_lag+1) 

    if z2z1.sel(lag=-30) > 0:
        return xr.Dataset(dict(z2z1=-z2z1, z1z1=z1z1, pcs=pcs))
    else:
        return xr.Dataset(dict(z2z1=z2z1, z1z1=z1z1, pcs=pcs))

# %%
def create_lubis_figure_5c(era5_data, amip_data, ace2_data, ngcm_data, save_path=None):
    fig = plt.figure(figsize=(7.5,9))
    gs = GridSpec(3,2,figure=fig,hspace=0.3, wspace=0.25,top=0.94,bottom=0.05, left=0.08, right=0.98)
    ax = fig.add_subplot(gs[2,1])
    lag = 120
    for i in range(37):
        #if ngcm1['z2z1'].sel(lag=-30).isel(member_id=i) > 0:
            #(-ngcm1['z2z1'].sel(lag=slice(-40,40)).isel(member_id=i)).plot.line(x='lag',c='silver',add_legend=False)
            #(-ngcm2['z2z1'].sel(lag=slice(-40,40)).isel(member_id=i)).plot.line(x='lag',c='silver',add_legend=False)
        #else:
            ngcm_cross['z2z1'].sel(lag=slice(-lag,lag)).isel(member_id=i).plot.line(x='lag',c='silver',add_legend=False)
            #ngcm2['z2z1'].sel(lag=slice(-lag,lag)).isel(member_id=i).plot.line(x='lag',c='silver',add_legend=False)

    era5_cross['z2z1'].sel(lag=slice(-lag,lag)).plot(c='k',linewidth=2.5,label='ERA5')

    ngcm_cross['z2z1'].sel(lag=slice(-lag,lag)).mean('member_id').plot(c='#1E88E5',linestyle='-',linewidth=2.5,label='NeuralGCM')
    #ngcm2['z2z1'].sel(lag=slice(-lag,lag)).mean('member_id').plot(c='#1E88E5',linewidth=2.5,label='NeuralGCM')

    #erai['z2z1'].sel(lag=slice(-lag,lag)).plot(c='k',linewidth=1.5,linestyle='--',label='ERA-Interim')
    ax.set_title(r'NGCM2.8 $\mathrm{z_{2}z_{1}}$')
    ax.set_xlabel('Lag (days)')
    ax.axhline(0,linestyle='--',c='k',linewidth=0.7)
    ax.axvline(0,linestyle='--',c='k',linewidth=0.7)
    ax.set_ylim([-0.25,0.25])
    ax.set_xlim([-40,40])
    ax.text(-0.1, 1.15, 'f', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    ax.minorticks_on()
    ax.text(0.12, 0.8, 'z2 leads', transform=ax.transAxes, fontsize=12, va='top')
    ax.text(0.65, 0.2, 'z1 leads', transform=ax.transAxes, fontsize=12, va='top')

    ax = fig.add_subplot(gs[2,0])
    for i in range(37):
        ngcm_cross['z1z1'].sel(lag=slice(-lag,lag)).isel(member_id=i).plot.line(x='lag',c='silver',add_legend=False)
        #ngcm2['z1z1'].sel(lag=slice(-lag,lag)).isel(member_id=i).plot.line(x='lag',c='silver',linewidth=0.7,add_legend=False)

    #erai['z1z1'].sel(lag=slice(-lag,lag)).plot(c='k',linewidth=1.5,linestyle='--')
    era5_cross['z1z1'].sel(lag=slice(-lag,lag)).plot(c='k',linewidth=2.5,label='ERA5')

    ngcm_cross['z1z1'].sel(lag=slice(-lag,lag)).mean('member_id').plot(c='#1E88E5',linestyle='-',linewidth=2.5,label='NGCM2.8')
    #ngcm2['z1z1'].sel(lag=slice(-lag,lag)).mean('member_id').plot(c='#1E88E5',linewidth=2.5)

    ax.set_title(r'NGCM2.8 $\mathrm{z_{1}}$ autocorrelation')
    ax.set_xlabel('Lag (days)')
    ax.axhline(0,linestyle='--',c='k',linewidth=0.7)
    ax.axvline(0,linestyle='--',c='k',linewidth=0.7)
    ax.set_xlim([-40,40])
    ax.set_ylim([-0.1,1.0])
    ax.text(-0.1, 1.15, 'e', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    ax.minorticks_on()
    plt.legend(fontsize=10, frameon=False)

    ax = fig.add_subplot(gs[1,1])
    for i in range(len(ace2.member_id)):
        ace2_cross['z2z1'].sel(lag=slice(-lag,lag)).isel(member_id=i).plot.line(x='lag',c='silver',add_legend=False)
        ace2_cross['z2z1'].sel(lag=slice(-lag,lag)).isel(member_id=i).plot.line(x='lag',c='silver',add_legend=False)

    era5_cross['z2z1'].sel(lag=slice(-lag,lag)).plot(c='k',linewidth=2.5,label='ERA5')

    ace2_cross['z2z1'].mean('member_id').sel(lag=slice(-lag,lag)).plot(c='#D81B60',linewidth=1.5,label='ACE2-ERA5')
    #ace2_train['z2z1'].mean('member_id').sel(lag=slice(-lag,lag)).plot(c='tab:green',linewidth=1.5,label='ACE2-ERA5 [2011-2017]')

    #erai['z2z1'].sel(lag=slice(-lag,lag)).plot(c='k',linewidth=1.5,linestyle='--',label='ERA-Interim')
    ax.set_title(r'ACE2-ERA5 $\mathrm{z_{2}z_{1}}$')
    ax.set_xlabel('')
    ax.axhline(0,linestyle='--',c='k',linewidth=0.7)
    ax.axvline(0,linestyle='--',c='k',linewidth=0.7)
    ax.set_ylim([-0.25,0.25])
    ax.set_xlim([-40,40])
    ax.text(-0.05, 1.15, 'd', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    ax.text(0.12, 0.8, 'z2 leads', transform=ax.transAxes, fontsize=12, va='top')
    ax.text(0.65, 0.2, 'z1 leads', transform=ax.transAxes, fontsize=12, va='top')
    ax.minorticks_on()
    ax = fig.add_subplot(gs[1,0])
    for i in range(len(ace2.member_id)):
        ace2['z1z1'].sel(lag=slice(-lag,lag)).isel(member_id=i).plot.line(x='lag',c='silver',add_legend=False)

    era5_cross['z1z1'].sel(lag=slice(-lag,lag)).plot(c='k',linewidth=2.5,label='ERA5')

    ace2_cross['z1z1'].mean('member_id').sel(lag=slice(-lag,lag)).plot(c='#D81B60',linewidth=1.5,label='ACE2-ERA5')
    #ace2_train['z1z1'].mean('member_id').sel(lag=slice(-lag,lag)).plot(c='tab:green',linewidth=1.5,label='ACE2-ERA5 [2011-2017]')

    #erai['z1z1'].sel(lag=slice(-lag,lag)).plot(c='k',linewidth=1.5,linestyle='--',label='ERA-I ')
    ax.set_title(r'ACE2-ERA5 $\mathrm{z_{1}}$ autocorrelation')
    ax.set_xlabel('')
    ax.axhline(0,linestyle='--',c='k',linewidth=0.7)
    ax.axvline(0,linestyle='--',c='k',linewidth=0.7)
    ax.set_xlim([-40,40])
    ax.set_ylim([-0.1,1.0])
    ax.text(-0.1, 1.15, 'c', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    ax.minorticks_on()
    plt.legend(fontsize=10, frameon=False)

    ax = fig.add_subplot(gs[0,1])
    for i in range(len(amip.member_id)):
        amip_cross['z2z1'].sel(lag=slice(-lag,lag)).isel(member_id=i).plot.line(x='lag',c='silver',add_legend=False)

    #cmip['z2z1'].mean('member_id').sel(lag=slice(-lag,lag)).plot(c='tab:purple',linewidth=1.5,label='CMIP')

    amip_cross['z2z1'].mean('member_id').sel(lag=slice(-lag,lag)).plot(c='tab:orange',linewidth=1.5,label='AMIP')
    #ace2_train['z2z1'].mean('member_id').sel(lag=slice(-lag,lag)).plot(c='tab:green',linewidth=1.5,label='ACE2-ERA5 [2011-2017]')

    #erai['z2z1'].sel(lag=slice(-lag,lag)).plot(c='k',linewidth=1.5,linestyle='--',label='ERA-Interim')
    era5_cross['z2z1'].sel(lag=slice(-lag,lag)).plot(c='k',linewidth=2.5,label='ERA5')

    ax.set_title(r'AMIP $\mathrm{z_{2}z_{1}}$')
    ax.set_xlabel('')
    ax.axhline(0,linestyle='--',c='k',linewidth=0.7)
    ax.axvline(0,linestyle='--',c='k',linewidth=0.7)
    ax.set_ylim([-0.25,0.25])
    ax.set_xlim([-40,40])
    ax.text(-0.1, 1.15, 'b', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    ax.text(0.65, 0.2, 'z1 leads', transform=ax.transAxes, fontsize=12, va='top')
    ax.text(0.12, 0.8, 'z2 leads', transform=ax.transAxes, fontsize=12, va='top')
    ax.minorticks_on()

    amip_member_ids = amip.member_id.values

    ax = fig.add_subplot(gs[0,0])
    for i in range(len(amip_member_ids)):
        amip_cross['z1z1'].sel(lag=slice(-lag,lag)).isel(member_id=i).plot.line(x='lag',c='silver',add_legend=False)

    era5_cross['z1z1'].sel(lag=slice(-lag,lag)).plot(c='k',linewidth=2.5,label='ERA5')


    #cmip['z1z1'].sel(lag=slice(-lag,lag)).mean('member_id').plot(c='tab:purple',linewidth=1.5,label='CMIP')

    amip_cross['z1z1'].sel(lag=slice(-lag,lag)).mean('member_id').plot(c='tab:orange',linewidth=1.5,label='AMIP')
    #ace2_train['z1z1'].mean('member_id').sel(lag=slice(-lag,lag)).plot(c='tab:green',linewidth=1.5,label='ACE2-ERA5 [2011-2017]')

    #erai['z1z1'].sel(lag=slice(-lag,lag)).plot(c='k',linewidth=1.5,linestyle='--',label='ERA-I ')
    ax.set_title(r'AMIP $\mathrm{z_{1}}$ autocorrelation')
    ax.set_xlabel('')
    ax.axhline(0,linestyle='--',c='k',linewidth=0.7)
    ax.axvline(0,linestyle='--',c='k',linewidth=0.7)
    ax.set_xlim([-40,40])
    ax.set_ylim([-0.1,1.0])
    #ax.set_ylim([-0.25,0.25])
    ax.text(-0.1, 1.15, 'a', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    ax.minorticks_on()

    plt.legend(fontsize=10, frameon=False)

    plt.tight_layout()
    plt.savefig('../../plots/Lubis_cross-EOF_feedbacks.png',dpi=300)
    
    return fig

def preprocess(file):
        eof_results = xr.open_dataset(file, decode_times=False).squeeze()
        print(eof_results)
        eof_results.coords['time'] = pd.date_range('1981-01-01', periods=len(eof_results.time), freq='D')
        z1 = eof_results['z1'] #.dropna('time')
        z1.coords['mode'] = [0,1]
        z1 = z1.where( z1 > -500, drop=True)
        return z1

def main():
    era5 = preprocess(sorted(glob.glob('./z1/z1_ERA5_*.nc'))[-1])

    files = sorted(glob.glob('./z1/*ua_CMIP6*.nc'))
    amip = [preprocess(files[i]) for idx, i in enumerate(range(len(files))) if 'ua_CMIP6' in files[i]]
    amip = xr.concat(amip, dim='member_id')
    amip.coords['member_id'] = np.arange(1,len(files)+1,step=1)

    files = sorted(glob.glob('./z1/*csp*.nc'))
    ngcm = [preprocess(files[i]) for idx, i in enumerate(range(len(files))) if '2.8' in files[i]]
    ngcm = xr.concat(ngcm, dim='member_id')
    ngcm.coords['member_id'] = np.arange(1,38,step=1)

    files = sorted(glob.glob('./z1/*eastward*.nc'))
    ace2 = [preprocess(files[i]) for idx, i in enumerate(range(len(files))) if 'eastward' in files[i]]
    ace2 = xr.concat(ace2, dim='member_id')
    ace2.coords['member_id'] = np.arange(1,38,step=1)


    era5_cross = compute_modes_v2(era5)
    era5_cross

    ace2_cross = [compute_modes_v2(ace2.sel(member_id=i)) for i in ace2.member_id]
    ace2_cross = xr.concat(ace2_cross, dim='member_id')
    ace2_cross

    ngcm_cross = [compute_modes_v2(ngcm.sel(member_id=i)) for i in ngcm.member_id]
    ngcm_cross = xr.concat(ngcm_cross, dim='member_id')
    ngcm_cross

    amip_cross = [compute_modes_v2(amip.sel(member_id=i)) for i in amip.member_id]
    amip_cross = xr.concat(amip_cross, dim='member_id')
    amip_cross

    

# %%
if __name__ == '__main__':
    main()

