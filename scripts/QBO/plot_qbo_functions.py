# %%
"""
QBO (Quasi-Biennial Oscillation) Analysis and Visualization

This script analyzes the QBO in equatorial zonal wind data from:
- ERA5 reanalysis
- AMIP6 models
- ACE2-ERA5 emulator
- NGCM2.8 emulator

Produces:
- Power spectrum analysis comparing QBO periodicity across datasets
- Time series comparisons of equatorial zonal winds
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from scipy.stats import linregress as _linregress
from scipy.signal import welch, find_peaks
import glob
import warnings
import argparse
import os
import sys
import gc

warnings.filterwarnings('ignore')

# Set matplotlib parameters
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 16

# %%
# ============================================================================
# Command Line Arguments
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='QBO Analysis and Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python plot_qbo_clean.py --data_dir /path/to/data
    python plot_qbo_clean.py --data_dir /path/to/data --output_dir /path/to/plots
        """
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/project/tas1/itbaxter/aimip/benchmarking_ai_variability/data',
        help='Base directory containing input data (default: /project/tas1/itbaxter/aimip/benchmarking_ai_variability/data)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../../plots',
        help='Directory to save output plots (default: ../../plots)'
    )
    
    return parser.parse_args()


# ============================================================================
# Utility Functions
# ============================================================================

def area_weighted_ave(ds):
    """Calculate area-weighted average accounting for latitude."""
    if 'lat' not in ds.dims:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    coslat = np.cos(np.deg2rad(ds.lat))
    ds, coslat = xr.broadcast(ds, coslat)
    ds = ds * coslat
    return ds.sum(('lat', 'lon'), skipna=True) / ((ds / ds) * coslat).sum(('lat', 'lon'), skipna=True)


def linregress(da_y, da_x, dim=None):
    """xarray-wrapped function of scipy.stats.linregress."""
    if dim is None:
        dim = [d for d in da_y.dims if d in da_x.dims][0]

    slope, intercept, r, p, stderr = xr.apply_ufunc(
        _linregress, da_x, da_y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        dask='allowed'
    )
    predicted = da_x * slope + intercept

    slope.attrs['long_name'] = 'slope of the linear regression'
    intercept.attrs['long_name'] = 'intercept of the linear regression'
    r.attrs['long_name'] = 'correlation coefficient'
    p.attrs['long_name'] = 'p-value'
    stderr.attrs['long_name'] = 'standard error of the estimated gradient'
    predicted.attrs['long_name'] = 'predicted values by the linear regression model'

    return xr.Dataset(dict(slope=slope, intercept=intercept, r=r, p=p, 
                          stderr=stderr, predicted=predicted))


def add_curved_arrow_label(ax, start_xy, end_xy, text, arrow_color='black', 
                           text_color='black', connectionstyle="arc3,rad=0.3"):
    """Add a curved arrow with label to plot."""
    ax.annotate(text, xy=end_xy, xytext=start_xy,
                arrowprops=dict(arrowstyle='->',
                              connectionstyle=connectionstyle,
                              color=arrow_color,
                              lw=1.5),
                fontsize=10,
                color=text_color,
                ha='center', va='center')


def stack_loop(dataarrays):
    """Stack multiple dataarrays with proper dimension handling."""
    stacked_dataarrays = []
    for data in dataarrays:
        # Drop problematic coordinates
        for coord in ['plev_bnds', 'lev_bnds', 'bnds']:
            if coord in data.coords:
                data = data.drop(coord)
            
        if 'member_id' not in data.dims:
            data = data.expand_dims('member_id')
            data['member_id'] = ('member_id', [data.attrs.get('member_id', 'r1i1p1f1')])
        
        if 'height' not in data.dims:
            data = data.expand_dims('height')
            data['height'] = ('height', [data.attrs.get('height', 2)])

        stacked_dataarrays.append(data)
    
    return stacked_dataarrays


class READER:
    """Class to read and process multiple CMIP data files."""
    
    def __init__(self, files):
        self.files = files

    def reader(self, f):
        """Read individual file and extract metadata."""
        source_id = f.split('/')[-1].split('_')[2]
        start = f.split('/')[-1].split('_')[5].split('-')[0]
        end = int(f.split('/')[-1].split('_')[5].split('-')[1].split('.')[0])
        
        try:
            ds = xr.open_dataset(f)
        except:
            ds = xr.open_dataset(f, decode_times=False)
            ds.coords['time'] = np.arange(f'{start}-01-01', f'{end}-01-01', 
                                         dtype='datetime64[M]')
            
        # Drop problematic coordinates
        for coord in ['plev_bnds', 'lev_bnds', 'bnds']:
            if coord in ds.coords:
                ds = ds.drop(coord)
            
        ds_ann = ds
        ds_ann.coords['source_id'] = source_id
        return ds_ann

    def process(self):
        """Process all files."""
        return [self.reader(f) for f in self.files]


def compute_power_spectrum(u, c='k', plot=False, label_peaks=False, i=0, label='ERA5'):
    """
    Compute power spectrum of zonal wind anomaly.
    
    Parameters:
    -----------
    u : xarray.DataArray
        Zonal wind data
    c : str
        Color for plotting
    plot : bool
        Whether to plot the spectrum
    label_peaks : bool
        Whether to label the peaks
    i : int
        Index for ensemble members
    label : str
        Label for the dataset
    
    Returns:
    --------
    period_months : array
        Period in months
    psd : array
        Power spectral density (normalized)
    peak_periods : array
        Periods of top 2 peaks
    """
    u_anom = u - u.mean(dim='time')
    fs = 1.0  # daily sampling
    freqs, psd = welch(u_anom, fs=fs, nperseg=len(u_anom)//2, detrend='linear')

    psd = psd / np.max(psd)

    # Convert to period in months
    period_months = (1 / freqs)

    # Find valid peaks (between 6 and 100 months)
    peaks, _ = find_peaks(psd, height=np.percentile(psd, 80))
    valid_peaks = [p for p in peaks if 6 <= period_months[p] <= 100]

    # Sort by PSD height (descending), pick top 2
    top2 = sorted(valid_peaks, key=lambda p: psd[p], reverse=True)[:2]

    # Plot if requested
    if plot:
        if i == 0:
            ax.semilogx(period_months, psd, lw=1.5, c=c, label=label)
        else:
            ax.semilogx(period_months, psd, lw=1.5, c=c)

    return period_months[:106], psd[:106], period_months[top2]

# %%
# ============================================================================
# Data Loading Functions
# ============================================================================

def preprocess_ace2(ds):
    """Preprocess ACE2 datasets."""
    if 'init_time' in ds.coords:
        ds = ds.drop('init_time')
    return ds['__xarray_dataarray_variable__'].sel(time=slice('1981-01-01', '2022-12-31')).squeeze()


def load_data(data_dir):
    """
    Load all required datasets.
    
    Parameters:
    -----------
    data_dir : str
        Base directory containing input data
    ace2_dir : str
        Directory containing ACE2 data
    
    Returns:
    --------
    dict : Dictionary containing all loaded datasets
    """
    data = {}
    
    # ERA5 reanalysis
    era5_file = os.path.join(data_dir, 'era5/eq_uwind/era5_uwind_qbo.nc')
    if not os.path.exists(era5_file):
        raise FileNotFoundError(f"ERA5 file not found: {era5_file}")
    data['era5'] = xr.open_dataset(era5_file)['__xarray_dataarray_variable__']
    gc.collect()
    
    # NGCM2.8
    ngcm_file = os.path.join(data_dir, 'ngcm/eq_uwind/ngcm_uwind_qbo.nc')
    if not os.path.exists(ngcm_file):
        raise FileNotFoundError(f"NGCM file not found: {ngcm_file}")
    data['csp'] = xr.open_dataset(ngcm_file)['__xarray_dataarray_variable__']
    gc.collect()
    
    # ACE2-ERA5
    ace2_pattern = os.path.join(data_dir, 'ace2/eq_uwind/ace2_qbo_mon*nc')
    ace2_files = glob.glob(ace2_pattern)
    if not ace2_files:
        raise FileNotFoundError(f"No ACE2 files found matching: {ace2_pattern}")
    
    data['ace2'] = xr.open_mfdataset(
        ace2_pattern,
        combine='nested',
        concat_dim='member_id',
        preprocess=preprocess_ace2
    )
    gc.collect()
    
    # AMIP6 models
    amip_pattern = os.path.join(data_dir, 'amip/eq_uwind/ua50*.nc')
    cmip_files = sorted(glob.glob(amip_pattern))
    if not cmip_files:
        raise FileNotFoundError(f"No AMIP files found matching: {amip_pattern}")
    
    cmip = READER(cmip_files).process()
    data['cmip_concat'] = xr.concat(stack_loop(cmip), dim='member_id').squeeze()
    
    # Extract equatorial mean zonal wind
    data['amip_ua'] = data['cmip_concat']['ua'].sel(lat=slice(-10, 10)).mean(dim=['lat', 'lon']).sel(
        time=slice('1979-01-01', '2014-12-31')).squeeze()
    
    return data

# %%
# ============================================================================
# Plotting Functions
# ============================================================================

def plot_power_spectrum(data, output_dir):
    """
    Create power spectrum analysis plot.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing all datasets (from load_data)
    output_dir : str
        Directory to save output plot
    
    Returns:
    --------
    str : Path to saved figure
    """
    # Make ax global for compute_power_spectrum function
    global ax
    
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], left=0.1, right=0.9, 
                           wspace=0.02, top=0.95)
    
    # Define colors
    colors = ['tab:orange', '#D81B60', '#1E88E5']
    
    # Panel A: Power Spectrum
    ax = plt.subplot(gs[0])
    
    # Define colors
    colors = ['tab:orange', '#D81B60', '#1E88E5']
    
    era5 = data['era5']
    csp = data['csp']
    ace2 = data['ace2']
    amip_ua = data['amip_ua']
    
    # ERA5
    period_months, era5_psd, era5_peaks = compute_power_spectrum(
        era5.resample(time='1MS').mean('time'), c='k', plot=True, 
        label_peaks=True, label='ERA5')
    
    add_curved_arrow_label(ax, (12, 0.9), (27.9, 1.0), f'ERA5: 28 mo', 
                           arrow_color='black', connectionstyle="arc3,rad=0.5")
    
    # AMIP ensemble
    amip_peaks = []
    amip_psd = []
    for i in range(amip_ua.member_id.size):
        period_months, psd, peaks = compute_power_spectrum(
            amip_ua.sel(time=slice('1979-01-01', '2014-12-31')).isel(member_id=i).dropna('time'),
            c='tab:orange', i=i, label='AMIP')
        amip_peaks.append(peaks)
        amip_psd.append(xr.DataArray(psd, dims='period'))
    
    ax.semilogx(period_months, xr.concat(amip_psd, dim='member_id').mean('member_id'), 
               lw=1.5, c='tab:orange', label='AMIP6')
    
    add_curved_arrow_label(ax, (3, 0.45), (6, 0.41), f'AMIP: 6 mo', 
                           arrow_color='tab:orange', connectionstyle="arc3,rad=0.5")
    add_curved_arrow_label(ax, (57, 0.65), (27.9, 0.62), f'AMIP: 27.9 mo', 
                           arrow_color='tab:orange', connectionstyle="arc3,rad=0.5")
    
    # NGCM2.8 ensemble
    csp_peaks = []
    csp_psd = []
    for i in range(csp.member_id.size):
        period_months, psd, peaks = compute_power_spectrum(
            csp.resample(time='1MS').mean('time').isel(member_id=i).dropna('time'),
            c='tab:blue', i=i, label='NGCM2.8')
        csp_peaks.append(peaks)
        csp_psd.append(xr.DataArray(psd, dims='period'))
    
    ax.semilogx(period_months, xr.concat(csp_psd, dim='member_id').mean('member_id'), 
               lw=1.5, c=colors[2], label='NGCM2.8')
    
    add_curved_arrow_label(ax, (3, 0.75), (12, np.max(xr.concat(csp_psd, dim='member_id').mean('member_id'))), 
                           f'NGCM: 12 mo', arrow_color=colors[2], connectionstyle="arc3,rad=0.5")
    
    # ACE2 ensemble
    ace2_peaks = []
    ace2_psd = []
    for i in range(ace2.member_id.size):
        period_months, psd, peaks = compute_power_spectrum(
            ace2.resample(time='1MS').mean('time').isel(member_id=i).dropna('time'),
            c='tab:green', i=i, label='ACE2')
        ace2_peaks.append(peaks)
        ace2_psd.append(xr.DataArray(psd, dims='period'))
    
    ax.semilogx(period_months, xr.concat(ace2_psd, dim='member_id').mean('member_id'), 
               lw=1.5, c=colors[1], label='ACE2-ERA5')
    
    ax.axvline(28, color='k', linestyle='--', linewidth=0.5)
    
    plt.gca().invert_xaxis()
    ax.set_xlabel('Period (months)')
    ax.set_ylabel('Power Spectral Density')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim([100, 0])
    ax.set_xticks([6, 12, 28, 50, 100])
    ax.set_xticklabels([6, 12, 28, 50, 100])
    ax.text(0.03, 0.9, 'a', fontsize=24, fontweight='bold', transform=ax.transAxes)
    
    # Panel B: Period vs Amplitude
    ax = plt.subplot(gs[1])
    ax.minorticks_on()
    ax.set_xscale('log')
    ax.set_xticks([6, 12, 28, 50, 100])
    ax.set_xticklabels([6, 12, 28, 50, 100])
    ax.text(0.03, 0.9, 'b', fontsize=24, fontweight='bold', transform=ax.transAxes)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    
    # Compute amplitudes (remove annual cycle, then get std)
    amip_rmac = amip_ua.groupby('time.month') - amip_ua.groupby('time.month').mean('time')
    amip_std = amip_rmac.std('time')
    
    # Plot scatter points
    ax.scatter(era5_peaks[0], 2**(0.5 * era5.std('time')), edgecolor='k', marker='X', 
              s=200, facecolor='k', label='ERA5')
    
    ax.scatter(np.concatenate(amip_peaks)[::2], 2**(0.5 * amip_std), 
              edgecolor=colors[0], facecolor=colors[0], label='AMIP', lw=1.5)
    
    ax.scatter(np.concatenate(csp_peaks)[::2], 
              2**(0.5 * csp.sel(time=slice('1981-01-01', '2020-12-31')).std('time')),
              edgecolor=colors[2], facecolor=colors[2], label='NGCM2.8', lw=1.5)
    
    # Highlight specific NGCM members
    for idx in [2, 16]:
        ax.scatter(np.concatenate(csp_peaks)[::2][idx], 
                  2**(0.5 * csp.isel(member_id=idx).sel(time=slice('1981-01-01', '2020-12-31')).std('time')),
                  edgecolor='k', facecolor=colors[2], lw=1.5)
    
    ax.scatter(np.concatenate(ace2_peaks)[::2], 
              2**(0.5 * ace2.sel(time=slice('1981-01-01', '2020-12-31')).std('time')),
              edgecolor=colors[1], facecolor=colors[1], label='ACE2-ERA5', lw=1.5)
    
    # Highlight specific ACE2 member
    ax.scatter(np.concatenate(ace2_peaks)[::2][4], 
              2**(0.5 * ace2.isel(member_id=4).sel(time=slice('1981-01-01', '2020-12-31')).std('time')),
              edgecolor='k', facecolor=colors[1], lw=1.5)
    
    # Annotate specific AMIP models
    add_curved_arrow_label(ax, (9, 34), 
                           (26, 1 + 2**(0.5 * amip_std).sel(member_id='CESM2-WACCM-FV2_r1i1p1f1') - 0.5), 
                           f'CESM2-WACCM-FV2 \n (nudged)', arrow_color='tab:orange',
                           connectionstyle="arc3,rad=-0.5")
    
    add_curved_arrow_label(ax, (49, 16), 
                           (28.5, 1 + 2**(0.5 * amip_std).sel(member_id='IPSL-CM6A-LR_r8i1p1f1') - 0), 
                           f'IPSL-CM6A-LR \n (prognostic)', arrow_color='tab:orange',
                           connectionstyle="arc3,rad=0.45")
    
    ax.scatter(27.5, 2**(0.5 * amip_ua.sel(member_id='IPSL-CM6A-LR_r8i1p1f1').std('time')),
              edgecolor='k', facecolor=colors[0], lw=1.5)
    
    add_curved_arrow_label(ax, (28, 44), 
                           (16.5, 2**(0.5 * amip_std).sel(member_id='E3SM-1-0_r2i1p1f1') - 1), 
                           f'E3SM-1-0', arrow_color='tab:orange',
                           connectionstyle="arc3,rad=0.55")
    
    ax.legend(loc='upper right', fontsize=9, frameon=False)
    ax.set_ylabel(r'Amplitude ($\mathrm{2^{0.5\sigma},\ m\ s^{-1}}$)')
    ax.set_xlabel('Dominant Periodicity (months)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim([0, 50])
    
    output_file = os.path.join(output_dir, 'qbo_power_spectrum-month_ensmean.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    return output_file


def plot_time_series(data, output_dir):
    """
    Create time series comparison plot.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing all datasets (from load_data)
    output_dir : str
        Directory to save output plot
    
    Returns:
    --------
    str : Path to saved figure
    """
    era5 = data['era5']
    csp = data['csp']
    ace2 = data['ace2']
    cmip_concat = data['cmip_concat']
    
    fig = plt.figure(figsize=(7.5, 8))
    mpl.rcParams['font.size'] = 14
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], top=0.98, left=0.12, 
                           right=0.98, bottom=0.11, hspace=0.15)
    
    # Panel A: AMIP models
    ax = plt.subplot(gs[0])
    era5.sel(time=slice('1979-01-01', '2023-12-31')).plot.line(
        x='time', c='k', add_legend=False, label='ERA5')
    cmip_concat['ua'].sel(lat=slice(-10, 10)).mean(dim=['lat', 'lon']).sel(
        time=slice('1979-01-01', '2014-12-31')).squeeze().sel(
        member_id='CESM2_r10i1p1f1').sel(time=slice('1979-01-01', '2023-12-31')).plot.line(
        x='time', linestyle='--', c='tab:orange', add_legend=False, label='CESM2')
    cmip_concat['ua'].sel(lat=slice(-10, 10)).mean(dim=['lat', 'lon']).sel(
        time=slice('1979-01-01', '2014-12-31')).squeeze().sel(
        member_id='IPSL-CM6A-LR_r8i1p1f1').sel(time=slice('1979-01-01', '2023-12-31')).plot.line(
        x='time', linestyle='-', c='tab:orange', add_legend=False, label='IPSL-CM6A-LR')
    
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel(' ')
    ax.legend(ncols=3, frameon=False, bbox_to_anchor=(0.9, 0.25))
    ax.set_ylim(-50, 35)
    ax.set_title('')
    ax.text(0.01, 0.9, 'a', fontsize=18, fontweight='bold', transform=ax.transAxes)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xlabel(' ')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.minorticks_on()
    
    # Panel B: ACE2-ERA5
    ax = plt.subplot(gs[1])
    validation = np.arange('1996-01-01', '2001-01-01', dtype='datetime64[M]')
    training = np.arange('2001-01-01', '2010-12-31', dtype='datetime64[M]')
    training2 = np.arange('2019-01-01', '2020-12-31', dtype='datetime64[M]')
    
    ax.fill_between(validation, np.zeros(len(validation)) - 40, 
                   np.zeros(len(validation)) + 25, color='silver', alpha=0.3)
    ax.fill_between(training, np.zeros(len(training)) - 40, 
                   np.zeros(len(training)) + 25, color='silver', alpha=0.5)
    ax.fill_between(training2, np.zeros(len(training2)) - 40, 
                   np.zeros(len(training2)) + 25, color='silver', alpha=0.5)
    
    era5.sel(time=slice('1979-01-01', '2023-12-31')).plot.line(
        x='time', c='k', add_legend=False)
    ace2.sel(time=slice('1979-01-01', '2023-12-31')).sel(member_id=4).plot.line(
        x='time', c='#D81B60', add_legend=False, label='ACE2-ERA5')
    
    ax.text(0.01, 0.9, 'b', fontsize=18, fontweight='bold', transform=ax.transAxes)
    ax.text(0.4, 0.9, 'tuning', fontsize=8, transform=ax.transAxes)
    ax.text(0.5, 0.9, 'testing', fontsize=8, transform=ax.transAxes)
    
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Tropical zonal mean zonal wind speed (m/s)')
    ax.legend(ncols=2, frameon=False, bbox_to_anchor=(0.45, 0.07))
    ax.set_ylim(-50, 35)
    ax.set_title('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.minorticks_on()
    ax.set_xlabel(' ')
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    # Panel C: NGCM2.8
    ax = plt.subplot(gs[2])
    training = np.arange('2018-01-01', '2024-01-01', dtype='datetime64[M]')
    ax.fill_between(training, np.zeros(len(training)) - 50, 
                   np.zeros(len(training)) + 25, color='silver', alpha=0.5)
    
    csp.isel(member_id=16).sel(time=slice('1979-01-01', '2023-12-31')).plot.line(
        x='time', c='#1E88E5', add_legend=False)
    csp.where(csp.isel(time=slice(-365*20, -1)).mean('time') < 0).isel(member_id=2).sel(
        time=slice('1979-01-01', '2023-12-31')).plot.line(
        x='time', c='#1E88E5', add_legend=False, label='NGCM2.8')
    era5.sel(time=slice('1979-01-01', '2023-12-31')).plot.line(
        x='time', c='k', add_legend=False)
    
    ax.text(0.835, 0.9, 'testing', fontsize=8, transform=ax.transAxes)
    ax.text(0.07, 0.89, 'westerly \n member (1 of 8)', fontsize=8, transform=ax.transAxes)
    ax.text(0.005, 0.1, 'easterly \n member \n (1 of 29)', fontsize=8, transform=ax.transAxes)
    
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('')
    ax.legend(ncols=2, bbox_to_anchor=(0.55, 0.95), frameon=False)
    ax.set_ylim(-50, 35)
    ax.set_title('')
    ax.text(0.01, 0.9, 'c', fontsize=18, fontweight='bold', transform=ax.transAxes)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.minorticks_on()
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'qbo_time_series-2member.png')
    plt.savefig(output_file, dpi=500)
    plt.close()
    
    return output_file

# %%
# ============================================================================
# Main Execution
# ============================================================================

def main(data_dir=None, output_dir=None):
    """Main execution function."""
    # Parse command line arguments
    if data_dir is None or output_dir is None:
        args = parse_args()
        data_dir = args.data_dir
        output_dir = args.output_dir

    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(' ')
    
    # Load data
    print("Loading data...")
    try:
        data = load_data(data_dir)
        print("Data loaded successfully.")
        print(' ')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create plots
    print("Creating power spectrum plot...")
    spectrum_file = plot_power_spectrum(data, output_dir)
    print(f"Saved power spectrum plot to: {spectrum_file}")
    print()
    
    print("Creating time series plot...")
    timeseries_file = plot_time_series(data, output_dir)
    print(f"Saved time series plot to: {timeseries_file}")
    print()
    
    print("QBO analysis complete. All figures saved.")


if __name__ == '__main__':
    args = parse_args()
    main(data_dir=args.data_dir, output_dir=args.output_dir)