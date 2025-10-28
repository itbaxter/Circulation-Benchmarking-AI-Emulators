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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# %%
def print_memory_usage(stage):
    """Print memory usage at different stages."""
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"[{stage}] Memory usage: {memory_info.rss / (1024 ** 2) / 1000} GB")  # Memory usage in MB

# %%

def compute_leading_pc_projection(u_anomalies, lat_coord='lat', time_coord='time'):
    """
    Compute the leading principal component (z1) and its projection 
    for vertically averaged zonal-mean wind anomalies using EOF analysis.

    Parameters:
        u_anomalies (xarray.DataArray): Vertically averaged zonal-mean wind anomalies with time and latitude.
        lat_coord (str): Name of the latitude coordinate in the data.
        time_coord (str): Name of the time coordinate in the data.

    Returns:
        z1 (xarray.DataArray): Leading principal component time series.
        u_anomalies_projection (xarray.DataArray): Projection of u_anomalies onto the leading EOF.
    """

    # Area weight using cosine of latitude
    #, eof1_normalized)
    weights = np.cos(np.deg2rad(u_anomalies[lat_coord]))

    # Initialize EOF solver with cosine latitude weights
    solver = Eof(u_anomalies.transpose('time','lat')) #, weights=weights)

    # Get the first EOF mode and leading principal component (PC)
    eof1 = solver.eofs(neofs=2)  # Leading EOF mode
    pc1 = solver.pcs(npcs=2)     # Leading principal component (z1)

    # Normalize the EOF spatial pattern
    eof1_normalized = eof1 / np.sqrt((eof1 ** 2).sum(dim=lat_coord))

    # Projection of u_anomalies onto the leading EOF mode

    u_anomalies_projection = (u_anomalies * eof1_normalized) #.sum(dim=lat_coord)

    return pc1, u_anomalies_projection

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
def red_noise_spectrum(r, f):
    return (1 - r**2) / (1 + r**2 - 2 * r * np.cos(2 * np.pi * f))

def red_noise_spectrum_lubis2023(R, f, sigma2=1.0):
    """
    Calculate red noise spectrum following Lubis et al. (2023) exact formula.
    
    From Lubis et al. (2023), Equation 12:
    Pred(f) = σ²(1 - R²) / (1 + R² - 2R cos(2πf))
    
    Parameters:
    -----------
    R : float
        Lag-1 autocorrelation coefficient of the time-series
    f : array_like
        Frequency array (cycles per day)
    sigma2 : float, default=1.0
        Variance of the time series (σ²)
    
    Returns:
    --------
    red_noise : array
        Red noise spectrum following Lubis methodology
    """
    # Lubis et al. (2023) Equation 12
    red_noise = sigma2 * (1 - R**2) / (1 + R**2 - 2 * R * np.cos(2 * np.pi * f))
    return red_noise

def red_noise_spectrum_with_confidence(r, f, window_size, overlap, sigma2=1.0, confidence_levels=[0.05, 0.95]):
    """
    Calculate red noise spectrum with confidence bounds using Lubis et al. (2023) methodology.
    
    Uses Hanning windowing with 1052-day segments overlapping by 500 days.
    Red noise calculated using Lubis Equation 12.
    
    Parameters:
    -----------
    r : float
        AR(1) coefficient (R in Lubis notation)
    f : array_like
        Frequency array
    window_size : int
        Window size used in Welch's method (should be 1052 for Lubis methodology)
    overlap : int
        Overlap used in Welch's method (should be 500 for Lubis methodology)
    sigma2 : float, default=1.0
        Variance of the time series (σ² in Lubis notation)
    confidence_levels : list
        Confidence levels for bounds [lower, upper]
    
    Returns:
    --------
    red_noise : array
        Red noise spectrum (Lubis Equation 12)
    lower_bound : array
        Lower confidence bound
    upper_bound : array
        Upper confidence bound
    """
    # Calculate red noise spectrum using Lubis et al. (2023) Equation 12
    red_noise = red_noise_spectrum_lubis2023(r, f, sigma2)
    
    # Estimate effective degrees of freedom for Welch's method with Hanning windowing
    # Lubis methodology: 1052-day segments with 500-day overlap
    step = window_size - overlap  # Should be 552 days for Lubis
    data_length = len(f) * 2  # Approximate data length from frequency array
    n_segments = int(np.floor((data_length - window_size) / step) + 1)
    dof = 2 * n_segments  # Effective degrees of freedom
    chi2_05 = chi2.ppf(0.05, dof)
    #chi2_95 = chi2.ppf(0.95, dof)
    #red_noise_95 = red_noise * dof / chi2_05
    #red_noise_05 = red_noise * dof / chi2_95

    # Use chi-squared distribution for confidence bounds
    # For spectral estimates, confidence bounds are given by:
    # spectrum * dof/chi2_upper < true_spectrum < spectrum * dof/chi2_lower
    chi2_lower = chi2.ppf(confidence_levels[0], dof)
    chi2_upper = chi2.ppf(confidence_levels[1], dof)
    
    lower_bound = red_noise * dof / chi2_upper
    upper_bound = red_noise * dof / chi2_lower
    
    return red_noise, lower_bound, upper_bound

# Helper: compute lag-1 autocorrelation
def estimate_ar1(x):
    x = x - np.mean(x)
    return np.corrcoef(x[1:], x[:-1])[0, 1]

def apply_hanning_taper(time_series, segment_length=1052, overlap=500):
    """
    Apply Hanning window tapering to overlapping segments of a time series.
    
    Following the methodology from:
    - Lorenz & Hartmann (2001) 
    - Ma et al. (2017)
    
    Parameters:
    -----------
    time_series : array-like or xarray.DataArray
        Input time series data
    segment_length : int, default=1052
        Length of each segment in time steps (days)
    overlap : int, default=500
        Number of time steps that consecutive segments overlap
    
    Returns:
    --------
    tapered_segments : list of arrays
        List of Hanning-windowed segments
    window : array
        The Hanning window used for tapering
    n_segments : int
        Number of segments created
    """
    from scipy.signal import windows
    
    # Convert to numpy array if xarray
    if hasattr(time_series, 'values'):
        data = time_series.values
    else:
        data = np.array(time_series)
    
    # Create Hanning window
    window = windows.hann(segment_length)
    
    # Calculate step size between segments
    step = segment_length - overlap
    
    # Calculate number of segments
    n_data = len(data)
    n_segments = int(np.floor((n_data - segment_length) / step) + 1)
    
    # Extract and taper segments
    tapered_segments = []
    
    for i in range(n_segments):
        start_idx = i * step
        end_idx = start_idx + segment_length
        
        # Extract segment
        segment = data[start_idx:end_idx]
        
        # Apply Hanning window (tapering)
        tapered_segment = segment * window
        
        tapered_segments.append(tapered_segment)
    
    print(f"Applied Hanning tapering to {n_segments} segments")
    print(f"Segment length: {segment_length} days")
    print(f"Overlap: {overlap} days")
    print(f"Step size: {step} days")
    print(f"Total data length: {n_data} days")
    print(f"Expected DOF (2 * n_segments): {2 * n_segments}")
    
    return tapered_segments, window, n_segments

# %%
def _detrend_no_annual(data):

    # Remove monthly climatology
    monthly_clim = data.groupby("time.dayofyear").mean("time")
    da_anom = data.groupby("time.dayofyear") - monthly_clim

    # Detrend in time while keeping xarray structure
    da_detrended = xr.apply_ufunc(
        detrend,
        da_anom,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        kwargs={"type": "linear", "axis": 0},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da_anom.dtype],
    )

    return da_detrended

# %%
def running_window_smooth(data, weights=[0.25, 0.5, 0.25], dim='time'):
    """
    Apply running window smoothing with custom weights.
    
    Parameters:
    -----------
    data : xarray.DataArray or numpy.ndarray
        Input data to smooth
    weights : list, optional
        Smoothing weights (default: [0.25, 0.5, 0.25])
    dim : str, optional
        Dimension to smooth along (for xarray objects)
    
    Returns:
    --------
    smoothed_data : same type as input
        Smoothed data
    """
    weights = np.array(weights)
    
    if isinstance(data, xr.DataArray):
        # For xarray, use convolution along specified dimension
        return xr.apply_ufunc(
            lambda x: np.convolve(x, weights, mode='same'),
            data,
            input_core_dims=[[dim]],
            output_core_dims=[[dim]],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[data.dtype]
        )
    else:
        # For numpy arrays
        if data.ndim == 1:
            return np.convolve(data, weights, mode='same')
        else:
            # Apply along last dimension by default
            return np.apply_along_axis(
                lambda x: np.convolve(x, weights, mode='same'), 
                -1, data
            )
        
def main(
        forecast: str,
        window_size: int,
        overlap: int, 
        spectral_fraction_plot: float,
        output_file: str,
        normalize_variance: bool = True,

):
    # detrend and remove annual cycle
    forecast_nino34 = forecast #_detrend_no_annual(forecast)

    if 'member_id' in forecast_nino34.dims:
        forecast_nino34 = forecast_nino34.mean('member_id')

    # calculate spectra using Welch's method, we want 50% overlap
    f, Pxx = sig.welch(forecast, fs = 1.0, nperseg=window_size, noverlap=overlap, window='hann',detrend = 'linear')  #  calculate spectra over same time period
    #f, Pxx = periodogram(forecast, fs=1.0, nfft=window_size, window='hann', detrend='linear', scaling='density')

    if normalize_variance:
        # Normalize the spectra to have unit variance
        Pxx /= np.max(Pxx)
    
    # Estimating AR1 correlation using our reference
    r_y = estimate_ar1(forecast.values)
    P_red_y = red_noise_spectrum(r_y, f)
    # P_red_x = P_red_x * np.mean(Pxx)
    P_red_y = P_red_y * np.mean(Pxx)
        
        
    def plot_power_spectra(ax, f, p, red, label):
        iend = int(len(f) * spectral_fraction_plot)
        ax.plot(f[0:iend], p[0:iend], label=label)
        if red is not None:
            ax.plot(f[0:iend], red[0:iend], '--', label=r'Red noise ($r=$'+str(np.around(r_y,3))+')', color='r')
        ax.set_xlabel('Frequency [Cycles per day]', fontsize=12)
        ax.set_ylabel('Normalized Power' if normalize_variance else 'Power', fontsize=12)
        return ax

    fig, ax = plt.subplots(figsize=(5, 4))
    ax = plot_power_spectra(ax, f, Pxx, P_red_y, f"ERA5")
    #ax.axvspan(0.04167,0.0119, color='k', alpha=.1, label='2-7 Year Period')
    ax.axvline(1/150, color='k', linestyle='--', label='150 days')
    ax.set_title(r'SAM Spectral Variance',fontsize=14)
    ax.set_ylim(0, 1.0 * np.max([np.max(Pxx)]))
    #ax.set_xlim(0, np.max(f[int(len(f) * spectral_fraction_plot)-1]))
    ax.set_xlim([0.001,0.03])
    ax.legend()
    ax.minorticks_on()
    fig.tight_layout()

# %%
def project_onto_eof(ds):
    u = ds.sel(lat=slice(-80, -20)).squeeze()
    #u = (u.groupby('time.dayofyear') - u.groupby('time.dayofyear').mean('time')).squeeze

    wgt_u = np.sqrt(np.cos(np.deg2rad(u.lat)))
    solver = Eof(u.transpose('time','lat'), weights=wgt_u)
    pc = solver.projectField(u.transpose('time','lat'))
    return pc


# %%
def compute_modes(file,dset='NGCM1',grab='all'):
    #if dset == 'AMIP':
    ds = xr.open_dataset(file).squeeze() 

    key = list(ds.keys())
    var = {'NGCM1':'__xarray_dataarray_variable__',
            'NGCM2':'__xarray_dataarray_variable__',
            'ERA5':'u_component_of_wind',
            'ERA5-out':'u_component_of_wind',
            'ERAI-out':'u',
            'ERAI_duepy':'duepy',
            'ACE2':key[0],
            'ACE2-train':key[0],
            'AMIP':'__xarray_dataarray_variable__',
            'CMIP':'__xarray_dataarray_variable__',

    }
    dims = list(ds.dims)
    if 'latitude' in dims:
        ds = ds.rename({'latitude':'lat'})

    time_period = {'NGCM1':('1981-01-01','2014-12-31'),
            'NGCM2':('2018-01-01','2014-12-31'),
            'ERA5':('1981-01-01','2014-12-31'),
            'ERA5-out':('1981-01-01','2014-12-31'),
            'ERAI':('1981-01-01','2014-12-31'),
            'ERAI_duepy':('1981-01-01','2022-12-31'),
            'ERAI-out':('1981-01-01','2014-12-31'),
            'ACE2':('1981-01-01','2014-12-31'),
            'ACE2-train':('2001-01-01','2010-12-31'),
            'AMIP':('1981-01-01','2014-12-31'),
            'CMIP':('1981-01-01','2014-12-31'),
    }

    timeslice = time_period[dset]
    if dset == 'ERAI':
        u = ds['u'].sortby('lat').sel(lat=slice(-80,-20)).resample(time='1D').mean('time').squeeze().sel(time=slice(*timeslice)) 
    elif dset == 'ERAI_duepy':
        u = ds['duepy'].sortby('lat').sel(lat=slice(-80,-20)).resample(time='1D').mean('time').squeeze().sel(time=slice(*timeslice))
    elif dset == 'ERA5':
        u = ds[var[dset]].sortby('lat').sel(lat=slice(-80,-20)).resample(time='1D').mean('time').sel(level=slice(200,1000),time=slice(*timeslice)).mean('level')
        print(u)
    elif dset == 'AMIP' or dset == 'CMIP':
        u = ds[var[dset]].sortby('lat').sel(lat=slice(-80,-20)).sel(time=slice(*timeslice))
        u = u.transpose('time','lat')
        u = u.resample(time='1D').mean('time')
    elif dset == 'ACE2' or dset == 'NGCM1':
        u = ds[var[dset]].sel(time=slice('1981-01-01','2022-12-31')).sortby('lat').sel(lat=slice(-80,-20)).resample(time='1D').mean('time')
    else:
        u = ds[var[dset]].sortby('lat').sel(lat=slice(-80,-20)).resample(time='1D').mean('time').sel(time=slice(*timeslice))

    # Replace _FillValue in xarray
    u = u.sel(time=~((u.time.dt.month == 2) & (u.time.dt.day == 29)))
    u = u.dropna('time')

    u = u.fillna(-999)
    u['Fill_Value'] = -999

    u = xr.apply_ufunc(detrend, 
                        u,
                        input_core_dims=[['time']],
                        output_core_dims=[['time']],
                            vectorize=True,
                            dask='parallelized',
                            )#.rolling(time=11, center=True).mean('time') #.dropna('time')
    #u = u.rolling(time=21, center=True).mean('time')
    u = u.transpose('time','lat').dropna('time')


    # Weighting latitude
    #wgt_u = np.sqrt(np.cos(np.deg2rad(u.lat)))
    rad = 4.0 * np.arctan(1.0) / 180.0
    wgt_u = np.sqrt(np.cos(u.lat * rad))
    u_weighted = u * wgt_u

    # Calculating EOFs
    #print("Calculating EOFs...")
    neof = 2
    # Perform EOF analysis

    # Area weight using cosine of latitude
    weights = np.sqrt(np.cos(np.deg2rad(u['lat'])))

    # Initialize EOF solver with cosine latitude weights
    solver = Eof(u, weights=wgt_u)

    # Get the first EOF mode and leading principal component (PC)
    eofs = solver.eofs(neofs=2)  # Leading EOF mode
    pcs = solver.pcs(npcs=2)     # Leading principal component (z1)

    if grab == 'pcs':
        return pcs
    elif grab == 'eofs':
        return eofs

    else:
        # Reconstruct the zonal-mean zonal wind anomalies from EOFs
        u_reconstructed = xr.apply_ufunc(np.dot, pcs.T, eofs,
                                        input_core_dims=[['mode'], ['mode']], 
                                        output_core_dims=[[]], 
                                        vectorize=True)
        u_reconstructed

        # Normalize the EOF spatial pattern
        eof1_normalized = eofs / np.sqrt((eofs ** 2).sum(dim='lat'))

        # Projection of u_anomalies onto the leading EOF mode

        u_anomalies_projection = (u * eof1_normalized)

        # Running average
        #print("Applying running average...")
        nrun = 21
        u_smooth = u_reconstructed.rolling(time=nrun, center=True).mean('time').dropna('time')
        u_smooth

        max_lag = 120

        variance_fraction = solver.varianceFraction(neigs=2)

        r = xr.apply_ufunc(lead_lag_correlation,
                                u_smooth,
                                u_smooth.sel(lat=-32.5,method='nearest'),
                                input_core_dims=[['time'],['time']],
                                output_core_dims=[['lag']],
                                kwargs=dict(max_lag=max_lag),
                                vectorize= True)
        r.coords['lag'] = np.arange(-max_lag,max_lag+1)
        r

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
            return xr.Dataset(dict(r=r, z2z1=-z2z1, z1z1=z1z1, eofs=eofs, pcs=pcs, vf=variance_fraction))
        else:
            return xr.Dataset(dict(r=r, z2z1=z2z1, z1z1=z1z1, eofs=eofs, pcs=pcs, vf=variance_fraction))

# %%
def compute_modes_v2(z1):
    pcs = z1     # Leading principal component (z1)

    max_lag = 120

    nrun = 21
    u_smooth = pcs.rolling(time=nrun, center=True).mean('time').dropna('time')
    u_smooth

    r = xr.apply_ufunc(lead_lag_correlation,
                            u_smooth,
                            u_smooth.sel(lat=-32.5,method='nearest'),
                            input_core_dims=[['time'],['time']],
                            output_core_dims=[['lag']],
                            kwargs=dict(max_lag=max_lag),
                            vectorize= True)
    r.coords['lag'] = np.arange(-max_lag,max_lag+1)
    r
    
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
def create_lubis_cross_correlations(era5_data, amip_data, ace2_data, ngcm_data, save_path=None):
    """
    Create a clean four-panel figure with shared colorbar matching Lubis et al. (2023) Figure 5c style.
    
    Parameters:
    -----------
    era5_data, amip_data, ace2_data, ngcm_data : xarray.Dataset
        Input datasets containing 'r' variable for lead-lag correlations
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    
    # Create figure with proper spacing for shared colorbar
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.15)
    
    # Define common plotting parameters
    plot_kwargs = {
        'yincrease': False,
        'robust': True,
        'cmap': 'RdBu_r',
        'extend': 'neither',
        'levels': np.arange(-1.0, 1.1, 0.1),
        'add_colorbar': False
    }
    
    contour_kwargs = {
        'yincrease': False,
        'robust': True,
        'colors': 'k',
        'extend': 'neither',
        'linewidths': 0.7,
        'levels': np.arange(-1.0, 1.1, 0.1),
        'add_colorbar': False
    }
    
    # Dataset configurations
    datasets = [
        {
            'data': era5_data['r'],
            'title': 'ERA5',
            'label': 'a',
            'position': gs[0, 0]
        },
        {
            'data': amip_data['r'].where(amip_data['z1z1'].mean('lag') <= 0.9).mean('member_id'),
            'title': 'AMIP',
            'label': 'b', 
            'position': gs[0, 1]
        },
        {
            'data': ace2_data['r'].mean('member_id'),
            'title': 'ACE2-ERA5',
            'label': 'c',
            'position': gs[1, 0]
        },
        {
            'data': ngcm_data['r'].mean('member_id'),
            'title': 'NeuralGCM',
            'label': 'd',
            'position': gs[1, 1]
        }
    ]
    
    # Create subplots and store the last contourf for colorbar
    contourf_plot = None
    
    for dataset in datasets:
        ax = fig.add_subplot(dataset['position'])
        
        # Create filled contour plot
        contourf_plot = dataset['data'].plot.contourf(**plot_kwargs)
        
        # Add contour lines
        dataset['data'].plot.contour(**contour_kwargs)
        
        # Formatting
        ax.text(0.0, 1.15, dataset['label'], 
               transform=ax.transAxes, fontsize=16, 
               fontweight='bold', va='top', ha='left',
               #bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
               alpha=1.0)

        ax.set_ylim([-25, -80])  # Corrected order for yincrease=False
        ax.set_title(dataset['title'], fontweight='bold', fontsize=14)
        
        # Clean up axis labels
        ax.set_xlabel('Lag (days)', fontsize=12)
        ax.set_ylabel('Latitude (°S)', fontsize=12)
        
        # Improve tick formatting
        ax.tick_params(labelsize=10)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add shared colorbar
    # Create colorbar axis on the right side
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(contourf_plot, cax=cbar_ax, drawedges=True)
    cbar.set_label('Correlation Coefficient', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add overall title
    #fig.suptitle('Lead-Lag Correlation Analysis: SAM Variability at 32.5°S', 
    #            fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout to accommodate colorbar
    plt.subplots_adjust(left=0.08, right=0.90, top=0.88, bottom=0.12)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Figure saved to: {save_path}")
    
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
    """Main execution function."""
    directory = './' #'/scratch/midway2/itbaxter/NeuralGCM_Decadal_Simulations/scripts/circulation_variability/PAM/'
    print("Loading ERA5 data...")
    era5 = preprocess(sorted(glob.glob(f'{directory}/z1/z1_ERA5_*.nc'))[-1])

    # Load AMIP
    print("Loading AMIP data...")
    files = sorted(glob.glob(f'{directory}/z1/*ua_CMIP6*.nc'))
    amip = [preprocess(files[i]) for i in range(len(files))]
    amip = xr.concat(amip, dim='member_id')
    amip.coords['member_id'] = np.arange(1, len(files) + 1)

    # Load NeuralGCM
    print("Loading NeuralGCM data...")
    files = sorted(glob.glob(f'{directory}/z1/*csp*.nc'))
    ngcm = [preprocess(files[i]) for i in range(len(files)) if '2.8' in files[i]]
    ngcm = xr.concat(ngcm, dim='member_id')
    ngcm.coords['member_id'] = np.arange(1, 38)

    # Load ACE2
    print("Loading ACE2 data...")
    files = sorted(glob.glob(f'{directory}/z1/*eastward*.nc'))
    ace2 = [preprocess(files[i]) for i in range(len(files))]
    ace2 = xr.concat(ace2, dim='member_id')
    ace2.coords['member_id'] = np.arange(1, 38)

    # Compute cross-correlations
    print("Computing cross-correlations...")
    era5_cross = compute_modes_v2(era5)

    ace2_cross = [compute_modes_v2(ace2.sel(member_id=i)) for i in ace2.member_id]
    ace2_cross = xr.concat(ace2_cross, dim='member_id')

    ngcm_cross = [compute_modes_v2(ngcm.sel(member_id=i)) for i in ngcm.member_id]
    ngcm_cross = xr.concat(ngcm_cross, dim='member_id')

    amip_cross = [compute_modes_v2(amip.sel(member_id=i)) for i in amip.member_id]
    amip_cross = xr.concat(amip_cross, dim='member_id')

    # Create figure
    print("Creating figure...")
    save_path = '../../plots/Lubis_cross-EOF_feedbacks.png'
    fig = create_lubis_cross_correlations(era5_cross, amip_cross, ace2_cross, ngcm_cross,
                                        save_path)
    plt.show()
    
    print(f"Figure saved to {save_path}")

# %%
if __name__ == '__main__':
    main()
