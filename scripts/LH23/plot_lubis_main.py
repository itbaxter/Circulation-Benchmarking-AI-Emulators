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
        
def plot_main(
        forecast: str,
        window_size: int,
        overlap: int, 
        spectral_fraction_plot: float,
        output_file: str,
        normalize_variance: bool = True,

):
    # detrend and remove annual cycle
    print(forecast)
    forecast_nino34 = forecast #_detrend_no_annual(forecast)

    if 'member_id' in forecast_nino34.dims:
        forecast_nino34 = forecast_nino34.mean('member_id')

    # calculate spectra using Welch's method, we want 50% overlap
    f, Pxx = sig.welch(forecast, fs = 1.0, nperseg=window_size, noverlap=overlap, window='hann',detrend = 'linear')  #  calculate spectra over same time period
    #f, Pxx = periodogram(forecast, fs=1.0, nfft=window_size, window='hann', detrend='linear', scaling='density')

    if normalize_variance:
        # Normalize the spectra to have unit variance
        print("Normalizing spectra to unit variance")
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
def red_noise_spectrum_with_confidence_fixed(r, f, data_length, window_size, overlap, confidence_levels=[0.05, 0.95]):
    """
    Calculate red noise spectrum with proper confidence bounds.
    
    Parameters:
    -----------
    r : float
        AR(1) coefficient 
    f : array_like
        Frequency array
    data_length : int
        ACTUAL length of the time series (in days)
    window_size : int
        Window size used in Welch's method
    overlap : int
        Overlap used in Welch's method
    confidence_levels : list
        Confidence levels for bounds [lower, upper]
    
    Returns:
    --------
    red_noise : array
        Red noise spectrum
    lower_bound : array
        Lower confidence bound
    upper_bound : array
        Upper confidence bound
    """
    # Calculate red noise spectrum using theoretical formula
    red_noise = (1 - r**2) / (1 + r**2 - 2 * r * np.cos(2 * np.pi * f))
    
    # Calculate correct degrees of freedom
    step = window_size - overlap
    n_segments = int(np.floor((data_length - window_size) / step) + 1)
    dof = 2 * n_segments  # Effective degrees of freedom
    
    print(f"Data length: {data_length} days")
    print(f"Window size: {window_size} days") 
    print(f"Overlap: {overlap} days")
    print(f"Step size: {step} days")
    print(f"Number of segments: {n_segments}")
    print(f"Degrees of freedom: {dof}")
    
    # Use chi-squared distribution for confidence bounds
    chi2_lower = chi2.ppf(confidence_levels[0], dof)
    chi2_upper = chi2.ppf(confidence_levels[1], dof)
    
    lower_bound = red_noise * dof / chi2_upper
    upper_bound = red_noise * dof / chi2_lower
    
    return red_noise, lower_bound, upper_bound

def validate_confidence_bounds(data_length=12419, window_size=1052, overlap=500):
    """
    Validate the confidence bound calculation with realistic parameters.
    """
    print("=== Confidence Bounds Validation ===")
    
    # Typical values for ERA5 1981-2014 data
    step = window_size - overlap
    n_segments = int(np.floor((data_length - window_size) / step) + 1)
    dof = 2 * n_segments
    
    print(f"For {data_length} days of data:")
    print(f"Window: {window_size} days, Overlap: {overlap} days")
    print(f"Step: {step} days")
    print(f"Number of segments: {n_segments}")
    print(f"Degrees of freedom: {dof}")
    
    # Calculate chi-squared values
    chi2_05 = chi2.ppf(0.05, dof)
    chi2_95 = chi2.ppf(0.95, dof)
    
    print(f"Chi-squared 5%: {chi2_05:.2f}")
    print(f"Chi-squared 95%: {chi2_95:.2f}")
    
    # Confidence bound ratios
    ratio_lower = dof / chi2_95
    ratio_upper = dof / chi2_05
    
    print(f"Lower bound ratio (red_noise * {ratio_lower:.3f})")
    print(f"Upper bound ratio (red_noise * {ratio_upper:.3f})")
    print(f"Confidence interval width: {ratio_upper/ratio_lower:.2f}x")
    
    # Expected ranges for DOF~29 (Lubis methodology)
    print("\n=== Comparison with Lubis et al. Expected Values ===")
    dof_lubis = 29
    chi2_05_lubis = chi2.ppf(0.05, dof_lubis)
    chi2_95_lubis = chi2.ppf(0.95, dof_lubis)
    ratio_lower_lubis = dof_lubis / chi2_95_lubis
    ratio_upper_lubis = dof_lubis / chi2_05_lubis
    
    print(f"Lubis DOF~29: Lower ratio: {ratio_lower_lubis:.3f}, Upper ratio: {ratio_upper_lubis:.3f}")
    print(f"Your DOF~{dof}: Lower ratio: {ratio_lower:.3f}, Upper ratio: {ratio_upper:.3f}")

# %%
def create_four_panel_lubis_figure(era5_data, amip_data, ace2_data, ngcm_data, 
                                   window_size=1052, overlap=500,  # Lubis methodology: 1052-day segments, 500-day overlap
                                   time_slice='1980-01-01:2014-12-31',  # 27 years for DOF~29
                                   save_path=None):
    """
    Create four-panel figure in Lubis et al. (2023) style showing power spectra
    for ERA5, AMIP, ACE2, and NGCM1 datasets.
    
    Parameters:
    -----------
    era5_data, amip_data, ace2_data, ngcm_data : xarray.Dataset
        Input datasets with 'pcs' variable containing EOF principal components
    window_size : int, default 1052
        Window size for Welch's method (Lubis methodology: 1052-day segments)
    overlap : int, default 500  
        Overlap for Welch's method (Lubis methodology: 500-day overlap)
    time_slice : str, default '1979-01-01:2005-12-31'
        Time slice for analysis (27 years for DOF~29, matching Lubis methodology)
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    
    # Setup figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8))
    axes = axes.flatten()
    
    # Dataset information
    datasets = [
        ('ERA5', era5_data, 'black'),
        ('AMIP', amip_data, 'tab:orange'), 
        ('ACE2', ace2_data, 'tab:red'),
        ('NGCM2.8', ngcm_data, 'tab:blue')
    ]
    
    def process_and_plot(ax, name, data, color, label):
        """Process individual dataset and create spectrum plot"""
        
        # Extract time slice for analysis
        start_year, end_year = time_slice.split(':')
        
        # Step 1: Detrend the principal component time series
        mode = 0
        ts_z1 = data.sel(mode=mode)

        data_detrend = xr.apply_ufunc(detrend, 
                                      (ts_z1).dropna('time').sel(time=slice(start_year, end_year)),
                                     input_core_dims=[['time']],
                                     output_core_dims=[['time']],
                                        vectorize=True,
                                        dask='parallelized',
                                        )
        
        # Step 3: Remove annual cycle and final detrend
        #smoothed_data = ts_z1.groupby('time.year') - ts_z1.groupby('time.year').mean('time') #_detrend_no_annual(smoothed_data)

        # Step 2.5: Apply additional window smoothing with weights [0.25, 0.5, 0.25]
        smoothed_data = running_window_smooth(data_detrend, weights=[0.25, 0.5, 0.25])

        # Step 2: Apply 21-day rolling mean smoothing (Lubis methodology)
        forecast_nino34 = smoothed_data.rolling(time=21, center=True).mean('time').dropna('time')

        Pxx_ens_member = []
        # Handle ensemble members if present
        if 'member_id' in forecast_nino34.dims:
            # Plot individual ensemble members in background
            for i, member in enumerate(forecast_nino34.member_id):
                member_data = forecast_nino34.sel(member_id=member)
                
                # Calculate spectrum for this member
                f_member, Pxx_member = sig.welch(member_data, fs=1.0, nperseg=window_size, noverlap=overlap, 
                                               window='hann')
                
                # Normalize the spectrum
                Pxx_member_norm = Pxx_member / np.max(Pxx_member)
                
                Pxx_ens_member.append(Pxx_member)
                # Plot individual member with low alpha
                iend = len(f_member)
                ax.plot(f_member[1:iend], Pxx_member_norm[1:iend], color=color, 
                       linewidth=0.5, alpha=0.3, zorder=100)

            Pxx_ens_member_array = np.mean(np.array(Pxx_ens_member),axis=0)
            Pxx_ens_member_norm = Pxx_ens_member_array / np.max(Pxx_ens_member_array)

            iend = len(f_member)
            ax.plot(f_member[1:iend], Pxx_ens_member_norm[1:iend], color=color, linewidth=2, 
                label=f'{name}', zorder=101)


        # Calculate ensemble mean for main plot after individual member spectra
        #if 'member_id' in forecast_nino34.dims:
        #    forecast_mean = forecast_nino34.mean('member_id')
             
        else:
            #forecast_mean = forecast_nino34

            # Calculate spectra using Welch's method with 50% overlap for ensemble mean
            f, Pxx = sig.welch(forecast_nino34, fs=1.0, nperseg=window_size, noverlap=overlap, 
                            window='hann', detrend='linear')
        
            # Normalize the spectra to have unit variance
            Pxx_norm = Pxx / np.max(Pxx)
            #Pxx_norm = Pxx / np.var(forecast_nino34)

                    # Plot power spectrum
            iend = len(f)
            ax.plot(f[1:iend], Pxx_norm[1:iend], color=color, linewidth=2, 
                label=f'{name}', zorder=101)
        

        if 'member_id' in forecast_nino34.dims:
            forecast_mean = forecast_nino34.mean('member_id')
        else:
            forecast_mean = forecast_nino34

        f, Pxx = sig.welch(forecast_mean, fs=1.0, nperseg=window_size, noverlap=overlap, 
                            window='hann', detrend='linear')

        Pxx_norm = Pxx / np.max(Pxx)

        # Estimate AR1 correlation for red noise background (using ensemble mean)
        r_y = estimate_ar1(forecast_mean.values)
        
        # Calculate variance (σ²) of the time series for Lubis red noise formula
        sigma2 = np.var(forecast_mean.values)

        # FIXED: Use actual data length for confidence bounds
        data_length = len(forecast_mean)
        
        # Calculate red noise with proper confidence bounds
        P_red_y, P_red_lower, P_red_upper = red_noise_spectrum_with_confidence_fixed(
            r_y, f, data_length, window_size, overlap
        )
        
        # FIXED: Proper normalization to match spectrum units
        # The red noise should be scaled to match the spectral density units
        mean_power = np.mean(Pxx_norm[1:20])  # Low-frequency average
        mean_red = np.mean(P_red_y[1:20])
        scale = mean_power / mean_red
        
        P_red_y_scaled = P_red_y * scale
        P_red_lower_scaled = P_red_lower * scale  
        P_red_upper_scaled = P_red_upper * scale
        
        # Plot red noise background
        ax.plot(f[1:iend], P_red_y_scaled[1:iend], '--', color='tab:blue', linewidth=1.5, 
                label=f'Red noise (r={r_y:.3f})', alpha=0.8)
        
        # FIXED: Proper confidence bounds
        ax.fill_between(f[1:iend], P_red_lower_scaled[1:iend], P_red_upper_scaled[1:iend], 
                    color='tab:blue', alpha=0.2, label='5-95% CI')
        
        # Print diagnostic inform
        
        # Add 150-day period line
        ax.axvline(1/(150), color='red', linestyle='-', linewidth=2, alpha=0.8, 
                  label='150 days')
        
        # Formatting to match Lubis style
        ax.set_xlabel('Frequency (cpd)', fontsize=12)
        ax.set_ylabel('Normalized Power', fontsize=12)
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.set_xlim([0.001, 0.03])  # Match Lubis frequency range
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.minorticks_on()
        ax.text(0.01,1.02, label, 
            ha='left', va='bottom', 
            weight='bold',
            transform=ax.transAxes,
            fontsize=16)
        
        # Add theoretical 150-day periodicity prediction line (as in Lubis figure)
        freq_150 = 1/150
        if freq_150 >= 0.001 and freq_150 <= 0.03:
            # Add shaded region around 150-day period to show uncertainty
            ax.axvspan(1/160, 1/140, color='red', alpha=0.1, 
                      label='150-day range')
        
        return ax

    # Process each dataset
    for i, (name, data, color) in enumerate(datasets):
        if data is not None:
            try:
                labels = ['a', 'b', 'c', 'd']
                process_and_plot(axes[i], name, data, color, labels[i])
            except Exception as e:
                print(f"Warning: Could not process {name}: {e}")
                axes[i].text(0.5, 0.5, f'{name}\nData not available', 
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=12)
                axes[i].set_xlabel('Frequency (cpd)')
                axes[i].set_ylabel('Normalized Power')
                axes[i].set_title(name, fontsize=14, fontweight='bold')

        else:
            axes[i].set_xlabel('Frequency (cpd)')
            axes[i].set_ylabel('Normalized Power') 
            axes[i].set_title(name, fontsize=14, fontweight='bold')
    
    # Overall figure formatting
    #fig.suptitle('SAM Power Spectra: 150-Day Periodicity Analysis', 
    #            fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for suptitle
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


# Alternative simplified usage if you have the data loaded:
def create_lubis_four_panel_simple(era5, amip, ace2, ngcm):
    """Simplified version - modify the dataset names as needed"""
    
    # Replace these with your actual loaded datasets
    datasets_dict = {
        'era5': era5,    
        'amip': amip.drop_sel(member_id=12),      
        'ace2': ace2,       
        'ngcm': ngcm         
    }

    fig = create_four_panel_lubis_figure(
        era5_data=datasets_dict['era5'],
        amip_data=datasets_dict['amip'], 
        ace2_data=datasets_dict['ace2'],
        ngcm_data=datasets_dict['ngcm'],
        window_size=1052,
        overlap=500,
        time_slice='1981-01-01:2014-12-31',
        save_path='../../plots/Lubis_power_spectra.png'
    )
    return fig

# %%

def preprocess(file):
        eof_results = xr.open_dataset(file, decode_times=False).squeeze()
        print(eof_results)
        eof_results.coords['time'] = pd.date_range('1981-01-01', periods=len(eof_results.time), freq='D')
        z1 = eof_results['z1'] #.dropna('time')
        z1.coords['mode'] = [0,1]
        z1 = z1.where( z1 > -500, drop=True)
        return z1.sel(time=slice('1981-01-01','2014-12-31'))

# %%
def main():
    test_directory = './'# Add file
    era5 = preprocess(sorted(glob.glob(f'{test_directory}/z1/z1_ERA5_*.nc'))[-1])

    files = sorted(glob.glob(f'{test_directory}/z1/*ua_CMIP6*.nc'))
    amip = [preprocess(files[i]) for idx, i in enumerate(range(len(files))) if 'ua_CMIP6' in files[i]]
    amip = xr.concat(amip, dim='member_id')
    amip.coords['member_id'] = np.arange(1,len(files)+1,step=1)

    files = sorted(glob.glob(f'{test_directory}/z1/*csp*.nc'))
    ngcm = [preprocess(files[i]) for idx, i in enumerate(range(len(files))) if '2.8' in files[i]]
    ngcm = xr.concat(ngcm, dim='member_id')
    ngcm.coords['member_id'] = np.arange(1,38,step=1)

    files = sorted(glob.glob(f'{test_directory}/z1/*eastward*.nc'))
    ace2 = [preprocess(files[i]) for idx, i in enumerate(range(len(files))) if 'eastward' in files[i]]
    ace2 = xr.concat(ace2, dim='member_id')
    ace2.coords['member_id'] = np.arange(1,38,step=1)

    fig = create_lubis_four_panel_simple(era5, amip, ace2, ngcm)
    plt.show(), gc.collect()

# %%
if __name__ == '__main__':
    main()


# %%
