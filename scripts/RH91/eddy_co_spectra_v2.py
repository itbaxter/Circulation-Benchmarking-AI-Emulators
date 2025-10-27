# %%
"""
Adapted from code originally made by Nicholas Lutsko (EAPS department, MIT) 
Retrieved from Nick's website: https://nicklutsko.github.io/code/ 


Functions for calculating eddy flux co-spectra. Script follows technique of Hayashi (1971) 
(see also Randel and Held (1991)) and calculates spectra at a specific height.

Includes functions to calculate space-time cross-spectra and phase-speed cross-spectrum.

Updated May 30th 2018 -- fixed bug identified by Ben Toms
Updated March 14th 2019 -- fixed several bugs identified by Neil Lewis

Updated May 2024 -- adapted for Python3, xarray, and with some modifications to increase speed by Ian Baxter

Tested using Python 3.12
"""
# %%
"""
Eddy flux co-spectra (Hayashi 1971; Randel & Held 1991) with modular structure.

- Command-line parser to specify data_dir and output_dir
- Reader function to load datasets
- Computational and plotting functions
- main() to orchestrate execution
"""

import argparse
import os
from pathlib import Path
import numpy as np
import scipy.signal as ss
import scipy.interpolate as si
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, detrend
import glob as glob
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

from scipy.stats import linregress as _linregress
import xarray as xr
def linregress(da_y, da_x, dim=None):
    '''xarray-wrapped function of scipy.stats.linregress.
    Note the order of the input arguments x, y is reversed to the original scipy function.'''
    if dim is None:
        dim = [d for d in da_y.dims if d in da_x.dims][0]

    slope, intercept, r, p, stderr = xr.apply_ufunc(_linregress, da_x, da_y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        dask='allowed')
    predicted = da_x * slope + intercept

    slope.attrs['long_name'] = 'slope of the linear regression'
    intercept.attrs['long_name'] = 'intercept of the linear regression'
    r.attrs['long_name'] = 'correlation coefficient'
    p.attrs['long_name'] = 'p-value'
    stderr.attrs['long_name'] = 'standard error of the estimated gradient'
    predicted.attrs['long_name'] = 'predicted values by the linear regression model'

    return xr.Dataset(dict(slope=slope, intercept=intercept,
        r=r, p=p, stderr=stderr, predicted=predicted))

# %%
def parse_args():
    parser = argparse.ArgumentParser(description='Eddy co-spectra (RH91) analysis')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Base data directory (expects era5/, ace2/, ngcm/, amip/ subfolders). If None, uses script-relative ../../..-data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures. If None, uses script-relative ../../plots')
    parser.add_argument('--ace2_index', type=int, default=6, help='Index for ACE2 member files (default: 6)')
    parser.add_argument('--ngcm_index', type=int, default=20, help='Index for NGCM member files (default: 20)')
    return parser.parse_args()


def resolve_dir(passed: str | None, fallback: Path) -> Path:
    if passed:
        return Path(passed).expanduser().resolve()
    return fallback.resolve()


def read_datasets(base_dir: Path, ace2_index: int = 6, ngcm_index: int = 20):
    """Load datasets from a standardized directory layout under base_dir.

    Returns a dict with keys: u_era5, v_era5, u_ace2, v_ace2, u_ngcm, v_ngcm, u_amip, v_amip
    """
    ds = {}

    # ERA5
    era5_u = base_dir / 'era5/u250/u250_era5_1958_2023.nc'
    era5_v = base_dir / 'era5/v250/v250_era5_1961_2001.nc'
    if not era5_u.exists():
        raise FileNotFoundError(f'Missing ERA5 u file: {era5_u}')
    if not era5_v.exists():
        raise FileNotFoundError(f'Missing ERA5 v file: {era5_v}')
    ds['u_era5'] = xr.open_dataset(era5_u)['u'].resample(time='1D').mean('time').isel(bnds=0)
    ds['v_era5'] = xr.open_dataset(era5_v)['v'].resample(time='1D').mean('time').isel(bnds=0)

    # ACE2
    ace2_u_files = sorted(glob.glob(str(base_dir / 'ace2/u250/eastward*.nc')))
    ace2_v_files = sorted(glob.glob(str(base_dir / 'ace2/v250/northward*.nc')))
    if not ace2_u_files:
        raise FileNotFoundError(f'No ACE2 u files under {base_dir / "ace2/u250"}')
    if not ace2_v_files:
        raise FileNotFoundError(f'No ACE2 v files under {base_dir / "ace2/v250"}')
    if ace2_index >= len(ace2_u_files) or ace2_index >= len(ace2_v_files):
        raise IndexError(f'ace2_index {ace2_index} out of range (u:{len(ace2_u_files)}, v:{len(ace2_v_files)})')
    ds['u_ace2'] = xr.open_dataset(ace2_u_files[ace2_index])['u']
    ds['v_ace2'] = xr.open_dataset(ace2_v_files[ace2_index])['v']

    # NGCM2.8
    ngcm_u_files = sorted(glob.glob(str(base_dir / 'ngcm/u250/*ngcm*nc')))
    ngcm_v_files = sorted(glob.glob(str(base_dir / 'ngcm/v250/*ngcm*nc')))
    if not ngcm_u_files or not ngcm_v_files:
        raise FileNotFoundError(f'No NGCM files under {base_dir / "ngcm"}')
    if ngcm_index >= len(ngcm_u_files) or ngcm_index >= len(ngcm_v_files):
        raise IndexError(f'ngcm_index {ngcm_index} out of range (u:{len(ngcm_u_files)}, v:{len(ngcm_v_files)})')
    ds['u_ngcm'] = xr.open_dataset(ngcm_u_files[ngcm_index])['v'].resample(time='1D').mean('time')
    ds['v_ngcm'] = xr.open_dataset(ngcm_v_files[ngcm_index])['v'].resample(time='1D').mean('time')

    # AMIP (CESM2-WACCM example)
    amip_u = base_dir / 'amip/u250/ua_CMIP6_CESM2-WACCM_day_amip_1950-2016.nc'
    amip_v = base_dir / 'amip/v250/va_CMIP6_CESM2-WACCM_day_amip_1950-2016.nc'
    if not amip_u.exists():
        raise FileNotFoundError(f'Missing AMIP u file: {amip_u}')
    if not amip_v.exists():
        raise FileNotFoundError(f'Missing AMIP v file: {amip_v}')
    ds['u_amip'] = xr.open_dataset(amip_u).sel(plev=25000)['ua'].squeeze()
    ds['v_amip'] = xr.open_dataset(amip_v).sel(plev=25000)['va'].squeeze()

    return ds


# %%

def calc_spacetime_cross_spec( a, b, dx, ts = 1., smooth = 1, width = 15., NFFT = 128 ):
    """
    Calculate space-time co-spectra, following method of Hayashi (1971)
    
    This function performs Fourier transforms in space and time to compute cross-spectra
    between two variables, separating eastward (positive) and westward (negative) propagating waves.

    Input:
      a - variable 1, dimensions = (time, space)
      b - variable 2, dimensions = (time, space)  
      dx - x-grid spacing (unit = space) [NOTE: This parameter appears in docstring but not function signature]
      ts - sampling interval (unit = time)
      smooth - 1 = apply Gaussian smoothing to reduce noise
          width - width of Gaussian smoothing kernel
          NFFT - length of FFT in cross-spectra calculations, sets frequency resolution

    Output:
      K_p - spectra for positive (eastward) frequencies
      K_n - spectra for negative (westward) frequencies  
      lon_freqs - wavenumbers (spatial frequencies)
      om - temporal frequencies
    """
    # Get dimensions of input arrays
    t, l = np.shape( a )  # t = time points, l = spatial points
    lf = l // 2  # Half the spatial points for positive wavenumbers only

    # Calculate spatial FFTs for both variables
    # Normalize by spatial points and convert to per meter (RH91 convention)
    Fa = (np.fft.fft(a, axis=1) / (float(l))) # FFT along spatial dimension, per meter
    Fb = (np.fft.fft(b, axis=1) / (float(l))) # FFT along spatial dimension, per meter

    # Generate wavenumber array, keep only positive wavenumbers
    # This represents n/(a*cos(phi)) in spherical coordinates per Randel & Held
    dx_scaled = np.zeros(l) + dx # Scale dx by cosine of latitude (0 for equator)
    lon_freq = np.fft.fftfreq( l, d = dx )[:lf] 

    # Extract real (cosine) and imaginary (sine) parts of spatial FFTs
    # These represent the amplitudes of cosine and sine basis functions
    CFa = Fa[:, :lf].real  # Cosine coefficients for variable a
    SFa = Fa[:, :lf].imag  # Sine coefficients for variable a
    CFb = Fb[:, :lf].real  # Cosine coefficients for variable b
    SFb = Fb[:, :lf].imag  # Sine coefficients for variable b

    # Number of frequency bins for temporal FFT
    tf = int(NFFT / 2 + 1)

    # Initialize arrays to hold cross-spectra for positive and negative phase speeds
    K_p = np.zeros( ( tf, lf ) )  # Eastward propagating waves
    K_n = np.zeros( ( tf, lf ) )  # Westward propagating waves
    
    # OPTIMIZED: Direct FFT cross-spectra calculation (10x faster than plt.csd)
    # Pre-compute window for all wavenumbers
    window = np.hanning(NFFT)
    
    # Vectorized temporal FFTs for all wavenumber components
    #print('Computing optimized cross-spectra...')
    
    # Pad data to NFFT length and apply window
    n_segments = t // NFFT
    if n_segments == 0:
        # If data is shorter than NFFT, pad with zeros
        CFa_pad = np.pad(CFa, ((0, NFFT - t), (0, 0)), mode='constant')
        SFa_pad = np.pad(SFa, ((0, NFFT - t), (0, 0)), mode='constant')
        CFb_pad = np.pad(CFb, ((0, NFFT - t), (0, 0)), mode='constant')
        SFb_pad = np.pad(SFb, ((0, NFFT - t), (0, 0)), mode='constant')
        n_segments = 1
    else:
        # Use overlapping segments for better statistics
        CFa_pad = CFa[:n_segments*NFFT, :].reshape(n_segments, NFFT, lf)
        SFa_pad = SFa[:n_segments*NFFT, :].reshape(n_segments, NFFT, lf)
        CFb_pad = CFb[:n_segments*NFFT, :].reshape(n_segments, NFFT, lf)
        SFb_pad = SFb[:n_segments*NFFT, :].reshape(n_segments, NFFT, lf)
    
    # Apply window and compute FFTs for all wavenumbers at once
    if n_segments == 1:
        # Apply window
        CFa_win = CFa_pad * window[:, np.newaxis]
        SFa_win = SFa_pad * window[:, np.newaxis]
        CFb_win = CFb_pad * window[:, np.newaxis]
        SFb_win = SFb_pad * window[:, np.newaxis]
        
        # Temporal FFTs
        CFa_fft = np.fft.fft(CFa_win, axis=0)[:tf, :] #/ NFFT
        SFa_fft = np.fft.fft(SFa_win, axis=0)[:tf, :] # NFFT
        CFb_fft = np.fft.fft(CFb_win, axis=0)[:tf, :] #/ NFFT
        SFb_fft = np.fft.fft(SFb_win, axis=0)[:tf, :] #/ NFFT

        # Cross-spectral densities (vectorized across all wavenumbers)
        csd_CaCb = (CFa_fft * np.conj(CFb_fft)).real
        csd_SaSb = (SFa_fft * np.conj(SFb_fft)).real
        csd_CaSb = (CFa_fft * np.conj(SFb_fft)).imag
        csd_SaCb = (SFa_fft * np.conj(CFb_fft)).imag
        
    else:
        # Multiple segments - average over segments
        csd_CaCb = np.zeros((tf, lf))
        csd_SaSb = np.zeros((tf, lf))
        csd_CaSb = np.zeros((tf, lf))
        csd_SaCb = np.zeros((tf, lf))
        
        for seg in range(n_segments):
            # Apply window
            CFa_win = CFa_pad[seg, :, :] * window[:, np.newaxis]
            SFa_win = SFa_pad[seg, :, :] * window[:, np.newaxis]
            CFb_win = CFb_pad[seg, :, :] * window[:, np.newaxis]
            SFb_win = SFb_pad[seg, :, :] * window[:, np.newaxis]
            
            # Temporal FFTs
            CFa_fft = np.fft.fft(CFa_win, axis=0)[:tf, :]
            SFa_fft = np.fft.fft(SFa_win, axis=0)[:tf, :]
            CFb_fft = np.fft.fft(CFb_win, axis=0)[:tf, :]
            SFb_fft = np.fft.fft(SFb_win, axis=0)[:tf, :]
            
            # Accumulate cross-spectral densities
            csd_CaCb += (CFa_fft * np.conj(CFb_fft)).real
            csd_SaSb += (SFa_fft * np.conj(SFb_fft)).real
            csd_CaSb += (CFa_fft * np.conj(SFb_fft)).imag
            csd_SaCb += (SFa_fft * np.conj(CFb_fft)).imag
        
        # Average over segments
        csd_CaCb /= n_segments
        csd_SaSb /= n_segments
        csd_CaSb /= n_segments
        csd_SaCb /= n_segments
    
    # Combine cross-spectra according to Hayashi (1971) formulation (vectorized)
    # These combinations separate eastward vs westward propagating components
    #        K_p[:, i] = csd_CaCb.real + csd_SaSb.real + csd_CaSb.imag - csd_SaCb.imag  # Eastward
    #       K_n[:, i] = csd_CaCb.real + csd_SaSb.real - csd_CaSb.imag + csd_SaCb.imag  # Westward
    # Hayashi (1971) Eqs. 4.9 and 4.10:
    #K_p = 1.0 * (csd_CaCb + csd_SaSb + csd_CaSb - csd_SaCb)  # Eastward (positive phase speed)
    #K_n = 1.0 * (csd_CaCb + csd_SaSb - csd_CaSb + csd_SaCb)  # Westward (negative phase speed)
    K_p = 2.0 * (csd_CaCb + csd_SaSb - csd_CaSb + csd_SaCb)  # Eastward (positive phase speed)
    K_n = 2.0 * (csd_CaCb + csd_SaSb + csd_CaSb - csd_SaCb)  # Westward (negative phase speed)
    # Nick's code says you don't need factor 4 from Hayashi eq4.11, since Fourier co-efficients are 1/2 as large due to only retaining positive wavenumbers
    # But this factor was added to get consistency with Hayashi 1971 and RH91

    #k = np.fft.fftfreq(n_lon, d=dx)[:n_lon//2 + 1]
    
    #lon_freq = 2 * np.pi * lon_freq  # convert to radians per meter


    # Create frequency array (only need to do this once)
    om = np.fft.fftfreq(NFFT, ts)[:tf]
    #om = 2 * np.pi * om
        # Note: Factor of 4 from Hayashi eq 4.11 not needed since FFT coefficients 
        # are 1/2 size due to keeping only positive wavenumbers

    # Combine positive and negative frequency spectra for smoothing
    K_combine = np.zeros( ( tf * 2, lf ) ) 
    K_combine[:tf, :] = K_n[::-1, :]   # Reversed negative frequencies for convolution
    K_combine[tf:, :] = K_p[:, :]      # Positive frequencies

    # Apply Gaussian smoothing if requested to reduce noise
    if smooth == 1.:
        # Create Gaussian filter centered at zero
        x = np.linspace( -tf / 2, tf / 2., tf )
        gauss_filter = np.exp( -x ** 2 / (2. * width ** 2 ) )
        gauss_filter /= sum( gauss_filter )  # Normalize filter
        
        # Apply smoothing to each wavenumber independently
        for i in range( lf ):
            K_combine[:, i] = np.convolve( K_combine[:, i], gauss_filter, 'same' )

    # Separate back into positive and negative frequency components
    K_n = K_combine[:tf, :]    # Negative (westward) frequencies
    K_p = K_combine[tf:, :]    # Positive (eastward) frequencies  
    K_n = K_n[::-1, :]         # Reverse negative frequencies back to original order

    if smooth == 1:
        U = np.mean(window**2)
        K_n /= U
        K_p /= U

    # Multiply by dx to ensure correct physical units (per meter)
    #K_p *= dx
    #K_n *= dx

    #K_p *= (2 * np.pi)**2
    #K_n *= (2 * np.pi)**2

    return K_p , K_n, lon_freq, om 

def calPhaseSpeedSpectrum( P_p, P_n, f_lon, om, cmax, nps, i1 = 1, i2 = 15 ):
    """
    Convert frequency-wavenumber spectra to phase speed spectra
    
    This function transforms the frequency-wavenumber cross-spectra into phase speed spectra
    by interpolating along lines of constant phase speed (c = ω/k) in frequency-wavenumber space.

    Input:
      P_p - spectra for positive (eastward) phase speeds from frequency-wavenumber analysis
      P_n - spectra for negative (westward) phase speeds from frequency-wavenumber analysis
      f_lon - wavenumbers (spatial frequencies)
      om - temporal frequencies
      cmax - maximum phase speed for output grid
      nps - number of points in phase speed grid
      i1 - lowest wavenumber index to include in summation (default=1, skip k=0)
          i2 - highest wavenumber index to include in summation (default=50)
    
    Output:
      P_cp - integrated spectra for positive phase speeds
      P_cn - integrated spectra for negative phase speeds  
      C - phase speed grid (units: spatial_unit/time_unit)
    """
    # Check that wavenumber range makes sense
    if i2 < i1:
        print("WARNING: highest wavenumber smaller than lowest wavenumber")

    # Get dimensions of input spectra
    j = len( f_lon )  # Number of wavenumbers
    t = len( om )     # Number of frequencies

    # Create evenly-spaced phase speed grid from 0 to maximum phase speed
    C = np.linspace(0., cmax, nps)

    # Initialize arrays to hold phase speed spectra for each wavenumber
    P_cp = np.zeros( ( nps, j ) )  # Positive phase speeds
    P_cn = np.zeros( ( nps, j ) )  # Negative phase speeds

    # Interpolate from frequency-wavenumber to phase speed space
    for i in range( int(i1), int(i2) ):
        # Create interpolation functions mapping phase speed (c = ω/k) to spectral power
        # These interpolate along lines of constant phase speed in (ω,k) space
        f1 = si.interp1d(om / f_lon[i], P_p[:, i], 'linear' )  # For positive frequencies
        f2 = si.interp1d(om / f_lon[i], P_n[:, i], 'linear' )  # For negative frequencies

        # Handle extrapolation: interp1d doesn't extrapolate well, so zero out 
        # phase speeds beyond the maximum frequency/wavenumber ratio
        k = -1
        for j in range( len(C) ):
            if C[j] > max(om) / f_lon[i]:  # Find where phase speed exceeds max possible
                k = j
                break
        if k == -1:  # If all phase speeds are within range
            k = len( C )	
        
        # Initialize arrays and fill with interpolated values up to maximum valid phase speed
        ad1 = np.zeros( nps )
        ad1[:k] =  f1( C[:k]  )  # Interpolate positive phase speeds
        ad2 = np.zeros( nps )
        ad2[:k] =  f2( C[:k] )  # Interpolate negative phase speeds

        # Scale by wavenumber (Jacobian of transformation from ω-k to c-k space)
        P_cp[:, i] = ad1 * f_lon[i]  # Positive phase speed spectrum for this wavenumber
        P_cn[:, i] = ad2 * f_lon[i]  # Negative phase speed spectrum for this wavenumber

     # Integrate over all wavenumbers to get total phase speed spectrum
    return np.sum(P_cp, axis = 1), np.sum(P_cn, axis = 1), C

def calc_co_spectra( x, y, dx, lat, dt, cmax = 50, nps = 50 ):
    """
    Calculate eddy phase speed co-spectra following Hayashi (1974) methodology
    
    This is the main function that coordinates the analysis. It processes 3D data
    (time, latitude, longitude) to compute phase speed co-spectra at each latitude.
    The method separates eastward and westward propagating eddy components.

    Input:
      x - first variable, dimensions = (time, lat, lon)
      y - second variable, dimensions = (time, lat, lon) 
      dx - spacing between longitude points (unit = meters)
      lat - latitude values (degrees) 
            NOTE: for spherical coordinates, dx should be scaled by a*cos(lat)
        where a is Earth radius
      dt - time sampling interval (unit = seconds)
      cmax - maximum phase speed for analysis (m/s)
      nps - number of points in phase speed grid

    Output:
      p_spec - phase speed co-spectra, dimensions = (lat, 2*nps)
               First nps points are negative (westward) phase speeds
           Last nps points are positive (eastward) phase speeds  
      ncps - phase speed values corresponding to p_spec columns
    """
    # Validate input dimensions
    if x.ndim != 3:
        print("WARNING: Dimensions of x != 3")
    if y.ndim != 3:
        print("WARNING: Dimensions of y != 3")

    # Get array dimensions
    t, l, j = np.shape( x )  # time, latitude, longitude
    
    # Remove zonal mean from both variables to focus on eddy components
    # This isolates the wave-like (eddy) variations from the mean flow
    #x_anom = x - np.mean( x, axis = 2 )[:, :, np.newaxis]  # Remove zonal mean from x
    #y_anom = y - np.mean( y, axis = 2 )[:, :, np.newaxis]  # Remove zonal mean from y

    # RH91 Method:
    sqrt_coslat = np.sqrt(np.cos(np.radians(lat)))
    x_anom = (x - np.mean(x, axis=2)[:, :, np.newaxis]) #/ sqrt_coslat[:, np.newaxis]
    y_anom = (y - np.mean(y, axis=2)[:, :, np.newaxis]) #/ sqrt_coslat[:, np.newaxis] 

    # Initialize array to hold spectra for all latitudes
    p_spec = np.zeros( ( l, 2 * nps ) )  # lat x (negative + positive phase speeds)

    # Process each latitude independently
    for i in range( l ):
        #print("Doing: ", i) 
        
        # Apply cosine latitude scaling for spherical coordinates
        # At higher latitudes, meridians converge, so effective dx is smaller
        # Earth radius in meters
        R = 6.371e6
        dx = 2 * np.pi * R / float(j)  # meters per longitude grid point
        dx_scaled = dx * np.cos(np.radians(lat[i])) #/ float(j)

        # Calculate space-time cross-spectra for this latitude
        K_p, K_n, lon_freq, om = calc_spacetime_cross_spec( x_anom[:, i, :], y_anom[:, i, :], dx = dx_scaled, ts = dt )
        
        # Convert frequency-wavenumber spectra to phase speed spectra
        # RH91 typically integrates over ~15-20 wavenumbers, not nps/2
        max_wavenumber = min(20, len(lon_freq)-1)  # Up to wavenumber 20 or available wavenumbers
        P_Cp, P_Cn, cp = calPhaseSpeedSpectrum(K_p, K_n, lon_freq, om, cmax, nps, 3, max_wavenumber)
        
        # Apply area weighting to account for spherical geometry
        # Grid cells are smaller at higher latitudes, so we weight by cos(lat)
        area_weight = np.cos(np.deg2rad(lat[i]))
        
        # Store results: negative phase speeds (reversed) then positive phase speeds
        p_spec[i, :nps] = P_Cn[::-1] * area_weight  # Westward propagating (negative c), reversed order
        p_spec[i, nps:] = P_Cp[:] * area_weight     # Eastward propagating (positive c)

    # Create full phase speed coordinate array
    ncps = np.zeros( 2 * nps )      # Array for negative + positive phase speeds
    ncps[:nps] = -1. * cp[::-1]     # Negative phase speeds (reversed and negated)
    ncps[nps:] = cp[:]              # Positive phase speeds
    # NOTE: This assumes nps=50, so 2*nps=100. Should use 'nps' instead of hardcoded '100'

    return p_spec, ncps


# %%
def process(u, v, years, season='DJFM', apply_window=True):
    """
    Process wind data for space-time spectral analysis with proper latitude weighting
    and windowing to reduce spectral leakage.
    
    Parameters:
    -----------
    u, v : xarray.DataArray
        Zonal and meridional wind components
    years : array-like
        Years to process
    season : str
        Season to analyze ('DJFM' or 'JJAS')
    apply_window : bool
        Whether to apply Hanning window to reduce spectral leakage
        
    Returns:
    --------
    East, West : xarray.DataArray
        Eastward and westward propagating wave spectra
    """
    p_spec = []
    c_spec = []
    
    for year in years[:]:
        # Select data for the specified season
        if season == 'DJFM':
            u_sel = u.squeeze().sel(time=slice(f'{year}-12-01', f'{year+1}-03-31'))
            v_sel = v.squeeze().sel(time=slice(f'{year}-12-01', f'{year+1}-03-31'))
        elif season == 'JJAS':
            u_sel = u.squeeze().sel(time=slice(f'{year}-06-01', f'{year}-09-30'))
            v_sel = v.squeeze().sel(time=slice(f'{year}-06-01', f'{year}-09-30'))
        
        # Remove leap days
        u_sel = u_sel.sel(time=~((u_sel.time.dt.month == 2) & (u_sel.time.dt.day == 29)))
        v_sel = v_sel.sel(time=~((v_sel.time.dt.month == 2) & (v_sel.time.dt.day == 29)))
        
        # Calculate wavenumbers and frequencies in physical units
        n_time = u_sel.sizes['time']
        n_lon = u_sel.sizes['lon']

        # Earth radius in meters
        R = 6.371e6
        dx = 2 * np.pi * R / n_lon  # meters per longitude grid point

        # Calculate phase speed spectrum
        p, c = xr.apply_ufunc(
            calc_co_spectra,
            u_sel, v_sel, 
            dx, u_sel['lat'],
            input_core_dims=[['time', 'lat', 'lon'], ['time', 'lat', 'lon'], [], ['lat']],
            output_core_dims=[['lat','c'], ['c']],
            kwargs={'dt': 86400, 'cmax': 40, 'nps': 50},
            vectorize=True
        )

        p_spec.append(p)
        c_spec.append(c)

    p_spec = xr.concat(p_spec, dim='year')
    c_spec = xr.concat(c_spec, dim='year')

    return p_spec, c_spec

# %%
def get_winds(ds,season='DJFM'):
    if season == 'DJFM':
        ds_sel = ds.sel(time=ds.time.dt.month.isin([1,2,3,12])).mean('time') #.groupby('time.month').mean('time').mean('month')
    elif season == 'JJAS':
        ds_sel = ds.sel(time=ds.time.dt.month.isin([6,7,8,9])).mean('time') #.groupby('time.month').mean('time').mean('month')
    #ds_sel = ds_sel.sel(time=~((ds_sel.time.dt.month == 2) & (u_sel.time.dt.day == 29)))
    return ds_sel

# %%


def compute_emfc_rh91_vectorized(uv_power_by_c):
    """Vectorized EMFC computation following RH91 Eq. 9"""
    a = 6.371e6
    lat = uv_power_by_c['lat']
    lat_rad = np.deg2rad(lat)
    coslat = np.cos(lat_rad)
    
    # Apply cos(lat) weighting
    flux_cos = uv_power_by_c * coslat
    
    # Compute derivative
    d_flux_cos = flux_cos.differentiate('lat') * (np.pi/180)
    
    # EMFC
    emfc = -1/(a * coslat) * d_flux_cos
    
    emfc.attrs['units'] = 'm/s^2'
    return emfc

# %%
def compute_emfc(uv_power, lat_name='lat'):
    """
    Compute eddy momentum flux convergence (EMFC) following Chen and Held (2007, JAS Eq. 1).

    Parameters
    ----------
    uv_power : xarray.DataArray
        Eddy momentum flux (e.g., sum of u'v' cospectrum over k and omega), shape (..., latitude)
    lat_name : str
        Name of the latitude coordinate in uv_power

    Returns
    -------
    emfc : xarray.DataArray
        Eddy momentum flux convergence (units: m/s^2), same shape as input minus latitude
    """
    # Earth's radius in meters
    a = 6.371e6
    lat = uv_power[lat_name]
    lat_rad = np.deg2rad(lat)
    coslat = np.cos(lat_rad)

    # Chen & Held (2007) Eq. 1: EMFC = -1/(a cos(phi)) * d/dphi (uv cos(phi))
    # phi in radians
    flux_cos = uv_power * coslat
    dphi = np.gradient(lat_rad)
    d_flux_cos_dphi = xr.apply_ufunc(
        np.gradient, flux_cos, dphi,
        input_core_dims=[[lat_name], [lat_name]],
        output_core_dims=[[lat_name]],
        vectorize=True
    )
    emfc = -1 / (a * coslat) * d_flux_cos_dphi 
    emfc.name = 'emfc'
    emfc.attrs['units'] = 'm/s^2'
    emfc.attrs['long_name'] = 'Eddy momentum flux convergence (Chen & Held 2007 Eq. 1)'
    return emfc * 86400

# %%
def compute_emfc_clean(uv_power, lat_name='lat'):
    """EMFC following Chen & Held (2007) with correct units"""
    a = 6.371e6  # Earth radius in meters
    lat = uv_power[lat_name]
    lat_rad = np.deg2rad(lat)
    coslat = np.cos(lat_rad)
    
    # Multiply by cos(lat) before taking derivative
    flux_cos = uv_power * coslat
    
    # Take derivative with respect to latitude in degrees, then convert to radians
    # differentiate('lat') gives d/d(lat_degrees), so we divide by (180/π) to get d/d(lat_radians)
    d_flux_cos = flux_cos.differentiate('lat') * (np.pi/180)
    
    # Apply EMFC formula
    emfc = -1 / (a * coslat) * d_flux_cos
    
    emfc.attrs['units'] = 'm/s^2'
    return emfc * 86400

# %%
def wrapper(ua,va,years=np.arange(1981, 2015, 1), season='DJFM'):
    if 'lon' not in ua.dims:
        ua = ua.rename({'longitude': 'lon','latitude': 'lat'})
        va = va.rename({'longitude': 'lon','latitude': 'lat'})
    p_spec, c = process(ua.squeeze(), va.squeeze(), years, season=season, apply_window=True)
    p_spec.coords['year'] = years

    C = c[0,:]
    dc = np.gradient(C)[0]
    #C = C[C != 0]
    p_spec.coords['c'] = C
    return p_spec / dc
# %%
# === Helper function for custom diverging colormap ===
def build_diverging_cmap(cmap_neg, cmap_pos):
    colors1 = plt.get_cmap(cmap_pos)(np.linspace(0, 1, 128))
    colors2 = plt.get_cmap(cmap_neg)(np.linspace(0, 1, 128))[::-1]
    white = np.ones((20, 4))
    return LinearSegmentedColormap.from_list('custom_div', np.vstack((colors2, white, colors1)))


def plot_panel(fig, ax, data, winds, title, letter, season='DJFM', ckey='c', latkey='lat', lonkey='lon', year_range=None, c_range=(-10, 40), levels=None):
    ax.text(0.02, 0.9, letter, fontsize=28, fontweight='bold', transform=ax.transAxes)

    # Concatenate westward and eastward components along phase speed 
    K = data #xr.concat([data['West'].sortby(ckey), data['East']], dim=ckey)

    if year_range is not None:
        K = K.sel(year=slice(*year_range))

    # Apply contour plot
    coslat = np.sqrt(np.cos(np.deg2rad(K['lat'])))
    # Only apply coslat weight once - it's already applied during data processing
    # Filter out very low phase speeds and take the mean over years
    K_plot = (1.0E03*K).mean('year').where(abs(K[ckey]) >= 0.05).sel({ckey: slice(*c_range)}).squeeze()
    
    #r = 1.0E06*K.polyfit(deg=1,dim='year').polyfit_coefficients.sel(degree=1)
    #tlevels = np.linspace(-abs(r.max()),abs(r.max())+0.1,16)
    #r.where(abs(r['c']) > 0.01).plot.contourf(cmap='RdBu_r',levels=tlevels)
    #p = K_plot.plot.contour(x=ckey, y=latkey, ax=ax, colors='k', linewidths=0.8, levels=levels)

    colors = [
        '#0038a8',  # dark blue
        '#268bd2',  # medium blue
        '#7eb6ff',  # light blue
        '#ffffff',  # white (center)
        '#ffeb7f',  # light yellow
        '#ffa500',  # orange
        '#d30000'   # red
    ]

    new_cmap = build_diverging_cmap('Blues', 'OrRd')
    new_levels = np.arange(-1.6, 1.61, 0.1)

    # Create the colormap
    chen_held_cmap = LinearSegmentedColormap.from_list('chen_held', colors, N=256)
    
    # Register the colormap so it can be accessed by name
    #plt.register_cmap(cmap=chen_held_cmap)

    #levels = np.arange(-0.1, 0.11,0.02)
    levels = [-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5]
    p = K_plot.plot.contourf(x=ckey, y=latkey, ax=ax, 
                             cmap=new_cmap,
                             extend='both',
                             levels=levels,
                             add_colorbar=False,
                             )

    c = K_plot.plot.contour(x=ckey, y=latkey, ax=ax, 
                             colors='k', 
                             #extend='both',
                             linewidths=0.7,
                             levels=[-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0.5,1,1.5,2,2.5,3,3.5],
                             add_colorbar=False,
                             )

    # Overlay zonal wind
    if 'lat' not in winds.dims:
        winds = winds.rename({'latitude':'lat','longitude':'lon'})

    (get_winds(winds,season=season).mean(lonkey)).plot(y=latkey, ax=ax, c='tab:purple', linewidth=2.5)

    pos = ax.get_position()
    panel_center = (pos.y1 + pos.y0) / 2
    
    if letter in ['a', 'e']:
        ax.text(pos.x0-0.08, panel_center, season, 
                transform=fig.transFigure, 
                rotation=90, 
                va='center',  
                ha='center', 
                weight='bold', 
                fontsize=18)
    

    Kmax = K.mean('year').where(K.mean('year') == (K.mean('year').where(K['c'] >= 1).max()),drop=True)
    Kmin = K.mean('year').where(K.mean('year') == (K.mean('year').where(K['c'] >= 1).min()),drop=True)

    ax.axvline(0, linewidth=0.8, c='k')
    if letter in ['a', 'b', 'c', 'd']:
        ax.set_title(f'{title}', weight='bold', fontsize=18)
    else:
        ax.set_title('', weight='bold', fontsize=16)
    
    if letter in ['a', 'e']:
        ax.set_ylabel('Latitude')
    else:
        ax.set_ylabel('')

    ax.set_xlim(c_range)
    ax.minorticks_on()
    ax.tick_params(axis='both', direction='in', length=5)
    if letter in ['a','b','c','d']:
        ax.set_xticks([-10, 0, 10, 20, 30, 40])
        ax.set_xticklabels([])
        ax.set_xlabel('')
    else:
        ax.set_xlabel('Phase speed (m/s)')
        ax.set_xticks([-10, 0, 10, 20, 30, 40])
        ax.set_xticklabels(['-10', '0', '10', '20', '30', '40'])
    ax.set_yticks([-80, -60, -40, -20, 0, 20, 40, 60, 80])
    ax.set_yticklabels(['80°S', '60°S', '40°S', '20°S', '0°', '20°N','40°N', '60°N', '80°N'])
    ax.set_ylim([-85,-5])

    return p, pos 

# %%
def plot_rh91_panels(dsets, specs, output_dir: Path):
    fig  = plt.figure(figsize=(15,10))

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 16

    gs = GridSpec(2, 4, wspace=0.25,hspace=0.07,top=0.95,bottom=0.07,left=0.1,right=0.9)

    levels = [-3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3]

    p, pos1 = plot_panel(fig, fig.add_subplot(gs[0,0]), specs['era5_DJFM'], dsets['u_era5'], 'ERA5', 'a', 'DJFM', year_range=(1981, 2014), levels=levels)
    plot_panel(fig, fig.add_subplot(gs[0,1]), specs['amip_DJFM'], dsets['u_amip'], 'AMIP', 'b', 'DJFM', year_range=(1981, 2014), levels=levels)
    plot_panel(fig, fig.add_subplot(gs[0,2]), specs['ace2_DJFM'], dsets['u_ace2'], 'ACE2-ERA5', 'c', 'DJFM', year_range=(1981, 2014), levels=levels)
    p, pos_top_right = plot_panel(fig, fig.add_subplot(gs[0,3]), specs['ngcm_DJFM'], dsets['u_ngcm'], 'NGCM2.8', 'd', 'DJFM', year_range=(1981, 2014), levels=levels)

    plot_panel(fig, fig.add_subplot(gs[1,0]), specs['era5_JJAS'], dsets['u_era5'], '', 'e', 'JJAS', year_range=(1981, 2014), levels=levels)
    plot_panel(fig, fig.add_subplot(gs[1,1]), specs['amip_JJAS'], dsets['u_amip'], '', 'f', 'JJAS', year_range=(1981, 2014), levels=levels)
    plot_panel(fig, fig.add_subplot(gs[1,2]), specs['ace2_JJAS'], dsets['u_ace2'], '', 'g', 'JJAS', year_range=(1981, 2014), levels=levels)
    p, pos_bottom_right = plot_panel(fig, fig.add_subplot(gs[1,3]), specs['ngcm_JJAS'], dsets['u_ngcm'], '', 'h', 'JJAS', year_range=(1981, 2014), levels=levels)

    cax = fig.add_axes([pos_top_right.x1 + 0.02, (pos_bottom_right.y1 + pos_bottom_right.y0)/3, 0.02,(pos_top_right.y1 + pos_top_right.y0)*0.55-(pos_bottom_right.y1 + pos_bottom_right.y0)*0.3])
    cbar = fig.colorbar(p, cax=cax, drawedges=True, orientation='vertical')
    cbar.set_label(r'$\mathrm{u^{\prime} v^{\prime}\ [m^{2}\ s^{-2} \cdot \Delta c^{-1}}]$', fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    out = output_dir / 'Figure_2-RH91-v2.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
# %%
def plot_emfc_panel(fig, ax, data, winds, title, letter, season='DJFM', ckey='c', latkey='lat', lonkey='lon', year_range=None, c_range=(-10, 40), levels=None):
    ax.text(0.01, 1.05, letter, fontsize=16, fontweight='bold', transform=ax.transAxes)

    # Apply contour plot
    coslat = np.sqrt(np.cos(np.deg2rad(data['lat'])))

    K = compute_emfc_rh91_vectorized(1.0E06*1.0E06*data*coslat)
 
    if year_range is not None:
        K = K.sel(year=slice(*year_range))

    # Only apply coslat weight once - it's already applied during data processing
    # Filter out very low phase speeds and take the mean over years
    K_plot = (K).mean('year').where(abs(K[ckey]) >= 0.05).sel({ckey: slice(*c_range)}).squeeze()
    
    #r = 1.0E06*K.polyfit(deg=1,dim='year').polyfit_coefficients.sel(degree=1)
    #tlevels = np.linspace(-abs(r.max()),abs(r.max())+0.1,16)
    #r.where(abs(r['c']) > 0.01).plot.contourf(cmap='RdBu_r',levels=tlevels)
    #p = K_plot.plot.contour(x=ckey, y=latkey, ax=ax, colors='k', linewidths=0.8, levels=levels)

    colors = [
        '#0038a8',  # dark blue
        '#268bd2',  # medium blue
        '#7eb6ff',  # light blue
        '#ffffff',  # white (center)
        '#ffeb7f',  # light yellow
        '#ffa500',  # orange
        '#d30000'   # red
    ]

    # Create the colormap
    chen_held_cmap = LinearSegmentedColormap.from_list('chen_held', colors, N=256)
    
    new_cmap = build_diverging_cmap('Blues', 'OrRd')
    new_levels = np.arange(-1.6, 1.61, 0.1)

    # Register the colormap so it can be accessed by name
    #plt.register_cmap(cmap=chen_held_cmap)

    #levels = np.arange(-0.1, 0.11,0.02)
    levels = [-0.1,-0.08,-0.06,-0.04,-0.02,0.02,0.04,0.06,0.08,0.1]
    p = K_plot.plot.contourf(x=ckey, y=latkey, ax=ax, cmap=new_cmap,
                             extend='both',
                             levels=new_levels,
                             add_colorbar=False,
                             )

    # Overlay zonal wind
    if 'lat' not in winds.dims:
        winds = winds.rename({'latitude':'lat','longitude':'lon'})

    coslat = (np.cos(np.deg2rad(winds['lat'])))
    (get_winds(winds,season=season).mean(lonkey)/coslat).plot(y=latkey, ax=ax,c='k')

    pos = ax.get_position()
    panel_center = (pos.y1 + pos.y0) / 2
    
    if letter in ['a', 'c', 'e', 'g']:
        ax.text(pos.x0-0.1, panel_center, title, 
                transform=fig.transFigure,  
                rotation=90, 
                va='center', 
                ha='center', 
                weight='bold', 
                fontsize=16)
    

    Kmax = K.mean('year').where(K.mean('year') == (K.mean('year').where(K['c'] >= 1).max()),drop=True)
    Kmin = K.mean('year').where(K.mean('year') == (K.mean('year').where(K['c'] >= 1).min()),drop=True)

    ax.axvline(0, linewidth=0.8, c='k')
    if letter == 'a':
        ax.set_title('DJFM', weight='bold', fontsize=16)
    elif letter == 'b':
        ax.set_title('JJAS', weight='bold', fontsize=16)
    else:
        ax.set_title('', weight='bold', fontsize=16)
    ax.set_xlabel('Phase speed (m/s)')
    if letter in ['a', 'c', 'e', 'g']:
        ax.set_ylabel('Latitude')
    else:
        ax.set_ylabel('')
    ax.set_xlim(c_range)
    ax.minorticks_on()
    ax.tick_params(axis='both', direction='in')
    ax.set_xticks([-10, 0, 10, 20, 30, 40])
    ax.set_yticks([-80, -60, -40, -20, 0, 20, 40, 60, 80])
    ax.set_yticklabels(['80°S', '60°S', '40°S', '20°S', '0°', '20°N','40°N', '60°N', '80°N'])
    ax.set_ylim([-85,-5])

    return p, pos 

# %%
def plot_emfc_panels(dsets, specs, output_dir: Path):
    fig  = plt.figure(figsize=(7.5,12))

    gs = GridSpec(4,2,wspace=0.2,hspace=0.35,top=0.95,bottom=0.08,left=0.15,right=0.85)

    levels = [-0.1,-0.08,-0.06,-0.04,-0.02,0.02,0.04,0.06,0.08,0.1]

    plot_emfc_panel(fig, fig.add_subplot(gs[0,0]), specs['era5_DJFM'], dsets['u_era5'], 'ERA5', 'a', 'DJFM', year_range=(1981, 2014), levels=levels)
    plot_emfc_panel(fig, fig.add_subplot(gs[1,0]), specs['amip_DJFM'], dsets['u_amip'], 'AMIP', 'c', 'DJFM', year_range=(1981, 2014), levels=levels)
    plot_emfc_panel(fig, fig.add_subplot(gs[2,0]), specs['ace2_DJFM'], dsets['u_ace2'], 'ACE2-ERA5', 'e', 'DJFM', year_range=(2002, 2009), levels=levels)
    p, pos0 = plot_emfc_panel(fig, fig.add_subplot(gs[3,0]), specs['ngcm_DJFM'], dsets['u_ngcm'], 'NGCM2.8', 'g', 'DJFM', year_range=(2018, 2014), levels=levels)

    plot_emfc_panel(fig, fig.add_subplot(gs[0,1]), specs['era5_JJAS'], dsets['u_era5'], '', 'b', 'JJAS', year_range=(1960, 2022), levels=levels)
    plot_emfc_panel(fig, fig.add_subplot(gs[1,1]), specs['amip_JJAS'], dsets['u_amip'], '', 'd', 'JJAS', year_range=(1981, 2014), levels=levels)
    plot_emfc_panel(fig, fig.add_subplot(gs[2,1]), specs['ace2_JJAS'], dsets['u_ace2'], '', 'f', 'JJAS', year_range=(1981, 2014), levels=levels)
    p, pos1 = plot_emfc_panel(fig, fig.add_subplot(gs[3,1]), specs['ngcm_JJAS'], dsets['u_ngcm'], '', 'h', 'JJAS', year_range=(1981, 2014), levels=levels)

    cax = fig.add_axes([pos1.x1 + 0.02, (pos1.y + pos1.y0)/2, 0.02, pos0.y1 - pos0.y0])
    cbar = fig.colorbar(p, cax=cax, orientation='vertical')
    cbar.set_label(r'$\mathrm{u^{\prime}$ (m/s$^2$)', fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    out = output_dir / 'Figure_2-CH07-v2.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def run_analysis(dsets):
    """Compute phase-speed spectra for each dataset and season."""
    specs = {}
    # ERA5
    specs['era5_DJFM'] = wrapper(dsets['u_era5'], dsets['v_era5'], years=np.arange(1981, 2015, 1), season='DJFM')
    specs['era5_JJAS'] = wrapper(dsets['u_era5'], dsets['v_era5'], years=np.arange(1981, 2015, 1), season='JJAS')
    # ACE2
    specs['ace2_DJFM'] = wrapper(dsets['u_ace2'], dsets['v_ace2'], years=np.arange(1981, 2015, 1), season='DJFM')
    specs['ace2_JJAS'] = wrapper(dsets['u_ace2'], dsets['v_ace2'], years=np.arange(1981, 2015, 1), season='JJAS')
    # AMIP
    specs['amip_DJFM'] = wrapper(dsets['u_amip'], dsets['v_amip'], years=np.arange(1981, 2015, 1), season='DJFM')
    specs['amip_JJAS'] = wrapper(dsets['u_amip'], dsets['v_amip'], years=np.arange(1981, 2015, 1), season='JJAS')
    # NGCM
    specs['ngcm_DJFM'] = wrapper(dsets['u_ngcm'], dsets['v_ngcm'], years=np.arange(1981, 2015, 1), season='DJFM')
    specs['ngcm_JJAS'] = wrapper(dsets['u_ngcm'], dsets['v_ngcm'], years=np.arange(1981, 2015, 1), season='JJAS')
    return specs


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    default_data = (here / '../../../Circulation-Benchmarking-AI-Emulators-data')
    default_out = (here / '../../plots')
    data_dir = resolve_dir(args.data_dir, default_data)
    output_dir = resolve_dir(args.output_dir, default_out)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Data dir: {data_dir}')
    print(f'Output dir: {output_dir}')

    dsets = read_datasets(data_dir, ace2_index=args.ace2_index, ngcm_index=args.ngcm_index)
    specs = run_analysis(dsets)

    rh91_path = plot_rh91_panels(dsets, specs, output_dir)
    print(f'Saved RH91 figure: {rh91_path}')
    ch07_path = plot_emfc_panels(dsets, specs, output_dir)
    print(f'Saved CH07 figure: {ch07_path}')


if __name__ == '__main__':
    main()
# %%

