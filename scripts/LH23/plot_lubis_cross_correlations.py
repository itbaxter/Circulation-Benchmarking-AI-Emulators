# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import glob as glob
from scipy.signal import correlate
from matplotlib.gridspec import GridSpec
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# %%

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
def plot_panel(ax, data_cross, era5_cross, member_cross, color, label, 
               correlation_type='z1z1', lag=120, panel_label='a', show_leads=False):
    """
    Plot a single panel of cross-correlations or autocorrelations.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    data_cross : xarray.Dataset
        Full dataset with member_id dimension.
    era5_cross : xarray.Dataset
        ERA5 reference data.
    member_cross : xarray.Dataset
        Ensemble mean or processed data.
    color : str
        Color for the model line.
    label : str
        Label for the model.
    correlation_type : str
        Either 'z1z1' or 'z2z1'.
    lag : int
        Maximum lag to plot.
    panel_label : str
        Panel label (a, b, c, etc.).
    show_leads : bool
        Whether to show "z1 leads" and "z2 leads" annotations.
    """
    # Plot individual members in gray
    for i in range(len(data_cross.member_id)):
        data_cross[correlation_type].sel(lag=slice(-lag, lag)).isel(member_id=i).plot.line(
            x='lag', c='silver', add_legend=False, ax=ax
        )

    # Plot ERA5
    era5_cross[correlation_type].sel(lag=slice(-lag, lag)).plot(
        c='k', linewidth=2.5, label='ERA5', ax=ax
    )

    # Plot ensemble mean
    member_cross[correlation_type].mean('member_id').sel(lag=slice(-lag, lag)).plot(
        c=color, linewidth=1.5 if color != '#1E88E5' else 2.5, 
        linestyle='-', label=label, ax=ax
    )

    # Formatting
    title = f'{label} $\\mathrm{{{correlation_type[:-2]}_{{{correlation_type[-2]}}}{correlation_type[-1]}}}$'
    if correlation_type == 'z1z1':
        title = f'{label} $\\mathrm{{z_{{1}}}}$ autocorrelation'
    
    ax.set_title(title)
    ax.set_xlabel('Lag (days)' if 'NGCM' in label else '')
    ax.axhline(0, linestyle='--', c='k', linewidth=0.7)
    ax.axvline(0, linestyle='--', c='k', linewidth=0.7)
    ax.set_xlim([-40, 40])
    
    if correlation_type == 'z2z1':
        ax.set_ylim([-0.25, 0.25])
        if show_leads:
            ax.text(0.12, 0.8, 'z2 leads', transform=ax.transAxes, fontsize=12, va='top')
            ax.text(0.65, 0.2, 'z1 leads', transform=ax.transAxes, fontsize=12, va='top')
    else:
        ax.set_ylim([-0.1, 1.0])
    
    ax.text(-0.1, 1.15, panel_label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    ax.minorticks_on()
    ax.legend(fontsize=10, frameon=False)


# %%
def create_lubis_cross_correlations(era5_cross, amip_cross, ace2_cross, ngcm_cross, 
                                    amip, ace2, ngcm, save_path=None):
    """
    Create the full Lubis cross-correlation figure.
    
    Parameters:
    -----------
    era5_cross : xarray.Dataset
        ERA5 cross-correlations.
    amip_cross : xarray.Dataset
        AMIP cross-correlations with member_id dimension.
    ace2_cross : xarray.Dataset
        ACE2 cross-correlations with member_id dimension.
    ngcm_cross : xarray.Dataset
        NeuralGCM cross-correlations with member_id dimension.
    amip, ace2, ngcm : xarray.Dataset
        Original datasets (for accessing member info).
    save_path : str, optional
        Path to save the figure.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    fig = plt.figure(figsize=(7.5, 9))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25, 
                  top=0.94, bottom=0.05, left=0.08, right=0.98)

    lag = 120

    # AMIP panels
    ax_amip_z1 = fig.add_subplot(gs[0, 0])
    plot_panel(ax_amip_z1, amip_cross, era5_cross, amip_cross, 'tab:orange', 
               'AMIP', 'z1z1', lag, 'a')

    ax_amip_z2z1 = fig.add_subplot(gs[0, 1])
    plot_panel(ax_amip_z2z1, amip_cross, era5_cross, amip_cross, 'tab:orange', 
               'AMIP', 'z2z1', lag, 'b', show_leads=True)

    # ACE2 panels
    ax_ace2_z1 = fig.add_subplot(gs[1, 0])
    plot_panel(ax_ace2_z1, ace2_cross, era5_cross, ace2_cross, '#D81B60', 
               'ACE2-ERA5', 'z1z1', lag, 'c')

    ax_ace2_z2z1 = fig.add_subplot(gs[1, 1])
    plot_panel(ax_ace2_z2z1, ace2_cross, era5_cross, ace2_cross, '#D81B60', 
               'ACE2-ERA5', 'z2z1', lag, 'd', show_leads=True)

    # NGCM panels
    ax_ngcm_z1 = fig.add_subplot(gs[2, 0])
    plot_panel(ax_ngcm_z1, ngcm_cross, era5_cross, ngcm_cross, '#1E88E5', 
               'NGCM2.8', 'z1z1', lag, 'e')

    ax_ngcm_z2z1 = fig.add_subplot(gs[2, 1])
    plot_panel(ax_ngcm_z2z1, ngcm_cross, era5_cross, ngcm_cross, '#1E88E5', 
               'NGCM2.8', 'z2z1', lag, 'f', show_leads=True)

    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    
    return fig

def preprocess(file):
        eof_results = xr.open_dataset(file, decode_times=False).squeeze()
        eof_results.coords['time'] = pd.date_range('1981-01-01', periods=len(eof_results.time), freq='D')
        z1 = eof_results['z1'] #.dropna('time')
        z1.coords['mode'] = [0,1]
        z1 = z1.where( z1 > -500, drop=True)
        return z1

def main():
    """Main execution function."""
    directory = './' 
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
                                         amip, ace2, ngcm, save_path)
    plt.show()
    
    print(f"Figure saved to {save_path}")


# %%
if __name__ == '__main__':
    main()


# %%
