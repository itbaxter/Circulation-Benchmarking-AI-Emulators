import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import argparse
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

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
        default='/project/tas1/itbaxter/aimip/benchmarking_ai_variability/Circulation-Benchmarking-AI-Emulators-data',
        help='Base directory containing input data (default: /project/tas1/itbaxter/aimip/benchmarking_ai_variability/Circulation-Benchmarking-AI-Emulators-data)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../../plots',
        help='Directory to save output plots (default: ../../plots)'
    )
    
    return parser.parse_args()

def area_weighted_ave(ds):
    """Calculate area-weighted average over lat/lon with cosine weighting."""
    weights = np.cos(np.deg2rad(ds.lat))
    return (ds * weights).mean(['lat', 'lon']) / weights.mean()


def preprocess(ds):
    """Preprocess ERA5-style dataset to equatorial mean (15S-15N)."""
    u_wind = ds['u'].squeeze()
    return area_weighted_ave(
        u_wind.sel(latitude=slice(15, -15)).rename({'latitude': 'lat', 'longitude': 'lon'})
    )


def preprocess_amip(ds):
    """Preprocess AMIP dataset to equatorial mean (15S-15N)."""
    u_wind = ds['ua'].squeeze()
    return area_weighted_ave(u_wind.sel(lat=slice(-15, 15)))


def build_diverging_cmap(cmap_neg, cmap_pos):
    """Build a custom diverging colormap with a white center band."""
    colors1 = plt.get_cmap(cmap_pos)(np.linspace(0, 1, 128))
    colors2 = plt.get_cmap(cmap_neg)(np.linspace(0, 1, 128))[::-1]
    white = np.ones((20, 4))
    return LinearSegmentedColormap.from_list('custom_div', np.vstack((colors2, white, colors1)))

def qbo_comparison(data_dir: str, output_dir: str):
    """
    Load NGCM/ERA5/AMIP equatorial zonal wind data and create a 3-panel comparison plot.

    Parameters:
    - data_dir: Base directory containing input data
    - output_file: Path to save the resulting PNG

    Returns: Path to saved figure
    """
    csp_pattern: str = f'{data_dir}/ngcm/eq_uwind/ngcm_qbo_mon_levels*.nc'
    #era5_glob: str = f'{data_dir}/era5/eq_uwind/*monmean.nc'
    era5_glob: str = f'/scratch/midway3/itbaxter/NeuralGCM_Decadal_Simulations/ERA5/ua/*nc'
    amip_glob: str = f'{data_dir}/amip/eq_uwind/*monmean.nc'
    amip_index: int = 20
    output_file: str = f'{output_dir}/qbo_comparison.png'

    print('Reading in files...')
    # NGCM2.8/ACE2 stack
    csp_files = sorted(glob.glob(csp_pattern))
    if len(csp_files) == 0:
        raise FileNotFoundError(f"No NGCM/ACE2 files matched pattern: {csp_pattern}")
    csp01 = xr.open_mfdataset(csp_files, combine='nested', concat_dim='member_id')
    csp01 = csp01.rename({'__xarray_dataarray_variable__': 'u'})

    # ERA5
    era5_files = sorted(glob.glob(era5_glob))
    if len(era5_files) == 0:
        raise FileNotFoundError(f"No ERA5 files matched glob: {era5_glob}")
    era5 = xr.open_mfdataset(
        era5_files[12 * (1979 - 1958):], combine='nested', concat_dim='time', preprocess=preprocess, engine='netcdf4'
    )
    era5 = era5.compute()

    # AMIP
    amip_files = sorted(glob.glob(amip_glob))
    if len(amip_files) == 0:
        raise FileNotFoundError(f"No AMIP files matched glob: {amip_glob}")
    if amip_index >= len(amip_files):
        raise IndexError(f"amip_index {amip_index} out of range for {len(amip_files)} files")
    amip = xr.open_dataset(amip_files[amip_index])
    amip = preprocess_amip(amip)
    amip.coords['plev'] = amip.coords['plev'] / 100  # Convert to hPa
    amip.coords['time'] = amip['time'].astype('datetime64[M]')

    print('Creating plot...')
    # Plot
    fig = plt.figure(figsize=(7.5, 6))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.4)

    # Colormap
    q2m_cmap = build_diverging_cmap('Blues', 'OrRd')
    q2m_levels = np.arange(-45, 45.1, 5)

    # Panel a: ERA5
    ax2 = fig.add_subplot(gs[0])
    era5.sel(time=slice('1980-01-01', '2022-12-31')).sel(level=slice(0, 200)).plot.contourf(
        yincrease=False, levels=q2m_levels, x='time', cmap=q2m_cmap, add_colorbar=False, ax=ax2
    )
    ax2.text(-0.1, 1.06, 'a', fontsize=18, fontweight='bold', transform=ax2.transAxes)
    ax2.set_yscale('log')
    ax2.axhline(50, c='k', linewidth=0.5, linestyle='--')
    ax2.set_xlabel('')
    ax2.set_ylabel('Pressure (hPa)')
    ax2.set_title('ERA5')
    ax2.set_ylim([200, 0])
    ax2.set_xlim([np.datetime64('1980-01-01'), np.datetime64('2022-12-31')])
    ax2.set_yticks([200., 100., 50., 20., 10., 5., 1.])
    ax2.set_yticklabels([200, 100, 50, 20, 10, 5, 1])
    ax2.minorticks_off()
    ax2.xaxis.minorticks_on()

    # Panel b: AMIP (single member)
    ax3 = fig.add_subplot(gs[1])
    p = ax3.contourf(
        amip['time'], amip['plev'].sel(plev=slice(200, 0)), amip.sel(plev=slice(200, 0)).T,
        levels=q2m_levels, cmap=q2m_cmap
    )
    ax3.text(-0.1, 1.06, 'b', fontsize=18, fontweight='bold', transform=ax3.transAxes)
    ax3.invert_yaxis()
    ax3.set_yscale('log')
    ax3.axhline(50, c='k', linewidth=0.5, linestyle='--')
    ax3.set_xlabel('')
    ax3.set_ylabel('Pressure (hPa)')
    model = str(amip['member_id'].values).split('_')[0]
    ax3.set_title(f'AMIP: {model}')
    ax3.set_ylim([200, 0])
    ax3.set_xlim([np.datetime64('1980-01-01'), np.datetime64('2022-12-31')])
    ax3.set_yticks([200., 100., 50., 20., 10., 5., 1.])
    ax3.set_yticklabels([200, 100, 50, 20, 10, 5, 1])
    ax3.minorticks_off()
    ax3.xaxis.minorticks_on()

    # Panel c: NGCM2.8 mean
    ax1 = fig.add_subplot(gs[2])
    p = csp01['u'].mean('member_id').sel(time=slice('1980-01-01', '2022-12-31')).sel(level=slice(0, 200)).plot.contourf(
        yincrease=False, levels=q2m_levels, x='time', cmap=q2m_cmap, add_colorbar=False, ax=ax1
    )
    ax1.text(-0.1, 1.06, 'c', fontsize=18, fontweight='bold', transform=ax1.transAxes)
    ax1.set_yscale('log')
    ax1.axhline(50, c='k', linewidth=0.5, linestyle='--')
    ax1.set_xlabel('Time (months)')
    ax1.set_ylabel('Pressure (hPa)')
    ax1.set_title('NGCM2.8')
    ax1.set_ylim([200, 0])
    ax1.set_xlim([np.datetime64('1980-01-01'), np.datetime64('2022-12-31')])
    ax1.set_yticks([200., 100., 50., 20., 10., 5., 1.])
    ax1.set_yticklabels([200, 100, 50, 20, 10, 5, 1])
    ax1.minorticks_off()
    ax1.xaxis.minorticks_on()
    ax1.axvline(np.datetime64('2018-01-01'), c='k', linewidth=0.5, linestyle='--')

    # Shared colorbar
    cb = plt.colorbar(p, ax=[ax1, ax2, ax3], orientation='vertical', drawedges=True, fraction=0.02, pad=0.02)
    cb.ax.set_ylabel('Zonal Wind Speed (m/s)')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    return output_file

if __name__ == '__main__':
    args = parse_args()
    out = qbo_comparison(data_dir=args.data_dir, output_dir=args.output_dir),
    print(f"Saved QBO comparison figure to: {out}")


