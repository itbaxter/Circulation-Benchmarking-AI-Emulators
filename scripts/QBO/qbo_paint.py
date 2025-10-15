# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
#from eofs.xarray import Eof
from scipy.signal import periodogram, welch
import psutil  # For memory profiling
#from eofs.xarray import Eof
import glob as glob
from scipy.signal import periodogram
from scipy.interpolate import interp1d
from scipy.signal import correlate
from matplotlib.gridspec import GridSpec
#from eofs.xarray import Eof
from scipy.signal import detrend

import matplotlib.path as mpath
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.ticker import ScalarFormatter,AutoLocator,MultipleLocator,AutoMinorLocator,FixedLocator
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec

# %%

files = sorted(glob.glob('/project2/tas1/itbaxter/NeuralGCM_Decadal_Simulations/data/processed/QBO/qbo_mon_levels*.nc'))
print(files)    

csp01 = xr.open_mfdataset(files,combine='nested',concat_dim='member_id')
csp01 = csp01.rename({'__xarray_dataarray_variable__':'u'})
csp01
# %%
p = csp01['u'].sel(time=slice('2018-01-01','2023-12-31')).plot.contourf(col='member_id',col_wrap=5,
                                                                        x='time',cmap='RdBu_r',
                                                                        levels=np.arange(-45,45.1,5),
                                                                        yincrease=False)
for ax in p.axes.flat:
    ax.set_yscale('log')
    ax.axhline(50,c='k',linewidth=0.5,linestyle='--')

# %%
files = sorted(glob.glob('/scratch/midway3/itbaxter/NeuralGCM_Decadal_Simulations/ERA5/ua/*monmean.nc'))
files

def area_weighted_ave(ds):
    """Calculate area-weighted average."""
    weights = np.cos(np.deg2rad(ds.lat))
    return (ds * weights).mean(['lat', 'lon']) / weights.mean()

def preprocess(ds):
    u_wind = ds['u'].squeeze()
    #u_wind.coords['time'] = u_wind['valid_time']
    #u_wind = u_wind.drop_vars(['valid_time','init_time'])
    #u_wind = u_wind.sel(time=slice('1995-01-01','2011-12-31'))
    return area_weighted_ave(u_wind.sel(latitude=slice(15,-15)).rename({'latitude':'lat','longitude':'lon'}))

era5 = xr.open_mfdataset(files[12*(1979-1958):],combine='nested',concat_dim='time',preprocess=preprocess,engine='netcdf4')
#era5 = era5.rename({'__xarray_dataarray_variable__':'u'})
era5

# %%
era5 = era5.compute()

# %%
# AMIP

#files = sorted(glob.glob('/project/tas1/itbaxter/for-tiffany/amip/121x240/ua/ua*nc'))
files = sorted(glob.glob('/scratch/midway3/itbaxter/NeuralGCM_Decadal_Simulations/amip/64x128/ua/ua*nc'))
files

#%%

def preprocess_amip(ds):
    u_wind = ds['ua'].squeeze() #resample(time='1MS').mean('time').squeeze()
    #u_wind.coords['time'] = u_wind['valid_time']
    #u_wind = u_wind.drop_vars(['valid_time','init_time'])
    #u_wind = u_wind.sel(time=slice('1995-01-01','2011-12-31'))
    return area_weighted_ave(u_wind.sel(lat=slice(-15,15)))

amip = xr.open_dataset(files[20])
amip = preprocess_amip(amip) #.isel(member_id=1)
amip.coords['plev'] = amip.coords['plev'] / 100  # Convert to hPa
amip.coords['time'] = amip['time'].astype('datetime64[M]')
amip

# %%
p = csp01['u'].mean('member_id').sel(time=slice('2018-01-01','2022-12-31')).plot.contourf(yincrease=False,levels=np.arange(-75,75.1,5),
                                                                        x='time',cmap='RdBu_r',
                                                                        )
p.axes.set_yscale('log')

# %%
p = era5.sel(time=slice('2018-01-01','2022-12-31')).plot.contourf(yincrease=False,levels=np.arange(-75,75.1,5),
                                                                        x='time',cmap='RdBu_r',
                                                                        )
p.axes.set_yscale('log')

# %%
p = amip.plot.contourf(yincrease=False,levels=np.arange(-75,75.1,5),
                                                                        x='time',cmap='RdBu_r',
                                                                        )
p.axes.set_yscale('log')

# %%
fig = plt.figure(figsize=(7.5, 6))
gs = GridSpec(3, 1, height_ratios=[1, 1, 1],
               hspace=0.4)  # Adjust hspace for spacing between subplots    

# === Custom diverging colormaps ===
def build_diverging_cmap(cmap_neg, cmap_pos):
    colors1 = plt.get_cmap(cmap_pos)(np.linspace(0, 1, 128))
    colors2 = plt.get_cmap(cmap_neg)(np.linspace(0, 1, 128))[::-1]
    white = np.ones((20, 4))
    return LinearSegmentedColormap.from_list('custom_div', np.vstack((colors2, white, colors1)))

# === Plot Q2m trends (left column) ===
q2m_cmap = build_diverging_cmap('Blues', 'OrRd')
q2m_levels = np.arange(-45, 45.1, 5)

ax2 = fig.add_subplot(gs[0])
era5.sel(time=slice('1980-01-01','2022-12-31')).sel(level=slice(0,200)).plot.contourf(yincrease=False,levels=q2m_levels,
                                                                        x='time',cmap=q2m_cmap,
                                                                        add_colorbar=False,
                                                                        #cbar_kwargs={'label': 'Zonal Wind Speed (m/s)'},
                                                                        ax=ax2)

ax2.text(-0.1, 1.06, 'a', fontsize=18, fontweight='bold', transform=ax2.transAxes)
ax2.set_yscale('log')
ax2.axhline(50,c='k',linewidth=0.5,linestyle='--')
ax2.set_xlabel('')
ax2.set_ylabel('Pressure (hPa)')
ax2.set_title('ERA5')
# Custom y-axis labels for log scale
ax2.set_ylim([200, 0])
ax2.set_xlim([np.datetime64('1980-01-01'), np.datetime64('2022-12-31')])
ax2.set_yticks([200., 100., 50., 20., 10., 5., 1.])
ax2.set_yticklabels([200, 100, 50, 20, 10, 5, 1])
ax2.minorticks_off()
ax2.xaxis.minorticks_on()

ax3 = fig.add_subplot(gs[1])
#amip.sel(time=slice('1980-01-01','2022-12-31')).sel(plev=slice(100,0)).plot.contourf(yincrease=False,levels=np.arange(-45,45.1,5),
#                                                                        x='time',cmap='RdBu_r',
#                                                                        cbar_kwargs={'label': 'Zonal Wind Speed (m/s)'},
#                                                                        ax=ax3)

p = ax3.contourf(amip['time'], amip['plev'].sel(plev=slice(200,0)), amip.sel(plev=slice(200,0)).T, 
                 levels=q2m_levels, 
                 cmap=q2m_cmap)

ax3.text(-0.1, 1.06, 'b', fontsize=18, fontweight='bold', transform=ax3.transAxes)
ax3.invert_yaxis()
ax3.set_yscale('log')
ax3.axhline(50,c='k',linewidth=0.5,linestyle='--')
ax3.set_xlabel('')
ax3.set_ylabel('Pressure (hPa)')
model = str(amip['member_id'].values).split('_')[0]
print(model)
ax3.set_title(f'AMIP: {model}')
ax3.set_ylim([200,0])
ax3.set_xlim([np.datetime64('1980-01-01'), np.datetime64('2022-12-31')])
ax3.set_yticks([200., 100., 50., 20., 10., 5., 1.])
ax3.set_yticklabels([200, 100, 50, 20, 10, 5, 1])
ax3.minorticks_off()
ax3.xaxis.minorticks_on()

ax1 = fig.add_subplot(gs[2])
p = csp01['u'].mean('member_id').sel(time=slice('1980-01-01','2022-12-31')).sel(level=slice(0,200)).plot.contourf(yincrease=False,
                                                                                                                  levels=q2m_levels,
                                                                        x='time',cmap=q2m_cmap,
                                                                        add_colorbar=False,
                                                                        #cbar_kwargs={'label': 'Zonal Wind Speed (m/s)'},
                                                                        ax=ax1)

ax1.text(-0.1, 1.06, 'c', fontsize=18, fontweight='bold', transform=ax1.transAxes)
ax1.set_yscale('log')
ax1.axhline(50,c='k',linewidth=0.5,linestyle='--')
ax1.set_xlabel('Time (months)')
ax1.set_ylabel('Pressure (hPa)')
ax1.set_title('NGCM2.8')
# Custom y-axis labels for log scale
ax1.set_ylim([200, 0])
ax1.set_xlim([np.datetime64('1980-01-01'), np.datetime64('2022-12-31')])
ax1.set_yticks([200., 100., 50., 20., 10., 5., 1.])
ax1.set_yticklabels([200, 100, 50, 20, 10, 5, 1])
ax1.minorticks_off()
ax1.xaxis.minorticks_on()
ax1.axvline(np.datetime64('2018-01-01'), c='k', linewidth=0.5, linestyle='--')  # Vertical line at 2018-01-01

pos1 = ax1.get_position()  # Get the position of the third subplot
pos2 = ax2.get_position()  # Get the position of the second subplot

cb = plt.colorbar(p, ax=[ax1, ax2, ax3], orientation='vertical', drawedges=True, fraction=0.02, pad=0.02)
cb.ax.set_ylabel('Zonal Wind Speed (m/s)')

plt.savefig('/scratch/midway2/itbaxter/NeuralGCM_Decadal_Simulations/plots/qbo_comparison.png',dpi=300,bbox_inches='tight')

# %%
files = glob.glob('/project/tas1/ockham/data1/data1/tas/climate_2.8_csp_pe/*zarr')
files


