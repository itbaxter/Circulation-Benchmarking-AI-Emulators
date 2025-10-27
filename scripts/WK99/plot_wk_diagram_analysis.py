"""
Uses code from Brian Mederios: https://github.com/brianpm/wavenumber_frequency
"""

# %%
import xarray as xr
# our local module:
import wavenumber_frequency_functions as wf
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob as glob
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

def plot_colorbar(img, pos0, pos1):
        pos = ax.get_position()
        #cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.05, pos.width, 0.01])  # [left, bottom, width, height]
        cbar_ax = fig.add_axes([pos0.x1 + 0.10, pos1.y0, 0.01, pos0.y1 - pos1.y0]) 
        cb = fig.colorbar(img, cax=cbar_ax, label='Normalized Power', orientation='vertical', pad=0.15)

def plot_normalized_symmetric_spectrum(s, ax, panel, label='ERA5', ylabel='None', ofil=None, cbar=False):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""
    fb = [0.01, 0.5]  # avoid zero for log-scale period

    # Get data for dispersion curves
    swfreq, swwn = wf.genDispersionCurves()
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)


    # Clean and subset data
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20, 20))
    z = z.where(z['frequency'] > 0)  # mask freq=0 to avoid divide-by-zero

    # Add tropical wave regions BEFORE plotting contours
    add_tropical_wave_regions(ax, component='sym', alpha=0.15, text_size=8, show_labels=True)

    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z, levels=np.arange(1.1, 1.71, 0.1), cmap='YlOrRd', extend='max')

    # Plot dispersion curves
    c = 'darkgray'
    for ii in range(3, 6):
        ax.plot(swk[ii, 0, :], swf[ii, 0, :], color=c)
        ax.plot(swk[ii, 1, :], swf[ii, 1, :], color=c)
        ax.plot(swk[ii, 2, :], swf[ii, 2, :], color=c)

    ax.axhline(1/60, linestyle='dashed', color='lightgray')
    ax.axhline(1/20, linestyle='dashed', color='lightgray')
    ax.axhline(1/5, linestyle='dashed', color='lightgray')
    ax.axhline(1/2.5, linestyle='dashed', color='lightgray')
    ax.axvline(0, linestyle='dashed', color='lightgray')
    ax.set_xlim([-20, 20])
    ax.set_xticks([-20,-15,-10,-5,0,5,10,15,20])
    ax.set_ylim(fb)
    ax.set_ylabel('Frequency (cpd)')
    ax.set_xlabel('Zonal Wavenumber')
#    ax.text("Normalized Symmetric Component")
    #if ylabel != 'None':
    #    ax.text(-0.22, 0.5, 'Normalized Symmetric Component', transform=ax.transAxes,
    #        rotation=90, va='center', ha='center', weight='bold', fontsize=12)
    ax.set_title(label,weight='bold', fontsize=16)
    #ax.text("Normalized Symmetric Component")
    
    #if cbar == True:
        #plt.colorbar(img, ax=ax, label='Normalized Power',pad=0.15)
    #    plot_colorbar(img, ax)

    ax.text(0.01,1.05,panel,fontsize=16,
            fontweight='bold',
            transform=ax.transAxes)

    # Add secondary y-axis for period (days)
    def freq_to_period(freq):
        return np.where(freq > 0, 1 / freq, np.nan)

    def period_to_freq(period):
        return np.where(period > 0, 1 / period, np.nan)

    secax = ax.secondary_yaxis('right', functions=(freq_to_period, freq_to_period))
    secax.set_ylabel('Period (days)')
    #secax.set_yscale('log')
    secax.set_yticks([1.25,1.33,1.43,1.54,1.67,1.82,2.00,2.50,3.33,5.00,10.0,20][::-1])
    secax.set_yticklabels([1.25,1.33,1.43,1.54,1.67,1.82,2.00,2.50,3.33,5.00,10.0,20][::-1])

    # Manually set safe limits
    min_freq = max(fb[0], 1e-3)
    max_freq = fb[1]
    period_ylim = freq_to_period(np.array([max_freq, min_freq]))  # reverse for log scale
    secax.set_ylim(period_ylim)

    if cbar == True:
        return img, ax.get_position()



def plot_normalized_asymmetric_spectrum(s, ax, panel, label='ERA5', ylabel='None',ofil=None, cbar=False):
    """Plot of normalized anti-symmetric power spectrum with shallow water curves."""

    fb = [0.01, 0.5]  # avoid zero to prevent divide-by-zero

    # Get data for dispersion curves
    swfreq, swwn = wf.genDispersionCurves()
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    # Clean and subset data
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20, 20))
    z = z.where(z['frequency'] > 0)  # mask freq=0 to avoid divide-by-zero

    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z, levels=np.arange(1.1, 1.71, 0.1), cmap='YlOrRd', extend='max')

    # Plot dispersion curves
    c = 'darkgray'
    for ii in range(0, 3):
        ax.plot(swk[ii, 0, :], swf[ii, 0, :], color=c)
        ax.plot(swk[ii, 1, :], swf[ii, 1, :], color=c)
        ax.plot(swk[ii, 2, :], swf[ii, 2, :], color=c)

    ax.axhline(1/60, linestyle='dashed', color='lightgray')
    ax.axhline(1/20, linestyle='dashed', color='lightgray')
    ax.axhline(1/5, linestyle='dashed', color='lightgray')
    ax.axhline(1/2.5, linestyle='dashed', color='lightgray')
    ax.axvline(0, linestyle='dashed', color='lightgray')
    ax.set_xlim([-20, 20])
    ax.set_xticks([-20,-15,-10,-5,0,5,10,15,20])
    ax.set_ylim(fb)
    ax.set_ylabel('Frequency (cpd)')
    ax.set_xlabel('Zonal Wavenumber')
    #if ylabel != 'None':
    #    ax.text(-0.22, 0.5, 'Normalized Anti-Symmetric Component', transform=ax.transAxes,
    #        rotation=90, va='center', ha='center', weight='bold', fontsize=12)
    ax.set_title(label,weight='bold', fontsize=16)
    
    #if cbar == True:
        #plt.colorbar(img, ax=ax, label='Normalized Power',pad=0.15)
    #    plot_colorbar(img, ax)
        #if i == 1:
        #    cb.ax.set_ylabel('Meridional moisture transport trend (kg/m2/s/41yr)')

    ax.text(0.01,1.05,panel,fontsize=16,
            fontweight='bold',
            transform=ax.transAxes)

    # Add secondary y-axis for period (days)
    def freq_to_period(freq):
        return np.where(freq > 0, 1 / freq, np.nan)

    def period_to_freq(period):
        return np.where(period > 0, 1 / period, np.nan)

    secax = ax.secondary_yaxis('right', functions=(freq_to_period, freq_to_period))
    secax.set_ylabel('Period (days)')
    #secax.set_yscale('log')
    secax.set_yticks([1.25,1.33,1.43,1.54,1.67,1.82,2.00,2.50,3.33,5.00,10.0,20][::-1])
    secax.set_yticklabels([1.25,1.33,1.43,1.54,1.67,1.82,2.00,2.50,3.33,5.00,10.0,20.0][::-1])

    # Set matching limits safely
    min_freq = max(fb[0], 1e-3)
    max_freq = fb[1]
    period_ylim = freq_to_period(np.array([max_freq, min_freq]))  # reversed for log axis
    secax.set_ylim(period_ylim)

    #if ofil is not None:
    #    fig.savefig(ofil, bbox_inches='tight', dpi=144)
    if cbar == True:
        return img, ax.get_position()

def plot_normalized_background_spectrum(s, ax, panel, label='ERA5', ylabel='None', ofil=None, cbar=False):
    """
    Plot the background spectrum of tropical precipitation.
    
    Parameters:
    -----------
    s : xarray.DataArray
        The background spectrum data
    ax : matplotlib.axes.Axes
        The axes to plot on
    panel : str
        Panel label (a, b, c, etc.)
    label : str
        Dataset label (ERA5, ACE2-ERA5, NGCM2.8)
    ylabel : str
        Y-axis label
    ofil : str, optional
        Output file path
    cbar : bool
        Whether to add a colorbar
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    fb = [0.01, 0.5]  # avoid zero for log-scale period

    # Clean and subset data
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20, 20))
    z = z.where(z['frequency'] > 0)  # mask freq=0 to avoid divide-by-zero

    #z = z / z.max()  # Normalize the spectrum

    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z, levels=np.arange(0, 1.4E-11, 0.2E-11), cmap='YlOrRd', extend='max')
    cl = ax.contour(kmesh0, vmesh0, z, levels=np.arange(1.1, 1.71, 0.1), colors='k', linewidths=0.4)

    # Reference lines
    ax.axhline(1/60, linestyle='dashed', color='lightgray', lw=0.7, alpha=0.7)
    ax.axhline(1/20, linestyle='dashed', color='lightgray', lw=0.7, alpha=0.7)
    ax.axhline(1/5, linestyle='dashed', color='lightgray', lw=0.7, alpha=0.7)
    ax.axhline(1/2.5, linestyle='dashed', color='lightgray', lw=0.7, alpha=0.7)
    ax.axvline(0, linestyle='dashed', color='lightgray', lw=0.7, alpha=0.7)

    # Add reference lines for specific wavenumbers
    for k in [-10, -5, 5, 10]:
        ax.axvline(k, linestyle='dotted', color='lightgray', alpha=0.7)
    
    ax.set_xlim([-20, 20])
    ax.set_xticks([-20,-15,-10,-5,0,5,10,15,20])
    ax.set_ylim(fb)
    ax.set_ylabel('Frequency (CPD)')
    ax.set_xlabel('Zonal Wavenumber')
    
    #if ylabel != 'None':
    #    ax.text(-0.22, 0.5, ylabel, transform=ax.transAxes,
    #        rotation=90, va='center', ha='center', weight='bold', fontsize=12)
    
    ax.set_title(label, weight='bold', fontsize=16)
    
    #if cbar:
        # This function should be defined in the main script
    #    plot_colorbar(img, ax)

    ax.text(0.01, 1.02, panel, fontsize=16,
            fontweight='bold',
            transform=ax.transAxes)

    # Add secondary y-axis for period (days)
    def freq_to_period(freq):
        return np.where(freq > 0, 1 / freq, np.nan)

    def period_to_freq(period):
        return np.where(period > 0, 1 / period, np.nan)

    secax = ax.secondary_yaxis('right', functions=(freq_to_period, freq_to_period))
    secax.set_ylabel('Period (days)')
    secax.set_yticks([1.25,1.33,1.43,1.54,1.67,1.82,2.00,2.50,3.33,5.00,10.0,20][::-1])
    secax.set_yticklabels([1.25,1.33,1.43,1.54,1.67,1.82,2.00,2.50,3.33,5.00,10.0,20][::-1])

    # Manually set safe limits
    min_freq = max(fb[0], 1e-3)
    max_freq = fb[1]
    period_ylim = freq_to_period(np.array([max_freq, min_freq]))  # reverse for log scale
    secax.set_ylim(period_ylim)

    #if ofil is not None:
    #    fig.savefig(ofil, bbox_inches='tight', dpi=144)
    if cbar == True:
        return img, ax.get_position()

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
# %%
if __name__ == "__main__":
    #
    # Input file and variable
    #
    fili = sorted(glob.glob(f"./data/ace2_37full*_asymComponent.nc")) 
    print(fili)
    asymComponent = xr.open_mfdataset(fili,combine='nested',concat_dim='member_id')['asymComponent']    # Load data

    era5 = xr.open_dataset('./data/era5_wk_Components.nc')['__xarray_dataarray_variable__']
    print(era5)

    era5.close()

    era5_background = xr.open_dataset('./data/era5_background.nc')['__xarray_dataarray_variable__']
    print(era5_background)

    # %%
    fili = sorted(glob.glob(f"./data/ace2_37full*_symComponent.nc")) 
    vari = "PRATEsfc"

    symComponent = xr.open_mfdataset(fili,combine='nested',concat_dim='member_id')['symComponent']    # Load data
    print(symComponent)

    fili = sorted(glob.glob(f"./data/ace2_37full*_asymComponent.nc")) 
    asymComponent = xr.open_mfdataset(fili,combine='nested',concat_dim='member_id')['asymComponent']    # Load data

    fili = sorted(glob.glob(f"./data/ace2_37full*_background.nc")) 
    background = xr.open_mfdataset(fili,combine='nested',concat_dim='member_id')['background']    # Load data

    ngcm = xr.open_dataset('./data/ngcm/ngcm_Components.nc')['__xarray_dataarray_variable__']
    print(ngcm)

    ngcm_background = xr.open_mfdataset(sorted(glob.glob(f'./data/ngcm/ngcm_background_*.nc')),combine='nested',concat_dim='member_id')['__xarray_dataarray_variable__']
    print(ngcm_background)

    fili = sorted(glob.glob(f"./data/amip/*nc"))
    amip = [xr.open_mfdataset(f,combine='nested',concat_dim='member_id')['__xarray_dataarray_variable__'].squeeze() for f in fili]
    #amip = xr.concat(amip, dim='component')
    print(amip)
   

    # %%
    # Plot averaged results
    #

    fig = plt.figure(figsize=(7.5,10))

    gs = GridSpec(4,2,wspace=0.5,hspace=0.5,right=0.9)

    ax = fig.add_subplot(gs[0])

    panel = 'a'
    plot_normalized_symmetric_spectrum(era5.sel(component='symmetric'), ax, panel, label='ERA5',ylabel='Normalized Symmetric Component')

    ax = fig.add_subplot(gs[1])

    panel ='b'
    plot_normalized_asymmetric_spectrum(era5.sel(component='antisymmetric'), ax, panel, label='ERA5', ylabel='Normalized Anti-Symmetric Component')

    ax = fig.add_subplot(gs[2])

    panel = 'c'
    plot_normalized_symmetric_spectrum(amip[2], ax, panel, label='AMIP',ylabel='Normalized Symmetric Component')

    ax = fig.add_subplot(gs[3])

    panel ='d'
    img, pos0 = plot_normalized_asymmetric_spectrum(amip[0], ax, panel, label='AMIP', ylabel='Normalized Anti-Symmetric Component',cbar=True)


    ax = fig.add_subplot(gs[4])

    panel = 'e'
    plot_normalized_symmetric_spectrum(symComponent.mean('member_id'), ax, panel, label='ACE2-ERA5')

    ax = fig.add_subplot(gs[5])

    panel ='f'
    img, pos1 = plot_normalized_asymmetric_spectrum(asymComponent.mean('member_id'), ax, panel, label='ACE2-ERA5',cbar=True)

    plot_colorbar(img, pos0, pos1)

    ax = fig.add_subplot(gs[6])

    panel = 'g'
    plot_normalized_symmetric_spectrum(ngcm.sel(component='symmetric').mean('member_id'), ax, panel, label='NGCM2.8',cbar=False)

    ax = fig.add_subplot(gs[7])

    panel ='h'
    plot_normalized_asymmetric_spectrum(ngcm.sel(component='antisymmetric').mean('member_id'), ax, panel, label='NGCM2.8', cbar=False)

    #plt.tight_layout()
    #if ofil is not None:
    outPlotName = '../../plots/Fig_1.WK_diagram.png'
    fig.savefig(outPlotName, bbox_inches='tight', dpi=300)

    print(symComponent)
# %%
fig = plt.figure(figsize=(7.5, 10))
gs = GridSpec(2, 1, wspace=0.5, hspace=0.5)
ax = fig.add_subplot(gs[0])

s = era5_background #background.mean('member_id')
fb = [0.01, 0.5]  # avoid zero to prevent divide-by-zero

# Clean and subset data
z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20, 20))
z = z.where(z['frequency'] > 0)  # mask freq=0 to avoid divide-by-zero

kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
z.plot.contourf(cmap='YlOrRd', vmin=0, vmax=1.8E-11, extend='both')

ax = fig.add_subplot(gs[1])
s = background.isel(member_id=0)
fb = [0.01, 0.5]  # avoid zero to prevent divide-by-zero

# Clean and subset data
z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20, 20))
z = z.where(z['frequency'] > 0)  # mask freq=0 to avoid divide-by-zero

kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
z.plot.contourf(cmap='YlOrRd',vmin=0, vmax=1.8E-11, extend='both')


# %%
def background_spectrum_plot(background):
    s = background
    fb = [0.01, 0.5]  # avoid zero to prevent divide-by-zero

    # Clean and subset data
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20, 20))
    z = z.where(z['frequency'] > 0)  # mask freq=0 to avoid divide-by-zero

    #z = z / z.max()  # Normalize the spectrum

    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    z.plot.contourf(cmap='YlOrRd', ax=ax, extend='max')

# %%
fig = plt.figure(figsize=(15, 15))
gs = GridSpec(4, 3)

for i in range(12):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    background_spectrum_plot(background.isel(member_id=i))
    ax.set_title(f'Member {i+1}', fontsize=12)
    ax.set_xlabel('Zonal Wavenumber')
    ax.set_ylabel('Frequency (CPD)')
    ax.axhline(1/60, linestyle='dashed', color='lightgray')
    ax.axhline(1/30, linestyle='dashed', color='lightgray')
    ax.axhline(1/6, linestyle='dashed', color='lightgray')
    ax.axhline(1/3, linestyle='dashed', color='lightgray')
    ax.axvline(0, linestyle='dashed', color='lightgray')

plt.tight_layout()
# %%

fig = plt.figure(figsize=(7.5,11))

gs = GridSpec(4,2,wspace=0.5,hspace=0.5,right=0.9,bottom=0.18)

ax = fig.add_subplot(gs[0])

panel = 'a'
plot_normalized_symmetric_spectrum(era5.sel(component='symmetric'), ax, panel, label='ERA5',ylabel='Normalized Symmetric Component')

ax = fig.add_subplot(gs[1])

panel ='b'
plot_normalized_background_spectrum(era5_background, ax, panel, label='ERA5', ylabel='Background Component')

ax = fig.add_subplot(gs[2])

panel = 'c'
plot_normalized_symmetric_spectrum(amip[2], ax, panel, label='AMIP',ylabel='Normalized Symmetric Component')

ax = fig.add_subplot(gs[3])

panel ='d'
plot_normalized_background_spectrum(amip[1], ax, panel, label='AMIP', ylabel='Background Component',cbar=False)


ax = fig.add_subplot(gs[4])

panel = 'e'
plot_normalized_symmetric_spectrum(symComponent.mean('member_id'), ax, panel, label='ACE2-ERA5')

ax = fig.add_subplot(gs[5])

panel ='f'
plot_normalized_background_spectrum(background.mean('member_id'), ax, panel, label='ACE2-ERA5',cbar=False)

#plot_colorbar(img, pos0, pos1)

ax = fig.add_subplot(gs[6])

panel = 'g'
img0, pos0 = plot_normalized_symmetric_spectrum(ngcm.sel(component='symmetric').mean('member_id'), ax, panel, label='NGCM2.8',cbar=True)

ax = fig.add_subplot(gs[7])

panel ='h'
img1, pos1 = plot_normalized_background_spectrum(ngcm_background.mean('member_id'), ax, panel, label='NGCM2.8', cbar=True)

cax = fig.add_axes([pos0.x0, pos1.y0-0.05, pos0.x1 - pos0.x0, 0.01])
cb = fig.colorbar(img0, cax=cax, label='Normalized Power', orientation='horizontal', drawedges=True, pad=0.15)

cax = fig.add_axes([pos1.x0, pos1.y0-0.05, pos1.x1 - pos1.x0, 0.01])
cb = fig.colorbar(img1, cax=cax, label='Raw background', orientation='horizontal', drawedges=True, pad=0.15)

#plt.tight_layout()
#if ofil is not None:
outPlotName = '../../plots/Fig_1.WK_diagram-background.png'
fig.savefig(outPlotName, bbox_inches='tight', dpi=300)


# %%
def background_spectrum_plot(background):
    s = background
    fb = [0.01, 0.5]  # avoid zero to prevent divide-by-zero

    # Clean and subset data
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20, 20))
    z = z.where(z['frequency'] > 0)  # mask freq=0 to avoid divide-by-zero

    #z = z / z.max()  # Normalize the spectrum

    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    z.plot.contourf(cmap='YlOrRd', ax=ax, vmin=0, vmax=1.4E-11, extend='max')

# %%
fig = plt.figure(figsize=(7.5, 7.5))
ax = fig.add_subplot(221)

background_spectrum_plot(era5_background)

ax = fig.add_subplot(222)

background_spectrum_plot(amip[1])

ax = fig.add_subplot(223)
background_spectrum_plot(background.mean('member_id'))

ax = fig.add_subplot(224)
background_spectrum_plot(ngcm_background.mean('member_id'))
# %%
def plot_normalized_background_spectrum(s, ax, panel, label='ERA5', ylabel='None', 
                                      ofil=None, cbar=False, show_xlabel=True, 
                                      show_ylabel=True, show_period_axis=True):
    """Plot the normalized background spectrum."""
    fb = [0.01, 0.5]

    # Clean and subset data
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20, 20))
    z = z.where(z['frequency'] > 0)
    
    # Normalize the spectrum    
    z_normalized = z / z.max()

    kmesh0, vmesh0 = np.meshgrid(z_normalized['wavenumber'], z_normalized['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z_normalized, 
                     levels=np.arange(0.2, 1.2, 0.1), 
                     cmap='YlOrRd', extend='max')
    cl = ax.contour(kmesh0, vmesh0, z_normalized, levels=np.arange(0.2, 1.2, 0.1), colors='k', linewidths=0.4)

    # Reference lines
    ax.axhline(1/60, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axhline(1/20, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axhline(1/5, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axhline(1/2.5, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axvline(0, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    
    # Add reference lines for specific wavenumbers
    for k in [0]:
        ax.axvline(k, linestyle='dotted', color='lightgray', lw=0.8, alpha=0.7)
    
    ax.set_xlim([-20, 20])

    ax.set_ylim(fb)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    #ax.set_yticklabels([0.1, 0.2, 0.3, 0.4, 0.5],size=14)
    ax.set_yticklabels([])

    # Conditional axis labels
    #if show_ylabel:
    #    ax.set_ylabel('Frequency (CPD)')
    if show_xlabel:
        ax.set_xlabel('Zonal Wavenumber')
        ax.set_xticks([-20,-15,-10,-5,0,5,10,15,20])
        ax.set_xticklabels([-20,-15,-10,-5,0,5,10,15,20],size=10)
    else:
        ax.set_xticks([-20,-15,-10,-5,0,5,10,15,20])
        ax.set_xticklabels([])

    ax.set_title(label, weight='bold', fontsize=12)

    ax.text(0.01, 1.03, panel, fontsize=16,
            fontweight='bold',
            transform=ax.transAxes)

    # Add secondary y-axis for period (days) - only if requested
    if show_period_axis:
        def freq_to_period(freq):
            return np.where(freq > 0, 1 / freq, np.nan)

        secax = ax.secondary_yaxis('right', functions=(freq_to_period, freq_to_period))
        secax.set_ylabel('Period (days)')
        secax.set_yticks([1.25,1.33,1.43,1.54,1.67,1.82,2.00,2.50,3.33,5.00,10.0,20][::-1])
        secax.set_yticklabels([1.25,1.33,1.43,1.54,1.67,1.82,2.00,2.50,3.33,5.00,10.0,20][::-1],size=10)
        #secax.set_yticklabels([])

        min_freq = max(fb[0], 1e-3)
        max_freq = fb[1]
        period_ylim = freq_to_period(np.array([max_freq, min_freq]))
        secax.set_ylim(period_ylim)
    else:
        ax.set_yticklabels([0.1, 0.2, 0.3, 0.4, 0.5],size=10)
        ax.set_ylabel('Frequency (cpd)',size=12)


    # Remove top and right spines
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)

    if cbar == True:
        return img, ax.get_position()

fig = plt.figure(figsize=(7.5, 7.5))

mpl.rcParams['font.size'] = 12

gs = GridSpec(2, 2, wspace=0.2, hspace=0.2, bottom=0.15, )
ax = fig.add_subplot(gs[0])

plot_normalized_background_spectrum(era5_background, ax, panel='a', label='ERA5', 
                                    show_xlabel=False, show_ylabel=True, show_period_axis=False,
                                    ylabel='Normalized Anti-Symmetric Component')

ax = fig.add_subplot(gs[1])

plot_normalized_background_spectrum(amip[1], ax, panel='b', label='AMIP', 
                                    show_xlabel=False, show_ylabel=True, show_period_axis=True,
                                    ylabel='Normalized Anti-Symmetric Component')

ax = fig.add_subplot(gs[2])
img0, pos0 = plot_normalized_background_spectrum(background.mean('member_id'), ax, panel='c', 
                                                 show_xlabel=True, show_ylabel=True, show_period_axis=False,
                                                 label='ACE2-ERA5', ylabel='Normalized Anti-Symmetric Component',cbar=True)

ax = fig.add_subplot(gs[3])
img1, pos1 = plot_normalized_background_spectrum(ngcm_background.mean('member_id'), ax, panel='d', 
                                                 show_xlabel=True, show_ylabel=True, show_period_axis=True,
                                                 label='NGCM2.8', ylabel='Normalized Anti-Symmetric Component',cbar=True)

cax = fig.add_axes([(pos0.x0+pos0.x1)/2, pos1.y0-0.08, (pos1.x1 - pos0.x0)/1.7, 0.01]) 
cb = fig.colorbar(img0, cax=cax, label='Normalized Power', orientation='horizontal', drawedges=True, pad=0.15)

plt.savefig('../../plots/Fig_1.WK_diagram-background_solo.png', bbox_inches='tight', dpi=300)


# %%
subplot_config = [
    (gs[0], 'a', era5.sel(component='symmetric'), 'plot_normalized_symmetric_spectrum', 'ERA5', 'Normalized Symmetric Component'),
    (gs[1], 'b', era5_background, 'plot_normalized_background_spectrum', 'ERA5', 'Background Component'),
    (gs[2], 'c', amip[2], 'plot_normalized_symmetric_spectrum', 'AMIP', 'Normalized Symmetric Component'),
    (gs[3], 'd', amip[1], 'plot_normalized_background_spectrum', 'AMIP', 'Background Component'),
    (gs[4], 'e', symComponent.mean('member_id'), 'plot_normalized_symmetric_spectrum', 'ACE2-ERA5', None),
    (gs[5], 'f', background.mean('member_id'), 'plot_normalized_background_spectrum', 'ACE2-ERA5', None),
    (gs[6], 'g', ngcm.sel(component='symmetric').mean('member_id'), 'plot_normalized_symmetric_spectrum', 'NGCM2.8', None),
    (gs[7], 'h', ngcm_background.mean('member_id'), 'plot_normalized_background_spectrum', 'NGCM2.8', None),
]

# %%
fig = plt.figure(figsize=(7.5, 7.5))

gs = GridSpec(2, 2, wspace=0.5, hspace=0.4, bottom=0.15, )
ax = fig.add_subplot(gs[0])

plot_normalized_asymmetric_spectrum(era5.sel(component='antisymmetric'), ax, panel='a', label='ERA5', ylabel='Normalized Anti-Symmetric Component')

ax = fig.add_subplot(gs[1])

plot_normalized_asymmetric_spectrum(amip[0], ax, panel='b', label='AMIP', ylabel='Normalized Anti-Symmetric Component')

ax = fig.add_subplot(gs[2])
img0, pos0 = plot_normalized_asymmetric_spectrum(asymComponent.mean('member_id'), ax, panel='c', label='ACE2-ERA5', ylabel='Normalized Anti-Symmetric Component',cbar=True)

ax = fig.add_subplot(gs[3])
img1, pos1 = plot_normalized_asymmetric_spectrum(ngcm.sel(component='antisymmetric').mean('member_id'), ax, panel='d', label='NGCM2.8', ylabel='Normalized Anti-Symmetric Component',cbar=True)

cax = fig.add_axes([(pos0.x0+pos0.x1)/2, pos1.y0-0.08, (pos1.x1 - pos0.x0)/1.7, 0.01]) 
cb = fig.colorbar(img0, cax=cax, label='Normalized Power', orientation='horizontal', drawedges=True, pad=0.15)

plt.savefig('../../plots/Fig_1.WK_diagram-asymmetric.png', bbox_inches='tight', dpi=300)

# %%


def add_tropical_wave_annotations(ax, component='sym'):
    """Add annotations for tropical wave modes on wavenumber-frequency plots."""
    swfreq, swwn = wf.genDispersionCurves()
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    if component == 'sym':
        ii = 3  # index for equivalent depth of 50m
        # MJO
        ax.text(8, 1/30, 'MJO', ha='center', va='center', fontsize=12, transform=ax.transData,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor='white', alpha=0.5))

        # Rossby waves
        ax.plot(np.zeros_like(np.linspace(0.01, swf[ii, 1, -1], len(swf[ii, 1, :])))+swk[ii, 2, 37], np.linspace(0.025, 0.088, len(swf[ii, 1, :])), color='tab:blue', lw=2)
        ax.plot(np.linspace(swk[ii, 2, 37],-0.8, len(swf[ii, 1, :])), np.linspace(0.025, 0.01, len(swf[ii, 1, :])), color='tab:blue', lw=2)
        if ii == 3:
            #continue
            ax.plot(swk[ii, 2, :38], swf[ii, 2, :38], color='tab:blue')
        
        ax.text(-14, 1/10, 'n=1 ER', ha='center', va='center', fontsize=12, transform=ax.transData,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor='white', alpha=0.5))

        # Kevlin waves
        ax.plot(np.linspace(4, 14, len(swf[ii, 1, :])), np.zeros_like((swf[ii, 1, :])) + (1/2.5), color='tab:red', lw=2)
        ax.plot(np.zeros_like(np.linspace(0.01, swf[ii, 1, -1], len(swf[ii, 1, :])))+14, np.linspace(0.4, 0.25, len(swf[ii, 1, :])), color='tab:red', lw=2)
        ax.plot(np.linspace(1, 3, len(swf[ii, 1, :])), np.zeros_like(np.linspace(0.4, 0.25, len(swf[ii, 1, :])))+1/20, color='tab:red', lw=2)
        ax.plot(np.linspace(3, 14, len(swf[ii, 1, :])), np.linspace(1/20, 0.25, len(swf[ii, 1, :])), color='tab:red', lw=2)
        ax.plot(np.linspace(1, 4, len(swf[ii, 1, :])), np.linspace(1/20, 0.4, len(swf[ii, 1, :])), color='tab:red', lw=2)
        #ax.plot(np.linspace(swk[ii, 2, 37],-0.8, len(swf[ii, 1, :])), np.linspace(0.025, 0.01, len(swf[ii, 1, :])), color='tab:red', lw=2)
        ax.text(9, 0.32, 'Kelvin', ha='center', va='center', fontsize=12, transform=ax.transData,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor='white', alpha=0.5))

        # WIG
        ii = 4
        ax.plot(np.zeros_like(np.linspace(0.01, swf[ii, 1, -1], len(swf[ii, 1, :])))+swk[5, 1, 25], np.linspace(swf[5, 1, 25], 0.7, len(swf[ii, 1, :])), color='tab:green', lw=2)
        ax.plot(np.zeros_like(np.linspace(0.01, swf[ii, 1, -1], len(swf[ii, 1, :])))+swk[5, 1, 41], np.linspace(swf[5, 1, 41], 0.7, len(swf[ii, 1, :])), color='tab:green', lw=2)

        ax.plot(swk[5, 1, 25:42], swf[5, 1, 25:42], color='tab:green', lw=2)

        ax.text(-6, 0.45, 'n=1 WIG', ha='center', va='center', fontsize=12, transform=ax.transData,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor='white', alpha=0.5))  
        

    else:
        ii = 0
        c = 'darkgrey'
        ax.plot(swk[ii, 0, :], swf[ii, 0, :], color=c)
        ax.plot(swk[ii, 1, :], swf[ii, 1, :], color=c)
        ax.plot(swk[ii, 2, :], swf[ii, 2, :], color=c)

        ax.plot((np.linspace(swk[1, 1, 8], 13, len(swf[ii, 1, :]))), np.linspace(swf[1, 1, 8], 1/1.82, len(swf[ii, 1, :])), color='tab:purple', lw=2) 

        ax.plot(swk[1, 1, 8:], swf[1, 1, 8:], color='tab:purple', lw=2)
        ax.plot(swk[1, 2, 8:], swf[1, 2, 8:], color='tab:purple', lw=2)
        ax.plot(np.zeros_like(np.linspace(swk[1, 2, 24], 0, len(swf[1, 2, 8:]))), np.linspace(swf[1, 2, 24], swf[1, 1, 24], len(swf[1, 2, 8:])), color='tab:purple', lw=2)

        ax.text(10, 0.32, 'n=0 EIG', ha='center', va='center', fontsize=12, transform=ax.transData,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor='white', alpha=0.8))  

        ax.plot((np.linspace(swk[1, 1, 8], 13, len(swf[ii, 1, :]))), np.linspace(swf[1, 1, 8], 1/1.82, len(swf[ii, 1, :])), color='tab:purple', lw=2) 

        ax.plot(swk[0, 1, 26:39], swf[0, 1, 26:39], color='tab:orange', lw=2)
        ax.plot(swk[0, 2, 26:39], swf[0, 2, 26:39], color='tab:orange', lw=2)
        ax.plot(np.zeros_like(np.linspace(swk[0, 2, 24], 0, len(swf[0, 2, 26:39])))+swk[0, 1, 26], np.linspace(swf[0, 2, 26], swf[0, 1, 26], len(swf[0, 2, 26:39])), color='tab:orange', lw=2)
        ax.plot(np.zeros_like(np.linspace(swk[0, 2, 24], 0, len(swf[0, 2, 26:39])))+swk[0, 1, 38], np.linspace(swf[0, 2, 38], swf[0, 1, 38], len(swf[0, 2, 26:39])), color='tab:orange', lw=2)

        ax.text(-12, 0.21, 'MRG', ha='center', va='center', fontsize=12, transform=ax.transData,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor='white', alpha=0.5))

def plot_normalized_symmetric_spectrum(s, ax, panel, label='ERA5', ylabel='None', 
                                     ofil=None, cbar=False, show_xlabel=True, 
                                     show_ylabel=True, show_period_axis=True):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""
    fb = [0.01, 0.5]  # avoid zero for log-scale period

    # Get data for dispersion curves
    swfreq, swwn = wf.genDispersionCurves()
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    # Clean and subset data
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20, 20))
    z = z.where(z['frequency'] > 0)  # mask freq=0 to avoid divide-by-zero

    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z, levels=np.arange(1.1, 1.71, 0.1), cmap='YlOrRd', extend='max')
    cl = ax.contour(kmesh0, vmesh0, z, levels=np.arange(1.1, 1.71, 0.1), colors='k', linewidths=0.4)

    # Plot dispersion curves
    c = 'darkgray'
    for ii in range(3, 6):
        ax.plot(swk[ii, 0, :], swf[ii, 0, :], color=c)
        ax.plot(swk[ii, 1, :], swf[ii, 1, :], color=c)
        ax.plot(swk[ii, 2, :], swf[ii, 2, :], color=c)

    # Reference lines
    ax.axhline(1/60, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axhline(1/20, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axhline(1/5, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axhline(1/2.5, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axvline(0, linestyle='dashed', color='lightgray',lw=0.8,  alpha=0.7)
    
    ax.set_xlim([-20, 20])
    ax.set_ylim(fb)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels([0.1, 0.2, 0.3, 0.4, 0.5],size=14)
    
    # Conditional axis labels
    if show_ylabel:
        ax.set_ylabel('Frequency (cpd)')
    if show_xlabel:
        ax.set_xlabel('Zonal Wavenumber')
        ax.set_xticks([-20,-15,-10,-5,0,5,10,15,20])
        ax.set_xticklabels([-20,-15,-10,-5,0,5,10,15,20],size=14)
    else:
        ax.set_xticks([-20,-15,-10,-5,0,5,10,15,20])
        ax.set_xticklabels([])
    
    ax.set_title(label, weight='bold', fontsize=16)

    ax.text(0.01, 1.05, panel, fontsize=16,
            fontweight='bold',
            transform=ax.transAxes)

    add_tropical_wave_annotations(ax, component='sym')

    # Add secondary y-axis for period (days) - only if requested
    if show_period_axis:
        def freq_to_period(freq):
            return np.where(freq > 0, 1 / freq, np.nan)

        secax = ax.secondary_yaxis('right', functions=(freq_to_period, freq_to_period))
        #secax.set_ylabel('Period (days)')
        secax.set_yticks([1.25,1.33,1.43,1.54,1.67,1.82,2.00,2.50,3.33,5.00,10.0,20][::-1])
        #secax.set_yticklabels([1.25,1.33,1.43,1.54,1.67,1.82,2.00,2.50,3.33,5.00,10.0,20][::-1])
        secax.set_yticklabels([])

        # Manually set safe limits
        min_freq = max(fb[0], 1e-3)
        max_freq = fb[1]
        period_ylim = freq_to_period(np.array([max_freq, min_freq]))
        secax.set_ylim(period_ylim)

    # Remove top and right spines
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)

    if cbar == True:
        return img, ax.get_position()


def plot_normalized_asymmetric_spectrum(s, ax, panel, label='ERA5', ylabel='None', 
                                     ofil=None, cbar=False, show_xlabel=True, 
                                     show_ylabel=True, show_period_axis=True):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""
    fb = [0.01, 0.5]  # avoid zero for log-scale period

    # Get data for dispersion curves
    swfreq, swwn = wf.genDispersionCurves()
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)


    # Clean and subset data
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20, 20))
    z = z.where(z['frequency'] > 0)  # mask freq=0 to avoid divide-by-zero

    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z, levels=np.arange(1.1, 1.71, 0.1), cmap='YlOrRd', extend='max')
    cl = ax.contour(kmesh0, vmesh0, z, levels=np.arange(1.1, 1.71, 0.1), colors='k', linewidths=0.4)

    # Plot dispersion curves
    c = 'darkgray'
    for ii in range(0, 3):
        ax.plot(swk[ii, 0, :], swf[ii, 0, :], color=c)
        ax.plot(swk[ii, 1, :], swf[ii, 1, :], color=c)
        ax.plot(swk[ii, 2, :], swf[ii, 2, :], color=c)

    ax.axhline(1/60, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axhline(1/20, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axhline(1/5, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axhline(1/2.5, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axvline(0, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)

    # Add reference lines for specific wavenumbers
    for k in [0]:
        ax.axvline(k, linestyle='dotted', color='lightgray', lw=0.8, alpha=0.7)
    
    ax.set_xlim([-20, 20])

    ax.set_ylim(fb)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    #ax.set_yticklabels([0.1, 0.2, 0.3, 0.4, 0.5],size=14)
    ax.set_yticklabels([])

    # Conditional axis labels
    #if show_ylabel:
    #    ax.set_ylabel('Frequency (CPD)')
    if show_xlabel:
        ax.set_xlabel('Zonal Wavenumber')
        ax.set_xticks([-20,-15,-10,-5,0,5,10,15,20])
        ax.set_xticklabels([-20,-15,-10,-5,0,5,10,15,20],size=14)
    else:
        ax.set_xticks([-20,-15,-10,-5,0,5,10,15,20])
        ax.set_xticklabels([])

    ax.set_title(label, weight='bold', fontsize=16)

    ax.text(0.01, 1.02, panel, fontsize=16,
            fontweight='bold',
            transform=ax.transAxes)


    add_tropical_wave_annotations(ax, component='asym')
    # Add secondary y-axis for period (days) - only if requested
    if show_period_axis:
        def freq_to_period(freq):
            return np.where(freq > 0, 1 / freq, np.nan)

        secax = ax.secondary_yaxis('right', functions=(freq_to_period, freq_to_period))
        secax.set_ylabel('Period (days)')
        secax.set_yticks([1.25,1.33,1.43,1.54,1.67,1.82,2.00,2.50,3.33,5.00,10.0,20][::-1])
        secax.set_yticklabels([1.25,1.33,1.43,1.54,1.67,1.82,2.00,2.50,3.33,5.00,10.0,20][::-1],size=14)
        #secax.set_yticklabels([])

        min_freq = max(fb[0], 1e-3)
        max_freq = fb[1]
        period_ylim = freq_to_period(np.array([max_freq, min_freq]))
        secax.set_ylim(period_ylim)

    # Remove top and right spines
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)

    if cbar == True:
        return img, ax.get_position()
# ---

# Main comparison figure
fig = plt.figure(figsize=(10, 15))
gs = GridSpec(4, 2, wspace=0.1, hspace=0.2, right=0.9, bottom=0.18)

mpl.rcParams.update({'font.size': 16, 'font.family': 'sans-serif'})

# Define subplot configurations
subplot_config = [
    (gs[0], 'a', era5.sel(component='symmetric'), 'plot_normalized_symmetric_spectrum', 'ERA5', 'Normalized Symmetric Component'),
    (gs[1], 'b', era5.sel(component='antisymmetric'), 'plot_normalized_asymmetric_spectrum', 'ERA5', 'Normalized Antisymmetric Component'),
    (gs[2], 'c', amip[2], 'plot_normalized_symmetric_spectrum', 'AMIP', 'Normalized Symmetric Component'),
    (gs[3], 'd', amip[0], 'plot_normalized_asymmetric_spectrum', 'AMIP', 'Normalized Antisymmetric Component'),
    (gs[4], 'e', symComponent.mean('member_id'), 'plot_normalized_symmetric_spectrum', 'ACE2-ERA5', None),
    (gs[5], 'f', asymComponent.mean('member_id'), 'plot_normalized_asymmetric_spectrum', 'ACE2-ERA5', None),
    (gs[6], 'g', ngcm.sel(component='symmetric').mean('member_id'), 'plot_normalized_symmetric_spectrum', 'NGCM2.8', None),
    (gs[7], 'h', ngcm.sel(component='antisymmetric').mean('member_id'), 'plot_normalized_asymmetric_spectrum', 'NGCM2.8', None),
]

# Store images and positions for colorbars
images = {}
positions = {}

# Create subplots
for subplot_pos, panel, data, plot_func, label, ylabel in subplot_config:
    ax = fig.add_subplot(subplot_pos)
    
    # Determine if colorbar should be shown
    cbar = panel in ['g', 'h']
    
    # Call the appropriate plotting function
    if plot_func == 'plot_normalized_symmetric_spectrum':
        if cbar:
            img, pos = plot_normalized_symmetric_spectrum(data, ax, panel, label=label, cbar=True)
            images['symmetric'] = img
            positions['symmetric'] = pos
        else:
            plot_normalized_symmetric_spectrum(data, ax, panel, label=label, show_xlabel=False, ylabel=ylabel)
    else:  # background spectrum
        if cbar:
            img, pos = plot_normalized_asymmetric_spectrum(data, ax, panel, label=label, cbar=True)
            images['background'] = img
            positions['background'] = pos
        else:
            cbar_flag = panel != 'd'  # Don't show colorbar for panel 'd'
            plot_normalized_asymmetric_spectrum(data, ax, panel, label=label, ylabel=ylabel, show_xlabel=False, cbar=cbar_flag)

# Add colorbars for the bottom row
if 'symmetric' in images and 'symmetric' in positions:
    pos0 = positions['symmetric']
    cax = fig.add_axes([(pos0.x1-pos0.x0)*0.5, pos0.y0-0.05, (pos1.x1 - pos0.x0)*0.85, 0.01])
    cb = fig.colorbar(images['symmetric'], cax=cax, label='Normalized Power', 
                     orientation='horizontal', drawedges=True, pad=0.15)

#if 'background' in images and 'background' in positions:
    #pos1 = positions['background']
    #cax = fig.add_axes([pos1.x0, pos1.y0-0.05, pos1.x1 - pos1.x0, 0.01])
    #cb = fig.colorbar(images['background'], cax=cax, label='Normalized ', 
    #                 orientation='horizontal', drawedges=True, pad=0.15)

# Save figure
outPlotName = 'Fig_1.WK_diagram-sym+asym.png'
fig.savefig(outPlotName, bbox_inches='tight', dpi=300)
# %%

def plot_normalized_symmetric_spectrum(s, ax, panel, label='ERA5', ylabel='None', 
                                     ofil=None, cbar=False, show_xlabel=True, 
                                     show_ylabel=True, show_period_axis=True):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""
    fb = [0.01, 0.5]  # avoid zero for log-scale period

    # Get data for dispersion curves
    swfreq, swwn = wf.genDispersionCurves()
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    # Clean and subset data
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20, 20))
    z = z.where(z['frequency'] > 0)  # mask freq=0 to avoid divide-by-zero

    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z, levels=np.arange(1.1, 1.71, 0.1), cmap='YlOrRd', extend='max')

    # Plot dispersion curves
    c = 'darkgray'
    for ii in range(3, 6):
        ax.plot(swk[ii, 0, :], swf[ii, 0, :], color=c)
        ax.plot(swk[ii, 1, :], swf[ii, 1, :], color=c)
        ax.plot(swk[ii, 2, :], swf[ii, 2, :], color=c)

    # Reference lines
    ax.axhline(1/60, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axhline(1/20, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axhline(1/5, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axhline(1/2.5, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axvline(0, linestyle='dashed', color='lightgray',lw=0.8,  alpha=0.7)
    
    ax.set_xlim([-20, 20])
    ax.set_ylim(fb)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels([0.1, 0.2, 0.3, 0.4, 0.5],size=14)
    
    # Conditional axis labels
    if show_ylabel:
        ax.set_ylabel('Frequency (cpd)')
    if show_xlabel:
        ax.set_xlabel('Zonal Wavenumber')
        ax.set_xticks([-20,-15,-10,-5,0,5,10,15,20])
        ax.set_xticklabels([-20,-15,-10,-5,0,5,10,15,20],size=14)
    else:
        ax.set_xticks([-20,-15,-10,-5,0,5,10,15,20])
        ax.set_xticklabels([])
    
    ax.set_title(label, weight='bold', fontsize=16)

    ax.text(0.01, 1.05, panel, fontsize=16,
            fontweight='bold',
            transform=ax.transAxes)

    # Add secondary y-axis for period (days) - only if requested
    if show_period_axis:
        def freq_to_period(freq):
            return np.where(freq > 0, 1 / freq, np.nan)

        secax = ax.secondary_yaxis('right', functions=(freq_to_period, freq_to_period))
        #secax.set_ylabel('Period (days)')
        secax.set_yticks([1.25,1.33,1.43,1.54,1.67,1.82,2.00,2.50,3.33,5.00,10.0,20][::-1])
        #secax.set_yticklabels([1.25,1.33,1.43,1.54,1.67,1.82,2.00,2.50,3.33,5.00,10.0,20][::-1])
        secax.set_yticklabels([])

        # Manually set safe limits
        min_freq = max(fb[0], 1e-3)
        max_freq = fb[1]
        period_ylim = freq_to_period(np.array([max_freq, min_freq]))
        secax.set_ylim(period_ylim)

    # Remove top and right spines
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)

    if cbar == True:
        return img, ax.get_position()


def plot_normalized_background_spectrum(s, ax, panel, label='ERA5', ylabel='None', 
                                      ofil=None, cbar=False, show_xlabel=True, 
                                      show_ylabel=True, show_period_axis=True):
    """Plot the normalized background spectrum."""
    fb = [0.01, 0.5]

    # Clean and subset data
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20, 20))
    z = z.where(z['frequency'] > 0)
    
    # Normalize the spectrum    
    z_normalized = z / z.max()

    kmesh0, vmesh0 = np.meshgrid(z_normalized['wavenumber'], z_normalized['frequency'])
    img = ax.contourf(kmesh0, vmesh0, z_normalized, 
                     levels=np.arange(0.1, 1.1, 0.1), 
                     cmap='YlOrRd', extend='max')

    # Reference lines
    ax.axhline(1/60, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axhline(1/20, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axhline(1/5, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axhline(1/2.5, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    ax.axvline(0, linestyle='dashed', color='lightgray', lw=0.8, alpha=0.7)
    
    # Add reference lines for specific wavenumbers
    for k in [0]:
        ax.axvline(k, linestyle='dotted', color='lightgray', lw=0.8, alpha=0.7)
    
    ax.set_xlim([-20, 20])

    ax.set_ylim(fb)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    #ax.set_yticklabels([0.1, 0.2, 0.3, 0.4, 0.5],size=14)
    ax.set_yticklabels([])

    # Conditional axis labels
    #if show_ylabel:
    #    ax.set_ylabel('Frequency (CPD)')
    if show_xlabel:
        ax.set_xlabel('Zonal Wavenumber')
        ax.set_xticks([-20,-15,-10,-5,0,5,10,15,20])
        ax.set_xticklabels([-20,-15,-10,-5,0,5,10,15,20],size=14)
    else:
        ax.set_xticks([-20,-15,-10,-5,0,5,10,15,20])
        ax.set_xticklabels([])

    ax.set_title(label, weight='bold', fontsize=16)

    ax.text(0.01, 1.02, panel, fontsize=16,
            fontweight='bold',
            transform=ax.transAxes)

    # Add secondary y-axis for period (days) - only if requested
    if show_period_axis:
        def freq_to_period(freq):
            return np.where(freq > 0, 1 / freq, np.nan)

        secax = ax.secondary_yaxis('right', functions=(freq_to_period, freq_to_period))
        secax.set_ylabel('Period (days)')
        secax.set_yticks([1.25,1.33,1.43,1.54,1.67,1.82,2.00,2.50,3.33,5.00,10.0,20][::-1])
        secax.set_yticklabels([1.25,1.33,1.43,1.54,1.67,1.82,2.00,2.50,3.33,5.00,10.0,20][::-1],size=14)
        #secax.set_yticklabels([])

        min_freq = max(fb[0], 1e-3)
        max_freq = fb[1]
        period_ylim = freq_to_period(np.array([max_freq, min_freq]))
        secax.set_ylim(period_ylim)

    # Remove top and right spines
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)

    if cbar == True:
        return img, ax.get_position()
# ---

# Main comparison figure
fig = plt.figure(figsize=(10, 15))
gs = GridSpec(4, 2, wspace=0.1, hspace=0.2, right=0.9, bottom=0.18)

mpl.rcParams.update({'font.size': 16, 'font.family': 'sans-serif'})

# Define subplot configurations
subplot_config = [
    (gs[0], 'a', era5.sel(component='symmetric'), 'plot_normalized_symmetric_spectrum', 'ERA5', 'Normalized Symmetric Component'),
    (gs[1], 'b', era5_background, 'plot_normalized_background_spectrum', 'ERA5', 'Background Component'),
    (gs[2], 'c', amip[2], 'plot_normalized_symmetric_spectrum', 'AMIP', 'Normalized Symmetric Component'),
    (gs[3], 'd', amip[1], 'plot_normalized_background_spectrum', 'AMIP', 'Background Component'),
    (gs[4], 'e', symComponent.mean('member_id'), 'plot_normalized_symmetric_spectrum', 'ACE2-ERA5', None),
    (gs[5], 'f', background.mean('member_id'), 'plot_normalized_background_spectrum', 'ACE2-ERA5', None),
    (gs[6], 'g', ngcm.sel(component='symmetric').mean('member_id'), 'plot_normalized_symmetric_spectrum', 'NGCM2.8', None),
    (gs[7], 'h', ngcm_background.mean('member_id'), 'plot_normalized_background_spectrum', 'NGCM2.8', None),
]

# Store images and positions for colorbars
images = {}
positions = {}

# Create subplots
for subplot_pos, panel, data, plot_func, label, ylabel in subplot_config:
    ax = fig.add_subplot(subplot_pos)
    
    # Determine if colorbar should be shown
    cbar = panel in ['g', 'h']
    
    # Call the appropriate plotting function
    if plot_func == 'plot_normalized_symmetric_spectrum':
        if cbar:
            img, pos = plot_normalized_symmetric_spectrum(data, ax, panel, label=label, cbar=True)
            images['symmetric'] = img
            positions['symmetric'] = pos
        else:
            plot_normalized_symmetric_spectrum(data, ax, panel, label=label, show_xlabel=False, ylabel=ylabel)
    else:  # background spectrum
        if cbar:
            img, pos = plot_normalized_background_spectrum(data, ax, panel, label=label, cbar=True)
            images['background'] = img
            positions['background'] = pos
        else:
            cbar_flag = panel != 'd'  # Don't show colorbar for panel 'd'
            plot_normalized_background_spectrum(data, ax, panel, label=label, ylabel=ylabel, show_xlabel=False, cbar=cbar_flag)

# Add colorbars for the bottom row
if 'symmetric' in images and 'symmetric' in positions:
    pos0 = positions['symmetric']
    cax = fig.add_axes([pos0.x0, pos0.y0-0.05, pos0.x1 - pos0.x0, 0.01])
    cb = fig.colorbar(images['symmetric'], cax=cax, label='Normalized Power', 
                     orientation='horizontal', drawedges=True, pad=0.15)

if 'background' in images and 'background' in positions:
    pos1 = positions['background']
    cax = fig.add_axes([pos1.x0, pos1.y0-0.05, pos1.x1 - pos1.x0, 0.01])
    cb = fig.colorbar(images['background'], cax=cax, label='Normalized background', 
                     orientation='horizontal', drawedges=True, pad=0.15)

# Save figure
outPlotName = 'Fig_1.WK_diagram-background.png'
fig.savefig(outPlotName, bbox_inches='tight', dpi=300)
# %%
# Main comparison figure
fig = plt.figure(figsize=(10, 15))
gs = GridSpec(4, 2, wspace=0.1, hspace=0.2, right=0.9, bottom=0.18)

mpl.rcParams.update({'font.size': 16, 'font.family': 'sans-serif'})

# Define subplot configurations
subplot_config = [
    (gs[0], 'a', era5.sel(component='symmetric'), 'plot_normalized_symmetric_spectrum', 'ERA5', 'Normalized Symmetric Component'),
    (gs[1], 'b', era5_background, 'plot_normalized_background_spectrum', 'ERA5', 'Background Component'),
    (gs[2], 'c', amip[2], 'plot_normalized_symmetric_spectrum', 'AMIP', 'Normalized Symmetric Component'),
    (gs[3], 'd', amip[1], 'plot_normalized_background_spectrum', 'AMIP', 'Background Component'),
    (gs[4], 'e', symComponent.mean('member_id'), 'plot_normalized_symmetric_spectrum', 'ACE2-ERA5', None),
    (gs[5], 'f', background.mean('member_id'), 'plot_normalized_background_spectrum', 'ACE2-ERA5', None),
    (gs[6], 'g', ngcm.sel(component='symmetric').mean('member_id'), 'plot_normalized_symmetric_spectrum', 'NGCM2.8', None),
    (gs[7], 'h', ngcm_background.mean('member_id'), 'plot_normalized_background_spectrum', 'NGCM2.8', None),
]

# Store images and positions for colorbars
images = {}
positions = {}

# Create subplots
for subplot_pos, panel, data, plot_func, label, ylabel in subplot_config:
    ax = fig.add_subplot(subplot_pos)

    # Determine if colorbar should be shown
    cbar = panel in ['g', 'h']

    # Call the appropriate plotting function
    if plot_func == 'plot_normalized_symmetric_spectrum':
        if cbar:
            img, pos = plot_normalized_symmetric_spectrum(data, ax, panel, label=label, cbar=True)
            images['symmetric'] = img
            positions['symmetric'] = pos
        else:
            plot_normalized_symmetric_spectrum(data, ax, panel, label=label, show_xlabel=False, ylabel=ylabel)
    else:  # background spectrum
        if cbar:
            img, pos = plot_normalized_background_spectrum(data, ax, panel, label=label, cbar=True)
            images['background'] = img
            positions['background'] = pos
        else:
            cbar_flag = panel != 'd'  # Don't show colorbar for panel 'd'
            plot_normalized_background_spectrum(data, ax, panel, label=label, ylabel=ylabel, show_xlabel=False, cbar=cbar_flag)

# Add colorbars for the bottom row
if 'symmetric' in images and 'symmetric' in positions:
    pos0 = positions['symmetric']
    cax = fig.add_axes([pos0.x0, pos0.y0-0.05, pos0.x1 - pos0.x0, 0.01])
    cb = fig.colorbar(images['symmetric'], cax=cax, label='Normalized Power',
                     orientation='horizontal', drawedges=True, pad=0.15)

if 'asymmetric' in images and 'asymmetric' in positions:
    pos1 = positions['asymmetric']
    cax = fig.add_axes([pos1.x0, pos1.y0-0.05, pos1.x1 - pos1.x0, 0.01])
    cb = fig.colorbar(images['asymmetric'], cax=cax, label='Normalized Power',
                     orientation='horizontal', drawedges=True, pad=0.15)

# Save figure
outPlotName = '../../plots/Fig_1.WK_diagram-sym+asym.png'
fig.savefig(outPlotName, bbox_inches='tight', dpi=300)


# %%
