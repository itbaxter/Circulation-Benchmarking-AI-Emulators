# %%
import xarray as xr
import numpy as np
from scipy.stats import linregress as _linregress
import matplotlib.pyplot as plt
import matplotlib as mpl
import xesmf as xe
import glob as glob
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings('ignore')

# %%
fig = plt.figure(figsize=(55,25))

import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, ConnectionPatch
from matplotlib.patches import FancyBboxPatch

def add_fancy_arrow_label(ax, start_xy, end_xy, text, color='black', 
                         arrow_style='->', mutation_scale=20, 
                         text_offset=(5, 5), fontsize=10, **kwargs):
    """
    Add a fancy arrow pointing to a curve with a text label.
    
    Parameters:
    -----------
    ax : matplotlib axis
    start_xy : tuple (x, y) - starting point of arrow (in data coordinates)
    end_xy : tuple (x, y) - ending point of arrow (pointing to curve)
    text : str - label text
    color : str - arrow and text color
    arrow_style : str - arrow style
    mutation_scale : int - arrow head size
    text_offset : tuple - offset for text from arrow start
    """
    # Create fancy arrow
    arrow = FancyArrowPatch(start_xy, end_xy,
                           arrowstyle=arrow_style,
                           mutation_scale=mutation_scale,
                           color=color,
                           linewidth=1.5,
                           **kwargs)
    ax.add_patch(arrow)
    
    # Add text label
    ax.annotate(text, 
                xy=start_xy, 
                xytext=text_offset,
                textcoords='offset points',
                fontsize=fontsize,
                color=color,
                ha='left', va='bottom')

def add_arrow_with_textbox(ax, start_xy, end_xy, text, color='black',
                          box_style="round,pad=0.3", alpha=0.8):
    """Add arrow with text in a fancy box."""
    # Arrow
    arrow = FancyArrowPatch(start_xy, end_xy,
                           arrowstyle='->',
                           mutation_scale=15,
                           color=color,
                           linewidth=1.5)
    ax.add_patch(arrow)
    
    # Text with fancy box
    bbox_props = dict(boxstyle=box_style, facecolor='white', 
                     edgecolor=color, alpha=alpha)
    ax.annotate(text, xy=start_xy, xytext=(10, 10),
                textcoords='offset points',
                bbox=bbox_props,
                fontsize=10,
                color=color)

# Option 3: Curved arrow (great for pointing to specific parts of curves)
def add_curved_arrow_label(ax, start_xy, end_xy, text, arrow_color='black', text_color='black',
                          connectionstyle="arc3,rad=0.3"):
    """Add a curved arrow with label."""
    ax.annotate(text, xy=end_xy, xytext=start_xy,
                arrowprops=dict(arrowstyle='->',
                              connectionstyle=connectionstyle,
                              color=arrow_color,
                              lw=1.5),
                fontsize=10,
                color=text_color,
                ha='center', va='center')

# Option 4: Multiple preset arrow styles
def add_preset_arrow(ax, start_xy, end_xy, text, style='fancy', color='black'):
    """Add arrow with preset styles."""
    
    styles = {
        'fancy': dict(arrowstyle='->', mutation_scale=20, 
                     connectionstyle="arc3,rad=0.1"),
        'bold': dict(arrowstyle='-|>', mutation_scale=25, lw=2),
        'curved': dict(arrowstyle='->', mutation_scale=15,
                      connectionstyle="arc3,rad=0.3"),
        'simple': dict(arrowstyle='->', mutation_scale=15),
        'wedge': dict(arrowstyle='wedge', mutation_scale=20)
    }
    
    arrow_props = styles.get(style, styles['fancy'])
    arrow_props['color'] = color
    
    ax.annotate(text, xy=end_xy, xytext=start_xy,
                arrowprops=arrow_props,
                fontsize=10,
                color=color,
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                         edgecolor=color, alpha=0.8))

# %%
def area_weighted_ave(ds):
    if 'lat' not in ds.dims:
        ds = ds.rename({'latitude':'lat','longitude':'lon'})
    coslat = np.cos(np.deg2rad(ds.lat))
    ds,coslat = xr.broadcast(ds,coslat)
    ds = ds * coslat
    #return ds.mean(('lat','lon'),skipna=True)
    return ds.sum(('lat','lon'),skipna=True)/((ds/ds)*coslat).sum(('lat','lon'),skipna=True)

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
era5 = xr.open_dataset('/project/tas1/itbaxter/aimip/benchmarking_ai_variability/data/era5/eq_uwind/era5_uwind_qbo.nc')['__xarray_dataarray_variable__']
era5
# %%
csp = xr.open_dataset('/project/tas1/itbaxter/aimip/benchmarking_ai_variability/data/ngcm/eq_uwind/ngcm_uwind_qbo.nc')['__xarray_dataarray_variable__'] #.sel(time=slice('2017-01-01','2023-12-31'))
csp

# %%
csp.member_id.values

# %%
#ace2 = xr.open_dataset('/project2/tas1/itbaxter/NeuralGCM_Decadal_Simulations/data/processed/QBO/ace2_era5_uwind_qbo.nc')['__xarray_dataarray_variable__'] #.sel(time=slice('2001-01-01','2010-12-31'))

def preprocess(ds):
    print(ds)
    if 'init_time' in ds.coords:
        ds = ds.drop('init_time')
    return ds['__xarray_dataarray_variable__'].sel(time=slice('1981-01-01','2022-12-31')).squeeze()

ace2 = xr.open_mfdataset('/project2/tas1/itbaxter/NeuralGCM_Decadal_Simulations/data/processed/QBO/ace2_qbo_mon*nc',combine='nested',concat_dim='member_id',preprocess=preprocess)
ace2

# %%
def stack_loop(dataarrays):
    stacked_dataarrays = []
    j = 0
    for data in dataarrays:
        # Drop problematic coordinates
        if 'plev_bnds' in data.coords:
            data = data.drop('plev_bnds')
        if 'lev_bnds' in data.coords:
            data = data.drop('lev_bnds')
        if 'bnds' in data.coords:
            data = data.drop('bnds')
            
        if 'member_id' not in data.dims:
            data = data.expand_dims('member_id')
            data['member_id'] = ('member_id', [data.attrs.get('member_id', 'r1i1p1f1')])
        
        #data.coords['member_id'] = ('member_id', [f'{data.source_id.values}_{member.values}' for member in data["member_id"]])
        print(data.member_id.values)
        if 'height' not in data.dims:
            data = data.expand_dims('height')
            data['height'] = ('height', [data.attrs.get('height', 2)])

        stacked_dataarrays.append(data)
        j += len(data.member_id)
    return stacked_dataarrays

class READER:
    def __init__(self,files):
        self.files = files

    def reader(self,f):
        source_id = f.split('/')[-1].split('_')[2]
        start = f.split('/')[-1].split('_')[5].split('-')[0]
        end = int(f.split('/')[-1].split('_')[5].split('-')[1].split('.')[0])
        print(source_id,start,end)
        try:
            ds = xr.open_dataset(f)
        except:
            ds = xr.open_dataset(f,decode_times=False)
            print(ds,start,end)
            ds.coords['time'] = np.arange(f'{start}-01-01',f'{end}-01-01',dtype='datetime64[M]')
            
        # Drop problematic coordinates before annual mean
        if 'plev_bnds' in ds.coords:
            ds = ds.drop('plev_bnds')
        if 'lev_bnds' in ds.coords:
            ds = ds.drop('lev_bnds')
        if 'bnds' in ds.coords:
            ds = ds.drop('bnds')
            
        ds_ann = ds # jja(ds).sel(year=slice(1950,2023))
        ds_ann.coords['source_id'] = source_id
        return ds_ann

    def process(self):
        ds = [self.reader(f) for f in self.files]
        print(ds)
        return ds

# %%
var = 'ua'
#cmip_files = sorted(glob.glob(f'/project2/tas1/itbaxter/MAPP_HumidityTrends/data/historical/192x288/{var}/*.nc'))
cmip_files = sorted(glob.glob(f'../../data/amip/eq_uwind/ua50*.nc'))

cmip = READER(cmip_files).process()
cmip

cmip_concat = xr.concat(stack_loop(cmip),dim='member_id').squeeze()
cmip_concat

# %%
cmip_concat.member_id.values

# %%
amip_ua = cmip_concat['ua'].sel(lat=slice(-10,10)).mean(dim=['lat','lon']).sel(time=slice('1979-01-01','2014-12-31')).squeeze()


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks

# %%
def compute_power_spectrum(u, c='k', plot=False, label_peaks=False, i=0, label='ERA5'):
    u_anom = u - u.mean(dim='time')
    fs = 1.0  # daily sampling
    freqs, psd = welch(u_anom, fs=fs, nperseg=len(u_anom)//2, detrend='linear')

    psd = psd / np.max(psd)

    # Convert to period in months
    period_months = (1 / freqs)

    # Find valid peaks
    peaks, _ = find_peaks(psd, height=np.percentile(psd, 80))
    valid_peaks = [p for p in peaks if 6 <= period_months[p] <= 100]

    # Sort by PSD height (descending), pick top 2
    top2 = sorted(valid_peaks, key=lambda p: psd[p], reverse=True)[:2]

    # Plot
    if plot:
        if i == 0:
            ax.semilogx(period_months, psd, lw=1.5, c=c, label=label)
        else:
            ax.semilogx(period_months, psd, lw=1.5, c=c)

    #if label_peaks:
        #for p in top2:
            #ax.plot(period_months[p], psd[p], 'o', color=c)
            #ax.text(period_months[p], psd[p]*1.01, f"{period_months[p]:.1f} mo", 
            #         fontsize=8, ha='center', va='bottom', color=c)



    return period_months[:106], psd[:106], period_months[top2] #.values  # return array of peak periods

# Plot
fig = plt.figure(figsize=(10, 5))

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 16

gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], left=0.1, right= 0.9, wspace=0.02, top= 0.95)

ax = plt.subplot(gs[0])

colors = ['tab:orange','#D81B60', '#1E88E5']

period_months,era5_psd,era5_peaks = compute_power_spectrum(era5.resample(time='1MS').mean('time'),c='k',plot=True,label_peaks=True, label='ERA5')

add_curved_arrow_label(ax, (12,0.9), (27.9,1.0), f'ERA5: {28} mo', arrow_color='black',
                connectionstyle="arc3,rad=0.5")

amip_peaks = []
amip_psd = []

for i in range(amip_ua.member_id.size):
    period_months,psd,peaks = compute_power_spectrum(amip_ua.sel(time=slice('1979-01-01','2014-12-31')).isel(member_id=i).dropna('time'),c='tab:orange', i =i, label='AMIP')
    amip_peaks.append(peaks)
    amip_psd.append(xr.DataArray(psd,dims='period'))

ax.semilogx(period_months, (xr.concat(amip_psd,dim='member_id').mean('member_id')), lw=1.5, c='tab:orange',label='AMIP6')

add_curved_arrow_label(ax, (3,0.45), (6,0.41), f'AMIP: 6 mo', arrow_color='tab:orange',
                connectionstyle="arc3,rad=0.5")

add_curved_arrow_label(ax, (57,0.65), (27.9,0.62), f'AMIP: 27.9 mo', arrow_color='tab:orange',
                connectionstyle="arc3,rad=0.5")

csp_peaks = [] 
csp_psd = []
for i in range(csp.member_id.size):
    period_months,psd,peaks = compute_power_spectrum(csp.resample(time='1MS').mean('time').isel(member_id=i).dropna('time'),c='tab:blue', i =i, label='NGCM2.8')
    csp_peaks.append(peaks)
    csp_psd.append(xr.DataArray(psd,dims='period'))

ax.semilogx(period_months, (xr.concat(csp_psd,dim='member_id').mean('member_id')), lw=1.5, c=colors[2],label='NGCM2.8')

add_curved_arrow_label(ax, (3,0.75), (12,np.max((xr.concat(csp_psd,dim='member_id').mean('member_id')))), f'NGCM: 12 mo', arrow_color=colors[2],
                connectionstyle="arc3,rad=0.5")

ace2_peaks = []
ace2_psd = []
for i in range(ace2.member_id.size):
    period_months,psd,peaks = compute_power_spectrum(ace2.resample(time='1MS').mean('time').isel(member_id=i).dropna('time'),c='tab:green', i =i, label='ACE2')
    ace2_peaks.append(peaks)
    ace2_psd.append(xr.DataArray(psd,dims='period'))

ax.semilogx(period_months, (xr.concat(ace2_psd,dim='member_id').mean('member_id')), lw=1.5, c=colors[1],label='ACE2-ERA5')


ax.axvline(28, color='k', linestyle='--',linewidth=0.5)

plt.gca().invert_xaxis()
ax.set_xlabel('Period (months)')
ax.set_ylabel('Power Spectral Density')
#fig.suptitle('QBO Power Spectrum (50 hPa, 10S-10N, Daily Data)')
ax.grid(True, linestyle='--', alpha=0.5)
#ax.legend()
ax.set_xlim([100,0])
ax.set_xticks([6,12,28,50,100])
ax.set_xticklabels([6,12,28,50,100])
ax.text(0.03, 0.9, 'a', fontsize=24, fontweight='bold', transform=ax.transAxes)

ax = plt.subplot(gs[1])
ax.minorticks_on()
ax.set_xscale('log')
ax.set_xticks([6,12,28,50,100])
ax.set_xticklabels([6,12,28,50,100])
ax.text(0.03, 0.9, 'b', fontsize=24, fontweight='bold', transform=ax.transAxes)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

ax.scatter(era5_peaks[0],2**(0.5*era5.std('time')),edgecolor='k',marker='X',s=200,facecolor='k', label='ERA5')

ax.scatter(np.concatenate(amip_peaks)[::2],2**(0.5*amip_ua.std('time')),edgecolor=colors[0],facecolor=colors[0],label='AMIP', lw=1.5)

ax.scatter(np.concatenate(csp_peaks)[::2],2**(0.5*csp.sel(time=slice('1981-01-01','2020-12-31')).std('time')),edgecolor=colors[2],facecolor=colors[2],label='NGCM2.8',lw=1.5)
ax.scatter(np.concatenate(csp_peaks)[::2][2],2**(0.5*csp.isel(member_id=2).sel(time=slice('1981-01-01','2020-12-31')).std('time')),edgecolor='k',facecolor=colors[2],lw=1.5)
ax.scatter(np.concatenate(csp_peaks)[::2][16],2**(0.5*csp.isel(member_id=16).sel(time=slice('1981-01-01','2020-12-31')).std('time')),edgecolor='k',facecolor=colors[2],lw=1.5)

ax.scatter(np.concatenate(ace2_peaks)[::2],2**(0.5*ace2.sel(time=slice('1981-01-01','2020-12-31')).std('time')),edgecolor=colors[1],facecolor=colors[1],label='ACE2-ERA5',lw=1.5)
ax.scatter(np.concatenate(ace2_peaks)[::2][4],2**(0.5*ace2.isel(member_id=4).sel(time=slice('1981-01-01','2020-12-31')).std('time')),edgecolor='k',facecolor=colors[1],lw=1.5)

#ax.scatter(amip_ua.max('time')-amip_ua.min('time'),np.concatenate(amip_peaks)[1::2],edgecolor='tab:orange',facecolor='None')

#ax.scatter(csp.max('time')-csp.min('time'),np.concatenate(csp_peaks)[1::2],edgecolor='tab:blue',facecolor='None')
#ax.scatter(ace2.max('time')-ace2.min('time'),np.concatenate(ace2_peaks)[1::2],edgecolor='tab:green',facecolor='None')
#amip_max = amip_ua.max('time')
#amip_min =  amip_ua.min('time')
amip_rmac = amip_ua.groupby('time.month') - amip_ua.groupby('time.month').mean('time')
amip_std =  amip_rmac.std('time')
add_curved_arrow_label(ax, (9,34), (26,1+2**(0.5*amip_std).sel(member_id='CESM2-WACCM-FV2_r1i1p1f1')-0.5), f'CESM2-WACCM-FV2 \n (nudged)', arrow_color='tab:orange',
                connectionstyle="arc3,rad=-0.5")

#add_curved_arrow_label(ax, (47,6), (26,(0.5*amip_std).sel(member_id='GISS-E2-2-G_r1i1p1f1')-0), f'GISS-E2-2-G \n (prognostic)', arrow_color='tab:orange',
#                connectionstyle="arc3,rad=-0.45")

add_curved_arrow_label(ax, (49,16), (28.5,1+2**(0.5*amip_std).sel(member_id='IPSL-CM6A-LR_r8i1p1f1')-0), f'IPSL-CM6A-LR \n (prognostic)', arrow_color='tab:orange',
                connectionstyle="arc3,rad=0.45")

ax.scatter(27.5,2**(0.5*amip_ua.sel(member_id='IPSL-CM6A-LR_r8i1p1f1').std('time')),edgecolor='k',facecolor=colors[0],lw=1.5)

#add_curved_arrow_label(ax, (32,42), (17,(0.5*amip_std).sel(member_id='E3SM-1-0_r1i1p1f1')-0.5), f'E3SM-1-0', arrow_color='tab:orange',
#                connectionstyle="arc3,rad=-0.5")

add_curved_arrow_label(ax, (28,44), (16.5,2**(0.5*amip_std).sel(member_id='E3SM-1-0_r2i1p1f1')-1), f'E3SM-1-0', arrow_color='tab:orange',
                connectionstyle="arc3,rad=0.55")

ax.legend(loc='upper right',fontsize=9,frameon=False)

ax.set_ylabel(r'Amplitude ($\mathrm{2^{0.5\sigma},\ m\ s^{-1}}$)')
ax.set_xlabel('Dominant Periodicity (months)')
#ax.set_xlim([-10,200])
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_ylim([0,50])
plt.savefig('../../plots/qbo_power_spectrum-month_ensmean.png',dpi=300)

# %%
amip_amp = 0.5 * amip_std.sortby(amip_std,ascending=False)
amip_amp

# %%
amip_max = amip_ua.max('time')
amip_min =  amip_ua.min('time')
for i,m in enumerate(amip_ua.member_id.values):
    print(i, m, amip_peaks[i][0],(amip_max-amip_min).sel(member_id=m).values)

# %%
ace_max = ace2.sel(time=slice('2001-01-01','2010-12-31')).max('time')
ace_min =  ace2.sel(time=slice('2001-01-01','2010-12-31')).min('time') #.min('time')
for i,m in enumerate(ace2.member_id.values):
    print(i, m, ace2_peaks[i][0],(ace_max-ace_min).sel(member_id=m).values)

# %%
ace2_peaks

# %%
member_ids = cmip_concat.member_id.values

for i,member_id in enumerate(member_ids):
    print(member_id, amip_peaks[i])

print(era5_peaks)
# %%
fig = plt.figure(figsize=(10,10))

gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.25)

ax = plt.subplot(gs[0])
#ax.fill_between(ace2['time'],ace2.min('member_id'),ace2.max('member_id'),color='silver',alpha=1.0)
cmip_concat['ua'].sel(lat=slice(-10,10)).mean(dim=['lat','lon']).sel(time=slice('1979-01-01','2014-12-31')).squeeze().sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='silver',alpha=0.5,add_legend=False)
cmip_concat['ua'].sel(lat=slice(-10,10)).mean(dim=['lat','lon']).sel(time=slice('1979-01-01','2014-12-31')).squeeze().mean('member_id').sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='tab:orange',add_legend=False,label='AMIP')
era5.sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='k',add_legend=False,label='ERA5')

ax.axhline(0,color='k',linewidth=0.5,linestyle='--')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Zonal wind speed (m/s)')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
ax.set_ylim(-50,35)
ax.set_title('')

ax = plt.subplot(gs[2])
#ax.fill_between(csp['time'],csp.where(csp.isel(time=slice(-365*20,-1)).mean('time') > 0).min('member_id'),csp.where(csp.isel(time=slice(-365*20,-1)).mean('time') > 0).max('member_id'),color='silver',alpha=1.0)
csp.where(csp.isel(time=slice(-365*20,-1)).mean('time') > 0).sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='silver',alpha=0.5,add_legend=False)
csp.where(csp.isel(time=slice(-365*20,-1)).mean('time') > 0).mean('member_id').sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='tab:blue',add_legend=False,label='NGCM2.8 westerly regime')
era5.sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='k',add_legend=False,label='ERA5')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Zonal wind speed (m/s)')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
ax.set_ylim(-50,35)
ax.set_title('')
ax.axhline(0,color='k',linewidth=0.5,linestyle='--')


ax = plt.subplot(gs[3])
#ax.fill_between(csp['time'],csp.where(csp.isel(time=slice(-365*20,-1)).mean('time') < 0).min('member_id'),csp.where(csp.isel(time=slice(-365*20,-1)).mean('time') < 0).max('member_id'),color='silver',alpha=1.0)
csp.where(csp.isel(time=slice(-365*20,-1)).mean('time') < 0).sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='silver',alpha=0.5,add_legend=False)
csp.where(csp.isel(time=slice(-365*20,-1)).mean('time') < 0).mean('member_id').sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='tab:blue',add_legend=False,label='NGCM2.8 easterly regime')
era5.sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='k',add_legend=False,label='ERA5')

ax.axhline(0,color='k',linewidth=0.5,linestyle='--')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Zonal wind speed (m/s)')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
ax.set_ylim(-50,35)
ax.set_title('')


ax = plt.subplot(gs[1])
#ax.fill_between(ace2['time'],ace2.min('member_id'),ace2.max('member_id'),color='silver',alpha=1.0)
ace2.sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='silver',alpha=0.5,add_legend=False)
ace2.sel(time=slice('1979-01-01','2023-12-31')).mean('member_id').plot.line(x='time',c='tab:green',add_legend=False,label='ACE2')
era5.sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='k',add_legend=False,label='ERA5')

ax.axhline(0,color='k',linewidth=0.5,linestyle='--')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Zonal wind speed (m/s)')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
ax.set_ylim(-50,35)
ax.set_title('')



#plt.tight_layout()
plt.savefig('/scratch/midway2/itbaxter/NeuralGCM_Decadal_Simulations/plots/qbo_time_series.png',dpi=300)

# %%
cmip_concat['ua'].sel(lat=slice(-5,5)).mean(dim=['lat','lon']).sel(time=slice('1979-01-01','2014-12-31')).squeeze().plot.line(x='time',add_legend=False)

# %%
fig = plt.figure(figsize=(7.5,8))

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 14

gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], top=0.98, left=0.12, right=0.98, bottom=0.11, hspace=0.15)

ax = plt.subplot(gs[0])
#ax.fill_between(ace2['time'],ace2.min('member_id'),ace2.max('member_id'),color='silver',alpha=1.0)
#cmip_concat['ua'].sel(lat=slice(-10,10)).mean(dim=['lat','lon']).sel(time=slice('1979-01-01','2014-12-31')).squeeze().sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='silver',alpha=0.5,add_legend=False)
#cmip_concat['ua'].sel(lat=slice(-10,10)).mean(dim=['lat','lon']).sel(time=slice('1979-01-01','2014-12-31')).squeeze().mean('member_id').sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='tab:orange',add_legend=False,label='AMIP')
era5.sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='k',add_legend=False,label='ERA5')
cmip_concat['ua'].sel(lat=slice(-10,10)).mean(dim=['lat','lon']).sel(time=slice('1979-01-01','2014-12-31')).squeeze().sel(member_id='CESM2_r10i1p1f1').sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time', linestyle='--',c='tab:orange',add_legend=False,label='CESM2')
#cmip_concat['ua'].sel(lat=slice(-10,10)).mean(dim=['lat','lon']).sel(time=slice('1979-01-01','2014-12-31')).squeeze().sel(member_id='GISS-E2-2-G_r1i1p1f1').sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',linestyle='-',c='tab:orange',add_legend=False,label='GISS-E2-2-G')

cmip_concat['ua'].sel(lat=slice(-10,10)).mean(dim=['lat','lon']).sel(time=slice('1979-01-01','2014-12-31')).squeeze().sel(member_id='IPSL-CM6A-LR_r8i1p1f1').sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',linestyle='-',c='tab:orange',add_legend=False,label='IPSL-CM6A-LR')

ax.axhline(0,color='k',linewidth=0.5,linestyle='--')
ax.set_ylabel(' ')
#ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(ncols=3,frameon=False,bbox_to_anchor=(0.9, 0.25))
ax.set_ylim(-50,35)
ax.set_title('')
ax.text(0.01, 0.9, 'a', fontsize=18, fontweight='bold', transform=ax.transAxes)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xlabel(' ')
ax.set_xticks([])
ax.set_xticklabels([])
ax.minorticks_on()

ax = plt.subplot(gs[1])
validation = np.arange('1996-01-01','2001-01-01',dtype='datetime64[M]')
training = np.arange('2001-01-01','2010-12-31',dtype='datetime64[M]')
training2 = np.arange('2019-01-01','2020-12-31',dtype='datetime64[M]')
ax.fill_between(validation, np.zeros(len(validation))-40, np.zeros(len(validation))+25, color='silver', alpha=0.3)
ax.fill_between(training, np.zeros(len(training))-40, np.zeros(len(training))+25, color='silver', alpha=0.5)
ax.fill_between(training2, np.zeros(len(training2))-40, np.zeros(len(training2))+25, color='silver', alpha=0.5)

#ax.fill_between(ace2['time'],ace2.min('member_id'),ace2.max('member_id'),color='silver',alpha=1.0)
era5.sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='k',add_legend=False)

ace2.sel(time=slice('1979-01-01','2023-12-31')).sel(member_id=4).plot.line(x='time',c='#D81B60',add_legend=False,label='ACE2-ERA5')
#ace2.sel(time=slice('1979-01-01','2023-12-31')).sel(member_id=2).plot.line(x='time',c='#D81B60',add_legend=False,label='ACE2-ERA5')

ax.text(0.01, 0.9, 'b', fontsize=18, fontweight='bold', transform=ax.transAxes)
ax.text(0.4, 0.9, 'tuning', fontsize=8, transform=ax.transAxes)
ax.text(0.5, 0.9, 'testing', fontsize=8, transform=ax.transAxes)

ax.axhline(0,color='k',linewidth=0.5,linestyle='--')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Tropical zonal mean zonal wind speed (m/s)')
#ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(ncols=2,frameon=False,bbox_to_anchor=(0.45, 0.07))
ax.set_ylim(-50,35)
ax.set_title('')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.minorticks_on()
ax.set_xlabel(' ')
ax.set_xticks([])
ax.set_xticklabels([])


ax = plt.subplot(gs[2])
training = np.arange('2018-01-01','2024-01-01',dtype='datetime64[M]')
ax.fill_between(training, np.zeros(len(training))-50, np.zeros(len(training))+25, color='silver', alpha=0.5)
#ax.fill_between(csp['time'],csp.where(csp.isel(time=slice(-365*20,-1)).mean('time') < 0).min('member_id'),csp.where(csp.isel(time=slice(-365*20,-1)).mean('time') < 0).max('member_id'),color='silver',alpha=1.0)
#csp.where(csp.isel(time=slice(-365*20,-1)).mean('time') < 0).sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='silver',alpha=0.5,add_legend=False)
#csp.where(csp.isel(time=slice(-365*20,-1)).mean('time') < 0).mean('member_id').sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='tab:blue',add_legend=False,label='NGCM2.8 easterly regime')
csp.isel(member_id=16).sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='#1E88E5',add_legend=False)
csp.where(csp.isel(time=slice(-365*20,-1)).mean('time') < 0).isel(member_id=2).sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='#1E88E5',add_legend=False,label='NGCM2.8')
#csp.where(csp.isel(time=slice(-365*20,-1)).mean('time') < 0).isel(member_id=0).sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='tab:blue',add_legend=False)

era5.sel(time=slice('1979-01-01','2023-12-31')).plot.line(x='time',c='k',add_legend=False)

ax.text(0.835, 0.9, 'testing', fontsize=8, transform=ax.transAxes)
ax.text(0.07, 0.89, 'westerly \n member (1 of 8)', fontsize=8, transform=ax.transAxes)
ax.text(0.005, 0.1, 'easterly \n member \n (1 of 29)', fontsize=8, transform=ax.transAxes)

#add_curved_arrow_label(ax, (2012,20), ((2009-1980)*12+6,csp.isel(member_id=16).sel(time='2009-06-01')), f'', arrow_color='tab:blue',
#                connectionstyle="arc3,rad=0.5")


#add_curved_arrow_label(ax, (32,43), (16,(amip_max-amip_min).sel(member_id='E3SM-1-0_r2i1p1f1')-0.5), f'', arrow_color='tab:blue',
#                connectionstyle="arc3,rad=0.5")

ax.axhline(0,color='k',linewidth=0.5,linestyle='--')
ax.set_xlabel('Time (years)')
ax.set_ylabel('')
#ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(ncols=2, bbox_to_anchor=(0.55, 0.95),frameon=False)
ax.set_ylim(-50,35)
ax.set_title('')
ax.text(0.01, 0.9, 'c', fontsize=18, fontweight='bold', transform=ax.transAxes)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.minorticks_on()

plt.tight_layout()
plt.savefig('/scratch/midway2/itbaxter/NeuralGCM_Decadal_Simulations/plots/qbo_time_series-2member.png',dpi=500)
# %%
fig = plt.figure(figsize=(55,25))

p = ace2.sel(time=slice('1979-01-01','2023-12-31')).plot.line(col='member_id',col_wrap=4,x='time',c='tab:green',alpha=0.5,add_legend=False)
for ax in p.axes.flat:
    ax.axhline(0,linestyle='--',c='k',linewidth=0.7)
   #era5.sel(time=slice('1979-01-01','2023-12-31')).plot.line(ax=ax,x='time',c='k',add_legend=False,label='ERA5') 

plt.savefig('/scratch/midway2/itbaxter/NeuralGCM_Decadal_Simulations/plots/qbo-ace2.png')

