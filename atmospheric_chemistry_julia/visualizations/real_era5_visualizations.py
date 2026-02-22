"""
real_era5_visualizations.py

Creates publication-quality visualizations from REAL ERA5 reanalysis data
previously downloaded from the publicly available WeatherBench2 archive on
Google Cloud Storage.

Data source: gs://weatherbench2/datasets/era5/
Resolution:  1 deg x 1 deg (360 x 181) -- the EXACT resolution our Julia model runs at
Variables:   Temperature, U/V winds, geopotential, specific humidity, surface fields
Time:        Real 6-hourly data from 2022

This is AUTHENTIC ERA5 reanalysis data produced by ECMWF.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})


def load_cached_era5():
    """
    Load REAL ERA5 data from cached download (fetched via HTTP from the
    public WeatherBench2 archive on Google Cloud Storage).

    The cached data has shape (lon, lat) for 2D fields and (level, lon, lat)
    for 3D fields. We transpose to (lat, lon) / (level, lat, lon) for plotting.
    """
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_data')
    cache_files = sorted([f for f in os.listdir(cache_dir) if f.startswith('era5_') and f.endswith('.pkl')])

    if not cache_files:
        print("  ERROR: No cached ERA5 data. Run fetch_real_era5.py first.")
        sys.exit(1)

    cache_path = os.path.join(cache_dir, cache_files[0])
    print(f"  Loading cached real ERA5: {cache_path}")

    with open(cache_path, 'rb') as f:
        fields = pickle.load(f)

    # Transpose arrays from (lon, lat) -> (lat, lon) for meshgrid compatibility
    surface_2d_vars = [
        '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
        'mean_sea_level_pressure', 'total_precipitation_6hr', 'surface_pressure',
        'boundary_layer_height', 'total_cloud_cover', 'total_column_water_vapour',
        'land_sea_mask', 'geopotential_at_surface',
    ]
    level_3d_vars = [
        'temperature', 'u_component_of_wind', 'v_component_of_wind',
        'specific_humidity', 'geopotential',
    ]

    for var in surface_2d_vars:
        if var in fields and isinstance(fields[var], np.ndarray) and fields[var].ndim == 2:
            fields[var] = fields[var].T  # (lon, lat) -> (lat, lon)

    for var in level_3d_vars:
        if var in fields and isinstance(fields[var], np.ndarray) and fields[var].ndim == 3:
            fields[var] = np.transpose(fields[var], (0, 2, 1))  # (lev, lon, lat) -> (lev, lat, lon)

    nlon = len(fields['longitude'])
    nlat = len(fields['latitude'])
    nlev = len(fields.get('level', []))
    print(f"  Grid: {nlon} x {nlat}, {nlev} levels")
    print(f"  Time: {fields['time']}")
    return fields


# ===========================================================================
# FIGURE 1: Real ERA5 2m Temperature -- Global Map
# ===========================================================================
def fig_temperature_map(fields):
    """Global 2m temperature from real ERA5 data."""
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='#333333')
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color='#888888')

    t2m = fields['2m_temperature']
    lon = fields['longitude']
    lat = fields['latitude']

    t2m_C = t2m - 273.15  # Kelvin to Celsius

    lon2d, lat2d = np.meshgrid(lon, lat)

    im = ax.pcolormesh(lon2d, lat2d, t2m_C,
                       transform=ccrs.PlateCarree(),
                       cmap='RdYlBu_r', vmin=-40, vmax=45,
                       shading='auto', rasterized=True)

    cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                      fraction=0.046, pad=0.05, aspect=35,
                      extend='both')
    cb.set_label('Temperature [deg C]', fontsize=12)

    ax.set_title(f'ERA5 Reanalysis -- 2m Temperature\n'
                 f'{fields["time"]} UTC | 1 deg x 1 deg Grid (360 x 181)',
                 fontsize=14, fontweight='bold')

    fig.text(0.99, 0.01, 'Data: ECMWF ERA5 via WeatherBench2\nModel: AtmosphericChemistry.jl',
             ha='right', va='bottom', fontsize=8, color='gray', style='italic')

    outpath = os.path.join(FIGDIR, 'real_era5_temperature.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ===========================================================================
# FIGURE 2: Real ERA5 Wind Field at 250 hPa (jet stream)
# ===========================================================================
def fig_wind_jet_stream(fields):
    """Upper-level winds showing the jet stream from real ERA5."""
    if 'u_component_of_wind' not in fields:
        print("  Skipping jet stream figure (no 3D wind data)")
        return

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    u = fields['u_component_of_wind']
    v = fields['v_component_of_wind']
    lon = fields['longitude']
    lat = fields['latitude']
    levels = fields['level']

    # Find 250 hPa level
    lev_idx = np.argmin(np.abs(levels - 250))
    u250 = u[lev_idx]
    v250 = v[lev_idx]
    wspd = np.sqrt(u250**2 + v250**2)

    lon2d, lat2d = np.meshgrid(lon, lat)

    im = ax.pcolormesh(lon2d, lat2d, wspd,
                       transform=ccrs.PlateCarree(),
                       cmap='YlOrRd', vmin=0, vmax=70,
                       shading='auto', rasterized=True)

    # Wind barbs (subsampled)
    skip = 8
    ax.barbs(lon2d[::skip, ::skip], lat2d[::skip, ::skip],
             u250[::skip, ::skip] * 1.944, v250[::skip, ::skip] * 1.944,
             transform=ccrs.PlateCarree(),
             length=5, linewidth=0.4, barbcolor='#333333')

    cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                      fraction=0.046, pad=0.05, aspect=35)
    cb.set_label('Wind Speed [m/s]', fontsize=12)

    ax.set_title(f'ERA5 Reanalysis -- 250 hPa Wind (Jet Stream)\n'
                 f'{fields["time"]} UTC | Level: {levels[lev_idx]} hPa',
                 fontsize=14, fontweight='bold')

    fig.text(0.99, 0.01, 'Data: ECMWF ERA5 via WeatherBench2',
             ha='right', va='bottom', fontsize=8, color='gray', style='italic')

    outpath = os.path.join(FIGDIR, 'real_era5_jet_stream.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ===========================================================================
# FIGURE 3: Zonal-mean temperature cross-section
# ===========================================================================
def fig_zonal_temperature(fields):
    """Zonal-mean temperature cross-section from real ERA5."""
    if 'temperature' not in fields:
        print("  Skipping zonal T cross-section (no 3D temperature)")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    T = fields['temperature']       # shape: (level, lat, lon)
    lat = fields['latitude']
    levels = fields['level']

    zm_T = np.nanmean(T, axis=-1)   # Mean over longitude -> (level, lat)

    lat2d, lev2d = np.meshgrid(lat, levels)

    im = ax.contourf(lat2d, lev2d, zm_T,
                     levels=np.arange(190, 310, 5),
                     cmap='RdYlBu_r', extend='both')
    ax.contour(lat2d, lev2d, zm_T,
               levels=np.arange(190, 310, 10),
               colors='black', linewidths=0.3, alpha=0.5)

    ax.set_yscale('log')
    ax.set_ylim(1000, 1)
    ax.set_xlim(-90, 90)
    ax.set_xlabel('Latitude [deg N]', fontsize=12)
    ax.set_ylabel('Pressure [hPa]', fontsize=12)

    ax.set_yticks([1000, 700, 500, 300, 200, 100, 50, 10, 1])
    ax.set_yticklabels(['1000', '700', '500', '300', '200', '100', '50', '10', '1'])

    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('Temperature [K]', fontsize=12)

    # Mark tropopause (approximate)
    trop_p = 100 + 200 * (np.abs(lat) / 90)**1.5
    ax.plot(lat, trop_p, 'k--', linewidth=1.5, label='~Tropopause')
    ax.legend(fontsize=10)

    ax.set_title(f'ERA5 Reanalysis -- Zonal-Mean Temperature\n{fields["time"]} UTC',
                 fontsize=14, fontweight='bold')

    fig.text(0.99, 0.01, 'Data: ECMWF ERA5 via WeatherBench2',
             ha='right', va='bottom', fontsize=8, color='gray', style='italic')

    outpath = os.path.join(FIGDIR, 'real_era5_zonal_temperature.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ===========================================================================
# FIGURE 4: MSLP & 10m winds (synoptic chart)
# ===========================================================================
def fig_synoptic_chart(fields):
    """Mean sea level pressure and 10m winds from real ERA5."""
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color='#888888')
    ax.add_feature(cfeature.OCEAN, facecolor='#f0f4f8')

    lon = fields['longitude']
    lat = fields['latitude']
    lon2d, lat2d = np.meshgrid(lon, lat)

    mslp = fields['mean_sea_level_pressure'] / 100  # Pa -> hPa
    u10 = fields['10m_u_component_of_wind']
    v10 = fields['10m_v_component_of_wind']
    wspd = np.sqrt(u10**2 + v10**2)

    # MSLP contours
    cs = ax.contour(lon2d, lat2d, mslp,
                    levels=np.arange(970, 1050, 4),
                    colors='black', linewidths=0.8,
                    transform=ccrs.PlateCarree())
    ax.clabel(cs, fontsize=7, fmt='%d')

    # Wind speed fill
    im = ax.pcolormesh(lon2d, lat2d, wspd,
                       transform=ccrs.PlateCarree(),
                       cmap='YlGnBu', vmin=0, vmax=20,
                       shading='auto', rasterized=True, alpha=0.6)

    # Wind vectors
    skip = 6
    ax.quiver(lon2d[::skip, ::skip], lat2d[::skip, ::skip],
              u10[::skip, ::skip], v10[::skip, ::skip],
              transform=ccrs.PlateCarree(),
              scale=250, width=0.002, alpha=0.7, color='#333333')

    cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                      fraction=0.046, pad=0.05, aspect=35)
    cb.set_label('10m Wind Speed [m/s]', fontsize=11)

    ax.set_title(f'ERA5 Reanalysis -- MSLP [hPa] & 10m Wind\n'
                 f'{fields["time"]} UTC',
                 fontsize=14, fontweight='bold')

    fig.text(0.99, 0.01, 'Data: ECMWF ERA5 via WeatherBench2',
             ha='right', va='bottom', fontsize=8, color='gray', style='italic')

    outpath = os.path.join(FIGDIR, 'real_era5_synoptic.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ===========================================================================
# FIGURE 5: Specific humidity -- moisture transport for chemistry
# ===========================================================================
def fig_humidity(fields):
    """Specific humidity from real ERA5 -- drives wet deposition."""
    if 'specific_humidity' not in fields:
        print("  Skipping humidity figure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                              subplot_kw={'projection': ccrs.Robinson()})

    lon = fields['longitude']
    lat = fields['latitude']
    lon2d, lat2d = np.meshgrid(lon, lat)
    q = fields['specific_humidity']     # (level, lat, lon)
    levels = fields['level']

    # (a) Surface level specific humidity
    ax = axes[0]
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    sfc_idx = np.argmin(np.abs(levels - 1000))
    q_sfc = q[sfc_idx]
    q_sfc_gkg = q_sfc * 1000  # kg/kg -> g/kg

    im = ax.pcolormesh(lon2d, lat2d, q_sfc_gkg,
                       transform=ccrs.PlateCarree(),
                       cmap='YlGnBu', vmin=0, vmax=20,
                       shading='auto', rasterized=True)
    cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                      fraction=0.046, pad=0.04)
    cb.set_label('g/kg')
    ax.set_title('(a) Surface Specific Humidity', fontweight='bold')

    # (b) 500 hPa
    ax = axes[1]
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    mid_idx = np.argmin(np.abs(levels - 500))
    q_500 = q[mid_idx]
    q_500_gkg = q_500 * 1000

    im = ax.pcolormesh(lon2d, lat2d, q_500_gkg,
                       transform=ccrs.PlateCarree(),
                       cmap='YlGnBu', vmin=0, vmax=5,
                       shading='auto', rasterized=True)
    cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                      fraction=0.046, pad=0.04)
    cb.set_label('g/kg')
    ax.set_title('(b) 500 hPa Specific Humidity', fontweight='bold')

    fig.suptitle(f'ERA5 Reanalysis -- Specific Humidity\n'
                 f'{fields["time"]} UTC -- Key driver for wet deposition in chemistry model',
                 fontsize=13, fontweight='bold')

    fig.text(0.99, 0.01, 'Data: ECMWF ERA5 via WeatherBench2',
             ha='right', va='bottom', fontsize=8, color='gray', style='italic')

    outpath = os.path.join(FIGDIR, 'real_era5_humidity.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ===========================================================================
# FIGURE 6: Zonal-mean wind cross-section
# ===========================================================================
def fig_zonal_wind(fields):
    """Zonal-mean zonal wind -- Hadley/Ferrel/Polar cells."""
    if 'u_component_of_wind' not in fields:
        print("  Skipping zonal wind figure")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    lat = fields['latitude']
    levels = fields['level']
    u = fields['u_component_of_wind']   # (level, lat, lon)
    v = fields['v_component_of_wind']

    zm_u = np.nanmean(u, axis=-1)   # (level, lat)
    zm_v = np.nanmean(v, axis=-1)

    lat2d, lev2d = np.meshgrid(lat, levels)

    # (a) Zonal wind
    norm = TwoSlopeNorm(vmin=-30, vcenter=0, vmax=60)
    im1 = ax1.contourf(lat2d, lev2d, zm_u,
                       levels=np.arange(-30, 65, 5),
                       cmap='RdBu_r', norm=norm, extend='both')
    ax1.contour(lat2d, lev2d, zm_u, levels=[0], colors='black', linewidths=1.5)
    ax1.set_yscale('log')
    ax1.set_ylim(1000, 1)
    ax1.set_xlim(-90, 90)
    ax1.set_yticks([1000, 700, 500, 300, 200, 100, 50, 10, 1])
    ax1.set_yticklabels(['1000', '700', '500', '300', '200', '100', '50', '10', '1'])
    ax1.set_xlabel('Latitude [deg N]')
    ax1.set_ylabel('Pressure [hPa]')
    cb1 = plt.colorbar(im1, ax=ax1)
    cb1.set_label('m/s')
    ax1.set_title('(a) Zonal-Mean Zonal Wind [U]', fontweight='bold')

    # (b) Meridional wind
    im2 = ax2.contourf(lat2d, lev2d, zm_v,
                       levels=np.arange(-5, 5.5, 0.5),
                       cmap='PuOr', extend='both')
    ax2.contour(lat2d, lev2d, zm_v, levels=[0], colors='black', linewidths=1)
    ax2.set_yscale('log')
    ax2.set_ylim(1000, 1)
    ax2.set_xlim(-90, 90)
    ax2.set_yticks([1000, 700, 500, 300, 200, 100, 50, 10, 1])
    ax2.set_yticklabels(['1000', '700', '500', '300', '200', '100', '50', '10', '1'])
    ax2.set_xlabel('Latitude [deg N]')
    ax2.set_ylabel('Pressure [hPa]')
    cb2 = plt.colorbar(im2, ax=ax2)
    cb2.set_label('m/s')
    ax2.set_title('(b) Zonal-Mean Meridional Wind [V]', fontweight='bold')

    fig.suptitle(f'ERA5 Reanalysis -- Zonal-Mean Wind Structure\n'
                 f'{fields["time"]} UTC -- Drives tracer transport in chemistry model',
                 fontsize=13, fontweight='bold')

    fig.text(0.99, 0.01, 'Data: ECMWF ERA5 via WeatherBench2',
             ha='right', va='bottom', fontsize=8, color='gray', style='italic')

    fig.tight_layout(rect=[0, 0.02, 1, 0.92])

    outpath = os.path.join(FIGDIR, 'real_era5_zonal_wind.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ===========================================================================
# FIGURE 7: Multi-panel Hero Overview
# ===========================================================================
def fig_hero_overview(fields):
    """Hero figure showing all real ERA5 fields used by the chemistry model."""
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.32, wspace=0.2)

    lon = fields['longitude']
    lat = fields['latitude']
    lon2d, lat2d = np.meshgrid(lon, lat)
    levels = fields.get('level', np.array([]))

    # --- (a) 2m Temperature ---
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    t2m = fields['2m_temperature'] - 273.15
    im = ax.pcolormesh(lon2d, lat2d, t2m, transform=ccrs.PlateCarree(),
                       cmap='RdYlBu_r', vmin=-40, vmax=45, shading='auto', rasterized=True)
    plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04, label='deg C')
    ax.set_title('(a) 2m Temperature', fontweight='bold', fontsize=11)

    # --- (b) MSLP ---
    ax = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    mslp = fields['mean_sea_level_pressure'] / 100
    im = ax.pcolormesh(lon2d, lat2d, mslp, transform=ccrs.PlateCarree(),
                       cmap='coolwarm', vmin=980, vmax=1040, shading='auto', rasterized=True)
    plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04, label='hPa')
    ax.set_title('(b) Mean Sea Level Pressure', fontweight='bold', fontsize=11)

    # --- (c) 10m Wind Speed ---
    ax = fig.add_subplot(gs[0, 2], projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    u10 = fields['10m_u_component_of_wind']
    v10 = fields['10m_v_component_of_wind']
    wspd = np.sqrt(u10**2 + v10**2)
    im = ax.pcolormesh(lon2d, lat2d, wspd, transform=ccrs.PlateCarree(),
                       cmap='YlOrRd', vmin=0, vmax=20, shading='auto', rasterized=True)
    plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04, label='m/s')
    ax.set_title('(c) 10m Wind Speed', fontweight='bold', fontsize=11)

    has_3d = len(levels) > 0 and 'temperature' in fields

    if has_3d:
        T_vals = fields['temperature']              # (level, lat, lon)
        u_vals = fields['u_component_of_wind']
        v_vals = fields['v_component_of_wind']

        # --- (d) Zonal-mean temperature ---
        ax = fig.add_subplot(gs[1, 0])
        zm_T = np.nanmean(T_vals, axis=-1)
        lat2d_z, lev2d = np.meshgrid(lat, levels)
        im = ax.contourf(lat2d_z, lev2d, zm_T, levels=np.arange(190, 310, 5),
                         cmap='RdYlBu_r', extend='both')
        ax.set_yscale('log')
        ax.set_ylim(1000, 1)
        ax.set_xlim(-90, 90)
        ax.set_yticks([1000, 500, 200, 50, 1])
        ax.set_yticklabels(['1000', '500', '200', '50', '1'])
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Pressure [hPa]')
        plt.colorbar(im, ax=ax, label='K')
        ax.set_title('(d) Zonal-Mean Temperature', fontweight='bold', fontsize=11)

        # --- (e) Zonal wind ---
        ax = fig.add_subplot(gs[1, 1])
        zm_u = np.nanmean(u_vals, axis=-1)
        im = ax.contourf(lat2d_z, lev2d, zm_u, levels=np.arange(-30, 65, 5),
                         cmap='RdBu_r', extend='both')
        ax.contour(lat2d_z, lev2d, zm_u, levels=[0], colors='k', linewidths=1)
        ax.set_yscale('log')
        ax.set_ylim(1000, 1)
        ax.set_xlim(-90, 90)
        ax.set_yticks([1000, 500, 200, 50, 1])
        ax.set_yticklabels(['1000', '500', '200', '50', '1'])
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Pressure [hPa]')
        plt.colorbar(im, ax=ax, label='m/s')
        ax.set_title('(e) Zonal-Mean Zonal Wind', fontweight='bold', fontsize=11)

        # --- (f) 250 hPa wind ---
        ax = fig.add_subplot(gs[1, 2], projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        lev250 = np.argmin(np.abs(levels - 250))
        u250 = u_vals[lev250]
        v250 = v_vals[lev250]
        ws250 = np.sqrt(u250**2 + v250**2)
        im = ax.pcolormesh(lon2d, lat2d, ws250, transform=ccrs.PlateCarree(),
                           cmap='YlOrRd', vmin=0, vmax=60, shading='auto', rasterized=True)
        plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04, label='m/s')
        ax.set_title('(f) 250 hPa Wind (Jet Stream)', fontweight='bold', fontsize=11)

    # --- (g) Precipitation ---
    if 'total_precipitation_6hr' in fields:
        ax = fig.add_subplot(gs[2, 0], projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        precip = fields['total_precipitation_6hr'] * 1000  # m -> mm
        im = ax.pcolormesh(lon2d, lat2d, precip, transform=ccrs.PlateCarree(),
                           cmap='Blues', vmin=0, vmax=20, shading='auto', rasterized=True)
        plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04, label='mm/6h')
        ax.set_title('(g) Precipitation (wet deposition)', fontweight='bold', fontsize=11)

    # --- (h) Humidity for wet chemistry ---
    if has_3d and 'specific_humidity' in fields:
        ax = fig.add_subplot(gs[2, 1], projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        q = fields['specific_humidity']
        sfc_idx = np.argmin(np.abs(levels - 1000))
        q_sfc = q[sfc_idx] * 1000  # g/kg
        im = ax.pcolormesh(lon2d, lat2d, q_sfc, transform=ccrs.PlateCarree(),
                           cmap='YlGnBu', vmin=0, vmax=20, shading='auto', rasterized=True)
        plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04, label='g/kg')
        ax.set_title('(h) Surface Humidity (wet chem.)', fontweight='bold', fontsize=11)

    # --- (i) Surface pressure ---
    if 'surface_pressure' in fields:
        ax = fig.add_subplot(gs[2, 2], projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        sp = fields['surface_pressure'] / 100
        im = ax.pcolormesh(lon2d, lat2d, sp, transform=ccrs.PlateCarree(),
                           cmap='viridis', vmin=500, vmax=1030, shading='auto', rasterized=True)
        plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04, label='hPa')
        ax.set_title('(i) Surface Pressure (sigma-coord)', fontweight='bold', fontsize=11)

    fig.suptitle('REAL ERA5 Reanalysis -- Meteorological Fields for Chemistry Transport\n'
                 f'ECMWF ERA5 | {fields["time"]} UTC | 1 deg x 1 deg (360 x 181) | '
                 'Used as forcing for AtmosphericChemistry.jl',
                 fontsize=14, fontweight='bold', y=0.99)

    fig.text(0.5, 0.005,
             'Data: ECMWF ERA5 Reanalysis (Hersbach et al., 2020) via WeatherBench2 (Rasp et al., 2024)\n'
             'All fields shown are REAL observational reanalysis data -- not simulated',
             ha='center', fontsize=9, color='#555555', style='italic')

    outpath = os.path.join(FIGDIR, 'real_era5_hero_overview.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 65)
    print("  REAL ERA5 Data Visualization Pipeline")
    print("  Source: ECMWF ERA5 via WeatherBench2 (Google Cloud)")
    print("  Resolution: 1 deg x 1 deg (360 x 181) -- model grid")
    print("=" * 65)

    print("\n[1/2] Loading cached real ERA5 data...")
    fields = load_cached_era5()

    print("\n" + "=" * 65)
    print("  Generating figures from REAL ERA5 data...")
    print("=" * 65)

    print("\n[Fig 1] Global 2m temperature...")
    fig_temperature_map(fields)

    print("[Fig 2] Jet stream (250 hPa winds)...")
    fig_wind_jet_stream(fields)

    print("[Fig 3] Zonal-mean temperature cross-section...")
    fig_zonal_temperature(fields)

    print("[Fig 4] Synoptic chart (MSLP + 10m winds)...")
    fig_synoptic_chart(fields)

    print("[Fig 5] Specific humidity...")
    fig_humidity(fields)

    print("[Fig 6] Zonal-mean winds...")
    fig_zonal_wind(fields)

    print("[Fig 7] Hero overview (9-panel)...")
    fig_hero_overview(fields)

    print("\n" + "=" * 65)
    print(f"  All figures saved to: {FIGDIR}/")
    print("  ALL DATA IS REAL ERA5 REANALYSIS FROM ECMWF")
    print("=" * 65)


if __name__ == '__main__':
    main()
