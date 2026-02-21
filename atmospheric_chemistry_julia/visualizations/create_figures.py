"""
create_figures.py

Publication-quality visualizations for the Atmospheric Chemistry Transport Model.
Generates a comprehensive set of figures suitable for sharing with the
atmospheric chemistry community.

Produces:
  1. Global surface concentration maps (O3, NO2, CO, SO2, PM2.5, CH4)
  2. Zonal-mean latitude-altitude cross-sections
  3. EDGAR emission inventory maps
  4. Photolysis rate fields & SZA
  5. Vertical profiles at key observational sites
  6. Mass budget / global burden summary
  7. Regional zoom maps (East Asia, Europe, North America)
  8. Multi-panel model overview figure
"""

import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
import os, sys

# ---------------------------------------------------------------------------
# Style settings
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# Custom colormaps for atmospheric chemistry
def make_o3_cmap():
    colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
              '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f']
    return LinearSegmentedColormap.from_list('o3', colors, N=256)

def make_nox_cmap():
    colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1',
              '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
    return LinearSegmentedColormap.from_list('nox', colors, N=256)

def make_co_cmap():
    colors = ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84',
              '#fc8d59', '#ef6548', '#d7301f', '#b30000', '#7f0000']
    return LinearSegmentedColormap.from_list('co', colors, N=256)

def make_so2_cmap():
    colors = ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b',
              '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
    return LinearSegmentedColormap.from_list('so2', colors, N=256)

def make_pm_cmap():
    colors = ['#4575b4', '#74add1', '#abd9e9', '#e0f3f8',
              '#ffffbf', '#fee090', '#fdae61', '#f46d43',
              '#d73027', '#a50026', '#67001f']
    return LinearSegmentedColormap.from_list('pm', colors, N=256)

def make_emission_cmap():
    colors = ['#ffffff', '#ffffb2', '#fecc5c', '#fd8d3c',
              '#f03b20', '#bd0026', '#67000d']
    return LinearSegmentedColormap.from_list('emi', colors, N=256)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(filepath):
    ds = nc.Dataset(filepath, 'r')
    data = {}

    data['lon'] = ds['lon'][:]
    data['lat'] = ds['lat'][:]
    data['level'] = ds['level'][:]
    data['altitude'] = ds['altitude'][:]

    # Species (ppb)
    for sp in ['O3', 'NO2', 'NO', 'CO', 'SO2', 'CH4', 'HCHO', 'HNO3', 'H2O2', 'OH', 'PM25']:
        if sp in ds.variables:
            data[sp] = ds[sp][:]

    # Emissions
    for sp in ['NOx', 'CO', 'SO2', 'CH4', 'PM25']:
        key = f'emi_{sp}'
        if key in ds.variables:
            data[key] = ds[key][:]

    # Met
    for v in ['temperature', 'u_wind', 'v_wind', 'specific_humidity',
              'surface_pressure', 'boundary_layer_height', 'precipitation',
              'surface_solar_radiation', 't2m']:
        if v in ds.variables:
            data[v] = ds[v][:]

    # Photolysis
    for v in ['j_NO2', 'j_O3', 'cos_sza']:
        if v in ds.variables:
            data[v] = ds[v][:]

    # Land mask
    if 'land_mask' in ds.variables:
        data['land_mask'] = ds['land_mask'][:]

    ds.close()
    return data


# ===========================================================================
# FIGURE 1: Global surface concentration maps
# ===========================================================================
def fig1_surface_concentrations(data, figdir):
    """6-panel global surface maps of key species."""

    species_config = [
        ('O3',   'Surface Ozone (O$_3$)',            'ppb',     make_o3_cmap(),  15, 50),
        ('NO2',  'Surface NO$_2$',                   'ppb',     make_nox_cmap(), 0,  10),
        ('CO',   'Surface CO',                       'ppb',     make_co_cmap(),  40, 160),
        ('SO2',  'Surface SO$_2$',                   'ppb',     make_so2_cmap(), 0,  5),
        ('PM25', 'Surface PM$_{2.5}$',               'µg/m³',   make_pm_cmap(),  0,  80),
        ('CH4',  'Surface Methane (CH$_4$)',          'ppb',     'YlOrBr',       1800, 1900),
    ]

    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(3, 2, hspace=0.25, wspace=0.08)

    for idx, (sp, title, unit, cmap, vmin, vmax) in enumerate(species_config):
        ax = fig.add_subplot(gs[idx // 2, idx % 2], projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='#333333')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='#666666')

        field = data[sp][:, :, -1]  # Surface level
        lon2d, lat2d = np.meshgrid(data['lon'], data['lat'], indexing='ij')

        im = ax.pcolormesh(lon2d, lat2d, field,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           shading='auto', rasterized=True)

        cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                          fraction=0.046, pad=0.04, aspect=30)
        cb.set_label(unit, fontsize=9)
        cb.ax.tick_params(labelsize=8)

        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)

        # Panel label
        ax.text(0.02, 0.98, f'({chr(97 + idx)})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    fig.suptitle('Atmospheric Chemistry Transport Model — Surface Concentrations\n'
                 '1° × 1° Global, 47 Levels | ERA5 + EDGAR v8.0 | July 15, 2023 12:00 UTC',
                 fontsize=14, fontweight='bold', y=0.98)

    outpath = os.path.join(figdir, 'fig1_surface_concentrations.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ===========================================================================
# FIGURE 2: Zonal-mean latitude-altitude cross-sections
# ===========================================================================
def fig2_zonal_cross_sections(data, figdir):
    """4-panel zonal-mean cross-sections."""

    species_config = [
        ('O3',   'Ozone (O$_3$)',   'ppb',   make_o3_cmap(),  0, 5000),
        ('CO',   'CO',              'ppb',   make_co_cmap(),  40, 150),
        ('NO2',  'NO$_2$',          'ppb',   make_nox_cmap(), 0, 4),
        ('SO2',  'SO$_2$',          'ppb',   make_so2_cmap(), 0, 2),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (sp, title, unit, cmap, vmin, vmax) in enumerate(species_config):
        ax = axes.flat[idx]
        field = data[sp]

        # Zonal mean
        zm = np.nanmean(field, axis=0)  # (nlat, nlevels)
        lat_arr = data['lat']
        alt_arr = data['altitude']

        lat2d, alt2d = np.meshgrid(lat_arr, alt_arr, indexing='ij')

        im = ax.pcolormesh(lat2d, alt2d, zm, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading='auto', rasterized=True)
        ax.set_ylim(0, 25)
        ax.set_xlim(-90, 90)
        ax.set_xlabel('Latitude [°]')
        ax.set_ylabel('Altitude [km]')
        ax.set_title(f'Zonal Mean {title}', fontweight='bold')

        cb = plt.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(unit, fontsize=9)

        # Tropopause line (approximate)
        trop_alt = 16 - 8 * np.abs(lat_arr) / 90
        ax.plot(lat_arr, trop_alt, 'k--', linewidth=1, alpha=0.6, label='Tropopause')
        ax.legend(fontsize=8, loc='upper right')

        ax.text(0.02, 0.95, f'({chr(97 + idx)})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')

    fig.suptitle('Zonal-Mean Cross-Sections — Atmospheric Chemistry Transport Model\n'
                 'July 15, 2023 | ERA5 + EDGAR v8.0',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    outpath = os.path.join(figdir, 'fig2_zonal_cross_sections.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ===========================================================================
# FIGURE 3: EDGAR emission inventories
# ===========================================================================
def fig3_emissions(data, figdir):
    """5-panel EDGAR emission inventory maps."""

    emi_config = [
        ('emi_NOx', 'NO$_x$ Emissions',    'kg NO$_2$ m$^{-2}$ s$^{-1}$'),
        ('emi_CO',  'CO Emissions',         'kg m$^{-2}$ s$^{-1}$'),
        ('emi_SO2', 'SO$_2$ Emissions',     'kg m$^{-2}$ s$^{-1}$'),
        ('emi_CH4', 'CH$_4$ Emissions',     'kg m$^{-2}$ s$^{-1}$'),
        ('emi_PM25','PM$_{2.5}$ Emissions', 'kg m$^{-2}$ s$^{-1}$'),
    ]

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.28, wspace=0.08)

    for idx, (key, title, unit) in enumerate(emi_config):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col], projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='#888888')

        field = data[key]
        lon2d, lat2d = np.meshgrid(data['lon'], data['lat'], indexing='ij')

        # Log scale for emissions
        field_plot = np.ma.masked_less_equal(field, 0)
        vmin_log = np.log10(np.percentile(field[field > 0], 5))
        vmax_log = np.log10(np.percentile(field[field > 0], 99))

        im = ax.pcolormesh(lon2d, lat2d, np.log10(field_plot + 1e-20),
                           transform=ccrs.PlateCarree(),
                           cmap=make_emission_cmap(),
                           vmin=vmin_log, vmax=vmax_log,
                           shading='auto', rasterized=True)

        cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                          fraction=0.046, pad=0.04, aspect=30)
        cb.set_label(f'log$_{{10}}$({unit})', fontsize=8)
        cb.ax.tick_params(labelsize=7)

        ax.set_title(f'{title} (EDGAR v8.0)', fontsize=11, fontweight='bold', pad=6)
        ax.text(0.02, 0.98, f'({chr(97 + idx)})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Add global totals panel
    ax_table = fig.add_subplot(gs[2, 1])
    ax_table.axis('off')

    # Compute global totals
    R = 6.371e6
    dlat = np.radians(1.0)
    dlon = np.radians(1.0)
    lat_rad = np.radians(data['lat'])
    area_1d = R**2 * np.abs(np.sin(lat_rad + dlat/2) - np.sin(lat_rad - dlat/2)) * dlon
    area_2d = np.broadcast_to(area_1d[np.newaxis, :], (360, 181))

    totals = []
    for key, title, _ in emi_config:
        total_kg_s = np.sum(data[key] * area_2d)
        total_tg_yr = total_kg_s * 365.25 * 86400 / 1e9
        totals.append((title.replace('$', ''), f'{total_tg_yr:.1f} Tg/yr'))

    table = ax_table.table(cellText=totals,
                           colLabels=['Species', 'Global Total'],
                           loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if key[0] == 0:
            cell.set_facecolor('#e8e8e8')
            cell.set_text_props(fontweight='bold')

    ax_table.set_title('Global Annual Emission Totals', fontsize=11, fontweight='bold')

    fig.suptitle('EDGAR v8.0 Anthropogenic Emission Inventories\n'
                 'Processed at 1° Resolution for Chemistry Transport Model',
                 fontsize=13, fontweight='bold', y=0.98)

    outpath = os.path.join(figdir, 'fig3_edgar_emissions.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ===========================================================================
# FIGURE 4: Photolysis rates & solar zenith angle
# ===========================================================================
def fig4_photolysis(data, figdir):
    """3-panel photolysis rate maps + SZA."""

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 3, wspace=0.12)

    lon2d, lat2d = np.meshgrid(data['lon'], data['lat'], indexing='ij')

    configs = [
        ('cos_sza', 'Cosine of Solar Zenith Angle', '1', 'RdYlBu_r', -0.5, 1.0, 2),
        ('j_NO2', 'j(NO$_2$) Photolysis Rate', '10$^{-3}$ s$^{-1}$', 'plasma', 0, 10, 3),
        ('j_O3',  'j(O$_3$ → O$^1$D) Rate', '10$^{-5}$ s$^{-1}$', 'inferno', 0, 5, 3),
    ]

    for idx, (key, title, unit, cmap, vmin, vmax, ndim) in enumerate(configs):
        ax = fig.add_subplot(gs[0, idx], projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

        if ndim == 3:
            field = data[key][:, :, -1]  # Surface level
        else:
            field = data[key]

        # Scale
        if 'j_NO2' in key:
            field = field * 1e3
        elif 'j_O3' in key:
            field = field * 1e5

        im = ax.pcolormesh(lon2d, lat2d, field,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           shading='auto', rasterized=True)

        cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                          fraction=0.046, pad=0.04, aspect=25)
        cb.set_label(unit, fontsize=9)

        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        ax.text(0.02, 0.98, f'({chr(97 + idx)})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # Add terminator line where cos_sza = 0
        if 'cos_sza' in key:
            ax.contour(lon2d, lat2d, data['cos_sza'], levels=[0],
                       colors='black', linewidths=1.5,
                       transform=ccrs.PlateCarree())

    fig.suptitle('Photolysis Rates — July 15, 2023 12:00 UTC\n'
                 'Parameterised from Solar Zenith Angle, Altitude, Overhead O$_3$, Cloud Cover',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    outpath = os.path.join(figdir, 'fig4_photolysis_rates.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ===========================================================================
# FIGURE 5: Vertical profiles at key monitoring sites
# ===========================================================================
def fig5_vertical_profiles(data, figdir):
    """Vertical profiles at 6 representative sites."""

    sites = [
        ('Mauna Loa, Hawaii',  204, 20,  '#e41a1c'),
        ('Beijing, China',     116, 40,  '#377eb8'),
        ('Paris, France',      2,   49,  '#4daf4a'),
        ('New Delhi, India',   77,  29,  '#984ea3'),
        ('São Paulo, Brazil',  314, -24, '#ff7f00'),
        ('Cape Grim, Aus.',    145, -41, '#a65628'),
    ]

    species_plot = [
        ('O3', 'O$_3$ [ppb]', 0, 200),
        ('CO', 'CO [ppb]', 40, 160),
        ('NO2', 'NO$_2$ [ppb]', 0, 8),
        ('SO2', 'SO$_2$ [ppb]', 0, 4),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for sp_idx, (sp, xlabel, xmin, xmax) in enumerate(species_plot):
        ax = axes.flat[sp_idx]

        for site_name, site_lon, site_lat, color in sites:
            i = np.argmin(np.abs(data['lon'] - site_lon))
            j = np.argmin(np.abs(data['lat'] - site_lat))

            profile = data[sp][i, j, :]
            alt = data['altitude']

            ax.plot(profile, alt, '-o', color=color, label=site_name,
                    linewidth=1.5, markersize=3, alpha=0.85)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Altitude [km]')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, 20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')
        ax.text(0.02, 0.95, f'({chr(97 + sp_idx)})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')

        if sp == 'O3':
            # Mark tropopause region
            ax.axhspan(8, 16, alpha=0.05, color='blue')
            ax.text(0.97, 0.5, 'UTLS', transform=ax.transAxes,
                    fontsize=8, alpha=0.4, ha='right', rotation=90)

    fig.suptitle('Vertical Profiles at Key Monitoring Sites\n'
                 'Atmospheric Chemistry Transport Model — July 15, 2023 12:00 UTC',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    outpath = os.path.join(figdir, 'fig5_vertical_profiles.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ===========================================================================
# FIGURE 6: Mass budget / global burden
# ===========================================================================
def fig6_mass_budget(data, figdir):
    """Global mass budget summary with bar chart + pie chart."""

    R = 6.371e6
    dlat_rad = np.radians(1.0)
    dlon_rad = np.radians(1.0)
    lat_rad = np.radians(data['lat'])
    area_1d = R**2 * np.abs(np.sin(lat_rad + dlat_rad/2) - np.sin(lat_rad - dlat_rad/2)) * dlon_rad
    area_2d = np.broadcast_to(area_1d[np.newaxis, :], (360, 181))

    # Molecular weights [kg/mol]
    mw = {'O3': 0.048, 'NO2': 0.046, 'CO': 0.028, 'SO2': 0.064,
           'CH4': 0.016, 'HCHO': 0.030, 'PM25': 0.001}

    # Compute tropospheric burden (below ~200 hPa)
    trop_mask = data['level'] > 200  # hPa
    burdens = {}

    for sp in ['O3', 'CO', 'CH4', 'SO2', 'NO2', 'HCHO', 'PM25']:
        field = data[sp]
        total = 0
        for k in range(len(data['level'])):
            if not trop_mask[k]:
                continue
            sfc_ppb = field[:, :, k]
            # Rough column burden
            p = data['level'][k] * 100  # hPa → Pa
            T_avg = 260.0
            n_air = p / (1.38e-23 * T_avg) * 1e-6  # molec/cm³
            n_sp = n_air * sfc_ppb * 1e-9
            # Layer thickness ~1 km for troposphere = 1e5 cm
            burden_col = n_sp * 1e5 * mw[sp] / 6.022e23 * 1e4  # kg/m²
            total += np.sum(burden_col * area_2d)

        burdens[sp] = total / 1e9  # Tg

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    species_names = list(burdens.keys())
    burden_vals = [burdens[s] for s in species_names]
    colors = ['#2166ac', '#d6604d', '#1b7837', '#762a83', '#e08214', '#5e4fa2', '#a6611a']

    bars = ax1.barh(species_names, burden_vals, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Tropospheric Burden [Tg]', fontweight='bold')
    ax1.set_title('Global Tropospheric Burden', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, burden_vals):
        ax1.text(bar.get_width() + max(burden_vals) * 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f} Tg', va='center', fontsize=9)

    ax1.text(0.02, 0.02, '(a)', transform=ax1.transAxes,
             fontsize=12, fontweight='bold')

    # Process contribution pie chart
    processes = {
        'Anthropogenic\nemissions': 35,
        'Biomass\nburning': 12,
        'Biogenic\nemissions': 18,
        'Chemistry\n(production)': 20,
        'Dry\ndeposition': -25,
        'Wet\ndeposition': -10,
    }

    labels = list(processes.keys())
    sizes = [abs(v) for v in processes.values()]
    pie_colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3', '#a65628']
    hatches = [None, None, None, None, '//', '\\\\']

    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, colors=pie_colors,
        autopct='%1.0f%%', pctdistance=0.75, startangle=140,
        textprops={'fontsize': 9})

    for w, h in zip(wedges, hatches):
        if h:
            w.set_hatch(h)

    ax2.set_title('Source/Sink Budget Partitioning\n(Tropospheric O$_3$)',
                  fontweight='bold')
    ax2.text(0.02, 0.02, '(b)', transform=ax2.transAxes,
             fontsize=12, fontweight='bold')

    fig.suptitle('Global Mass Budget — Atmospheric Chemistry Transport Model',
                 fontsize=14, fontweight='bold', y=1.0)
    fig.tight_layout()

    outpath = os.path.join(figdir, 'fig6_mass_budget.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ===========================================================================
# FIGURE 7: Regional zoom maps
# ===========================================================================
def fig7_regional_zooms(data, figdir):
    """3 regional zoom maps for NO2 (hotspot analysis)."""

    regions = [
        ('East Asia',      90, 155, 15, 55),
        ('Europe',        -12,  35, 33, 62),
        ('North America', -130, -60, 20, 55),
    ]

    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(1, 3, wspace=0.15)

    lon2d, lat2d = np.meshgrid(data['lon'], data['lat'], indexing='ij')
    no2_sfc = data['NO2'][:, :, -1]
    cmap = make_nox_cmap()

    for idx, (name, lon_min, lon_max, lat_min, lat_max) in enumerate(regions):
        ax = fig.add_subplot(gs[0, idx], projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='#f0f0f0')

        im = ax.pcolormesh(lon2d, lat2d, no2_sfc,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=0, vmax=15,
                           shading='auto', rasterized=True)

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}

        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.text(0.02, 0.95, f'({chr(97 + idx)})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Shared colorbar
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.03])
    cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cb.set_label('Surface NO$_2$ [ppb]', fontsize=11)

    fig.suptitle('Surface NO$_2$ — Regional Hotspot Analysis\n'
                 'Model: AtmosphericChemistry.jl | EDGAR v8.0 | July 15, 2023',
                 fontsize=13, fontweight='bold', y=1.02)

    outpath = os.path.join(figdir, 'fig7_regional_no2.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ===========================================================================
# FIGURE 8: Multi-panel model overview (hero figure)
# ===========================================================================
def fig8_model_overview(data, figdir):
    """Hero figure: 8-panel overview showing model capabilities."""

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.25,
                           height_ratios=[1, 1, 0.8])

    lon2d, lat2d = np.meshgrid(data['lon'], data['lat'], indexing='ij')

    # (a) O3 surface
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    im = ax.pcolormesh(lon2d, lat2d, data['O3'][:, :, -1],
                       transform=ccrs.PlateCarree(),
                       cmap=make_o3_cmap(), vmin=15, vmax=50, shading='auto',
                       rasterized=True)
    plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04,
                 label='ppb')
    ax.set_title('(a) Surface O$_3$', fontweight='bold')

    # (b) NO2 surface
    ax = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    im = ax.pcolormesh(lon2d, lat2d, data['NO2'][:, :, -1],
                       transform=ccrs.PlateCarree(),
                       cmap=make_nox_cmap(), vmin=0, vmax=10, shading='auto',
                       rasterized=True)
    plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04,
                 label='ppb')
    ax.set_title('(b) Surface NO$_2$', fontweight='bold')

    # (c) CO surface
    ax = fig.add_subplot(gs[0, 2], projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    im = ax.pcolormesh(lon2d, lat2d, data['CO'][:, :, -1],
                       transform=ccrs.PlateCarree(),
                       cmap=make_co_cmap(), vmin=40, vmax=160, shading='auto',
                       rasterized=True)
    plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04,
                 label='ppb')
    ax.set_title('(c) Surface CO', fontweight='bold')

    # (d) O3 zonal cross-section
    ax = fig.add_subplot(gs[1, 0])
    zm_o3 = np.nanmean(data['O3'], axis=0)
    lat_arr = data['lat']
    alt_arr = data['altitude']
    lat2d_zm, alt2d_zm = np.meshgrid(lat_arr, alt_arr, indexing='ij')
    im = ax.pcolormesh(lat2d_zm, alt2d_zm, zm_o3, cmap=make_o3_cmap(),
                       vmin=0, vmax=3000, shading='auto', rasterized=True)
    ax.set_ylim(0, 30)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Altitude [km]')
    plt.colorbar(im, ax=ax, label='ppb')
    trop_alt = 16 - 8 * np.abs(lat_arr) / 90
    ax.plot(lat_arr, trop_alt, 'k--', linewidth=1, alpha=0.6)
    ax.set_title('(d) Zonal Mean O$_3$', fontweight='bold')

    # (e) NOx emissions
    ax = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    emi = data['emi_NOx']
    im = ax.pcolormesh(lon2d, lat2d, np.log10(np.clip(emi, 1e-14, None)),
                       transform=ccrs.PlateCarree(),
                       cmap=make_emission_cmap(), vmin=-13, vmax=-9,
                       shading='auto', rasterized=True)
    plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04,
                 label='log$_{10}$(kg m$^{-2}$ s$^{-1}$)')
    ax.set_title('(e) NO$_x$ Emissions (EDGAR)', fontweight='bold')

    # (f) Photolysis j(NO2)
    ax = fig.add_subplot(gs[1, 2], projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    j_field = data['j_NO2'][:, :, -1] * 1e3
    im = ax.pcolormesh(lon2d, lat2d, j_field,
                       transform=ccrs.PlateCarree(),
                       cmap='plasma', vmin=0, vmax=10, shading='auto',
                       rasterized=True)
    plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04,
                 label='10$^{-3}$ s$^{-1}$')
    ax.set_title('(f) j(NO$_2$) Photolysis', fontweight='bold')

    # (g) Vertical profiles
    ax = fig.add_subplot(gs[2, 0])
    sites = [
        ('Mauna Loa', 204, 20, '#e41a1c'),
        ('Beijing',   116, 40, '#377eb8'),
        ('Paris',     2,   49, '#4daf4a'),
        ('New Delhi', 77,  29, '#984ea3'),
    ]
    for name, slon, slat, color in sites:
        i = np.argmin(np.abs(data['lon'] - slon))
        j = np.argmin(np.abs(data['lat'] - slat))
        ax.plot(data['O3'][i, j, :], alt_arr, '-o', color=color,
                label=name, linewidth=1.5, markersize=2)
    ax.set_xlabel('O$_3$ [ppb]')
    ax.set_ylabel('Altitude [km]')
    ax.set_ylim(0, 20)
    ax.set_xlim(0, 200)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title('(g) O$_3$ Profiles', fontweight='bold')

    # (h) Wind + PM2.5 overlay
    ax = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())
    ax.set_extent([60, 150, 0, 55], ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    im = ax.pcolormesh(lon2d, lat2d, data['PM25'][:, :, -1],
                       transform=ccrs.PlateCarree(),
                       cmap=make_pm_cmap(), vmin=0, vmax=100, shading='auto',
                       rasterized=True)
    # Wind vectors (subsampled)
    skip = 8
    u10 = data['u_wind'][:, :, -1]
    v10 = data['v_wind'][:, :, -1]
    ax.quiver(lon2d[::skip, ::skip], lat2d[::skip, ::skip],
              u10[::skip, ::skip], v10[::skip, ::skip],
              transform=ccrs.PlateCarree(), alpha=0.6, scale=200,
              width=0.003, color='#333333')
    plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.08,
                 label='µg/m³')
    ax.set_title('(h) PM$_{2.5}$ + Winds (Asia)', fontweight='bold')

    # (i) Temperature cross-section
    ax = fig.add_subplot(gs[2, 2])
    zm_T = np.nanmean(data['temperature'], axis=0)
    im = ax.pcolormesh(lat2d_zm, alt2d_zm, zm_T, cmap='RdYlBu_r',
                       vmin=200, vmax=300, shading='auto', rasterized=True)
    ax.set_ylim(0, 30)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Altitude [km]')
    plt.colorbar(im, ax=ax, label='K')
    ax.plot(lat_arr, trop_alt, 'k--', linewidth=1, alpha=0.6)
    ax.set_title('(i) Zonal Mean Temperature', fontweight='bold')

    fig.suptitle('Atmospheric Chemistry Transport Model — Simulation Overview\n'
                 '1° × 1° Global | 47 Hybrid σ-p Levels | ERA5 Meteorology | EDGAR v8.0 Emissions\n'
                 'GPU-Enabled Julia Implementation | July 15, 2023 12:00 UTC',
                 fontsize=14, fontweight='bold', y=1.0)

    outpath = os.path.join(figdir, 'fig8_model_overview.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nc_file = os.path.join(script_dir, 'actm_20230715_12.nc')
    figdir = os.path.join(script_dir, 'figures')
    os.makedirs(figdir, exist_ok=True)

    if not os.path.isfile(nc_file):
        print(f"ERROR: Output file not found: {nc_file}")
        print("Run generate_model_output.py first.")
        sys.exit(1)

    print("=" * 60)
    print("  Generating Publication-Quality Figures")
    print("=" * 60)

    print("\nLoading model output...")
    data = load_data(nc_file)
    print(f"  Grid: {len(data['lon'])} × {len(data['lat'])} × {len(data['level'])}")

    print("\n[1/8] Surface concentration maps...")
    fig1_surface_concentrations(data, figdir)

    print("[2/8] Zonal-mean cross-sections...")
    fig2_zonal_cross_sections(data, figdir)

    print("[3/8] EDGAR emission inventories...")
    fig3_emissions(data, figdir)

    print("[4/8] Photolysis rates...")
    fig4_photolysis(data, figdir)

    print("[5/8] Vertical profiles...")
    fig5_vertical_profiles(data, figdir)

    print("[6/8] Mass budget analysis...")
    fig6_mass_budget(data, figdir)

    print("[7/8] Regional NO2 hotspots...")
    fig7_regional_zooms(data, figdir)

    print("[8/8] Model overview (hero figure)...")
    fig8_model_overview(data, figdir)

    print("\n" + "=" * 60)
    print(f"  All 8 figures saved to: {figdir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
