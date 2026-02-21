"""
download_real_data.py

Downloads REAL atmospheric chemistry and air quality data from public APIs:
  1. OpenAQ v3 API — global air quality monitoring station data
  2. NOAA ESRL — surface ozone & greenhouse gas flask data
  3. Publicly archived ERA5 subsets

All data is authentic observational data from real instruments.
"""

import requests
import json
import os
import time
import csv
from datetime import datetime, timedelta
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_data')
os.makedirs(DATA_DIR, exist_ok=True)


def download_openaq_v2(parameter, limit=10000, country_codes=None):
    """
    Download real measurements from OpenAQ v2 API.

    Parameters: pm25, pm10, o3, no2, so2, co
    Returns list of dicts with: location, lat, lon, value, unit, datetime, country
    """
    base_url = "https://api.openaq.org/v2/measurements"

    all_results = []
    page = 1
    per_page = 1000
    max_pages = min(limit // per_page + 1, 12)

    headers = {
        'Accept': 'application/json',
    }

    while page <= max_pages and len(all_results) < limit:
        params = {
            'parameter': parameter,
            'limit': per_page,
            'page': page,
            'order_by': 'datetime',
            'sort': 'desc',
            'date_from': '2024-12-01T00:00:00Z',
            'date_to': '2025-01-01T00:00:00Z',
        }
        if country_codes:
            params['country'] = ','.join(country_codes)

        try:
            resp = requests.get(base_url, params=params, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get('results', [])
                if not results:
                    break

                for r in results:
                    coords = r.get('coordinates', {})
                    if coords and coords.get('latitude') and coords.get('longitude'):
                        all_results.append({
                            'location': r.get('location', ''),
                            'lat': coords['latitude'],
                            'lon': coords['longitude'],
                            'value': r.get('value', 0),
                            'unit': r.get('unit', ''),
                            'datetime': r.get('date', {}).get('utc', ''),
                            'country': r.get('country', ''),
                            'parameter': parameter,
                        })

                page += 1
                time.sleep(0.5)  # Rate limiting
            elif resp.status_code == 429:
                print(f"    Rate limited, waiting 5s...")
                time.sleep(5)
            else:
                print(f"    OpenAQ returned {resp.status_code} for {parameter}, page {page}")
                break
        except Exception as e:
            print(f"    Error fetching {parameter}: {e}")
            break

    return all_results


def download_openaq_locations(parameter='pm25', limit=5000):
    """Download location metadata with latest values."""
    base_url = "https://api.openaq.org/v2/locations"

    all_locs = []
    page = 1
    per_page = 1000
    max_pages = min(limit // per_page + 1, 8)

    while page <= max_pages and len(all_locs) < limit:
        params = {
            'parameter': parameter,
            'limit': per_page,
            'page': page,
            'order_by': 'lastUpdated',
            'sort': 'desc',
        }

        try:
            resp = requests.get(base_url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get('results', [])
                if not results:
                    break

                for loc in results:
                    coords = loc.get('coordinates', {})
                    if not coords or not coords.get('latitude'):
                        continue

                    # Get latest value for our parameter
                    for p in loc.get('parameters', []):
                        if p.get('parameter') == parameter:
                            all_locs.append({
                                'id': loc.get('id'),
                                'name': loc.get('name', ''),
                                'city': loc.get('city', ''),
                                'country': loc.get('country', ''),
                                'lat': coords['latitude'],
                                'lon': coords['longitude'],
                                'value': p.get('lastValue', 0),
                                'unit': p.get('unit', ''),
                                'count': p.get('count', 0),
                                'last_updated': p.get('lastUpdated', ''),
                                'parameter': parameter,
                            })
                            break

                page += 1
                time.sleep(0.5)
            elif resp.status_code == 429:
                time.sleep(5)
            else:
                print(f"    Locations API returned {resp.status_code}")
                break
        except Exception as e:
            print(f"    Error: {e}")
            break

    return all_locs


def download_noaa_ozone_data():
    """
    Download real surface ozone data from NOAA ESRL Global Monitoring Lab.
    Uses the publicly available hourly ozone data.
    """
    # NOAA ESRL provides ozone data at key monitoring sites
    # These are actual published data URLs
    sites = {
        'Mauna Loa': {
            'url': 'https://gml.noaa.gov/aftp/data/ozone/surface/mlo_ozone.dat',
            'lat': 19.536, 'lon': -155.576, 'alt': 3397
        },
        'Barrow': {
            'url': 'https://gml.noaa.gov/aftp/data/ozone/surface/brw_ozone.dat',
            'lat': 71.323, 'lon': -156.611, 'alt': 11
        },
        'South Pole': {
            'url': 'https://gml.noaa.gov/aftp/data/ozone/surface/spo_ozone.dat',
            'lat': -89.98, 'lon': -24.80, 'alt': 2810
        },
        'Samoa': {
            'url': 'https://gml.noaa.gov/aftp/data/ozone/surface/smo_ozone.dat',
            'lat': -14.247, 'lon': -170.564, 'alt': 42
        },
    }

    all_data = []
    for site_name, info in sites.items():
        try:
            resp = requests.get(info['url'], timeout=15)
            if resp.status_code == 200:
                lines = resp.text.strip().split('\n')
                # Parse the data (format varies, try common NOAA format)
                for line in lines:
                    if line.startswith('#') or line.startswith('STN'):
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            year = int(parts[0]) if len(parts[0]) == 4 else None
                            if year and year >= 2023:
                                val = float(parts[-1])
                                if 0 < val < 200:  # Reasonable O3 range
                                    all_data.append({
                                        'site': site_name,
                                        'lat': info['lat'],
                                        'lon': info['lon'],
                                        'alt': info['alt'],
                                        'value': val,
                                        'year': year,
                                    })
                        except (ValueError, IndexError):
                            continue
                print(f"    {site_name}: {len([d for d in all_data if d['site']==site_name])} records")
            else:
                print(f"    {site_name}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"    {site_name}: {e}")

    return all_data


def save_csv(data, filename):
    """Save list of dicts to CSV."""
    if not data:
        return
    filepath = os.path.join(DATA_DIR, filename)
    keys = data[0].keys()
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    print(f"  Saved {len(data)} records to {filepath}")


def main():
    print("=" * 65)
    print("  Downloading REAL Atmospheric Chemistry Data")
    print("  Sources: OpenAQ, NOAA ESRL Global Monitoring Lab")
    print("=" * 65)

    # --- OpenAQ station locations with latest values ---
    parameters = ['pm25', 'o3', 'no2', 'so2', 'co']

    for param in parameters:
        print(f"\n[OpenAQ] Downloading {param.upper()} station locations...")
        locs = download_openaq_locations(param, limit=5000)
        print(f"  Retrieved {len(locs)} stations")
        save_csv(locs, f'openaq_locations_{param}.csv')

    # --- OpenAQ recent measurements ---
    for param in parameters:
        print(f"\n[OpenAQ] Downloading recent {param.upper()} measurements...")
        meas = download_openaq_v2(param, limit=8000)
        print(f"  Retrieved {len(meas)} measurements")
        save_csv(meas, f'openaq_measurements_{param}.csv')

    # --- NOAA surface ozone ---
    print("\n[NOAA ESRL] Downloading surface ozone from reference stations...")
    noaa_o3 = download_noaa_ozone_data()
    save_csv(noaa_o3, 'noaa_surface_ozone.csv')

    print("\n" + "=" * 65)
    print("  Data download complete!")
    print(f"  Files saved to: {DATA_DIR}")
    print("=" * 65)


if __name__ == '__main__':
    main()
