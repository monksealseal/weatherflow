"""
NetCDF I/O for GCM state

Provides functions to save and load model state to/from NetCDF files
"""

import numpy as np


def save_state(state, filename):
    """
    Save model state to NetCDF file

    Parameters
    ----------
    state : ModelState
        Model state to save
    filename : str
        Output filename

    Note
    ----
    Requires netCDF4 library
    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        print("Warning: netCDF4 not available. Saving to numpy format instead.")
        np.savez(filename.replace('.nc', '.npz'),
                u=state.u, v=state.v, T=state.T, q=state.q,
                qc=state.qc, qi=state.qi, ps=state.ps,
                time=state.time)
        return

    with Dataset(filename, 'w', format='NETCDF4') as nc:
        # Create dimensions
        nlev, nlat, nlon = state.u.shape
        nc.createDimension('lon', nlon)
        nc.createDimension('lat', nlat)
        nc.createDimension('lev', nlev)
        nc.createDimension('time', 1)

        # Create coordinate variables
        lon_var = nc.createVariable('lon', 'f4', ('lon',))
        lat_var = nc.createVariable('lat', 'f4', ('lat',))
        lev_var = nc.createVariable('lev', 'f4', ('lev',))
        time_var = nc.createVariable('time', 'f4', ('time',))

        # Assume grid coordinates are available
        # (In practice, would get from state.grid)
        lon_var[:] = np.linspace(0, 360, nlon, endpoint=False)
        lat_var[:] = np.linspace(-90, 90, nlat)
        lev_var[:] = np.arange(nlev)
        time_var[:] = state.time

        # Attributes
        lon_var.units = 'degrees_east'
        lat_var.units = 'degrees_north'
        lev_var.units = 'level'
        time_var.units = 'seconds since start'

        # Create state variables
        u_var = nc.createVariable('u', 'f4', ('time', 'lev', 'lat', 'lon'))
        v_var = nc.createVariable('v', 'f4', ('time', 'lev', 'lat', 'lon'))
        T_var = nc.createVariable('T', 'f4', ('time', 'lev', 'lat', 'lon'))
        q_var = nc.createVariable('q', 'f4', ('time', 'lev', 'lat', 'lon'))
        qc_var = nc.createVariable('qc', 'f4', ('time', 'lev', 'lat', 'lon'))
        qi_var = nc.createVariable('qi', 'f4', ('time', 'lev', 'lat', 'lon'))
        ps_var = nc.createVariable('ps', 'f4', ('time', 'lat', 'lon'))

        # Units
        u_var.units = 'm/s'
        v_var.units = 'm/s'
        T_var.units = 'K'
        q_var.units = 'kg/kg'
        qc_var.units = 'kg/kg'
        qi_var.units = 'kg/kg'
        ps_var.units = 'Pa'

        # Long names
        u_var.long_name = 'Zonal wind'
        v_var.long_name = 'Meridional wind'
        T_var.long_name = 'Temperature'
        q_var.long_name = 'Specific humidity'
        qc_var.long_name = 'Cloud liquid water'
        qi_var.long_name = 'Cloud ice'
        ps_var.long_name = 'Surface pressure'

        # Write data
        u_var[0, :, :, :] = state.u
        v_var[0, :, :, :] = state.v
        T_var[0, :, :, :] = state.T
        q_var[0, :, :, :] = state.q
        qc_var[0, :, :, :] = state.qc
        qi_var[0, :, :, :] = state.qi
        ps_var[0, :, :] = state.ps

        # Global attributes
        nc.title = 'GCM Model State'
        nc.institution = 'GCM Development Team'
        nc.source = 'Sophisticated General Circulation Model'

    print(f"State saved to {filename}")


def load_state(state, filename):
    """
    Load model state from NetCDF file

    Parameters
    ----------
    state : ModelState
        Model state object to populate
    filename : str
        Input filename

    Note
    ----
    Requires netCDF4 library
    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        print("Warning: netCDF4 not available. Loading from numpy format.")
        data = np.load(filename.replace('.nc', '.npz'))
        state.u = data['u']
        state.v = data['v']
        state.T = data['T']
        state.q = data['q']
        state.qc = data['qc']
        state.qi = data['qi']
        state.ps = data['ps']
        state.time = float(data['time'])
        state.update_diagnostics()
        return

    with Dataset(filename, 'r') as nc:
        state.u[:] = nc.variables['u'][0, :, :, :]
        state.v[:] = nc.variables['v'][0, :, :, :]
        state.T[:] = nc.variables['T'][0, :, :, :]
        state.q[:] = nc.variables['q'][0, :, :, :]
        state.qc[:] = nc.variables['qc'][0, :, :, :]
        state.qi[:] = nc.variables['qi'][0, :, :, :]
        state.ps[:] = nc.variables['ps'][0, :, :]
        state.time = float(nc.variables['time'][0])

    state.update_diagnostics()
    print(f"State loaded from {filename}")
