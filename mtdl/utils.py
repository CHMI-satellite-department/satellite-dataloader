import numpy as np
from pyproj import CRS
from pyproj import Transformer
import xarray as xr

def image2xr(path, georef=None, require_georef=True):
    """Load georeferenced TIFF"""
    da = xr.open_rasterio(path)

    if georef is not None:
        if not isinstance(georef, xr.DataArray):
            georef = image2xr(georef)

        da.coords['lat'] = (('y', 'x'), georef.lat.data)
        da.coords['lon'] = (('y', 'x'), georef.lon.data)

        for attr in ['transform', 'res', 'crs']:
            da.attrs[attr] = georef.attrs.get(attr)

    elif 'crs' in da.attrs:
        da_crs = CRS.from_proj4(da.attrs['crs'])
        wgs84 = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(da_crs, wgs84)

        xx, yy = np.meshgrid(da.x, da.y)
        lat, lon = transformer.transform(xx, yy)

        da.coords['lat'] = (('y', 'x'), lat)
        da.coords['lon'] = (('y', 'x'), lon)

    if require_georef and ('lat' not in da.coords or 'lon' not in da.coords):
        raise ValueError(f'cannot georeference image {path}')

    return da
