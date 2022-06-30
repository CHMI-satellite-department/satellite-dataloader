from pathlib import Path
from satdl.datasets import StaticImageFolderDataset
import xarray as xr

import pytest

FIXTURE_DIR = Path(__file__).parent / 'test_data'


@pytest.mark.datafiles(FIXTURE_DIR / 'images')
@pytest.mark.datafiles(FIXTURE_DIR / '201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif')
def test_sifd(datafiles):
    sifd = StaticImageFolderDataset(datafiles, '{projection}-{resolution}.{product}.{datetime:%Y%m%d.%H%M}.0.jpg',
                                    georef=Path(datafiles) / '201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif')
    assert len(sifd) == 12
    for f in Path(datafiles).glob('*.jpg'):
        assert Path(f).name in sifd.keys()

    for attr in sifd.attrs.values():
        assert attr['projection'] == 'msgce'

    for key in sifd.keys():
        im = sifd[key]

        assert isinstance(im, xr.DataArray)
        assert im.shape == (3, 800, 1160)
        assert im.lat.min() > 42.27232
        assert im.lat.max() < 56.64341
        assert im.lon.min() > -1.687554
        assert im.lon.max() < 30.715534


@pytest.mark.datafiles(FIXTURE_DIR / 'images')
@pytest.mark.datafiles(FIXTURE_DIR / '201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif')
def test_sifd_iter(datafiles):
    sifd = StaticImageFolderDataset(datafiles, '{projection}-{resolution}.{product}.{datetime:%Y%m%d.%H%M}.0.jpg',
                                    georef=Path(datafiles) / '201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif')
    i = 0
    for key in sifd:
        assert (Path(datafiles) / key).exists()
        i += 1

    assert i == 12


@pytest.mark.datafiles(FIXTURE_DIR / 'images')
@pytest.mark.datafiles(FIXTURE_DIR / '201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif')
def test_sifd_cache(datafiles):
    sifd = StaticImageFolderDataset(datafiles, '{projection}-{resolution}.{product}.{datetime:%Y%m%d.%H%M}.0.jpg',
                                    georef=Path(datafiles) / '201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif',
                                    max_cache=50)

    for key in sifd.keys():
        im = sifd[key]

        assert isinstance(im, xr.DataArray)

    for key in sifd.keys():
        im = sifd[key]

        assert isinstance(im, xr.DataArray)


@pytest.mark.datafiles(FIXTURE_DIR / 'images')
@pytest.mark.datafiles(FIXTURE_DIR / '201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif')
def test_sifd_groupby(datafiles):
    sifd = StaticImageFolderDataset(datafiles, '{projection}-{resolution}.{product}.{datetime:%Y%m%d.%H%M}.0.jpg',
                                    georef=Path(datafiles) / '201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif',
                                    max_cache=None)

    group = sifd.groupby('datetime', sortby=['datetime', 'product'])

    assert len(group) == 3
    assert len(group[0]) == 4

    group = sifd.groupby('product', sortby='datetime')

    assert len(group) == 4
    assert len(group[0]) == 3

    i = 0
    for _ in group:
        i += 1

    assert i == 4
