from pathlib import Path
import pytest
from satdl.utils import image2xr

FIXTURE_DIR = Path(__file__).parent / 'test_data'


@pytest.mark.datafiles(FIXTURE_DIR / '201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif')
def test_image2xr_georeferencing(datafiles):
    da = image2xr(str(datafiles / '201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif'))
    assert da.lat.min() > 42.27232
    assert da.lat.max() < 56.64341
    assert da.lon.min() > -1.687554
    assert da.lon.max() < 30.715534
