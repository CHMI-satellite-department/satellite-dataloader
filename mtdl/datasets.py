import logging
from pathlib import Path
from tqdm.auto import tqdm
import warnings
import xarray as xr
import satpy
import random
import torch

INSTRUMENTS = {}
INSTRUMENTS['SEVIRI'] = {'reader': 'seviri_l1b_hrit',
                         'channels': ['HRV', 'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134',
                                      'VIS006', 'VIS008', 'WV_062', 'WV_073']}


def walk_and_convert(converter, source, dest, level=0, mask='*'):
    """Walk directory structure and convert data

    :param converter: class to convert the data
    :param source: source directory
    :param dest: destination directory
    :param level: number of sublevels (default: 0)
    :param mask: mask of sublevels, (default: '*')
    :return:
    """
    path = Path(source)
    files_or_dirs = list(path.glob(mask))
    if level <= 0:
        for p in tqdm(files_or_dirs, leave=False, desc=str(source), total=len(files_or_dirs)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converter(p, dest)
    else:
        for p in tqdm(files_or_dirs, leave=False, desc=str(source), total=len(files_or_dirs)):
            walk_and_convert(converter, p, dest, level=level - 1, mask=mask)


class SatpyDir2H5:
    def __init__(self, proj, instrument='SEVIRI', out_mask='{proj}-{instrument}-{start_time}.nc', overwrite=False,
                 radius_of_influence=50000, **proj_kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.proj = proj
        self.instrument = instrument
        self.reader = INSTRUMENTS[instrument]['reader']
        self.channels = INSTRUMENTS[instrument]['channels']
        self.proj_kwargs = {'radius_of_influence': radius_of_influence}
        self.proj_kwargs.update(proj_kwargs)
        self.out_mask = out_mask

        self.overwrite = overwrite

    def __call__(self, source, dest):
        """Create netcdf cube from satelite data
        
        :param source:  path to source data (directory)
        :param dest: output dir
        :return: xarray dataset
        """
        files = satpy.find_files_and_readers(base_dir=source, reader=self.reader)
        scene = satpy.Scene(filenames=files)

        out_name_dict = {'instrument': self.instrument, 'proj': self.proj}
        out_name_dict.update(scene.attrs)
        fname = Path(dest) / self.out_mask.format(**out_name_dict)
        if not self.overwrite and fname.is_file():
            self.logger.info(f'SKIPPING {fname}, already exists')
            return

        scene.load(self.channels)
        scene = scene.resample(self.proj, **self.proj_kwargs)

        loc_ds = {}
        for ch in self.channels:
            da = scene[ch]
            da = da.drop_vars(['acq_time', 'crs'], errors='ignore')
            da.attrs = {}
            loc_ds[ch] = da
        loc_ds = xr.Dataset(loc_ds)

        loc_ds.to_netcdf(fname, engine='h5netcdf', encoding={n: {'zlib': True, 'complevel': 9} for n in self.channels})
        self.logger.info(f'CREATED {fname}')


class H5Dataset:
    def __init__(self, source, size=(256, 256), mask='*.nc'):
        self._datafiles = tuple(Path(source).glob(mask))
        self._size = size

    def __len__(self):
        return len(self._datafiles)

    def __getitem__(self, index):
        data = self._read_data(self._datafiles[index])
        coords = list(data.coords)
        x0 = random.randint(0, len(data[coords[0]]) - 1 - self._size[0])
        y0 = random.randint(0, len(data[coords[1]]) - 1 - self._size[1])
        data = data.isel(**{coords[0]: slice(x0, x0+self._size[0]), coords[1]:slice(y0, y0+self._size[1])})
        data = data.to_array(dim='channel')
        return data.values

    def _read_data(self, source):
        return xr.open_dataset(source)

class H5DatasetTorch(H5Dataset):
    def __getitem__(self, index):
        return torch.FloatTensor(super().__getitem__(index))