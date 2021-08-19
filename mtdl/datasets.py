from pathlib import Path
from trollsift import Parser
from typing import Any, Dict, List, Tuple
import xarray as xr

from .utils import image2xr

class ImageFolderDataset:
    def __init__(self, base_folder: str or Path, file_mask: str or Parser,
                 georef: str or Path or xr.DataArray=None) -> None:
        """Create ImageFolderDataset

        Parameters
        ----------
        base_folder : str or Path
            root folder of the data
        file_mask : str or trollsift.Parser
            mask of image names specifying attributes in the file name. Must not contain wildcards '*' or '?'
        georef : str or Path or xr.DataArray or None
            external georeference for plain images, optional
        """
        self._base_folder = Path(base_folder)
        if not self._base_folder.exists():
            raise ValueError(f'base folder {base_folder} does not exist')
        self._file_mask = Parser(file_mask)
        self._georef = georef  # TODO: validate georeference
        self._files = list(self._base_folder.rglob(self._file_mask.globify()))

    def __len__(self) -> int:
        return len(self._files)

    def _extract_attrs(self, filename: str) -> Dict[str, Any]:
        return self._file_mask.parse(filename)

    def keys(self) -> List[str]:
        """Return all image keys"""
        return [str(f.relative_to(self._base_folder)) for f in self._files]

    def items(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Return list of (key, attributes) pairs"""
        return [(key, self._extract_attrs(key)) for key in self.keys()]

    def __getitem__(self, key: str) -> xr.DataArray:
        """Return image as DataArray from key"""
        da = image2xr(self._base_folder / key, georef=self._georef)
        da.attrs.update(self._extract_attrs(key))

        return da

    def iloc(self, i: int) -> xr.DataArray:
        """Return i-th image as DataArrray

        Raises IndexError if i >= len(self)
        """
        return self[str(self._files[i].relative_to(self._base_folder))]

    def random(self) -> xr.DataArray:
        """Return random image as DataArray"""
        return self.iloc(np.rand.randint(len(self)))
