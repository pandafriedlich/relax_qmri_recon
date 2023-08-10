from .qmridata import (QuantitativeMappingSubjectData,
                       ACCELERATION_FOLDER_INV_MAP)
from .utils import get_acs_mask
import os
from torch.utils.data import Dataset
import typing
from pathlib import Path
import torch


class QuantitativeMRIAccelerationXDataset(Dataset):
    """
    The dataset of all subjects of a certain acceleration factor.
    """
    def __init__(self, dataset_base: typing.Union[str, bytes, os.PathLike],
                 transforms: typing.Optional[typing.Callable] = None):
        super().__init__()
        self.dataset_base = Path(dataset_base)
        self.transforms = transforms
        self.acceleration_factor = ACCELERATION_FOLDER_INV_MAP[self.dataset_base.stem]
        self.subject_paths = list(self.dataset_base.glob("P*"))

    def __len__(self) -> int:
        return len(self.subject_paths)

    def __getitem__(self, ind) -> typing.Dict[str, typing.Any]:
        subject_path = self.subject_paths[ind]
        subject_data = QuantitativeMappingSubjectData(subject_path,
                                                      acceleration_factor=self.acceleration_factor)
        # split the subject along the slice dimension
        _slice_dimension = 3
        t1_n_slices = subject_data.t1_map_kspace.shape[_slice_dimension]
        t2_n_slices = subject_data.t2_map_kspace.shape[_slice_dimension]
        t1 = [dict(acc_kspace=subject_data.t1_map_kspace[:, :, :, s, :],
                   us_mask=subject_data.t1_us_mask,
                   acs_mask=get_acs_mask(subject_data.t1_us_mask),
                   tvec=subject_data.t1_map_t_inv[s])
              for s in range(t1_n_slices)]
        t2 = [dict(acc_kspace=subject_data.t2_map_kspace[:, :, :, s, :],
                   us_mask=subject_data.t2_us_mask,
                   acs_mask=get_acs_mask(subject_data.t2_us_mask),
                   tvec=subject_data.t2_map_t_echo)
              for s in range(t2_n_slices)]

        # transforms
        if self.transforms is not None:
            t1 = [self.transforms(d) for d in t1]
            t2 = [self.transforms(d) for d in t2]

        return dict(t1=t1, t2=t2, path=subject_path.relative_to(self.dataset_base))
