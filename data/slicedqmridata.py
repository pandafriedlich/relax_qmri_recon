import numpy as np
import torch
from pathlib import Path
import typing
import os
from .qmridata import ACCELERATION_FOLDER_MAP
from sklearn.model_selection import KFold
from .utils import dumped_data_reader, get_acs_mask
from numbers import Number


class SlicedQuantitativeMRIDatasetListSplit:
    """
    List all the files of
    """
    def __init__(self, dataset_base: typing.Union[str, bytes, os.PathLike],
                 acceleration_factors: typing.Tuple[Number, ...],
                 modalities: typing.Tuple[str, ...]) -> None:
        """

        Go through the dataset and get all the files.
        :param dataset_base: Path to the dataset base like `.../TrainingSet`.
        :param acceleration_factors: The acceleration factors to be included.
        :param modalities: The modalities to be included, can be ('t1map', ), ('t2map', ) or ('t1map', 't2map')
        """
        self.dataset_base = Path(dataset_base)
        self.acceleration_factors = acceleration_factors
        self.modalities = modalities
        assert all([m.lower() in ('t1map', 't2map') for m in self.modalities]),\
            f"Invalid modalities: {str(self.modalities)}! "

        self.ground_truth_folder = self.dataset_base / ACCELERATION_FOLDER_MAP[1.0]
        self.acceleration_sub_folders = [self.dataset_base / ACCELERATION_FOLDER_MAP[float(r)]
                                         for r in self.acceleration_factors]
        # List all accelerated data files.
        self.list_of_acc_files = []
        for acc_sub_folder in self.acceleration_sub_folders:
            for subject_folder in acc_sub_folder.glob("P*"):
                modality_sub_folders = [subject_folder / m for m in self.modalities]
                for modality_folder in modality_sub_folders:
                    self.list_of_acc_files += list(modality_folder.glob('slice_*.dat'))

        # List all corresponding GT files.
        self.list_of_gt_files = []
        for acc_file in self.list_of_acc_files:
            filename = acc_file.name
            modality = acc_file.parent.name
            subject_id = acc_file.parent.parent.name
            gt_files = self.ground_truth_folder / subject_id / modality / filename
            assert gt_files.exists(), f"Ground truth file {gt_files.as_posix()} not found!"
            self.list_of_gt_files.append(gt_files)

        assert len(self.list_of_acc_files) == len(self.list_of_gt_files), \
            "Acceleration files and GT files have mismatching lengths!"

    def split(self, k: int = 5) -> typing.List[dict]:
        """
        Split the dataset
        :param k: number of splits.
        :return: List of training and validation files.
        """
        indices = np.arange(len(self.list_of_acc_files))
        splitter = KFold(n_splits=k, shuffle=True)
        splits = list(splitter.split(indices[:, None]))
        split_files = []
        for training_indices, valid_indices in splits:
            split_files.append(dict(training=([self.list_of_acc_files[ind] for ind in training_indices],
                                                [self.list_of_gt_files[ind] for ind in training_indices]),
                                    validation=([self.list_of_acc_files[ind] for ind in valid_indices],
                                                [self.list_of_gt_files[ind] for ind in valid_indices]),
                                    )
                               )
        return split_files


class SlicedQuantitativeMRIDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_acc_files: typing.List[typing.Union[str, bytes, os.PathLike]],
                 list_of_gt_files: typing.List[typing.Union[str, bytes, os.PathLike]],
                 transforms: typing.Optional[typing.Callable] = None) -> None:
        """
        Construct a dataset object.
        :param list_of_acc_files: List of acceleration data files.
        :param list_of_gt_files: List of ground truth data files.
        :param transforms: Data transforms.
        """
        super(SlicedQuantitativeMRIDataset, self).__init__()
        self.list_of_acc_files = list_of_acc_files
        self.list_of_gt_files = list_of_gt_files
        assert len(self.list_of_acc_files) == len(self.list_of_gt_files), \
            "Acceleration files and GT files have mismatching lengths!"
        self.transforms = transforms

    def __len__(self):
        return len(self.list_of_acc_files)

    def __getitem__(self, index):
        acc_file = self.list_of_acc_files[index]
        gt_file = self.list_of_gt_files[index]
        acc = dumped_data_reader(acc_file)
        gt = dumped_data_reader(gt_file)
        ti_in_keys = 'ti' in acc.keys()

        # relaxation time vector, can be 'ti' fit T1 mapping or 'te' for T2 mapping.
        tvec = acc['ti'] if ti_in_keys else acc['te']
        acc_kspace = acc['kspace']
        acc_sensitivity = acc['senstivity']
        acc_under_sampling_mask = acc['us']
        acs_mask = get_acs_mask(acc_under_sampling_mask, half_bandwidth=12)

        gt_kspace = gt['kspace']
        sos_recon = gt['sos']
        sample = dict(tvec=tvec,
                    acc_kspace=acc_kspace,              # (kx, ky, nc, nt)
                    us_mask=acc_under_sampling_mask,    # (kx, ky)
                    acs_mask=acs_mask,                  # (kx, ky)
                    init_sensitivity=acc_sensitivity,   # (kx, ky, nc, nt)
                    full_kspace=gt_kspace,              # (kx, ky, nc, nt)
                    full_sos=sos_recon                  # (kx, ky, nt)
                    )
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


