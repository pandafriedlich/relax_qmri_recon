import joblib
import numpy as np
import torch
from pathlib import Path
import typing
import os
from .qmridata import ACCELERATION_FOLDER_MAP
from sklearn.model_selection import KFold
from .utils import dumped_data_reader, get_acs_mask
from numbers import Number
import hashlib


class SlicedQuantitativeMRIDatasetListSplit:
    """
    List all the files of the preprocessed (sliced) dataset.
    """
    def __init__(self, dataset_base: typing.Union[str, bytes, os.PathLike],
                 acceleration_factors: typing.Tuple[Number, ...],
                 modalities: typing.Tuple[str, ...],
                 make_split: bool = True,
                 overwrite_split: bool = False) -> None:
        """

        Go through the dataset and get all the files.
        :param dataset_base: Path to the dataset base like `.../TrainingSet`.
        :param acceleration_factors: The acceleration factors to be included.
        :param modalities: The modalities to be included, can be ('t1map', ), ('t2map', ) or ('t1map', 't2map')
        :param make_split: If the dataset should be split.
        :param overwrite_split: If the split info file should be overwritten (Make a new split of the dataset).
        """
        self.dataset_base = Path(dataset_base)
        self.acceleration_factors = acceleration_factors
        self.modalities = modalities
        self.overwrite_split = overwrite_split

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

        # make splits
        if make_split:
            identifier = (self.modalities, self.acceleration_factors)
            identifier = str(identifier).encode()
            identifier = hashlib.sha1(identifier).hexdigest()
            split_file_path = self.dataset_base / f"split_{identifier:s}.info"
            if not split_file_path.exists() or self.overwrite_split:
                # splits of file lists
                self.splits = self.split()
                joblib.dump(self.splits, split_file_path)
            else:
                self.splits = joblib.load(split_file_path)
        else:
            self.splits = {'all': [self.list_of_acc_files, self.list_of_gt_files]}

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
        split_files = {ind: sf for ind, sf in zip(range(k), split_files)}
        split_files['all'] = [self.list_of_acc_files, self.list_of_gt_files]
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
        # acc_sensitivity = acc['senstivity']
        acc_under_sampling_mask = acc['us']
        acs_mask = get_acs_mask(acc_under_sampling_mask, half_bandwidth=12)

        gt_kspace = gt['kspace']
        # sos_recon = gt['sos']
        sample = dict(tvec=tvec,
                    acc_kspace=acc_kspace,              # (kx, ky, nc, nt)
                    us_mask=acc_under_sampling_mask,    # (kx, ky)
                    acs_mask=acs_mask,                  # (kx, ky)
                    # init_sensitivity=acc_sensitivity,   # (kx, ky, nc, nt)
                    full_kspace=gt_kspace,              # (kx, ky, nc, nt)
                    # full_sos=sos_recon                  # (kx, ky, nt)
                    )
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


def qmri_data_collate_fn(list_of_samples: typing.List[typing.Dict[str, typing.Any]]):
    """
    Customized collate function which stacks the samples along the batch axis (dim=0).
    :param list_of_samples: List of samples
    :return:
    """
    keys = list_of_samples[0].keys()
    batch = dict()
    for k in keys:
        val = [sample[k] for sample in list_of_samples]
        val = torch.stack(val, dim=0)
        batch[k] = val
    return batch



