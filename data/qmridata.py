import typing
import os
from pathlib import Path
import csv
import mat73
import numpy as np
import tqdm
from .utils import SoS_Reconstruction
import joblib
from .utils import estimate_sensitivity_map

ACCELERATION_FACTORS = (1., 4., 8., 10.)
ACCELERATION_FOLDER_MAP = {1.: 'FullSample', 4.: 'AccFactor04', 8.: 'AccFactor08', 10.: 'AccFactor10'}
ACCELERATION_FOLDER_INV_MAP = {v: k for k, v in ACCELERATION_FOLDER_MAP.items()}
ACCELERATION_K_SPACE_KEYWORDS = {r: 'kspace' + ['_full', f'_sub{int(r):02d}'][r > 1.] for r in ACCELERATION_FACTORS}
ACCELERATION_UNDER_SAMPLING_KEYWORDS = {r: [None, f'mask{int(r):02d}'][r > 1.] for r in ACCELERATION_FACTORS}


class SubjectData:
    """
    An encapsulation for all data of a certain subject, this class handles the I/O of the provided .mat .csv and .nii files.
    """

    def __init__(self, subject_base: typing.Union[str, bytes, os.PathLike],
                 acceleration_factor: typing.Union[int, float] = 1):
        """
        Initialize a subject data object.
        :param subject_base: Path to the subject data.
        :param acceleration_factor: Acceleration factor R.
        """
        self.subject_base = Path(subject_base)
        self.acceleration_factor = float(acceleration_factor)

    @staticmethod
    def _load_k_space_data(filepath: typing.Union[str, bytes, os.PathLike],
                           acceleration_factor: typing.Union[int, float]) -> np.ndarray:
        """
        Load k-space data from a mat73 file.
        :param filepath: path to the .mat file.
        :return: k-space data of shape (kx, ky, nc, ns, nt).
        """
        dat = mat73.loadmat(filepath)
        k_space = dat[ACCELERATION_K_SPACE_KEYWORDS[acceleration_factor]]
        return k_space

    @staticmethod
    def _load_under_sampling_mask(filepath: typing.Union[str, bytes, os.PathLike],
                                  acceleration_factor: typing.Union[int, float]) -> np.ndarray:
        """
        Load under-sampling mask from a mat73 file.
        :param filepath: path to the .mat file.
        :return: mask of shape (kx, ky).
        """
        dat = mat73.loadmat(filepath)
        mask = dat[ACCELERATION_UNDER_SAMPLING_KEYWORDS[float(acceleration_factor)]]
        return mask


class QuantitativeMappingSubjectData(SubjectData):
    """
    An encapsulation for all qMRI data of a certain subject.
    """

    def __init__(self, subject_base: typing.Union[str, bytes, os.PathLike],
                 acceleration_factor: typing.Union[int, float] = 1):
        """
        Initialize a subject data object.
        :param subject_base: Path to the subject data e.g. `*/Mapping/TraininingSet/AccFactor04/P001`.
        :param acceleration_factor: Acceleration factor R.
        """
        super().__init__(subject_base, acceleration_factor)

        # Load all related data
        self.t1_map_kspace = super()._load_k_space_data(self.subject_base / "T1map.mat",
                                                        acceleration_factor=self.acceleration_factor)
        self.t2_map_kspace = super()._load_k_space_data(self.subject_base / "T2map.mat",
                                                        acceleration_factor=self.acceleration_factor)
        self.t1_map_t_inv = QuantitativeMappingSubjectData._load_inversion_time_csv(self.subject_base / "T1map.csv")
        self.t2_map_t_echo = QuantitativeMappingSubjectData._load_echo_time_csv(self.subject_base / "T2map.csv")
        # self.t1_sos_recon = None
        # self.t2_sos_recon = None
        # self.t1_sensitivity = None
        # self.t2_sensitivity = None
        self.t1_us_mask = None
        self.t2_us_mask = None

        # For fully sampled subject, there is no sampling mask. It makes sense, huh?
        if self.acceleration_factor > 1.:
            self.t1_us_mask = QuantitativeMappingSubjectData._load_under_sampling_mask(
                self.subject_base / "T1map_mask.mat", self.acceleration_factor)
            self.t2_us_mask = QuantitativeMappingSubjectData._load_under_sampling_mask(
                self.subject_base / "T2map_mask.mat", self.acceleration_factor)
            # estimate t1-sensitivity-map
            # sensitivity = np.stack([estimate_sensitivity_map(self.t1_map_kspace[..., j, :],
            #                        self.t1_us_mask) for j in range(self.t1_map_kspace.shape[-2])],
            #                        axis=-2)
            # self.t1_sensitivity = sensitivity

            # estimate t2-sensitivity-map
            # sensitivity = np.stack([estimate_sensitivity_map(self.t2_map_kspace[..., j, :],
            #                        self.t2_us_mask) for j in range(self.t2_map_kspace.shape[-2])],
            #                        axis=-2)
            # self.t2_sensitivity = sensitivity
        else:
            pass
            # self.t1_sos_recon = SoS_Reconstruction(self.t1_map_kspace, fft_dim=(0, 1), coil_dim=2,
            #                                        complex_input=False, centered=True, normalized=True)
            # self.t2_sos_recon = SoS_Reconstruction(self.t2_map_kspace, fft_dim=(0, 1), coil_dim=2,
            #                                        complex_input=False, centered=True, normalized=True)

    def slicing(self) -> dict:
        """
        Get per-slice k-space data, under-sampling mask and inversion time/echo time.
        :return: dictionary of k-space, inversion/echo time and under-sampling mask.
        """
        n_slices_t1 = self.t1_map_kspace.shape[-2]
        n_slices_t2 = self.t2_map_kspace.shape[-2]
        t1 = [dict(ti=self.t1_map_t_inv[j, :] if self.t1_map_t_inv.size >= 9 else None,
                   kspace=self.t1_map_kspace[..., j, :],
                   # senstivity=self.t1_sensitivity[..., j, :] if self.acceleration_factor > 1. else None,
                   # sos=None if self.acceleration_factor > 1. else self.t1_sos_recon[..., j, :],
                   us=self.t1_us_mask)
              for j in range(n_slices_t1)]
        t2 = [dict(te=self.t2_map_t_echo,
                   kspace=self.t2_map_kspace[..., j, :],
                   # senstivity=self.t2_sensitivity[..., j, :] if self.acceleration_factor > 1. else None,
                   # sos=None if self.acceleration_factor > 1. else self.t2_sos_recon[..., j, :],
                   us=self.t2_us_mask)
              for j in range(n_slices_t2)]
        return dict(t1=t1, t2=t2)

    def get_kx_ky_shape(self) -> dict:
        """
        Get kx- and ky-direction size.
        :return:
        """
        t1_spatial_size = self.t1_map_kspace.shape
        t2_spatial_size = self.t2_map_kspace.shape
        return dict(t1_shape=t1_spatial_size, t2_shape=t2_spatial_size)

    @staticmethod
    def _load_inversion_time_csv(filepath: typing.Union[str, bytes, os.PathLike]) -> np.ndarray:
        """
        Load TI for T1 mapping.
        :param filepath: path to *.csv file.
        :return: Inversion time array of shape (#slices, #readouts).
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return np.array([])
        with open(filepath) as stream:
            reader = csv.reader(stream, delimiter=',')
            rows = list(reader)
        rows_inversion_times = rows[1:]
        inversion_times = []
        for row in rows_inversion_times:
            inversion_time = [float(col) for col in row if (not col.startswith('TI')) and (col != '')]
            inversion_times.append(inversion_time)
        inversion_times = np.array(inversion_times)  # (#readouts, #slices)
        inversion_times = inversion_times.T  # (#slices, #readouts)
        return inversion_times

    @staticmethod
    def _load_echo_time_csv(filepath: typing.Union[str, bytes, os.PathLike]) -> np.ndarray:
        """
        Load TE for T2 mapping.
        :param filepath: path to *.csv file.
        :return: Echo time array of shape (#slices, ).
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return np.array([])
        with open(filepath) as stream:
            reader = csv.reader(stream, delimiter=',')
            rows = list(reader)
        rows_echo_times = rows[1:]
        echo_times = []
        for row in rows_echo_times:
            # assert len(row) == 2, "Invalid echo time row detected!"
            echo_time = float(row[1])
            echo_times.append(echo_time)
        echo_times = np.array(echo_times)  # (#readouts, )
        return echo_times


class CMRxReconQuantitativeAccelerationXDataset:
    """
    A quantitative MRI dataset of a certain acceleration factor.
    """

    def __init__(self, dataset_base: typing.Union[str, bytes, os.PathLike]):
        """
        Initialize the object.
        :param dataset_base: path to .../AccFactorXX or .../FullSample
        """
        self.dataset_base = Path(dataset_base)
        self.acceleration_factor = ACCELERATION_FOLDER_INV_MAP[self.dataset_base.stem]
        self.subject_paths = list(self.dataset_base.glob("P*"))
        self.n_subjects = len(self.subject_paths)
        self.t1_shapes = []
        self.t2_shapes = []

    def make_split_files(self, sliced_data_base: typing.Union[str, bytes, os.PathLike]) -> None:
        """
        Split and save the patient to data per-slice.
        :return: None
        """
        sliced_data_base = Path(sliced_data_base)
        pbar = tqdm.tqdm(self.subject_paths)
        for subject_path in pbar:
            subject_id = subject_path.stem
            pbar.set_description(subject_id)
            subject = QuantitativeMappingSubjectData(subject_path,
                                                     acceleration_factor=self.acceleration_factor)
            shapes = subject.get_kx_ky_shape()
            self.t1_shapes.append(shapes['t1_shape'])
            self.t2_shapes.append(shapes['t2_shape'])
            subject_sliced = subject.slicing()

            # T1-mapping file
            t1_map_base = sliced_data_base / subject_id / 't1map'
            t1_map_base.mkdir(exist_ok=True, parents=True)
            t1 = subject_sliced['t1']  # keys: (kspace, ti, us, sos)
            for slice_ind, t1_dat in enumerate(t1):
                filename = f"slice_{slice_ind:02d}.dat"
                joblib.dump(t1_dat, t1_map_base / filename)

            # T2-mapping file
            t2_map_base = sliced_data_base / subject_id / 't2map'
            t2_map_base.mkdir(exist_ok=True, parents=True)
            t2 = subject_sliced['t2']
            for slice_ind, t2_dat in enumerate(t2):
                filename = f"slice_{slice_ind:02d}.dat"
                joblib.dump(t2_dat, t2_map_base / filename)

        joblib.dump(self.t1_shapes, sliced_data_base / 't1shapes.dat')
        joblib.dump(self.t2_shapes, sliced_data_base / 't2shapes.dat')


class CMRxReconQuantitativeRawDataset:
    """
    A quantitative MRI dataset.
    """

    def __init__(self, dataset_base: typing.Union[str, bytes, os.PathLike]):
        """
        Initialize the object.
        :param dataset_base: path to `.../[TrainingSet, ValidationSet]`
        """
        self.dataset_base = Path(dataset_base)
        self.acceleration_data = {
            r: CMRxReconQuantitativeAccelerationXDataset(self.dataset_base / ACCELERATION_FOLDER_MAP[r])
            for r in ACCELERATION_FACTORS if r > 1.}
        self.ground_truth_data = CMRxReconQuantitativeAccelerationXDataset(
            self.dataset_base / ACCELERATION_FOLDER_MAP[1.])
        self.ground_truth_available = self.ground_truth_data.n_subjects > 0
        self.t1_shapes = []
        self.t2_shapes = []

    def save_split_files(self, sliced_base: typing.Union[str, bytes, os.PathLike]) -> None:
        """
        Save data per slice to reduce meaningless IO.
        :return: None
        """
        sliced_base = Path(sliced_base)
        # process acceleration folders
        for r in ACCELERATION_FACTORS:
            if r > 1.:
                acc_data = self.acceleration_data[r]
                acc_data.make_split_files(sliced_data_base=sliced_base / ACCELERATION_FOLDER_MAP[r])
            else:
                self.ground_truth_data.make_split_files(sliced_data_base=sliced_base / ACCELERATION_FOLDER_MAP[r])

