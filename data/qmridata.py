import typing
import os
from pathlib import Path
import csv
import mat73
import numpy as np

ACCELERATION_FACTORS = (1., 4., 8., 10.)
ACCELERATION_FOLDER_MAP = {1.: 'FullSample', 4. : 'AccFactor04', 8.: 'AccFactor08', 10.: 'AccFactor10'}
ACCELERATION_FOLDER_INV_MAP = {v: k for k, v in ACCELERATION_FOLDER_MAP.items()}
ACCELERATION_K_SPACE_KEYWORDS = {r: 'kspace' + [f'sub_{r:02d}', 'full'][r > 1.] for r in ACCELERATION_FACTORS}
ACCELERATION_UNDER_SAMPLING_KEYWORDS = {r: [f'mask{r: 02d}', None][r > 1.] for r in ACCELERATION_FACTORS}

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
        mask = dat[ACCELERATION_UNDER_SAMPLING_KEYWORDS[acceleration_factor]]
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
        self.t1_map_kspace = super()._load_k_space_data(self.subject_base / "T1map.mat")
        self.t2_map_kspace = super()._load_k_space_data(self.subject_base / "T2map.mat")
        self.t1_map_t_inv = QuantitativeMappingSubjectData._load_inversion_time_csv(self.subject_base / "T1map.csv")
        self.t2_map_t_echo = QuantitativeMappingSubjectData._load_echo_time_csv(self.subject_base / "T2map.csv")
        self.t1_us_mask = None
        self.t2_us_mask = None

        # For fully sampled subject, there is no sampling mask. It makes sense, huh?
        if self.acceleration_factor > 1. :
            self.t1_us_mask = QuantitativeMappingSubjectData._load_under_sampling_mask(
                self.subject_base / "T1map_mask.mat", self.acceleration_factor)
            self.t2_us_mask = QuantitativeMappingSubjectData._load_under_sampling_mask(
                self.subject_base / "T2map_mask.mat", self.acceleration_factor
            )

    @staticmethod
    def _load_inversion_time_csv(filepath: typing.Union[str, bytes, os.PathLike]) -> np.ndarray:
        """
        Load TI for T1 mapping.
        :param filepath: path to *.csv file.
        :return: Inversion time array of shape (#slices, #readouts).
        """
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
        with open(filepath) as stream:
            reader = csv.reader(stream, delimiter=',')
            rows = list(reader)
        rows_echo_times = rows[1:]
        echo_times = []
        for row in rows_echo_times:
            assert len(row) == 2, "Invalid echo time row detected!"
            echo_time = float(row[1])
            echo_times.append(echo_time)
        echo_times = np.array(echo_times)  # (#readouts, )
        return echo_times



class CMRxReconQuantitativeRawDataset:
    """
    A quantitative MRI data set.
    """
    def __init__(self, dataset_base: typing.Union[str, bytes, os.PathLike]):
        """
        Initialize the object.
        :param dataset_base: path to `.../[TrainingSet, ValidationSet]`
        """
        self.dataset_base = Path(dataset_base)
        self.acceleration_folders = None

