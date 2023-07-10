from pathlib import Path
import typing
import os
import yaml


def load_yaml_for_paths(yaml_file: typing.Union[str, bytes, os.PathLike] = "./cmrxrecon_dataset.yaml"):
    """
    Load a YAML file for the dataset path configurations. The YAML file must contain the following keys with string values:
        "dataset_base": path to the dataset, which ends with "*/CMRxRecon/ChallengeData".
    :param yaml_file: Path to the YAML file.
    :return: Dataset paths as a dictionary. 
    """
    yaml_file = Path(yaml_file)
    with open(yaml_file) as stream:
        cmrxrecon_paths = yaml.load(stream, yaml.Loader)
    return cmrxrecon_paths


class CMRxReconDatasetPath:
    def __init__(self, yaml_file: typing.Union[str, bytes, os.PathLike]):
        """
        Initializer.
        :param yaml_file: path to the YAML file.
        """
        self.yaml_file = yaml_file
        basis_paths = load_yaml_for_paths(self.yaml_file)
        self.dataset_base = Path(basis_paths['dataset_base'])
        self.sliced_dataset_base = Path(basis_paths['sliced_dataset_base'])

    def get_raw_data_path(self, *args: typing.List[str]):
        folder = '/'.join(args)
        return self.dataset_base / folder

    def get_sliced_data_path(self, *args: typing.List[str]):
        folder = '/'.join(args)
        return self.sliced_dataset_base / folder



