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
        self.expr_dump_base = Path(basis_paths["expr_dump_base"])
        self.inference_dump_base = Path(basis_paths["inference_dump_base"])
        self.pretrained_base = Path(basis_paths["pretrained_base"])

    @staticmethod
    def _get_sub_folder_path(base: Path, *args: typing.List[str]) -> Path:
        """
        Get path to "base/args[0]/args[1]/...".
        :param base: The base path.
        :param args: Sub-folders as a list of strings.
        :return: Path("base/args[0]/args[1]/...").
        """
        folder = '/'.join(args)
        return base / folder

    def get_raw_data_path(self, *args: typing.List[str]):
        return self._get_sub_folder_path(self.dataset_base, *args)

    def get_sliced_data_path(self, *args: typing.List[str]):
        return self._get_sub_folder_path(self.sliced_dataset_base, *args)

    def get_dump_data_path(self,*args: typing.List[str]):
        return self._get_sub_folder_path(self.expr_dump_base, *args)



