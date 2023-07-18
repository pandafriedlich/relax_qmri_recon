import typing
import os
from .utils import parse_inference_yaml, build_models_from_config, ensemble_prediction
import tqdm
import torch
import numpy as np
from scipy.io import savemat
from data.paths import CMRxReconDatasetPath
from data.qmritestloader import QuantitativeMRIAccelerationXDataset
from data.transforms import get_default_raw_qmr_transform


# setup models
class QuantitativeMRIReconTester:
    def __init__(self, run_name: str,
                 path_handler: CMRxReconDatasetPath,
                 inference_file: typing.Union[str, bytes, os.PathLike]):
        """
        Initialize the tester.
        :param run_name: Name for the test run, the folder for saving test data will have the same name.
        :param path_handler: Handling the Dataset folders.
        :param inference_file: Inference configuration file.
        """
        self.run_name = run_name  # folder name for saving
        self.path_handler = path_handler
        self.test_set_base = path_handler.get_raw_data_path("MultiCoil",
                                                            "Mapping",
                                                            "ValidationSet")
        self.save_base = path_handler.inference_dump_base / self.run_name
        self.test_configs = parse_inference_yaml(inference_file)

    def run_inference_on_acceleration(self, acceleration: str = 'AccFactor04') -> None:
        """
        Run inference on data of a specific acceleration factor.
        :param acceleration: Can be one of ('AccFactor04', 'AccFactor08', 'AccFactor10').
        :return: None
        """
        assert acceleration in ('AccFactor04', 'AccFactor08', 'AccFactor10'), f"Unknown acceleration {acceleration}"
        dataset_base = self.test_set_base / acceleration
        config_acc = self.test_configs[acceleration]
        save_base = self.save_base / "MultiCoil" / "Mapping" / "ValidationSet" / acceleration
        save_base.mkdir(exist_ok=True, parents=True)

        # Build-up t1 & t2 models
        config_acc_t1 = config_acc['t1']
        t1_models = build_models_from_config(dump_base=self.path_handler.expr_dump_base,
                                             model_config=config_acc_t1)
        config_acc_t2 = self.test_configs[acceleration]['t2']
        t2_models = build_models_from_config(dump_base=self.path_handler.expr_dump_base,
                                             model_config=config_acc_t2)

        # initialize test dataset
        test_set = QuantitativeMRIAccelerationXDataset(dataset_base,
                                                       transforms=get_default_raw_qmr_transform())

        # here we go!
        pbar = tqdm.tqdm(test_set)
        with torch.no_grad():
            # turn-off backward graph to save GPU memory
            for sample in pbar:
                pbar.set_description(f"Subject: {sample['path']}")
                t1in, t2in = sample['t1'], sample['t2']
                t1_pred = [ensemble_prediction(t1_models, d1) for d1 in t1in]
                t1_pred_rss = [t1p['rss'].detach().cpu().numpy() for t1p in t1_pred]
                t1_pred_rss = np.transpose(np.stack(t1_pred_rss, axis=-1),  # (kx, ky, nt, ns)
                                           (0, 1, 3, 2))                    # (kx, ky, ns, nt)
                t2_pred = [ensemble_prediction(t2_models, d2) for d2 in t2in]
                t2_pred_rss = [t2p['rss'].detach().cpu().numpy() for t2p in t2_pred]
                t2_pred_rss = np.transpose(np.stack(t2_pred_rss, axis=-1),  # (kx, ky, nt, ns)
                                           (0, 1, 3, 2))                    # (kx, ky, ns, nt)

                subject_save_base = save_base / sample['path']
                subject_save_base.mkdir(exist_ok=True, parents=True)
                savemat(subject_save_base / "T1map.mat",
                        dict(img4ranking=t1_pred_rss),
                        do_compression=True)
                savemat(subject_save_base / "T2map.mat",
                        dict(img4ranking=t2_pred_rss),
                        do_compression=True)
