import typing
import torch
import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from direct.data.transforms import (complex_multiplication,
                                    modulus,
                                    conjugate,
                                    ifft2,
                                    fft2,
                                    root_sum_of_squares)
import direct.functionals as direct_func
from models.loss import SSIMLoss, NuclearNormLoss
from direct.nn.unet.unet_2d import UnetModel2d
import models.utils as mutils
from data.slicedqmridata import (SlicedQuantitativeMRIDatasetListSplit,
                                 SlicedQuantitativeMRIDataset,
                                 qmri_data_collate_fn)
from data.paths import CMRxReconDatasetPath
from data.transforms import get_default_sliced_qmr_transform
from configuration.config import (TrainerConfig,
                                  ReconstructionBackboneConfig,
                                  SensitivityRefinementModuleConfig,
                                  DataSetConfiguration,
                                  TrainingTrackingConfig)
from models.recurrentvarnet import RecurrentVarNet
from models.tricathlon import QuantitativeMRIReconstructionNet
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from dataclasses import asdict
import numpy as np


class QuantitativeMRITrainer(object):
    def __init__(self, run_name: str, path_handler: CMRxReconDatasetPath,
                 split: int = 0,
                 disable_tracker: bool = False,
                 recon_config: typing.Optional[ReconstructionBackboneConfig] = None,
                 ser_config: typing.Optional[SensitivityRefinementModuleConfig] = None,
                 data_set_config: typing.Optional[DataSetConfiguration] = None,
                 training_config: typing.Optional[TrainerConfig] = None,
                 tracker_config: typing.Optional[TrainingTrackingConfig] = None
                 ) -> None:
        """
        Initialize the training object.
        :param run_name: Set a name for a certain training run.
        :param path_handler: Dataset path handler object, with which we can get raw/prepared dataset paths and training dump base paths.
        :param split: Dataset split, integer number in [0, 5) for 5-fold cross validation.
        :param disable_tracker: Disable the training tracker.
        :param recon_config: Reconstruction network configuration, currently only RecurrentVarNet is supported.
        :param ser_config: Sensitivity estimation network configuration, currently only U-Net 2D is supported.
        :param data_set_config: Dataset loading configuration with attributes acceleration_factors and modality.
        :param training_config: Training configuration file.
        :param tracker_config: Tracker configuration file.
        """
        self.run_name = run_name
        self.split: int = split
        self.disable_tracker: int = disable_tracker
        self.path_handler = path_handler

        # Configurable variables
        self.recon_model_config = ReconstructionBackboneConfig() if recon_config is None else recon_config
        self.sensitivity_model_config = SensitivityRefinementModuleConfig() if ser_config is None else ser_config
        self.training_config = TrainerConfig() if training_config is None else training_config
        self.tracker_config = TrainingTrackingConfig() if tracker_config is None else tracker_config
        self.dataset_config = DataSetConfiguration() if training_config is None else data_set_config
        self.save_optimizer_every = self.training_config.save_optimizer_factor * self.training_config.save_checkpoint_every

        # training status
        self.epoch = 0
        self.global_step = 0

        # 1. load data set, we load the related paths from YAML, and then create the split dataset
        coil_type = "MultiCoil"
        training_set_base = self.path_handler.get_sliced_data_path(coil_type,
                                                                   "Mapping",
                                                                   "TrainingSet")
        sliced_dataset_files = SlicedQuantitativeMRIDatasetListSplit(training_set_base,
                                                                     acceleration_factors=self.dataset_config.acceleration_factors,
                                                                     modalities=(self.dataset_config.modality,),
                                                                     make_split=True,
                                                                     overwrite_split=False)

        # Now we have the list of file paths.
        file_lists_dict = sliced_dataset_files.splits[self.split]
        self.training_file_lists = file_lists_dict['training']
        self.validation_file_lists = file_lists_dict['validation']

        # set up transforms to be performed on the raw data, create Dataset & DataLoader.
        transforms = get_default_sliced_qmr_transform()
        self.training_set = SlicedQuantitativeMRIDataset(*self.training_file_lists,
                                                         transforms=transforms)
        self.validation_set = SlicedQuantitativeMRIDataset(*self.validation_file_lists,
                                                           transforms=transforms)
        self.training_loader = DataLoader(self.training_set,
                                          batch_size=self.training_config.batch_size,
                                          shuffle=True,
                                          num_workers=self.training_config.dataloader_num_workers,
                                          pin_memory=True,
                                          collate_fn=qmri_data_collate_fn)
        self.validation_loader = DataLoader(self.validation_set,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=self.training_config.dataloader_num_workers,
                                            pin_memory=True,
                                            collate_fn=qmri_data_collate_fn
                                            )
        self.training_steps_per_epoch = len(self.training_loader)

        # 2. set-up models
        recon_model_config = asdict(self.recon_model_config)
        _supported_operators = {
            'fft2': fft2,
            'ifft2': ifft2
        }
        recon_model_config['forward_operator'] = _supported_operators[recon_model_config['forward_operator']]
        recon_model_config['backward_operator'] = _supported_operators[recon_model_config['backward_operator']]

        model = RecurrentVarNet(
            **recon_model_config
        ).cuda().float()

        sensitivity_model_config = asdict(self.sensitivity_model_config)
        additional_model = UnetModel2d(
            **sensitivity_model_config
        ).cuda().float()
        self.recon_model = QuantitativeMRIReconstructionNet(model,
                                                            additional_model)

        # set up optimizers and lr_schedulers
        self.optimizer = torch.optim.Adam(self.recon_model.parameters(),
                                          lr=self.training_config.lr_init)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lr_lambda=lambda e: (
                                                                                           1 - e / self.training_config.max_epochs) ** self.training_config.lr_decay)

        # 3. set up loss functions
        self.loss_functions = dict()
        for loss_name in self.training_config.combined_loss_weight.keys():
            if loss_name.lower() == 'nmse':
                loss_fn = direct_func.nmse.NMSELoss()
            elif loss_name.lower() == 'ssim':
                loss_fn = SSIMLoss()
            elif loss_name.lower() == 'nuc':
                loss_fn = NuclearNormLoss()
            else:
                raise ValueError(f"Loss function with name {loss_name} is not supported yet!")
            self.loss_functions[loss_name] = loss_fn

        # 4. set-up paths
        self.expr_dump_base = self.path_handler.get_dump_data_path(self.run_name, f"fold_{self.split}")
        self.model_dump_base = self.expr_dump_base / "models"
        self.validation_save_base = self.expr_dump_base / "validation"
        self.model_log_base = self.expr_dump_base / "logs"

        self.model_dump_base.mkdir(parents=True, exist_ok=True)
        self.validation_save_base.mkdir(parents=True, exist_ok=True)
        self.model_log_base.mkdir(parents=True, exist_ok=True)

        # 5. tensorboard/weight&bias, whatever...
        if not self.disable_tracker:
            self.training_tracker = SummaryWriter(self.model_log_base)
        else:
            self.training_tracker = None

        # 6. resume if applicable
        self._initialize_network()
        if training_config.load_pretrained_latest:
            self.resume_latest()

    def _initialize_network(self):
        """
        Initialize the network weights from pretraining or Kaiming intialization.
        """
        if self.training_config.load_pretrained_weight is not None:
            state_dict = torch.load(self.training_config.load_pretrained_weight)
            self.recon_model.load_state_dict(state_dict, strict=True)
        else:
            mutils.kaiming_init_model(self.recon_model)

    def _compute_loss(self, pred: typing.Dict[str, torch.Tensor],
                      gt: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
        """
        Compute all the loss functions for prediction the ground truth.
        :param pred: prediction of the network with keys ('sensitivity', 'pred_kspace', 'rss', 'rss_flattened')
        :param gt: ground truth of the network with keys ('full_kspace', 'rss', 'rss_flattened')
        :return: The loss dictionary.
        """
        loss_value_dict = dict()
        total_loss: torch.Tensor = 0.
        dt, dev = pred['rss_flattened'].dtype, pred['rss_flattened'].device
        for loss_name, loss_fn in self.loss_functions.items():
            if loss_name.lower() == 'ssim':
                loss_val = loss_fn(pred['rss_flattened'],
                                   gt['rss_flattened'].to(dtype=dt, device=dev),
                                   torch.amax(gt['rss_flattened'].to(dtype=dt, device=dev), dim=(1, 2, 3)),
                                   reduced=True)
            elif loss_name.lower() == 'nuc':
                loss_val = loss_fn(pred['rss'])
            else:
                loss_val = loss_fn(pred['rss'],
                                   gt['rss'].to(dtype=dt, device=dev))
            total_loss += self.training_config.combined_loss_weight[loss_name] * loss_val
            loss_value_dict[loss_name] = loss_val
        loss_value_dict['total'] = total_loss
        return loss_value_dict

    def train(self) -> None:
        """
        The training routine, loading data, forward pass, loss computation and backward.
        """
        self.recon_model.cuda().float()
        self.recon_model.train()
        n_steps_per_epoch = len(self.training_loader)
        self.global_step = self.epoch * n_steps_per_epoch

        for epoch in range(self.epoch, self.training_config.max_epochs):
            pbar = tqdm.tqdm(self.training_loader)
            for ind, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                # forward pass
                prediction = self.recon_model(batch)

                # convert k-space to RSS image
                pred_for_loss = mutils.get_rearranged_prediction(prediction, 'pred_kspace')
                full_for_loss = mutils.get_rearranged_prediction(batch, 'full_kspace')

                # compute loss
                loss_value_dict = self._compute_loss(pred_for_loss, full_for_loss)

                # backward pass
                total_loss = loss_value_dict['total']
                total_loss.backward()
                self.optimizer.step()

                # record the losses
                description_str = f'Epoch: {self.epoch + 1} '
                self.global_step += 1
                for k, l in loss_value_dict.items():
                    description_str += f'{k:s}: {l:.4f} '
                    if self.global_step % self.tracker_config.save_training_loss_every > 0 or self.disable_tracker:
                        self.training_tracker.add_scalar(f'loss/{k}', l.item(), self.global_step)
                pbar.set_description(description_str)

                if self.global_step % self.tracker_config.save_training_image_every > 0 or self.disable_tracker:
                    grid_gt = make_grid(full_for_loss['rss_flattened'], nrow=9,
                                        normalize=True, scale_each=True)
                    grid_pred = make_grid(pred_for_loss['rss_flattened'], nrow=9,
                                          normalize=True, scale_each=True)
                    self.training_tracker.add_image("training/pred", grid_pred,
                                                    global_step=self.global_step)
                    self.training_tracker.add_image("training/gt", grid_gt,
                                                    global_step=self.global_step)

            self.scheduler.step(self.epoch)
            self.save_latest_checkpoint()
            self.save_per_epoch()
            if self.epoch % self.training_config.validation_every == 0:
                self.validation()
        self.training_tracker.flush()
        self.training_tracker.close()

    def save_checkpoint(self, filename: str, save_optimizer: bool = False) -> None:
        """
        Save checkpoint to filename.model.
        :param filename: The filename of the checkpoint file.
        :param save_optimizer: If the optimizer related status dictionary should be saved.
        """
        save_dict = dict(epoch=self.epoch,  # end of this epoch
                         model=self.recon_model.state_dict())
        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['scheduler'] = self.scheduler.state_dict()
        torch.save(save_dict, self.model_dump_base / f"{filename}.model")

    def save_per_epoch(self) -> None:
        """
        Save checkpoints for every certain number of epochs.
        """
        if self.epoch % self.training_config.save_checkpoint_every == 0:
            save_optimizer = self.epoch % self.save_optimizer_every
            filename = f"epoch_{self.epoch :04d}"
            self.save_checkpoint(filename, save_optimizer=save_optimizer)

    def save_latest_checkpoint(self) -> None:
        """
        Save the latest epoch.
        """
        self.save_checkpoint("model_latest", save_optimizer=True)

    def validation(self) -> None:
        """
        Let's validate the results.
        """
        # keep the current mode and switch to evaluation mode
        model_status = self.recon_model.training
        self.recon_model.eval()
        validation_losses = {k: [] for k in self.training_config.combined_loss_weight.keys()}
        validation_losses['total'] = []
        pbar = tqdm.tqdm(self.validation_loader)
        for ind, batch in enumerate(pbar):
            prediction = self.recon_model(batch)

            # convert k-space to RSS image
            pred_for_loss = mutils.get_rearranged_prediction(prediction, 'pred_kspace')
            full_for_loss = mutils.get_rearranged_prediction(batch, 'full_kspace')

            # compute loss
            loss_value_dict = self._compute_loss(pred_for_loss, full_for_loss)
            for k in validation_losses.keys():
                validation_losses[k].append(loss_value_dict[k].item())

            if not self.disable_tracker and ind == 0:
                grid_gt = make_grid(full_for_loss['rss_flattened'], nrow=9,
                                    normalize=True, scale_each=True)
                grid_pred = make_grid(pred_for_loss['rss_flattened'], nrow=9,
                                      normalize=True, scale_each=True)
                self.training_tracker.add_image("validation/pred", grid_pred,
                                                global_step=self.global_step)
                self.training_tracker.add_image("validation/gt", grid_gt,
                                                global_step=self.global_step)

        if not self.disable_tracker:
            for k, l in validation_losses.items():
                self.training_tracker.add_scalar(
                    f'validation_loss/{k}', np.mean(l), self.global_step
                )
        self.recon_model.train(model_status)

    def resume_from_checkpoint(self, checkpoint: Path) -> None:
        """
        Resume training from a checkpoint.
        :param checkpoint: path to checkpoint.
        :return: None
        """
        state_dict = torch.load(checkpoint)
        self.epoch = state_dict['epoch'] + 1        # next epoch
        self.recon_model.load_state_dict(state_dict['model'])
        optim_sd = state_dict.get('optimizer', None)
        sched_sd = state_dict.get('scheduler', None)
        if optim_sd is not None:
            self.optimizer.load_state_dict(optim_sd)
        if sched_sd is not None:
            self.scheduler.load_state_dict(sched_sd)

    def resume_latest(self) -> None:
        checkpoint = self.model_dump_base / "model_latest.model"
        self.resume_from_checkpoint(checkpoint)
