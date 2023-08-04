import typing
import torch
import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from dataclasses import asdict
import models.utils as mutils
from data.slicedqmridata import (SlicedQuantitativeMRIDatasetListSplit,
                                 SlicedQuantitativeMRIDataset,
                                 qmri_data_robust_collate_fn)
from data.paths import CMRxReconDatasetPath
from data.transforms import get_default_sliced_qmr_transform
from configuration.config import (TrainerConfig,
                                  MappingModuleConfig,
                                  DataSetConfiguration,
                                  ImageDomainAugmentationConfig,
                                  PostTrainingValidationConfig,
                                  TrainingTrackingConfig)
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from models.tricathlon import MOLLIMappingNet, T2RelaxMappingNet
from .utils import make_heatmap_grid


class QuantitativeMappingTrainer(object):
    def __init__(self, run_name: str, path_handler: CMRxReconDatasetPath,
                 fold: int = 0,
                 disable_tracker: bool = False,
                 disable_tqdm: bool = True,
                 map_config: typing.Optional[MappingModuleConfig] = None,
                 data_set_config: typing.Optional[DataSetConfiguration] = None,
                 augmentation_config: typing.Optional[ImageDomainAugmentationConfig] = None,
                 training_config: typing.Optional[TrainerConfig] = None,
                 tracker_config: typing.Optional[TrainingTrackingConfig] = None,
                 post_training_config: typing.Optional[PostTrainingValidationConfig] = None
                 ) -> None:
        """
        Initialize the training object.
        :param run_name: Set a name for a certain training run.
        :param path_handler: Dataset path handler object, with which we can get raw/prepared dataset paths and training dump base paths.
        :param fold: Dataset fold, integer number in [0, 5) for 5-fold cross validation.
        :param disable_tracker: Disable the training tracker.
        :param disable_tqdm: Disable the tqdm output.
        :param map_config: Mapping net configuration, currently only U-Net 2D is supported.
        :param data_set_config: Dataset loading configuration with attributes acceleration_factors and modality.
        :param augmentation_config: Data augmentation configuration.
        :param training_config: Training configuration.
        :param tracker_config: Tracker configuration.
        :param post_training_config: Post training configuration.
        """
        self.run_name = run_name
        self.fold: int = fold
        self.disable_tracker: bool = disable_tracker
        self.disable_tqdm: bool = disable_tqdm
        self.path_handler = path_handler

        # Configurable variables
        self.mapping_model_config = MappingModuleConfig() if map_config is None else map_config
        self.training_config = TrainerConfig() if training_config is None else training_config
        self.tracker_config = TrainingTrackingConfig() if tracker_config is None else tracker_config
        self.dataset_config = DataSetConfiguration() if training_config is None else data_set_config
        self.augment_config = ImageDomainAugmentationConfig() if augmentation_config is None else augmentation_config
        self.post_training_config = PostTrainingValidationConfig() if post_training_config is None else post_training_config
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
        file_lists_dict = sliced_dataset_files.splits[self.fold]
        self.training_file_lists = file_lists_dict['training']
        self.validation_file_lists = file_lists_dict['validation']

        # set up transforms to be performed on the raw data, create Dataset & DataLoader.
        transforms = get_default_sliced_qmr_transform(self.augment_config)
        self.training_set = SlicedQuantitativeMRIDataset(*self.training_file_lists,
                                                         transforms=transforms)
        self.validation_set = SlicedQuantitativeMRIDataset(*self.validation_file_lists,
                                                           transforms=transforms)
        self.training_loader = DataLoader(self.training_set,
                                          batch_size=self.training_config.batch_size,
                                          shuffle=True,
                                          num_workers=self.training_config.dataloader_num_workers,
                                          pin_memory=True,
                                          collate_fn=qmri_data_robust_collate_fn)
        self.validation_loader = DataLoader(self.validation_set,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=self.training_config.dataloader_num_workers,
                                            pin_memory=True,
                                            collate_fn=qmri_data_robust_collate_fn
                                            )
        self.training_steps_per_epoch = len(self.training_loader)

        # 2. set-up models
        if self.dataset_config.modality == 't1map':
            self.mapping_model = MOLLIMappingNet(**asdict(self.mapping_model_config))
        elif self.dataset_config.modality == 't2map':
            self.mapping_model = T2RelaxMappingNet(**asdict(self.mapping_model_config))
        else:
            raise ValueError(f"Unknown modality {self.dataset_config.modality}!")

        # set up optimizers and lr_schedulers
        self.optimizer = torch.optim.Adam(self.mapping_model.parameters(),
                                          lr=self.training_config.lr_init)

        def lr_scheduling_epoch(e):
            """lr scheduler function supporting warm-up"""
            if e < self.training_config.warm_up_epochs:
                lr = self.training_config.warm_up_lr / self.training_config.lr_init
            else:
                lr = (1. - e / self.training_config.max_epochs) ** self.training_config.lr_decay
            return lr

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lr_lambda=lr_scheduling_epoch)

        # 3. set up loss functions
        self.loss_functions = dict()
        for loss_name in self.training_config.combined_loss_weight.keys():
            if loss_name.lower() == 'mse':
                loss_fn = lambda pred, gt, weight: ((pred - gt) ** 2).sum() / weight.sum()
            elif loss_name.lower() == 'l1':
                loss_fn = lambda pred, gt, weight: torch.abs(pred - gt).sum() / weight.sum()
            else:
                raise ValueError(f"Loss function with name {loss_name} is not supported yet!")
            self.loss_functions[loss_name] = loss_fn

        # 4. set-up paths
        self.expr_dump_base = self.path_handler.get_dump_data_path(self.run_name, f"fold_{self.fold}")
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

    def _update_augmentation_p(self) -> None:
        """
        Update augmentation probability during training.
        """
        decay_every = self.augment_config.decay_every
        decay_factor = self.augment_config.decay_factor

        if (self.epoch + 1) % decay_every == 0:
            # decay
            self.augment_config.p_contamination *= decay_factor
            self.augment_config.p_new_mask *= decay_factor
            self.augment_config.p_affine *= decay_factor
            self.augment_config.p_flip *= decay_factor
            self.augment_config.contamination_max_rel *= decay_factor

            # new transforms for the dataset
            transforms = get_default_sliced_qmr_transform(self.augment_config)
            self.training_set.update_transforms(transforms)

            # update training_loader
            self.training_loader = DataLoader(self.training_set,
                                          batch_size=self.training_config.batch_size,
                                          shuffle=True,
                                          num_workers=self.training_config.dataloader_num_workers,
                                          pin_memory=True,
                                          collate_fn=qmri_data_robust_collate_fn)

    def _initialize_network(self):
        """
        Initialize the network weights from pretraining or Kaiming intialization.
        """
        if self.training_config.load_pretrained_weight is not None:
            # Search for the pretrained model in the training dump base
            pretrained_config = self.training_config.load_pretrained_weight
            strict = pretrained_config.get('strict', True)
            pretrained_base = self.path_handler.expr_dump_base / pretrained_config['pretrained_run_name']
            fold = pretrained_config['fold']
            if fold == 'auto':
                fold = self.fold
            pretrained_base = pretrained_base / f"fold_{fold}" / "models"
            model_name = pretrained_config['model_checkpoint']
            if model_name == 'latest':
                model_name = 'model_latest.model'
            elif isinstance(model_name, int):
                model_name = f'epoch_{model_name:04d}.model'
            else:
                model_name = f'{model_name}.model'
            pretrained = pretrained_base / model_name
            assert pretrained.exists(), f"Cannot find pretrained model {pretrained}"
            state_dict = torch.load(pretrained)["model"]
            self.mapping_model.load_state_dict(state_dict, strict=strict)
        else:
            mutils.kaiming_init_model(self.mapping_model)

    def _compute_loss(self, pred: torch.Tensor,
                      gt: torch.Tensor, weight: typing.Optional[torch.Tensor] = None) -> typing.Dict[str, torch.Tensor]:
        """
        Compute all the loss functions for prediction the ground truth.
        :param pred: signal model prediction
        :param gt: ground truth signal
        :return: The loss dictionary.
        """
        loss_value_dict = dict()
        total_loss: torch.Tensor = 0.
        dt, dev = pred.dtype, pred.device
        for loss_name, loss_fn in self.loss_functions.items():
            loss_val = loss_fn(pred,
                               gt.to(dtype=dt, device=dev),
                               weight.to(dtype=dt, device=dev).expand(*pred.shape))
            total_loss += self.training_config.combined_loss_weight[loss_name] * loss_val
            loss_value_dict[loss_name] = loss_val
        loss_value_dict['total'] = total_loss
        return loss_value_dict

    def train(self) -> None:
        """
        The training routine, loading data, forward pass, loss computation and backward.
        """
        self.mapping_model.cuda().float()
        self.mapping_model.train()
        n_steps_per_epoch = len(self.training_loader)
        self.global_step = self.epoch * n_steps_per_epoch

        epoch_start = self.epoch
        gradient_accumulation = 2
        for self.epoch in range(epoch_start, self.training_config.max_epochs):
            pbar = tqdm.tqdm(self.training_loader, disable=self.disable_tqdm)
            self.optimizer.zero_grad()

            for ind, batch in enumerate(pbar):
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].cuda().float()
                t_relax = batch['tvec']
                if t_relax is None:
                    continue
                t_relax = t_relax * 1e-3
                rss_gt = mutils.get_rearranged_prediction(batch, 'full_kspace')['rss']                  # (nb, kx, ky, kt)
                rss_gt = torch.permute(rss_gt, (0, 3, 1, 2))
                t_relax = t_relax.expand(*rss_gt.shape)
                net_input = torch.cat((t_relax, rss_gt), dim=1)
                pmap = self.mapping_model(net_input)
                signal_pred = torch.abs(self.mapping_model.signal_model(t_relax, pmap))
                weight = 1. - self.mapping_model.get_air_mask(rss_gt)
                loss_value_dict = self._compute_loss(signal_pred, rss_gt, weight)
                total_loss = loss_value_dict['total']
                total_loss.backward()
                if ((ind + 1) % gradient_accumulation == 0) or (ind + 1 == len(self.training_loader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # record the losses
                curr_lr = self.scheduler.get_last_lr()[0]
                description_str = f'Epoch: {self.epoch + 1} LR: {curr_lr:.3e} '
                self.global_step += 1
                for k, l in loss_value_dict.items():
                    description_str += f'{k:s}: {l:.3e} '
                    if self.global_step % self.tracker_config.save_training_loss_every == 0 and (
                    not self.disable_tracker):
                        self.training_tracker.add_scalar(f'loss/{k}', l.item(), self.global_step)
                pbar.set_description(description_str)

                if self.global_step % self.tracker_config.save_training_image_every == 0 and (not self.disable_tracker):
                    with torch.no_grad():
                        t_relax_map = self.mapping_model.get_t_pred(pmap)
                        air = self.mapping_model.get_air_mask(rss_gt)
                        vmax = 2.0 if self.dataset_config.modality == 't1map' else 0.12
                        colormap = 'jet' if self.dataset_config.modality == 't1map' else 'plasma'
                        pred_images = make_heatmap_grid(t_relax_map, air=air, vmax=vmax, cmap=colormap, nrow=4)
                    self.training_tracker.add_image("training/pred", pred_images,
                                                    global_step=self.global_step)

            # end of epoch routines
            self._update_augmentation_p()                       # decay augmentation p if applicable
            self.scheduler.step(self.epoch)                     # lr update
            self.save_latest_checkpoint()                       # save latest
            self.save_per_epoch()                               # save checkpoint if applicable

            # validation
            if (self.epoch + 1) % self.training_config.validation_every == 0:
                torch.cuda.empty_cache()
                self.validation()
        if not self.disable_tracker:
            self.training_tracker.flush()
            self.training_tracker.close()

    def save_checkpoint(self, filename: str, save_optimizer: bool = False) -> None:
        """
        Save checkpoint to filename.model.
        :param filename: The filename of the checkpoint file.
        :param save_optimizer: If the optimizer related status dictionary should be saved.
        """
        save_dict = dict(epoch=self.epoch,  # end of this epoch
                         model=self.mapping_model.state_dict())
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

    def validation(self, dump: bool = False) -> typing.Dict[str, typing.Any]:
        pass

    def resume_from_checkpoint(self, checkpoint: Path) -> None:
        """
        Resume training from a checkpoint.
        :param checkpoint: path to checkpoint.
        :return: None
        """
        self.mapping_model.cuda()
        state_dict = torch.load(checkpoint, map_location='cuda')
        self.epoch = state_dict['epoch'] + 1  # next epoch
        self.global_step = state_dict['epoch'] * self.training_steps_per_epoch
        self.mapping_model.load_state_dict(state_dict['model'])
        optim_sd = state_dict.get('optimizer', None)
        sched_sd = state_dict.get('scheduler', None)
        if optim_sd is not None:
            self.optimizer.load_state_dict(optim_sd)
        if sched_sd is not None:
            self.scheduler.load_state_dict(sched_sd)

    def resume_latest(self) -> None:
        checkpoint = self.model_dump_base / "model_latest.model"
        self.resume_from_checkpoint(checkpoint)
