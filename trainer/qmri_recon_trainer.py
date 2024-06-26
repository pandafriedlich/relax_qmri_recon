import typing
import joblib
import torch
import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from direct.data.transforms import (ifft2,
                                    fft2)
import direct.functionals as direct_func
from models.loss import SSIMLoss, NuclearNormLoss
from direct.nn.unet.unet_2d import UnetModel2d
import models.utils as mutils
from data.slicedqmridata import (SlicedQuantitativeMRIDatasetListSplit,
                                 SlicedQuantitativeMRIDataset,
                                 qmri_data_robust_collate_fn)
from data.paths import CMRxReconDatasetPath
from data.transforms import get_default_sliced_qmr_transform
from configuration.config import (TrainerConfig,
                                  ReconstructionBackboneConfig,
                                  SensitivityRefinementModuleConfig,
                                  DataSetConfiguration,
                                  ImageDomainAugmentationConfig,
                                  MappingModuleConfig,
                                  PostTrainingValidationConfig,
                                  TrainingTrackingConfig)
from models.recurrentunet import RecurrentVarNet
from models.tricathlon import QuantitativeMRIReconstructionNet, MOLLIMappingNet, T2RelaxMappingNet
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from dataclasses import asdict
import numpy as np
from .utils import make_heatmap_grid
from models.loss import weighted_l1_loss
freeze_mapping = True


class QuantitativeMRITrainer(object):
    def __init__(self, run_name: str, path_handler: CMRxReconDatasetPath,
                 fold: int = 0,
                 disable_tracker: bool = False,
                 disable_tqdm: bool = True,
                 recon_config: typing.Optional[ReconstructionBackboneConfig] = None,
                 ser_config: typing.Optional[SensitivityRefinementModuleConfig] = None,
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
        :param recon_config: Reconstruction network configuration, currently only RecurrentVarNet is supported.
        :param ser_config: Sensitivity estimation network configuration, currently only U-Net 2D is supported.
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
        self.recon_model_config = ReconstructionBackboneConfig() if recon_config is None else recon_config
        self.sensitivity_model_config = SensitivityRefinementModuleConfig() if ser_config is None else ser_config
        self.mapping_net_config = map_config
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

        # qMRI mapping models
        self.mapping_model: typing.Union[MOLLIMappingNet, T2RelaxMappingNet, None] = None
        if self.mapping_net_config is not None:
            if self.dataset_config.modality == 't1map':
                self.mapping_model = MOLLIMappingNet(**asdict(self.mapping_net_config))
            elif self.dataset_config.modality == 't2map':
                self.mapping_model = T2RelaxMappingNet(**asdict(self.mapping_net_config))
            else:
                raise ValueError("Modality unknown!")

        # set up optimizers and lr_schedulers
        if self.mapping_model is not None and (not freeze_mapping):
            params = list(self.mapping_model.parameters()) + list(self.recon_model.parameters())
        else:
            params = self.recon_model.parameters()
        self.optimizer = torch.optim.Adam(params,
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
            if loss_name.lower() == 'nmse':
                loss_fn = direct_func.nmse.NMSELoss()
            elif loss_name.lower() == 'l1':
                loss_fn = torch.nn.L1Loss(reduction='mean')
            elif loss_name.lower() == 'ssim':
                loss_fn = SSIMLoss()
            elif loss_name.lower() == 'nuc':
                loss_fn = NuclearNormLoss()
            elif loss_name.lower().startswith('map_'):
                loss_fn = None
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

    def _load_local_weight(self, pretrained_config: typing.Dict[str, typing.Any], key: str='model') -> typing.Dict[str, torch.Tensor]:
        """
        Load weight from local checkpoint file.
        :param pretrained_config: Dictionary containing run name, fold, and check point name.
        :param key: The key to find the checkpoint, if not found, use 'model'.
        :return: Network state dict.
        """
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
        # Load recon model
        checkpoint = torch.load(pretrained, map_location='cuda')
        if key in checkpoint.keys():
            state_dict = checkpoint[key]
        else:
            state_dict = checkpoint['model']
        return state_dict

    def _initialize_network(self):
        """
        Initialize the network weights from pretraining or Kaiming intialization.
        """
        if self.training_config.load_pretrained_weight is not None:
            self.recon_model.cuda().float()
            pretrained_config = self.training_config.load_pretrained_weight
            state_dict = self._load_local_weight(pretrained_config, key='model')
            self.recon_model.load_state_dict(state_dict, strict=pretrained_config.get('strict', True))
        else:
            mutils.kaiming_init_model(self.recon_model)

        if self.mapping_model is not None and self.training_config.load_pretrained_mapping is not None:
            self.mapping_model.cuda().float()
            pretrained_config = self.training_config.load_pretrained_mapping
            state_dict = self._load_local_weight(pretrained_config, key='mapping_model')
            self.mapping_model.load_state_dict(state_dict, strict=pretrained_config.get('strict', True))

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
            elif loss_name.lower().startswith('map_'):
                continue
            else:
                loss_val = loss_fn(pred['rss'],
                                   gt['rss'].to(dtype=dt, device=dev))
            total_loss += self.training_config.combined_loss_weight[loss_name] * loss_val
            loss_value_dict[loss_name] = loss_val
        loss_value_dict['total'] = total_loss
        return loss_value_dict

    def _mapping_forward(self, image: torch.Tensor, t_relax: torch.Tensor) -> torch.Tensor:
        """
        Compute quantitative mapping prediction.
        :param image: Input image of shape (nb, kt, kx, ky).
        :param t_relax: Relaxation time  (nb, kt, 1, 1)
        :return: Predicted parameter map.
        """
        t_relax = t_relax.expand(*image.shape)
        net_input = torch.cat((t_relax, image), dim=1)
        pmap = self.mapping_model(net_input)
        return pmap

    def _compute_pinn_loss(self, im: torch.Tensor, pmap: torch.Tensor, t_relax: torch.Tensor):
        """
        Compute the difference between signal predicted with the parameter map and the true signal, this is a special case of the Physics-informed Neural Network (PINN).
        :param im: image of shape (nb, kt, kx, ky)
        :param pmap: parameter map of shape (nb, #params, kx, ky)
        :param t_relax: Expanded relaxation time (Ti or TE) of shape (nb, kt, kx, ky).
        :return: The unsupervised PINN loss.
        """
        assert self.mapping_model is not None, "Mapping model is None!"
        signal_pred = torch.abs(self.mapping_model.signal_model(t_relax, pmap))     # (nb, kt, kx, ky)
        air = self.mapping_model.get_air_mask(im)
        weight = 1. - air
        loss_fn = weighted_l1_loss
        weight = weight.expand(*signal_pred.shape).to(dtype=signal_pred.dtype,
                                                      device=signal_pred.device)
        loss = loss_fn(signal_pred, im, weight)
        return loss

    def _compute_mapping_related_loss(self, rss_pred: torch.Tensor,
                                      rss_gt: torch.Tensor,
                                      t_relax: torch.Tensor) -> typing.Dict[str, torch.Tensor]:
        """
        Compute relaxometry related losses.
        :param rss_gt: Ground-truth magnitude image of shape (nb, kx, ky, kt).
        :param rss_pred: Predicted magnitude image (nb, kx, ky, kt)
        :param t_relax: Relaxation time vector (T_inv or T_echo) of shape (nb, kt, 1, 1)
        :return: Dictionary of all loss items.
        """
        rss_gt = torch.permute(rss_gt, (0, 3, 1, 2))            # (nb, kt, kx, ky)
        rss_pred = torch.permute(rss_pred, (0, 3, 1, 2))        # (nb, kt, kx, ky)
        air_gt = self.mapping_model.get_air_mask(rss_gt)
        air_pred = self.mapping_model.get_air_mask(rss_pred)

        rss_gt = rss_gt.to(dtype=rss_pred.dtype, device=rss_pred.device)
        t_relax = t_relax.to(dtype=rss_pred.dtype, device=rss_pred.device)

        # unsupervised fitting loss for gt
        pmap_gt = self._mapping_forward(rss_gt, t_relax)
        pinn_gt = self._compute_pinn_loss(rss_gt, pmap_gt, t_relax)
        Tx_map_gt = self.mapping_model.get_t_pred(pmap_gt)                          # (nb, 1, kx, ky)

        # unsupervised fitting loss for prediction
        pmap_pred = self._mapping_forward(rss_pred, t_relax)
        pinn_pred = self._compute_pinn_loss(rss_pred, pmap_pred, t_relax)
        Tx_map_pred = self.mapping_model.get_t_pred(pmap_pred)                      # (nb, 1, kx, ky)

        # Mapping difference.
        air = self.mapping_model.get_air_mask(rss_pred)
        weight = 1. - air.expand(*Tx_map_pred.shape).to(device=rss_pred.device, dtype=rss_pred.dtype)
        diff = weighted_l1_loss(Tx_map_pred, Tx_map_gt, weight)

        return dict(map_pinn_gt=pinn_gt,            # The loss items starts with 'map_'
                    map_pinn_pred=pinn_pred,
                    map_tx_diff=diff,
                    pmap_pred=pmap_pred,            # Below are intermediate variables
                    pmap_gt=pmap_gt,
                    air_pred=air_pred,
                    air_gt=air_gt,
                    tx_pred=Tx_map_pred,
                    tx_gt=Tx_map_gt)

    def train(self) -> None:
        """
        The training routine, loading data, forward pass, loss computation and backward.
        """
        self.recon_model.cuda().float()
        self.recon_model.train()
        n_steps_per_epoch = len(self.training_loader)
        self.global_step = self.epoch * n_steps_per_epoch
        if freeze_mapping and self.mapping_model is not None:
            self.mapping_model.eval()

        epoch_start = self.epoch
        gradient_accumulation = 8
        for self.epoch in range(epoch_start, self.training_config.max_epochs):
            pbar = tqdm.tqdm(self.training_loader, disable=self.disable_tqdm)
            self.optimizer.zero_grad()

            for ind, batch in enumerate(pbar):
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].cuda().float()

                prediction = self.recon_model(batch)

                # convert k-space to RSS image
                pred_for_loss = mutils.get_rearranged_prediction(prediction, 'pred_kspace')
                full_for_loss = mutils.get_rearranged_prediction(batch, 'full_kspace')

                # compute reconstruction loss
                loss_value_dict = self._compute_loss(pred_for_loss, full_for_loss)

                # mapping loss
                if self.mapping_model is not None:
                    t_relax = batch['tvec'] * 1e-3
                    mapping_value_dict = self._compute_mapping_related_loss(pred_for_loss['rss'],
                                                                            full_for_loss['rss'],
                                                                            t_relax)
                    for name, val in mapping_value_dict.items():
                        if name.startswith('map_'):
                            loss_value_dict[name] = val
                            loss_value_dict['total'] += val * self.training_config.combined_loss_weight[name]
                else:
                    mapping_value_dict = dict()

                # backward pass
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
                    if k == 'total':
                        description_str += f'{k:s}: {l:.3e} '
                    if self.global_step % self.tracker_config.save_training_loss_every == 0 and (not self.disable_tracker):
                        self.training_tracker.add_scalar(f'loss/{k}', l.item(), self.global_step)
                pbar.set_description(description_str)

                if self.global_step % self.tracker_config.save_training_image_every == 0 and (not self.disable_tracker):
                    # reconstruction
                    grid_gt = make_grid(full_for_loss['rss_flattened'], nrow=9,
                                        normalize=True, scale_each=True)
                    grid_pred = make_grid(pred_for_loss['rss_flattened'], nrow=9,
                                          normalize=True, scale_each=True)
                    self.training_tracker.add_image("training/pred", grid_pred,
                                                    global_step=self.global_step)
                    self.training_tracker.add_image("training/gt", grid_gt,
                                                    global_step=self.global_step)

                    # mapping
                    if self.mapping_model is not None:
                        cmap = {'t1map': 'jet', 't2map': 'plasma'}[self.dataset_config.modality]
                        vmax = {'t1map': 2., 't2map': 0.12}[self.dataset_config.modality]
                        grid_map_pred = make_heatmap_grid(mapping_value_dict['tx_pred'],
                                                          air=mapping_value_dict['air_pred'],
                                                          vmax=vmax, cmap=cmap)
                        grid_map_gt = make_heatmap_grid(mapping_value_dict['tx_gt'],
                                                        air=mapping_value_dict['air_gt'],
                                                        vmax=vmax, cmap=cmap)
                        self.training_tracker.add_image("training/pred_map", grid_map_pred,
                                                        global_step=self.global_step)
                        self.training_tracker.add_image("training/gt_map", grid_map_gt,
                                                        global_step=self.global_step)


            # end of epoch routines
            self._update_augmentation_p()                       # decay augmentation p if applicable
            self.scheduler.step(self.epoch)                     # lr update
            self.save_latest_checkpoint()                       # save latest
            self.save_per_epoch()                               # save checkpoint if applicable

            # validation
            if (self.epoch) % self.training_config.validation_every == 0:
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
                         model=self.recon_model.state_dict())
        if self.mapping_model is not None:
            save_dict['mapping_model'] = self.mapping_model.state_dict()
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

    def validation(self, post_training=False, dump: bool = False) -> typing.Dict[str, typing.Any]:
        """
        Let's validate the results.
        :param post_training: Indicates if it is a post-training validation.
        :param dump: Indicates if the validation results should be dumped.
        """
        if post_training:
            # load model weights
            if self.post_training_config.swa:
                epoch_start, epoch_end, update_bn_steps = (self.post_training_config.swa_epoch_start,
                                                           self.post_training_config.swa_epoch_end,
                                                           self.post_training_config.swa_update_bn_steps)
                swa_filename = f"model_swa_{epoch_start:04d}_{epoch_end:04d}_{update_bn_steps:04d}.model"
                swa_filepath = self.model_dump_base / swa_filename
                if (swa_filepath.exists() and self.post_training_config.swa_overwrite) or (not swa_filepath.exists()):
                    swa_state_dict = self.stochastic_weight_averaging(epoch_start=epoch_start, epoch_end=epoch_end,
                                                                      update_bn_steps=update_bn_steps)
                else:
                    swa_state_dict = torch.load(swa_filepath)
                self.epoch = -1
                self.recon_model.load_state_dict(swa_state_dict['model'])
                checkpoint_path = swa_filepath
            else:
                self.resume_latest()
                checkpoint_path = self.model_dump_base / "model_latest.model"
        else:
            checkpoint_path = self.model_dump_base / f"epoch_{self.epoch:04d}.model"

        # keep the current mode and switch to evaluation mode
        model_status = self.recon_model.training
        self.recon_model.eval()
        validation_losses = {k: [] for k in self.training_config.combined_loss_weight.keys()}
        validation_losses['total'] = []
        pbar = tqdm.tqdm(self.validation_loader, disable=self.disable_tqdm)
        for ind, batch in enumerate(pbar):
            with torch.no_grad():
                pbar.set_description(f"Epoch: {self.epoch} ")
                prediction = self.recon_model(batch)

            # convert k-space to RSS image
            pred_for_loss = mutils.get_rearranged_prediction(prediction, 'pred_kspace')
            full_for_loss = mutils.get_rearranged_prediction(batch, 'full_kspace')

            # compute loss
            loss_value_dict = self._compute_loss(pred_for_loss, full_for_loss)
            if self.mapping_model is not None:
                t_relax = batch['tvec'] * 1e-3
                mapping_value_dict = self._compute_mapping_related_loss(pred_for_loss['rss'],
                                                                        full_for_loss['rss'],
                                                                        t_relax)
                for name, val in mapping_value_dict.items():
                    if name.startswith('map_'):
                        loss_value_dict[name] = val
                        loss_value_dict['total'] += val * self.training_config.combined_loss_weight[name]
            else:
                mapping_value_dict = dict()
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
                # mapping
                if self.mapping_model is not None:
                    cmap = {'t1map': 'jet', 't2map': 'plasma'}[self.dataset_config.modality]
                    vmax = {'t1map': 2., 't2map': 0.12}[self.dataset_config.modality]
                    grid_map_pred = make_heatmap_grid(mapping_value_dict['tx_pred'],
                                                      air=mapping_value_dict['air_pred'],
                                                      vmax=vmax, cmap=cmap)
                    grid_map_gt = make_heatmap_grid(mapping_value_dict['tx_gt'],
                                                    air=mapping_value_dict['air_gt'],
                                                    vmax=vmax, cmap=cmap)
                    self.training_tracker.add_image("validation/pred_map", grid_map_pred,
                                                    global_step=self.global_step)
                    self.training_tracker.add_image("validation/gt_map", grid_map_gt,
                                                    global_step=self.global_step)

        if not self.disable_tracker:
            for k, l in validation_losses.items():
                self.training_tracker.add_scalar(
                    f'validation_loss/{k}', np.mean(l), self.global_step
                )
        self.recon_model.train(model_status)

        if dump:
            # dump validation results
            filename = f"valid_epoch_{self.epoch:d}.dat"
            filepath = self.validation_save_base / filename
            joblib.dump(dict(checkpoint_path=checkpoint_path, validation_losses=validation_losses), filepath)
        return validation_losses

    def resume_from_checkpoint(self, checkpoint: Path) -> None:
        """
        Resume training from a checkpoint.
        :param checkpoint: path to checkpoint.
        :return: None
        """
        self.recon_model.cuda().float()
        state_dict = torch.load(checkpoint, map_location='cuda')
        self.epoch = state_dict['epoch'] + 1  # next epoch
        self.global_step = state_dict['epoch'] * self.training_steps_per_epoch
        self.recon_model.load_state_dict(state_dict['model'])
        if self.mapping_model is not None and state_dict.get('mapping_model', None) is not None:
            self.mapping_model.cuda().float()
            self.mapping_model.load_state_dict(state_dict['mapping_model'])
        optim_sd = state_dict.get('optimizer', None)
        sched_sd = state_dict.get('scheduler', None)
        if optim_sd is not None:
            self.optimizer.load_state_dict(optim_sd)
        if sched_sd is not None:
            self.scheduler.load_state_dict(sched_sd)

    def resume_latest(self) -> None:
        checkpoint = self.model_dump_base / "model_latest.model"
        self.resume_from_checkpoint(checkpoint)

    def stochastic_weight_averaging(self, epoch_start: int,
                                    epoch_end: int,
                                    update_bn_steps: int = 100) -> typing.Dict[str, typing.Any]:
        """
        Perform stochastic weight averaging (SWA) in the weight space. SWA can counteract the randomness introduced by stochastic gradient descent.
        :param epoch_start: The first epoch number for SWA.
        :param epoch_end: The last ending epoch number for SWA.
        :param update_bn_steps: Steps of forward passes for batch normalization buffer update.
        :return: The SWA dictionary.
        """
        weight_aver = dict()  # initialize the averaged weight.
        num_models_average: float = 0.  # keep track of number of models that have been averaged.
        epoch_numbers_averaged = []  # keep track of epoch numbers that have been averaged.

        # weight space averaging
        pbar = tqdm.tqdm(range(epoch_start, epoch_end), disable=self.disable_tqdm)
        for epoch in pbar:
            filename = f"epoch_{epoch:04d}.model"
            filepath = self.model_dump_base / filename  # go through all the checkpoints
            if filepath.exists():
                weight_t = torch.load(filepath)['model']
                for key, w_t in weight_t.items():
                    # SWA: w_aver[t] = w_aver[t-1] * n / (n + 1) + w_t / (n + 1)
                    w_aver_t_prev = weight_aver.get(key, 0.)
                    w_aver_t = w_aver_t_prev * num_models_average / (num_models_average + 1.) + w_t / (
                                num_models_average + 1.)
                    weight_aver[key] = w_aver_t
                num_models_average += 1.
                epoch_numbers_averaged.append(epoch)
                pbar.set_description(f"SWA of epoch {epoch:4d}.")
            else:
                pbar.set_description(f"Checkpoint {epoch:4d} not found!")

        # update normalization layers for a few steps
        self.recon_model.load_state_dict(weight_aver)
        self.recon_model.train()
        if update_bn_steps > 0:
            pbar = tqdm.tqdm(self.training_loader, disable=self.disable_tqdm)
            for ind, batch in enumerate(pbar):
                pbar.set_description("SWA updating normalization layers ...")
                self.recon_model(batch)
                if ind >= (update_bn_steps - 1):
                    break

        # save swa weights
        state_dict_swa = dict(
            epoch=-1,
            num_models_average=num_models_average,
            epoch_start=epoch_start,
            update_bn_steps=update_bn_steps,
            epoch_end=epoch_end,
            epoch_numbers_averaged=epoch_numbers_averaged,
            model=self.recon_model.state_dict()
        )
        filename = f"model_swa_{epoch_start:04d}_{epoch_end:04d}_{update_bn_steps:04d}.model"
        torch.save(state_dict_swa,
                   self.model_dump_base / filename)
        return state_dict_swa
