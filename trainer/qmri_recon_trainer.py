import torch
import tqdm
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
from data.transforms import (ToTensor,
                             ViewAsRealTransform,
                             EstimateSensitivityTransform,
                             NormalizeKSpaceTransform
                             )
from torchvision.transforms import Compose
from configuration.config import (TrainerConfig,
                                  ReconstructionBackboneConfig,
                                  SensitivityRefinementModuleConfig)
from models.recurrentvarnet import RecurrentVarNet
from models.tricathlon import QuantitativeMRIReconstructionNet
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class QuantitativeMRITrainer(object):
    def __init__(self):
        """
        TODO: make trainer configurable
        """
        # Configurable variables
        self.split = 0
        self.acceleration_factors = (4., 8., 10.)
        self.modality = ('t1map', )
        self.baseline_images = dict(t1map=9, t2map=3)[self.modality]
        self.batch_size = 2
        self.loader_num_workers = 8
        self.weight_decay = 0.
        self.lr_decay = 0.9
        self.lr_init = 1e-3
        self.max_epochs = 1000
        self.combined_loss_weight = dict(nmse=1., ssim=1.)
        self.run_name = 'dryrun'
        self.epoch = 0
        self.save_every = 10
        self.global_step = 0

        # 1. load data set, we load the related paths from YAML, and then create the split dataset
        path_yaml_file = "../cmrxrecon_dataset.yaml"
        dataset_path = CMRxReconDatasetPath(path_yaml_file)
        coil_type = "MultiCoil"
        training_set_base = dataset_path.get_sliced_data_path(coil_type,
                                                              "Mapping",
                                                              "TrainingSet")
        sliced_dataset_files = SlicedQuantitativeMRIDatasetListSplit(training_set_base,
                                                                     acceleration_factors=self.acceleration_factors,
                                                                     modalities=self.modality)
        # Now we have the list of file paths.
        file_lists_dict = sliced_dataset_files.splits[self.split]
        self.training_file_lists = file_lists_dict['training']
        self.validation_file_lists = file_lists_dict['validation']

        # set up transforms to be performed on the raw data, create Dataset & DataLoader.
        transforms = Compose([ToTensor(keys=('acc_kspace', 'us_mask', 'acs_mask', 'full_kspace')),
                              ViewAsRealTransform(keys=('acc_kspace', 'us_mask', 'acs_mask', 'full_kspace')),
                              EstimateSensitivityTransform(),
                              NormalizeKSpaceTransform(keys=('acc_kspace', 'full_kspace'))
                              ])
        self.training_set = SlicedQuantitativeMRIDataset(*self.training_file_lists,
                                                         transforms=transforms)
        self.validation_set = SlicedQuantitativeMRIDataset(*self.validation_file_lists,
                                                           transforms=transforms)
        self.training_loader = DataLoader(self.training_set,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=self.loader_num_workers,
                                          pin_memory=True,
                                          collate_fn=qmri_data_collate_fn)
        self.validation_loader = DataLoader(self.validation_set,
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            num_workers=self.loader_num_workers,
                                            pin_memory=True,
                                            collate_fn=qmri_data_collate_fn
                                            )

        # 2. set-up models
        model = RecurrentVarNet(
            forward_operator=fft2,
            backward_operator=ifft2,
            in_channels=2 * self.baseline_images,
            num_steps=4,
            recurrent_num_layers=3,
            recurrent_hidden_channels=96,
            initializer_initialization='sense',
            learned_initializer=True,
            initializer_channels=[32, 32, 64, 64],
            initializer_dilations=[1, 1, 2, 4],
            initializer_multiscale=3
        ).cuda().float()

        additional_model = UnetModel2d(
            in_channels=2 * self.baseline_images,
            out_channels=2 * self.baseline_images,
            num_filters=8,
            num_pool_layers=4,
            dropout_probability=0.0
        ).cuda().float()
        self.recon_model = QuantitativeMRIReconstructionNet(model,
                                                            additional_model)

        # set up optimizers and lr_schedulers
        self.optimizer = torch.optim.Adam(self.recon_model.parameters(),
                                          lr=self.lr_init)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lr_lambda=lambda e: (
                                                                                           1 - e / self.max_epochs) ** self.lr_decay)

        # 3. set up loss functions
        self.loss_functions = dict()
        for loss_name in self.combined_loss_weight.keys():
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
        self.expr_dump_base = dataset_path.get_dump_data_path(self.run_name, f"fold_{self.split}")
        self.model_dump_base = self.expr_dump_base / "models"
        self.validation_save_base = self.expr_dump_base / "validation"
        self.model_log_base = self.expr_dump_base / "logs"
        # self.expr_dump_base.mkdir(parents=True, exist_ok=True)
        self.model_dump_base.mkdir(parents=True, exist_ok=True)
        self.validation_save_base.mkdir(parents=True, exist_ok=True)
        self.model_log_base.mkdir(parents=True, exist_ok=True)

        # 5. tensorboard/weight&bias, whatever...
        self.training_tracker = SummaryWriter(self.model_log_base)

    def train(self, restore_latest: bool = True) -> None:
        """
        The training routine, loading data, forward pass, loss computation and backward.
        :param restore_latest: If the latest checkpoint should be restored.
        :return:
        """
        self.recon_model.cuda().float()
        self.recon_model.train()
        n_steps_per_epoch = len(self.training_loader)
        self.global_step = self.epoch * n_steps_per_epoch

        for epoch in range(self.epoch, self.max_epochs):
            pbar = tqdm.tqdm(self.training_loader)
            for batch in pbar:
                self.optimizer.zero_grad()
                # forward pass
                prediction = self.recon_model(batch)
                y_pred = prediction["pred_kspace"]
                y_full = batch['full_kspace'].cuda().float()

                # convert k-space to RSS image
                x_gt = mutils.root_sum_of_square_recon(
                    y_full,
                    backward_operator=ifft2,
                    spatial_dim=(2, 3),
                    coil_dim=1
                )
                x_pred = mutils.root_sum_of_square_recon(
                    y_pred,
                    backward_operator=ifft2,
                    spatial_dim=(2, 3),
                    coil_dim=1
                )
                x_gt_flattened = x_gt.permute(0, 3, 1, 2).flatten(0, 1).unsqueeze(1)
                x_pred_flattened = x_pred.permute(0, 3, 1, 2).flatten(0, 1).unsqueeze(1)

                # compute loss
                loss_value_dict = dict()
                total_loss = 0.
                for loss_name, loss_fn in self.loss_functions.items():
                    if loss_name.lower() == 'ssim':
                        loss_val = loss_fn(x_pred_flattened,
                                           x_gt_flattened,
                                           torch.amax(x_gt_flattened, dim=(1, 2, 3)),
                                           reduced=True)
                    elif loss_name.lower() == 'nuc':
                        loss_val = loss_fn(x_pred)
                    else:
                        loss_val = loss_fn(x_pred, x_gt)
                    total_loss += self.combined_loss_weight[loss_name] * loss_val
                    loss_value_dict[loss_name] = loss_val.item()
                loss_value_dict['total'] = total_loss.item()

                # backward pass
                total_loss.backward()
                self.optimizer.step()

                # record the losses
                description_str = ''
                self.global_step += 1
                for k, l in loss_value_dict.items():
                    description_str += f'{k:s}: {l:.4f} '
                    if self.global_step % 10 == 0:
                        self.training_tracker.add_scalar(
                            f'loss/{k}',
                            l,
                            self.global_step
                        )
                pbar.set_description(description_str)

                if self.global_step % 10 == 0:
                    grid_gt = make_grid(x_gt_flattened, nrow=9,
                                        normalize=True, scale_each=True)
                    grid_pred = make_grid(x_pred_flattened, nrow=9,
                                        normalize=True, scale_each=True)
                    self.training_tracker.add_image("training/pred", grid_pred,
                                                    global_step=self.global_step)
                    self.training_tracker.add_image("training/gt", grid_gt,
                                                    global_step=self.global_step)

            self.scheduler.step(self.epoch)
            self.save_latest_checkpoint()
            self.save_per_epoch()
            # self.validation()
        self.training_tracker.flush()
        self.training_tracker.close()

    def save_checkpoint(self, filename: str, save_optimizer: bool = False) -> None:
        """
        Save checkpoint to filename.model.
        :param filename: The filename of the checkpoint file.
        :param save_optimizer: If the optimizer related status dictionary should be saved.
        """
        save_dict = dict(epoch=self.epoch, model=self.recon_model.state_dict())
        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['scheduler'] = self.scheduler.state_dict()
        torch.save(save_dict, self.model_dump_base / f"{filename}.model")

    def save_per_epoch(self):
        if self.epoch % self.save_every == 0:
            filename = f"epoch_{self.epoch:04d}"
            self.save_checkpoint(filename, save_optimizer=False)

    def save_latest_checkpoint(self):
        self.save_checkpoint("model_latest", save_optimizer=True)

    def validation(self):
        self.recon_model.eval()
        batch = next(iter(self.validation_set))
        with torch.no_grad():
            prediction = self.recon_model(batch)
            y_pred = prediction["pred_kspace"]
            y_full = batch['full_kspace'].cuda().float()

            # convert k-space to RSS image
            x_gt = mutils.root_sum_of_square_recon(
                y_full,
                backward_operator=ifft2,
                spatial_dim=(2, 3),
                coil_dim=1
            )
            x_pred = mutils.root_sum_of_square_recon(
                y_pred,
                backward_operator=ifft2,
                spatial_dim=(2, 3),
                coil_dim=1
            )

            x_gt_flattened = x_gt.permute(0, 3, 1, 2).flatten(0, 1).unsqueeze(1)
            x_pred_flattened = x_pred.permute(0, 3, 1, 2).flatten(0, 1).unsqueeze(1)

            # compute loss
            loss_value_dict = dict()
            total_loss = 0.
            for loss_name, loss_fn in self.loss_functions.items():
                if loss_name.lower() == 'ssim':
                    loss_val = loss_fn(x_pred_flattened,
                                       x_gt_flattened,
                                       torch.amax(x_gt_flattened, dim=(1, 2, 3)),
                                       reduced=True)
                elif loss_name.lower() == 'nuc':
                    loss_val = loss_fn(x_pred)
                else:
                    loss_val = loss_fn(x_pred, x_gt)
                total_loss += self.combined_loss_weight[loss_name] * loss_val
                loss_value_dict[loss_name] = loss_val.item()
            loss_value_dict['total'] = total_loss.item()

            # make image grid
            grid_gt = make_grid(x_gt_flattened, nrow=9,
                                normalize=True, scale_each=True)
            grid_pred = make_grid(x_pred_flattened, nrow=9,
                                  normalize=True, scale_each=True)
            self.training_tracker.add_image("validation/pred", grid_pred,
                                            global_step=self.global_step)
            self.training_tracker.add_image("validation/gt", grid_gt,
                                            global_step=self.global_step)

            for k, l in loss_value_dict.items():
                self.training_tracker.add_scalar(
                    f'validation_loss/{k}',
                    l,
                    self.global_step
                )
