import torch
from torch.utils.data import DataLoader
import numpy as np
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
from matplotlib import pyplot as plt
from data.transforms import (ToTensor,
                             ViewAsRealTransform,
                             EstimateSensitivityTransform,
                             NormalizeKSpaceTransform
                             )
from torchvision.transforms import Compose
from models.recurrentvarnet import RecurrentVarNet

# load dataset
dataset_paths = CMRxReconDatasetPath("../cmrxrecon_dataset.yaml")
sliced_multi_coil_mapping_training = dataset_paths.get_sliced_data_path("MultiCoil", "Mapping", "TrainingSet")
sliced_dataset_files = SlicedQuantitativeMRIDatasetListSplit(sliced_multi_coil_mapping_training,
                                                             acceleration_factors=(4., 8., 10.,),
                                                             modalities=('t1map',))

# make splits
splits = sliced_dataset_files.split(k=5)
training, validation = splits[0]['training'], splits[0]['validation']

# transforms
transforms = Compose([ToTensor(keys=('acc_kspace', 'us_mask', 'acs_mask', 'full_kspace')),
                      ViewAsRealTransform(keys=('acc_kspace', 'us_mask', 'acs_mask', 'full_kspace')),
                      EstimateSensitivityTransform(),
                      NormalizeKSpaceTransform(keys=('acc_kspace', 'full_kspace'))
                      ])

# split 0 training set
training_set = SlicedQuantitativeMRIDataset(*training, transforms=transforms)
print("Number of training samples: ", len(training_set))

training_loader = DataLoader(training_set, batch_size=2,
                             shuffle=True, num_workers=1,
                             collate_fn=qmri_data_collate_fn)

# define a model
model = RecurrentVarNet(
    forward_operator=fft2,
    backward_operator=ifft2,
    in_channels=2 * 9,         # 2 for complex numbers, 9 for #baseline images
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
    in_channels=2 * 9,
    out_channels=2 * 9,
    num_filters=8,
    num_pool_layers=4,
    dropout_probability=0.0
).cuda().float()

l2_loss = direct_func.nmse.NMSELoss()
ssim_loss = SSIMLoss()
nuc_loss = NuclearNormLoss(relax_dim=3, spatial_dim=(1, 2))

for sample in training_loader:
    # Get a sample
    y = sample['acc_kspace'].cuda().float()
    U = sample['us_mask'].cuda().float()
    S = sample['sensitivity'].cuda().float()
    y_full = sample['full_kspace'].cuda().float()

    Sref = mutils.refine_sensitivity_map(additional_model,
                                         S, coil_dim=1, spatial_dim=(2, 3),
                                         relax_dim=4, complex_dim=5)
    y_pred = model(y, U, Sref)
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
    nuc = nuc_loss(x_pred)
    l2 = l2_loss(x_gt, x_pred)
    x_gt_flattened = x_gt.permute(0, 3, 1, 2).flatten(0, 1).unsqueeze(1)
    x_pred_flattened = x_pred.permute(0, 3, 1, 2).flatten(0, 1).unsqueeze(1)
    ssim = ssim_loss(x_pred_flattened,
                     x_gt_flattened,
                     torch.amax(x_gt_flattened, dim=(1, 2, 3)),
                     reduced=True)

    # visualize SENSE initialization
    y = y[0]
    S = S[0]
    y_full = y_full[0]

    # get SENSE init recon
    S_conj = conjugate(S)  # (nc, kx, ky, kt, 2)
    x_rec = ifft2(y, dim=(1, 2),  # (nc, kx, ky, kt, 2)
                  centered=True,
                  normalized=True, complex_input=True)
    x_rec = complex_multiplication(S_conj, x_rec)   # (nc, kx, ky, kt, 2)
    x_rec = x_rec.sum(axis=0)                       # (kx, ky, kt, 2)
    x_rec = modulus(x_rec)                          # (kx, ky, kt)

    x_gt = ifft2(y_full, dim=(1, 2),                # (nc, kx, ky, kt, 2)
                 centered=True,
                 normalized=True, complex_input=True)
    x_gt = root_sum_of_squares(x_gt, dim=0, complex_dim=-1)

    # show reconstruction
    fig, axes = plt.subplots(2, x_rec.shape[-1])
    for j in range(x_rec.shape[-1]):
        ax0 = axes[0, j]
        ax1 = axes[1, j]

        ax0.imshow(x_rec[..., j].detach().cpu().numpy(), cmap='gray')
        ax0.axis('off')
        ax0.set_aspect('equal')

        ax1.imshow(x_gt[..., j].detach().cpu().numpy(), cmap='gray')
        ax1.axis('off')
        ax1.set_aspect('equal')

    plt.savefig('temp.png')
    plt.show()
    plt.close()
    break
