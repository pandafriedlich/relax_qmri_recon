import torch
from torch.utils.data import DataLoader
import numpy as np
from direct.data.transforms import (complex_multiplication,
                                    modulus,
                                    conjugate,
                                    ifft2,
                                    root_sum_of_squares)
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
for sample in training_loader:
    # Get a sample
    y = sample['acc_kspace'][0]
    S = sample['sensitivity'][0]
    y_full = sample['full_kspace'][0]

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
