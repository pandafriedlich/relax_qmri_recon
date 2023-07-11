import numpy as np
from direct.data.transforms import (complex_multiplication,
                                    modulus,
                                    conjugate,
                                    ifft2,
                                    root_sum_of_squares)
from data.slicedqmridata import SlicedQuantitativeMRIDatasetListSplit, SlicedQuantitativeMRIDataset
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
                      ViewAsRealTransform(keys=('acc_kspace', 'full_kspace')),
                      EstimateSensitivityTransform(),
                      NormalizeKSpaceTransform(keys=('acc_kspace', 'full_kspace'))
                      ])

# split 0 training set
training_set = SlicedQuantitativeMRIDataset(*training, transforms=transforms)
print("Number of training samples: ", len(training_set))

# Get a sample
sample = training_set[0]
y = sample['acc_kspace']
S = sample['sensitivity']
y_full = sample['full_kspace']

# get SENSE init recon
S_conj = conjugate(S)  # (kt, nc, kx, ky, 2)
x_rec = ifft2(y, dim=(2, 3),  # (kt, nc, kx, ky, 2)
              centered=True,
              normalized=True, complex_input=True)
x_rec = complex_multiplication(S_conj, x_rec)  # (kt, nc, kx, ky, 2)
x_rec = x_rec.sum(axis=1)  # (kt, kx, ky, 2)
x_rec = modulus(x_rec)  # (kt, kx, ky)

x_gt = ifft2(y_full, dim=(2, 3),  # (kt, nc, kx, ky, 2)
             centered=True,
             normalized=True, complex_input=True)
x_gt = root_sum_of_squares(x_gt, dim=1, complex_dim=-1)

# show reconstruction
fig, axes = plt.subplots(2, x_rec.shape[0])
for j in range(x_rec.shape[0]):
    ax0 = axes[0, j]
    ax1 = axes[1, j]

    ax0.imshow(x_rec[j, ...].detach().cpu().numpy(), cmap='gray')
    ax0.axis('off')
    ax0.set_aspect('equal')

    ax1.imshow(x_gt[j, ...].detach().cpu().numpy(), cmap='gray')
    ax1.axis('off')
    ax1.set_aspect('equal')

plt.savefig('temp.png')
plt.show()
plt.close()
