import numpy as np
from data.utils import ifft2
from data.slicedqmridata import SlicedQuantitativeMRIDatasetListSplit, SlicedQuantitativeMRIDataset
from data.paths import CMRxReconDatasetPath
from matplotlib import pyplot as plt

# load dataset
dataset_paths = CMRxReconDatasetPath("../cmrxrecon_dataset.yaml")
sliced_multi_coil_mapping_training = dataset_paths.get_sliced_data_path("MultiCoil", "Mapping", "TrainingSet")
sliced_dataset_files = SlicedQuantitativeMRIDatasetListSplit(sliced_multi_coil_mapping_training,
                                                             acceleration_factors=(4., 8., 10.,),
                                                             modalities=('t1map', ))

# make splits
splits = sliced_dataset_files.split(k=5)
training, validation = splits[0]['training'], splits[0]['validation']

# split 0 training set
training_set = SlicedQuantitativeMRIDataset(*training)
print("Number of training samples: ", len(training_set))

sample = training_set[0]
y = sample['acc_kspace']
S = sample['init_sensitivity']
y_full = sample['gt_kspace']
gt_sos = sample['gt_sos']

# get SENSE init recon
S_H = np.conjugate(S)
x_rec = ifft2(S_H * y, dim=(0, 1), centered=True, normalized=True, complex_input=False)
x_rec = S_H * x_rec
x_rec = x_rec.sum(axis=2)
x_rec = np.abs(x_rec)

# show reconstruction
fig, axes = plt.subplots(2, gt_sos.shape[-1])
for j in range(gt_sos.shape[-1]):
    ax0 = axes[0, j]
    ax1 = axes[1, j]
    ax0.imshow(x_rec[..., j], cmap='gray')
    ax0.axis('off')
    ax0.set_aspect('equal')

    ax1.imshow(gt_sos[..., j], cmap='gray')
    ax1.axis('off')
    ax1.set_aspect('equal')

plt.savefig('temp.png')
plt.show()
plt.close()

