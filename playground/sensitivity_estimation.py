import numpy as np
import tqdm
from direct.algorithms.mri_algorithms import EspiritCalibration
from direct.data import transforms as direct_transform
import yaml
from pathlib import Path
import mat73
import torch
from matplotlib import pyplot as plt
from data.utils import estimate_sensitivity_map


def get_acs_mask(mask: np.ndarray, half_bandwidth: int = 12) -> np.ndarray:
    """
    Get auto-calibration mask from the true Cartesian under-sampling mask along ky.
    :param mask: Under-sampling mask of shape (..., kx, ky). DC component should be already shifted to k-space center.
    :param half_bandwidth: DC ± half_bandwidth is always sampled, this band will be used for calibration.
    :return: ACS mask of shape (..., kx, ky).
    """
    ky = mask.shape[-1]
    dc_ky_ind = ky // 2
    ky_slicer = slice(dc_ky_ind - half_bandwidth, dc_ky_ind + half_bandwidth, 1)
    assert np.all(mask[..., ky_slicer] == 1),\
        "Central lines around ky-DC not fully sampled!"
    acs_mask = np.zeros_like(mask)
    acs_mask[..., ky_slicer] = 1.
    return acs_mask


_multi_coil = "MultiCoil"
_task_type = "Mapping"
_split = "TrainingSet"
_acceleration = "FullSample"
_subject_code = "P001"

dataset_paths = yaml.load(Path("../cmrxrecon_dataset.yaml").open('r'),
                          yaml.Loader)
dataset_base = Path(dataset_paths["dataset_base"])
subject_base = dataset_base / _multi_coil / _task_type / _split / _acceleration / _subject_code


def load_subject(base, filenames=('T1map', 'T1map_mask')):
    base = Path(base)
    data = dict()
    for key in filenames:
        mat_file = mat73.loadmat(base / f"{key}.mat")
        data[key] = list(mat_file.values())[0]
    return data


subject = load_subject(subject_base, filenames=('T1map', ))
y = subject['T1map'][..., 0, :]                         # (kx, ky, nc)
U = np.ones(y.shape[:2])
Uacs = get_acs_mask(U)                                  # (kx, ky)

S = estimate_sensitivity_map(y, U)
S = torch.from_numpy(S)
S = direct_transform.view_as_real(S)
S_H = direct_transform.conjugate(S)
y = direct_transform.view_as_real(torch.from_numpy(y))
x = direct_transform.complex_multiplication(
    S_H,
    direct_transform.ifft2(y, dim=(0, 1),
                           complex_input=True)
)
x = x.sum(dim=-3)

I = x.detach().cpu().numpy()
I = (I ** 2).sum(axis=-1) ** 0.5
fig, ax = plt.subplots(dpi=200)
ax.imshow(I[..., 0], cmap='gray')
ax.set_aspect('equal')
ax.axis('off')
plt.show()
plt.close()
