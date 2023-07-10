import numpy as np
from typing import Tuple
from direct.algorithms.mri_algorithms import EspiritCalibration
from direct.data import transforms as direct_transform
import torch


def ifft2(
    data: np.ndarray,
    dim: Tuple[int, ...] = (1, 2),
    centered: bool = True,
    normalized: bool = True,
    complex_input: bool = True,
) -> np.ndarray:
    """
    Perform centered inverse Fourier Transform (FT).
    :param data: Complex-valued k-space data.
    :param dim: The two dimensions along with inverse FT will be performed.
    :param centered: if the k-space is already centered.
    :param normalized: if the k-space should be normalized.
    :param complex_input: True for (..., 2) np.float inputs, False for (...) np.complex inputs.
    :return: inverse Fourier Transform results.
    """
    if complex_input:
        data = data[..., 0] + 1j * data[..., 1]
    if centered:
        data = np.fft.ifftshift(data, axes=dim)

    data = np.fft.ifftn(
        data,
        axes=dim,
        norm="ortho" if normalized else None,
    )

    if centered:
        data = np.fft.fftshift(data, axes=dim)
    if complex_input:
        data = np.stack((np.real(data), np.imag(data)), axis=-1)
    return data


def SoS_Reconstruction(data: np.ndarray,
                       fft_dim: Tuple[int, ...],
                       coil_dim: int,
                       complex_input: bool = False,
                       centered: bool = True,
                       normalized: bool = True) -> np.ndarray:
    """
    SoS reconstruction for fully-sampled k-space.
    :param data: k-space data.
    :param fft_dim: dimensions for kx and ky.
    :param coil_dim: dimension for coils.
    :param centered: if k-space is centered.
    :param complex_input: if k-space has separate channels for real and imaginary parts.
    :param normalized: if k-space data are already normalized.
    :return: SOS reconstruction of the magnitude image.
    """
    data = ifft2(data, fft_dim, centered=centered,
                 complex_input=complex_input, normalized=normalized)
    sum_of_square = np.abs(data * data).sum(coil_dim)
    root_sum_of_square = sum_of_square ** 0.5
    return root_sum_of_square


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


def estimate_sensitivity_map(k_space: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Estimation of sensitivity map.
    :param k_space: (kx, ky, nc, nb)
    :param mask: (kx, ky)
    :return: Sensitivity map of shape (kx, ky, nc, nb) complex.
    """
    k_space = np.transpose(k_space, (3, 2, 0, 1))
    Uacs = get_acs_mask(mask, half_bandwidth=12)
    k_space = torch.from_numpy(k_space).cuda()
    Uacs = torch.from_numpy(Uacs).cuda()

    def backward_operator(*args, **kwargs):
        kwargs['normalized'] = True
        return direct_transform.ifft2(*args, **kwargs)

    sensitivity = []
    for b in range(k_space.shape[0]):
        sensitivity_estimator = EspiritCalibration(
            threshold=0.02,
            max_iter=30,
            crop=0.9,
            backward_operator=backward_operator
        )
        y = direct_transform.view_as_real(k_space[b, ...])
        S = sensitivity_estimator.calculate_sensitivity_map(Uacs, y)
        S = direct_transform.view_as_complex(S)
        sensitivity.append(S.detach().cpu().numpy())
    sensitivity = np.array(sensitivity)
    sensitivity = np.transpose(sensitivity, (2, 3, 1, 0))
    return sensitivity

