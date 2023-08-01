import typing
import torch
import numpy as np
from direct.data import transforms as dtrans
from configuration.config import ImageDomainAugmentationConfig
import torchvision.transforms as vt


class GenerateNewMaskTransform:
    """
    Generate a new under-sampling mask and the corresponding k-space measurements with the same acceleration factor as the original mask.
    """
    def __init__(self, full_kspace_key: str = 'full_kspace',
                 acc_kspace_key: str = 'acc_kspace',
                 mask_key: str = 'us_mask',
                 acs_half_bandwidth: int = 12,
                 spatial_dim: typing.Tuple[int, int] = (1, 2)):
        self.full_kspace_key = full_kspace_key
        self.acc_kspace_key = acc_kspace_key
        self.mask_key = mask_key
        self.acs_half_bandwidth = acs_half_bandwidth
        self.spatial_dim = spatial_dim

    def __call__(self, sample: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
        """
        Generate a new random Cartesian under-sampling mask with the same acceleration factor as before.
        :param sample: A dictionary with fully sampled k-space and the original under-sampling mask.
        :return: Augmented sample, note: under-sampling mask and accelerated k-space will change.
        """
        full_kspace = sample[self.full_kspace_key]
        raw_mask = sample[self.mask_key]  # The original k-space under-sampling mask
        kx_dim, ky_dim = self.spatial_dim
        num_ky_lines = int(
            raw_mask.sum(dim=ky_dim).mean(dim=kx_dim).squeeze().item())  # Cartesian under-sampling budget
        kx, ky = full_kspace.shape[kx_dim], full_kspace.shape[ky_dim]
        new_mask = torch.zeros((kx, ky))

        # ACS region
        dc_ky_ind = ky // 2
        ky_slicer = slice(dc_ky_ind - self.acs_half_bandwidth, dc_ky_ind + self.acs_half_bandwidth, 1)
        new_mask[:, ky_slicer] = 1.

        # Non-ACS region
        num_ky_lines_budget = num_ky_lines - 2 * self.acs_half_bandwidth
        non_acs_ky_locations = list(range(dc_ky_ind - self.acs_half_bandwidth)) \
                               + list(range(dc_ky_ind + self.acs_half_bandwidth, ky))
        chosen_ky_locations = np.random.choice(non_acs_ky_locations, size=num_ky_lines_budget, replace=False)
        new_mask[:, chosen_ky_locations] = 1.
        new_num_ky_lines = int(new_mask[0, :].sum().item())
        assert new_num_ky_lines == num_ky_lines, f"Acceleration factor mismatch, before: {num_ky_lines}, after: {new_num_ky_lines}!"

        # new us_mask
        new_mask = new_mask.view(*raw_mask.shape)               # expand dimensions
        new_acc_kspace = full_kspace * new_mask                 # simulate k-space measurement
        sample[self.acc_kspace_key] = new_acc_kspace
        sample[self.mask_key] = new_mask

        return sample


class ImageDomainAugmentation:
    """
    Perform image domain augmentation, including random v-flip, h-flip, rotation ...
    """
    def __init__(self, image_space_config: ImageDomainAugmentationConfig,
                 forward_operator: typing.Callable = dtrans.fft2,
                 backward_operator: typing.Callable = dtrans.ifft2,
                 full_kspace_key: str = 'full_kspace',
                 acc_kspace_key: str = 'acc_kspace',
                 mask_key: str = 'us_mask',
                 image_key: str = 'gt_image'):
        self.forward_operator = forward_operator
        self.backward_operator = backward_operator
        self.full_kspace_key = full_kspace_key
        self.acc_kspace_key = acc_kspace_key
        self.mask_key = mask_key
        self.image_key = image_key

        # augmentations in the image domain
        self.aug_config = image_space_config
        self.affine = vt.RandomAffine(degrees=self.aug_config.rotation,
                                      translate=(self.aug_config.translation, self.aug_config.translation),
                                      scale=(self.aug_config.scale_min, self.aug_config.scale_max),
                                      shear=self.aug_config.shearing,
                                      interpolation=vt.InterpolationMode.BILINEAR,
                                      fill=0)
        self.vflip = vt.RandomVerticalFlip(p=self.aug_config.p_flip)
        self.hflip = vt.RandomHorizontalFlip(p=self.aug_config.p_flip)
        self.new_mask = GenerateNewMaskTransform(full_kspace_key=self.full_kspace_key,
                                                 acc_kspace_key=self.acc_kspace_key,
                                                 mask_key=self.mask_key,
                                                 spatial_dim=(1, 2))
        self.p_contamination = self.aug_config.p_contamination
        self.contamination_max_rel = self.aug_config.contamination_max_rel

    def __call__(self, sample: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
        """
        Performs image domain augmentation. After that the fully sampled k-space will be updated via fft, then the original us mask will be applied on it to simulate the accelerated k-space.
        :param sample: A dictionary with the image reconstructed from the fully sampled k-space.
        :return: The same sample but images augmented in k-space.
        """
        # print(self.aug_config.p_affine,
        #       self.contamination_max_rel)
        gt_image = sample[self.image_key]                       # (kc, kx, ky, kt, 2)
        gt_image = torch.permute(gt_image, (0, 3, 4, 1, 2))     # (kc, kt, 2, kx, ky)
        raw_shape = gt_image.shape

        # Augment
        gt_image = self.vflip(gt_image)
        gt_image = self.hflip(gt_image)
        gt_image = gt_image.flatten(0, 1)
        if np.random.random() < self.aug_config.p_affine:
            gt_image = self.affine(gt_image)
        gt_image = gt_image.unflatten(0, raw_shape[:2])
        gt_image = torch.permute(gt_image, (0, 3, 4, 1, 2)).contiguous()

        # Simulation additive noise
        full_kspace = self.forward_operator(gt_image)
        if np.random.random() < self.aug_config.p_contamination:
            kspace_energy = (full_kspace ** 2).mean() ** 0.5
            noise = torch.randn(full_kspace.shape) * (np.random.random() * self.contamination_max_rel * kspace_energy)
            full_kspace += noise

        # Simulation under-sampling
        us_mask = sample[self.mask_key]
        acc_kspace = us_mask * full_kspace
        sample[self.full_kspace_key] = full_kspace
        sample[self.acc_kspace_key] = acc_kspace
        sample[self.image_key] = gt_image

        # Under-sampling Augmentation
        if np.random.random() < self.aug_config.p_new_mask:
            sample = self.new_mask(sample)
        return sample
