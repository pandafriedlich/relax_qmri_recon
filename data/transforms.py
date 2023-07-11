import torch
from direct.data import transforms as dtrans
from direct.data import mri_transforms as mtrans
import typing


class ToTensor(object):
    def __init__(self, keys: typing.Union[None, typing.Tuple[str]] = None):
        """
        Initializer.
        :param keys: keys to transform.
        """
        if keys is None:
            self.transform_keys = ('acc_kspace',
                                   'us_mask',
                                   'acs_mask',
                                   'full_kspace')
        else:
            self.transform_keys = keys

    def __call__(self, sample: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        """
        Convert sample to tensor.
        :param sample: Dictionary of loaded numpy objects.
        :return: Dictionary of Tensor objects.
        """
        tensor_sample = dict()
        for k in self.transform_keys:
            val = torch.from_numpy(sample[k])
            if k in ('acc_kspace', 'full_kspace', 'sensitivity'):
                val = torch.permute(val, (3, 2, 0, 1))                  # After permutation: (nt, nc, kx, ky)
            if k.endswith('_mask'):                                      # Masks are of shape (1, 1, kx, ky)
                val = val[None, None,   ...]
            tensor_sample[k] = val
        return tensor_sample


class ViewAsRealTransform:
    def __init__(self, keys: typing.Union[None, typing.Tuple[str]] = None):
        """
        Initializer.
        :param keys: keys to transform.
        """
        if keys is None:
            self.transform_keys = ('acc_kspace',
                                   'full_kspace')
        else:
            self.transform_keys = keys

    def __call__(self, sample: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
        """
        Convert all tensors to plain complex numbers.
        :param sample: data sample.
        :return: converted data sample.
        """
        for key in self.transform_keys:
            sample[key] = dtrans.view_as_real(sample[key])       # (..., kx, ky, 2)
        return sample


class EstimateSensitivityTransform:
    def __init__(self, estimation_type: str = 'rss_estimation',
                 kx_ky_dim: typing.Tuple[int] = (2, 3),
                 coil_dim: int = 1,
                 complex_dim: int = 4):
        """
        Initializer.
        :param estimation_type: Method for sensitivity estimation. Supported values are ('rss_estimation', 'espirit').
        :param kx_ky_dim: Dimensions for kx and ky.
        :param coil_dim: Dimension for coils.
        :param complex_dim: Dimension for the real/imag parts.
        """
        self.estimation_type = estimation_type
        self.kx_ky_dim = kx_ky_dim
        self.coil_dim = coil_dim
        self.complex_dim = complex_dim

    def __call__(self, sample: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
        """
        Estimating initial sensitivity map.
        :param sample: Sample dictionary with keys `masked_kspace` and `us_mask`.
        :return: Sample dictionary with an additional key `sensitivity`.
        """
        if self.estimation_type == 'rss_estimation':
            masked_kspace = sample["acc_kspace"]                    # (nt, nc, kx, ky, 2)
            acs_mask = sample["acs_mask"]                           # (1, 1, kx, ky)
            acs_mask = acs_mask.unsqueeze(self.complex_dim)         # (1, 1, kx, ky, 1)
            acs_kspace = masked_kspace * acs_mask
            acs_image = dtrans.ifft2(
                acs_kspace.float(),
                dim=self.kx_ky_dim,
                centered=True,
                normalized=True,
                complex_input=True
            )                                          # (nt, nc, kx, ky, 2)
            acs_image_rss = dtrans.root_sum_of_squares(acs_image,
                                                      dim=self.coil_dim)    # (nt, kx, ky)
            acs_image_rss = acs_image_rss[:, None, :, :, None]              # (nt, 1, kx, ky, 1)
            sensitivity_map = dtrans.safe_divide(acs_image, acs_image_rss)  # (nt, nc, kx, ky, 2)
            sensitivity_map_norm = torch.sqrt(
                (sensitivity_map ** 2).sum(self.complex_dim).sum(self.coil_dim)
            )                                                               # (nt, kx, ky)
            sensitivity_map_norm = sensitivity_map_norm.unsqueeze(self.coil_dim).unsqueeze(self.complex_dim)    # (nt, 1, kx, ky, 1)
            sensitivity_map = dtrans.safe_divide(sensitivity_map, sensitivity_map_norm)                         # (nt, nc, kx, ky, 2)
            sample['sensitivity'] = sensitivity_map
        else:
            raise ValueError(f"Currently only `rss_estimation` is supported, got {self.estimation_type}!")
        return sample


class NormalizeKSpaceTransform:
    def __init__(self, percentile: float = 0.99,
                 keys: typing.Union[None, typing.Tuple[str]] = None,
                 kx_ky_dim: typing.Tuple[int] = (2, 3),
                 coil_dim: int = 1,
                 complex_dim: int = 4):
        """
        Initializer.
        :param percentile: Percentile as maximal k-space, default is 0.99.
        :param keys: Transform keys.
        :param kx_ky_dim: Dimensions for kx and ky.
        :param coil_dim: Dimension for coils.
        :param complex_dim: Dimension for the real/imag parts.
        """
        assert 0. <= percentile <= 1., f"Invalid percentile {percentile}, should be in [0., 1.]"
        if keys is None:
            self.transform_keys = ('acc_kspace',
                                   'us_mask',
                                   'acs_mask',
                                   'full_kspace')
        else:
            self.transform_keys = keys

        self.percentile = percentile
        self.kx_ky_dim = kx_ky_dim
        self.coil_dim = coil_dim
        self.complex_dim = complex_dim

    def __call__(self, sample: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
        """
        Normalize the k-space data.
        :param sample: Data sample.
        :return: Normalized sample.
        """
        masked_kspace = sample['acc_kspace']                                        # (nt, nc, kx, ky, 2)
        kspace_modulus = dtrans.modulus(masked_kspace,
                                        complex_axis=self.complex_dim)              # (nt, nc, kx, ky)
        kspace_modulus = kspace_modulus.flatten(*self.kx_ky_dim)
        amax = torch.quantile(kspace_modulus, self.percentile, dim=-1)              # (nt, nc)
        amax = amax[..., None, None, None]                                                   # (nt, nc, kx, ky, 2)
        sample['scaling_factor'] = amax
        for key in self.transform_keys:
            sample[key] = dtrans.safe_divide(sample[key], amax)
        return sample



