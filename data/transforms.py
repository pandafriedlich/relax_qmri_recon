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
                val = torch.permute(val, (2, 0, 1, 3))                      # After permutation: (nc, kx, ky, nt)
            if k.endswith('_mask'):                                         # Masks are of shape (1, kx, ky, 1)
                val = val.unsqueeze(0).unsqueeze(-1)
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
                                   'us_mask',
                                   'acs_mask',
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
            if key.endswith('_mask'):
                sample[key] = sample[key].unsqueeze(-1)
            else:
                sample[key] = dtrans.view_as_real(sample[key])       # (..., kx, ky, 2)
        return sample


class EstimateSensitivityTransform:
    def __init__(self, estimation_type: str = 'rss_estimation',
                 kx_ky_dim: typing.Tuple[int] = (1, 2),
                 coil_dim: int = 0,
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
            masked_kspace = sample["acc_kspace"]                    # (nc, kx, ky, nt, 2)
            acs_mask = sample["acs_mask"]                           # (1, kx, ky, 1, 1)
            acs_kspace = masked_kspace * acs_mask
            acs_image = dtrans.ifft2(
                acs_kspace.float(),
                dim=self.kx_ky_dim,
                centered=True,
                normalized=True,
                complex_input=True
            )                                                               # (nc, kx, ky, nt, 2)
            acs_image_rss = dtrans.root_sum_of_squares(acs_image,
                                                      dim=self.coil_dim)    # (kx, ky, nt)
            acs_image_rss = (acs_image_rss
                             .unsqueeze(self.coil_dim)
                             .unsqueeze(self.complex_dim))                  # (1, kx, ky, nt, 1)
            sensitivity_map = dtrans.safe_divide(acs_image, acs_image_rss)  # (nc, kx, ky, nt, 2)
            sensitivity_map_norm = torch.sqrt(
                (sensitivity_map ** 2).sum(self.complex_dim).sum(self.coil_dim)
            )                                                               # (kx, ky, nt)
            sensitivity_map_norm = sensitivity_map_norm.unsqueeze(self.coil_dim).unsqueeze(self.complex_dim)    # (1, kx, ky, nt, 1)
            sensitivity_map = dtrans.safe_divide(sensitivity_map, sensitivity_map_norm)                         # (nc, kx, ky, nt, 2)
            sample['sensitivity'] = sensitivity_map
        else:
            raise ValueError(f"Currently only `rss_estimation` is supported, got {self.estimation_type}!")
        return sample


class NormalizeKSpaceTransform:
    def __init__(self, percentile: float = 0.99,
                 keys: typing.Union[None, typing.Tuple[str]] = None,
                 kx_ky_dim: typing.Tuple[int] = (1, 2),
                 coil_dim: int = 0,
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
        masked_kspace = sample['acc_kspace']                                        # (nc, kx, ky, nt, 2)
        kspace_modulus = dtrans.modulus(masked_kspace,
                                        complex_axis=self.complex_dim)              # (nc, kx, ky, nt)
        kspace_modulus = kspace_modulus.flatten(*self.kx_ky_dim)                    # (nc, kx * ky, nt)
        amax = torch.quantile(kspace_modulus, self.percentile, dim=1)               # (nc, nt)
        amax = amax[:, None, None, :, None]                                         # (nc, 1, 1, nt, 1)
        sample['scaling_factor'] = amax
        for key in self.transform_keys:
            sample[key] = dtrans.safe_divide(sample[key], amax)
        return sample


