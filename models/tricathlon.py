import torch
import typing

import direct.nn.unet.unet_2d
from .utils import refine_sensitivity_map, root_sum_of_square_recon
import direct.data.transforms as dtrans


class QuantitativeMRIReconstructionNet(torch.nn.Module):
    def __init__(self, recon_backbone: torch.nn.Module,
                 sensitivity_net: typing.Optional[torch.nn.Module] = None,
                 mapping_net: typing.Optional[torch.nn.Module] = None):
        """
        Initialize the qMRI reconstruction net from backbone and SER.
        :param recon_backbone: Reconstruction backbone network.
        :param sensitivity_net: Sensitivity refinement network.
        :param mapping_net: Mapping network
        """
        super().__init__()
        self.recon_net = recon_backbone
        self.sensitivity_net = sensitivity_net
        self.mapping_net = mapping_net

    def forward(self,
                data: typing.Dict[str, typing.Any]) -> typing.Dict[str, torch.Tensor]:
        """
        Estimate the full k-space.
        :param data: input data as a dictionary with keys 'acc_kspace' and 'us_mask'.
        :return: estimated sensitivity map and full k-space as a dict.
        """
        masked_kspace = data['acc_kspace'].cuda().float()
        us_mask = data['us_mask'].cuda().float()
        sensitivity = data['sensitivity'].cuda().float()
        if self.sensitivity_net is not None:
            sensitivity = refine_sensitivity_map(self.sensitivity_net,
                                                 sensitivity,
                                                 coil_dim=1,
                                                 spatial_dim=(2, 3),
                                                 relax_dim=4,
                                                 complex_dim=5)
        pred_kspace = self.recon_net(masked_kspace,
                                     us_mask,
                                     sensitivity)
        if self.mapping_net is not None:
            rss = root_sum_of_square_recon(pred_kspace.float(),
                                           backward_operator=dtrans.ifft2,
                                           spatial_dim=(2, 3),
                                           coil_dim=1,
                                           complex_dim=-1)  # (nb, kx, ky, [nt])
            rss = torch.permute(rss, (0, 3, 1, 2))
            relaxation = self.mapping_net(rss)
        else:
            relaxation = None

        return dict(sensitivity=sensitivity,
                    pred_kspace=pred_kspace,
                    relaxation=relaxation)


class MOLLIMappingNet(direct.nn.unet.unet_2d.UnetModel2d):
    """
    Predict (A, k, T1_app) in the signal model s(t) = A * (1 - k * exp(-t/T1_app)).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activ_A = torch.nn.Softplus()
        self.activ_k = torch.nn.Softplus()
        self.activ_T1 = torch.nn.Softplus()

    @staticmethod
    def signal_model(t, pmap):
        s = (1 - pmap[:, [1], ...] * torch.exp(-t / pmap[:, [2], ...])) * pmap[:, [0], ...]
        return s

    @staticmethod
    def get_air_mask(image, percentage=0.05):
        with torch.no_grad():
            # maximal signal intensity during relaxation of each pixel
            recovered_signal = torch.amax(image, dim=1, keepdim=True)
            # maximal signal intensity among all pixels
            signal_max = torch.amax(recovered_signal, dim=(2, 3), keepdim=True)
            air = recovered_signal < signal_max * percentage
        return air.float()

    @staticmethod
    def get_t_pred(pmap):
        return (pmap[:, [1], ...] - 1.) * pmap[:, [2], ...]

    def forward(self, x):
        out = super().forward(x)
        A = self.activ_A(out[:, [0], :, :])
        k = self.activ_k(out[:, [1], :, :]) + 1.
        T1 = self.activ_T1(out[:, [2], :, :])
        return torch.cat((A, k, T1), dim=1)


class T2RelaxMappingNet(direct.nn.unet.unet_2d.UnetModel2d):
    """
    Predict (A, T2) in the signal model s(t) = A * exp(-t/T2).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activ_A = torch.nn.Softplus()
        self.activ_T2 = torch.nn.Softplus()

    @staticmethod
    def signal_model(t, pmap):
        s = torch.exp(- t / pmap[:, [1], ...]) * pmap[:, [0], ...]
        return s

    @staticmethod
    def get_air_mask(image, percentage=0.05):
        with torch.no_grad():
            # maximal signal intensity during relaxation of each pixel
            recovered_signal = torch.amax(image, dim=1, keepdim=True)
            # maximal signal intensity among all pixels
            signal_max = torch.amax(recovered_signal, dim=(2, 3), keepdim=True)
            air = recovered_signal < signal_max * percentage
        return air.float()

    @staticmethod
    def get_t_pred(pmap):
        return pmap[:, [1], ...]

    def forward(self, x):
        out = super().forward(x)
        A = self.activ_A(out[:, [0], :, :])
        T2 = self.activ_T2(out[:, [1], :, :])
        return torch.cat((A, T2), dim=1)
