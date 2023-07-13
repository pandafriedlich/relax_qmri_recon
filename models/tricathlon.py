import torch
import typing
from .utils import refine_sensitivity_map


class QuantitativeMRIReconstructionNet(torch.nn.Module):
    def __init__(self, recon_backbone: torch.nn.Module,
                 sensitivity_net: typing.Optional[torch.nn.Module] = None):
        """
        Initialize the qMRI reconstruction net from backbone and SER.
        :param recon_backbone: Reconstruction backbone network.
        :param sensitivity_net: Sensitivity refinement network.
        """
        super().__init__()
        self.recon_net = recon_backbone
        self.sensitivity_net = sensitivity_net

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
        return dict(sensitivity=sensitivity,
                    pred_kspace=pred_kspace)

