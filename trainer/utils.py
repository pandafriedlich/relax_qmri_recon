from matplotlib import pyplot as plt
import torch
import numpy as np
from torchvision.utils import make_grid


def make_heatmap_grid(image, air=None, vmin=0., vmax=2., cmap='jet', *args, **kwargs):
    assert image.shape[1] == 1, "Image has more than one channels!"
    b, _, h, w = image.shape
    if air is not None:
        image = image * (1. - air.to(dtype=image.dtype, device=image.device))
    new_image = torch.zeros((b, 3, h, w))
    for ind in range(b):
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colormap = plt.get_cmap(cmap)
        quantity = image[ind, 0, ...].detach().cpu().numpy()
        rgb = colormap(norm(quantity))[:, :, :3]
        # rgb = (rgb * 255).astype(np.uint8)
        new_image[ind, :, ...] = torch.from_numpy(rgb).permute((2, 0, 1))
    return make_grid(new_image, *args, **kwargs)

