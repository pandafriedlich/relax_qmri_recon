import typing
import torch
import direct.data.transforms as dtrans


def group_tensor_view(x: torch.Tensor):
    """
    Flatten the input of shape (..., nt, 2) to (..., 2 * nt)
    :param x: Tensor of shape (..., nt, 2)
    :return: Tensor of shape (..., 2 * nt)
    """
    x = x.view(*x.shape[:-2], -1)
    return x


def separate_tensor_view(x: torch.Tensor):
    """
    Flatten the input of shape (..., nt * 2) to (..., nt, 2)
    :param x: Tensor of shape (..., nt * 2)
    :return: Tensor of shape (..., nt, 2)
    """
    last_dim_shape = x.shape[-1]
    x = x.view(*x.shape[:-1], last_dim_shape // 2, 2)
    return x


def normalize_sensitivity_map(sensitivity: torch.Tensor,
                              coil_dim: int,
                              complex_dim: int = -1):
    """
    Normalize the sensitivity map.
    :param sensitivity: (..., nc, ..., 2).
    :param complex_dim: Dimension for complex Real/Imag.
    :param coil_dim: Dimension for coils.
    :return: Normalized sensitivity map of the same shape as input.
    """
    sensitivity_map_norm = torch.sqrt(
        (sensitivity ** 2).sum(complex_dim).sum(coil_dim)
    )  # (...)
    sensitivity_map_norm = sensitivity_map_norm.unsqueeze(coil_dim).unsqueeze(
        complex_dim)  # (..., nc, ..., 1)
    sensitivity = dtrans.safe_divide(sensitivity, sensitivity_map_norm)
    return sensitivity


def root_sum_of_square_recon(y: torch.Tensor,
                             backward_operator: torch.Callable,
                             spatial_dim: typing.Tuple[int] = (2, 3),
                             coil_dim: int = 1,
                             complex_dim: int = -1) -> torch.Tensor:
    """
    Get SOS reconstruction of given k-space data.
    :param y:  k-space data of shape (..., nc, ... kx, ky, ..., 2).
    :param backward_operator: ifft operator.
    :param spatial_dim: kx, ky dimensions, default (2, 3)
    :param coil_dim:  coil dimension, default 1.
    :param complex_dim: complex real/imag dimension, default -1.
    :return: RSS reconstruction image of shape (..., kx, ky, ...), the coil dim and complex dim vanishes.
    """
    x = backward_operator(y, dim=spatial_dim)                       # inverse FT
    # RSS transform
    rss = dtrans.root_sum_of_squares(x,
                                     dim=coil_dim,
                                     complex_dim=complex_dim)       # (..., kx, ky, ...)
    return rss


def refine_sensitivity_map(model: torch.nn.Module,
                           sensitivity: torch.Tensor,
                           coil_dim: int = 1,
                           spatial_dim: typing.Tuple[int] = (2, 3),
                           relax_dim: typing.Union[int, None] = 4,
                           complex_dim: int = -1
                           ) -> torch.Tensor:
    """
    Refine sensitivity model.
    :param model: Sensitivity refinement model.
    :param sensitivity: Tensor of shape (nb, nc, kx, ky, [nt], 2).
    :param coil_dim: Dimension coil, default is 1.
    :param spatial_dim: Dimension for kx, ky, default is (2, 3).
    :param relax_dim: Dimension for relaxometry, default is 4.
    :param complex_dim: Dimension for complex numbers, default is -1.
    :return: Refined sensitivity as a tensor of shape (nb, nc, kx, ky, [nt], 2).
    """
    if relax_dim is not None:
        sensitivity = group_tensor_view(sensitivity)        # (nb, nc, kx, ky, 2 * nt)
        channel_dim = relax_dim
    else:
        channel_dim = complex_dim
    sensitivity = torch.permute(sensitivity,
                                (0, coil_dim, channel_dim, *spatial_dim)        # (nb, nc, 2 * nt, kx, ky)
                                )
    output = []
    for idx in range(sensitivity.shape[coil_dim]):
        refined = model(sensitivity.select(coil_dim, idx))       # (0, #channel, kx, ky)
        output.append(refined)
    output = torch.stack(output, dim=coil_dim)
    output = output.permute((0, coil_dim,
                             output.ndim - 2,
                             output.ndim - 1,
                             2
                             ))                                 # (nb, nc, kx, ky, nt * 2)
    if relax_dim is not None:
        output = separate_tensor_view(output)
    output = normalize_sensitivity_map(output, coil_dim=coil_dim,
                                       complex_dim=complex_dim)
    return output

