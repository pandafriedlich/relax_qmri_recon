import typing
import dataclasses
import os


class BasicConfig:
    def update(self, new: dict):
        """
        Update the configuration from a dict.
        :param new:
        :return:
        """
        for key, value in new.items():
            setattr(self, key, value)
        return self


@dataclasses.dataclass
class ReconstructionBackboneConfig(BasicConfig):
    forward_operator: str = 'fft2'
    backward_operator: str = 'ifft2'
    in_channels: int = 18
    num_steps: int = 4
    recurrent_hidden_channels: int = 64
    recurrent_num_layers: int = 4
    no_parameter_sharing: bool = True
    learned_initializer: bool = True
    initializer_initialization: str = None,
    initializer_channels: typing.Tuple[int, ...] = (32, 32, 64, 64),
    initializer_dilations: typing.Tuple[int, ...] = (1, 1, 2, 4),
    initializer_multiscale: int = 1
    normalized: bool = False


@dataclasses.dataclass
class SensitivityRefinementModuleConfig(BasicConfig):
    in_channels: int = 2
    out_channels: int = 2
    num_filters: int = 8
    num_pool_layers: int = 4
    dropout_probability: int = 0.0


@dataclasses.dataclass
class DataSetConfiguration(BasicConfig):
    acceleration_factors: typing.Tuple[int] = (4., 8., 10.)
    modality: str = 't1map'
    overwrite_split: bool = False


@dataclasses.dataclass
class TrainerConfig(BasicConfig):
    batch_size: int = 2
    dataloader_num_workers: int = 8
    weight_decay: float = 0.
    lr_decay: float = 0.9
    lr_init: float = 1e-3
    max_epochs: int = 1000
    save_checkpoint_every: int = 10
    save_optimizer_factor: int = 2
    validation_every: int = 10
    load_pretrained_weight: typing.Optional[typing.Union[str, bytes, os.PathLike]] = None
    load_pretrained_latest: bool = False
    combined_loss_weight: typing.Dict[str, float] = \
        dataclasses.field(default_factory=
                          lambda: {"nmse": 1., "ssim": 1.})


@dataclasses.dataclass
class TrainingTrackingConfig(BasicConfig):
    save_training_loss_every: int = 10  # in steps not epochs
    save_training_image_every: int = 500  # in steps not epochs
