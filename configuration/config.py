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
    initializer_initialization: str = None
    initializer_channels: typing.Tuple[int, ...] = (32, 32, 64, 64)
    initializer_dilations: typing.Tuple[int, ...] = (1, 1, 2, 4)
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
class MappingModuleConfig(BasicConfig):
    in_channels: int = 18
    out_channels: int = 3
    num_filters: int = 64
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
    warm_up_lr: float = 1e-3
    warm_up_epochs: float = 2
    max_epochs: int = 1000
    save_checkpoint_every: int = 10
    save_optimizer_factor: int = 2
    validation_every: int = 10
    load_pretrained_weight: typing.Optional[typing.Optional[typing.Dict]] = None
    load_pretrained_mapping: typing.Optional[typing.Optional[typing.Dict]] = None
    load_pretrained_latest: bool = False
    combined_loss_weight: typing.Dict[str, float] = \
        dataclasses.field(default_factory=
                          lambda: {"nmse": 1., "ssim": 1.})


@dataclasses.dataclass
class TrainingTrackingConfig(BasicConfig):
    save_training_loss_every: int = 10  # in steps not epochs
    save_training_image_every: int = 500  # in steps not epochs


@dataclasses.dataclass
class PostTrainingValidationConfig(BasicConfig):
    swa: bool = True                        # if SWA will be performed
    swa_overwrite: bool = False             # if the old SWA results should be overwritten
    swa_epoch_start: int = 150              # The first epoch number for SWA (inclusive).
    swa_epoch_end: int = 200                # The last epoch number for SWA (exclusive).
    swa_update_bn_steps: int = 100          # number of forward passes to update normalization layers
    validate_with_swa: bool = True          # if the validation should be performed with the SWA weight.


@dataclasses.dataclass
class ImageDomainAugmentationConfig(BasicConfig):
    rotation: float = 45.
    translation: float = 0.10                   # 10% of image shape
    shearing: float = 20.
    scale_min: float = 0.7
    scale_max: float = 1.4
    contamination_max_rel: float = 0.1
    p_affine: float = 0.2
    p_flip: float = 0.2
    p_new_mask: float = 0.2
    p_contamination: float = 0.1
    # augmentation scheduling
    decay_factor: float = 0.5           # p_new = p_old * decay_factor, 1.0 for no decay
    decay_every: int = 100              # decay after this many epochs

