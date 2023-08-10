import copy
import typing
import os
import yaml
from copy import deepcopy
from pathlib import Path
from models.recurrentunet import RecurrentVarNet
from direct.nn.unet import UnetModel2d
from models.tricathlon import QuantitativeMRIReconstructionNet
import torch
from direct.data.transforms import fft2, ifft2
from models.utils import root_sum_of_square_recon
from models.tricathlon import MOLLIMappingNet, T2RelaxMappingNet


def parse_inference_yaml(inference_file: typing.Union[str, bytes, os.PathLike]) -> typing.Dict[
        str, typing.Any]:
    """
    Load inference configuration from the yaml file.
    :param inference_file:
    :return:
    """
    with open(inference_file) as stream:
        inference_configs = yaml.load(stream, yaml.Loader)
    default_config = inference_configs['Default']
    configs = dict()

    for acc in ('AccFactor04', 'AccFactor08', 'AccFactor10'):
        # initialize with default configuration
        configs[acc] = deepcopy(default_config)

        # update configuration if applicable
        update = inference_configs.get(acc, {})
        if update is None:
            update = {}
        for modality in update.keys():                                  # modalities
            m_config = update.get(modality, {})
            if m_config is None:
                m_config = {}
            for key, val in m_config.items():                           # 'recon_backbone_config' etc.
                configs[acc][modality][key].update(val)
    return configs


def get_checkpoints_paths(dump_base: typing.Union[str, bytes, os.PathLike],
                          checkpoints_config: typing.Dict[str, typing.Any]) -> typing.List[Path]:
    """
    Get absolute paths to the dumped checkpoints.
    :param dump_base: Path to the dumped training checkpoints.
    :param checkpoints_config: Dictionary of the selected checkpoints.
    :return:
    """
    dump_base = Path(dump_base)
    run_name = checkpoints_config['run_name']
    dump_base = dump_base / run_name
    checkpoint_paths = []
    for fold in range(5):
        fold_name = f"fold_{fold:d}"
        selected = checkpoints_config[fold_name]
        for s in selected:
            if s == 'latest':
                filename = 'model_latest.model'
            elif isinstance(s, int):
                filename = f'epoch_{s:04d}.model'
            elif s.startswith('model_swa'):
                filename = f'{s}.model'
            else:
                raise ValueError(f"invalid checkpoint name: {s}")
            checkpoint_paths.append(dump_base / fold_name / "models" / filename)
    return checkpoint_paths


def build_models_from_config(dump_base: typing.Union[str, bytes, os.PathLike],
                             model_config: typing.Dict[str, typing.Any]) -> typing.List[QuantitativeMRIReconstructionNet]:
    """
    Build models from configuration.
    :param dump_base: Path to the training dump base.
    :param model_config: Configuration of model architecture and checkpoints.
    :return: List of models for ensemble.
    """
    checkpoints = get_checkpoints_paths(dump_base, model_config['model_checkpoints'])
    num_models = len(checkpoints)

    _supported_operators = {
        'fft2': fft2,
        'ifft2': ifft2
    }
    model_config['recon_backbone_config']['forward_operator'] = _supported_operators[model_config['recon_backbone_config']['forward_operator']]
    model_config['recon_backbone_config']['backward_operator'] = _supported_operators[model_config['recon_backbone_config']['backward_operator']]

    models = [
        QuantitativeMRIReconstructionNet(
            RecurrentVarNet(**model_config['recon_backbone_config']),
            UnetModel2d(**model_config['sensitivity_refinement_config'])
        )
        for _ in range(num_models)
    ]
    for m in range(num_models):
        checkpoint = torch.load(checkpoints[m])['model']
        models[m].load_state_dict(checkpoint)
        models[m].eval()
        models[m].cuda().float()
    return models


def ensemble_prediction(models: typing.List[QuantitativeMRIReconstructionNet],
                        sample: typing.Dict[str, typing.Any],
                        mapping_model: typing.Optional[typing.Union[MOLLIMappingNet, T2RelaxMappingNet]] = None):
    """
    Make ensemble prediction with vmap. Reference: https://pytorch.org/docs/stable/func.migrating.html
    :param models: List of models in ensemble.
    :param sample: Input to the model.
    :param mapping_model: Mapping model.
    :return: Ensemble output.
    """
    base_model = copy.deepcopy(models[0])
    base_model.to('meta')
    params, buffers = torch.func.stack_module_state(models)

    def call_single_model(params, buffers, data):
        return torch.func.functional_call(base_model, (params, buffers), (data,))
    output = torch.vmap(call_single_model, (0, 0, None))(params, buffers, sample)
    pred_kspace = output['pred_kspace'].mean(dim=0).squeeze(0)                                              # (nc, kx, ky, nt, 2)
    rss = root_sum_of_square_recon(pred_kspace, ifft2, spatial_dim=(1, 2), coil_dim=0, complex_dim=-1)      # (kx, ky, nt)
    if mapping_model is not None:
        map_pred = mapping_prediction(mapping_model, rss, sample['tvec'])
    else:
        map_pred = dict(pmap=None, air=None)
    rss = rss.double()
    rss *= sample['scaling_factor'].cuda().double().squeeze()
    return dict(rss=rss, pmap=map_pred['pmap'], air=map_pred['air'])


def build_mapping_model(dump_base: typing.Union[str, bytes, os.PathLike],
                        modality: str='t1') -> torch.nn.Module:
    if modality == 't1':
        model = MOLLIMappingNet(in_channels=18, out_channels=3,
                                num_filters=256, num_pool_layers=1, dropout_probability=0.)
        ckpt_path = get_checkpoints_paths(dump_base, dict(run_name='t1mapping_net',
                                                          fold_0=[1000], fold_1=[],
                                                          fold_2=[], fold_3=[],
                                                          fold_4=[]))
        checkpoint = torch.load(ckpt_path[0])
        model.load_state_dict(checkpoint['mapping_model'])
    elif modality == 't2':
        model = T2RelaxMappingNet(in_channels=6, out_channels=3,
                                num_filters=256, num_pool_layers=1, dropout_probability=0.)
        ckpt_path = get_checkpoints_paths(dump_base, dict(run_name='t2mapping_net',
                                                          fold_0=['latest'], fold_1=[],
                                                          fold_2=[], fold_3=[],
                                                          fold_4=[]))
        checkpoint = torch.load(ckpt_path[0])
        model.load_state_dict(checkpoint['model'])
    else:
        raise ValueError(f"Unknown modality {modality}!")

    return model


def mapping_prediction(model: typing.Union[MOLLIMappingNet, T2RelaxMappingNet],
                       rss: torch.Tensor,
                       t_relax: torch.Tensor):
    model.cuda().float()
    rss = torch.permute(rss, (2, 0, 1))[None, ...]         # (kx, ky, kt) -> (1, kt, kx, ky)
    t_relax = t_relax.expand(*rss.shape) * 1e-3
    rss = rss.cuda().float()
    t_relax = t_relax.cuda().float()
    x = torch.cat([t_relax, rss], dim=1)
    pmap = model(x.cuda().float())
    air = model.get_air_mask(rss)
    return dict(pmap=pmap, air=air)
