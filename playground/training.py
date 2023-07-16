import yaml
import click
from configuration.config import (TrainerConfig,
                                  TrainingTrackingConfig,
                                  ReconstructionBackboneConfig,
                                  SensitivityRefinementModuleConfig,
                                  DataSetConfiguration)
from trainer.qmri_recon_trainer import QuantitativeMRITrainer
from data.paths import CMRxReconDatasetPath


@click.command()
@click.option('-s',
              '--split',
              default=0,
              type=int,
              help='Perform training on the given split, valid values are integers in [0, 5)')
@click.option('-d',
              '--dataset-config',
              'dataset_config_path',
              default='../yamls/cmrxrecon_dataset.yaml',
              type=str,
              help='Path to dataset configuration file.'
              )
@click.option('-r',
              '--running-config',
              'running_config_path',
              required=True,
              type=str,
              help='Path to the running configuration file')
@click.option('--disable-tqdm',
              'disable_tqdm',
              is_flag=True,
              default=False,
              help="Disable tqdm, you may want to do this when redirecting stdout to a file.")
def train(split, dataset_config_path, running_config_path, disable_tqdm):
    """The main function for training with command line interface."""
    # get datapath handler
    data_path_handler = CMRxReconDatasetPath(dataset_config_path)

    # load configuration files
    with open(running_config_path) as stream:
        training_configs = yaml.load(stream, yaml.Loader)
    recon_config = ReconstructionBackboneConfig().update(
        training_configs['recon_backbone_config']
    )
    ser_config = SensitivityRefinementModuleConfig().update(
        training_configs['sensitivity_refinement_config']
    )
    dataset_config = DataSetConfiguration().update(
        training_configs['dataset_config']
    )
    trainer_config = TrainerConfig().update(
        training_configs['trainer_config']
    )
    tracking_config = TrainingTrackingConfig().update(
        training_configs['training_tracking_config']
    )

    trainer = QuantitativeMRITrainer(run_name=training_configs['run_name'],
                                     path_handler=data_path_handler,
                                     split=split,
                                     disable_tracker=False,
                                     disable_tqdm=disable_tqdm,
                                     recon_config=recon_config,
                                     ser_config=ser_config,
                                     data_set_config=dataset_config,
                                     training_config=trainer_config,
                                     tracker_config=tracking_config
                                     )
    trainer.train()


if __name__ == '__main__':
    train()
