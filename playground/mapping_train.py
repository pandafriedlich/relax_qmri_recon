import yaml
import click
from configuration.config import (TrainerConfig,
                                  TrainingTrackingConfig,
                                  DataSetConfiguration,
                                  MappingModuleConfig,
                                  ImageDomainAugmentationConfig,
                                  PostTrainingValidationConfig)
from trainer.mapping_net_trainer import QuantitativeMappingTrainer
from data.paths import CMRxReconDatasetPath


@click.command()
@click.option('-f',
              '--fold',
              default=0,
              type=int,
              help='Perform training on the given fold, valid values are integers in [0, 5)')
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
@click.option('-a', '--action',
              multiple=True,
              default=['train'],
              help='Specify the trainer action, can be `train` and `validate`.')
@click.option('--disable-tqdm',
              'disable_tqdm',
              is_flag=True,
              default=False,
              help="Disable tqdm, you may want to do this when redirecting stdout to a file.")
@click.option('--disable-tracker',
              'disable_tracker',
              is_flag=True,
              default=False,
              help="Disable loss tracker (tensorboard summary writer or something similar.")
def train(fold,
          dataset_config_path,
          running_config_path,
          action,
          disable_tqdm,
          disable_tracker):
    """The main function for training with command line interface."""
    for a in action:
        assert a.lower() in ('train',), f"Unknown action {a}!"

    # get datapath handler
    data_path_handler = CMRxReconDatasetPath(dataset_config_path)

    # load configuration files
    with open(running_config_path) as stream:
        training_configs = yaml.load(stream, yaml.Loader)
    dataset_config = DataSetConfiguration().update(
        training_configs['dataset_config']
    )
    augmentation_config = ImageDomainAugmentationConfig().update(
        training_configs['augmentation_config']
    )
    trainer_config = TrainerConfig().update(
        training_configs['trainer_config']
    )
    tracking_config = TrainingTrackingConfig().update(
        training_configs['training_tracking_config']
    )
    mapping_network_config = MappingModuleConfig().update(
        training_configs['mapping_network_config']
    )

    trainer = QuantitativeMappingTrainer(run_name=training_configs['run_name'],
                                         path_handler=data_path_handler,
                                         fold=fold,
                                         disable_tracker=disable_tracker,
                                         disable_tqdm=disable_tqdm,
                                         map_config=mapping_network_config,
                                         data_set_config=dataset_config,
                                         augmentation_config=augmentation_config,
                                         training_config=trainer_config,
                                         tracker_config=tracking_config
                                         )
    if 'train' in action:
        trainer.train()


if __name__ == '__main__':
    train()
