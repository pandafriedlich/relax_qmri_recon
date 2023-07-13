import yaml
from configuration.config import (TrainerConfig,
                                  TrainingTrackingConfig,
                                  ReconstructionBackboneConfig,
                                  SensitivityRefinementModuleConfig,
                                  DataSetConfiguration)
from trainer.qmri_recon_trainer import QuantitativeMRITrainer
from data.paths import CMRxReconDatasetPath

dataset_path_yaml = "../yamls/cmrxrecon_dataset.yaml"
training_config_yaml = "../yamls/all_acc_t1_5_fold.yaml"

# get datapath handler
data_path_handler = CMRxReconDatasetPath(dataset_path_yaml)

# load configuration files
with open(training_config_yaml) as stream:
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
                                 split=0,
                                 disable_tracker=False,
                                 recon_config=recon_config,
                                 ser_config=ser_config,
                                 data_set_config=dataset_config,
                                 training_config=trainer_config,
                                 tracker_config=tracking_config
                                 )
trainer.train()
