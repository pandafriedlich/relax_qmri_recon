from data.qmridata import CMRxReconQuantitativeRawDataset
from data.paths import CMRxReconDatasetPath

# load dataset
dataset_paths = CMRxReconDatasetPath("../cmrxrecon_dataset.yaml")
multi_coil_mapping_training = dataset_paths.get_raw_data_path("MultiCoil", "Mapping", "TrainingSet")
sliced_multi_coil_mapping_training = dataset_paths.get_sliced_data_path("MultiCoil", "Mapping", "TrainingSet")

# split the files
training = CMRxReconQuantitativeRawDataset(multi_coil_mapping_training)
training.save_split_files(sliced_multi_coil_mapping_training)

