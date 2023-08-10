from tester.tester import QuantitativeMRIReconTester
from data.paths import CMRxReconDatasetPath

data_path_handler = CMRxReconDatasetPath("../yamls/cmrxrecon_dataset.yaml")

tester = QuantitativeMRIReconTester('dummy',
                                    data_path_handler,
                                    "../yamls/inference.yaml")
tester.run_inference_on_acceleration('AccFactor10', modality='t1')
# tester.run_inference_on_acceleration('AccFactor08')
# tester.run_inference_on_acceleration('AccFactor10')
