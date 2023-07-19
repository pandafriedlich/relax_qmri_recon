# Quantitative MRI Reconstruction Challenge.

## 1. Run the training scripts.
- Create a conda environment with requirements in `direct.yml`.
- Set up the paths to the datasets, experiments dump base in `yamls/cmrxrecon_dataset.yml`, please check the comments in the yaml file for more information.
- Slice the data into smaller pieces, such that the IO speed can be improved during training.
```bash
cd playground 
export PYTHONPATH=$(realpath $(pwd)/..)
python prepare_data.py
```
- Run the training script. 
```bash 
cd playground 
# Add the project base to PYTHONPATH
export PYTHONPATH=$(realpath $(pwd)/..)
split=0 # 5-fold cross-validation, split can be 0-4
running_config=../yamls/all_acc_t1_5_fold.yml
python training.py -s 0 -r $running_config -a train 
```
