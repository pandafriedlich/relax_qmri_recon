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

## Acknowledgment
This project borrows code from the [NKI direct](https://github.com/NKI-AI/direct) project.

## Cite this work 
```bibtex
@inproceedings{zhao2023relaxometry,
  title={Relaxometry Guided Quantitative Cardiac Magnetic Resonance Image Reconstruction},
  author={Zhao, Yidong and Zhang, Yi and Tao, Qian},
  booktitle={International Workshop on Statistical Atlases and Computational Models of the Heart},
  pages={349--358},
  year={2023},
  organization={Springer}
}
```


