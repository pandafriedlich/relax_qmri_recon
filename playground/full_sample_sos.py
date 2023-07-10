import yaml
from pathlib import Path
import mat73
from matplotlib import pyplot as plt
from data.utils import SoS_Reconstruction


_multi_coil = "MultiCoil"
_task_type = "Mapping"
_split = "TrainingSet"
_acceleration = "FullSample"
_subject_code = "P001"

dataset_paths = yaml.load(Path("../cmrxrecon_dataset.yaml").open('r'),
                          yaml.Loader)
dataset_base = Path(dataset_paths["dataset_base"])
subject_base = dataset_base / _multi_coil / _task_type / _split / _acceleration / _subject_code


def load_subject(base, filenames=('T1map', 'T1map_mask')):
    base = Path(base)
    data = dict()
    for key in filenames:
        mat_file = mat73.loadmat(base / f"{key}.mat")
        data[key] = list(mat_file.values())[0]
    return data


subject = load_subject(subject_base, filenames=('T1map', ))
y = subject['T1map']                         # (kx, ky, nc, ns, nt)
x = SoS_Reconstruction(y, fft_dim=(0, 1),
                       coil_dim=2,
                       centered=True,
                       complex_input=False)

fig, axes = plt.subplots(2, 2, dpi=200)
for sl_ind in range(2):
    for t_ind in range(2):
        ax = axes[sl_ind, t_ind]
        ax.imshow(x[..., sl_ind, t_ind].T, cmap='gray')
        ax.set_aspect('equal')
        ax.axis('off')
plt.show()
plt.close()
