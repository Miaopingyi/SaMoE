# Data preparation

Pass the dataset name with `--dataset` and the parent directory with `--root`.

For BTMRI:

```text
data/
  BTMRI/
    BTMRI/
      glioma_tumor/image001.jpg
      ...
    split_BTMRI.json
```

The split file contains `train`, `val` and `test` lists. Each entry is `[relative_image_path, integer_label, class_name]`. Paths are relative to the inner `BTMRI` directory. The four class names are `glioma tumor`, `meningioma tumor`, `pituitary tumor` and `normal brain`.

Use the same splits and class ordering across runs. Few-shot samples are cached in the dataset directory by seed and shot count. Where an adapter can generate a random split, that split is not a substitute for the original evaluation split. Patient-level separation must be established in the input split.

| Adapter | Directory below `--root` |
|---|---|
| `btmri.py` | `BTMRI` |
| `busbra_birads.py` | `BUSBRA` |
| `busi.py` | `BUSI` |
| `chestxray14.py` | `ChestXray14` |
| `chmnist.py` | `CHMNIST` |
| `covid.py` | `COVID_19` |
| `ctkidney.py` | `CTKidney` |
| `cxr8.py` | `CXR8` |
| `derm7pt.py` | `Derm7pt` |
| `derm7pt_clin.py` | `Derm7pt` |
| `derm7pt_derm.py` | `Derm7pt` |
| `dermamnist.py` | `DermaMNIST` |
| `gazi.py` | `Gazi` |
| `isic2017.py` | `ISIC2017` |
| `kather16.py` | `Kather16` |
| `kather18.py` | `Kather18` |
| `kneexray.py` | `KneeXray` |
| `kvasir.py` | `Kvasir` |
| `lungcolon.py` | `LungColon` |
| `milk10k.py` | `MILK10k` |
| `motum.py` | `MOTUM` |
| `octmnist.py` | `OCTMNIST` |
| `ph2.py` | `PH2` |
| `retina.py` | `RETINA` |
| `rsnapd.py` | `RSNAPD` |

See the relevant adapter for annotation fields and preprocessing requirements. Images, original split manifests and model checkpoints are not included.
