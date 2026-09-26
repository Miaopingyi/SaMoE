# Training and Evaluation

Run commands from the repository root. Use `CUDA_VISIBLE_DEVICES` to select a GPU. Dataset names are case-sensitive; `python train.py --help` lists the available adapters.

## Base-to-novel generalization

The script trains on base classes and evaluates the same checkpoint on base and novel classes:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/base2new.sh /path/to/data BTMRI 1
```

Repeat with seeds 2 and 3. The default configuration uses 16 shots per base class and 50 training epochs. To run each stage directly:

```bash
python train.py --dataset BTMRI --root /path/to/data --shots 16 --seed 1 \
  --config-file configs/base_to_novel.yaml --output-dir output/BTMRI/base/seed1

python train.py --dataset BTMRI --root /path/to/data --seed 1 \
  --config-file configs/base_to_novel.yaml --eval-only \
  --model-dir output/BTMRI/base/seed1 --output-dir output/BTMRI/new/seed1 \
  DATASET.SUBSAMPLE_CLASSES new
```

By default, evaluation loads `model-best.pth.tar`, selected on base-class validation data. To evaluate an epoch checkpoint instead, add `--load-epoch N`. `--model-dir` is the training run directory, not an individual checkpoint file.

## Cross-site generalization

Train on DermaMNIST and evaluate a target dataset:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/cross_domain.sh /path/to/data DermaMNIST PH2 1
```

Other targets include `Derm7pt_clin`, `Derm7pt_derm`, and `MILK10k`. Target names must match the adapter registry.

## Cross-protocol generalization

Train on BTMRI and evaluate Gazi MRI images. Pass one of `flair`, `t1w`, `t1w-ce`, or `t2w`:

```bash
CUDA_VISIBLE_DEVICES=0 TARGET_MODALITY=flair \
  bash scripts/cross_domain.sh /path/to/data BTMRI Gazi 1
```

The Gazi adapter expects the prepared 2D images and uses all three extracted slices. Its current implementation does not use `--slice-index` to select a single slice.

## Cross-modality generalization

Use OCTMNIST as the source, for example:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/cross_domain.sh /path/to/data OCTMNIST BTMRI 1
```

This protocol transfers prompts across modalities and may change organs and label spaces. The target class tokens are rebuilt when loading the source prompt checkpoint.

## Cross-domain stages

The cross-domain script trains the source only when no completed best checkpoint and final epoch checkpoint exist. It then evaluates the requested target. To evaluate another target directly without source training:

```bash
python train.py --dataset PH2 --root /path/to/data --seed 1 --shots -1 \
  --config-file configs/cross_domain.yaml --eval-only \
  --model-dir output/cross_domain/DermaMNIST/seed1 \
  --output-dir output/cross_domain/DermaMNIST-to-PH2/seed1
```

Training uses 100 epochs and 16 shots per source class. The target uses all available test examples. Use the same architecture settings for training and evaluation. Set `OUTPUT_ROOT` to change the scripts' output directory or `PYTHON` to select an interpreter.

The result tables in the README reproduce the paper's reported values. The checks included here exercise configuration, core operations and model loading; they do not constitute a new reproduction of those benchmark results.
