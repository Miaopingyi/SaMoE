"""Train and evaluate biomedical prompt learning."""
import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))

import torch
from yacs.config import CfgNode as CN
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import setup_logger, set_random_seed
import trainers.samoe

DATASETS = {'BTMRI': 'btmri', 'BUSBRA_BIRADS': 'busbra_birads', 'BUSI': 'busi', 'ChestXray14': 'chestxray14', 'CHMNIST': 'chmnist', 'COVID_19': 'covid', 'CTKidney': 'ctkidney', 'CXR8': 'cxr8', 'Derm7pt': 'derm7pt', 'Derm7pt_clin': 'derm7pt_clin', 'Derm7pt_derm': 'derm7pt_derm', 'DermaMNIST': 'dermamnist', 'Gazi': 'gazi', 'ISIC2017': 'isic2017', 'Kather16': 'kather16', 'Kather18': 'kather18', 'KneeXray': 'kneexray', 'Kvasir': 'kvasir', 'LungColon': 'lungcolon', 'MILK10k': 'milk10k', 'MOTUM': 'motum', 'OCTMNIST': 'octmnist', 'PH2': 'ph2', 'RETINA': 'retina', 'RSNAPD': 'rsnapd'}


def extend_cfg(cfg):
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.DATASET.MODALITY = "t1"
    cfg.DATASET.SLICE_INDEX = 0
    cfg.TRAINER.SAMOE = CN({'N_CTX': 8, 'CTX_INIT': 'a photo of a', 'PREC': 'fp32', 'PROMPT_DEPTH': 9, 'NUM_EXPERTS': 4, 'TOP_K': 2, 'ROUTER_HIDDEN': 256, 'LAMBDA_BALANCE': 0.1, 'ROUTING': 'image-conditioned', 'ROUTER_SHARING': 'shared', 'EXPERT_BANK_SHARING': 'independent', 'SAVE_PROMPT_ONLY': True})
    cfg.TRAINER.NAME = 'SaMoE_CLIP'


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--root", default="data", help="dataset root directory")
    parser.add_argument("--config-file", default="configs/base_to_novel.yaml")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--modality", default=None, help="MRI modality for MOTUM or Gazi")
    parser.add_argument("--slice-index", type=int, default=None, help="slice offset (unused by the Gazi adapter)")
    parser.add_argument("--resume", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--load-epoch", type=int)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    return parser


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.DATASET.NAME = args.dataset
    cfg.DATASET.ROOT = args.root
    cfg.DATASET.NUM_SHOTS = args.shots
    cfg.OUTPUT_DIR = args.output_dir
    cfg.SEED = args.seed
    cfg.RESUME = args.resume
    if args.modality is not None:
        cfg.DATASET.MODALITY = args.modality
    if args.slice_index is not None:
        cfg.DATASET.SLICE_INDEX = args.slice_index
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    if cfg.DATASET.NAME not in DATASETS:
        raise ValueError(f"Unknown dataset: {cfg.DATASET.NAME}")
    importlib.import_module("datasets." + DATASETS[cfg.DATASET.NAME])
    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    print(cfg)
    if cfg.USE_CUDA and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    trainer = build_trainer(cfg)
    if args.eval_only:
        if not args.model_dir:
            raise ValueError("--eval-only requires --model-dir")
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
    elif not args.no_train:
        trainer.train()


if __name__ == "__main__":
    main(build_parser().parse_args())
