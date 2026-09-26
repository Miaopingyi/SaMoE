# Installation

Use Python 3.10 and a CUDA-compatible PyTorch installation. The tested development environment uses PyTorch 2.0.1 and CUDA 11.8.

```bash
conda create -n samoe python=3.10 -y
conda activate samoe
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Run these commands from the repository root:

```bash
python train.py --help
python -m unittest discover -s tests -v
```

The required Dassl runtime is included under `vendor/dassl`; no separate Dassl installation is needed. `clip_maple` contains the CLIP backbone used by the trainer.

## Pretrained weights

The default backbone is OpenAI CLIP ViT-B/16. It is downloaded on first use and cached as `~/.cache/clip/ViT-B-16.pt`. Download it in advance on a machine with internet access when preparing an offline environment.

Dataset images and trained SaMoE checkpoints are not included.
