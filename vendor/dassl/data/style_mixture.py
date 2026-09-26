"""Deterministic acquisition-like style components for compound-shift tests.

The source training mixture is assigned independently within every class, so
the style component cannot become a class proxy. Evaluation renders every
original test image under one forced component at a time. Downstream analysis
can therefore form exact mixture-weight shifts without sampling noise.
"""

import hashlib
import random
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter

from dassl.utils import read_image

from .data_manager import DatasetWrapper


STYLE_NAMES = ("gamma", "noise", "blur", "resolution")


def _stable_int(text):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _largest_remainder_counts(total, weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(STYLE_NAMES),):
        raise ValueError(
            f"SOURCE_WEIGHTS must contain {len(STYLE_NAMES)} values"
        )
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("SOURCE_WEIGHTS must be finite and non-negative")
    if weights.sum() <= 0:
        raise ValueError("SOURCE_WEIGHTS must have positive sum")
    weights = weights / weights.sum()
    expected = weights * total
    counts = np.floor(expected).astype(np.int64)
    remainder = total - int(counts.sum())
    order = np.argsort(-(expected - counts), kind="stable")
    counts[order[:remainder]] += 1
    return counts.tolist()


class StyleMixtureDatasetWrapper(DatasetWrapper):
    """Apply one style component before the ordinary Dassl transform."""

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        super().__init__(
            cfg, data_source, transform=transform, is_train=is_train
        )
        style_cfg = cfg.INPUT.STYLE_MIXTURE
        self.style_cfg = style_cfg
        self.forced_style = int(style_cfg.STYLE_ID)
        if self.forced_style not in (-1, 0, 1, 2, 3):
            raise ValueError("STYLE_ID must be -1 or one of 0,1,2,3")
        if not is_train and self.forced_style < 0:
            raise ValueError(
                "Evaluation with STYLE_MIXTURE enabled requires STYLE_ID"
            )
        self.style_ids = self._build_assignments(cfg)
        counts = np.bincount(self.style_ids, minlength=4).tolist()
        mode = "train-source" if is_train else "forced-evaluation"
        print(
            "Controlled style mixture: "
            f"mode={mode}, counts={counts}, styles={STYLE_NAMES}"
        )

    def _build_assignments(self, cfg):
        if not self.is_train:
            return [self.forced_style] * len(self.data_source)

        groups = defaultdict(list)
        for index, item in enumerate(self.data_source):
            groups[int(item.label)].append(index)

        assignments = [-1] * len(self.data_source)
        salt = int(self.style_cfg.ASSIGNMENT_SEED) + int(cfg.SEED) * 10007
        for label, indices in sorted(groups.items()):
            ordered = sorted(
                indices,
                key=lambda index: _stable_int(
                    f"{salt}|{self.data_source[index].impath}"
                ),
            )
            counts = _largest_remainder_counts(
                len(ordered), self.style_cfg.SOURCE_WEIGHTS
            )
            style_ids = []
            for style_id, count in enumerate(counts):
                style_ids.extend([style_id] * count)
            rng = random.Random(salt + label * 104729)
            rng.shuffle(style_ids)
            for index, style_id in zip(ordered, style_ids):
                assignments[index] = style_id

        if any(style_id < 0 for style_id in assignments):
            raise RuntimeError("Incomplete style assignment")
        return assignments

    def _apply_style(self, image, style_id, impath):
        if style_id == 0:
            array = np.asarray(image).astype(np.float32) / 255.0
            array = np.power(
                np.clip(array, 0.0, 1.0), float(self.style_cfg.GAMMA)
            )
            return Image.fromarray(
                np.round(array * 255.0).astype(np.uint8), mode=image.mode
            )

        if style_id == 1:
            array = np.asarray(image).astype(np.float32) / 255.0
            noise_seed = _stable_int(f"compound-noise|{impath}") % (2**32)
            rng = np.random.default_rng(noise_seed)
            noise = rng.normal(
                0.0, float(self.style_cfg.NOISE_STD), size=array.shape
            )
            array = np.clip(array + noise, 0.0, 1.0)
            return Image.fromarray(
                np.round(array * 255.0).astype(np.uint8), mode=image.mode
            )

        if style_id == 2:
            return image.filter(
                ImageFilter.GaussianBlur(
                    radius=float(self.style_cfg.BLUR_RADIUS)
                )
            )

        if style_id == 3:
            scale = float(self.style_cfg.RESOLUTION_SCALE)
            if not 0.0 < scale < 1.0:
                raise ValueError("RESOLUTION_SCALE must lie in (0, 1)")
            width, height = image.size
            small_size = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            low_resolution = image.resize(small_size, Image.BICUBIC)
            return low_resolution.resize((width, height), Image.BICUBIC)

        raise ValueError(f"Unknown style component {style_id}")

    def __getitem__(self, idx):
        item = self.data_source[idx]
        style_id = int(self.style_ids[idx])
        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "style_id": style_id,
            "style_name": STYLE_NAMES[style_id],
        }

        image = read_image(item.impath)
        image = self._apply_style(image, style_id, item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for transform_index, transform in enumerate(self.transform):
                    transformed = self._transform_image(transform, image)
                    key = "img" if transform_index == 0 else (
                        f"img{transform_index + 1}"
                    )
                    output[key] = transformed
            else:
                output["img"] = self._transform_image(self.transform, image)

        if self.return_img0:
            output["img0"] = self.to_tensor(image)

        return output
