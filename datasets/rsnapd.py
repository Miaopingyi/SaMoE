import os
import pickle
import math
import random
import json
from collections import defaultdict

import sys

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dassl_path = os.path.join(current_dir, 'Dassl.pytorch')
sys.path.append(dassl_path)

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing


@DATASET_REGISTRY.register()
class RSNAPD(DatasetBase):
    """RSNA Pneumonia Detection Challenge Dataset.
    
    This dataset contains chest X-ray images for pneumonia detection.
    The task is binary classification: Normal vs Pneumonia.
    
    Original labels from the JSON:
    - subset_init_label: 0 (Normal), 1 (No Lung Opacity / Not Normal), 2 (Lung Opacity)
    
    We simplify to:
    - Normal (label 0): subset_init_label == 0
    - Pneumonia (label 1): subset_init_label in [1, 2]
    
    This dataset is used as a TARGET DOMAIN for Cross-site Domain Generalization.
    """

    dataset_dir = "RSNAPD"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(
            self.dataset_dir, 
            "mdai_rsna_project_x9N20BZa_images_2018-07-20-153330"
        )
        self.json_file = os.path.join(
            self.dataset_dir,
            "pneumonia-challenge-dataset-mappings_2018.json"
        )
        self.split_path = os.path.join(self.dataset_dir, "split_RSNAPD.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_and_split_data(
                self.json_file, self.image_dir
            )
            self.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_and_split_data(json_file, image_dir, p_trn=0.5, p_val=0.2):
        """Read and split RSNAPD dataset from JSON file."""
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"Total samples in RSNAPD: {len(data)}")
        
        # Simplified binary classification
        classnames = ['Normal', 'Pneumonia']
        
        # Group samples by label
        label_to_samples = defaultdict(list)
        
        for entry in data:
            img_id = entry['img_id']
            subset_label = entry['subset_init_label']
            
            # Map to binary classes
            # 0: Normal, 1/2: Pneumonia
            if subset_label == 0:
                label = 0  # Normal
                classname = 'Normal'
            else:
                label = 1  # Pneumonia (includes label 1 and 2)
                classname = 'Pneumonia'
            
            # Construct image path
            # Images are organized in subdirectories
            img_path = None
            for subdir in os.listdir(image_dir):
                subdir_path = os.path.join(image_dir, subdir)
                if os.path.isdir(subdir_path):
                    potential_path = os.path.join(subdir_path, img_id)
                    if os.path.exists(potential_path):
                        img_path = potential_path
                        break
            
            if img_path and os.path.exists(img_path):
                label_to_samples[label].append((img_path, label, classname))
        
        # Split data
        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")
        
        train, val, test = [], [], []
        
        for label, samples in label_to_samples.items():
            if len(samples) == 0:
                continue
            
            random.shuffle(samples)
            n_total = len(samples)
            n_train = max(1, round(n_total * p_trn))
            n_val = max(1, round(n_total * p_val))
            n_test = max(1, n_total - n_train - n_val)
            
            classname = samples[0][2]
            print(f"  Class '{classname}': {n_total} samples "
                  f"({n_train} train, {n_val} val, {n_test} test)")
            
            # Create Datum objects
            for i, (impath, lbl, cn) in enumerate(samples):
                item = Datum(impath=impath, label=lbl, classname=cn)
                if i < n_train:
                    train.append(item)
                elif i < n_train + n_val:
                    val.append(item)
                else:
                    test.append(item)
        
        print(f"Split: {len(train)} train, {len(val)} val, {len(test)} test")
        return train, val, test

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Subsample classes for evaluation.
        
        For Cross-site DG, this dataset is used as target domain (zero-shot test).
        We typically use subsample="all" to test on all classes.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args
        
        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]
        else:
            selected = labels[m:]
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output

