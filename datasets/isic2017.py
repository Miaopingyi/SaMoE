import os
import pandas as pd
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class ISIC2017(DatasetBase):
    """ISIC2017 dataset for dermatoscopy images.
    
    This dataset contains 600 dermoscopy images with diagnosis information.
    For cross-domain evaluation, we use the diagnosis_1 column which provides
    binary classification: Benign vs Malignant.
    
    Classes:
    - benign
    - malignant
    
    For cross-domain evaluation, all samples are used for testing (zero-shot).
    """

    dataset_dir = "ISIC2017"

    def __init__(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        
        # Load metadata
        meta_file = os.path.join(self.dataset_dir, "metadata.csv")
        meta_df = pd.read_csv(meta_file)
        
        # Only keep samples with diagnosis_1
        meta_df = meta_df[meta_df['diagnosis_1'].notna()].copy()
        
        # Normalize diagnosis labels
        meta_df['diagnosis_label'] = meta_df['diagnosis_1'].str.lower()
        
        # Get unique classes
        all_classes = sorted(meta_df['diagnosis_label'].unique())
        
        # Create test set (all samples for zero-shot evaluation)
        test = self._process_split(meta_df, all_classes)
        
        # For zero-shot evaluation datasets, we use test data for class info extraction
        # but the dataset won't be used for training (eval-only mode in cross-dataset transfer)
        super().__init__(train_x=test, val=[], test=test)

    def _process_split(self, meta_df, classnames):
        """Process samples into Datum format."""
        items = []
        
        for _, row in meta_df.iterrows():
            img_name = row['isic_id']
            img_path = os.path.join(self.dataset_dir, f"{img_name}.jpg")
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            
            label = row['diagnosis_label']
            classname = label
            label_idx = classnames.index(classname)
            
            item = Datum(
                impath=img_path,
                label=label_idx,
                classname=classname
            )
            items.append(item)
        
        return items

