import os
import pandas as pd
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class MILK10k(DatasetBase):
    """MILK10k dataset for dermatoscopy images.
    
    This dataset contains dermoscopy images from the MILK study.
    For cross-domain evaluation with DermaMNIST, we keep only matching classes
    and use only dermoscopic images (not clinical close-up images).
    
    Classes (aligned with DermaMNIST):
    - actinic_keratosis (from AKIEC)
    - basal_cell_carcinoma (from BCC)
    - benign_keratosis (from BKL)
    - dermatofibroma (from DF)
    - melanocytic_nevus (from NV)
    - melanoma (from MEL)
    - vascular_lesion (from VASC)
    
    Note: Only dermoscopic images are used for cross-domain evaluation.
    Labels are read from MILK10k_Training_GroundTruth.csv.
    Other labels (BEN_OTH, INF, MAL_OTH, SCCKA) are excluded.
    """
    
    dataset_dir = "MILK10k"
    
    def __init__(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "MILK10k_Training_Input")
        
        # Load ground truth labels
        gt_file = os.path.join(self.dataset_dir, "MILK10k_Training_GroundTruth.csv")
        gt_df = pd.read_csv(gt_file)
        
        # Load metadata to get dermoscopic image ISIC IDs
        meta_file = os.path.join(self.dataset_dir, "MILK10k_Training_Metadata.csv")
        meta_df = pd.read_csv(meta_file)
        
        # Only keep dermoscopic images (not clinical close-up)
        meta_df = meta_df[meta_df['image_type'] == 'dermoscopic'].copy()
        
        # Map MILK10k labels to DermaMNIST labels
        gt_df['dermamnist_label'] = gt_df.apply(self._map_to_dermamnist, axis=1)
        
        # Remove samples that don't match DermaMNIST classes
        gt_df = gt_df[gt_df['dermamnist_label'].notna()].copy()
        
        # Merge with metadata to get ISIC IDs for dermoscopic images
        merged_df = gt_df.merge(
            meta_df[['lesion_id', 'isic_id']], 
            on='lesion_id', 
            how='inner'
        )
        
        print(f"Kept {len(merged_df)} samples with matching DermaMNIST classes")
        
        # Get unique classes (sorted to ensure consistent ordering)
        all_classes = sorted(merged_df['dermamnist_label'].unique())
        print(f"Classes in MILK10k (aligned with DermaMNIST): {all_classes}")
        
        # Print class distribution
        print("\nClass distribution:")
        for cls in all_classes:
            count = (merged_df['dermamnist_label'] == cls).sum()
            print(f"  {cls}: {count}")
        
        # Create test set (all samples for zero-shot evaluation)
        test = self._process_split(merged_df, all_classes)
        
        # For zero-shot evaluation datasets, we use test data for class info extraction
        # but the dataset won't be used for training (eval-only mode in cross-dataset transfer)
        super().__init__(train_x=test, val=[], test=test)
    
    def _map_to_dermamnist(self, row):
        """Map MILK10k labels to DermaMNIST labels.
        
        Mapping:
        - AKIEC -> actinic_keratosis
        - BCC -> basal_cell_carcinoma
        - BKL -> benign_keratosis
        - DF -> dermatofibroma
        - NV -> melanocytic_nevus
        - MEL -> melanoma
        - VASC -> vascular_lesion
        
        Other labels (BEN_OTH, INF, MAL_OTH, SCCKA) are excluded.
        
        Returns None for labels that don't match DermaMNIST classes.
        """
        # Map MILK10k labels to DermaMNIST labels
        label_mapping = {
            'AKIEC': 'actinic_keratosis',
            'BCC': 'basal_cell_carcinoma',
            'BKL': 'benign_keratosis',
            'DF': 'dermatofibroma',
            'NV': 'melanocytic_nevus',
            'MEL': 'melanoma',
            'VASC': 'vascular_lesion',
        }
        
        # Find which label is active (value == 1.0)
        for milk10k_label, dermamnist_label in label_mapping.items():
            if row[milk10k_label] == 1.0:
                return dermamnist_label
        
        # If no matching label found (e.g., BEN_OTH, INF, MAL_OTH, SCCKA), return None
        return None
    
    def _process_split(self, merged_df, classnames):
        """Process samples into Datum format."""
        items = []
        
        for _, row in merged_df.iterrows():
            lesion_id = row['lesion_id']
            isic_id = row['isic_id']
            
            # Image path: MILK10k_Training_Input/{lesion_id}/{isic_id}.jpg
            img_path = os.path.join(self.image_dir, lesion_id, f"{isic_id}.jpg")
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            
            label = row['dermamnist_label']
            classname = label
            label_idx = classnames.index(classname)
            
            item = Datum(
                impath=img_path,
                label=label_idx,
                classname=classname
            )
            items.append(item)
        
        return items
