import os
import pandas as pd
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class Derm7pt_derm(DatasetBase):
    """Derm7pt dataset - Dermoscopy modality only.
    
    This dataset contains dermoscopy images from Derm7pt.
    Used for cross-domain evaluation with DermaMNIST as source domain.
    
    Classes (aligned with DermaMNIST):
    - basal_cell_carcinoma
    - melanoma (combined: all melanoma subtypes)
    - melanocytic_nevus (combined: clark nevus, reed/spitz nevus, dermal nevus, blue nevus, 
                         congenital nevus, combined nevus, recurrent nevus)
    - vascular_lesion
    - dermatofibroma
    """

    dataset_dir = "Derm7pt"

    def __init__(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        self.meta_dir = os.path.join(self.dataset_dir, "release_v0", "meta")
        self.image_dir = os.path.join(self.dataset_dir, "release_v0", "images")

        # Load metadata
        meta_file = os.path.join(self.meta_dir, "meta.csv")
        meta_df = pd.read_csv(meta_file)
        
        # Only keep samples with dermoscopy images
        meta_df = meta_df[meta_df['derm'].notna()].copy()
        
        # Simplify diagnosis classes (aligned with DermaMNIST)
        meta_df['simplified_diagnosis'] = meta_df['diagnosis'].apply(self._simplify_diagnosis)
        
        # Remove samples that don't match DermaMNIST classes
        meta_df = meta_df[meta_df['simplified_diagnosis'].notna()].copy()
        
        print(f"[Derm7pt_derm] Kept {len(meta_df)} dermoscopy samples with matching DermaMNIST classes")
        
        # Get unique classes
        all_classes = sorted(meta_df['simplified_diagnosis'].unique())
        print(f"[Derm7pt_derm] Classes: {all_classes}")
        
        # Create test set (all samples for zero-shot evaluation)
        test = self._process_split(meta_df, all_classes, 'derm')
        
        super().__init__(train_x=test, val=[], test=test)

    def _simplify_diagnosis(self, diagnosis):
        """Simplify diagnosis to match DermaMNIST classes."""
        diagnosis = diagnosis.lower()
        
        # Melanoma types (must check before nevus to avoid misclassification)
        if 'melanoma' in diagnosis:
            if 'metastasis' in diagnosis:
                return None
            return 'melanoma'
        
        # Nevus types -> melanocytic_nevus
        if any(keyword in diagnosis for keyword in ['nevus', 'naevus']):
            return 'melanocytic_nevus'
        
        # Basal cell carcinoma
        if 'basal cell carcinoma' in diagnosis:
            return 'basal_cell_carcinoma'
        
        # Vascular lesion
        if 'vascular' in diagnosis:
            return 'vascular_lesion'
        
        # Dermatofibroma
        if 'dermatofibroma' in diagnosis:
            return 'dermatofibroma'
        
        return None

    def _process_split(self, meta_df, classnames, modality_col):
        """Process samples into Datum format."""
        items = []
        
        for _, row in meta_df.iterrows():
            img_filename = row[modality_col]
            img_path = os.path.join(self.image_dir, img_filename)
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            
            label = row['simplified_diagnosis']
            classname = label
            label_idx = classnames.index(classname)
            
            item = Datum(
                impath=img_path,
                label=label_idx,
                classname=classname
            )
            items.append(item)
        
        return items

