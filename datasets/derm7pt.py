import os
import pandas as pd
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden


@DATASET_REGISTRY.register()
class Derm7pt(DatasetBase):
    """Derm7pt dataset for dermatoscopy images.
    
    This dataset contains 1011 dermoscopy images with 20 diagnosis types.
    For cross-domain evaluation with DermaMNIST, we keep only matching classes.
    
    Classes (aligned with DermaMNIST):
    - basal_cell_carcinoma
    - melanoma (combined: all melanoma subtypes)
    - melanocytic_nevus (combined: clark nevus, reed/spitz nevus, dermal nevus, blue nevus, 
                         congenital nevus, combined nevus, recurrent nevus)
    - vascular_lesion
    - dermatofibroma
    
    Note: Other diagnosis types (seborrheic_keratosis, lentigo, melanosis, etc.) are excluded
    to ensure class consistency with DermaMNIST.
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
        
        print(f"Kept {len(meta_df)} samples with matching DermaMNIST classes")
        
        # Get unique classes
        all_classes = sorted(meta_df['simplified_diagnosis'].unique())
        print(f"Classes in Derm7pt (aligned with DermaMNIST): {all_classes}")
        
        # Create test set (all samples for zero-shot evaluation)
        test = self._process_split(meta_df, all_classes)
        
        # For zero-shot evaluation datasets, we use test data for class info extraction
        # but the dataset won't be used for training (eval-only mode in cross-dataset transfer)
        super().__init__(train_x=test, val=[], test=test)

    def _simplify_diagnosis(self, diagnosis):
        """Simplify diagnosis to match DermaMNIST classes.
        
        Returns None for diagnosis types that don't match DermaMNIST classes.
        """
        diagnosis = diagnosis.lower()
        
        # Melanoma types (must check before nevus to avoid misclassification)
        if 'melanoma' in diagnosis:
            # Exclude melanoma metastasis
            if 'metastasis' in diagnosis:
                return None
            return 'melanoma'
        
        # Nevus types -> melanocytic_nevus (to match DermaMNIST naming)
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
        
        # Exclude other diagnosis types (seborrheic keratosis, lentigo, melanosis, etc.)
        return None

    def _process_split(self, meta_df, classnames):
        """Process samples into Datum format."""
        items = []
        
        for _, row in meta_df.iterrows():
            img_filename = row['derm']  # Dermoscopy image path
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
