import os
import pandas as pd
import sys

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取项目根目录
dassl_path = os.path.join(current_dir, 'Dassl.pytorch')
sys.path.append(dassl_path)

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class MOTUM(DatasetBase):
    """MOTUM dataset for brain MRI images.
    
    This dataset contains brain MRI images with different imaging protocols.
    For cross-protocol domain generalization, we load specific MRI modalities.
    
    Supported modalities: t1, t1ce, t2, flair
    
    Classes (aligned with BTMRI):
    - glioma_tumor (all Glioma cases map to this class)
    
    Note on class distribution:
    - MOTUM contains 29 Glioma cases (primary brain tumors)
    - All Glioma subtypes (Glioblastoma, Oligodendroglioma, Astrocytoma) 
      are mapped to 'glioma_tumor' to align with BTMRI
    - Other case types (LungMeta, BreastMeta, etc.) are metastases and 
      cannot be mapped to BTMRI classes, so they are excluded
    
    Important: This dataset has only 1 class (glioma_tumor). In zero-shot 
    evaluation, the model outputs probabilities for 4 classes (BTMRI classes),
    but all test samples are glioma_tumor. This means accuracy will be high
    if the model correctly identifies glioma_tumor, but this primarily tests
    cross-protocol generalization (different MRI sequences) rather than 
    cross-class generalization.
    
    For cross-domain evaluation, all samples are used for testing (zero-shot).
    The modality is specified via DATASET.MODALITY config parameter.
    """
    
    dataset_dir = "MOTUM"
    supported_modalities = ["t1", "t1ce", "t2", "flair"]
    
    def __init__(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        
        # Check if 2D images exist, otherwise use original NIfTI directory
        images_2d_dir = os.path.join(self.dataset_dir, "images_2d")
        images_dir = os.path.join(self.dataset_dir, "images")
        
        if os.path.exists(images_2d_dir):
            self.image_dir = images_2d_dir
            self.use_2d_images = True
            print("Using 2D converted images from images_2d/")
        else:
            self.image_dir = images_dir
            self.use_2d_images = False
            print("Using original NIfTI files from images/")
            print("Note: For better performance, consider converting to 2D using:")
            print("  python scripts/preprocessing/convert_motum_nifti_to_2d.py")
        
        # Get modality from config (default to t1 if not specified)
        modality = getattr(cfg.DATASET, 'MODALITY', 't1')
        if modality not in self.supported_modalities:
            raise ValueError(f"Modality must be one of {self.supported_modalities}, got {modality}")
        
        self.modality = modality
        
        # Load metadata from Participants.xlsx
        excel_file = os.path.join(self.dataset_dir, "Participants.xlsx")
        meta_df = pd.read_excel(excel_file)
        
        # Get diagnosis column name (may vary)
        diag_col = None
        for col in meta_df.columns:
            if 'diagnosis' in col.lower() or 'pathologic' in col.lower():
                diag_col = col
                break
        
        if diag_col is None:
            diag_col = meta_df.columns[4]  # Usually the 5th column
        
        # Process Glioma cases only
        samples = []
        for _, row in meta_df.iterrows():
            case_id = row['ID']
            
            # Construct glioma directory name
            if '-' in case_id:
                parts = case_id.split('-')
                glioma_dir = f"{parts[0]}-Glioma-{parts[-1]}"
            else:
                glioma_dir = f"{case_id}-Glioma"
            
            glioma_path = os.path.join(self.image_dir, glioma_dir)
            
            if not os.path.exists(glioma_path):
                continue
            
            # Check if the specified modality file exists
            if self.use_2d_images:
                # Use 2D converted images (middle slice)
                modality_file = f"{self.modality}_middle.png"
                img_path = os.path.join(glioma_path, modality_file)
                # Try jpg if png doesn't exist
                if not os.path.exists(img_path):
                    modality_file = f"{self.modality}_middle.jpg"
                    img_path = os.path.join(glioma_path, modality_file)
            else:
                # Use original NIfTI files
                modality_file = f"{self.modality}.nii.gz"
                img_path = os.path.join(glioma_path, modality_file)
            
            if not os.path.exists(img_path):
                print(f"Warning: Modality {self.modality} not found for {case_id}: {img_path}")
                continue
            
            # All Glioma cases map to glioma_tumor (to align with BTMRI)
            samples.append({
                'case_id': case_id,
                'glioma_dir': glioma_dir,
                'img_path': img_path,
                'diagnosis': row[diag_col] if pd.notna(row[diag_col]) else 'Glioma',
                'label': 'glioma_tumor'
            })
        
        samples_df = pd.DataFrame(samples)
        
        if len(samples_df) == 0:
            raise ValueError(f"No samples found for modality {self.modality}")
        
        # Get unique classes (should only be glioma_tumor)
        all_classes = sorted(samples_df['label'].unique())
        
        print(f"MOTUM Dataset - Modality: {self.modality}")
        print(f"  Total samples: {len(samples_df)}")
        print(f"  Classes: {all_classes}")
        
        # Print diagnosis distribution for reference
        from collections import Counter
        diag_counts = Counter(samples_df['diagnosis'])
        print(f"\n  Diagnosis distribution:")
        for diag, count in diag_counts.most_common():
            print(f"    {diag}: {count}")
        
        # Create test set (all samples for zero-shot evaluation)
        test = self._process_split(samples_df, all_classes)
        
        # For zero-shot evaluation datasets, we use test data for class info extraction
        # but the dataset won't be used for training (eval-only mode in cross-dataset transfer)
        super().__init__(train_x=test, val=[], test=test)
    
    def _process_split(self, samples_df, classnames):
        """Process samples into Datum format."""
        items = []
        
        for _, row in samples_df.iterrows():
            img_path = row['img_path']
            label = row['label']
            classname = label
            label_idx = classnames.index(classname)
            
            item = Datum(
                impath=img_path,
                label=label_idx,
                classname=classname
            )
            items.append(item)
        
        return items

