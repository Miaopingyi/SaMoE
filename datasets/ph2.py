import os
import pandas as pd
import sys

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取项目根目录
dassl_path = os.path.join(current_dir, 'Dassl.pytorch')
sys.path.append(dassl_path)

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase



@DATASET_REGISTRY.register()
class PH2(DatasetBase):
    """PH2 dataset for dermatoscopy images.
    
    This dataset contains 200 dermoscopy images with 3 clinical diagnoses.
    For cross-domain evaluation with DermaMNIST, labels are aligned as follows:
    
    Classes (aligned with DermaMNIST):
    - melanocytic_nevus (from Common Nevus and Atypical Nevus)
    - melanoma
    
    Original PH2 classes:
    - Common Nevus → melanocytic_nevus
    - Atypical Nevus → melanocytic_nevus
    - Melanoma → melanoma
    
    For cross-domain evaluation, all samples are used for testing (zero-shot).
    """

    dataset_dir = "PH2"

    def __init__(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "trainx")
        
        # Load metadata from Excel file
        excel_file = os.path.join(self.dataset_dir, "PH2_dataset.xlsx")
        
        # Read Excel file, skipping the first 12 rows (header starts at row 13)
        # Row 13 contains column names, data starts from row 14
        meta_df = pd.read_excel(excel_file, skiprows=12, header=0)
        
        # The columns are: Image Name, Histological Diagnosis, Common Nevus, Atypical Nevus, Melanoma, ...
        # We'll extract the diagnosis from the X marks in the columns
        
        # Process the data and map to DermaMNIST labels
        samples = []
        for _, row in meta_df.iterrows():
            img_name = row.iloc[0]  # Image Name
            if pd.isna(img_name) or 'IMD' not in str(img_name):
                continue
            
            # Check which diagnosis column has 'X' and map to DermaMNIST labels
            dermamnist_label = None
            if row.iloc[2] == 'X':  # Common Nevus → melanocytic_nevus
                dermamnist_label = 'melanocytic_nevus'
            elif row.iloc[3] == 'X':  # Atypical Nevus → melanocytic_nevus
                dermamnist_label = 'melanocytic_nevus'
            elif row.iloc[4] == 'X':  # Melanoma → melanoma
                dermamnist_label = 'melanoma'
            
            if dermamnist_label:
                samples.append({
                    'image_name': img_name,
                    'dermamnist_label': dermamnist_label
                })
        
        samples_df = pd.DataFrame(samples)
        
        # Get unique classes (aligned with DermaMNIST)
        all_classes = sorted(samples_df['dermamnist_label'].unique())
        
        print(f"Kept {len(samples_df)} samples with matching DermaMNIST classes")
        print(f"Classes in PH2 (aligned with DermaMNIST): {all_classes}")
        
        # Print class distribution
        from collections import Counter
        class_counts = Counter(samples_df['dermamnist_label'])
        print("\nClass distribution:")
        for cls in all_classes:
            count = class_counts[cls]
            print(f"  {cls}: {count}")
        
        # Create test set (all samples for zero-shot evaluation)
        test = self._process_split(samples_df, all_classes)
        
        # For zero-shot evaluation datasets, we use test data for class info extraction
        # but the dataset won't be used for training (eval-only mode in cross-dataset transfer)
        super().__init__(train_x=test, val=[], test=test)

    def _process_split(self, samples_df, classnames):
        """Process samples into Datum format."""
        items = []
        
        for _, row in samples_df.iterrows():
            img_name = row['image_name']
            img_path = os.path.join(self.image_dir, f"{img_name}.bmp")
            
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
