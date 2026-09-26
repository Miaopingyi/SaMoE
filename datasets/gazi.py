import os
import sys

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取项目根目录
dassl_path = os.path.join(current_dir, 'Dassl.pytorch')
sys.path.append(dassl_path)

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class Gazi(DatasetBase):
    """Gazi Brains 2020 dataset for brain MRI images.
    
    This dataset contains brain MRI images with different imaging protocols.
    For cross-protocol domain generalization, we load specific MRI modalities.
    
    Dataset information:
    - Total: 100 subjects (100 scans)
    - Subjects 1-50: HGG (High Grade Glial tumors) → glioma_tumor
    - Subjects 51-100: Normal healthy subjects → normal_brain
    
    Supported modalities:
    - t1w: T1-weighted (all subjects)
    - t2w: T2-weighted (all subjects)
    - flair: FLAIR sequence (all subjects)
    - t1w-ce: Gadolinium enhanced T1-weighted (HGG subjects + 12 normal subjects)
    
    Classes (aligned with BTMRI):
    - normal_brain: Normal healthy subjects (subjects 51-100)
    - glioma_tumor: HGG patients (subjects 1-50)
    
    For cross-domain evaluation, all samples are used for testing (zero-shot).
    The modality is specified via DATASET.MODALITY config parameter.
    """
    
    dataset_dir = "Gazi"
    supported_modalities = ["t1w", "t2w", "flair", "t1w-ce"]
    
    # Mapping from config modality names to file name patterns
    modality_file_patterns = {
        "t1w": "t1w.nii",
        "t2w": "t2w.nii",
        "flair": "flair.nii",
        "t1w-ce": "ce-gadolinium_t1w.nii"
    }
    
    def __init__(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        self.data_dir = os.path.join(self.dataset_dir, "data")
        
        # Get modality from config (default to t1w if not specified)
        modality = getattr(cfg.DATASET, 'MODALITY', 't1w')
        if modality not in self.supported_modalities:
            raise ValueError(f"Modality must be one of {self.supported_modalities}, got {modality}")
        
        # For mri_cross_protocol_dg experiments, we use all three layers (t-1, middle, t+1)
        # SLICE_INDEX config is ignored - all three layers are loaded
        self.modality = modality
        self.file_pattern = self.modality_file_patterns[modality]
        
        # Check if 2D images exist, otherwise use original NIfTI directory
        images_2d_dir = os.path.join(self.dataset_dir, "images_2d")
        
        if os.path.exists(images_2d_dir):
            self.use_2d_images = True
            self.image_base_dir = images_2d_dir
            print("Using 2D converted images from images_2d/")
        else:
            self.use_2d_images = False
            self.image_base_dir = self.data_dir
            print("Using original NIfTI files from data/")
            print("Note: For better performance, consider converting to 2D using:")
            print("  python scripts/preprocessing/convert_gazi_nifti_to_2d.py")
        
        # Process all subjects (1-100)
        samples = []
        
        for subject_id in range(1, 101):
            subject_dir = f"{subject_id:03d}"  # 001, 002, ..., 100
            subject_path = os.path.join(self.image_base_dir, subject_dir)
            
            if not os.path.exists(subject_path):
                continue
            
            # Determine label based on subject ID
            # Subjects 1-50: HGG → glioma_tumor
            # Subjects 51-100: Normal → normal_brain
            if subject_id <= 50:
                label = 'glioma_tumor'
                group = 'HGG'
            else:
                label = 'normal_brain'
                group = 'Normal'
            
            # Find all three slice layers (t-1, middle, t+1) for mri_cross_protocol_dg
            if self.use_2d_images:
                modality_base = self.modality.replace('-', '_')
                
                # Try to load all three layers: t-1, middle, t+1
                slice_names = ['t-1', 'middle', 't+1']
                layer_samples = []
                
                for slice_name in slice_names:
                    # Look for pattern: {modality}_{slice_name}.{ext}
                    # e.g., t1w_middle.png, t1w_t-1.png, t1w_t+1.png
                    img_file = f"{modality_base}_{slice_name}.png"
                    img_path = os.path.join(subject_path, img_file)
                    # Try jpg if png doesn't exist
                    if not os.path.exists(img_path):
                        img_file = f"{modality_base}_{slice_name}.jpg"
                        img_path = os.path.join(subject_path, img_file)
                    
                    # If file exists, add it as a sample
                    if os.path.exists(img_path):
                        layer_samples.append({
                            'subject_id': subject_id,
                            'subject_dir': subject_dir,
                            'img_path': img_path,
                            'label': label,
                            'group': group,
                            'slice_layer': slice_name
                        })
                
                # Add all found layer samples to the main samples list
                # If we found at least one 2D image, skip NIfTI handling
                if layer_samples:
                    samples.extend(layer_samples)
                    continue
            else:
                # For NIfTI files, look for pattern like: sub-01_t1w.nii
                # Search for files matching the pattern
                all_nii_files = [f for f in os.listdir(subject_path) if f.endswith('.nii')]
                
                # Special handling for t1w to avoid matching ce-gadolinium_t1w
                if self.modality == "t1w":
                    # Match t1w but exclude ce-gadolinium_t1w
                    img_files = [f for f in all_nii_files 
                                if 't1w.nii' in f and 'ce-gadolinium' not in f]
                else:
                    # For other modalities, match the pattern
                    img_files = [f for f in all_nii_files 
                                if self.file_pattern in f]
                
                # Prefer ROI-stripped version if available
                roi_files = [f for f in img_files if 'desc-roi' in f]
                if roi_files:
                    img_file = roi_files[0]
                elif img_files:
                    img_file = img_files[0]
                else:
                    # Skip if modality not found
                    continue
                
                img_path = os.path.join(subject_path, img_file)
            
                if not os.path.exists(img_path):
                    # For t1w-ce modality, it's only available for HGG subjects (1-50)
                    # and 12 normal subjects (51-100), so skip if not found
                    continue
                
                # For NIfTI files, only add middle slice (extracted on-the-fly)
                samples.append({
                    'subject_id': subject_id,
                    'subject_dir': subject_dir,
                    'img_path': img_path,
                    'label': label,
                    'group': group,
                    'slice_layer': 'middle'  # NIfTI files use middle slice
                })
        
        if len(samples) == 0:
            raise ValueError(f"No samples found for modality {self.modality}")
        
        # Use Gazi's actual 2 classes (not BTMRI's 4 classes)
        # Prompt learning methods (CoOp, BiomedCoOp, DCSP, etc.) will automatically
        # reinitialize the prompt learner to adapt to the new number of classes
        gazi_classes = ['normal_brain', 'glioma_tumor']
        
        print(f"Gazi Dataset - Modality: {self.modality}")
        print(f"  Loading all three slice layers: t-1, middle, t+1")
        print(f"  Total samples: {len(samples)}")
        print(f"  Using Gazi classes: {gazi_classes}")
        
        # Count samples by label and slice layer
        from collections import Counter
        label_counts = Counter([s['label'] for s in samples])
        layer_counts = Counter([s.get('slice_layer', 'unknown') for s in samples])
        print(f"\n  Label distribution:")
        for label, count in label_counts.most_common():
            print(f"    {label}: {count}")
        print(f"\n  Slice layer distribution:")
        for layer, count in sorted(layer_counts.items()):
            print(f"    {layer}: {count}")
        
        # Create test set using Gazi's actual class structure
        test = self._process_split(samples, gazi_classes)
        
        # For cross-domain evaluation datasets:
        # - train_x uses test data (required by DatasetBase to extract classnames)
        #   but will not be used for training (eval-only mode)
        # - val is empty (no validation data)
        # - test contains all samples for evaluation
        # - classnames use Gazi's 2 classes; prompt learning methods will adapt
        super().__init__(train_x=test, val=[], test=test)
        
        # Use Gazi's actual 2 classes
        # Prompt learning methods (CoOp, BiomedCoOp, DCSP, etc.) handle different
        # class numbers by reinitializing prompt learner parameters
        self._classnames = gazi_classes
        self._num_classes = len(gazi_classes)
        self._lab2cname = {i: name for i, name in enumerate(gazi_classes)}
    
    def _process_split(self, samples, classnames):
        """Process samples into Datum format.
        
        Args:
            samples: List of sample dictionaries
            classnames: List of class names (Gazi's 2 classes: normal_brain, glioma_tumor)
        """
        items = []
        
        for sample in samples:
            img_path = sample['img_path']
            label = sample['label']
            
            # Map to BTMRI's class index
            label_idx = classnames.index(label)
            classname = label
            
            item = Datum(
                impath=img_path,
                label=label_idx,
                classname=classname
            )
            items.append(item)
        
        return items

