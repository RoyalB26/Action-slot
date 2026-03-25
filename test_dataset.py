import sys
sys.path.append('datasets')
import argparse
from taco import TACO

# Mock args for testing
class Args:
    root = r'C:\TACO'  # Đường dẫn dữ liệu TACO của bạn
    model_name = 'action_slot'
    seq_len = 16
    taco_class = 'Action'
    box = False
    allocated_slot = False
    num_slots = 21
    gt = False
    bg_mask = False
    obj_mask = False
    plot = False
    num_objects = 20
    backbone = 'x3d'  # Added for dataset transform pipeline compatibility

args = Args()

try:
    # Test loading train set
    print("Testing TACO dataset load...")
    dataset = TACO(args=args, split='train')
    print(f"Dataset loaded successfully. Length: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Actor shape: {sample['actor'].shape}")
        print(f"Ego: {sample['ego']}")
        print("Test passed!")
    else:
        print("Dataset is empty, check data path.")
except Exception as e:
    print(f"Error loading dataset: {e}")