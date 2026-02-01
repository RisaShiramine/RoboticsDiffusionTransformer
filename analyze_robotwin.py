import h5py
import numpy as np
import json
import os
import sys

def analyze_hdf5(file_path):
    print(f"Analyzing {file_path}...")
    try:
        with h5py.File(file_path, 'r') as f:
            def print_attrs(name, obj):
                print(name, obj)
                if isinstance(obj, h5py.Dataset):
                    print(f"  Shape: {obj.shape}, Type: {obj.dtype}")
                    if np.issubdtype(obj.dtype, np.number):
                         data = obj[:]
                         print(f"  Min: {np.min(data)}, Max: {np.max(data)}, Mean: {np.mean(data)}")

            f.visititems(print_attrs)
            
            # Check joint_action specifically
            if 'joint_action' in f:
                ja = f['joint_action']
                if 'left_arm' in ja:
                    left_arm = ja['left_arm'][:]
                    print(f"\nJoint Action Left Arm (First 5): \n{left_arm[:5]}")
                if 'right_arm' in ja:
                    right_arm = ja['right_arm'][:]
                    print(f"\nJoint Action Right Arm (First 5): \n{right_arm[:5]}")
                    
    except Exception as e:
        print(f"Error reading hdf5: {e}")

def find_instruction(hdf5_path):
    # Logic from hdf5_vla_dataset.py
    file_name = os.path.basename(hdf5_path).replace('.hdf5', '')
    
    # Try multiple possible locations
    possible_dirs = [
        os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(hdf5_path)), "../instructions")),
        "/mnt/hdd/RoboTwin/data/stack_blocks_three/demo_clean/instructions",
        "data/datasets/robotwin_instructions" 
    ]
    
    found = False
    for instr_dir in possible_dirs:
        instr_path = os.path.join(instr_dir, f"{file_name}.json")
        if os.path.exists(instr_path):
            print(f"\nFound instruction file at: {instr_path}")
            with open(instr_path, 'r') as fp:
                data = json.load(fp)
                print(json.dumps(data, indent=2))
            found = True
            break
    
    if not found:
        print("\nInstruction file not found in examined directories.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # Default to episode0.hdf5 in the linked directory
        target_file = "data/datasets/robotwin/episode0.hdf5"
    
    if os.path.exists(target_file):
        analyze_hdf5(target_file)
        find_instruction(target_file)
    else:
        print(f"File {target_file} does not exist.")
