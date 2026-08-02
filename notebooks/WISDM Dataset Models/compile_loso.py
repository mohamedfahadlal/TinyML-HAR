import os
import numpy as np

# =====================================================================
# 1. DIRECTORY CONFIGURATION (MATCHED TO YOUR EXACT DISK LAYOUT)
# =====================================================================
BASE_DIR = r"D:\Github\TinyML-HAR\data\WISDM Dataset"
PHONE_ACCEL_DIR = os.path.join(BASE_DIR, "accel")
PHONE_GYRO_DIR = os.path.join(BASE_DIR, "gyro")

print(f"✅ Target Accel Directory: {PHONE_ACCEL_DIR}")
print(f"✅ Target Gyro Directory : {PHONE_GYRO_DIR}")

# Quick double-check safety guard
if not os.path.exists(PHONE_ACCEL_DIR) or not os.path.exists(PHONE_GYRO_DIR):
    print("❌ Error: Verification failed. Paths not recognized.")
    exit()

# =====================================================================
# 2. SLIDING WINDOW & ACTIVITY TARGET PARAMETERS
# =====================================================================
WINDOW_SIZE = 50
STEP_SIZE = 25       # 50% Overlap configuration
NUM_CHANNELS = 11    # 3 Accel + 3 Gyro + 1 Accel_SMV + 1 Gyro_SMV + 3 Filtered Accel

# Core classification mapping matching your neural network topology
activity_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

all_windows = []
all_labels = []
all_subjects = []

print("\n⏳ Starting synchronized multi-sensor parsing for all 51 subjects...")
print("=" * 70)

# Helper function to cleanly parse raw text lines into mapping arrays
def parse_wisdm_file(file_path):
    data_dict = {act: [] for act in activity_mapping}
    with open(file_path, 'r') as f:
        for line in f:
            clean = line.strip().rstrip(';')
            if not clean: 
                continue
            parts = clean.split(',')
            if len(parts) < 6: 
                continue
            
            act = parts[1].strip()
            if act in activity_mapping:
                try:
                    # Append raw x, y, z floats
                    data_dict[act].append([float(parts[3]), float(parts[4]), float(parts[5])])
                except ValueError:
                    continue  # Safely ignore un-parseable lines
    return {act: np.array(lst) for act, lst in data_dict.items() if len(lst) > 0}

# =====================================================================
# 3. CORE SYNCHRONIZATION LOOP (SUBJECT RANGE: 1600 - 1650)
# =====================================================================
for sub_id in range(1600, 1651):
    accel_file = os.path.join(PHONE_ACCEL_DIR, f"data_{sub_id}_accel_phone.txt")
    gyro_file = os.path.join(PHONE_GYRO_DIR, f"data_{sub_id}_gyro_phone.txt")
    
    # Skip if a specific subject file is missing
    if not os.path.exists(accel_file) or not os.path.exists(gyro_file):
        continue
        
    print(f"Syncing Streams -> Subject {sub_id}...")
    
    accel_data = parse_wisdm_file(accel_file)
    gyro_data = parse_wisdm_file(gyro_file)
    
    # Process activity intersections across both data sensors
    for act in activity_mapping:
        if act in accel_data and act in gyro_data:
            # Sync matching row constraints to mitigate minor sensor drop-outs
            min_len = min(len(accel_data[act]), len(gyro_data[act]))
            a_stream = accel_data[act][:min_len]
            g_stream = gyro_data[act][:min_len]
            
            # Step over timelines using sliding window step size
            for start in range(0, min_len - WINDOW_SIZE + 1, STEP_SIZE):
                end = start + WINDOW_SIZE
                
                accel_win = a_stream[start:end]
                gyro_win = g_stream[start:end]
                
                # --- PRODUCTION FEATURE ENGINE SELECTION ---
                # 1. Compute Euclidean Magnitude Vectors (SMV) to lock in rotational invariance
                accel_smv = np.sqrt(np.sum(accel_win**2, axis=1, keepdims=True))
                gyro_smv = np.sqrt(np.sum(gyro_win**2, axis=1, keepdims=True))
                
                # 2. Extract local gravity offsets via baseline high-pass averaging
                filtered_accel = accel_win - np.mean(accel_win, axis=0)
                
                # 3. Stack all features into your 11-Channel array architecture
                feature_matrix = np.hstack([
                    accel_win, gyro_win, accel_smv, gyro_smv, filtered_accel
                ])
                
                all_windows.append(feature_matrix)
                all_labels.append(activity_mapping[act])
                all_subjects.append(sub_id)

# =====================================================================
# 4. STORAGE & DATA DUMP EXPORT
# =====================================================================
print("=" * 70)
print("⏳ Writing finalized matrices to disk arrays...")

np.save("X_all_windows.npy", np.array(all_windows, dtype=np.float32))
np.save("y_all_labels.npy", np.array(all_labels, dtype=np.int32))
np.save("subject_ids.npy", np.array(all_subjects, dtype=np.int32))

print("\n" + "="*55)
print("💾 SUCCESS: DUAL-SENSOR COMPILATION COMPLETE")
print("="*55)
print(f"Total Synchronized Windows : {len(all_windows)}")
print(f"Unified Window Matrix Shape: {np.array(all_windows).shape} (11 Channels)")
print(f"Output Target Arrays       : X_all_windows.npy, y_all_labels.npy, subject_ids.npy")
print("="*55)