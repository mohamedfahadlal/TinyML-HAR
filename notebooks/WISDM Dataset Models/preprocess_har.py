import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor

import tensorflow as tf
print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def _worker_process_subject(args):
    """
    Independent worker function executed in parallel across CPU cores.
    Parses and aligns data pairs for a single subject.
    """
    acc_file, raw_dir, columns, activity_mapping = args
    subject_id = os.path.basename(acc_file).split('_')[1]
    gyro_file = os.path.join(raw_dir, 'gyro', f'data_{subject_id}_gyro_phone.txt')
    
    if not os.path.exists(gyro_file):
        return None
        
    def parse_file(path):
        data = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip().replace(';', '')
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) == 6:
                    data.append(parts)
        df = pd.DataFrame(data, columns=columns)
        for col in ['timestamp', 'x', 'y', 'z']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna().sort_values('timestamp')

    # Load and clean individual sensor streams
    df_acc = parse_file(acc_file)
    df_gyro = parse_file(gyro_file)
    
    # Asynchronous sensor polling correction using nearest-match (25ms tolerance)
    aligned = pd.merge_asof(
        df_acc, df_gyro[['timestamp', 'x', 'y', 'z']], 
        on='timestamp', suffixes=('_acc', '_gyro'),
        direction='nearest', tolerance=25000000  # 25ms in nanoseconds
    ).dropna()
    
    return aligned


class ProductionHARPreprocessor:
    def __init__(self, raw_dir_path, sampling_rate=20, window_sec=2.5, overlap_pct=0.5):
        """
        Orchestrates an end-to-end parallel preprocessing pipeline for the WISDM dataset.
        
        Args:
            raw_dir_path (str): Path to the folder containing 'accel' and 'gyro' subdirectories.
            sampling_rate (int): Base sensor polling frequency in Hz. Default is 20Hz.
            window_sec (float): Time window duration in seconds. Default is 2.5s.
            overlap_pct (float): Step overlap ratio between sequential windows (e.g., 0.5 = 50%).
        """
        self.raw_dir = raw_dir_path
        self.fs = sampling_rate
        self.window_size = int(window_sec * sampling_rate)  # 50 timesteps
        self.step_size = int(self.window_size * (1 - overlap_pct))  # 25 timesteps
        
        # Core WISDM activity letter-to-integer mapping
        self.activity_mapping = {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
            'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'O': 13, 'P': 14, 
            'Q': 15, 'R': 16, 'S': 17
        }
        # Class 18 = Postural Transitions (Dynamic shifts)
        # Class 19 = Not-On-Body Anomaly (Static phone surfaces)

    def load_and_align_raw_data(self):
        """Dispatches multi-subject file parsing jobs across parallel CPU processes."""
        acc_path_pattern = os.path.join(self.raw_dir, 'accel', 'data_*_accel_phone.txt')
        acc_files = glob.glob(acc_path_pattern)
        columns = ['subject_id', 'activity', 'timestamp', 'x', 'y', 'z']
        
        print(f"Spawning parallel workers across CPU cores to ingest files...")
        worker_args = [(f, self.raw_dir, columns, self.activity_mapping) for f in acc_files]
        
        all_aligned_data = []
        # Leverage concurrent pooling to bypass single-threaded Python bottlenecks
        with ProcessPoolExecutor() as executor:
            results = executor.map(_worker_process_subject, worker_args)
            for res in results:
                if res is not None:
                    all_aligned_data.append(res)
                    
        if not all_aligned_data:
            raise FileNotFoundError(f"No matching file pairs found. Verify layout inside: {self.raw_dir}")
            
        final_df = pd.concat(all_aligned_data, ignore_index=True)
        final_df = final_df[final_df['activity'].isin(self.activity_mapping.keys())].copy()
        final_df['label'] = final_df['activity'].map(self.activity_mapping)
        return final_df

    def _butter_filter(self, data, cutoff, btype, order=4):
        """Applies a zero-phase digital Butterworth filter over continuous arrays."""
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        return filtfilt(b, a, data, axis=0)

    def compute_kinematic_channels(self, acc_raw, gyro_raw):
        """Generates 11 total features, combining raw spatial tracks and rotation-invariant magnitudes."""
        # 1. Total Accelerometer Magnitude
        acc_mag = np.linalg.norm(acc_raw, axis=1, keepdims=True)
        
        # 2. Component Isolation via Butterworth Frequency Filtering
        gravity = self._butter_filter(acc_raw, cutoff=0.5, btype='low')
        body_acc = self._butter_filter(acc_raw, cutoff=0.5, btype='high')
        
        body_acc_mag = np.linalg.norm(body_acc, axis=1, keepdims=True)
        gravity_mag = np.linalg.norm(gravity, axis=1, keepdims=True)
        
        # 3. Trigonometric Tilt Angle Relative to Gravity
        dot_prod = np.sum(acc_raw * gravity, axis=1, keepdims=True)
        tilt_angle = np.arccos(np.clip(dot_prod / (acc_mag * gravity_mag + 1e-8), -1.0, 1.0))
        
        # 4. Total and Dynamic Gyroscope Magnitudes
        gyro_mag = np.linalg.norm(gyro_raw, axis=1, keepdims=True)
        gyro_body_mag = np.linalg.norm(self._butter_filter(gyro_raw, cutoff=0.5, btype='high'), axis=1, keepdims=True)
        
        # Horizontal fusion: 6 raw spatial axes + 5 rotation-invariant metrics = 11 final channels
        return np.hstack([
            acc_raw, acc_mag, body_acc_mag, tilt_angle,
            gyro_raw, gyro_mag, gyro_body_mag
        ])

    def generate_not_on_body_anomalies(self, num_windows=1000):
        """Synthesizes flat-surface resting blocks (Class 19) in the exact 11-channel matrix footprint."""
        windows = []
        for _ in range(num_windows):
            # Generate a random gravity vector orientation in 3D Space
            g = np.random.uniform(-1, 1, size=3)
            g = (g / np.linalg.norm(g)) * 9.81
            
            # Constant gravity with slight Gaussian white noise to replicate sensor jitter
            acc_raw = np.tile(g, (self.window_size, 1)) + np.random.normal(0, 0.02, (self.window_size, 3))
            gyro_raw = np.random.normal(0, 0.005, (self.window_size, 3))
            
            channels = self.compute_kinematic_channels(acc_raw, gyro_raw)
            windows.append(channels)
        return np.array(windows), np.full((num_windows,), 19)

    def build_random_rotation_matrix(self):
        """Generates a random 3D rotation matrix to physically simulate device placement variants."""
        tx, ty, tz = np.random.uniform(0, 2*np.pi, size=3)
        Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
        Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
        Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def augment_via_3d_rotation(self, X_raw_axes, y_labels, num_aug_copies=1):
        """Rotates raw coordinate slices and recalculates all tracking invariant features."""
        X_aug, y_aug = [], []
        for i in range(len(X_raw_axes)):
            acc_raw = X_raw_axes[i, :, 0:3]
            gyro_raw = X_raw_axes[i, :, 6:9]
            
            for _ in range(num_aug_copies):
                R = self.build_random_rotation_matrix()
                rot_acc = acc_raw @ R.T
                rot_gyro = gyro_raw @ R.T
                
                channels = self.compute_kinematic_channels(rot_acc, rot_gyro)
                X_aug.append(channels)
                y_aug.append(y_labels[i])
        return np.array(X_aug), np.array(y_aug)

    def process_dataset(self):
        """Executes full operational pipeline and outputs standardized, augmented arrays."""
        print("Step 1: Starting Accelerated Data Ingestion & Alignment...")
        df = self.load_and_align_raw_data()
        
        print(f"\nStep 2: Performing Kinematic Feature Extraction ({df.shape[0]} aligned rows)...")
        features = self.compute_kinematic_channels(
            df[['x_acc', 'y_acc', 'z_acc']].values, 
            df[['x_gyro', 'y_gyro', 'z_gyro']].values
        )
        labels = df['label'].values
        
        print("\nStep 3: Constructing Time-Series Sliding Windows...")
        X_list, y_list = [], []
        for start in range(0, len(features) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            X_list.append(features[start:end])
            y_list.append(labels[start:end])
            
        X = np.array(X_list)
        y_chunked = np.array(y_list)
        
        print("\nStep 4: Isolating Postural Transition States (Class 18)...")
        y_final = np.zeros(len(y_chunked), dtype=int)
        for i in range(len(y_chunked)):
            modes = pd.Series(y_chunked[i]).value_counts()
            # If the most common activity represents less than 85% of the sequence, code as Transition
            if (modes.iloc[0] / self.window_size) < 0.85:
                y_final[i] = 18
            else:
                y_final[i] = modes.index[0]
                
        # Split data to ensure validation sets have no contamination from data augmentation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_final, test_size=0.2, random_state=42, stratify=y_final
        )
        
        print("\nStep 5: Applying 3D Spatial Rotation Augmentations to Train Partition...")
        X_train_rot, y_train_rot = self.augment_via_3d_rotation(X_train, y_train, num_aug_copies=1)
        
        print("\nStep 6: Injecting Synthetic Not-On-Body Anomalies (Class 19)...")
        X_nob_train, y_nob_train = self.generate_not_on_body_anomalies(num_windows=1200)
        X_nob_test, y_nob_test = self.generate_not_on_body_anomalies(num_windows=300)
        
        # Combine base, rotated, and static anomaly matrices
        X_train_final = np.concatenate([X_train, X_train_rot, X_nob_train], axis=0)
        y_train_final = np.concatenate([y_train, y_train_rot, y_nob_train], axis=0)
        
        X_test_final = np.concatenate([X_test, X_nob_test], axis=0)
        y_test_final = np.concatenate([y_test, y_nob_test], axis=0)
        
        print("\nStep 7: Executing Global Scaler Normalization...")
        # Fit normalization parameters ONLY on training sets to avoid data leakage
        mean = X_train_final.mean(axis=(0, 1), keepdims=True)
        std = X_train_final.std(axis=(0, 1), keepdims=True) + 1e-8
        
        X_train_norm = (X_train_final - mean) / std
        X_test_norm = (X_test_final - mean) / std
        
        print("\n================== FINAL PIPELINE DATA REPORT ==================")
        print(f"X_train array shape: {X_train_norm.shape} -> y_train labels: {y_train_final.shape}")
        print(f"X_test array shape : {X_test_norm.shape}  -> y_test labels : {y_test_final.shape}")
        print(f"Total Model Tracking Classes Outputted: {len(np.unique(y_train_final))}")
        print("=================================================================")
        
        return X_train_norm, X_test_norm, y_train_final, y_test_final


# --- SCRIPT INFERENCE START ---
if __name__ == "__main__":
    # Point directly to your Windows local repository subdirectory
    LOCAL_DATA_DIR = r"D:\Github\TinyML-HAR\data\WISDM Dataset"
    
    # Initialize Preprocessing Object
    preprocessor = ProductionHARPreprocessor(
        raw_dir_path=LOCAL_DATA_DIR, 
        sampling_rate=20, 
        window_sec=2.5, 
        overlap_pct=0.5
    )
    
    # Run the parallel engine
    X_train, X_test, y_train, y_test = preprocessor.process_dataset()

# --- AT THE BOTTOM OF YOUR PREPROCESS_HAR.PY FILE ---
if __name__ == "__main__":
    LOCAL_DATA_DIR = r"D:\Github\TinyML-HAR\data\WISDM Dataset"
    
    preprocessor = ProductionHARPreprocessor(
        raw_dir_path=LOCAL_DATA_DIR, 
        sampling_rate=20, 
        window_sec=2.5, 
        overlap_pct=0.5
    )
    
    # Process the dataset
    X_train, X_test, y_train, y_test = preprocessor.process_dataset()
    
    # Save arrays to disk so you don't have to process them again
    print("Saving processed arrays to disk...")
    np.save("X_train_processed.npy", X_train)
    np.save("X_test_processed.npy", X_test)
    np.save("y_train_processed.npy", y_train)
    np.save("y_test_processed.npy", y_test)
    print("All arrays saved successfully! Preprocessing script finished.")