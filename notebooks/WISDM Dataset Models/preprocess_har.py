import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor

def _worker_process_subject(args):
    """
    Parallel worker process. Parses and aligns data pairs for a single subject,
    filtering for target activities immediately to save memory.
    """
    acc_file, raw_dir, columns, target_letters = args
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
                    # IMMEDIATE FILTER: Only keep rows belonging to your 6 specified activities
                    if parts[1] in target_letters:
                        data.append(parts)
                        
        df = pd.DataFrame(data, columns=columns)
        for col in ['timestamp', 'x', 'y', 'z']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna().sort_values('timestamp')

    df_acc = parse_file(acc_file)
    if df_acc.empty:
        return None
        
    df_gyro = parse_file(gyro_file)
    if df_gyro.empty:
        return None
    
    # Time-sync the remaining frames
    aligned = pd.merge_asof(
        df_acc, df_gyro[['timestamp', 'x', 'y', 'z']], 
        on='timestamp', suffixes=('_acc', '_gyro'),
        direction='nearest', tolerance=25000000
    ).dropna()
    
    return aligned


class ProductionHARPreprocessor:
    def __init__(self, raw_dir_path, sampling_rate=20, window_sec=2.5, overlap_pct=0.5):
        self.raw_dir = raw_dir_path
        self.fs = sampling_rate
        self.window_size = int(window_sec * sampling_rate)  # 50 timesteps
        self.step_size = int(self.window_size * (1 - overlap_pct))  # 25 timesteps
        
        # STRICT FILTER: Only map the 5 core codes representing your specified activities
        self.activity_mapping = {
            'A': 0,  # Walking
            'B': 1,  # Jogging
            'C': 2,  # Stairs (Upstairs/Downstairs baseline)
            'D': 3,  # Sitting
            'E': 4   # Standing
        }
        # Class 5 = Postural Transitions (Dynamic intermediate states)
        # Class 6 = Not-On-Body Anomaly (Static phone surfaces)

    def load_and_align_raw_data(self):
        """Dispatches multi-subject jobs across cores using the new target filter."""
        acc_path_pattern = os.path.join(self.raw_dir, 'accel', 'data_*_accel_phone.txt')
        acc_files = glob.glob(acc_path_pattern)
        columns = ['subject_id', 'activity', 'timestamp', 'x', 'y', 'z']
        
        print(f"Spawning parallel workers filtering strictly for your activities...")
        target_letters = set(self.activity_mapping.keys())
        worker_args = [(f, self.raw_dir, columns, target_letters) for f in acc_files]
        
        all_aligned_data = []
        with ProcessPoolExecutor() as executor:
            results = executor.map(_worker_process_subject, worker_args)
            for res in results:
                if res is not None:
                    all_aligned_data.append(res)
                    
        if not all_aligned_data:
            raise FileNotFoundError(f"No matching files found or filtered out. Check path: {self.raw_dir}")
            
        final_df = pd.concat(all_aligned_data, ignore_index=True)
        final_df['label'] = final_df['activity'].map(self.activity_mapping)
        return final_df

    def _butter_filter(self, data, cutoff, btype, order=4):
        nyq = 0.5 * self.fs
        b, a = butter(order, cutoff / nyq, btype=btype, analog=False)
        return filtfilt(b, a, data, axis=0)

    def compute_kinematic_channels(self, acc_raw, gyro_raw):
        acc_mag = np.linalg.norm(acc_raw, axis=1, keepdims=True)
        gravity = self._butter_filter(acc_raw, cutoff=0.5, btype='low')
        body_acc = self._butter_filter(acc_raw, cutoff=0.5, btype='high')
        
        body_acc_mag = np.linalg.norm(body_acc, axis=1, keepdims=True)
        gravity_mag = np.linalg.norm(gravity, axis=1, keepdims=True)
        
        dot_prod = np.sum(acc_raw * gravity, axis=1, keepdims=True)
        tilt_angle = np.arccos(np.clip(dot_prod / (acc_mag * gravity_mag + 1e-8), -1.0, 1.0))
        
        gyro_mag = np.linalg.norm(gyro_raw, axis=1, keepdims=True)
        gyro_body_mag = np.linalg.norm(self._butter_filter(gyro_raw, cutoff=0.5, btype='high'), axis=1, keepdims=True)
        
        return np.hstack([acc_raw, acc_mag, body_acc_mag, tilt_angle, gyro_raw, gyro_mag, gyro_body_mag])

    def generate_not_on_body_anomalies(self, num_windows=1000):
        windows = []
        for _ in range(num_windows):
            g = np.random.uniform(-1, 1, size=3)
            g = (g / np.linalg.norm(g)) * 9.81
            acc_raw = np.tile(g, (self.window_size, 1)) + np.random.normal(0, 0.02, (self.window_size, 3))
            gyro_raw = np.random.normal(0, 0.005, (self.window_size, 3))
            channels = self.compute_kinematic_channels(acc_raw, gyro_raw)
            windows.append(channels)
        return np.array(windows), np.full((num_windows,), 6) # Target index 6 for Static Anomalies

    def build_random_rotation_matrix(self):
        tx, ty, tz = np.random.uniform(0, 2*np.pi, size=3)
        Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
        Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
        Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def augment_via_3d_rotation(self, X_raw_axes, y_labels, num_aug_copies=1):
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
        print("Step 1: Ingesting and Filtering Target Activities...")
        df = self.load_and_align_raw_data()
        
        print(f"\nStep 2: Processing 11 Kinematic Channels ({df.shape[0]} aligned rows)...")
        features = self.compute_kinematic_channels(
            df[['x_acc', 'y_acc', 'z_acc']].values, 
            df[['x_gyro', 'y_gyro', 'z_gyro']].values
        )
        labels = df['label'].values
        
        print("\nStep 3: Constructing Window Sequences...")
        X_list, y_list = [], []
        for start in range(0, len(features) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            X_list.append(features[start:end])
            y_list.append(labels[start:end])
            
        X = np.array(X_list)
        y_chunked = np.array(y_list)
        
        print("\nStep 4: Mapping Dynamic Postural Transitions (Class 5)...")
        y_final = np.zeros(len(y_chunked), dtype=int)
        for i in range(len(y_chunked)):
            modes = pd.Series(y_chunked[i]).value_counts()
            if (modes.iloc[0] / self.window_size) < 0.85:
                y_final[i] = 5  # Target index 5 for dynamic intermediate changes
            else:
                y_final[i] = modes.index[0]
                
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_final, test_size=0.2, random_state=42, stratify=y_final
        )
        
        print("\nStep 5: Applying 3D Spatial Rotation Augmentations...")
        X_train_rot, y_train_rot = self.augment_via_3d_rotation(X_train, y_train, num_aug_copies=1)
        
        print("\nStep 6: Simulating Synthetic Not-On-Body Anomalies...")
        X_nob_train, y_nob_train = self.generate_not_on_body_anomalies(num_windows=2000)
        X_nob_test, y_nob_test = self.generate_not_on_body_anomalies(num_windows=500)
        
        X_train_final = np.concatenate([X_train, X_train_rot, X_nob_train], axis=0)
        y_train_final = np.concatenate([y_train, y_train_rot, y_nob_train], axis=0)
        
        X_test_final = np.concatenate([X_test, X_nob_test], axis=0)
        y_test_final = np.concatenate([y_test, y_nob_test], axis=0)
        
        print("\nStep 7: Applying Normalization Scaling...")
        mean = X_train_final.mean(axis=(0, 1), keepdims=True)
        std = X_train_final.std(axis=(0, 1), keepdims=True) + 1e-8
        
        X_train_norm = (X_train_final - mean) / std
        X_test_norm = (X_test_final - mean) / std
        
        print("\n================== RECONFIGURED PIPELINE REPORT ==================")
        print(f"X_train array shape: {X_train_norm.shape} -> y_train labels: {y_train_final.shape}")
        print(f"X_test array shape : {X_test_norm.shape}  -> y_test labels : {y_test_final.shape}")
        print(f"Total Model Tracking Classes (5 Core + Transitions + Anomaly): {len(np.unique(y_train_final))}")
        print("==================================================================")
        
        print("Saving filtered arrays to disk...")
        np.save("X_train_processed.npy", X_train_norm)
        np.save("X_test_processed.npy", X_test_norm)
        np.save("y_train_processed.npy", y_train_final)
        np.save("y_test_processed.npy", y_test_final)
        print("Done!")

if __name__ == "__main__":
    LOCAL_DATA_DIR = r"D:\Github\TinyML-HAR\data\WISDM Dataset"
    preprocessor = ProductionHARPreprocessor(raw_dir_path=LOCAL_DATA_DIR)
    preprocessor.process_dataset()