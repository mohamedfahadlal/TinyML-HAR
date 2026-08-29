import os
import subprocess
import time
import re

ADB_PATH = r"C:\Users\hp\AppData\Local\Android\Sdk\platform-tools\adb.exe"
MODELS_DIR = r"D:\Github\TinyML-HAR\notebooks\WISDM Dataset Models"

MODELS = {
    "Model 1: Baseline LSTM": "deepconvlstm_quantized.tflite",
    "Model 2: CNN-LSTM Hybrid": "har_model_quantized.tflite",
    "Model 3: Pure 1D-CNN (Ours)": "pure_cnn_quantized.tflite"
}

def run_adb_cmd(args):
    cmd = [ADB_PATH] + args
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore', shell=True)
    return result.stdout if result.stdout else "", result.stderr if result.stderr else ""

print("⏳ Initializing Deep Logcat Search Pipeline...")
_, _ = run_adb_cmd(["devices"])

for model_name, filename in MODELS.items():
    local_path = os.path.join(MODELS_DIR, filename)
    phone_dest = f"/data/local/tmp/{filename}"
    
    if not os.path.exists(local_path):
        continue
        
    print(f"\n🚀 Running: {model_name}...")
    run_adb_cmd(["push", local_path, phone_dest])
    
    # Clean and launch
    run_adb_cmd(["shell", "am force-stop org.tensorflow.lite.benchmark"])
    run_adb_cmd(["logcat", "-c"])
    
    run_adb_cmd(["shell", f"am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args '\"--graph={phone_dest} --num_runs=50\"'"])
    time.sleep(6.0)
    
    # 🔓 THE CATCH-ALL UNFILTERED DUMP:
    # We grab the last 150 lines of system logs to see what the phone actually wrote!
    logs, _ = run_adb_cmd(["logcat", "-d", "-t", "150"])
    
    print(f"--- DIAGNOSTIC OUT LOG FOR {model_name} ---")
    
    # Search for ANY row containing tflite benchmark tags
    found_any = False
    for line in logs.split("\n"):
        if any(x in line.lower() for x in ["tflite", "inference", "timings", "avg", "us"]):
            print(line.strip())
            found_any = True
            
    if not found_any:
        print("⚠️ Direct metrics hidden by OS. Dumping top log signatures instead:")
        # Print first 5 active lines to see what your specific phone kernel is doing
        for line in logs.split("\n")[:8]:
            if line.strip(): print(f"  > {line.strip()}")
            
    print("-" * 80)