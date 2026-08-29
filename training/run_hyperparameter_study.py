import yaml
import subprocess
import os
import json
import sys
import threading
import time

# Try to import the existing SystemMonitor
try:
    # Adjust path so we can import from eda.utils
    sys.path.insert(0, os.path.abspath('..'))
    from eda.utils.system_monitor import SystemMonitor
    monitor_available = True
except ImportError:
    monitor_available = False

def run_study(config_idx=None):
    configs = [
        {"optimizer": "adam", "lr": 1e-3, "wd": 1e-4},
        {"optimizer": "adam", "lr": 1e-4, "wd": 1e-4},
        {"optimizer": "adamw", "lr": 1e-3, "wd": 1e-2},
        {"optimizer": "adamw", "lr": 1e-4, "wd": 1e-2},
        {"optimizer": "sgd", "lr": 1e-2, "wd": 1e-4},
        {"optimizer": "sgd", "lr": 1e-3, "wd": 1e-4},
    ]

    if config_idx is not None:
        if 0 <= config_idx < len(configs):
            configs = [configs[config_idx]]
        else:
            print(f"Invalid config index: {config_idx}")
            return

    base_config_path = r"config\training_config_p01_small.yaml"
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    results = []
    log_file_path = "hyperparameter_study.log"
    
    with open(log_file_path, "w") as log_f:
        def log_print(msg):
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()

        log_print("=== P01 Hyperparameter Study Started ===")
        log_print(f"Total configurations to test: {len(configs)}")
        
        # --- System Monitoring Thread ---
        stop_monitor = False
        def monitor_loop():
            if not monitor_available:
                log_print("[SYSTEM] SystemMonitor not available. Skipping real-time system monitoring.")
                return
            
            sys_mon = SystemMonitor()
            while not stop_monitor:
                try:
                    snap = sys_mon.snapshot()
                    cpu_pct = snap["cpu"]["percent"]
                    ram_pct = snap["memory"]["percent"]
                    
                    gpu_str = ""
                    if snap.get("gpu"):
                        gpus = []
                        for g in snap["gpu"]:
                            temp = g.get('temperature_c', 'N/A')
                            gpus.append(f"GPU {g['id']} ({g['load_percent']}% | {temp}C)")
                        gpu_str = " | " + " | ".join(gpus)
                    
                    log_print(f"[SYSTEM] CPU: {cpu_pct}% | RAM: {ram_pct}%{gpu_str}")
                except Exception as e:
                    pass
                time.sleep(60) # Log every 60 seconds
                
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        # --------------------------------

        for i, c in enumerate(configs):
            log_print(f"\n--- Running config {i+1}/{len(configs)}: {c} ---")
            cfg = base_config.copy()
            cfg["optimizer"]["name"] = c["optimizer"]
            cfg["optimizer"]["lr"] = c["lr"]
            cfg["optimizer"]["weight_decay"] = c["wd"]
            
            cfg["wandb"]["tags"] = ["picek", "small_subset", "binary", "temporal", "gru", "p01", f"opt_{c['optimizer']}", f"lr_{c['lr']}", f"wd_{c['wd']}"]
            cfg["experiment"]["name"] = f"P01 Hyperopt: {c['optimizer']} lr={c['lr']} wd={c['wd']}"

            temp_config_path = r"config\temp_hyper_config.yaml"
            with open(temp_config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)

            log_print(f"Launching training process for {c}...")
            
            # Run training and pipe output directly to the log file
            try:
                subprocess.run(
                    ["python", "train.py", "--config", temp_config_path],
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                log_print(f"\n[ERROR] Run failed with exit code {e.returncode}")
                continue

            # Get latest output manifest to find best metrics
            try:
                dirs = [os.path.join("outputs", d) for d in os.listdir("outputs") if os.path.isdir(os.path.join("outputs", d)) and d != "latest" and d != "logs"]
                if not dirs:
                    continue
                latest_real_dir = max(dirs, key=os.path.getmtime)
                
                manifest_path = os.path.join(latest_real_dir, "manifest.json")
                if os.path.exists(manifest_path):
                    with open(manifest_path, "r") as f:
                        manifest = json.load(f)
                    
                    best_metric = manifest.get("execution", {}).get("best_metric", 0.0)
                    
                    results.append({
                        "config": c,
                        "best_metric": best_metric,
                        "output_dir": latest_real_dir
                    })
                    log_print(f"\n[SUCCESS] Completed {c}. Best validation F1 (accident): {best_metric} in {latest_real_dir}")
                else:
                    log_print(f"\n[WARNING] Completed {c} but manifest.json not found in {latest_real_dir}")
            except Exception as e:
                log_print(f"\n[ERROR] Could not read metrics: {e}")
                
        # Stop monitor
        stop_monitor = True

        log_print("\n\n=== Hyperparameter Study Complete ===")
        log_print("Results Summary:")
        best_res = None
        for r in results:
            log_print(f"Config: {r['config']} -> Best Metric: {r['best_metric']} ({r['output_dir']})")
            if best_res is None or r['best_metric'] > best_res['best_metric']:
                best_res = r
                
        if best_res:
            log_print(f"\nWINNING CONFIGURATION: {best_res['config']} with score {best_res['best_metric']}")

if __name__ == '__main__':
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_study(idx)
