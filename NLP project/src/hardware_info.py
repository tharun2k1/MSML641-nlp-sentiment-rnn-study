# src/hardware_info.py
import os, json, platform, multiprocessing, subprocess

def _ram_gb():
    # Try psutil first (optional)
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        pass
    # macOS fallback via sysctl
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            return round(int(out) / (1024**3), 2)
    except Exception:
        pass
    # Linux fallback via /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024**2), 2)
    except Exception:
        pass
    return None

def main(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": multiprocessing.cpu_count(),
        "cuda_available": False,
        "mps_available": False,
        "device": "cpu",
    }
    # Optional: torch device info
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        info["device"] = "cuda" if info["cuda_available"] else ("mps" if info["mps_available"] else "cpu")
        if info["cuda_available"]:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    info["ram_total_gb"] = _ram_gb()

    out_path = os.path.join(out_dir, "hardware.json")
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[OK] Wrote hardware snapshot to {out_path}")

if __name__ == "__main__":
    main()
