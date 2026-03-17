# AGENTS.md

## Tracker Launching

- Use `.\run_tracker.ps1` for tracker runs. It auto-selects the best available environment and prefers `.venv_ros2` when CUDA/TensorRT is available.
- `.\run_tracker_clean.ps1` is kept for compatibility, but now also defaults to the same auto-selection behavior. Only use `-ForceClean` when you explicitly want the CPU-only fallback.
- `.\run_tracker_ros2.ps1` forces the GPU/TensorRT environment.
- For quick checks without starting the tracker, run `.\run_tracker.ps1 -ProbeOnly` to see which environment would be selected.

## Performance Notes

- `.venv_ros2` is the high-performance tracker environment. On this machine it is the one that exposes CUDA and TensorRT.
- `.venv_clean` is CPU-oriented and should not be used for performance-sensitive tracker validation unless GPU is unavailable.
