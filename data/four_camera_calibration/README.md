Four-camera intrinsics calibration data.

Each session sits under `data/four_camera_calibration/<session>` with one subdirectory per camera serial and a `session.json` manifest.
Raw PNGs are written by `calibration/four_camera_intrinsics_capture.py`.
Per-session intrinsics outputs are written by `calibration/four_camera_intrinsics_calibrate.py`.

The `.gitignore` in this folder prevents the large image sets from being checked in.
